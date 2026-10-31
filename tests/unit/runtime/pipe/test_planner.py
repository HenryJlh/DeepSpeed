# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import torch.nn as nn

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.pipe.planner import (LayerCost, load_costs, model_state_bytes, optimizer_step_bytes, profile_layer, save_costs, saved_activation_bytes, StageMemoryModel)

from unit.common import DistributedTest

GiB = 1024**3


class TestCostModel:

    def test_scaling_keeps_model_states(self):
        layer = LayerCost(1.0, 2.0, activation_bytes=10, input_bytes=1, state_bytes=100, transient_bytes=3).scaled(4)
        assert (layer.fwd_ms, layer.bwd_ms, layer.activation_bytes, layer.input_bytes, layer.state_bytes,
                layer.transient_bytes) == (4.0, 8.0, 40, 4, 100, 12)

    def test_checkpointed_layer_runs_its_forward_twice(self):
        layer = LayerCost(1.0, 2.0, activation_bytes=10, input_bytes=1)
        assert (layer.time_ms(), layer.time_ms(checkpointed=True)) == (3.0, 4.0)
        assert (layer.kept_bytes(), layer.kept_bytes(checkpointed=True)) == (10, 1)

    def test_state_bytes(self):
        # bf16 parameters and gradients, fp32 master weights and Adam moments sharded by ZeRO-1
        assert model_state_bytes(10) == 10 * 16
        assert model_state_bytes(10, dp_size=4) == 10 * (2 + 2 + 3)
        # the step's fp32 gradient partition and Adam temporaries, sharded the same way
        assert optimizer_step_bytes(10, dp_size=2) == 10 * 3


class TestCostsFile:

    def test_round_trip_and_linear_scaling(self, tmp_path):
        block = LayerCost(1.0, 2.0, activation_bytes=GiB, input_bytes=0.1 * GiB, state_bytes=GiB)
        moe = LayerCost(3.0, 6.0, activation_bytes=2 * GiB, input_bytes=0.1 * GiB, state_bytes=4 * GiB)
        head = LayerCost(4.0, 8.0, activation_bytes=3 * GiB, transient_bytes=5 * GiB)
        path = tmp_path / "costs.json"
        save_costs(path, {"block": block, "moe": moe}, ["block", "moe", "moe"], last_stage_extra=[head])
        layers, first, last = load_costs(str(path))(1)
        assert [l.fwd_ms for l in layers] == [1.0, 3.0, 3.0] and [l.name for l in layers] == ["block", "moe", "moe"]
        assert first == [] and last == [head]
        layers, _, last = load_costs(str(path))(2)
        assert layers[1].activation_bytes == 4 * GiB and layers[1].state_bytes == 4 * GiB
        assert last[0].transient_bytes == 10 * GiB

    def test_measured_sizes_take_precedence(self):
        spec = {
            "micro_batch_size": 1,
            "layer_types": {
                "block": {
                    "fwd_ms": 1.0,
                    "bwd_ms": 2.0
                }
            },
            "layer_sequence": ["block"] * 2,
            "per_micro_batch_size": {
                "2": {
                    "layer_types": {
                        "block": {
                            "fwd_ms": 1.5,
                            "bwd_ms": 3.0
                        }
                    }
                }
            },
        }
        assert [l.fwd_ms for l in load_costs(spec)(2)[0]] == [1.5, 1.5]
        assert [l.fwd_ms for l in load_costs(spec)(4)[0]] == [4.0, 4.0]

    def test_rejects_unknown_layer_types(self, tmp_path):
        with pytest.raises(ValueError, match="without costs"):
            save_costs(tmp_path / "costs.json", {"block": LayerCost(1, 2)}, ["block", "moe"])


class TestProfiler:

    def test_saved_activation_bytes(self):
        x = torch.randn(4, 16, requires_grad=True)
        linear = nn.Linear(16, 16)
        # Linear saves its input for the weight gradient; the weight itself is a parameter
        assert saved_activation_bytes(linear, (x, ))[0] == x.numel() * 4
        # tanh additionally saves its output
        block = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        assert saved_activation_bytes(block, (x, ))[0] == 2 * x.numel() * 4


class _HeadLoss(nn.Module):
    """A vocabulary head + loss: its backward materializes an fp32 logits gradient it never keeps."""

    def __init__(self, hidden, vocab):
        super().__init__()
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, x):
        logits = self.lm_head(x).float()
        return torch.nn.functional.cross_entropy(logits, torch.zeros(x.size(0), dtype=torch.long, device=x.device))


class _ExpertBlock(nn.Module):
    """A dense projection and an "expert" marked the way DeepSpeed MoE marks expert parameters."""

    def __init__(self, hidden):
        super().__init__()
        self.dense = nn.Linear(hidden, hidden)
        self.expert = nn.Linear(hidden, hidden)
        for p in self.expert.parameters():
            p.allreduce = False

    def forward(self, x):
        return self.expert(torch.tanh(self.dense(x)))


class TestProfileLayer(DistributedTest):
    # profiles on the accelerator, which a forked test process may only initialize under DistributedTest
    world_size = 1

    def test_profile_layer(self):
        device = get_accelerator().device_name()
        layer = nn.Sequential(nn.Linear(16, 16), nn.Tanh()).to(device)
        cost = profile_layer(layer, (torch.randn(4, 16, device=device), ), warmup=1, iters=2, name="block")
        assert cost.name == "block" and cost.fwd_ms > 0 and cost.bwd_ms >= 0
        assert cost.input_bytes == 4 * 16 * 4
        assert cost.activation_bytes == 2 * 4 * 16 * 4
        assert cost.state_bytes == model_state_bytes(16 * 16 + 16)
        assert cost.transient_bytes >= 0

    def test_expert_states_are_sharded_over_expert_data_parallel_ranks(self):
        device = get_accelerator().device_name()
        block = _ExpertBlock(16).to(device)
        cost = profile_layer(block, (torch.randn(4, 16, device=device), ), warmup=1, iters=1, dp_size=8, ep_size=4)
        params = 16 * 16 + 16
        assert cost.state_bytes == pytest.approx(model_state_bytes(params, 8) + model_state_bytes(params, 2))
        assert cost.step_bytes == pytest.approx(optimizer_step_bytes(params, 8))
        assert cost.expert_step_bytes == pytest.approx(optimizer_step_bytes(params, 2))
        # other settings give their own states; the step is then not modelled
        other = profile_layer(block, (torch.randn(4, 16, device=device), ), warmup=1, iters=1, state_bytes=123)
        assert (other.state_bytes, other.step_bytes, other.expert_step_bytes) == (123, 0, 0)
        with pytest.raises(ValueError, match="must divide"):
            profile_layer(block, (torch.randn(4, 16, device=device), ), warmup=1, iters=1, dp_size=2, ep_size=4)

    def test_backward_transient_of_a_head(self):
        if get_accelerator().device_name() == "cpu":
            pytest.skip("the cpu accelerator's peak counter is the process RSS")
        tokens, vocab = 512, 8192
        device = get_accelerator().device_name()
        head = _HeadLoss(64, vocab).to(device)
        cost = profile_layer(head, (torch.randn(tokens, 64, device=device), ), warmup=1, iters=1)
        # the fp32 logits gradient exists only during the backward
        assert cost.transient_bytes >= tokens * vocab * 4


class TestStageMemoryModel:

    def test_inflight_follows_1f1b(self):
        layer = [LayerCost(1, 2, activation_bytes=10, input_bytes=1)]
        model = StageMemoryModel(num_stages=4, micro_batches=8)
        assert [model.stage_bytes(s, layer, 0) for s in range(4)] == [40, 30, 20, 10]
        # never more in flight than there are micro-batches
        assert StageMemoryModel(num_stages=4, micro_batches=2).stage_bytes(0, layer, 0) == 20

    def test_terms(self):
        block = LayerCost(1, 2, activation_bytes=10, input_bytes=1, state_bytes=100, transient_bytes=3)
        head = LayerCost(1, 2, activation_bytes=20, state_bytes=50, transient_bytes=40)
        model = StageMemoryModel(num_stages=2, micro_batches=4)
        # stage 0 keeps 2 micro-batches; only one layer's backward transient is live at a time
        assert model.stage_bytes(0, [block] * 3, 0) == 300 + 2 * 30 + 3
        # a checkpointed layer keeps its input and rebuilds its activations during its backward
        assert model.stage_bytes(0, [block] * 3, 2) == 300 + 2 * (1 + 1 + 10) + (3 + 10)
        # the last stage keeps one micro-batch, and the head's logits gradient is the largest transient
        assert model.stage_bytes(1, [block], 0, [head]) == 150 + (10 + 20) + 40

    def test_optimizer_step_can_be_the_peak(self):
        block = LayerCost(1,
                          2,
                          activation_bytes=10,
                          input_bytes=1,
                          state_bytes=100,
                          step_bytes=40,
                          expert_step_bytes=30)
        embed = LayerCost(1, 2, state_bytes=50, step_bytes=20)
        model = StageMemoryModel(num_stages=2, micro_batches=4)
        # the dense group (3 x 40 + 20) is stepped apart from the expert group (3 x 30); activations are gone
        assert model.stage_bytes(0, [block] * 3, 0, [embed]) == 350 + 140
        # with more activations in flight the 1F1B phase is the peak again
        many = LayerCost(1, 2, activation_bytes=100, input_bytes=1, state_bytes=100, step_bytes=40)
        assert model.stage_bytes(0, [many] * 3, 0, [embed]) == 350 + 2 * 300
