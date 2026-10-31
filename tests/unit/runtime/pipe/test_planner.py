# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import itertools
import json
import random

import pytest
import torch
import torch.nn as nn

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.pipe.planner import (LayerCost, PipelinePlan, StageMemoryModel, evaluate_partition, load_costs,
                                            model_state_bytes, optimizer_step_bytes, pipeline_step_ms, profile_layer,
                                            save_costs, saved_activation_bytes, search_micro_batch_size, solve_stages)
from deepspeed.runtime.pipe.planner.__main__ import main as planner_main
from deepspeed.runtime.pipe.planner.solver import _StageCosts

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


def random_layers(rng, n):
    layers = []
    for _ in range(n):
        fwd = rng.uniform(1, 5)
        act = rng.uniform(1, 4) * GiB
        layers.append(
            LayerCost(fwd_ms=fwd,
                      bwd_ms=2 * fwd,
                      activation_bytes=act,
                      input_bytes=act / rng.uniform(5, 20),
                      state_bytes=rng.uniform(1, 3) * GiB,
                      transient_bytes=rng.uniform(0, 2) * GiB,
                      step_bytes=rng.uniform(0, 4) * GiB,
                      expert_step_bytes=rng.uniform(0, 4) * GiB))
    return layers


def brute_force(layers, num_stages, limit, micro_batches, first=(), last=()):
    """Every split and, per stage, the fewest checkpointed layers that fit, via the reference memory model."""
    model = StageMemoryModel(num_stages, micro_batches)
    n, best = len(layers), None
    for cuts in itertools.combinations(range(1, n), num_stages - 1):
        bounds = (0, ) + cuts + (n, )
        worst = 0.0
        for s in range(num_stages):
            chunk = layers[bounds[s]:bounds[s + 1]]
            extras = (list(first) if s == 0 else []) + (list(last) if s == num_stages - 1 else [])
            for c in range(len(chunk) + 1):
                if model.stage_bytes(s, chunk, c, extras) <= limit:
                    t = sum(e.time_ms() for e in extras) + sum(l.time_ms(i < c) for i, l in enumerate(chunk))
                    break
            else:
                t = float("inf")
            worst = max(worst, t)
        best = worst if best is None else min(best, worst)
    return best


class TestSolver:

    @pytest.mark.parametrize("seed", range(12))
    def test_matches_brute_force(self, seed):
        rng = random.Random(seed)
        n, stages, mb = rng.randint(4, 8), rng.randint(2, 4), rng.randint(2, 8)
        layers = random_layers(rng, n)
        first = [LayerCost(rng.uniform(1, 4), rng.uniform(2, 8), activation_bytes=GiB, state_bytes=3 * GiB)]
        last = [LayerCost(rng.uniform(2, 8), rng.uniform(4, 16), activation_bytes=2 * GiB, state_bytes=4 * GiB)]
        limit = rng.uniform(10, 40) * GiB
        expected = brute_force(layers, stages, limit, mb, first, last)
        plan = solve_stages(layers, stages, limit, mb, first_stage_extra=first, last_stage_extra=last)
        if expected == float("inf"):
            assert plan is None
            return
        assert max(s.time_ms for s in plan) == pytest.approx(expected)
        assert sum(s.num_layers for s in plan) == n
        assert all(s.memory_bytes <= limit for s in plan)

    def test_fast_stage_cost_matches_reference_model(self):
        rng = random.Random(0)
        layers = random_layers(rng, 6)
        extra = [LayerCost(1, 2, activation_bytes=GiB, state_bytes=GiB, transient_bytes=3 * GiB)]
        model = StageMemoryModel(3, 5)
        cost = _StageCosts(layers, model, float("inf"), extra, extra)
        for s in range(3):
            # an infinite budget returns c = 0; check every c through the incremental path
            for c in range(len(layers) + 1):
                # the two paths sum in a different order, so allow for rounding at the boundary
                cost.limit = model.stage_bytes(s, layers, c, cost.extras(s)) * (1 + 1e-9)
                cost.cache.clear()
                _, chosen, memory = cost(s, 0, len(layers))
                assert chosen <= c
                assert memory == pytest.approx(model.stage_bytes(s, layers, chosen, cost.extras(s)))

    def test_memory_pressure_checkpoints_early_stages_more(self):
        layers = [LayerCost(1, 2, activation_bytes=GiB, input_bytes=0.01 * GiB, state_bytes=0.5 * GiB)] * 16
        plan = solve_stages(layers, 4, memory_limit_bytes=8 * GiB, micro_batches=8)
        ckpt = [s.num_checkpointed for s in plan]
        # stage 0 holds 4 micro-batches and must recompute; the last stage holds one and need not
        assert ckpt[0] > 0 and ckpt[-1] == 0
        assert all(s.memory_bytes <= 8 * GiB for s in plan)
        # without memory pressure nothing is checkpointed and the split is uniform
        plan = solve_stages(layers, 4, memory_limit_bytes=1000 * GiB, micro_batches=8)
        assert [s.num_layers for s in plan] == [4, 4, 4, 4]
        assert all(s.num_checkpointed == 0 for s in plan)

    def test_optimizer_step_moves_layers_off_a_stage(self):
        layers = [LayerCost(1, 2, state_bytes=GiB, step_bytes=2 * GiB)] * 8
        embed = [LayerCost(0, 0, state_bytes=2 * GiB, step_bytes=4 * GiB)]
        # k layers on stage 0 step at (k + 2) + (2k + 4) GiB, so at most 3 fit in 15 GiB; recomputation cannot help
        plan = solve_stages(layers, 2, memory_limit_bytes=15 * GiB, micro_batches=4, first_stage_extra=embed)
        assert [s.num_layers for s in plan] == [3, 5]
        assert [s.num_checkpointed for s in plan] == [0, 0]
        # without the step the split would balance time
        plan = solve_stages([LayerCost(1, 2, state_bytes=GiB)] * 8,
                            2,
                            15 * GiB,
                            4,
                            first_stage_extra=[LayerCost(0, 0, state_bytes=2 * GiB)])
        assert [s.num_layers for s in plan] == [4, 4]

    def test_heavy_last_stage_gets_fewer_layers(self):
        layers = [LayerCost(1, 2)] * 12
        plan = solve_stages(layers, 3, float("inf"), 4, last_stage_extra=[LayerCost(3, 6)])
        assert [s.num_layers for s in plan] == [5, 5, 2]

    def test_infeasible(self):
        layers = [LayerCost(1, 2, state_bytes=10 * GiB)] * 4
        assert solve_stages(layers, 2, memory_limit_bytes=GiB, micro_batches=2) is None

    def test_search_skips_sizes_that_do_not_divide(self):
        layers = [LayerCost(1, 2, activation_bytes=GiB, input_bytes=0.1 * GiB)] * 8

        def costs(mbs):
            return [l.scaled(mbs) for l in layers], [], []

        plan = search_micro_batch_size(costs, [1, 3, 4],
                                       global_batch_size=32,
                                       data_parallel_size=2,
                                       num_stages=2,
                                       memory_limit_bytes=100 * GiB)
        # mbs=3 does not divide 32 / 2; with linear costs the smaller micro-batch has the smaller fill/drain
        # bubble: mbs=1 is 2 * 12 + 15 * 12 = 204 ms, mbs=4 is 2 * 48 + 3 * 48 = 240 ms
        assert plan.micro_batch_size == 1 and plan.micro_batches == 16
        assert plan.step_ms == pytest.approx(204.0)
        assert pipeline_step_ms([48.0, 48.0], 4) == pytest.approx(240.0)

    def test_evaluate_partition(self):
        layers = [LayerCost(1, 2, activation_bytes=GiB, input_bytes=0.0)] * 4
        stages = evaluate_partition(layers, [2, 2], [2, 0], micro_batches=4)
        assert [s.time_ms for s in stages] == [8.0, 6.0]
        assert stages[0].memory_bytes == pytest.approx(GiB)  # only the recomputed layer
        assert stages[1].memory_bytes == pytest.approx(2 * GiB)  # 1F1B: last stage holds one micro-batch


class TestPlanFile:

    def test_round_trip(self, tmp_path):
        layers = [LayerCost(1, 2, activation_bytes=GiB, input_bytes=0.01 * GiB)] * 8
        stages = solve_stages(layers, 2, memory_limit_bytes=3 * GiB, micro_batches=4)
        plan = PipelinePlan(stages=stages, micro_batches=4, partition_layer_type="Block")
        path = tmp_path / "plan.json"
        plan.save(path)
        loaded = PipelinePlan.load(path)
        assert loaded.layers_per_stage == plan.layers_per_stage
        assert loaded.checkpoint_layers_per_stage == plan.checkpoint_layers_per_stage
        assert loaded.partition_method == "custom:" + ",".join(map(str, plan.layers_per_stage))
        assert loaded.partition_layer_type == "Block"

    def test_hand_written_plan(self, tmp_path):
        path = tmp_path / "plan.json"
        path.write_text(json.dumps({"layers_per_stage": [3, 5], "checkpoint_layers_per_stage": [2, 0]}))
        plan = PipelinePlan.load(path)
        assert plan.partition_method == "custom:3,5"
        assert plan.checkpoint_layers_per_stage == [2, 0]


class TestCommandLine:

    def costs(self, tmp_path):
        dense = LayerCost(1.0, 2.0, activation_bytes=GiB, input_bytes=0.05 * GiB, state_bytes=0.2 * GiB)
        moe = LayerCost(2.5, 5.0, activation_bytes=2 * GiB, input_bytes=0.05 * GiB, state_bytes=GiB)
        head = LayerCost(4.0, 8.0, activation_bytes=2 * GiB, state_bytes=2 * GiB)
        path = tmp_path / "costs.json"
        save_costs(path, {"dense": dense, "moe": moe}, ["dense"] * 2 + ["moe"] * 10, last_stage_extra=[head])
        return str(path)

    def test_plan(self, tmp_path, capsys):
        ppath = tmp_path / "plan.json"
        args = ["--costs", self.costs(tmp_path), "--stages", "4", "--global-batch-size", "32", "--dp", "2"]
        assert planner_main(
            args +
            ["--memory-gb", "16", "--micro-batch-sizes", "1,2", "--partition-layer-type", "Block", "-o",
             str(ppath)]) == 0
        out = capsys.readouterr().out
        assert "uniform, full checkpointing" in out
        plan = PipelinePlan.load(ppath)
        assert sum(plan.layers_per_stage) == 12 and plan.partition_layer_type == "Block"
        # the default headroom keeps every stage under 90 % of the device memory
        assert plan.notes["headroom_gb"] == pytest.approx(1.6)
        assert all(s.memory_bytes <= 14.4 * GiB for s in plan.stages)

    def test_headroom_is_left_out_of_the_budget(self, tmp_path):
        ppath = tmp_path / "plan.json"
        args = ["--costs", self.costs(tmp_path), "--stages", "4", "--global-batch-size", "8", "-o", str(ppath)]
        assert planner_main(args + ["--memory-gb", "16", "--headroom-gb", "4"]) == 0
        assert max(s.memory_bytes for s in PipelinePlan.load(ppath).stages) <= 12 * GiB
        # 1 GiB is left, less than the head's states alone
        assert planner_main(args + ["--memory-gb", "16", "--headroom-gb", "15"]) == 1
