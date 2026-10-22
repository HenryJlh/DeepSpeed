# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.moe import aux_loss
from deepspeed.moe.aux_loss import attach_aux_loss
from deepspeed.moe.layer import MoE
from deepspeed.pipe import PipelineModule
from deepspeed.utils import groups

from unit.common import DistributedTest

HIDDEN = 8
COEF = 0.5


class AuxLayer(nn.Module):
    """A layer with a load-balancing-style loss on a gate that does not feed the output.

    Without the auxiliary loss its gate gets no gradient, so a lost auxiliary loss is visible.
    """

    def __init__(self, attach=True):
        super().__init__()
        self.fc = nn.Linear(HIDDEN, HIDDEN)
        self.gate = nn.Linear(HIDDEN, 4)
        self.attach = attach
        self.last_aux = None

    def forward(self, x):
        aux = self.gate(x).pow(2).mean()
        self.last_aux = aux
        out = x + torch.tanh(self.fc(x))
        return attach_aux_loss(out, aux, COEF) if self.attach else out


class TestAttachAuxLoss:

    def setup_method(self):
        aux_loss.set_aux_loss_grad_scale(1.0)
        aux_loss.pop_recorded_aux_loss()

    def test_gradient_is_that_of_the_combined_loss(self):
        torch.manual_seed(0)
        layer = AuxLayer()
        ref = copy.deepcopy(layer)
        ref.attach = False
        x = torch.randn(3, HIDDEN)

        aux_loss.set_aux_loss_grad_scale(0.25)
        (layer(x).sum() * 0.25).backward()
        out = ref(x)
        ((out.sum() + COEF * ref.last_aux) * 0.25).backward()
        for p, q in zip(layer.parameters(), ref.parameters()):
            assert torch.allclose(p.grad, q.grad, atol=1e-6)
        total, count = aux_loss.pop_recorded_aux_loss()
        assert count == 1 and torch.allclose(total, COEF * ref.last_aux.detach())

    def test_no_grad_returns_output_unchanged(self):
        layer = AuxLayer()
        x = torch.randn(3, HIDDEN)
        with torch.no_grad():
            assert torch.equal(layer(x), x + torch.tanh(layer.fc(x)))

    def test_recorded_once_under_non_reentrant_checkpointing(self):
        layer = AuxLayer()
        x = torch.randn(3, HIDDEN, requires_grad=True)
        checkpoint(layer, x, use_reentrant=False).sum().backward()
        assert aux_loss.pop_recorded_aux_loss()[1] == 1
        assert layer.gate.weight.grad.abs().sum() > 0


def pipeline_config(gas):
    return {
        "train_batch_size": gas,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": gas,
        # the reference does not clip; DeepSpeed clips at 1.0 by default
        "gradient_clipping": 0.0,
        "steps_per_print": 1000,
        "optimizer": {
            "type": "SGD",
            "params": {
                "lr": 0.1
            }
        },
        "zero_allow_untested_optimizer": True,
    }


class TestPipelineAuxLoss(DistributedTest):
    world_size = 2

    def test_first_stage_aux_loss_reaches_its_gate(self):
        gas, lr = 4, 0.1
        torch.manual_seed(0)
        layers = [AuxLayer(), AuxLayer(), nn.Linear(HIDDEN, HIDDEN), nn.Linear(HIDDEN, HIDDEN)]
        reference = copy.deepcopy(nn.Sequential(*layers))
        for layer in reference:
            if isinstance(layer, AuxLayer):
                layer.attach = False
        gate_before = layers[0].gate.weight.detach().clone()
        data = [(torch.randn(1, HIDDEN), torch.randn(1, HIDDEN)) for _ in range(gas)]

        model = PipelineModule(layers=layers, num_stages=2, loss_fn=nn.MSELoss(), partition_method="uniform")
        engine, _, _, _ = deepspeed.initialize(config=pipeline_config(gas),
                                               model=model,
                                               model_parameters=model.parameters())
        engine.train_batch(iter(data))

        # reference: mean over micro-batches of (mse + COEF * sum of the aux losses), one SGD step
        for x, y in data:
            out = reference(x)
            aux = sum(l.last_aux for l in reference if isinstance(l, AuxLayer))
            ((F.mse_loss(out, y) + COEF * aux) / gas).backward()
        with torch.no_grad():
            for p in reference.parameters():
                p -= lr * p.grad

        ref = dict(reference.named_parameters())
        for name, p in engine.module.named_parameters():
            diff = (p.detach().cpu() - ref[name]).abs().max().item()
            assert diff < 1e-6, f"{name}: max |pipeline - reference| = {diff}"
        if engine.stage_id == 0:
            assert not torch.allclose(engine.module.get_submodule("0").gate.weight.detach().cpu(), gate_before)

    @pytest.mark.parametrize("precision", ["bf16", "fp16"])
    def test_aux_loss_is_scaled_like_the_loss_in_mixed_precision(self, precision):
        # The gates only get gradient from the aux loss, so their update shows whether the injected
        # gradient carries the same 1/gas and fp16 loss scale as the main loss: a missing factor of
        # 128 or 1/4 is far outside the tolerance, bf16/fp16 rounding is not.
        acc = get_accelerator()
        if precision == "fp16" and not acc.is_fp16_supported():
            pytest.skip("fp16 is not supported on this accelerator")
        if precision == "bf16" and not acc.is_bf16_supported():
            pytest.skip("bf16 is not supported on this accelerator")
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        gas, lr = 4, 1.0
        torch.manual_seed(0)
        layers = [AuxLayer(), AuxLayer(), nn.Linear(HIDDEN, HIDDEN), nn.Linear(HIDDEN, HIDDEN)]
        for layer in layers:
            layer.to(dtype)
        reference = copy.deepcopy(nn.Sequential(*layers)).float()
        for layer in reference:
            if isinstance(layer, AuxLayer):
                layer.attach = False
        gates_before = [layers[i].gate.weight.detach().float().clone() for i in (0, 1)]
        data = [(torch.randn(1, HIDDEN).to(dtype), torch.randn(1, HIDDEN).to(dtype)) for _ in range(gas)]

        config = pipeline_config(gas)
        config["optimizer"]["params"]["lr"] = lr
        config[precision] = {"enabled": True, "loss_scale": 128} if precision == "fp16" else {"enabled": True}
        model = PipelineModule(layers=layers, num_stages=2, loss_fn=nn.MSELoss(), partition_method="uniform")
        engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=model.parameters())
        device = acc.device_name()
        engine.train_batch(iter([(x.to(device), y.to(device)) for x, y in data]))

        for x, y in data:
            out = reference(x.float())
            aux = sum(l.last_aux for l in reference if isinstance(l, AuxLayer))
            ((F.mse_loss(out, y.float()) + COEF * aux) / gas).backward()
        if engine.stage_id == 0:
            for i, before in zip((0, 1), gates_before):
                moved = engine.module.get_submodule(str(i)).gate.weight.detach().float().cpu() - before
                expected = -lr * reference[i].gate.weight.grad
                assert (moved - expected).norm() / expected.norm() < 0.1, f"gate {i}"

    def test_legacy_moe_layer_on_every_stage(self):

        class MoEBlock(nn.Module):

            def __init__(self, coef):
                super().__init__()
                self.moe = MoE(hidden_size=HIDDEN,
                               expert=nn.Linear(HIDDEN, HIDDEN),
                               num_experts=2,
                               k=1,
                               use_rts=False,
                               drop_tokens=False,
                               aux_loss_coef=coef)

            def forward(self, x):
                return self.moe(x)[0]

        def run(coef):
            torch.manual_seed(0)
            model = PipelineModule(layers=[MoEBlock(coef), MoEBlock(coef)],
                                   num_stages=2,
                                   loss_fn=nn.MSELoss(),
                                   partition_method="uniform")
            engine, _, _, _ = deepspeed.initialize(config=pipeline_config(2),
                                                   model=model,
                                                   model_parameters=model.parameters())
            torch.manual_seed(1)
            data = [(torch.randn(1, 4, HIDDEN), torch.randn(1, 4, HIDDEN)) for _ in range(2)]
            engine.train_batch(iter(data))
            gate = [p for n, p in engine.module.named_parameters() if "gate" in n][0]
            return gate.detach().clone(), engine.agg_moe_aux_loss

        without_gate, without_aux = run(None)
        with_gate, with_aux = run(0.1)
        assert without_aux is None
        assert with_aux is not None and torch.isfinite(with_aux)
        assert not torch.allclose(with_gate, without_gate)


class TestExpertLocalLayoutWarning(DistributedTest):
    world_size = 4

    @pytest.mark.parametrize("hosts_per_node,warns", [(4, False), (1, True)])
    def test_warns_when_ep_group_spans_nodes(self, monkeypatch, hosts_per_node, warns):
        rank = dist.get_rank()
        monkeypatch.setattr(groups.socket, "gethostname", lambda: f"node{rank // hosts_per_node}")
        warnings = []
        monkeypatch.setattr(groups.logger, "warning", lambda msg, *a, **k: warnings.append(msg))
        groups._EXPERT_PARALLEL_GROUP.pop("ep_size_2", None)
        groups._EXPERT_DATA_PARALLEL_GROUP.pop("ep_size_2", None)
        groups._create_expert_and_data_parallel(2, pp_size=2)
        is_first_of_group = rank in (0, 2)
        assert bool(warnings) == (warns and is_first_of_group)
        if warnings:
            assert "spans 2 nodes" in warnings[0]
