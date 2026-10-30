# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import torch.nn as nn

import deepspeed
from deepspeed.moe.layer import MoE
from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
from deepspeed.pipe import PipelineModule
from deepspeed.accelerator import get_accelerator

from unit.common import DistributedTest

HIDDEN = 8
EXPERTS = 2


class MoEBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.moe = MoE(hidden_size=HIDDEN, expert=nn.Linear(HIDDEN, HIDDEN), num_experts=EXPERTS, ep_size=1, k=1)

    def forward(self, x):
        return x + self.moe(x)[0]


def train(max_group_size, steps=2):
    torch.manual_seed(0)
    # 3 MoE blocks on the first stage, 1 on the second
    layers = [MoEBlock() for _ in range(4)] + [nn.Linear(HIDDEN, HIDDEN)]
    model = PipelineModule(layers=layers, num_stages=2, loss_fn=nn.MSELoss(), partition_method="uniform")
    group = {"params": list(model.parameters()), "name": "params"}
    groups = split_params_into_different_moe_groups_for_optimizer(group, max_group_size=max_group_size)
    config = {
        "train_batch_size": 2,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_clipping": 1.0,
        "steps_per_print": 1000,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.01
            }
        },
        "zero_optimization": {
            "stage": 1
        },
    }
    engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=groups)
    assert engine.module.parts == [0, 3, 5]
    device = get_accelerator().device_name()
    gen = torch.Generator().manual_seed(1)
    data = [(torch.randn(1, 4, HIDDEN, generator=gen).to(device), torch.randn(1, 4, HIDDEN, generator=gen).to(device))
            for _ in range(2)]
    losses, norms = [], []
    for _ in range(steps):
        losses.append(engine.train_batch(iter(data)).item())
        norms.append(float(engine.get_global_grad_norm()))
    return losses, norms, len(engine.optimizer.param_groups)


class TestPipeMoEGradNorm(DistributedTest):
    world_size = 2

    def test_stages_with_different_numbers_of_expert_groups(self):
        # One group per expert group: both stages hold the same groups.
        ref_losses, ref_norms, _ = train(max_group_size=None)
        # Expert parameters chunked one MoE block per group, as max_group_size does for large models:
        # the first stage gets 3 expert groups and the second 1. The grad-norm reduction across the
        # stages must not depend on the number of groups on each stage.
        losses, norms, num_groups = train(max_group_size=EXPERTS * (HIDDEN * HIDDEN + HIDDEN))
        assert num_groups == (4 if deepspeed.comm.get_rank() == 0 else 2)
        assert losses == pytest.approx(ref_losses, rel=1e-5)
        assert norms == pytest.approx(ref_norms, rel=1e-5)
