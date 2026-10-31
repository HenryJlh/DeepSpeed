# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import copy

import pytest
import torch
import torch.nn as nn

import deepspeed
from deepspeed.pipe import PipelineModule
from deepspeed.accelerator import get_accelerator

from unit.common import DistributedTest

HIDDEN_DIM = 16


class Pre(nn.Linear):

    def __init__(self):
        super().__init__(HIDDEN_DIM, HIDDEN_DIM)


class Block(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)

    def forward(self, x):
        return x + torch.tanh(self.fc(x))


class Post(nn.Linear):

    def __init__(self):
        super().__init__(HIDDEN_DIM, HIDDEN_DIM)


def gpt_like_layers(blocks=8):
    # two leading and three trailing layers that are not partitionable, as in a GPT pipe model
    torch.manual_seed(0)
    return [Pre(), Pre()] + [Block() for _ in range(blocks)] + [Post(), Post(), Post()]


def stage_layer_types(model):
    return [type(f).__name__ for f in model.forward_funcs]


class TestCustomPartition(DistributedTest):
    world_size = 4

    def test_boundaries_follow_partitionable_layers(self):
        model = PipelineModule(layers=gpt_like_layers(),
                               num_stages=4,
                               partition_method="custom:1,3,2,2",
                               partition_layer_type="Block")
        # Block layers sit at 2..9: stage s starts at its first Block, except stage 0 at 0
        assert model.parts == [0, 3, 6, 8, 13]
        expected = [
            ["Pre", "Pre", "Block"],
            ["Block"] * 3,
            ["Block"] * 2,
            ["Block", "Block", "Post", "Post", "Post"],
        ]
        assert stage_layer_types(model) == expected[model.stage_id]

    def test_every_layer_counts_without_layer_type(self):
        model = PipelineModule(layers=gpt_like_layers(), num_stages=4, partition_method="custom:4,3,3,3")
        assert model.parts == [0, 4, 7, 10, 13]

    def test_matches_uniform_when_counts_are_uniform(self):
        layers = gpt_like_layers()
        custom = PipelineModule(layers=copy.deepcopy(layers), num_stages=4, partition_method="custom:4,3,3,3")
        uniform = PipelineModule(layers=copy.deepcopy(layers), num_stages=4, partition_method="uniform")
        assert custom.parts == uniform.parts

    @pytest.mark.parametrize("method,match", [
        ("custom:2,2,2", "3 entries for 4 stages"),
        ("custom:2,2,2,1", "assigns 7 layers, but there are 8"),
        ("custom:0,3,3,2", "at least one layer"),
        ("custom:2,a,2,4", "comma-separated list of ints"),
    ])
    def test_rejects_bad_counts(self, method, match):
        with pytest.raises(ValueError, match=match):
            PipelineModule(layers=gpt_like_layers(),
                           num_stages=4,
                           partition_method=method,
                           partition_layer_type="Block")


class TestCustomPartitionTwoStages(DistributedTest):
    world_size = 2

    def test_unmatched_layers_stay_with_the_layer_before(self):
        torch.manual_seed(0)
        layers = []
        for _ in range(4):
            layers += [Block(), nn.LayerNorm(HIDDEN_DIM)]
        model = PipelineModule(layers=layers,
                               num_stages=2,
                               partition_method="custom:2,2",
                               partition_layer_type="Block")
        assert model.parts == [0, 4, 8]
        assert stage_layer_types(model) == ["Block", "LayerNorm", "Block", "LayerNorm"]


def train_losses(layers, config, steps=3, **module_kwargs):
    model = PipelineModule(layers=copy.deepcopy(layers), num_stages=2, loss_fn=nn.MSELoss(), **module_kwargs)
    engine, _, _, _ = deepspeed.initialize(config=config,
                                           model=model,
                                           model_parameters=[p for p in model.parameters()])
    gen = torch.Generator().manual_seed(1)
    data = [(torch.randn(1, HIDDEN_DIM, generator=gen), torch.randn(1, HIDDEN_DIM, generator=gen)) for _ in range(4)]
    device = get_accelerator().device_name()
    data = [(x.to(device), y.to(device)) for x, y in data]
    losses = []
    for _ in range(steps):
        losses.append(engine.train_batch(iter(data)).item())
    norm = engine.get_global_grad_norm()
    return losses, None if norm is None else float(norm)


class TestCustomPartitionTraining(DistributedTest):
    world_size = 2

    def test_same_losses_as_uniform(self):
        config = {
            "train_batch_size": 4,
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
        layers = gpt_like_layers()
        reference, ref_norm = train_losses(layers, config, partition_method="uniform")
        custom, norm = train_losses(layers, config, partition_method="custom:3,5", partition_layer_type="Block")
        assert custom == pytest.approx(reference, rel=1e-5)
        assert norm == pytest.approx(ref_norm, rel=1e-5)


class TestCheckpointAcrossPartitions(DistributedTest):
    world_size = 2

    def test_uniform_checkpoint_loads_into_custom_partition(self, tmpdir):
        config = {
            "train_batch_size": 4,
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1000,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.01
                }
            },
        }
        gen = torch.Generator().manual_seed(1)
        device = get_accelerator().device_name()
        data = [(torch.randn(1, HIDDEN_DIM, generator=gen).to(device), torch.randn(1, HIDDEN_DIM,
                                                                                   generator=gen).to(device))
                for _ in range(4)]

        def engine_for(layers, **kwargs):
            model = PipelineModule(layers=layers, num_stages=2, loss_fn=nn.MSELoss(), **kwargs)
            engine, _, _, _ = deepspeed.initialize(config=config,
                                                   model=model,
                                                   model_parameters=[p for p in model.parameters()])
            return engine

        saved = engine_for(gpt_like_layers(), partition_method="uniform")
        saved.train_batch(iter(data))
        expected = saved.eval_batch(iter(data)).item()
        saved.save_checkpoint(tmpdir)

        torch.manual_seed(7)  # different weights, so only the checkpoint can make the losses agree
        fresh = [Pre(), Pre()] + [Block() for _ in range(8)] + [Post(), Post(), Post()]
        loaded = engine_for(fresh, partition_method="custom:3,5", partition_layer_type="Block")
        assert loaded.module.parts != saved.module.parts
        assert loaded.eval_batch(iter(data)).item() != pytest.approx(expected, rel=1e-3)
        loaded.load_checkpoint(tmpdir, load_optimizer_states=False, load_lr_scheduler_states=False)
        assert loaded.eval_batch(iter(data)).item() == pytest.approx(expected, rel=1e-5)
