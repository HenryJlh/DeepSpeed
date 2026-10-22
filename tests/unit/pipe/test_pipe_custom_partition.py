# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import copy

import pytest
import torch
import torch.nn as nn

import deepspeed
import deepspeed.comm as dist
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


class TestProfilePlan(DistributedTest):
    world_size = 2

    def write_plan(self, tmpdir):
        from deepspeed.runtime.pipe.planner import PipelinePlan, StagePlan
        plan = PipelinePlan(stages=[StagePlan(0, 3, 2, 0.0, 0.0),
                                    StagePlan(1, 5, 0, 0.0, 0.0)],
                            partition_layer_type="Block")
        path = f"{tmpdir}/plan_rank{dist.get_rank()}.json"
        plan.save(path)
        return path

    def test_plan_sets_partition_and_checkpointing(self, tmpdir):
        model = PipelineModule(layers=gpt_like_layers(),
                               num_stages=2,
                               partition_method=f"profile:{self.write_plan(tmpdir)}")
        assert model.parts == [0, 5, 13]
        assert model.partition_layer_type == "Block"
        assert model.checkpoint_layers_per_stage == [2, 0]

    def test_arguments_override_the_plan(self, tmpdir):
        model = PipelineModule(layers=gpt_like_layers(),
                               num_stages=2,
                               partition_method=f"profile:{self.write_plan(tmpdir)}",
                               checkpoint_layers_per_stage=[1, 1])
        assert model.checkpoint_layers_per_stage == [1, 1]

    def test_profile_without_plan_explains(self):
        with pytest.raises(NotImplementedError, match="deepspeed.runtime.pipe.planner"):
            PipelineModule(layers=gpt_like_layers(), num_stages=2, partition_method="profile")


class TestSelectiveCheckpointing(DistributedTest):
    world_size = 4

    def build(self, counts, interval=1, method="custom:1,3,2,2"):
        model = PipelineModule(layers=gpt_like_layers(),
                               num_stages=4,
                               partition_method=method,
                               partition_layer_type="Block",
                               activation_checkpoint_interval=interval,
                               checkpoint_layers_per_stage=counts)
        model._precompute_checkpointable_values()
        return model

    def segments(self, model):
        return [(s, e, c) for (s, e), c in zip(model.checkpoint_segment_bounds, model.is_checkpointable_results)]

    def test_first_layers_of_each_stage(self):
        model = self.build([1, 2, 0, 1])
        expected = [
            # Pre, Pre are not partitionable, so stage 0 checkpoints its single Block only
            [(0, 2, False), (2, 3, True)],
            [(0, 1, True), (1, 2, True), (2, 3, False)],
            [(0, 2, False)],
            [(0, 1, True), (1, 5, False)],
        ]
        assert self.segments(model) == expected[model.stage_id]

    def test_interval_caps_a_run(self):
        model = self.build([1, 3, 2, 2], interval=2)
        expected = [
            [(0, 2, False), (2, 3, True)],
            [(0, 2, True), (2, 3, True)],
            [(0, 2, True)],
            [(0, 2, True), (2, 5, False)],
        ]
        assert self.segments(model) == expected[model.stage_id]

    def test_interval_zero_enables_checkpointing(self):
        model = self.build([1, 1, 1, 1], interval=0)
        assert model.activation_checkpoint_interval == 1

    def test_works_with_uniform_partition(self):
        model = self.build([0, 0, 1, 0], method="uniform")
        # uniform gives parts [0, 4, 7, 10, 13]: stage 2 holds three Blocks, stage 3 only Post layers
        if model.stage_id == 2:
            assert self.segments(model) == [(0, 1, True), (1, 3, False)]
        else:
            assert not any(model.is_checkpointable_results)

    @pytest.mark.parametrize("counts,match", [
        ([1, 1, 1], "3 entries for 4 stages"),
        ([2, 0, 0, 0], r"must be an int in \[0, 1\]"),
        ([0, -1, 0, 0], r"must be an int in \[0, 3\]"),
    ])
    def test_rejects_bad_counts(self, counts, match):
        with pytest.raises(ValueError, match=match):
            self.build(counts)

    def test_setter_rebuilds_segments(self):
        model = self.build([0, 0, 0, 0])
        assert not any(model.is_checkpointable_results)
        model.set_checkpoint_layers_per_stage([1, 3, 2, 2])
        assert any(model.is_checkpointable_results)


class TestDualPipeVSelectiveCheckpointing(DistributedTest):
    world_size = 2

    def test_matches_sequential_model(self):
        from deepspeed.pipe import DualPipeVModule
        from deepspeed.utils import RepeatingLoader

        micro_batches, hidden, lr = 5, 8, 0.1
        config = {
            "train_batch_size": micro_batches,
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": micro_batches,
            "optimizer": {
                "type": "SGD",
                "params": {
                    "lr": lr
                }
            },
            "zero_allow_untested_optimizer": True,
        }
        torch.manual_seed(0)
        layers = []
        for _ in range(8):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        reference = copy.deepcopy(nn.Sequential(*layers))
        data = [(torch.randn(1, hidden), torch.randn(1, hidden)) for _ in range(micro_batches)]
        loss_fn = nn.MSELoss()

        # 2 ranks = 4 virtual stages; rank r runs stage r (phase 0) and stage 3 - r (phase 1)
        model = DualPipeVModule(layers=layers,
                                num_stages=2,
                                loss_fn=loss_fn,
                                partition_method="custom:3,1,2,2",
                                partition_layer_type="Linear",
                                checkpoint_layers_per_stage=[2, 1, 0, 2])
        engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=model.parameters())
        engine.set_dataiterator(RepeatingLoader(data))

        # each Linear is followed by its Tanh; checkpointed Linears pull in nothing else
        expected = {
            0: [[(0, 1, True), (1, 2, False), (2, 3, True), (3, 6, False)],
                [(0, 1, True), (1, 2, False), (2, 3, True), (3, 4, False)]],
            1: [[(0, 1, True), (1, 2, False)], [(0, 4, False)]],
        }[engine.stage_id]
        for phase in (0, 1):
            model.activate_phase(phase)
            assert [(s, e, c) for (s, e), c in zip(model.checkpoint_segment_bounds, model.is_checkpointable_results)
                    ] == expected[phase]

        for _ in range(2):
            loss = engine.train_batch()
            losses = []
            for x, y in data:
                ref_loss = loss_fn(reference(x), y)
                (ref_loss / micro_batches).backward()
                losses.append(ref_loss.detach())
            with torch.no_grad():
                for p in reference.parameters():
                    p -= lr * p.grad
                    p.grad = None
            assert torch.allclose(loss.cpu(), torch.stack(losses).mean(), atol=1e-5)


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
        custom, norm = train_losses(layers,
                                    config,
                                    partition_method="custom:3,5",
                                    partition_layer_type="Block",
                                    checkpoint_layers_per_stage=[3, 2])
        assert custom == pytest.approx(reference, rel=1e-5)
        assert norm == pytest.approx(ref_norm, rel=1e-5)

    def test_config_sets_checkpoint_layers_per_stage(self):
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
            "pipeline": {
                "checkpoint_layers_per_stage": [1, 1]
            },
        }
        model = PipelineModule(layers=gpt_like_layers(),
                               num_stages=2,
                               loss_fn=nn.MSELoss(),
                               partition_method="custom:4,4",
                               partition_layer_type="Block")
        engine, _, _, _ = deepspeed.initialize(config=config,
                                               model=model,
                                               model_parameters=[p for p in model.parameters()])
        assert engine.module.checkpoint_layers_per_stage == [1, 1]
        assert sum(engine.module.is_checkpointable_results) == 1
        dist.barrier()


class TestCheckpointSegments(DistributedTest):
    world_size = 1

    def test_results_without_segments_keep_interval_blocks(self):
        # code written before segments existed fills is_checkpointable_results itself, one per interval block
        layers = gpt_like_layers()
        reference = nn.Sequential(*copy.deepcopy(layers))
        model = PipelineModule(layers=layers, num_stages=1, activation_checkpoint_interval=4)
        model.is_checkpointable_results = [False, True, False, True]
        model.checkpoint_segment_bounds = []
        x = torch.randn(2, HIDDEN_DIM, device=get_accelerator().device_name())
        assert torch.allclose(model(x).cpu(), reference(x.cpu()), atol=1e-6)

    def test_mismatch_raises_instead_of_skipping_layers(self):
        model = PipelineModule(layers=gpt_like_layers(), num_stages=1, activation_checkpoint_interval=4)
        model._precompute_checkpointable_values()
        model.is_checkpointable_results = model.is_checkpointable_results[:-1]
        with pytest.raises(RuntimeError, match="checkpointing decisions"):
            model(torch.randn(2, HIDDEN_DIM, device=get_accelerator().device_name()))


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
        loaded = engine_for(fresh,
                            partition_method="custom:3,5",
                            partition_layer_type="Block",
                            checkpoint_layers_per_stage=[3, 2])
        assert loaded.module.parts != saved.module.parts
        assert loaded.eval_batch(iter(data)).item() != pytest.approx(expected, rel=1e-3)
        loaded.load_checkpoint(tmpdir, load_optimizer_states=False, load_lr_scheduler_states=False)
        assert loaded.eval_batch(iter(data)).item() == pytest.approx(expected, rel=1e-5)
