# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import itertools
import json
import random

import pytest
import torch
import torch.nn as nn

from deepspeed.runtime.pipe.planner import (LayerCost, PipelinePlan, StageMemoryModel, evaluate_partition,
                                            model_state_bytes, pipeline_step_ms, profile_layer, saved_activation_bytes,
                                            search_micro_batch_size, solve_stages)
from deepspeed.accelerator import get_accelerator
from deepspeed.pipe import PipelineModule
from deepspeed.runtime.pipe.planner import Calibration, fit_calibration, prediction_error, record_stage_memory
from deepspeed.runtime.pipe.planner.__main__ import main as planner_main
from deepspeed.runtime.pipe.planner.solver import _StageCosts

from unit.common import DistributedTest

GiB = 1024**3


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
                      transient_bytes=rng.uniform(0, 2) * GiB))
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


class TestCostModel:

    def test_xmoe4d_formula(self):
        layer = LayerCost.from_gemm_comm(gemm_ms=2.0, comm_ms=1.0)
        assert layer.time_ms(checkpointed=False) == pytest.approx(3 * 2.0 + 2 * 1.0)
        assert layer.time_ms(checkpointed=True) == pytest.approx(4 * 2.0 + 3 * 1.0)

    def test_scaling_keeps_model_states(self):
        layer = LayerCost(1.0, 2.0, activation_bytes=10, input_bytes=1, state_bytes=100).scaled(4)
        assert (layer.fwd_ms, layer.bwd_ms, layer.activation_bytes, layer.input_bytes, layer.state_bytes) == \
            (4.0, 8.0, 40, 4, 100)

    def test_state_bytes(self):
        assert model_state_bytes(10, zero_stage=0) == 10 * 16
        assert model_state_bytes(10, zero_stage=1, dp_size=4) == 10 * (2 + 2 + 3)
        with pytest.raises(ValueError):
            model_state_bytes(10, zero_stage=2)

    def test_inflight_follows_1f1b(self):
        layer = [LayerCost(1, 2, activation_bytes=10, input_bytes=1)]
        model = StageMemoryModel(num_stages=4, micro_batches=8)
        assert [model.stage_bytes(s, layer, 0) for s in range(4)] == [40, 30, 20, 10]
        # never more in flight than there are micro-batches
        assert StageMemoryModel(num_stages=4, micro_batches=2).stage_bytes(0, layer, 0) == 20


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
        model = StageMemoryModel(3, 5, overhead_bytes=0.5 * GiB, state_factor=1.2, activation_factor=0.9)
        cost = _StageCosts(layers, 3, model, float("inf"), extra, extra)
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
        stages = evaluate_partition(layers, [2, 2], [2, 0], float("inf"), micro_batches=4)
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

    def test_cli(self, tmp_path, capsys):
        costs = {
            "micro_batch_size": 1,
            "layer_types": {
                "dense": {
                    "fwd_ms": 1.0,
                    "bwd_ms": 2.0,
                    "activation_bytes": GiB,
                    "input_bytes": 0.05 * GiB,
                    "state_bytes": 0.2 * GiB
                },
                "moe": {
                    "fwd_gemm_ms": 1.5,
                    "fwd_comm_ms": 1.0,
                    "activation_bytes": 2 * GiB,
                    "input_bytes": 0.05 * GiB,
                    "state_bytes": 1.0 * GiB
                },
            },
            "layer_sequence": ["dense"] * 2 + ["moe"] * 10,
            "last_stage_extra": [{
                "fwd_ms": 4.0,
                "bwd_ms": 8.0,
                "activation_bytes": 2 * GiB,
                "state_bytes": 2 * GiB
            }],
        }
        cpath, ppath = tmp_path / "costs.json", tmp_path / "plan.json"
        cpath.write_text(json.dumps(costs))
        assert planner_main([
            "--costs",
            str(cpath), "--stages", "4", "--memory-gb", "16", "--global-batch-size", "32", "--dp", "2",
            "--micro-batch-sizes", "1,2", "--partition-layer-type", "Block", "-o",
            str(ppath)
        ]) == 0
        out = capsys.readouterr().out
        assert "uniform, full checkpointing" in out
        plan = PipelinePlan.load(ppath)
        assert sum(plan.layers_per_stage) == 12 and plan.partition_layer_type == "Block"


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

    def test_backward_transient_of_a_head(self):
        if get_accelerator().device_name() == "cpu":
            pytest.skip("the cpu accelerator's peak counter is the process RSS")
        tokens, vocab = 512, 8192
        device = get_accelerator().device_name()
        head = _HeadLoss(64, vocab).to(device)
        cost = profile_layer(head, (torch.randn(tokens, 64, device=device), ), warmup=1, iters=1)
        # the fp32 logits gradient exists only during the backward
        assert cost.transient_bytes >= tokens * vocab * 4


def synthetic_runs(layers, first, last, truth, partitions, micro_batches=8):
    """Calibration runs whose peaks a memory model with factors ``truth`` would measure exactly."""
    runs = []
    for counts, ckpt in partitions:
        model = StageMemoryModel(len(counts),
                                 micro_batches,
                                 overhead_bytes=truth.overhead_bytes,
                                 state_factor=truth.state_factor,
                                 activation_factor=truth.activation_factor)
        peaks, start = [], 0
        for s, (n, c) in enumerate(zip(counts, ckpt)):
            extras = (first if s == 0 else []) + (last if s == len(counts) - 1 else [])
            peaks.append(model.stage_bytes(s, layers[start:start + n], c, extras))
            start += n
        runs.append({
            "layers_per_stage": counts,
            "checkpoint_layers_per_stage": ckpt,
            "micro_batch_size": 1,
            "micro_batches": micro_batches,
            "peak_allocated_bytes": peaks,
            "peak_reserved_bytes": peaks,
        })
    return runs


class TestCalibration:
    partitions = [([3, 3, 3, 3], [3, 3, 3, 3]), ([2, 4, 4, 2], [1, 0, 0, 0]), ([4, 3, 3, 2], [2, 1, 0, 0])]

    def costs(self):
        rng = random.Random(3)
        layers = random_layers(rng, 12)
        first = [LayerCost(1, 2, activation_bytes=0.1 * GiB, state_bytes=4 * GiB)]
        last = [LayerCost(5, 10, activation_bytes=2 * GiB, state_bytes=4 * GiB, transient_bytes=3 * GiB)]
        return layers, first, last

    def test_recovers_the_factors_that_produced_the_peaks(self):
        layers, first, last = self.costs()
        truth = Calibration(state_factor=1.3, activation_factor=0.75, overhead_bytes=2.5 * GiB)
        runs = synthetic_runs(layers, first, last, truth, self.partitions)
        fit = fit_calibration(runs, lambda mbs: (layers, first, last))
        assert fit.state_factor == pytest.approx(1.3, rel=1e-6)
        assert fit.activation_factor == pytest.approx(0.75, rel=1e-6)
        assert fit.overhead_bytes == pytest.approx(2.5 * GiB, rel=1e-6)
        assert fit.fit_mae_bytes < 1e-3 * GiB and fit.loo_mae_bytes < 1e-3 * GiB
        assert fit.num_points == 12

    def test_factors_stay_non_negative(self):
        layers, first, last = self.costs()
        runs = synthetic_runs(layers, first, last, Calibration(), self.partitions)
        for run in runs:  # peaks below what the states alone need would ask for a negative overhead
            run["peak_reserved_bytes"] = [p - 20 * GiB for p in run["peak_reserved_bytes"]]
        fit = fit_calibration(runs, lambda mbs: (layers, first, last))
        assert min(fit.state_factor, fit.activation_factor, fit.overhead_bytes) >= 0

    def test_calibrated_prediction_matches_an_unseen_partition(self):
        layers, first, last = self.costs()
        truth = Calibration(state_factor=1.25, activation_factor=0.8, overhead_bytes=3 * GiB)
        seen = synthetic_runs(layers, first, last, truth, self.partitions[:2])
        unseen = synthetic_runs(layers, first, last, truth, self.partitions[2:])
        fit = fit_calibration(seen, lambda mbs: (layers, first, last))
        for predicted, measured in prediction_error(unseen, lambda mbs: (layers, first, last), fit)[0]:
            assert predicted == pytest.approx(measured, rel=1e-6)
        raw = prediction_error(unseen, lambda mbs: (layers, first, last))[0]
        assert any(abs(p - m) > GiB for p, m in raw)  # uncalibrated, the same prediction is off

    def test_cli_plans_with_the_calibrated_model(self, tmp_path, capsys):
        layers, first, last = self.costs()
        runs = synthetic_runs(layers, first, last, Calibration(state_factor=1.25, overhead_bytes=4 * GiB),
                              self.partitions)
        costs = {
            "layers": [l.to_dict() for l in layers],
            "first_stage_extra": [first[0].to_dict()],
            "last_stage_extra": [last[0].to_dict()]
        }
        cpath, rpath, ppath = tmp_path / "costs.json", tmp_path / "runs.json", tmp_path / "plan.json"
        cpath.write_text(json.dumps(costs))
        rpath.write_text(json.dumps(runs))
        args = [
            "--costs",
            str(cpath), "--stages", "4", "--memory-gb", "200", "--global-batch-size", "8", "--micro-batch-sizes", "1",
            "-o",
            str(ppath)
        ]
        assert planner_main(args + ["--calibration", str(rpath)]) == 0
        assert "state x1.250" in capsys.readouterr().out
        calibrated = PipelinePlan.load(ppath)
        assert calibrated.notes["calibration"]["overhead_bytes"] == pytest.approx(4 * GiB, rel=1e-6)
        assert planner_main(args) == 0
        # the calibrated model predicts more memory for the same partition
        assert max(s.memory_bytes for s in calibrated.stages) > max(s.memory_bytes
                                                                    for s in PipelinePlan.load(ppath).stages)


class TestRecordStageMemory(DistributedTest):
    world_size = 2

    def test_records_one_run_per_call(self, tmpdir):
        import deepspeed
        layers = [nn.Linear(8, 8) for _ in range(6)]
        model = PipelineModule(layers=layers,
                               num_stages=2,
                               loss_fn=nn.MSELoss(),
                               partition_method="custom:2,4",
                               checkpoint_layers_per_stage=[1, 0])
        config = {
            "train_batch_size": 4,
            "train_micro_batch_size_per_gpu": 1,
            "steps_per_print": 1000,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.01
                }
            }
        }
        engine, _, _, _ = deepspeed.initialize(config=config, model=model, model_parameters=list(model.parameters()))
        device = get_accelerator().device_name()
        engine.train_batch(iter([(torch.randn(1, 8, device=device), torch.randn(1, 8, device=device))] * 4))
        path = str(tmpdir / f"runs_{engine.global_rank}.json")
        for _ in range(2):
            run = record_stage_memory(engine, path, partition="custom:2,4")
        if engine.global_rank == 0:
            assert run["layers_per_stage"] == [2, 4]
            assert run["checkpoint_layers_per_stage"] == [1, 0]
            assert run["micro_batches"] == 4 and run["micro_batch_size"] == 1
            assert len(run["peak_reserved_bytes"]) == 2 and min(run["peak_allocated_bytes"]) > 0
            assert len(json.load(open(path))) == 2
        else:
            assert run is None
