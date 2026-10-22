# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Memory-aware pipeline partitioning with per-stage selective checkpointing.

This is the ELM-PP planner of X-MoE-4D (SC'26): choose how many layers each stage gets and how
many of them it recomputes so that the slowest stage is as fast as possible while every stage
fits in memory. For a stage, recomputing more layers only adds time, so it checkpoints the fewest
layers that fit; a minimax dynamic program over the cut points then balances the stages.
"""

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Callable, List, Optional, Sequence

from .cost import LayerCost
from .memory import StageMemoryModel

PLAN_VERSION = 1
INF = float("inf")


@dataclass
class StagePlan:
    stage_id: int
    num_layers: int
    num_checkpointed: int
    time_ms: float
    memory_bytes: float


@dataclass
class PipelinePlan:
    stages: List[StagePlan]
    micro_batch_size: int = 1
    micro_batches: int = 1
    step_ms: float = 0.0
    samples_per_sec: float = 0.0
    partition_layer_type: Optional[str] = None
    notes: dict = field(default_factory=dict)

    @property
    def layers_per_stage(self):
        return [s.num_layers for s in self.stages]

    @property
    def checkpoint_layers_per_stage(self):
        return [s.num_checkpointed for s in self.stages]

    @property
    def partition_method(self):
        """The ``PipelineModule`` partition method that executes this plan."""
        return "custom:" + ",".join(str(n) for n in self.layers_per_stage)

    def to_dict(self):
        return {
            "version": PLAN_VERSION,
            "micro_batch_size": self.micro_batch_size,
            "micro_batches": self.micro_batches,
            "layers_per_stage": self.layers_per_stage,
            "checkpoint_layers_per_stage": self.checkpoint_layers_per_stage,
            "partition_layer_type": self.partition_layer_type,
            "predicted_step_ms": self.step_ms,
            "predicted_samples_per_sec": self.samples_per_sec,
            "stages": [asdict(s) for s in self.stages],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d):
        if d.get("version", PLAN_VERSION) != PLAN_VERSION:
            raise ValueError(f"unsupported pipeline plan version {d.get('version')}")
        if "stages" in d:
            stages = [StagePlan(**s) for s in d["stages"]]
        else:  # a hand-written plan may only give the two lists
            stages = [
                StagePlan(i, n, c, 0.0, 0.0)
                for i, (n, c) in enumerate(zip(d["layers_per_stage"], d["checkpoint_layers_per_stage"]))
            ]
        return cls(stages=stages,
                   micro_batch_size=d.get("micro_batch_size", 1),
                   micro_batches=d.get("micro_batches", 1),
                   step_ms=d.get("predicted_step_ms", 0.0),
                   samples_per_sec=d.get("predicted_samples_per_sec", 0.0),
                   partition_layer_type=d.get("partition_layer_type"),
                   notes=d.get("notes", {}))

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))


def pipeline_step_ms(stage_ms, micro_batches, per_micro_batch_overhead_ms=0.0, optimizer_ms=0.0):
    """1F1B step time: one micro-batch fills and drains the pipeline, the rest run at the slowest stage."""
    times = [t + per_micro_batch_overhead_ms for t in stage_ms]
    return sum(times) + (micro_batches - 1) * max(times) + optimizer_ms


class _StageCosts:
    """Cheapest feasible way to run layers [a, b) as stage s, with prefix sums over the layer list."""

    def __init__(self, layers, num_stages, memory_model, memory_limit_bytes, first_stage_extra, last_stage_extra):
        self.layers = layers
        self.num_stages = num_stages
        self.memory_model = memory_model
        self.limit = memory_limit_bytes
        self.first = list(first_stage_extra or [])
        self.last = list(last_stage_extra or [])
        self.cache = {}

    def extras(self, stage_id):
        extras = []
        if stage_id == 0:
            extras += self.first
        if stage_id == self.num_stages - 1:
            extras += self.last
        return extras

    def __call__(self, stage_id, a, b):
        key = (stage_id, a, b)
        if key not in self.cache:
            self.cache[key] = self._solve(stage_id, a, b)
        return self.cache[key]

    def _solve(self, stage_id, a, b):
        # Same quantities as StageMemoryModel.stage_bytes, updated in O(1) per extra checkpointed layer.
        layers = self.layers[a:b]
        extras = self.extras(stage_id)
        model = self.memory_model
        inflight = model.inflight(stage_id, model.num_stages, model.micro_batches)
        states = sum(l.state_bytes for l in layers) + sum(e.state_bytes for e in extras)
        kept = sum(l.activation_bytes for l in layers) + sum(e.activation_bytes for e in extras)
        time = sum(l.time_ms() for l in layers) + sum(e.time_ms() for e in extras)
        transient = 0.0
        for c in range(len(layers) + 1):
            if c > 0:
                layer = layers[c - 1]
                kept += layer.input_bytes - layer.activation_bytes
                transient = max(transient, layer.activation_bytes)
                time += layer.fwd_ms
            memory = states + inflight * kept + transient + model.overhead_bytes
            if memory <= self.limit:
                return time, c, memory
        return INF, len(layers), INF


def solve_stages(layers: Sequence[LayerCost],
                 num_stages: int,
                 memory_limit_bytes: float,
                 micro_batches: int,
                 first_stage_extra: Sequence[LayerCost] = (),
                 last_stage_extra: Sequence[LayerCost] = (),
                 inflight="1f1b",
                 overhead_bytes: float = 0.0) -> Optional[List[StagePlan]]:
    """Split ``layers`` (the partitionable layers, in order) into ``num_stages`` contiguous stages.

    Minimizes the slowest stage's time over all splits that give every stage at least one layer
    and keep every stage within ``memory_limit_bytes``. Returns None if no split fits.
    """
    n = len(layers)
    if n < num_stages:
        raise ValueError(f"{n} layers cannot fill {num_stages} stages")
    model = StageMemoryModel(num_stages, micro_batches, inflight=inflight, overhead_bytes=overhead_bytes)
    cost = _StageCosts(list(layers), num_stages, model, memory_limit_bytes, first_stage_extra, last_stage_extra)

    # best[k][i]: slowest-stage time when the first i layers form stages 0..k-1
    best = [[INF] * (n + 1) for _ in range(num_stages + 1)]
    cut = [[-1] * (n + 1) for _ in range(num_stages + 1)]
    best[0][0] = 0.0
    for k in range(1, num_stages + 1):
        stage_id = k - 1
        # leave at least one layer for every later stage
        for i in range(k, n - (num_stages - k) + 1):
            for j in range(k - 1, i):
                if best[k - 1][j] == INF:
                    continue
                t = max(best[k - 1][j], cost(stage_id, j, i)[0])
                if t < best[k][i]:
                    best[k][i], cut[k][i] = t, j
    if best[num_stages][n] == INF:
        return None

    plans, i = [], n
    for k in range(num_stages, 0, -1):
        j = cut[k][i]
        time, c, memory = cost(k - 1, j, i)
        plans.append(StagePlan(k - 1, i - j, c, time, memory))
        i = j
    return plans[::-1]


def search_micro_batch_size(layer_costs: Callable[[int], tuple],
                            candidates: Sequence[int],
                            global_batch_size: int,
                            data_parallel_size: int,
                            num_stages: int,
                            memory_limit_bytes: float,
                            inflight="1f1b",
                            overhead_bytes: float = 0.0,
                            per_micro_batch_overhead_ms: float = 0.0,
                            optimizer_ms: float = 0.0,
                            partition_layer_type: Optional[str] = None) -> Optional[PipelinePlan]:
    """Plan for every micro-batch size in ``candidates`` and keep the fastest step.

    ``layer_costs(mbs)`` returns ``(layers, first_stage_extra, last_stage_extra)`` for that micro-batch
    size. Sizes that do not divide ``global_batch_size`` over ``data_parallel_size`` ranks are skipped.
    """
    best = None
    for mbs in candidates:
        if mbs <= 0 or global_batch_size % (mbs * data_parallel_size):
            continue
        micro_batches = global_batch_size // (mbs * data_parallel_size)
        layers, first, last = layer_costs(mbs)
        stages = solve_stages(layers,
                              num_stages,
                              memory_limit_bytes,
                              micro_batches,
                              first_stage_extra=first,
                              last_stage_extra=last,
                              inflight=inflight,
                              overhead_bytes=overhead_bytes)
        if stages is None:
            continue
        step = pipeline_step_ms([s.time_ms for s in stages], micro_batches, per_micro_batch_overhead_ms, optimizer_ms)
        plan = PipelinePlan(stages=stages,
                            micro_batch_size=mbs,
                            micro_batches=micro_batches,
                            step_ms=step,
                            samples_per_sec=global_batch_size / (step / 1000.0) if step > 0 else math.inf,
                            partition_layer_type=partition_layer_type)
        if best is None or plan.step_ms < best.step_ms:
            best = plan
    return best


def evaluate_partition(layers,
                       layers_per_stage,
                       checkpoint_layers_per_stage,
                       memory_limit_bytes,
                       micro_batches,
                       first_stage_extra=(),
                       last_stage_extra=(),
                       inflight="1f1b",
                       overhead_bytes=0.0):
    """Predicted time and memory of a given partition, e.g. a uniform baseline, under the same model."""
    num_stages = len(layers_per_stage)
    model = StageMemoryModel(num_stages, micro_batches, inflight=inflight, overhead_bytes=overhead_bytes)
    cost = _StageCosts(list(layers), num_stages, model, memory_limit_bytes, first_stage_extra, last_stage_extra)
    stages, start = [], 0
    for s, (n, c) in enumerate(zip(layers_per_stage, checkpoint_layers_per_stage)):
        chunk = layers[start:start + n]
        extras = cost.extras(s)
        memory = model.stage_bytes(s, chunk, c, extras)
        time = sum(e.time_ms() for e in extras) + sum(layer.time_ms(i < c) for i, layer in enumerate(chunk))
        stages.append(StagePlan(s, n, c, time, memory))
        start += n
    return stages
