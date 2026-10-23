# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Calibrate the planner's memory model against measured stage peaks.

Per-layer costs cannot see everything that occupies a GPU: optimizer-step temporaries, allocator
fragmentation, communication buffers. X-MoE-4D (SC'26) covers them with a fixed overhead chosen by
hand for its model and hardware; this module measures instead. Run a couple of short training
jobs with different partitions, record each stage's peak with :func:`record_stage_memory`, and fit

    peak = state_factor * states + activation_factor * dynamic + overhead

(``StageMemoryModel``) with :func:`fit_calibration`. The planner then plans with the fitted model
(``python -m deepspeed.runtime.pipe.planner ... --calibration runs.json``).
"""

import itertools
import json
import os

from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator

from .memory import Calibration, StageMemoryModel

TARGETS = ("reserved", "allocated")


def record_stage_memory(engine, path=None, **notes):
    """Each pipeline stage's peak memory so far, as one calibration run.

    Collective: every rank must call it. The peak of a stage is the maximum over its data-parallel
    replicas. Rank 0 returns the run and, if ``path`` is given, appends it to that JSON file; the
    other ranks return None. Call it after a few training steps of the partition to measure.
    """
    acc = get_accelerator()
    module = engine.module
    local = (engine.stage_id, acc.max_memory_allocated(), acc.max_memory_reserved())
    peaks = [None] * dist.get_world_size()
    dist.all_gather_object(peaks, local)
    if dist.get_rank() != 0:
        return None
    num_stages = module.num_stages
    allocated, reserved = [0] * num_stages, [0] * num_stages
    for stage, alloc, resv in peaks:
        allocated[stage] = max(allocated[stage], alloc)
        reserved[stage] = max(reserved[stage], resv)
    run = {
        "layers_per_stage": [module._partitionable_count(s) for s in range(num_stages)],
        "checkpoint_layers_per_stage": list(module.checkpoint_layers_per_stage or [0] * num_stages),
        "micro_batch_size": engine.train_micro_batch_size_per_gpu(),
        "micro_batches": engine.micro_batches,
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
        "notes": notes,
    }
    if path is not None:
        runs = load_runs(path) if os.path.exists(path) else []
        with open(path, "w") as f:
            json.dump(runs + [run], f, indent=2)
    return run


def load_runs(path):
    with open(path) as f:
        return json.load(f)


def _points(runs, layer_costs, inflight, target):
    """One (states, dynamic, measured) triple per stage per run."""
    points = []
    for run in runs:
        layers, first, last = layer_costs(run["micro_batch_size"])
        counts = run["layers_per_stage"]
        num_stages = len(counts)
        if sum(counts) != len(layers):
            raise ValueError(f"calibration run has {sum(counts)} layers, the costs describe {len(layers)}")
        model = StageMemoryModel(num_stages, run["micro_batches"], inflight=inflight)
        start = 0
        for s, (n, c) in enumerate(zip(counts, run["checkpoint_layers_per_stage"])):
            extras = (list(first) if s == 0 else []) + (list(last) if s == num_stages - 1 else [])
            states, dynamic = model.components(s, layers[start:start + n], c, extras)
            points.append((states, dynamic, float(run[f"peak_{target}_bytes"][s])))
            start += n
    return points


def _least_squares(points):
    """Non-negative (state_factor, activation_factor, overhead) minimizing the squared error.

    With three unknowns, trying every subset of them held at zero and keeping the best non-negative
    unconstrained solution is exact.
    """
    # Columns are bytes (~1e10) next to a constant 1; scale each to unit magnitude, or the normal
    # equations lose most of their precision.
    scales = [max(abs(p[0]) for p in points) or 1.0, max(abs(p[1]) for p in points) or 1.0, 1.0]
    y_scale = max(abs(p[2]) for p in points) or 1.0
    best = None
    for free in itertools.product((True, False), repeat=3):
        idx = [i for i in range(3) if free[i]]
        if not idx:
            continue
        rows = [[(p[0], p[1], 1.0)[i] / scales[i] for i in idx] for p in points]
        ys = [p[2] / y_scale for p in points]
        sol = _solve_normal_equations(rows, ys)
        if sol is None or any(v < 0 for v in sol):
            continue
        coef = [0.0, 0.0, 0.0]
        for i, v in zip(idx, sol):
            coef[i] = v * y_scale / scales[i]
        err = sum((coef[0] * p[0] + coef[1] * p[1] + coef[2] - p[2])**2 for p in points)
        if best is None or err < best[1]:
            best = (coef, err)
    if best is None:
        raise ValueError("could not fit the memory model to the calibration runs")
    return best[0]


def _solve_normal_equations(rows, ys):
    k = len(rows[0])
    ata = [[sum(r[i] * r[j] for r in rows) for j in range(k)] for i in range(k)]
    aty = [sum(r[i] * y for r, y in zip(rows, ys)) for i in range(k)]
    # Gaussian elimination with partial pivoting on the k x k system (k <= 3)
    m = [row[:] + [b] for row, b in zip(ata, aty)]
    for col in range(k):
        pivot = max(range(col, k), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12 * max(1.0, max(abs(v) for row in ata for v in row)):
            return None
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(k):
            if r != col:
                f = m[r][col] / m[col][col]
                m[r] = [a - f * b for a, b in zip(m[r], m[col])]
    return [m[i][k] / m[i][i] for i in range(k)]


def _mae(coef, points):
    return sum(abs(coef[0] * p[0] + coef[1] * p[1] + coef[2] - p[2]) for p in points) / len(points)


def fit_calibration(runs, layer_costs, inflight="1f1b", target="reserved"):
    """Fit ``Calibration`` factors to measured stage peaks.

    ``runs``: records from :func:`record_stage_memory`. ``layer_costs(mbs)`` returns
    ``(layers, first_stage_extra, last_stage_extra)``, as for the planner. ``target`` picks the peak
    to model: ``"reserved"`` (what the allocator holds, which is what runs out) or ``"allocated"``.
    With two or more runs, the leave-one-run-out error estimates the error on an unseen partition.
    """
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    points = _points(runs, layer_costs, inflight, target)
    coef = _least_squares(points)
    loo = float("nan")
    if len(runs) > 1:
        errors = []
        for i in range(len(runs)):
            held_out = _points([runs[i]], layer_costs, inflight, target)
            rest = _points(runs[:i] + runs[i + 1:], layer_costs, inflight, target)
            errors.append(_mae(_least_squares(rest), held_out))
        loo = sum(errors) / len(errors)
    return Calibration(state_factor=coef[0],
                       activation_factor=coef[1],
                       overhead_bytes=coef[2],
                       target=target,
                       fit_mae_bytes=_mae(coef, points),
                       loo_mae_bytes=loo,
                       num_points=len(points))


def prediction_error(runs, layer_costs, calibration=None, inflight="1f1b", target="reserved"):
    """Per-stage (predicted, measured) bytes for each run under ``calibration`` (None: uncalibrated)."""
    cal = calibration or Calibration()
    out = []
    for run in runs:
        stages = []
        for states, dynamic, measured in _points([run], layer_costs, inflight, target):
            predicted = cal.state_factor * states + cal.activation_factor * dynamic + cal.overhead_bytes
            stages.append((predicted, measured))
        out.append(stages)
    return out
