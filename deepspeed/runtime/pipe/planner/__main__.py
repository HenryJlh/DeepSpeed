# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Plan a pipeline partition from per-layer costs.

    python -m deepspeed.runtime.pipe.planner --costs costs.json --stages 4 --memory-gb 64 \\
        --global-batch-size 256 --dp 8 --micro-batch-sizes 1,2,4 --partition-layer-type Block -o plan.json

``costs.json`` is written by :func:`~deepspeed.runtime.pipe.planner.save_costs` from
:func:`~deepspeed.runtime.pipe.planner.profile_layer` measurements (see :func:`load_costs` for the
format). The plan keeps every stage's predicted peak within ``--memory-gb`` minus ``--headroom-gb``,
is compared with uniform partitions, and is written for ``partition_method="profile:plan.json"``.
"""

import argparse
import sys

from .cost import load_costs
from .solver import evaluate_partition, pipeline_step_ms, search_micro_batch_size

GiB = 1024**3

# Share of the device memory left out of the plan by default, for what per-layer costs cannot see:
# allocator fragmentation and communication buffers. On 4x MI210 (64 GiB) every plan measured ran with
# 6 GiB left out, while plans with 3 GiB ran out of memory; the stages ran 0.3-2.2 GiB above prediction.
DEFAULT_HEADROOM_FRACTION = 0.10


def _uniform(n, stages):
    q, r = divmod(n, stages)
    return [q + (1 if s < r else 0) for s in range(stages)]


def _report(name, stages, micro_batches, budget):
    times = [s.time_ms for s in stages]
    fits = all(s.memory_bytes <= budget for s in stages)
    print(f"{name:<28} step {pipeline_step_ms(times, micro_batches):10.1f} ms  slowest stage {max(times):8.1f} ms  "
          f"peak {max(s.memory_bytes for s in stages) / GiB:7.2f} GiB  {'fits' if fits else 'exceeds the budget'}")
    for s in stages:
        print(f"    stage {s.stage_id}: {s.num_layers:3d} layers, {s.num_checkpointed:3d} checkpointed, "
              f"{s.time_ms:9.2f} ms, {s.memory_bytes / GiB:7.2f} GiB")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--costs", required=True, help="per-layer costs (JSON, see save_costs)")
    p.add_argument("--stages", type=int, required=True, help="pipeline stages")
    p.add_argument("--memory-gb", type=float, required=True, help="device memory per GPU, GiB")
    p.add_argument("--global-batch-size", type=int, required=True)
    p.add_argument("--dp", type=int, default=1, help="data-parallel replicas of the pipeline")
    p.add_argument("--micro-batch-sizes", default="1", help="comma-separated candidates")
    p.add_argument("--headroom-gb",
                   type=float,
                   default=None,
                   help="memory per GPU left out of the plan for allocator fragmentation and communication "
                   f"buffers (default {DEFAULT_HEADROOM_FRACTION:.0%} of --memory-gb); "
                   "raise it if a plan runs out of memory")
    p.add_argument("--partition-layer-type", default=None, help="regex of the partitionable layers' class names")
    p.add_argument("-o", "--output", default="pipeline_plan.json")
    a = p.parse_args(argv)

    layer_costs = load_costs(a.costs)
    headroom_gb = a.memory_gb * DEFAULT_HEADROOM_FRACTION if a.headroom_gb is None else a.headroom_gb
    budget = (a.memory_gb - headroom_gb) * GiB
    plan = search_micro_batch_size(layer_costs, [int(x) for x in a.micro_batch_sizes.split(",")],
                                   a.global_batch_size,
                                   a.dp,
                                   a.stages,
                                   budget,
                                   partition_layer_type=a.partition_layer_type)
    if plan is None:
        print(
            f"no partition fits {budget / GiB:.1f} GiB ({a.memory_gb} GiB - {headroom_gb:.1f} GiB headroom) "
            "for any candidate micro-batch size",
            file=sys.stderr)
        return 1

    layers, first, last = layer_costs(plan.micro_batch_size)
    print(f"micro-batch size {plan.micro_batch_size}, {plan.micro_batches} micro-batches, {len(layers)} layers, "
          f"{a.stages} stages, {budget / GiB:.1f} GiB per GPU ({a.memory_gb} GiB - {headroom_gb:.1f} GiB headroom)")
    _report("plan", plan.stages, plan.micro_batches, budget)
    for label, ckpt in (("uniform, no checkpointing", lambda k: 0), ("uniform, full checkpointing", lambda k: k)):
        counts = _uniform(len(layers), a.stages)
        stages = evaluate_partition(layers, counts, [ckpt(k) for k in counts], plan.micro_batches, first, last)
        _report(label, stages, plan.micro_batches, budget)

    plan.notes = {"costs": str(a.costs), "memory_gb": a.memory_gb, "headroom_gb": headroom_gb}
    plan.save(a.output)
    print(f"wrote {a.output}: partition_method='profile:{a.output}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
