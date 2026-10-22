# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Plan a pipeline partition from per-layer costs.

    python -m deepspeed.runtime.pipe.planner --costs costs.json --stages 4 --memory-gb 60 \\
        --global-batch-size 256 --dp 8 --micro-batch-sizes 1,2,4 -o plan.json

``costs.json`` lists the partitionable layers in order, either explicitly or by type::

    {"micro_batch_size": 1,
     "layer_types": {"dense": {"fwd_ms": 3.1, "bwd_ms": 6.0, "activation_bytes": 2.1e9,
                               "input_bytes": 1.6e7, "state_bytes": 3.2e9}, "moe": {...}},
     "layer_sequence": ["dense", "dense", "moe", ...],          # or "layers": [{...}, ...]
     "first_stage_extra": [{...}], "last_stage_extra": [{...}],  # embedding, head + loss
     "per_micro_batch_size": {"2": {...same keys...}}}           # optional, else costs scale linearly

Each entry has the fields of :class:`LayerCost`; ``fwd_gemm_ms`` + ``fwd_comm_ms`` may replace
``fwd_ms`` + ``bwd_ms`` to use the X-MoE-4D cost formula.
"""

import argparse
import json
import sys

from .cost import LayerCost
from .solver import evaluate_partition, pipeline_step_ms, search_micro_batch_size

GiB = 1024**3


def _cost(d):
    d = dict(d)
    if "fwd_gemm_ms" in d:
        return LayerCost.from_gemm_comm(d.pop("fwd_gemm_ms"), d.pop("fwd_comm_ms", 0.0), **d)
    return LayerCost.from_dict(d)


def _parse(section):
    if "layers" in section:
        layers = [_cost(d) for d in section["layers"]]
    else:
        types = {k: _cost({"name": k, **v}) for k, v in section["layer_types"].items()}
        layers = [types[t] for t in section["layer_sequence"]]
    first = [_cost(d) for d in section.get("first_stage_extra", [])]
    last = [_cost(d) for d in section.get("last_stage_extra", [])]
    return layers, first, last


def load_costs(spec):
    """Returns ``layer_costs(mbs) -> (layers, first_stage_extra, last_stage_extra)`` for a costs dict."""
    base_mbs = spec.get("micro_batch_size", 1)
    base = _parse(spec)
    per_mbs = {int(k): _parse({**spec, **v}) for k, v in spec.get("per_micro_batch_size", {}).items()}

    def layer_costs(mbs):
        if mbs in per_mbs:
            return per_mbs[mbs]
        f = mbs / base_mbs
        return tuple([c.scaled(f) for c in group] for group in base)

    return layer_costs


def _uniform(n, stages):
    q, r = divmod(n, stages)
    return [q + (1 if s < r else 0) for s in range(stages)]


def _report(name, stages, micro_batches, limit, overhead_ms):
    times = [s.time_ms for s in stages]
    step = pipeline_step_ms(times, micro_batches, overhead_ms)
    fits = all(s.memory_bytes <= limit for s in stages)
    print(f"{name:<28} step {step:10.1f} ms  slowest stage {max(times):8.1f} ms  "
          f"peak {max(s.memory_bytes for s in stages) / GiB:7.2f} GiB  {'fits' if fits else 'OOM'}")
    for s in stages:
        print(f"    stage {s.stage_id}: {s.num_layers:3d} layers, {s.num_checkpointed:3d} checkpointed, "
              f"{s.time_ms:9.2f} ms, {s.memory_bytes / GiB:7.2f} GiB")
    return step


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--costs", required=True, help="per-layer costs (JSON)")
    p.add_argument("--stages", type=int, required=True)
    p.add_argument("--memory-gb", type=float, required=True, help="memory budget per GPU, GiB")
    p.add_argument("--global-batch-size", type=int, required=True)
    p.add_argument("--dp", type=int, default=1, help="data-parallel replicas of the pipeline")
    p.add_argument("--micro-batch-sizes", default="1", help="comma-separated candidates")
    p.add_argument("--inflight", default="1f1b", choices=["1f1b", "gpipe"])
    p.add_argument("--overhead-gb", type=float, default=0.0, help="per-GPU memory reserved outside the model")
    p.add_argument("--per-micro-batch-overhead-ms", type=float, default=0.0)
    p.add_argument("--partition-layer-type", default=None, help="stored in the plan for PipelineModule")
    p.add_argument("-o", "--output", default="pipeline_plan.json")
    a = p.parse_args(argv)

    with open(a.costs) as f:
        layer_costs = load_costs(json.load(f))
    limit = a.memory_gb * GiB
    plan = search_micro_batch_size(layer_costs, [int(x) for x in a.micro_batch_sizes.split(",")],
                                   a.global_batch_size,
                                   a.dp,
                                   a.stages,
                                   limit,
                                   inflight=a.inflight,
                                   overhead_bytes=a.overhead_gb * GiB,
                                   per_micro_batch_overhead_ms=a.per_micro_batch_overhead_ms,
                                   partition_layer_type=a.partition_layer_type)
    if plan is None:
        print("no partition fits the memory budget for any candidate micro-batch size", file=sys.stderr)
        return 1

    layers, first, last = layer_costs(plan.micro_batch_size)
    n = len(layers)
    print(f"micro-batch size {plan.micro_batch_size}, {plan.micro_batches} micro-batches, {n} layers, "
          f"{a.stages} stages, budget {a.memory_gb} GiB")
    _report("plan", plan.stages, plan.micro_batches, limit, a.per_micro_batch_overhead_ms)
    for label, ckpt in (("uniform, no checkpointing", lambda k: 0), ("uniform, full checkpointing", lambda k: k)):
        counts = _uniform(n, a.stages)
        stages = evaluate_partition(layers,
                                    counts, [ckpt(k) for k in counts],
                                    limit,
                                    plan.micro_batches,
                                    first,
                                    last,
                                    inflight=a.inflight,
                                    overhead_bytes=a.overhead_gb * GiB)
        _report(label, stages, plan.micro_batches, limit, a.per_micro_batch_overhead_ms)

    plan.notes = {"costs": a.costs, "memory_gb": a.memory_gb, "inflight": a.inflight}
    plan.save(a.output)
    print(f"wrote {a.output}: partition_method='profile:{a.output}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
