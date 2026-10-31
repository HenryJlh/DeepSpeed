# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Per-layer costs used by the pipeline planner, and the costs file that stores them."""

import json
from dataclasses import dataclass, asdict, replace


@dataclass(frozen=True)
class LayerCost:
    """What one layer costs one pipeline rank for one micro-batch.

    Args:
        fwd_ms: forward time.
        bwd_ms: backward time.
        activation_bytes: memory the forward keeps alive for the backward when the layer is not checkpointed.
        input_bytes: memory kept when the layer is checkpointed, i.e. its input.
        state_bytes: resident model states (parameters, gradients and optimizer states) of the layer.
        transient_bytes: memory its backward needs only while it runs (gradient temporaries, the fp32
            logits gradient of a head), on top of what it keeps between forward and backward.
        step_bytes: memory the optimizer step needs for the layer's dense parameters on top of the model
            states; the step processes one parameter group at a time, see :func:`optimizer_step_bytes`.
        expert_step_bytes: the same for its expert parameters, which form a separate group.
        name: free-form label, e.g. the layer type.
    """
    fwd_ms: float
    bwd_ms: float
    activation_bytes: float = 0.0
    input_bytes: float = 0.0
    state_bytes: float = 0.0
    transient_bytes: float = 0.0
    step_bytes: float = 0.0
    expert_step_bytes: float = 0.0
    name: str = ""

    def time_ms(self, checkpointed=False):
        """A checkpointed layer runs its forward a second time before the backward."""
        return self.fwd_ms + self.bwd_ms + (self.fwd_ms if checkpointed else 0.0)

    def kept_bytes(self, checkpointed=False):
        return self.input_bytes if checkpointed else self.activation_bytes

    def scaled(self, factor):
        """Time and activations grow linearly with the micro-batch size; model states and the
        optimizer step do not."""
        return replace(self,
                       fwd_ms=self.fwd_ms * factor,
                       bwd_ms=self.bwd_ms * factor,
                       activation_bytes=self.activation_bytes * factor,
                       input_bytes=self.input_bytes * factor,
                       transient_bytes=self.transient_bytes * factor)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def model_state_bytes(num_params, dp_size=1):
    """Resident model-state bytes of ``num_params`` parameters on one rank under bf16 + ZeRO-1 + Adam.

    2-byte parameters and gradients are replicated; the 12 bytes of fp32 master weights and Adam
    moments are sharded over the ``dp_size`` ranks that hold the same parameters (for expert
    parameters, the expert data-parallel group). This is the model-state term of X-MoE-4D (SC'26).
    Other precisions, ZeRO stages and optimizers keep different states; pass ``state_bytes`` to
    :func:`~deepspeed.runtime.pipe.planner.profile_layer` for them.
    """
    return num_params * (2 + 2 + 12 / dp_size)


def optimizer_step_bytes(num_params, dp_size=1):
    """Memory the bf16 + ZeRO-1 + Adam optimizer step needs for a parameter group, on top of the model states.

    6 bytes per parameter of the group's partition (the group over ``dp_size``), from the step of
    ``DeepSpeedZeroOptimizer`` with bf16 gradients: it flattens the group's bf16 gradients into a new
    tensor (2 bytes) and casts that to an fp32 gradient partition (4 bytes) while the bf16 gradients are
    still allocated. It then frees the flat copy and the partition's bf16 gradients before Adam runs, so
    Adam's temporaries fit in the same 6 bytes, and frees the fp32 partition before the next group.
    The fp32 partition is one tensor, so its 4 bytes must be contiguous.
    """
    return num_params * 6 / dp_size


def save_costs(path, layer_types, layer_sequence, first_stage_extra=(), last_stage_extra=(), micro_batch_size=1):
    """Write a costs file for ``python -m deepspeed.runtime.pipe.planner``.

    Args:
        layer_types: ``{name: LayerCost}`` for each distinct partitionable layer.
        layer_sequence: the names of the partitionable layers, in pipeline order.
        first_stage_extra / last_stage_extra: costs of the layers that always stay on the first / last
            stage (embedding; final norm, head and loss), never checkpointed.
        micro_batch_size: the micro-batch size the costs were measured with.
    """
    missing = sorted(set(layer_sequence) - set(layer_types))
    if missing:
        raise ValueError(f"layer_sequence names layer types without costs: {missing}")
    spec = {
        "micro_batch_size": micro_batch_size,
        "layer_types": {
            name: cost.to_dict()
            for name, cost in layer_types.items()
        },
        "layer_sequence": list(layer_sequence),
        "first_stage_extra": [c.to_dict() for c in first_stage_extra],
        "last_stage_extra": [c.to_dict() for c in last_stage_extra],
    }
    with open(path, "w") as f:
        json.dump(spec, f, indent=2)


def _parse(section):
    if "layers" in section:
        layers = [LayerCost.from_dict(d) for d in section["layers"]]
    else:
        types = {k: LayerCost.from_dict({**v, "name": k}) for k, v in section["layer_types"].items()}
        layers = [types[t] for t in section["layer_sequence"]]
    first = [LayerCost.from_dict(d) for d in section.get("first_stage_extra", [])]
    last = [LayerCost.from_dict(d) for d in section.get("last_stage_extra", [])]
    return layers, first, last


def load_costs(spec):
    """Read a costs file (path or dict) into ``layer_costs(mbs) -> (layers, first_extra, last_extra)``.

    The file lists the partitionable layers in order, either as ``"layers": [...]`` or as
    ``"layer_types"`` plus ``"layer_sequence"``. Costs measured at other micro-batch sizes can be given
    under ``"per_micro_batch_size": {"2": {...same keys...}}``; any other size is scaled linearly from
    ``"micro_batch_size"``.
    """
    if not isinstance(spec, dict):
        with open(spec) as f:
            spec = json.load(f)
    base_mbs = spec.get("micro_batch_size", 1)
    base = _parse(spec)
    per_mbs = {int(k): _parse({**spec, **v}) for k, v in spec.get("per_micro_batch_size", {}).items()}

    def layer_costs(mbs):
        if mbs in per_mbs:
            return per_mbs[mbs]
        factor = mbs / base_mbs
        return tuple([c.scaled(factor) for c in group] for group in base)

    return layer_costs
