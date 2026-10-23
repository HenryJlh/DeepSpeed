# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Per-stage memory model used by the pipeline planner."""

from dataclasses import asdict, dataclass


@dataclass
class Calibration:
    """Factors fitted to measured stage peaks; see ``StageMemoryModel`` and ``calibrate.fit_calibration``."""
    state_factor: float = 1.0
    activation_factor: float = 1.0
    overhead_bytes: float = 0.0
    target: str = "reserved"
    fit_mae_bytes: float = 0.0
    loo_mae_bytes: float = float("nan")
    num_points: int = 0

    def to_dict(self):
        return asdict(self)


def inflight_1f1b(stage_id, num_stages, micro_batches):
    """Micro-batches whose activations stage ``stage_id`` holds at once under the 1F1B schedule.

    Stage s runs p - s forwards before its first backward, so it keeps that many micro-batches alive.
    """
    return max(1, min(num_stages - stage_id, micro_batches))


def inflight_gpipe(stage_id, num_stages, micro_batches):
    """All forwards before any backward."""
    return micro_batches


INFLIGHT = {"1f1b": inflight_1f1b, "gpipe": inflight_gpipe}


def layer_transient_bytes(layer, checkpointed):
    """Memory a layer needs only while its backward runs: its backward temporaries, plus the
    activations it rebuilds first when it is checkpointed."""
    return layer.transient_bytes + (layer.activation_bytes if checkpointed else 0.0)


class StageMemoryModel:
    """Peak memory of one pipeline stage.

    peak = state_factor * states + activation_factor * dynamic + overhead_bytes, where

        states  = model states of the stage's layers and extras
        dynamic = in-flight micro-batches * (activations of plain layers + inputs of checkpointed ones
                  + activations of extras) + the largest backward transient of any one layer

    With the factors at 1 and no overhead this is what the profiled costs predict. The factors and
    the overhead absorb what per-layer costs cannot see (optimizer-step temporaries, allocator
    fragmentation, communication buffers); ``calibrate.fit_calibration`` fits them to measured peaks.
    """

    def __init__(self,
                 num_stages,
                 micro_batches,
                 inflight="1f1b",
                 overhead_bytes=0.0,
                 state_factor=1.0,
                 activation_factor=1.0):
        self.num_stages = num_stages
        self.micro_batches = micro_batches
        self.inflight = INFLIGHT[inflight] if isinstance(inflight, str) else inflight
        self.overhead_bytes = overhead_bytes
        self.state_factor = state_factor
        self.activation_factor = activation_factor

    def combine(self, stage_id, states, kept, transient):
        """Peak bytes from a stage's states, per-micro-batch kept activations and largest transient."""
        inflight = self.inflight(stage_id, self.num_stages, self.micro_batches)
        return (self.state_factor * states + self.activation_factor * (inflight * kept + transient) +
                self.overhead_bytes)

    def components(self, stage_id, layers, num_checkpointed, extras=()):
        """``(states, dynamic)`` bytes of a stage before the factors and the overhead are applied.

        ``layers``: the stage's partitionable layers in order; the first ``num_checkpointed`` are
        checkpointed. ``extras``: its non-partitionable layers (embedding, head), never checkpointed.
        """
        kept, states, transient = 0.0, 0.0, 0.0
        for i, layer in enumerate(layers):
            checkpointed = i < num_checkpointed
            kept += layer.kept_bytes(checkpointed)
            states += layer.state_bytes
            transient = max(transient, layer_transient_bytes(layer, checkpointed))
        for layer in extras:
            kept += layer.activation_bytes
            states += layer.state_bytes
            transient = max(transient, layer_transient_bytes(layer, False))
        inflight = self.inflight(stage_id, self.num_stages, self.micro_batches)
        return states, inflight * kept + transient

    def stage_bytes(self, stage_id, layers, num_checkpointed, extras=()):
        states, dynamic = self.components(stage_id, layers, num_checkpointed, extras)
        return self.state_factor * states + self.activation_factor * dynamic + self.overhead_bytes
