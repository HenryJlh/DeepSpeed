# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Per-stage memory model used by the pipeline planner."""


def inflight_1f1b(stage_id, num_stages, micro_batches):
    """Micro-batches whose activations stage ``stage_id`` holds at once under the 1F1B schedule.

    Stage s runs p - s forwards before its first backward, so it keeps that many micro-batches alive.
    """
    return max(1, min(num_stages - stage_id, micro_batches))


def inflight_gpipe(stage_id, num_stages, micro_batches):
    """All forwards before any backward."""
    return micro_batches


INFLIGHT = {"1f1b": inflight_1f1b, "gpipe": inflight_gpipe}


class StageMemoryModel:
    """Peak memory of one pipeline stage.

    peak = model states of its layers and extras
         + in-flight micro-batches * (activations of plain layers + inputs of checkpointed ones + extras)
         + one recomputed layer's activations, if anything is checkpointed
         + overhead_bytes (allocator slack, communication buffers, ...)
    """

    def __init__(self, num_stages, micro_batches, inflight="1f1b", overhead_bytes=0.0):
        self.num_stages = num_stages
        self.micro_batches = micro_batches
        self.inflight = INFLIGHT[inflight] if isinstance(inflight, str) else inflight
        self.overhead_bytes = overhead_bytes

    def stage_bytes(self, stage_id, layers, num_checkpointed, extras=()):
        """``layers``: the stage's partitionable layers in order; the first ``num_checkpointed`` are checkpointed.
        ``extras``: its non-partitionable layers (embedding, head), never checkpointed."""
        kept, states, transient = 0.0, 0.0, 0.0
        for i, layer in enumerate(layers):
            checkpointed = i < num_checkpointed
            kept += layer.kept_bytes(checkpointed)
            states += layer.state_bytes
            if checkpointed:
                transient = max(transient, layer.activation_bytes)
        for layer in extras:
            kept += layer.activation_bytes
            states += layer.state_bytes
        inflight = self.inflight(stage_id, self.num_stages, self.micro_batches)
        return states + inflight * kept + transient + self.overhead_bytes
