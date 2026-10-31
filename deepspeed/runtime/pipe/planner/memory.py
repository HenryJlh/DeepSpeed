# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Per-stage memory model used by the pipeline planner."""


def inflight_1f1b(stage_id, num_stages, micro_batches):
    """Micro-batches whose activations stage ``stage_id`` holds at once under the 1F1B schedule.

    Stage s runs p - s forwards before its first backward, so it keeps that many micro-batches alive,
    or all of them if there are fewer.
    """
    return max(1, min(num_stages - stage_id, micro_batches))


def layer_transient_bytes(layer, checkpointed):
    """Memory a layer needs only while its backward runs: its backward temporaries, plus the
    activations it rebuilds first when it is checkpointed."""
    return layer.transient_bytes + (layer.activation_bytes if checkpointed else 0.0)


def step_peak_bytes(states, layers):
    """Peak of the optimizer step: model states plus the largest parameter group's step memory.

    A stage's dense parameters form one optimizer group and its expert parameters another, and the
    step processes one group at a time. Activations are freed by then.
    """
    dense = sum(l.step_bytes for l in layers)
    expert = sum(l.expert_step_bytes for l in layers)
    return states + max(dense, expert)


class StageMemoryModel:
    """Peak memory of one pipeline stage: the larger of its 1F1B phase and its optimizer step.

    1F1B phase = states + in-flight micro-batches * kept + transient, where

        states    = model states of the stage's layers, including the embedding / head it holds
        kept      = per micro-batch, activations of plain layers + inputs of checkpointed ones
                    + activations of the embedding / head
        in-flight = min(p - s, micro_batches), see :func:`inflight_1f1b`
        transient = the largest backward transient of any one layer of the stage, see
                    :func:`layer_transient_bytes`; only one layer runs its backward at a time

    optimizer step = states + the largest parameter group's step memory, see :func:`step_peak_bytes`

    The 1F1B phase is Eq. 12 of X-MoE-4D (SC'26) with its terms measured instead of derived: the
    paper's model states and in-flight activations, plus the backward working set that X-MoE-4D's
    code adds once a stage recomputes. The optimizer step is not in X-MoE-4D's model; with ZeRO-1 at
    a small data-parallel size it is the larger of the two. The paper's fixed overhead, for what
    per-layer costs cannot see (allocator fragmentation, communication buffers), is not part of the
    model: plan against the device memory minus a headroom instead.
    """

    def __init__(self, num_stages, micro_batches):
        self.num_stages = num_stages
        self.micro_batches = micro_batches

    def inflight(self, stage_id):
        return inflight_1f1b(stage_id, self.num_stages, self.micro_batches)

    def stage_bytes(self, stage_id, layers, num_checkpointed, extras=()):
        """Peak bytes of stage ``stage_id``.

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
        one_f_one_b = states + self.inflight(stage_id) * kept + transient
        return max(one_f_one_b, step_peak_bytes(states, list(layers) + list(extras)))
