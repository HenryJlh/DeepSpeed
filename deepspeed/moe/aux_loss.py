# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Route an MoE auxiliary loss through the backward pass of the activations it is attached to.

A pipeline stage only passes activations to the next one, and only the last stage computes the
loss, so a load-balancing loss returned by an MoE layer on an earlier stage never reaches it.
:func:`attach_aux_loss` instead ties the auxiliary loss to the layer's output: the forward returns
the output unchanged, and its backward feeds ``d(total loss)/d(aux) = coef`` into the auxiliary
loss, scaled like the main loss (1 / gradient accumulation steps, times the fp16 loss scale).
This works wherever the layer runs, with or without pipeline parallelism.

The engine sets that scale before each backward pass. Values seen by the backward are recorded,
once per micro-batch even under activation checkpointing, for logging.
"""

import torch


class _State:
    # set once any auxiliary loss is attached; until then the engine does no aux-loss work at all
    active = False
    grad_scale = 1.0
    total = None
    count = 0


def aux_loss_attached():
    """Whether this process has attached an auxiliary loss with :func:`attach_aux_loss`."""
    return _State.active


def set_aux_loss_grad_scale(scale):
    """The factor the engine applies to the main loss before its backward pass.

    A float, or a 0-dim tensor (e.g. a GradScaler's device-side scale) to avoid a host sync.
    """
    _State.grad_scale = scale


def pop_recorded_aux_loss():
    """Sum and count of the scaled auxiliary losses whose backward ran since the last call."""
    total, count = _State.total, _State.count
    _State.total, _State.count = None, 0
    return total, count


class _AttachAuxLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, output, aux_loss):
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        aux_loss, = ctx.saved_tensors
        value = aux_loss.detach().float()
        _State.total = value if _State.total is None else _State.total + value
        _State.count += 1
        scale = _State.grad_scale
        if torch.is_tensor(scale):
            return grad_output, scale.to(aux_loss.device, aux_loss.dtype).expand_as(aux_loss).clone()
        return grad_output, torch.full_like(aux_loss, scale)


def attach_aux_loss(output, aux_loss, coef=1.0):
    """Return ``output`` with ``coef * aux_loss`` added to the training objective through its backward.

    Do not also add the auxiliary loss to the model's loss, or it is counted twice. Outside of
    autograd (evaluation, the no-grad pass of reentrant checkpointing) ``output`` is returned as is.
    """
    if not torch.is_grad_enabled() or not torch.is_tensor(aux_loss) or not aux_loss.requires_grad:
        return output
    _State.active = True
    return _AttachAuxLoss.apply(output, aux_loss * coef)
