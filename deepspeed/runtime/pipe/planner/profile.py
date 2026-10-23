# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Measure the per-layer costs the planner needs on the current device."""

import time

import torch

from deepspeed.accelerator import get_accelerator
from .cost import LayerCost, model_state_bytes


def _tensor_bytes(t):
    return t.numel() * t.element_size()


def saved_activation_bytes(module, inputs):
    """Bytes autograd keeps alive between ``module(*inputs)`` and its backward.

    Counts each saved storage once, skips the module's own parameters, and is exact on any device.
    """
    params = {p.untyped_storage().data_ptr() for p in module.parameters()}
    seen = set()
    total = [0]

    def pack(t):
        ptr = t.untyped_storage().data_ptr()
        if ptr not in params and ptr not in seen:
            seen.add(ptr)
            total[0] += t.untyped_storage().nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = module(*inputs)
    return total[0], out


def backward_transient_bytes(module, inputs):
    """Peak memory the backward needs beyond what the forward left allocated, gradients excluded.

    Gradients of parameters are model states; the rest (the gradient flowing in, gradients of the
    inputs, kernel workspaces, an fp32 logits gradient) only lives while the backward runs.
    Measured with the accelerator's peak-memory counter, so it is only as exact as that counter.
    """
    acc = get_accelerator()
    module.zero_grad(set_to_none=True)
    for x in inputs:
        if torch.is_tensor(x):
            x.grad = None  # the gradient sent to the previous stage is new memory in the pipeline
    acc.synchronize()
    out = _first_tensor(module(*inputs))
    acc.synchronize()
    after_forward = acc.memory_allocated()
    acc.reset_peak_memory_stats()
    out.backward(torch.ones_like(out))
    acc.synchronize()
    peak = acc.max_memory_allocated()
    grads = sum(_tensor_bytes(p.grad) for p in module.parameters() if p.grad is not None)
    module.zero_grad(set_to_none=True)
    return max(0.0, float(peak - after_forward - grads))


def _first_tensor(out):
    if torch.is_tensor(out):
        return out
    for o in out:
        if torch.is_tensor(o) and o.requires_grad:
            return o
    raise ValueError("layer output has no tensor that requires grad")


def _timed(fn, iters):
    acc = get_accelerator()
    acc.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    acc.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def profile_layer(module, inputs, warmup=3, iters=10, name="", zero_stage=0, dp_size=1, state_bytes=None):
    """Forward/backward time, saved activations and model states of one layer for one micro-batch.

    ``inputs`` is the tuple the layer receives in the pipeline. Floating inputs are made to require
    grad so the backward runs as it would mid-pipeline. Time is wall clock between device
    synchronizations, averaged over ``iters`` runs after ``warmup`` runs. Collectives the layer issues
    (e.g. the expert all-to-all) are timed with it, so profile MoE layers under their real EP group.
    """
    inputs = tuple(x.detach().requires_grad_(x.is_floating_point()) if torch.is_tensor(x) else x for x in inputs)

    def fwd():
        return module(*inputs)

    def fwd_bwd():
        out = _first_tensor(fwd())
        out.backward(torch.ones_like(out))

    for _ in range(warmup):
        fwd_bwd()
    fwd_ms = _timed(fwd, iters)
    total_ms = _timed(fwd_bwd, iters)
    module.zero_grad(set_to_none=True)

    activation, out = saved_activation_bytes(module, inputs)
    del out
    transient = backward_transient_bytes(module, inputs)
    input_bytes = sum(_tensor_bytes(x) for x in inputs if torch.is_tensor(x) and x.is_floating_point())
    if state_bytes is None:
        num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        state_bytes = model_state_bytes(num_params, zero_stage=zero_stage, dp_size=dp_size)
    return LayerCost(fwd_ms=fwd_ms,
                     bwd_ms=max(total_ms - fwd_ms, 0.0),
                     activation_bytes=float(activation),
                     input_bytes=float(input_bytes),
                     state_bytes=float(state_bytes),
                     name=name or type(module).__name__,
                     transient_bytes=float(transient))
