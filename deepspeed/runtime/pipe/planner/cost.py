# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Per-layer cost model used by the pipeline planner."""

from dataclasses import dataclass, asdict, replace


@dataclass(frozen=True)
class LayerCost:
    """What one layer costs one pipeline rank for one micro-batch.

    Args:
        fwd_ms: forward time.
        bwd_ms: backward time.
        activation_bytes: memory the forward keeps alive for the backward when the layer is not checkpointed.
        input_bytes: memory kept when the layer is checkpointed, i.e. its input.
        state_bytes: resident model states (parameters, gradients and optimizer states) for this layer.
        name: free-form label, e.g. the layer type.
    """
    fwd_ms: float
    bwd_ms: float
    activation_bytes: float = 0.0
    input_bytes: float = 0.0
    state_bytes: float = 0.0
    name: str = ""

    def time_ms(self, checkpointed=False):
        """A checkpointed layer runs its forward a second time before the backward."""
        return self.fwd_ms + self.bwd_ms + (self.fwd_ms if checkpointed else 0.0)

    def kept_bytes(self, checkpointed=False):
        return self.input_bytes if checkpointed else self.activation_bytes

    @classmethod
    def from_gemm_comm(cls, gemm_ms, comm_ms, **kwargs):
        """Cost of an expert-parallel MoE layer from its forward GEMM and all-to-all time (X-MoE-4D, SC'26).

        The backward runs the GEMMs twice and the all-to-all once, which gives the paper's
        3g + 2c per layer, and 4g + 3c when it is checkpointed.
        """
        return cls(fwd_ms=gemm_ms + comm_ms, bwd_ms=2 * gemm_ms + comm_ms, **kwargs)

    def scaled(self, factor):
        """Time and activations grow linearly with the micro-batch size; model states do not."""
        return replace(self,
                       fwd_ms=self.fwd_ms * factor,
                       bwd_ms=self.bwd_ms * factor,
                       activation_bytes=self.activation_bytes * factor,
                       input_bytes=self.input_bytes * factor)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


ZERO_LAYER = LayerCost(0.0, 0.0)


def model_state_bytes(num_params, zero_stage=0, dp_size=1, param_bytes=2, grad_bytes=2, optimizer_bytes=12):
    """Resident model-state bytes for ``num_params`` parameters on one rank.

    The defaults are bf16/fp16 mixed precision with Adam: 2-byte parameters and gradients and
    12 bytes of fp32 master weights and moments. Pipeline parallelism supports ZeRO-0 and ZeRO-1;
    ZeRO-1 shards the optimizer states over the ``dp_size`` ranks that hold the same parameters.
    """
    if zero_stage not in (0, 1):
        raise ValueError(f"pipeline parallelism supports ZeRO stage 0 and 1, got {zero_stage}")
    optimizer = optimizer_bytes / dp_size if zero_stage == 1 else optimizer_bytes
    return num_params * (param_bytes + grad_bytes + optimizer)
