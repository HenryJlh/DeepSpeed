# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Memory-aware pipeline partitioning.

Measure each distinct layer with :func:`profile_layer`, write the costs with :func:`save_costs`,
and plan with :func:`search_micro_batch_size`. The plan sets both the layers per stage and how
many of them each stage checkpoints.
"""

from .cost import LayerCost, model_state_bytes, optimizer_step_bytes, save_costs, load_costs
from .memory import StageMemoryModel, inflight_1f1b
from .solver import (PipelinePlan, StagePlan, solve_stages, search_micro_batch_size, evaluate_partition,
                     pipeline_step_ms)
from .profile import profile_layer, saved_activation_bytes, backward_transient_bytes
