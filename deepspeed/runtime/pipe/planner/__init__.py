# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Profile-guided pipeline partitioning (``PipelineModule(partition_method="profile:<plan.json>")``).

Build a plan from per-layer costs with :func:`search_micro_batch_size` (or the command line,
``python -m deepspeed.runtime.pipe.planner``), save it with :meth:`PipelinePlan.save`, and pass
its path to ``PipelineModule``. The plan sets both the layers per stage and how many of them each
stage checkpoints.
"""

from .cost import LayerCost, model_state_bytes
from .memory import StageMemoryModel, inflight_1f1b, inflight_gpipe
from .solver import (PipelinePlan, StagePlan, solve_stages, search_micro_batch_size, evaluate_partition,
                     pipeline_step_ms)
from .profile import profile_layer, saved_activation_bytes
