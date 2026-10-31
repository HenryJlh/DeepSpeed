# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Memory-aware pipeline partitioning.

Measure each distinct layer with :func:`profile_layer` and write the costs with :func:`save_costs`;
:class:`StageMemoryModel` predicts each stage's peak memory from them.
"""

from .cost import LayerCost, model_state_bytes, optimizer_step_bytes, save_costs, load_costs
from .memory import StageMemoryModel, inflight_1f1b
from .profile import profile_layer, saved_activation_bytes, backward_transient_bytes
