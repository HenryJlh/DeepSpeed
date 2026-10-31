Pipeline Parallelism
====================

Model Specification
--------------------
.. autoclass:: deepspeed.pipe.PipelineModule
    :members:

.. autoclass:: deepspeed.pipe.DualPipeVModule
    :members:

.. autoclass:: deepspeed.pipe.LayerSpec
    :members:

.. autoclass:: deepspeed.pipe.TiedLayerSpec
    :members:

.. autoclass:: deepspeed.runtime.pipe.ProcessTopology
    :members:

Training
--------
.. automodule:: deepspeed.runtime.pipe.engine
    :members:

.. autoclass:: deepspeed.runtime.pipe.dualpipev.DualPipeVEngine
    :members:

Planning
--------
See the `memory-aware pipeline partitioning tutorial <https://www.deepspeed.ai/tutorials/pipeline-planner/>`_.

.. autofunction:: deepspeed.runtime.pipe.planner.profile_layer

.. autofunction:: deepspeed.runtime.pipe.planner.save_costs

.. autofunction:: deepspeed.runtime.pipe.planner.load_costs

.. autoclass:: deepspeed.runtime.pipe.planner.LayerCost
    :members:

.. autofunction:: deepspeed.runtime.pipe.planner.model_state_bytes

.. autofunction:: deepspeed.runtime.pipe.planner.optimizer_step_bytes

.. autoclass:: deepspeed.runtime.pipe.planner.StageMemoryModel
    :members:

.. autofunction:: deepspeed.runtime.pipe.planner.search_micro_batch_size

.. autofunction:: deepspeed.runtime.pipe.planner.solve_stages

.. autoclass:: deepspeed.runtime.pipe.planner.PipelinePlan
    :members:

Extending Pipeline Parallelism
------------------------------
.. automodule:: deepspeed.runtime.pipe.schedule
    :members:
