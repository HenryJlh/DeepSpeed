---
title: "Memory-Aware Pipeline Partitioning"
tags: training large-model
---

Pipeline stages rarely need the same memory. Under the 1F1B schedule, stage `s`
of `P` keeps up to `P - s` micro-batches of activations alive, so the first
stage holds `P` times what the last one holds, while the first and last stages
also carry the embedding and the head. Balancing layer counts or parameters
then leaves the first stages out of memory and the last ones idle, and the
usual fix, recomputing activations on every stage, slows down every stage.

The pipeline planner chooses, for every stage, how many layers it gets and
how many of them it recomputes, so that the slowest stage is as fast as
possible while every stage fits in memory. It implements ELM-PP from
[X-MoE-4D (SC'26)](#relation-to-x-moe-4d) on top of `PipelineModule`.

This tutorial assumes familiarity with the [pipeline parallelism
tutorial](/tutorials/pipeline/), in particular the `custom:` partition method
and `checkpoint_layers_per_stage`, which execute the plans.

## Usage

Planning takes three steps: measure what each kind of layer costs, plan
offline, and train with the plan.

### 1. Measure the layers

`profile_layer` runs one layer forward and backward on the current device for
one micro-batch and returns a `LayerCost`: forward and backward time, the
activations its forward keeps for the backward, its input (what it keeps when
recomputed), the memory its backward needs only while it runs, and its model
states. Measure each distinct layer once, with the input it receives inside the
pipeline, and write the costs with `save_costs`:

```python
from deepspeed.runtime.pipe.planner import profile_layer, save_costs

hidden = torch.randn(mbs, seq, d_model, device=device, dtype=torch.bfloat16)
tokens = torch.randint(0, vocab, (mbs, seq), device=device)
costs = {
    "Block": profile_layer(TransformerBlock(cfg).to(device), (hidden, ), dp_size=dp, ep_size=ep),
}
embed = profile_layer(Embedding(cfg).to(device), (tokens, ), dp_size=dp)
head = profile_layer(HeadAndLoss(cfg).to(device), (hidden, ), dp_size=dp)
save_costs("costs.json", costs, ["Block"] * num_blocks,
           first_stage_extra=[embed], last_stage_extra=[head], micro_batch_size=mbs)
```

`layer_sequence` (here `["Block"] * num_blocks`) lists the partitionable layers
in pipeline order; models that mix layer types, such as dense and MoE blocks,
list each by its type name. The embedding and the head always stay on the
first and last stage.

Collectives a layer issues are timed with it, so profile MoE layers on a rank
of their real expert-parallel group. Model states assume bf16 mixed precision
with ZeRO stage 1 and Adam (2 bytes of parameters and gradients, plus 12 bytes
of fp32 master weights and moments sharded over the `dp_size` data-parallel
ranks, or `dp_size // ep_size` for expert parameters). For other settings pass
`state_bytes` to `profile_layer`.

### 2. Plan

```bash
python -m deepspeed.runtime.pipe.planner --costs costs.json --stages 4 --memory-gb 64 \
    --global-batch-size 256 --dp 8 --micro-batch-sizes 1,2 --partition-layer-type Block -o plan.json
```

The planner prints the plan with its predicted time and memory per stage next
to uniform partitions without and with full recomputation, and writes
`plan.json`. `--memory-gb` is the device memory. A share of it,
`--headroom-gb` (10 % by default), is left out of the plan for
what per-layer costs cannot see: allocator fragmentation and communication
buffers. If a plan runs out of memory, raise `--headroom-gb`.

Fragmentation matters most when a stage allocates single tensors of several
GiB, such as the fp32 logits of a large vocabulary or, with ZeRO stage 1 at a
small data-parallel size, the fp32 gradient partition of the optimizer step:
the allocator may hold enough free memory in total but no block large enough.
Keeping it from splitting large cached blocks helps; on ROCm, where
`expandable_segments` is not supported, set
`PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512` (`PYTORCH_ALLOC_CONF` on
recent PyTorch).

### 3. Train with the plan

```python
net = PipelineModule(layers=specs, num_stages=4, loss_fn=loss_fn,
                     partition_method="profile:plan.json")
```

The plan sets the partition (as `custom:`), `partition_layer_type` and
`checkpoint_layers_per_stage`; arguments given explicitly take precedence. Use
the micro-batch size the plan was made for (`micro_batch_size` and
`micro_batches` in `plan.json`); the engine warns otherwise. Every rank reads
the plan itself, so all ranks must see the same file: ranks partitioned
differently cannot exchange activations and hang.

## How the planner works

**Stage time** is the sum of the forward and backward times of the stage's
layers, plus one more forward for every recomputed layer, plus the embedding or
head on the first or last stage.

**Stage memory** is the larger of two phases of a training step. During the
forward and backward passes (1F1B):

```
model states + min(P - s, micro-batches) x kept activations + largest backward transient
```

where kept activations are, per micro-batch, the activations of the plain
layers and the inputs of the recomputed ones, and the backward transient is
the memory one layer's backward needs while it runs; a recomputed layer
rebuilds its activations first, so its transient includes them. Only one layer
runs its backward at a time, so the largest transient counts, not their sum.
During the optimizer step, when the activations are freed:

```
model states + step memory of the stage's largest parameter group
```

With bf16 and ZeRO stage 1 the step needs 6 bytes per parameter of a group
(sharded over the ranks that share its optimizer partition), one group at a
time; a stage's dense parameters form one group and its expert parameters
another. At a small data-parallel size the step can be the larger phase, and
recomputation cannot shrink it, so such a stage must hold fewer parameters.

**Search.** Recomputing more layers of a stage only adds time, so a stage
recomputes the fewest layers that fit. A minimax dynamic program over the cut
points then minimizes the slowest stage, which sets the 1F1B step time
`sum(stage times) + (micro-batches - 1) x slowest stage time`. An outer loop
repeats this for every candidate micro-batch size that divides the global
batch and keeps the shortest predicted step.

## Relation to X-MoE-4D

The planner is ELM-PP from X-MoE-4D (SC'26). Its search is unchanged; its cost
model is adapted to generic `PipelineModule` models and measured on the device
instead of derived from Megatron-specific formulas.

### Search: unchanged

The objective (minimize the slowest stage under a per-stage memory limit), the
choice within a stage (fewest recomputed layers that fit), the dynamic program,
the micro-batch loop and its step-time formula, and the linear scaling of costs
to unmeasured micro-batch sizes are those of X-MoE-4D. Three details differ:

| X-MoE-4D | DeepSpeed | Kind | Reason |
|---|---|---|---|
| One cost for all layers | A cost per layer, from layer types | Necessary | A `PipelineModule` can mix layer types (dense and MoE blocks); with one layer type the search is the same |
| A stage may be empty | Every stage gets at least one layer | Necessary | `custom:` partitions require it |
| Embedding and head count toward memory only | They count toward time too | Improvement | A large-vocabulary head can take as long as two blocks or more (2.1-2.4 blocks for a 128k vocabulary in our measurements); ignoring it overloads the last stage |

X-MoE-4D also adds 55 ms per micro-batch and 10 % of the forward/backward time
for the optimizer to the step time. Neither changes the partition: a constant
added to every stage leaves the slowest stage where it is, and the 10 % scales
all candidates alike. The 55 ms only shifts the choice between micro-batch
sizes and was measured for X-MoE-4D's Megatron setup, so it is left out; the
predicted step time therefore excludes stage-to-stage transfers and the
optimizer step, which favors smaller micro-batch sizes when several are given.

### Cost model: adapted

| Term | X-MoE-4D | DeepSpeed | Kind | Reason |
|---|---|---|---|---|
| Layer time | Forward GEMM time `g` and all-to-all time `c` parsed from the log of an instrumented Megatron run; `3g + 2c` per layer, `4g + 3c` when recomputed | Forward and backward time measured per layer type by `profile_layer`; a recomputed layer adds one forward | Necessary | A generic model has no instrumented GEMM and communication timers. Measuring the backward also avoids assuming it costs `2g + c` |
| Model states | `2 + 2 + 12 / d` bytes per parameter, `d = dp` (dense) or `dp / ep` (experts) | Same, from the layer's parameters (experts recognized as DeepSpeed MoE marks them) | Same | Holds for bf16 + ZeRO-1 + Adam only; other settings pass `state_bytes` |
| Activations | Closed-form formula for X-MoE-4D's attention and MoE layers | Tensors autograd saves for the backward, measured per layer type | Necessary | The formula describes X-MoE-4D's own kernels; for DeepSpeed MoE blocks it was 17-23 % below the measured activations |
| In-flight micro-batches | `P - s` | `min(P - s, micro-batches)` | Improvement | With fewer micro-batches than stages, a stage cannot hold more than there are |
| Embedding and head | Formula terms for the vocabulary (two fp32 logits-sized buffers on the last stage) | Measured like any layer, on the first and last stage | Necessary | Same reason as activations |
| Backward working set | Twice one layer's activations (+30 %) once a stage recomputes anything; otherwise inside the fixed overhead | Largest measured backward transient of the stage; a recomputed layer adds its rebuilt activations | Improvement | The same quantity, measured per layer instead of a fixed multiple |
| Optimizer step | Not modelled | Model states + 6 / d bytes per parameter of the stage's largest parameter group; the stage's peak is the larger of this and the 1F1B phase | Improvement | With ZeRO-1 at a small data-parallel size the step is the peak: on 4x MI210 (PP4, DP1) a stage of 2.5B dense parameters needs a 9.5 GiB fp32 gradient partition, and plans without this term ran out of memory in the step |
| Overhead | 7 GiB | `--headroom-gb`, 10 % of the device memory by default | Necessary | Fragmentation and communication buffers depend on the device and the framework |

The paper's memory model (Eq. 12) lists model states, in-flight activations
and the overhead; the backward working set appears in X-MoE-4D's code, the
optimizer step in neither.

### Not ported

* X-MoE-4D's Megatron fork replaces the model-parallel group of the ZeRO
  gradient-norm reduction with a group inside one stage. With an uneven
  partition this avoids a hang, but each stage then clips its gradients by its
  own norm. In DeepSpeed, MoE models trained with ZeRO stage 1 and an uneven
  partition can hang in the gradient-norm reduction when stages hold different
  numbers of expert parameter groups. Until the reduction is fixed, build the
  parameter groups yourself with one group per expert group and pass them to
  `deepspeed.initialize`, which then leaves them as they are:
  ```python
  from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
  groups = split_params_into_different_moe_groups_for_optimizer(
      {"params": list(net.parameters()), "name": "params"}, max_group_size=None)
  engine, _, _, _ = deepspeed.initialize(model=net, model_parameters=groups, config=config)
  ```
* The memory-balanced and ablation modes of X-MoE-4D's planner script.

## Limitations

* The model-state and optimizer-step formulas hold for bf16 mixed precision
  with ZeRO stage 1 and Adam; pass `state_bytes` for anything else (the step
  is then not modelled). With default MoE parameter groups, which split the
  expert parameters into chunks, the step needs less than modelled.
* Plans assume the 1F1B schedule of `PipelineModule`. DualPipeV runs
  `checkpoint_layers_per_stage` but is not planned for.
* The memory of stage-to-stage transfers and allocator fragmentation is not
  modelled and is covered by `--headroom-gb`. The time of transfers and of the
  optimizer step is not modelled either.
* Costs measured at one micro-batch size are scaled linearly to the others.
  To compare micro-batch sizes, measure each one and add it to the costs file
  under `"per_micro_batch_size"`.

Contact: [@HenryJlh](https://github.com/HenryJlh)
