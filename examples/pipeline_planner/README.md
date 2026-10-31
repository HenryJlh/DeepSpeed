# Memory-aware pipeline partitioning (X-MoE-4D ELM-PP)

This example shows how to let DeepSpeed choose, for every pipeline stage, how many layers it gets and how many of them
it recomputes, so that the slowest stage is as fast as possible while every stage fits in GPU memory. It uses the
pipeline planner of DeepSpeed (`deepspeed.runtime.pipe.planner`), which implements ELM-PP from X-MoE-4D (SC'26). See
the [tutorial](https://www.deepspeed.ai/tutorials/pipeline-planner/) for how the planner works.

The model (`model.py`) is a GPT-like `PipelineModule`: an embedding, 32 transformer blocks (dense MLP, or a DeepSpeed
MoE layer with `--experts`) and a 128k-vocabulary head.

## Run

`run.sh` runs the three steps on 4 GPUs with a pipeline of 4 stages:

```bash
# 1. Measure what each kind of layer costs (time, activations, backward transient, model states)
deepspeed --num_gpus 1 profile_layers.py --experts 8 -o costs.json

# 2. Plan for 4 stages of 64 GiB GPUs and a global batch of 16 sequences
python -m deepspeed.runtime.pipe.planner --costs costs.json --stages 4 --memory-gb 64 \
    --global-batch-size 16 --partition-layer-type Block -o plan.json

# 3. Train with the plan
deepspeed --num_gpus 4 train.py --experts 8 --partition profile:plan.json --deepspeed_config ds_config.json
```

`train.py --partition parameters --recompute-all` runs DeepSpeed's default partition with every block recomputed, for
comparison. Each stage prints its step time and peak memory.

Notes:

* Profile MoE layers on as many GPUs as the expert-parallel size (`deepspeed --num_gpus <ep> profile_layers.py --ep
  <ep> --dp <dp>`), so that the all-to-all is timed with the layer.
* The model states assume bf16 with ZeRO stage 1 and Adam, as in `ds_config.json`.
* If a plan runs out of memory although every stage's predicted peak is below the device memory, the allocator could
  not find a block large enough for one of the large single tensors (the fp32 logits, or the optimizer step's fp32
  gradient partition at a small data-parallel size). On ROCm set `PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512`, then
  raise `--headroom-gb`.
* `train.py` builds the MoE optimizer groups with `max_group_size=None`: with ZeRO stage 1, pipeline stages that hold
  different numbers of expert parameter groups hang in the gradient-norm reduction, and uneven partitions cause that.

## Results

4x MI210 (64 GiB), PP 4 x DP 1, bf16 with ZeRO stage 1, 16 micro-batches of 1 sequence, 128k vocabulary, plans made
with 6 GiB of headroom (the default is 10 % of the device memory, 6.4 GiB). Step time in seconds; DeepSpeed's default
partition is `parameters`.

| Model | Plan (layers / recomputed per stage) | Plan | `parameters`, no recomputation | `parameters`, all recomputed |
|---|---|---|---|---|
| Dense, 32 blocks of 201M, seq 4096 | 8,9,9,6 / 0,0,0,0 | 9.96 | 10.18 | 12.72 |
| MoE, 32 blocks with 8 experts, seq 8192 | 8,9,9,6 / 0,0,0,0 | 9.83 | 10.40 | 13.08 |
| Dense, 40 blocks of 201M, seq 8192 (*) | 10,10,11,9 / 6,2,0,0 | 29.36 | not run | 34.80 |

(*) With `PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512`; without it every partition tried, including DeepSpeed's
default with all blocks recomputed, runs out of memory through fragmentation.

The planner's predicted peak memory was below the measured peak by 0.3-2.2 GiB per stage, and its step time below the
measured one by 1-8 %.
