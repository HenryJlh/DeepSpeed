#!/bin/bash
# Memory-aware pipeline partitioning on 4 GPUs: a 32-block MoE model (8 experts per block), PP=4.
set -e
MODEL="--layers 32 --hidden 2048 --seq 8192 --vocab 128000 --experts 8 --ffn 4096"

# 1. Measure the Block, the embedding and the head (one GPU here: expert parallelism is 1).
deepspeed --num_gpus 1 profile_layers.py $MODEL -o costs.json

# 2. Plan for 4 stages of 64 GiB GPUs and 16 micro-batches of size 1 per step.
python -m deepspeed.runtime.pipe.planner --costs costs.json --stages 4 --memory-gb 64 \
    --global-batch-size 16 --partition-layer-type Block -o plan.json

# 3. Train with the plan, then with DeepSpeed's default partition and full recomputation for comparison.
deepspeed --num_gpus 4 train.py $MODEL --partition profile:plan.json --deepspeed_config ds_config.json
deepspeed --num_gpus 4 train.py $MODEL --partition parameters --recompute-all --deepspeed_config ds_config.json
