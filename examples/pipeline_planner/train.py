# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Step 3: train with a plan, or with a baseline partition, and report step time and peak memory per stage.

    deepspeed --num_gpus 4 train.py --experts 8 --partition profile:plan.json
    deepspeed --num_gpus 4 train.py --experts 8 --partition parameters --recompute-all
"""

import argparse
import os
import time

import torch

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.pipe import PipelineModule

from model import add_model_args, layer_specs, lm_loss, random_batches

p = add_model_args(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
p.add_argument("--stages", type=int, default=4)
p.add_argument("--partition", default="profile:plan.json", help="partition_method of PipelineModule")
p.add_argument("--recompute-all", action="store_true", help="baseline: recompute every Block")
p.add_argument("--steps", type=int, default=5, help="timed steps, after 2 warm-up steps")
p.add_argument("--local_rank", type=int, default=-1)
p = deepspeed.add_config_arguments(p)
args = p.parse_args()

deepspeed.init_distributed()
acc = get_accelerator()
acc.set_device(int(os.environ.get("LOCAL_RANK", 0)))

# "profile:plan.json" sets the Blocks per stage and how many of them each stage recomputes
torch.manual_seed(1234)  # the same weights in every run of the same partition
kwargs = dict(partition_method=args.partition)
if args.recompute_all:
    kwargs.update(activation_checkpoint_interval=1, checkpointable_layers=["Block"])
net = PipelineModule(layers=layer_specs(args), num_stages=args.stages, loss_fn=lm_loss, **kwargs)
params = [p for p in net.parameters()]
if args.experts:
    # With ZeRO-1, stages that hold different numbers of expert parameter groups hang in the
    # gradient-norm reduction, which uneven partitions cause; one group per expert group avoids it.
    from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
    group = {"params": params, "name": "params"}
    params = split_params_into_different_moe_groups_for_optimizer(group, max_group_size=None)
engine, _, _, _ = deepspeed.initialize(args=args, model=net, model_parameters=params)

data = random_batches(args, engine.gradient_accumulation_steps(), acc.current_device_name(),
                      engine.train_micro_batch_size_per_gpu())
times = []
try:
    for step in range(2 + args.steps):
        acc.synchronize()
        start = time.perf_counter()
        loss = engine.train_batch(iter(data))
        acc.synchronize()
        if step >= 2:
            times.append(time.perf_counter() - start)
except torch.OutOfMemoryError:
    # the other stages then wait for this one until the launcher stops them
    print(f"stage {engine.stage_id}: out of memory", flush=True)
    raise

if engine.grid.get_data_parallel_id() == 0:  # one line per stage
    print(
        f"stage {engine.stage_id}: {sum(times) / len(times) * 1000:.0f} ms/step, "
        f"peak {acc.max_memory_allocated() / 2**30:.2f} GiB allocated / {acc.max_memory_reserved() / 2**30:.2f} GiB "
        f"reserved, loss {float(loss):.4f}",
        flush=True)
