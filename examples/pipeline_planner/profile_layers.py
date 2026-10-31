# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Step 1: measure what each kind of layer costs and write costs.json for the planner.

    deepspeed --num_gpus <ep> profile_layers.py --experts 8 --ep <ep> --dp <dp> -o costs.json

Run one process per expert-parallel rank so that the MoE all-to-all is timed with the layer.
"""

import argparse
import os

import torch
import torch.nn as nn

import deepspeed
import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.pipe.planner import profile_layer, save_costs

from model import Block, Embed, Head, add_model_args, lm_loss

p = add_model_args(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
p.add_argument("--dp", type=int, default=1, help="data-parallel replicas the plan is for (ZeRO-1 shards over them)")
p.add_argument("--micro-batch-size", type=int, default=1)
p.add_argument("-o", "--output", default="costs.json")
p.add_argument("--local_rank", type=int, default=-1)
args = p.parse_args()

deepspeed.init_distributed()
acc = get_accelerator()
acc.set_device(int(os.environ.get("LOCAL_RANK", 0)))
device, dtype = acc.current_device_name(), torch.bfloat16
mbs = args.micro_batch_size
tokens = torch.randint(0, args.vocab, (mbs, args.seq), device=device)
hidden = torch.randn(mbs, args.seq, args.hidden, device=device, dtype=dtype)


class HeadAndLoss(nn.Module):
    """The last stage runs the head and the loss; the fp32 logits gradient belongs to its cost."""

    def __init__(self):
        super().__init__()
        self.head = Head(args)

    def forward(self, x):
        return lm_loss(self.head(x), tokens)


block = Block(args).to(device, dtype)
if args.experts:
    block.mlp.set_deepspeed_parallelism()  # creates the expert-parallel group; the engine does this in training
costs = {"Block": profile_layer(block, (hidden, ), dp_size=args.dp, ep_size=args.ep)}
del block
embed = profile_layer(Embed(args).to(device, dtype), (tokens, ), dp_size=args.dp)
head = profile_layer(HeadAndLoss().to(device, dtype), (hidden, ), dp_size=args.dp)

if dist.get_rank() == 0:
    for name, c in [("Block", costs["Block"]), ("Embed", embed), ("Head", head)]:
        print(
            f"{name:6s} fwd {c.fwd_ms:7.1f} ms  bwd {c.bwd_ms:7.1f} ms  activations {c.activation_bytes / 2**30:6.2f} GiB  "
            f"backward transient {c.transient_bytes / 2**30:6.2f} GiB  states {c.state_bytes / 2**30:6.2f} GiB")
    save_costs(args.output, costs, ["Block"] * args.layers, [embed], [head], micro_batch_size=mbs)
    print(f"wrote {args.output}")
