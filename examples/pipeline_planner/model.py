# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""A GPT-like model written as a list of pipeline layers: Embed, N x Block (dense or MoE), Head."""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepspeed.pipe import LayerSpec


def add_model_args(parser: argparse.ArgumentParser):
    g = parser.add_argument_group("model")
    g.add_argument("--layers", type=int, default=32, help="transformer blocks")
    g.add_argument("--hidden", type=int, default=2048)
    g.add_argument("--seq", type=int, default=8192)
    g.add_argument("--vocab", type=int, default=128000)
    g.add_argument("--experts", type=int, default=0, help="experts per block; 0 for a dense MLP")
    g.add_argument("--ffn", type=int, default=4096, help="hidden size of the MLP / of each expert")
    g.add_argument("--topk", type=int, default=2)
    g.add_argument("--ep", type=int, default=1, help="expert-parallel size")
    return parser


class Embed(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.emb = nn.Embedding(args.vocab, args.hidden)

    def forward(self, ids):
        return self.emb(ids)


class MLP(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(args.hidden, args.ffn), nn.Linear(args.ffn, args.hidden)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    """Pre-LN causal attention and MLP. With --experts the MLP is a DeepSpeed MoE layer."""

    def __init__(self, args):
        super().__init__()
        self.hidden, self.heads = args.hidden, args.hidden // 128
        self.ln1, self.ln2 = nn.LayerNorm(args.hidden), nn.LayerNorm(args.hidden)
        self.qkv, self.proj = nn.Linear(args.hidden, 3 * args.hidden), nn.Linear(args.hidden, args.hidden)
        if args.experts:
            from deepspeed.moe.layer import MoE
            self.mlp = MoE(hidden_size=args.hidden,
                           expert=MLP(args),
                           num_experts=args.experts,
                           ep_size=args.ep,
                           k=args.topk)
        else:
            self.mlp = MLP(args)

    def forward(self, x):
        b, s, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, s, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(b, s, -1))
        out = self.mlp(self.ln2(x))
        # an MoE layer returns (output, aux loss, expert counts); pipeline layers pass one tensor on
        return x + (out[0] if isinstance(out, tuple) else out)


class Head(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.ln = nn.LayerNorm(args.hidden)
        self.lm_head = nn.Linear(args.hidden, args.vocab, bias=False)

    def forward(self, x):
        return self.lm_head(self.ln(x))


def lm_loss(logits, labels):
    return F.cross_entropy(logits.float().flatten(0, 1), labels.flatten())


def layer_specs(args):
    """The pipeline layers. Only the Blocks are partitioned; Embed and Head stay on the first / last stage."""
    return [LayerSpec(Embed, args)] + [LayerSpec(Block, args) for _ in range(args.layers)] + [LayerSpec(Head, args)]


def random_batches(args, num, device, mbs=1):
    g = torch.Generator().manual_seed(1)
    batches = []
    for _ in range(num):
        tokens = torch.randint(0, args.vocab, (mbs, args.seq), generator=g)
        labels = torch.randint(0, args.vocab, (mbs, args.seq), generator=g)
        batches.append((tokens.to(device), labels.to(device)))
    return batches
