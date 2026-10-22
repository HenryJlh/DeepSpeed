# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""AutoEP inside a PipelineModule: pipeline stages across ranks, expert parallelism inside a stage."""

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import deepspeed
from deepspeed.pipe import PipelineModule

from unit.common import DistributedTest

transformers = pytest.importorskip("transformers")

STEPS, MICRO_BATCHES, SEQ, LR, CLIP = 2, 4, 16, 1e-3, 1.0


def tiny_mixtral():
    cfg = transformers.MixtralConfig(num_hidden_layers=4,
                                     hidden_size=64,
                                     intermediate_size=128,
                                     num_attention_heads=4,
                                     num_key_value_heads=2,
                                     vocab_size=256,
                                     num_local_experts=4,
                                     num_experts_per_tok=2,
                                     tie_word_embeddings=False,
                                     attn_implementation="sdpa")
    torch.manual_seed(0)
    return transformers.MixtralForCausalLM(cfg).float()


def lm_loss(logits, labels):
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(), labels[:, 1:].reshape(-1))


# Pipeline wrappers that keep each HF module under its HF path, e.g. "<idx>.model.layers.2.mlp",
# so the stock AutoEP preset finds the MoE blocks.
class Embed(nn.Module):

    def __init__(self, hf):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = hf.model.embed_tokens

    def forward(self, ids):
        return self.model.embed_tokens(ids)


class Decoder(nn.Module):

    def __init__(self, hf, i):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleDict({str(i): hf.model.layers[i]})
        self.rotary = hf.model.rotary_emb
        self.index = str(i)

    def forward(self, h):
        pos = torch.arange(h.size(1), device=h.device).unsqueeze(0)
        out = self.model.layers[self.index](h, position_ids=pos, position_embeddings=self.rotary(h, pos))
        return out[0] if isinstance(out, tuple) else out


class Head(nn.Module):

    def __init__(self, hf):
        super().__init__()
        self.model = nn.Module()
        self.model.norm = hf.model.norm
        self.lm_head = hf.lm_head

    def forward(self, h):
        return self.lm_head(self.model.norm(h))


def reference_run(hf, data):
    model = copy.deepcopy(hf)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    out = []
    for step in range(STEPS):
        opt.zero_grad()
        loss = sum(lm_loss(model(input_ids=x).logits, x) for x in data[step]) / MICRO_BATCHES
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        out.append((loss.item(), norm.item()))
    return out


class TestAutoEPInPipeline(DistributedTest):
    world_size = 4

    @pytest.mark.parametrize("zero_stage", [0, 1])
    def test_matches_hf_reference(self, zero_stage):
        hf = tiny_mixtral()
        g = torch.Generator().manual_seed(1)
        data = [[torch.randint(0, 256, (1, SEQ), generator=g) for _ in range(MICRO_BATCHES)] for _ in range(STEPS)]
        reference = reference_run(hf, data)

        layers = [Embed(hf)] + [Decoder(hf, i) for i in range(4)] + [Head(hf)]
        model = PipelineModule(layers=layers, num_stages=2, loss_fn=lm_loss, partition_method="uniform")
        config = {
            # 2 stages x 2 data-parallel replicas: every replica sees the same micro-batches
            "train_batch_size": 2 * MICRO_BATCHES,
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": MICRO_BATCHES,
            "gradient_clipping": CLIP,
            "steps_per_print": 1000,
            "zero_optimization": {
                "stage": zero_stage
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": LR,
                    "weight_decay": 0.0,
                    "torch_adam": True
                }
            },
            "expert_parallel": {
                "enabled": True,
                "autoep_size": 2,
                "preset_model": "mixtral",
                "use_grouped_mm": False
            },
        }
        engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=list(model.parameters()), config=config)
        replaced = [n for n, m in engine.module.named_modules() if type(m).__name__ == "AutoEPMoELayer"]
        assert len(replaced) == 2, replaced  # two Mixtral blocks per stage
        for step in range(STEPS):
            loss = engine.train_batch(iter([(x, x) for x in data[step]]))
            if zero_stage == 1:  # the basic fp32 optimizer path does not record the norm
                assert engine.get_global_grad_norm() == pytest.approx(reference[step][1], rel=1e-4)
            assert loss.item() == pytest.approx(reference[step][0], rel=1e-5)
