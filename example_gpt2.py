import os
import math
import tiktoken

from typing import *
from dataclasses import dataclass

import smolgrad.nn as nn
from smolgrad import Tensor
from smolgrad.core import no_grad, _get_d


@dataclass
class GPT2Config:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


# --------------------- MODEL --------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPT2Config, device="gpu"):
        super().__init__(device=device)
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, device=device)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, device=device)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # mask
        _b = self._d.ones((config.block_size, config.block_size))
        _b = self._d.tril(_b)
        self.bias = self._d.reshape(_b, (1, 1, config.block_size, config.block_size))

    def forward(self, x: Tensor):
        B, T, C = x.size()
        qkv: Tensor = self.c_attn(x)
        q, k, v = qkv.split(sections=3, dim=-1)
        q = q.reshape((B, T, self.n_head, C // self.n_head)).T(axes=[1, 2])
        k = k.reshape((B, T, self.n_head, C // self.n_head)).T(axes=[1, 2])
        v = v.reshape((B, T, self.n_head, C // self.n_head)).T(axes=[1, 2])

        attn: Tensor = (q @ k.T([-2, -1])) * (1.0 / math.sqrt(k.size(-1)))
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        attn = nn.softmax(attn, dim=-1)
        y: Tensor = attn @ v
        y = y.T([1, 2]).reshape((B, T, C))
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: GPT2Config, device="gpu"):
        super().__init__(device=device)

        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x: Tensor):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPT2Config, device="gpu"):
        super().__init__(device=device)

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPT2Config, device="gpu"):
        super().__init__(device=device)

        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd, device=device),
            wpe = nn.Embedding(config.block_size, config.n_embd, device=device),
            h = nn.ModuleList([Block(config, device=device) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, device=device)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, use_bias=False, device=device)

    def forward(self, idx: Tensor):
        assert idx._d == self._d, "need all tensors of same type either mlx or np."

        # idx shape is (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, max sequence length is {self.config.block_size}"

        pos = self._d.arange(0, T, dtype=self._d.int32)
        pos = Tensor(pos, dtype=pos.dtype, use_np=self.use_np)
        pos_emb = self.transformer.wpe(pos)     # (T, n_embd)
        tok_emb = self.transformer.wte(idx)     # (B, T, n_embd)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)    # (B, T, vocab_size)

        return logits   # softmax over logits to get probabilities

    @classmethod
    def from_pretrained(cls, model_type, device="gpu"):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        _d = _get_d(device)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPT2Config(**config_args)
        model = GPT(config)
        sd = model.state_dict()

        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                weights_data = sd_hf[k].t().tolist()
                print(">> copying transposed weights...")
                sd[k].data[:] = _d.array(weights_data)
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                weights_data = sd_hf[k].tolist()
                print(">> copying other weights...")
                sd[k].data[:] = _d.array(weights_data)

        return model


model = GPT.from_pretrained("gpt2")
for k, v in model.state_dict().items():
    print(k, v.shape)
