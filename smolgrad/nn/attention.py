import math
from typing import *

from smolgrad import Tensor
from ._module import Module
from .linear import Linear
from ._activations import softmax


class SelfAttention(Module):
    def __init__(
            self, context_size: int, d_embed: int, 
            n_heads: int, device: str = "gpu"
        ) -> None:
        # TODO: dropout and attention mask
        super().__init__(device)
        self.use_np = False if device=="gpu" else True

        self.w_proj = Linear(d_embed, 3 * d_embed)
        self.o_proj = Linear(d_embed, d_embed)
        self.n_heads = n_heads
        self.d_embed = d_embed

    def forward(self, x: Tensor) -> Tensor:
        bs, sl, ed = x.shape
        q, k, v = self.w_proj(x).split(sections=3, dim=-1)
        q = q.reshape((bs, sl, self.n_heads, ed // self.n_heads)).T(axes=[1, 2])
        k = k.reshape((bs, sl, self.n_heads, ed // self.n_heads)).T(axes=[1, 2])
        v = v.reshape((bs, sl, self.n_heads, ed // self.n_heads)).T(axes=[1, 2])

        kT: Tensor = k.T(axes=[-1, -2])
        attn: Tensor = (q @ kT) * (1 / math.sqrt(k.shape[-1]))
        attn = softmax(attn, axis=-1)

        # TODO: add dropout and attention mask
        y = attn @ v
        y = y.T(axes=[1, 2]).reshape(bs, sl, ed)
        y = self.o_proj(y)

        return y


