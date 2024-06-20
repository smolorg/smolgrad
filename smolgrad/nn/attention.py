import math
from typing import *

from smolgrad import Tensor
from ._module import Module
from .linear import Linear
from .dropout import Dropout
from ._activations import softmax


class CausalSelfAttention(Module):
    def __init__(
            self, context_size: int, d_embed: int, 
            n_heads: int, attn_pdrop: float = 0.7,
            resd_pdrop: float = 0.6, device: str = "gpu"
        ) -> None:
        # TODO: dropout and attention mask
        super().__init__(device)
        self.use_np = False if device=="gpu" else True
        self.context_size = context_size

        self.w_proj = Linear(d_embed, 3 * d_embed, device=self.device)
        self.o_proj = Linear(d_embed, d_embed, device=self.device)
        self.attn_drop = Dropout(p=attn_pdrop, device=self.device)
        self.resd_drop = Dropout(p=resd_pdrop, device=self.device)

        self.mask = self._d.tril(
            self._d.ones((context_size, context_size))
        )
        self.mask = self._d.reshape(self.mask, (1, 1, context_size, context_size))
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
        _mask = (self.mask == 0).astype(self._d.int8)
        attn = attn.masked_fill(_mask, float("-inf"))
        attn = softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        y = attn @ v
        y = y.T(axes=[1, 2]).reshape(bs, sl, ed)
        y = self.o_proj(y)
        y = self.resd_drop(y)
        return y


