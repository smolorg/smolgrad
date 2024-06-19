from typing import *

from ._module import Module
from ..core import Tensor


class Dropout(Module):
    """
    Randomly zero out elements during training.
    """
    def __init__(self, p: float = 0.5, device: str = "gpu") -> None:
        super().__init__(device)
        self.p = p
        self.use_np = False if device=="gpu" else True

    def forward(self, x: Tensor) -> Tensor:
        if not self.is_training:
            return x
        
        mask = None
        if self.use_np:
            mask = self._d.reshape(self._d.random.binomial(size=x.shape), x.shape)
        else:
            mask = self._d.random.bernoulli(shape=x.shape)
        mask = mask > self.p    # randomly 0 out 
        mask = Tensor(
            mask.astype(self._d.int8), dtype=self._d.int8,
            requires_grad=False, use_np=self.use_np
        )

        # compensate for the expected value change during training
        out = x * mask / (1 - self.p)
        return out




        