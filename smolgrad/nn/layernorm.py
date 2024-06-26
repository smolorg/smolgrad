from typing import *
from ..core import Tensor
from ._module import Module

class LayerNorm(Module):
    def __init__(self, normalized_shape: Union[int, List[int]], eps: float = 1e-5, device: str = "gpu"):
        super().__init__(device)
        self.normalized_shape = normalized_shape if isinstance(normalized_shape, (list, tuple)) else [normalized_shape]
        self.eps = eps
        self.use_np = False if device == "gpu" else True

        # Initialize gamma and beta parameters
        self.weight = Tensor(
            self._d.ones(self.normalized_shape),
            dtype=self._d.float32, requires_grad=True, use_np=self.use_np
        )
        self.bias = Tensor(
            self._d.zeros(self.normalized_shape),
            dtype=self._d.float32, requires_grad=True, use_np=self.use_np
        )

    def forward(self, x: Tensor) -> Tensor:
        # Calculate the dimensions to normalize over
        norm_dims = tuple(range(x.ndim - len(self.normalized_shape), x.ndim))
        
        # Calculate mean and variance along the normalized dimensions
        mean = x.mean(axis=norm_dims, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=norm_dims, keepdims=True)

        # Normalize the input
        x_norm = (x - mean) / ((var + self.eps) ** 0.5)

        # Scale and shift
        return self.weight * x_norm + self.bias
