from typing import *

from ..core import Tensor
from ._module import Module
from ._activations import *


class ReLU(Module):
    """
    ReLU module returns the input value if it is positive, 
    and 0 if the input value is negative or zero
    """
    def __init__(self) -> None:
        pass

    def forward(self, x: Tensor) -> Tensor:
        out = relu(x)
        return out


class Sigmoid(Module):
    """
    Sigmoid module returns calculated sigmoid of input tensor x
    """
    def __init__(self) -> None:
        pass

    def forward(self, x: Tensor) -> Tensor:
        out = sigmoid(x)
        return out


class Softmax(Module):
    """
    Calculates the softmax function for an input tensor given some axis
    """
    def __init__(self, dim: int = None) -> None:
        self.dim = dim or -1

    def forward(self, x: Tensor) -> Tensor:
        out = softmax(x, axis = self.dim)
        return out


class GELU(Module):
    def __init__(
            self, 
            approximate: Literal['none', 'tanh'] = 'none', 
            device: str = "gpu"
        ):
        super().__init__(device)
        self.approximate = approximate

    def forward(self, x: Tensor) -> Tensor:
        if self.approximate == 'tanh':
            return self._tanh_approximation(x)
        else:
            return self._exact_gelu(x)

    def _exact_gelu(self, x: Tensor) -> Tensor:
        y: Tensor = (x * 0.7978845608 * (1 + 0.044715 * x * x))
        return 0.5 * x * (1 + tanh(y))

    def _tanh_approximation(self, x: Tensor) -> Tensor:
        y: Tensor = (0.7978845608 * (x + 0.044715 * x * x * x))
        return 0.5 * x * (1 + tanh(y))
