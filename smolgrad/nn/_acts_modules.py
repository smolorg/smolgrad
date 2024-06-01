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
