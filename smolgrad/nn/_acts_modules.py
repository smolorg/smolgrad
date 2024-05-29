from typing import *

from ..core import Tensor
from ._module import Module
from ._activations import relu


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
