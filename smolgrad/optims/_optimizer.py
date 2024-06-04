from typing import *
from ..core import Tensor, _get_d


class Optimizer:
    """
    Base class for all optimizers
    """
    def __init__(self, parameters: List[Tensor], lr: float, device: str = "gpu") -> None:
        self.device = device
        self._d = _get_d(self.device)
        self.parameters = parameters
        self.lr = lr
        
        assert self._d == self.parameters[0]._d, f"Mismatched devices in Optimizer."

    def step(self) -> None:
        """
        Update the parameters.
        """
        raise NotImplementedError("The 'step' method must be implemented in child class.")
    
    def zero_grad(self) -> None:
        """
        Reset gradients to 0 values
        """
        for param in self.parameters:
            param._reset_grad()
