from typing import *

from ..core import Tensor
from ._optimizer import Optimizer


class SGD(Optimizer):
    """
    Stochastic gradient descent with momentum
    """
    def __init__(
            self, parameters: List[Tensor], lr: float,
            momentum: float = 0.9, device: str = "gpu"
        ) -> None:
        super().__init__(parameters, lr, device)

        self.momentum = momentum
        self.velocities = []
        for p in self.parameters:
            self.velocities.append(self._d.zeros_like(p.data))

    def step(self):
        """
        Update parameters using momentum:

        ```
        v = momentum * v + (1 - momentum) * p.grad
        p.data -= lr * v
        ```
        """
        for i in range(len(self.parameters)):
            p, v = self.parameters[i], self.velocities[i]

            v[:] = (self.momentum * v) + ((1 - self.momentum) * p.grad)
            p.data -= (self.lr * v)
