from typing import *

from ._optimizer import Optimizer
from ..core import Tensor


class RMSProp(Optimizer):
    """
    RMSProp (Root Mean Square Propagation) optimization
    """
    def __init__(
            self, parameters: List[Tensor], 
            lr: float, decay: float = 0.95, eps: float = 1e-8, 
            device: str = "gpu"
        ) -> None:
        super().__init__(parameters, lr, device)

        self.decay = decay
        self.eps = eps
        self.vs = [self._d.zeros_like(p.data) for p in self.parameters]

    def step(self):
        """
        Use adaptive learning rate for parameters. Infrequent params get
        more boost while frequent params get less boost.
        """
        for i in range(len(self.parameters)):
            p, v = self.parameters[i], self.vs[i]

            v[:] = (self.decay * v) + ((1 - self.decay) * p.grad ** 2)
            p.data -= (self.lr / (self._d.sqrt(v) + self.eps)) * p.grad
