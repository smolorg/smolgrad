from typing import *

from ._optimizer import Optimizer
from ..core import Tensor


class Adam(Optimizer):
    """
    Update using Adam optimization
    """
    def __init__(
            self, parameters: List[Tensor], 
            lr: float, beta1: float = 0.9, beta2: float = 0.96,
            eps: float = 1e-6, device: str = "gpu"
        ) -> None:
        super().__init__(parameters, lr, device)

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.mo1 = [self._d.zeros_like(p.data) for p in self.parameters]
        self.mo2 = [self._d.zeros_like(p.data) for p in self.parameters]
        
        # time step
        self.timestep = 0

    def step(self):
        self.timestep += 1
        for i in range(len(self.parameters)):
            p, m, v = self.parameters[i], self.mo1[i], self.mo2[i]

            m[:] = ((1 - self.beta1) * p.grad) + (m * self.beta1)
            v[:] = ((1 - self.beta2) * p.grad**2) + (v * self.beta2)

            m_hat = m / (1 - self.beta1 ** self.timestep)
            v_hat = v / (1 - self.beta2 ** self.timestep)

            # update weights
            p.data -= (self.lr * m_hat / self._d.sqrt(v_hat) + self.eps)
