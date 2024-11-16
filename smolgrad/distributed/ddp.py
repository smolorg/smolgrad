from typing import *
from mpi4py import MPI

from ..core import Tensor
from ..nn import Module, ModuleList, ModuleDict


class DistributedDataParallel:
    def __init__(
            self, model: Union[Module, ModuleList, ModuleDict], 
            comm: MPI.Intracomm, root: int = 0
        ):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.world_size = comm.Get_size()
        self.root = root
        self.model = model

        self._broadcast_model()

    def train(self) -> None:
        self.model.train()
    
    def eval(self) -> None:
        self.model.eval()

    def zero_grad(self):
        """
        Reset gradients of all parameters
        """
        self.model.zero_grad()

    def parameters(self) -> List[Tensor]:
        return self.model.parameters()
    
    def __call__(self, *args, **kwargs) -> Any:
        return self.model(*args, **kwargs)

    def _broadcast_model(self) -> None:
        """
        Method to broadcast the whole model to all the processes. Each
        process should then contain the same parameters of the model.
        """
        if self.rank == self.root:
            state_dict = self.model.state_dict()
        else:
            state_dict = None
        state_dict = self.comm.bcast(state_dict, root=self.root)
        if self.rank != self.root:
            self.model.load_state_dict(state_dict)