from typing import *
from mpi4py import MPI

from ..core import Tensor
from ..nn import Module, ModuleList, ModuleDict


def gradient_allreduce(comm: MPI.Intracomm, param: Tensor):
    """
    This hook (function) starts a non-blocking allreduce (average)
    communication for the parameters as soon as their final gradients are calculated.
    Non-blocking call helps us start calculating the gradients for the next layer which
    basically interleaves communication (this layer) with computation (next layer).

    TODO: optimization - Starting a communication for each parameter is not optimal. Pytorch' DDP handles this
    by putting the parameters in buckets.
    """
    if param.requires_grad and param.grad is not None:
        # inplace all-reduce (average will be calculated later)
        param._request = comm.Iallreduce(
            MPI.IN_PLACE, param.grad, op=MPI.SUM
        )

def gradients_wait_for_all(params: List[Tensor], world_size: int):
    """
    After the full backward pass, we will wait for all the reduction
    communication to finish, and only then we can ensure that the gradients
    on all the processes are the same.
    """
    requests = [
        param._request for param in params
        if param.requires_grad and param.grad is not None and 
        param._request is not None
    ]
    MPI.Request.Waitall(requests)

    # average the gradients on all processes now
    # gradients are all summed, we just need to 
    # divide by the world_size
    for param in params:
        if param.requires_grad and param.grad is not None:
            param.grad[:] = param.grad / world_size


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
        self.register_hooks()
    
    def eval(self) -> None:
        self.model.eval()

    def zero_grad(self):
        """
        Reset gradients of all parameters
        """
        self.model.zero_grad()

    def parameters(self) -> List[Tensor]:
        return self.model.parameters()
    
    def synchronize(self) -> None:
        """
        Across all the processes, let the gradients be reduced
        """
        if self.model.is_training:
            gradients_wait_for_all(self.model.parameters(), self.world_size)
            for param in self.model.parameters():
                param.reset_grad_hooks()

            # hooks for next iteration
            self.register_hooks()
    
    def register_hooks(self) -> None:
        if self.model.is_training:
            for param in self.model.parameters():
                param.register_grad_hook(
                    lambda p: gradient_allreduce(self.comm, p)
                )

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

    def __call__(self, *args, **kwargs) -> Any:
        return self.model(*args, **kwargs)