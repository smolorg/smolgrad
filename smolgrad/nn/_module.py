from typing import *

from ..core import Tensor, _get_d


class Module:
    """
    Base class for the NN (neural network) modules 
    """

    def __init__(self, device: str = "gpu") -> None:
        self.device = device
        self._d = _get_d(device=self.device)
        self.is_training = True

    def _get_tensors(self) -> List[Tensor]:
        """
        Go through the class and get all tensors defined
        """
        tensors: List[Tensor] = []

        for _, value in self.__dict__.items():
            if isinstance(value, Tensor):
                tensors.append(value)
            elif isinstance(value, Module):
                tensors += value._get_tensors()
        
        return tensors
    
    def train(self) -> None:
        """
        Set the module to training
        """
        self.is_training = True
    
    def eval(self) -> None:
        """
        Set module to evaluation
        """
        self.is_training = False
    
    def parameters(self) -> List[Tensor]:
        """
        Returns the parameters which requires gradients to be calculated
        """
        return [t for t in self._get_tensors() if t.requires_grad]
    
    def zero_grad(self):
        """
        Reset gradients of all parameters
        """
        for param in self.parameters():
            param._reset_grad()

    def forward(self, *args: Any, **kwargs: Any):
        """
        To overwritten in the children class
        """
        raise NotImplementedError("Module base class's forward method is not implemented")
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Call the `forward` method with the given arguments and keyword arguments
        """
        return self.forward(*args, **kwargs)
