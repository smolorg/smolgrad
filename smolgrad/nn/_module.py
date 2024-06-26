from typing import *

from ..core import Tensor, _get_d


class Module:
    """
    Base class for the NN (neural network) modules 
    """
    def __init__(self, device: str = "gpu") -> None:
        self.device = device
        self._d = _get_d(device=self.device)
        self.use_np = False if device=="gpu" else True
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
    
    def state_dict(self, prefix: str = '') -> Dict[str, Any]:
        """
        Returns a dictionary containing a whole state of the module.
        Both parameters and persistent buffers (e.g. running averages) are included.
        """
        state_dict = {}
        for name, value in self.__dict__.items():
            pref = f"{prefix}.{name}" if prefix else name
            if isinstance(value, Tensor):
                state_dict[pref] = value
            elif isinstance(value, (Module, ModuleList, ModuleDict)):
                state_dict = state_dict | value.state_dict(prefix=pref)
    
        return state_dict

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Call the `forward` method with the given arguments and keyword arguments
        """
        return self.forward(*args, **kwargs)


class ModuleList(Module):
    def __init__(self, modules: List[Module] = None, device: str = "gpu"):
        super().__init__(device=device)
        self._modules = modules or []

    def __getitem__(self, idx: int) -> Module:
        return self._modules[idx]

    def __setitem__(self, idx: int, module: Module):
        self._modules[idx] = module

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules)

    def append(self, module: Module):
        self._modules.append(module)

    def extend(self, modules: List[Module]):
        self._modules.extend(modules)

    def insert(self, index: int, module: Module):
        self._modules.insert(index, module)

    def parameters(self) -> List[Tensor]:
        params = []
        for module in self._modules:
            params.extend(module.parameters())
        return params

    def state_dict(self, prefix: str = '') -> Dict[str, Any]:
        sd = {}
        for index, mod in enumerate(self._modules):
            pref = f"{prefix}.{index}" if prefix else index
            sd |= mod.state_dict(prefix=pref)
        return sd
        
    def forward(self, x: Any) -> Any:
        for module in self._modules:
            x = module(x)
        return x

class ModuleDict(Module):
    def __init__(
            self, modules: Dict[str, Union[Module, ModuleList, 'ModuleDict']] = None,
            device: str = "gpu"
        ):
        super().__init__(device=device)
        self._modules = modules or {}

    def __getitem__(self, key: str) -> Union[Module, ModuleList, 'ModuleDict']:
        return self._modules[key]

    def __setitem__(self, key: str, module: Union[Module, ModuleList, 'ModuleDict']):
        self._modules[key] = module

    def __delitem__(self, key: str):
        del self._modules[key]

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[str]:
        return iter(self._modules)

    def __contains__(self, key: str) -> bool:
        return key in self._modules

    def clear(self):
        self._modules.clear()

    def pop(self, key: str) -> Union[Module, ModuleList, 'ModuleDict']:
        return self._modules.pop(key)

    def keys(self) -> List[str]:
        return list(self._modules.keys())

    def items(self) -> List[tuple]:
        return list(self._modules.items())

    def values(self) -> List[Union[Module, ModuleList, 'ModuleDict']]:
        return list(self._modules.values())

    def update(self, modules: Dict[str, Union[Module, ModuleList, 'ModuleDict']]):
        self._modules.update(modules)

    def parameters(self) -> List[Tensor]:
        params = []
        for module in self._modules.values():
            params.extend(module.parameters())
        return params
    
    def state_dict(self, prefix: str = '') -> Dict[str, Any]:
        sd = {}
        for k, mod in self._modules.items():
            pref = f"{prefix}.{k}" if prefix else k
            sd |= mod.state_dict(prefix=pref)
        return sd
    
    def forward(self, x: Any) -> Any:
        for module in self._modules.values():
            x = module(x)
        return x