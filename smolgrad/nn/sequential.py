from typing import List, Union, Any

from ._module import Module
from ..core import Tensor


class Sequential(Module):
    """
    A sequential container.

    Modules will be added to it in the order they are passed in the constructor. 
    Alternatively, an OrderedDict of modules can be passed in. 
    The forward() method of Sequential accepts any input and forwards 
    it to the first module it contains. It then “chains” outputs to inputs sequentially 
    for each subsequent module, finally returning the output of the last module.
    """
    def __init__(self, *modules: Module, device: str = "gpu") -> None:
        super().__init__(device=device)

        self.modules = list(modules)

    def append(self, module: Module):
        """
        add the given module at the end of the sequential list
        """
        self.modules.append(module)

    def parameters(self) -> List[Tensor]:
        t = []
        for mod in self.modules:
            t += mod.parameters()
        
        return t

    def forward(self, x: Union[Tensor, Any]) -> Union[Tensor, Any]:
        if len(self.modules) == 0:
            return x
        
        for i in range(len(self.modules)):
            x = self.modules[i](x)

        return x
