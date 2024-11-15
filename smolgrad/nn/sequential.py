from typing import List, Union, Any

from ._module import Module, ModuleList
from ..core import Tensor


class Sequential(ModuleList):
    """
    A sequential container.

    Modules will be added to it in the order they are passed in the constructor. 
    Alternatively, an OrderedDict of modules can be passed in. 
    The forward() method of Sequential accepts any input and forwards 
    it to the first module it contains. It then “chains” outputs to inputs sequentially 
    for each subsequent module, finally returning the output of the last module.
    """
    def __init__(self, *modules: Module, device: str = "gpu") -> None:
        super().__init__(modules=modules, device=device)
