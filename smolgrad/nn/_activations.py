from typing import *

from ..core import Tensor


def relu(tn: Tensor) -> Tensor:
    """
    The Rectified Linear Unit (ReLU) activation function is a widely used 
    activation function in neural networks. It is defined as follows:
    
    Formula:
    `ReLU(x) = max(0, x)`

    In other words, the ReLU function returns the input value if it is positive, 
    and 0 if the input value is negative or zero. It introduces non-linearity into the 
    network, allowing it to learn more complex patterns and representations.
    """

    out = Tensor(tn._d.maximum(0, tn.data), _children=(tn, ), _op="relu", use_np=tn.is_np_tensor)

    if tn.requires_grad:
        # gradients will be 0 where tn's data is 0
        # elsewhere they will be just copied backwards
        def _relu_backward():
            tn.grad += (tn.data > 0) * out.grad
        
        out.grad_fn = _relu_backward
        out.set_requires_grad(True)

    return out
