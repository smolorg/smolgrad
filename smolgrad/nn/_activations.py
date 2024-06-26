# This file is for various functional activation functions. 
# They introduce non-linearity into the network, allowing it to 
# learn more complex patterns and representations.

from typing import *

from ..core import Tensor


def relu(tn: Tensor) -> Tensor:
    """
    The Rectified Linear Unit (ReLU) activation function is a widely used 
    activation function in neural networks. It is defined as follows:
    
    Formula:
    `ReLU(x) = max(0, x)`

    In other words, the ReLU function returns the input value if it is positive, 
    and 0 if the input value is negative or zero.
    """

    out = Tensor(
        tn._d.maximum(0, tn.data), dtype=tn.dtype, _children=(tn, ), 
        _op="relu", use_np=tn.is_np_tensor
    )

    if tn.requires_grad and Tensor.grad_is_enabled:
        # gradients will be 0 where tn's data is 0
        # elsewhere they will be just copied backwards
        def _relu_backward():
            tn.grad += (tn.data > 0) * out.grad
        
        out.grad_fn = _relu_backward
        out.set_requires_grad(True)

    return out


def tanh(tn: Tensor) -> Tensor:
    """
    Apply the hyperbolic tangent (tanh) activation function element-wise.

    Args:
        val (Tensor): The input tensor.

    Returns:
        Tensor: The tensor with tanh applied element-wise.
    """

    out = Tensor(
        tn._d.tanh(0, tn.data), dtype=tn.dtype, _children=(tn, ), 
        _op="tanh", use_np=tn.is_np_tensor
    )

    if tn.requires_grad and Tensor.grad_is_enabled:
        def _tanh_backward():
            tn.grad += (1 - out.data**2) * out.grad

        out._backward = _tanh_backward
        out.requires_grad = True

    return out


def sigmoid(tn: Tensor) -> Tensor:
    """
    Computes the expit (also known as the logistic sigmoid function) of the elements of input.

    Formula:

    `sigmoid(x) = 1 / (1 + exp(-x))`
    """

    e_1x = tn._d.exp(-tn.data)
    out = Tensor(
        1 / (1 + e_1x), _children=(tn, ), dtype=tn.dtype,
        _op="sigmoid", use_np=tn.is_np_tensor
    )

    # since d/dx (1 / (1 + e^-x)) = e^-x / (1 + e^-x) ^ 2
    if tn.requires_grad and Tensor.grad_is_enabled:
        def _sigmoid_backward():
            tn.grad += (e_1x / (1 + e_1x) ** 2) * out.grad

        out.grad_fn = _sigmoid_backward
        out.set_requires_grad(True)

    return out


def softmax(tn: Tensor, axis: int = -1) -> Tensor:
    """
    Applies the Softmax function to an n-dimensional input Tensor.

    Rescales them so that the elements of the n-dimensional output Tensor 
    lie in the range [0,1] and sum to 1.
    """

    # note: for numerical stability and not letting the softmax output
    # underflow or overflow, we shift the original tensor t by max(t)
    # also, no need to write the backward gradient flow function for this
    # since it uses atomic tensor operations.
    max_tn = Tensor(
        tn._d.max(tn.data, axis=axis, keepdims=True), 
        dtype=tn.dtype, use_np=tn.is_np_tensor
    )
    shifted_exp = (tn - max_tn).exp()

    shifted_exp_sum = shifted_exp.sum(axis=axis, keepdims=True)
    out = shifted_exp / shifted_exp_sum

    return out
