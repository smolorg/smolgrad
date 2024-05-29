# this file uses numpy / mlx in order to perform tensor operations

import numpy as np
import mlx.core as mx

from typing import *

from ..utils import broadcast_axis


# constants
DEFAULT_MIN = 1e-7
DEFAULT_MAX = 1 - 1e-7

# base type
Array = Union[np.ndarray, mx.array]


class Tensor:
    """
    holds elements having the same dtype
    """
    def __init__(
            self,
            data: Union[Array, Any],
            dtype = None,
            _children: tuple = (),
            _op = None,
            requires_grad: bool = False,
            use_np: bool = False
        ) -> None:
        
        self.is_np_tensor = use_np
        self._d = self._get_d(device="gpu" if not use_np else "cpu")
        self.dtype = dtype or self._d.float32
        
        # actual data
        self.data = (
            self._d.array(data, self.dtype) if not isinstance(data, Array) else data.astype(dtype=self.dtype)
        )

        # operation this tensor originates from along with children
        self._prev = set([c for c in _children if c.requires_grad])
        self._op = _op

        # gradient
        self.requires_grad = requires_grad
        self.grad = self._d.zeros_like(self.data) if self.requires_grad else None
        self.grad_fn = None

        self.shape = self.data.shape
        self.ndim = len(self.shape)

    def _get_d(self, device: str = "gpu"):
        # this is a bit misleading because mlx has unified cpu/gpu ram
        # will be fixing this soon
        if device == "cpu":    return np
        if device == "gpu":    return mx

    def set_requires_grad(self, val: bool):
        if not isinstance(val, bool):
            raise ValueError("Value should be boolean")
        
        if self.grad is None and val == True:
            self.grad = mx.zeros_like(self.data)

        self.requires_grad = val

    def backward(self):
        """
        sort the graph topologically.
        run the grad function from the last node to first
        """
        ordering = []
        
        visited = set()
        recursion_stack = set()
        
        def _tsort(curr: "Tensor"):
            if curr in recursion_stack:
                raise ValueError("Graph contains a cycle")
            if curr not in visited:
                visited.add(curr)
                recursion_stack.add(curr)

                for child in curr._prev:
                    _tsort(child)
                
                recursion_stack.remove(curr)
                ordering.append(curr)

        _tsort(self)

        # gradient wrt to self is always 1
        self.grad = self._d.ones_like(self.data)

        # gradient on each previous node
        for node in reversed(ordering):
            if node.grad_fn is not None:
                node.grad_fn()
    
    def clip(
            self, min: float = DEFAULT_MIN, 
            max: float = DEFAULT_MAX, clip_grad: bool = False, 
            grad_min: float = DEFAULT_MIN, grad_max: float = DEFAULT_MAX
        ):
        """
        constrain the Tensor/gradient values between given min max range
        """

        self.data = self._d.clip(self.data, min, max)

        if clip_grad and self.grad is not None:
            self.grad = self._d.clip(self.grad, grad_min, grad_max)

        return self

    # ----------------------- UNARY OPS --------------------------------

    def sum(self, axis: Tuple[int] = None, keepdims: bool = False):
        """
        sum values of tensor along given axes
        """
        out = Tensor(
            self._d.sum(self.data, axis=axis, keepdims=keepdims),
            _children=(self, ), _op="sum", use_np=self.is_np_tensor
        )

        if self.requires_grad:
            ex_axis = axis if axis and not keepdims else None

            def _sum_backward():
                if ex_axis:
                    self.grad += self._d.ones_like(self.grad) * self._d.expand_dims(
                        out.grad, axis=ex_axis
                    )
                else:
                    self.grad += self._d.ones_like(self.grad) * out.grad

            out.grad_fn = _sum_backward
            out.set_requires_grad(True)

        return out
    
    def half(self):
        """
        convert the data and gradients to half precision i.e. float32 -> float16
        """
        if self.dtype == self._d.float32:
            out = Tensor(
                self.data, dtype=self._d.float16, 
                _children = (self, ), _op = "half",
                use_np=self.is_np_tensor
            )

            if self.requires_grad:
                # just copy the gradients backward
                def _half_backward():
                    self.grad += out.grad
                
                out.grad_fn = _half_backward
                out.set_requires_grad(True)

            return out
        
        else:
            raise ValueError(f"Cannot convert Tensor with dtype {self.dtype} to half precision.")

    def T(self, axes: Iterable = None):
        """
        transposes a given tensor along the given axes
        """

        out = Tensor(
            self._d.transpose(self.data, axes=axes),
            _children=(self, ), _op='T', use_np=self.is_np_tensor
        )

        if self.requires_grad:
            def _transpose_backward():
                self.grad += self._d.transpose(out.grad, axes=axes)
            
            out.grad_fn = _transpose_backward
            out.set_requires_grad(True)
        
        return out
    
    def exp(self):
        """
        elementwise e to the power data
        """

        out = Tensor(self._d.exp(self.data), _children=(self, ), _op='exp', use_np=self.is_np_tensor)
        
        if self.requires_grad:
            def _exp_backward():
                self.grad += self.data * out.grad

            out.grad_fn = _exp_backward
            out.set_requires_grad(True)
        
        return out

    # ------------------------ BINARY OPS -------------------------

    def __matmul__(self, other):
        """
        matrix multiplication with tensors
        """

        assert self._d == other._d, f"Tensors must be of the same type i.e. numpy or mlx"

        other = other if isinstance(other, Tensor) else Tensor(other, use_np=self.is_np_tensor)
        
        out = Tensor(self.data @ other.data, _children=(self, other), _op="@", use_np=self.is_np_tensor)
        if not self.requires_grad and not other.requires_grad:
            return out

        # for 1D tensors, expand first dimension for first tensor
        # and expand last dimension for second tensor
        # example: (3,) @ (3,) becomes (1, 3) and (3, 1)
        # which is compatible for matrix multiplication
        le_axis = (0, ) if self.data.ndim == 1 else ()
        re_axis = (-1, ) if other.data.ndim == 1 else ()

        # resultant tensor's grad should be expanded by both le_axis and re_axis
        rese_axis = le_axis + re_axis

        # we need to take broadcasting into account
        # except last two dimensions of shape (since they will be used for matmul)
        # gradients will be summed along the broadcasted axes for both tensors
        l, r = broadcast_axis(self.data.shape[:-2], other.data.shape[:-2])

        # for 2D (can be generalized for more dimensions too):
        #
        # self.grad = out.grad @ other.data.T
        # other.grad = self.data.T @ out.grad

        def _matmul_backward():
            if self.requires_grad:
                self.grad = self._d.reshape(
                    self._d.sum(
                        self._d.expand_dims(out.grad, axis=rese_axis) @
                        self._d.expand_dims(other.data, axis=re_axis).swapaxes(-1, -2),
                        axis = l
                    ),
                    self.data.shape
                )
            if other.requires_grad:
                other.grad = self._d.reshape(
                    self._d.sum(
                        self._d.expand_dims(self.data, axis=le_axis).swapaxes(-1, -2) @
                        self._d.expand_dims(out.grad, axis=rese_axis),
                        axis = r
                    ),
                    other.data.shape
                )

        out.grad_fn = _matmul_backward
        out.set_requires_grad(True)

        return out

    def __add__(self, other):
        """
        elementwise add (takes broadcasting into account)
        """

        if isinstance(other, (int, float)):
            out = Tensor(self.data + other, _children=(self, ), _op='+', use_np=self.is_np_tensor)

            if self.requires_grad:
                def _add_backward_scalar():
                        self.grad += out.grad

                out.grad_fn = _add_backward_scalar
                out.set_requires_grad(True)
            
            return out
                        
        else:
            other = other if isinstance(other, Tensor) else Tensor(other, use_np=self.is_np_tensor)
            out = Tensor(self.data + other.data, _children=(self, other), _op='+')

            if self.requires_grad == False and other.requires_grad == False:
                return out
            
            if self.shape == other.shape:
                # same shape, so gradient for addition will be just propagated
                # backwards equally to self and other from the resultant Tensor (out)
                def _add_backward_same():
                    if self.requires_grad:
                        self.grad += out.grad
                    if other.requires_grad:
                        other.grad += out.grad   

                out.grad_fn = _add_backward_same
            
            else:
                # different shapes, broadcast occurs
                # gradient will be summed along the broadcasted axes
                # since the out Tensor is result of broadcasting and addition
                # in essence, broadcasted axes are copied and added, so gradients from 
                # all the copies should be added
                laxis, raxis = broadcast_axis(self.data.shape, other.data.shape)

                def _add_backward_diff():
                    if self.requires_grad:
                        self.grad += self._d.reshape(
                            mx.sum(out.grad, axis=laxis), self.shape
                        )
                    if other.requires_grad:
                        other.grad += self._d.reshape(
                            mx.sum(out.grad, axis=raxis), other.shape
                        )
                
                out.grad_fn = _add_backward_diff

        out.set_requires_grad(True)
        return out
    
    def __mul__(self, other):
        """
        element wise multiply (takes broadcasting into account)
        """

        if isinstance(other, (int, float)):
            out = Tensor(self.data * other, _children=(self, ), _op='*', use_np=self.is_np_tensor)

            if self.requires_grad:
                def _mul_backward_scalar():
                    self.grad += other * out.grad

            out.grad_fn = _mul_backward_scalar
            out.set_requires_grad(True)
            return out
        
        else:
            other = other if isinstance(other, Tensor) else Tensor(other, use_np=self.is_np_tensor)
            out = Tensor(self.data * other.data, _children=(self, other), _op='*')
            
            if self.requires_grad == False and other.requires_grad == False:
                return out
            
            if self.shape == other.shape:
                def _mul_backward_same():
                    if self.requires_grad:
                        self.grad += other.data * out.grad
                    if other.requires_grad:
                        other.grad += self.data * out.grad

                out.grad_fn = _mul_backward_same

            else:
                # for broadcast multiply
                # different shapes, broadcast occurs
                # gradient will be summed along the broadcasted axes
                # since the out Tensor is result of broadcasting and addition
                # in essence, broadcasted axes are copied and added, so gradients from 
                # all the copies should be added
                laxis, raxis = broadcast_axis(self.data.shape, other.data.shape)

                def _mul_backward_diff():
                    if self.requires_grad:
                        self.grad += self._d.reshape(
                            mx.sum(other.data * out.grad, axis=laxis), self.shape
                        )
                    if other.requires_grad:
                        other.grad += self._d.reshape(
                            mx.sum(self.data * out.grad, axis=raxis), other.shape
                        )
                
                out.grad_fn = _mul_backward_diff

        out.set_requires_grad(True)
        return out
    
    def __pow__(self, other: Union[int, float]):
        """
        raise the tensor to some int or float power
        """

        assert isinstance(other, (int, float)), f"Tensor power for {type(other)} is not supported."

        # numpy and mlx don't allow raising by negative values
        def _neg_pow(a, b: Union[int, float]):
            assert isinstance(b, (int, float))
            return 1 / (a ** abs(b)) if b < 0 else a ** b
        
        out = Tensor(
            _neg_pow(self.data, other), _children=(self, ), _op="pow", use_np=self.is_np_tensor 
        )

        # gradient is: p * a^(p-1)
        if self.requires_grad:
            def _pow_backward():
                self.grad += other * _neg_pow(self.data, other - 1) * out.grad

            out.grad_fn = _pow_backward
            out.set_requires_grad(True)

        return out
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other: "Tensor"):
        return self + (-other)
    
    def __radd__(self, other: "Tensor"):  # other + self
        return self + other
    
    def __rmul__(self, other: "Tensor"):  # other * self
        return self * other

    def __truediv__(self, other: "Tensor"):
        return self * (other ** -1)
    
    def __rtruediv__(self, other: "Tensor"):    # other / self
        return (self ** -1) * other
    
    def __repr__(self) -> str:
        if self.requires_grad:
            return f"Tensor({self.data}, requires_grad={self.requires_grad}, is_mlx_tensor={not self.is_np_tensor})"
        
        return f"Tensor({self.data}, is_mlx_tensor={not self.is_np_tensor})"
