from ..core import Tensor
from ._module import Module


class Linear(Module):
    def __init__(
        self, in_features: int, out_features: int, 
        use_bias: bool = True, device: str = "gpu", dtype = None
    ):
        """
        Fully connected linear layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool, optional): Whether to include bias terms. Default is True.
            device (str, optional): Device on which the layer's tensors should reside. Default is "cpu".
            dtype (str, optional): Data type of the tensors. Default is "float32".
        """

        super().__init__(device)

        self.use_bias = use_bias
        self.use_np = False if device=="gpu" else True

        self.weight = Tensor(
            self._d.random.uniform(-1, 1, (out_features, in_features)),
            dtype=dtype, requires_grad=True, use_np=self.use_np
        )
        if self.use_bias:
            self.bias = Tensor(
                self._d.random.uniform(-1, 1, (out_features, )),
                dtype=dtype, requires_grad=True, use_np=self.use_np
            )

    def forward(self, X: Tensor) -> Tensor:
        """
        fully connected layer's output is:
        
        `o = w @ x + b`
        """
        out = X @ self.weight.T() + self.bias
        return out
