from typing import Optional
from ..core import Tensor
from ._module import Module

class Embedding(Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: str = "gpu"
    ):
        super().__init__(device)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.use_np = False if device == "gpu" else True

        # initialize the embedding weight matrix
        self.weight = Tensor(
            self._d.random.normal((num_embeddings, embedding_dim)),
            requires_grad=True,
            use_np=self.use_np
        )

    def forward(self, input: Tensor) -> Tensor:
        # ensure input is of integer type
        if input.dtype not in [self._d.int32, self._d.int64]:
            input.data = input.data.astype(self._d.int32)
            input.dtype = self._d.int32

        # perform the embedding lookup
        out = self.weight[input.data]
        return out
