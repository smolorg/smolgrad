from smolgrad import Tensor
from smolgrad.core.engine import (
    no_grad
)

a = Tensor(
    [1, 2], 
    requires_grad=True,
    use_np=True
)

with no_grad():
    x = a + a
    y = x.sum()
    # will raise an error
    y.backward()
