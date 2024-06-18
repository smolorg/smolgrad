from smolgrad import Tensor
from smolgrad.core.engine import (
    no_grad
)

a = Tensor(
    [1, 2], 
    requires_grad=True
)

with no_grad():
    x = a + a
    y = x.sum()
    y.backward()

# will be 0
print(a.grad)
