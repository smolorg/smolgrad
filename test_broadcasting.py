from smolgrad import Tensor
import numpy as np
import torch

# Simple broadcasting test (forwards and backwards)

arr1 = np.array([1, 2, 3, 4, 5]).astype(np.float32)
arr2 = np.broadcast_to(np.array([1, 2, 3, 4, 5]), (5, 5)).astype(np.float32)
a = Tensor(
    arr1,
    requires_grad=True
)

atorch = torch.tensor(arr1, requires_grad=True)

b = Tensor(
    arr2,
    requires_grad=True
)

btorch = torch.tensor(arr2, requires_grad=True)


def test_broadcasting(fn):
    x = fn(a, b)
    xtorch = fn(atorch, btorch)
    x.sum().backward()
    xtorch.sum().backward()

    assert np.allclose(x.data, xtorch.data.numpy())
    assert np.allclose(a.grad, atorch.grad.numpy())
    assert np.allclose(b.grad, btorch.grad.numpy())
    print("All tests pass")
    # clear the gradients
    a._reset_grad()
    b._reset_grad()
    atorch.grad.zero_()
    btorch.grad.zero_()

test_broadcasting(lambda x, y: x + y)
test_broadcasting(lambda x, y: x - y)
test_broadcasting(lambda x, y: x * y)
test_broadcasting(lambda x, y: x / y)