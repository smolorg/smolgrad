from smolgrad.core import Tensor

a = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
b = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)

c = a + b
d = a - b
d = c * d
e = d.sum()

e.backward()

print(a.grad, b.grad)

# array([[2, 4, 6],
# [8, 10, 12]], dtype=float32) array([[-2, -4, -6],
# [-8, -10, -12]], dtype=float32)
