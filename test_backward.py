from smolgrad.core import Tensor

a = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
b = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)

c = a.cat(b, axis=1)
d = c + c
e = d.log().sum()

e.backward()

print(a.grad, b.grad)

# array([[1, 0.5, 0.333333],
# [0.25, 0.2, 0.166667]], dtype=float32) array([[1, 0.5, 0.333333],
# [0.25, 0.2, 0.166667]], dtype=float32)
