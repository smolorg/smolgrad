from smolgrad.core import Tensor

a = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)

a = a + 1
a = a * 2
a = a ** 2
a = a.sum()

a.backward()

print(a.grad)
