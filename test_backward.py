from smolgrad.core import Tensor

a = Tensor([[1, 2], [3, 4]], requires_grad=True)

b = a.var(axis=1, correction=1)
c = b.sum()

c.backward()

print("output of std: ", b)
print("gradient of a: ", a.grad)
