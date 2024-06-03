from smolgrad.core import Tensor

a = Tensor([[1, 2], [3, 4]], requires_grad=True)

b = a.mean(axis = 1)
c = b.sum()

c.backward()

print("output of mean: ", b)
print("gradient of a: ", a.grad)
