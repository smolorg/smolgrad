from smolgrad.core import Tensor

x = Tensor([1, 2, 3, 4], requires_grad=True)

a, b = x.split(2, dim=-1)

y = (a * b).sum()

y.backward()

print("output of std: ", y)
print("gradient of a: ", x.grad)
