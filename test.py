from smolgrad.core.engine import Tensor


a = Tensor([1, 2, 3]); a.set_requires_grad(True)
b = Tensor([2, 3, 1]); b.set_requires_grad(True)

c = a + b
d = c + c ** 2 + c ** 3
e = d.sum()

e.backward()

print(a, b)
print(f"Gradient of a:\n{a.grad}")
print(f"Gradient of b:\n{b.grad}")
