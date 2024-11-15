from smolgrad import Tensor


def print_hook(tensor: Tensor):
    print(">> in gradient hook for tensor of shape: ", tensor.shape)


a = Tensor([1, 2, 3, 4], use_np=True, requires_grad=True)
a.register_grad_hook(print_hook)
b = Tensor([1, 2, 3, 4], use_np=True, requires_grad=True)
b.register_grad_hook(print_hook)

c = a.cat([b])
c.register_grad_hook(print_hook)
loss = c.sum()
loss.register_grad_hook(print_hook)
loss.backward()
