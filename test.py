from smolgrad import Tensor
import smolgrad.nn as nn

pred = Tensor([num for num in range(1, 7)], requires_grad=True)
actual = Tensor([1, 3, 2, 4, 5, 6])

loss = nn.MSELoss()

output: Tensor = loss(pred, actual)
output.backward()

print("Output of the loss function: ", output)
print(pred.grad)


# Example output
# Output of the loss function:  Tensor(array(2, dtype=float32), requires_grad=True, is_mlx_tensor=True)
# array([0, -2, 2, 0, 0, 0], dtype=float32)