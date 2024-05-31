from smolgrad import Tensor
import smolgrad.nn as nn

a = Tensor([num for num in range(1, 7)], requires_grad=True)

mod = nn.Sequential(
    nn.Linear(in_features=a.shape[0], out_features=3),
    nn.ReLU(),
    nn.Linear(in_features=3, out_features=3),
    nn.Sigmoid()
)

out: Tensor = mod(a)
print("Output of the model: ", out)

loss = out.sum()

loss.backward()

# usually we don't care about gradients of input
# but this is just for an example of backprop from 
# output layer to input layer
print(a.grad)


# Example output
# Output of the model:  Tensor(array([0.469774, 0.389416, 0.504945], dtype=float32), requires_grad=True, is_mlx_tensor=True)
# array([0.0401431, 0.0918337, -0.0767768, 0.0143847, 0.0724958, -0.110774], dtype=float32)