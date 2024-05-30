import smolgrad.core as sc
import smolgrad.nn as nn

inp = sc.Tensor([1, 2, 3, 4, 5])

layer = nn.Linear(5, 10)
# output between 0 and 1
act = nn.Sigmoid()

out: sc.Tensor = act(layer(inp))

out.backward()

print("Output from Linear layer: ", out)
print("Input shape: ", inp.shape)
print("Output shape: ", out.shape)
print("Layer weights shape: ", layer.w.shape)
print("Layer bias shape: ", layer.b.shape)
print("Bias grad for example: ", layer.b.grad)

# Output from Linear layer:  Tensor(array([0.0110387, 0.998355, 0.118756, ..., 0.99935, 0.322409, 0.000104427], dtype=float32), requires_grad=True, is_mlx_tensor=True)
# Input shape:  (5,)
# Output shape:  (10,)
# Layer weights shape:  (10, 5)
# Layer bias shape:  (10,)
# Bias grad for example:  array([0.0109169, 0.00164224, 0.104653, ..., 0.000649192, 0.218462, 0.000104416], dtype=float32)
