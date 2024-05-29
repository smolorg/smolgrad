import smolgrad.core as sc
import smolgrad.nn as nn

inp = sc.Tensor([1, 2, 3, 4, 5])

layer = nn.Linear(5, 10)

out: sc.Tensor = layer(inp)

print("Output from Linear layer: ", out)
print("Input shape: ", inp.shape)
print("Output shape: ", out.shape)
print("Layer weights shape: ", layer.w.shape)
print("Layer bias shape: ", layer.b.shape)

# Output from Linear layer:  Tensor(array([-0.451915, -6.91314, -1.51139, ..., 5.53345, 0.63165, -0.392751], dtype=float32), requires_grad=True, is_mlx_tensor=True)
# Input shape:  (5,)
# Output shape:  (10,)
# Layer weights shape:  (5, 10)
# Layer bias shape:  (10,)