from smolgrad import Tensor
import smolgrad.nn as nn

in_feats, out_feats = 2, 1

model = nn.Sequential(
    nn.Linear(in_feats, 5),
    nn.ReLU(),
    nn.Linear(5, 10),
    nn.ReLU(),
    nn.Linear(10, out_feats),
    nn.Sigmoid()    
)

print(model.parameters())

# Example output
# Output of the loss function:  Tensor(array(2, dtype=float32), requires_grad=True, is_mlx_tensor=True)
# array([0, -2, 2, 0, 0, 0], dtype=float32)