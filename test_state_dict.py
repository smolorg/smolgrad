import smolgrad.nn as nn

model1 = nn.Sequential(
    nn.Linear(2, 5),
    nn.ReLU(),
    nn.Linear(5, 10),
    nn.ReLU(),
    nn.Linear(10, 1),
    nn.Sigmoid()
)
print("\n>> Model 1's state dict:")
print(model1.state_dict())


model2 = nn.Sequential(
    nn.Linear(2, 5),
    nn.ReLU(),
    nn.Linear(5, 10),
    nn.ReLU(),
    nn.Linear(10, 1),
    nn.Sigmoid()
)

print("\n>> Before loading from state dict of model 1:")
print("Are state dicts equal?: ", model2.state_dict() == model1.state_dict())
print("\n>> After loading:")
model2.load_state_dict(model1.state_dict())
print("Are state dicts equal now?: ", model2.state_dict() == model1.state_dict())

