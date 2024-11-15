import smolgrad.nn as nn


def check_params_equal(model1: nn.Sequential, model2: nn.Sequential):
    params1, params2 = model1.parameters(), model2.parameters()
    for i, param in enumerate(params1):
        if not (param == params2[i]):
            return False
    return True


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
print("Are parameters equal?", check_params_equal(model1, model2))
print("\n>> After loading:")
model2.load_state_dict(model1.state_dict())
print("Are parameters equal now?", check_params_equal(model1, model2))
