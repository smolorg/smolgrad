import smolgrad
import numpy as np

import smolgrad.distributed


learning_rate = 0.001

comm = smolgrad.distributed.DDPCOMM.COMM_WORLD
model = smolgrad.nn.Sequential(
    smolgrad.nn.Linear(2, 5),
    smolgrad.nn.ReLU(),
    smolgrad.nn.Linear(5, 10),
    smolgrad.nn.ReLU(),
    smolgrad.nn.Linear(10, 1),
    smolgrad.nn.Sigmoid()
)
ddp_model = smolgrad.distributed.DistributedDataParallel(model, comm=comm, root=0)
optimizer = smolgrad.optims.SGD(ddp_model.parameters(), learning_rate)

ddp_model.train()

micro_batch = smolgrad.Tensor(np.random.normal(size=(4, 2)).astype(np.float32))
output = ddp_model(micro_batch)
loss = output.sum()     # dummy loss
optimizer.zero_grad()
loss.backward()
ddp_model.synchronize()
optimizer.step()

# After the step, the parameters of the model
# should be the same on all processes
print(f"Rank {comm.rank}:")
print(f"Last parameter value: {ddp_model.parameters()[-1]}")
