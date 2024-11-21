import numpy as np
from mpi4py import MPI

import smolgrad.nn as nn
from smolgrad import Tensor
from smolgrad.optims import SGD
from smolgrad.distributed import DistributedDataParallel as DDP

ROOT_PROCESS_ID = 0
comm = MPI.COMM_WORLD
model = nn.Sequential(
    nn.Linear(2, 5),
    nn.ReLU(),
    nn.Linear(5, 10),
    nn.ReLU(),
    nn.Linear(10, 1),
    nn.Sigmoid()
)
lr = 0.001
ddp_model = DDP(model, comm=comm, root=ROOT_PROCESS_ID)
optimizer = SGD(ddp_model.parameters(), lr)
ddp_model.train()

# simulates 1 step of training, 1 micro_batch
micro_batch = Tensor(np.random.normal(size=(4, 2)).astype(np.float32))
output = ddp_model(micro_batch)
loss = output.sum()
optimizer.zero_grad()
loss.backward()
ddp_model.synchronize()
optimizer.step()

print(f"Rank {comm.rank}:")
print(f"Last param's gradient: {ddp_model.parameters()[-1].grad}")