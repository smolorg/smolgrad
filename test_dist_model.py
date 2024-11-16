from mpi4py import MPI

import smolgrad.nn as nn
from smolgrad import Tensor
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
ddp_model = DDP(model, comm=comm, root=ROOT_PROCESS_ID)


example_input = Tensor([[1, 2], [3, 4]], use_np=True)
output = ddp_model(example_input)

print(f"Rank {comm.rank}:")
print(f"Model input: {example_input}, Model output: {output}")
