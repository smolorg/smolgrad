I think the `DistributedDataParallel` (DDP) class can be initialized like:

```python
from mpi4py import MPI
from smolgrad.distributed import DistributedDataParallel as DDP

ROOT_PROCESS_ID = 0
comm = MPI.COMM_WORLD

ddp_model = DDP(model, comm=comm, root=ROOT_PROCESS_ID)
```

where within the DDP class, we create the model's `state_dict` at the root process and then broadcast it to other processes. After recieving the state dict on other processes, we initialize / overwrite the model's parameters using the recieved state dict.
