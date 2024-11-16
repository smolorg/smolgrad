import smolgrad.nn as nn
from mpi4py import MPI


ROOT_PROCESS_ID = 0
in_features, out_features = 10, 10

comm = MPI.COMM_WORLD


def create_model(comm: MPI.Intracomm):
    rank, tprocs = comm.Get_rank(), comm.Get_size()
    model = nn.Sequential(
        nn.Linear(2, 5),
        nn.ReLU(),
        nn.Linear(5, 10),
        nn.ReLU(),
        nn.Linear(10, 1),
        nn.Sigmoid()
    )
    if rank == ROOT_PROCESS_ID:
        print(f">> total number of processes/cores: {tprocs}")
        state_dict = model.state_dict()
    else:
        state_dict = None
    state_dict = comm.bcast(state_dict, root=ROOT_PROCESS_ID)
    if rank != ROOT_PROCESS_ID:
        print(f"Loading model from state dict in rank {rank} ...")
        model.load_state_dict(state_dict)
    print(f"Last parameter of model in rank {rank}:\n{model.parameters()[-1]}")


create_model(comm)
