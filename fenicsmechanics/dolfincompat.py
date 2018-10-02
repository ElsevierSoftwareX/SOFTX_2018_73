import dolfin as dlf

if dlf.__version__.startswith('2018'):
    MPI_COMM_WORLD = dlf.MPI.comm_world
else:
    MPI_COMM_WORLD = dlf.mpi_comm_world()
