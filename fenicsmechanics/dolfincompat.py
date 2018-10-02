import dolfin as _dlf

__all__ = ["MPI_COMM_WORLD"]

if _dlf.__version__.startswith('2018'):
    MPI_COMM_WORLD = _dlf.MPI.comm_world
else:
    MPI_COMM_WORLD = _dlf.mpi_comm_world()
