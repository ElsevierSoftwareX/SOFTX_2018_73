"""
This module checks the FEniCS version and provides handles to
variables that have changed in between different FEniCS versions.

Variables defined:
* MPI_COMM_WORLD
* LOG_LEVEL_ERROR

"""
import dolfin as _dlf

__all__ = ["MPI_COMM_WORLD", "LOG_LEVEL_ERROR"]

if _dlf.__version__.startswith('2018'):
    MPI_COMM_WORLD = _dlf.MPI.comm_world
    LOG_LEVEL_ERROR = _dlf.LogLevel.ERROR
else:
    MPI_COMM_WORLD = _dlf.mpi_comm_world()
    LOG_LEVEL_ERROR = _dlf.ERROR
