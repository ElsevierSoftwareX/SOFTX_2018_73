import dolfin as _dlf

from collections import namedtuple as _nt

__all__ = ["MPI_COMM_WORLD"]

_version_info = _nt("_version_info", ["major", "minor", "micro"])
_dlf_version_info = _version_info(*tuple(map(int, _dlf.__version__.split("."))))
if _dlf_version_info.major >= 2018:
    MPI_COMM_WORLD = _dlf.MPI.comm_world
else:
    MPI_COMM_WORLD = _dlf.mpi_comm_world()
