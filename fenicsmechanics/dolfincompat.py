"""
This module checks the FEniCS version and provides handles to
variables that have changed in between different FEniCS versions.

Variables defined:
* MPI_COMM_WORLD
* LOG_LEVEL_ERROR

"""
import dolfin as _dlf

__all__ = ["MPI_COMM_WORLD",
           "LOG_LEVEL_ERROR",
           "DOLFIN_VERSION_INFO",
           "convert_version_info"]


def convert_version_info(version_string):
    from collections import namedtuple
    version_info = namedtuple("version_info", ["major", "minor", "micro"])
    version_tuple = version_info(*map(int, version_string.split(".")))
    return version_tuple

DOLFIN_VERSION_INFO = convert_version_info(_dlf.__version__)

if DOLFIN_VERSION_INFO.major == 2018:
    MPI_COMM_WORLD = _dlf.MPI.comm_world
    LOG_LEVEL_ERROR = _dlf.LogLevel.ERROR
else:
    MPI_COMM_WORLD = _dlf.mpi_comm_world()
    LOG_LEVEL_ERROR = _dlf.ERROR
