# Import the objects necessary to use package.
from .solidmechanics import SolidMechanicsProblem, SolidMechanicsSolver
from .fluidmechanics import *
from .materials.solid_materials import *
from .materials.fluids import *
from .utils import *

import sys as _sys
import dolfin as _dlf
_rank = _dlf.MPI.rank(_dlf.mpi_comm_world())
if _sys.version_info[0] < 3:
    try:
        from .mechanicssolver import MechanicsBlockSolver
    except ImportError:
        _s = """
        **********************************************************
        *                        WARNING:                        *
        *  The CBC-Block FEniCS App, which MechanicsBlockSolver  *
        *       depends on, does not seem to be installed.       *
        **********************************************************
        """
        if _rank == 0: print(_s)
        del _s
else:
    _s = """
    **********************************************************
    *                        WARNING:                        *
    *  The CBC-Block FEniCS App, which MechanicsBlockSolver  *
    *      depends on, is not compatible with python 3.      *
    **********************************************************
    """
    if _rank == 0: print(_s)
    del _s

del _sys, _dlf, _rank

# Users can still create a MechanicsProblem object, but will
# not be able to use the MechanicsBlockSolver is version < 3.
from .mechanicsproblem import MechanicsProblem


def init(quad_degree=2):
    import dolfin as dlf
    dlf.parameters['form_compiler']['cpp_optimize'] = True
    dlf.parameters['form_compiler']['representation'] = "uflacs"
    dlf.parameters['form_compiler']['quadrature_degree'] = quad_degree
    dlf.parameters['form_compiler']['optimize'] = True

init()

__version__ = "1.0.0"
