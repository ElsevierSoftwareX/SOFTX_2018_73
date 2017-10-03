# Import the objects necessary to use package.
from .solidmechanics import SolidMechanicsProblem, SolidMechanicsSolver
from .materials.solid_materials import *
from .materials.fluids import *
from .utils import *

import sys as _sys
if _sys.version_info[0] < 3:
    from .mechanicsproblem import MechanicsProblem
    from .mechanicssolver import MechanicsBlockSolver
else:
    s = """
    **********************************************************
    *                        WARNING:                        *
    *  The CBC-Block FEniCS App, which MechanicsBlockSolver  *
    *      depends on, is not compatible with python 3.      *
    **********************************************************
    """
    print(s)
