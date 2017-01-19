# Import the objects necessary to use package.
from .mechanicsproblem import MechanicsProblem
from .mechanicssolver import MechanicsSolver
from .materials.solid_materials import *
from .utils import *

# Delete variables to clean up dir.
del mechanicsproblem
del mechanicssolver
del materials
del utils
del dlf

# Setup for "import *"
__all__ = list()
__all__ += dir()
