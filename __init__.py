# Import the objects necessary to use package.
from .mechanicsproblem import MechanicsProblem
from .mechanicssolver import MechanicsSolver
from .materials.solid_materials import *
from .utils import *

# Check if module is being reloaded.
try:
    reloading
except NameError:
    reloading = False # Module is being imported.
else:
    reloading = True # Module is being reloaded.

# Delete variables to clean up dir.
if not reloading:
    del mechanicsproblem
    del mechanicssolver
    del materials
    del utils
    del dlf

# Setup for "import *"
__all__ = list()
__all__ += dir()
