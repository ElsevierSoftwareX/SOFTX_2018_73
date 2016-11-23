from .mechanicsproblem import MechanicsProblem
from .mechanicssolver import MechanicsSolver
from .materials.elastic import lin_elastic, neo_hookean

import dolfin as _dlf
_dlf.parameters['form_compiler']['representation'] = 'uflacs'
del _dlf
