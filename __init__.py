# # import dolfin as _dlf
# import dolfin as dlf

# from ufl.domain import find_geometric_dimension as _geo_dim

from .mechanicsproblem import *
from .materials.elastic import *

import dolfin as dlf
dlf.parameters['form_compiler']['representation'] = 'uflacs'
