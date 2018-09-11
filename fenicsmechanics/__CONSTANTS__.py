dict_implemented = {
    'materials': {
        'elastic': ('lin_elastic', 'neo_hookean', 'guccione', 'fung'),
        'viscous': ('newtonian', 'stokes')
    }
}

import os as _os
_dirname = _os.path.dirname(_os.path.realpath(__file__))
base_mesh_dir = _os.path.abspath(_os.path.join(_dirname, "../meshfiles"))
del _os, _dirname
