import dolfin as dlf
import fenicsmechanics as fm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--save',
                    help='save solution',
                    action='store_true')
args = parser.parse_args()

mesh_dir = '../meshfiles/lshape/'
mesh_file = mesh_dir + 'lshape-mesh.xml.gz'
mesh_function = mesh_dir + 'lshape-mesh_function.xml.gz'

# Region IDs
ALL_ELSE = 0
INFLOW = 1
OUTFLOW = 2
NOSLIP = 3

# Inlet pressure
p_in = dlf.Expression('sin(3.0*t)', t=0.0, degree=2)

# Material subdictionary
mat_dict = {
    'const_eqn': 'newtonian',
    'type': 'viscous',
    'incompressible': True,
    'density': 1.0,
    'nu': 0.01,
}

# Mesh subdictionary
mesh_dict = {
    'mesh_file': mesh_file,
    'mesh_function': mesh_function,
    'element': 'p2-p1'
}

# Formulation subdictionary
formulation_dict = {
    'time': {
        'unsteady': True,
        'dt': 0.01,
        'interval': [0., 3.],
        'theta': 1.0
    },
    'domain': 'eulerian',
    'body_force': dlf.Constant([0.]*2),
    'bcs': {
        'dirichlet': {
            'velocity': [dlf.Constant([0.]*2)],
            'regions': [NOSLIP]
        },
        'neumann': {
            'regions': [INFLOW, OUTFLOW],
            'types': ['pressure', 'pressure'],
            'values': [p_in, dlf.Constant(0.0)]
        }
    }
}

config = {
    'material': mat_dict,
    'mesh': mesh_dict,
    'formulation': formulation_dict
}

if args.save:
    vel_file = 'results/lshape-unsteady-velocity.pvd'
    pressure_file = 'results/lshape-unsteady-pressure.pvd'
else:
    vel_file = None
    pressure_file = None

problem = fm.MechanicsProblem(config)
solver = fm.MechanicsSolver(problem)
solver.solve(fname_vel=vel_file, fname_pressure=pressure_file)
