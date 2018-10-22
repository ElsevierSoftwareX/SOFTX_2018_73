import dolfin as dlf
import fenicsmechanics as fm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--save',
                    help='save solution',
                    action='store_true')
parser.add_argument('-bs', '--block-solver',
                    help='Use MechanicsBlockSolver',
                    action='store_true')
args = parser.parse_args()

mesh_file, boundaries = fm.get_mesh_file_names("lshape", ret_facets=True, refinements="fine")

# Region IDs
ALL_ELSE = 0
INFLOW = 1
OUTFLOW = 2
NOSLIP = 3

# Inlet pressure
p_in = "sin(3.0*t)"

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
    'boundaries': boundaries
}

# Formulation subdictionary
formulation_dict = {
    'time': {
        'dt': 0.01,
        'interval': [0., 3.],
        'theta': 1.0
    },
    'element': 'p2-p1',
    'domain': 'eulerian',
    'body_force': dlf.Constant([0., 0.]),
    'bcs': {
        'dirichlet': {
            'velocity': [dlf.Constant([0.]*2)],
            'regions': [NOSLIP]
        },
        'neumann': {
            'regions': [INFLOW, OUTFLOW],
            'types': ['pressure', 'pressure'],
            'values': [p_in, 0.0]
        }
    }
}

config = {
    'material': mat_dict,
    'mesh': mesh_dict,
    'formulation': formulation_dict
}

if args.save:
    vel_file = 'results/lshape-%s-unsteady-velocity.pvd'
    pressure_file = 'results/lshape-%s-unsteady-pressure.pvd'
    if args.block_solver:
        vel_file = vel_file % "bs"
        pressure_file = pressure_file % "bs"
    else:
        vel_file = vel_file % "mixed"
        pressure_file = pressure_file % "mixed"
else:
    vel_file = None
    pressure_file = None

if args.block_solver:
    problem = fm.MechanicsProblem(config)
    solver = fm.MechanicsBlockSolver(problem, fname_vel=vel_file,
                                     fname_pressure=pressure_file)
    solver.solve()
else:
    problem = fm.FluidMechanicsProblem(config)
    solver = fm.FluidMechanicsSolver(problem, fname_vel=vel_file,
                                     fname_pressure=pressure_file)
    solver.full_solve()
