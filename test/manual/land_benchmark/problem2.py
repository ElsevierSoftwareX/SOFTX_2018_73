import os
import sys
import argparse
import dolfin as dlf
import fenicsmechanics as fm
from fenicsmechanics.dolfincompat import MPI_COMM_WORLD

# For use with emacs python shell.
try:
    sys.argv.remove("--simple-prompt")
except ValueError:
    pass

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--pressure",
                    default=10.0, type=float,
                    help="Pressure to be applied at z = 0.")

default_mesh_dir = fm._get_mesh_file_names("ellipsoid", ret_dir=True, ret_mesh=False)
default_mesh_file = os.path.join(default_mesh_dir, "ellipsoid_1000um.h5")
default_fiber_file = os.path.join(default_mesh_dir, "fibers/n%i-p0-1000um.xml.gz")
default_fiber_files = [default_fiber_file % i for i in [1, 2]]
parser.add_argument("--mesh-file",
                    type=str, default=default_mesh_file,
                    help="Name of mesh file to use for mesh, facet function, " \
                    + "and fiber directions.")
parser.add_argument("--fiber-files",
                    default=default_fiber_files, nargs=2, type=str,
                    help="Name of files for vector fiber directions.")
parser.add_argument("--incompressible",
                    action="store_true", help="Model as incompressible material.")
parser.add_argument("--bulk-modulus",
                    type=float, default=1e3, dest="kappa",
                    help="Bulk modulus of the material.")
parser.add_argument("--loading-steps", "-ls",
                    type=int, default=100,
                    help="Number of loading steps to use.")
parser.add_argument("--polynomial-degree", "-pd",
                    type=int, default=2, dest="pd", choices=[1, 2, 3],
                    help="Polynomial degree to be used for displacement.")
args = parser.parse_args()

# Region IDs
HNBC  = 0  # homogeneous Neumann BC
HDBC  = 10  # homogeneous Dirichlet BC
INBC  = 20  # inhomogeneous Neumann BC

# Time parameters
t0, tf = interval = [0., 1.]
dt = (tf - t0)/args.loading_steps

KAPPA = 1e100 if args.incompressible else args.kappa

# Material subdictionary
material = {
    'const_eqn': 'guccione',
    'type': 'elastic',
    'incompressible': args.incompressible,
    'density': 0.0,
    'kappa': KAPPA,
    'bt': 1.0,
    'bf': 1.0,
    'bfs': 1.0,
    'C': 10.0,
    'fibers': {
        'fiber_files': args.fiber_files,
        'fiber_names': ['n1', 'n2'],
        'element': 'p0'
    }
}

# Mesh subdictionary
mesh = {
    'mesh_file': args.mesh_file,
    'boundaries': args.mesh_file
}

# Formulation subdictionary
formulation = {
    'time':{'dt': dt, 'interval': interval},
    'domain': 'lagrangian',
    'inverse': False,
    'bcs':{
        'dirichlet': {
            'displacement': [[0.]*3],
            'regions': [HDBC],
            'velocity': [[0.]*3]
        },
        'neumann': {
            'regions': [INBC],
            'types': ['pressure'],
            'values': ["%f*t" % args.pressure]
        }
    }
}

if args.incompressible:
    formulation['element'] = "p%i-p%i" % (args.pd, args.pd - 1)
else:
    formulation['element'] = "p%i" % args.pd

# Overall configuration
config = {
    'material': material,
    'mesh': mesh,
    'formulation': formulation
}

if args.incompressible:
    fname_disp = "results/displacement-incompressible.pvd"
    fname_pressure = "results/pressure.pvd"
else:
    fname_disp = "results/displacement.pvd"
    fname_pressure = None
problem = fm.SolidMechanicsProblem(config)
solver = fm.SolidMechanicsSolver(problem, fname_disp=fname_disp,
                                 fname_pressure=fname_pressure)
solver.set_parameters(linear_solver="mumps")
solver.full_solve()

rank = dlf.MPI.rank(MPI_COMM_WORLD)
if rank == 0:
    print("DOF(u) = ", problem.displacement.function_space().dim())

import numpy as np
vals = np.zeros(3)
x_endocardium = np.array([0., 0., -17.])
x_epicardium = np.array([0., 0., -20.])
try:
    problem.displacement.eval(vals, x_endocardium)
    print("(rank %i, endocardium) vals + x = " % rank, vals + x_endocardium)
    problem.displacement.eval(vals, x_epicardium)
    print("(rank %i, endocardium) vals + x = " % rank, vals + x_epicardium)
except RuntimeError:
    pass
