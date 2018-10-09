import os
import sys
import argparse
import numpy as np

from mpi4py import MPI

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

parser.add_argument("--mesh-size",
                    type=int, choices=[200, 250, 300, 500, 1000], default=1000,
                    help="Mesh size for the ellipsoid.")
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

mesh_file =  fm._get_mesh_file_names("ellipsoid", ext="h5",
                                     refinements=["%ium" % args.mesh_size])
fiber_files = mesh_file

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
        'fiber_files': fiber_files,
        'fiber_names': [['fib1', 'fib2', 'fib3'],
                        ['she1', 'she2', 'she3']],
        'elementwise': True
    }
}

# Mesh subdictionary
mesh = {
    'mesh_file': mesh_file,
    'boundaries': mesh_file
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
    fname_disp = "results/ellipsoid-displacement-incompressible.xml.gz"
    fname_pressure = "results/ellipsoid-pressure.xml.gz"
    fname_hdf5 = "results/ellipsoid-incompressible.h5"
    fname_xdmf = "results/ellipsoid-incompressible-viz.xdmf"
else:
    fname_disp = "results/ellipsoid-displacement.xml.gz"
    fname_pressure = None
    fname_hdf5 = "results/ellipsoid.h5"
    fname_xdmf = "results/ellipsoid-viz.xdmf"
problem = fm.SolidMechanicsProblem(config)
solver = fm.SolidMechanicsSolver(problem, fname_disp=fname_disp,
                                 fname_pressure=fname_pressure,
                                 fname_xdmf=fname_xdmf)
solver.set_parameters(linear_solver="mumps")
solver.full_solve()

rank = dlf.MPI.rank(MPI_COMM_WORLD)
disp_dof = problem.displacement.function_space().dim()
if rank == 0:
    print("DOF(u) = ", disp_dof)

import numpy as np
disp_endo = np.zeros(3)
disp_epi = np.zeros(3)
x_endocardium = np.array([0., 0., -17.])
x_epicardium = np.array([0., 0., -20.])
fmt_str = ("%i",) + ("%f",)*3

def find_and_write_loc(vals, x, u, fname):
    try:
        u.eval(vals, x)
        write_to_file = True
    except RuntimeError:
        write_to_file = False

    if write_to_file:
        final_vals = x + vals
        print("(rank %i, %s) x + vals = %s" \
              % (rank, str(tuple(x)), str(tuple(final_vals))))
        data = np.hstack((disp_dof, final_vals)).reshape([1, -1])
        with open(fname, "ab") as f:
            np.savetxt(f, data, fmt=fmt_str)

fname = "ellipsoid-final_location-%s.dat"
u = problem.displacement
find_and_write_loc(disp_endo, x_endocardium, u, fname % "endocardium")
find_and_write_loc(disp_epi, x_epicardium, u, fname % "epicardium")
