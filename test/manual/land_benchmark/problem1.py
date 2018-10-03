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

parser = argparse.ArgumentParser()
parser.add_argument("--pressure",
                    default=0.004, type=float,
                    help="Pressure to be applied at z = 0 (default 0.004 kPa).")

mesh_dir = fm._get_mesh_file_names("beam", ret_dir=True, ret_mesh=False)
mesh_file = os.path.join(mesh_dir, "beam_1000.h5")
parser.add_argument("--mesh-file",
                    type=str, default=mesh_file,
                    help="Name of mesh file to use for mesh and facet function.")
parser.add_argument("--generate-mesh",
                    action="store_true", help="Generates mesh using mshr.")
parser.add_argument("--resolution",
                    default=70, type=int,
                    help="Resolution used to generate mesh with mshr \
                          (default: 70).")
parser.add_argument("--incompressible",
                    action="store_true", help="Model as incompressible material.")
parser.add_argument("--bulk-modulus",
                    type=float, default=1e3, dest="kappa",
                    help="Bulk modulus of the material (default: 1e3).")
parser.add_argument("--loading-steps", "-ls",
                    type=int, default=10,
                    help="Number of loading steps to use (default: 10).")
parser.add_argument("--polynomial-degree", "-pd",
                    type=int, default=2, dest="pd", choices=[1, 2, 3],
                    help="Polynomial degree to be used for displacement \
                          (default: 2).")
args = parser.parse_args()

# Region IDs
HNBC  = 0  # homogeneous Neumann BC
HDBC  = 10  # homogeneous Dirichlet BC
INBC  = 20  # inhomogeneous Neumann BC

KAPPA = 1e100 if args.incompressible else args.kappa

if args.generate_mesh:
    import mshr
    domain = mshr.Box(dlf.Point(), dlf.Point(10, 1, 1))
    mesh_file = mshr.generate_mesh(domain, args.resolution)

    boundaries = dlf.MeshFunction("size_t", mesh_file, 2)
    boundaries.set_all(HNBC)

    hdbc = dlf.CompiledSubDomain("near(x[0], 0.0) && on_boundary")
    hdbc.mark(boundaries, HDBC)

    inbc = dlf.CompiledSubDomain("near(x[2], 0.0) && on_boundary")
    inbc.mark(boundaries, INBC)
else:
    mesh_file = boundaries = args.mesh_file

mesh = {
    'mesh_file': mesh_file,
    'boundaries': boundaries
}

cf = dlf.Constant([1., 0., 0.])
cs = dlf.Constant([0., 1., 0.])
material = {
    'type': 'elastic',
    'const_eqn': 'guccione',
    'incompressible': args.incompressible,
    'density': 0.0,
    'C': 2.0,
    'bf': 8.0,
    'bt': 2.0,
    'bfs': 4.0,
    'kappa': KAPPA,
    'fibers': {
        'fiber_files': [cf, cs],
        'fiber_names': ['e1', 'e2'],
        'element': None
    }
}

interval = [0., 1.]
dt = (interval[1] - interval[0])/args.loading_steps
formulation = {
    'time': {
        'dt': dt,
        'interval': interval
    },
    'domain': 'lagrangian',
    'bcs': {
        'dirichlet': {
            'displacement': [[0., 0., 0.]],
            'regions': [HDBC]
        },
        'neumann': {
            'regions': [INBC],
            'values': ["%f*t" % args.pressure],
            'types': ['pressure']
        }
    }
}

if args.incompressible:
    formulation['element'] = 'p%i-p%i' % (args.pd, args.pd - 1)
else:
    formulation['element'] = 'p%i' % args.pd

config = {'mesh': mesh, 'material': material, 'formulation': formulation}

if args.incompressible:
    fname_disp = "results/displacement-incompressible.pvd"
    fname_pressure = "results/pressure.pvd"
else:
    fname_disp = "results/displacement.pvd"
    fname_pressure = None
problem = fm.SolidMechanicsProblem(config)
solver = fm.SolidMechanicsSolver(problem, fname_disp=fname_disp,
                                 fname_pressure=fname_pressure)
solver.full_solve()

rank = dlf.MPI.rank(MPI_COMM_WORLD)
if rank == 0:
    print("DOF(u) = ", problem.displacement.function_space().dim())

import numpy as np
vals = np.zeros(3)
x = np.array([10., 0.5, 1.])
try:
    problem.displacement.eval(vals, x)
    print("(rank %i) vals + x = " % rank, vals + x)
except RuntimeError:
    pass
