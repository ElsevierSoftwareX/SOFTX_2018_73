import dolfin as dlf
import fenicsmechanics as fm

# Region IDs
HNBC  = 0  # homogeneous Neumann BC
HDBC  = 10  # homogeneous Dirichlet BC
INBC  = 20  # inhomogeneous Neumann BC

# fname = "../../../meshfiles/beam/beam_1000.h5"
# mesh = fm.load_mesh(fname)

import mshr
domain = mshr.Box(dlf.Point(), dlf.Point(10, 1, 1))
mesh = mshr.generate_mesh(domain, 80)

boundaries = dlf.MeshFunction("size_t", mesh, 2)
boundaries.set_all(HNBC)

hdbc = dlf.CompiledSubDomain("near(x[0], 0.0) && on_boundary")
hdbc.mark(boundaries, HDBC)

inbc = dlf.CompiledSubDomain("near(x[2], 0.0) && on_boundary")
inbc.mark(boundaries, INBC)

mesh = {
    # 'mesh_file': fname,
    # 'boundaries': fname
    'mesh_file': mesh,
    'boundaries': boundaries
}

cf = dlf.Constant([1., 0., 0.])
cs = dlf.Constant([0., 1., 0.])

material = {
    'type': 'elastic',
    'const_eqn': 'guccione',
    'incompressible': True,
    'density': 0.0,
    'C': 2.0,
    'bf': 8.0,
    'bt': 2.0,
    'bfs': 4.0,
    'kappa': 1e30,
    'fibers': {
        'fiber_files': [cf, cs],
        'fiber_names': ['e1', 'e2'],
        'element': None
    }
}

formulation = {
    'time': {
        'dt': 0.1,
        'interval': [0., 1.]
    },
    'element': 'p2-p1',
    # 'element': 'p1',
    'domain': 'lagrangian',
    'bcs': {
        'dirichlet': {
            'displacement': [[0., 0., 0.]],
            'regions': [HDBC]
        },
        'neumann': {
            'regions': [INBC],
            'values': ["0.004*t"],
            'types': ['pressure']
        }
    }
}

config = {'mesh': mesh, 'material': material, 'formulation': formulation}

problem = fm.SolidMechanicsProblem(config)
solver = fm.SolidMechanicsSolver(problem, fname_disp="results/displacement.pvd")
solver.full_solve()

print("DOF(u) = ", problem.displacement.function_space().dim())

import numpy as np
vals = np.zeros(3)
x = np.array([10., 0.5, 1.])
# problem.displacement.set_allow_extrapolation(True)
rank = dlf.MPI.rank(dlf.mpi_comm_world())
try:
    problem.displacement.eval(vals, x)
    print("(rank %i) vals + x = " % rank, vals + x)
except RuntimeError:
    pass
