from __future__ import print_function

import sys
import argparse
import numpy as np
import dolfin as dlf
import fenicsmechanics as fm

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--material",
                    help="constitutive relation",
                    default='guccione',
                    choices=['linear','neo-hooke','aniso','guccione'])
parser.add_argument("--output_dir",
                    help="output directory",
                    default='problem_2')
parser.add_argument("--meshname",
                    help="mesh file",
                    default='../meshfiles/ellipsoid/new-ellipsoid_1000um.h5')
parser.add_argument("-i","--inverse",
                    help="activate inverse elasticity",
                    action='store_true')
parser.add_argument("-pd", "--polynomial_degree",
                    help="polynomial degree of the ansatz functions",
                    default=1,
                    type=int)
parser.add_argument("-ic","--incompressible",
                    help="block formulation for incompressible materials",
                    action='store_true')
parser.add_argument("--pressure",
                    help="target pressure of inflation test in kPA",
                    default=10,
                    type=float)
parser.add_argument("-nu", "--poissons_ratio",
                    help="poissons ratio",
                    default=0.4,
                    type=float)
parser.add_argument("-kappa", "--bulk_modulus",
                    help="bulk modulus",
                    default=1000.0,
                    type=float)
parser.add_argument("-mu", "--shear_modulus",
                    help="shear modulus",
                    default=100.,
                    type=float)
parser.add_argument("-ls", "--loading_steps",
                    help="number of loading steps",
                    default=50,
                    type=int)
parser.add_argument("--solver",
                    help="choose solving method",
                    default='mumps',
                    choices=['umfpack','mumps','pastix','hypre_amg','ml_amg','petsc_amg'])
args = parser.parse_args()

if args.incompressible:
    # name_dims = ('incomp_' + args.material, dim_str)
    element_type = 'p%i-p%i' % (args.polynomial_degree,
                                args.polynomial_degree - 1)
else:
    # name_dims = ('comp_' + args.material, dim_str)
    element_type = 'p%i' % args.polynomial_degree

# Optimization options for the form compiler
dlf.parameters['form_compiler']['cpp_optimize'] = True
dlf.parameters['form_compiler']['representation'] = "uflacs"
dlf.parameters['form_compiler']['quadrature_degree'] = 4
dlf.parameters['form_compiler']['optimize'] = True

rank = dlf.MPI.rank(dlf.mpi_comm_world())
if rank !=0:
    dlf.set_log_level(dlf.ERROR)

# --- GLOBAL VARIABLES --------------------------------------------------------
output_dir = args.output_dir

# Region IDs
HNBC  = 0  # homogeneous Neumann BC
HDBC  = 10  # homogeneous Dirichlet BC
INBC  = 20  # inhomogeneous Neumann BC

# Elasticity parameters
nu = args.poissons_ratio
mu = args.shear_modulus
kappa = args.bulk_modulus
bf = bt = bfs = 1.0
C = 10.0

# Time parameters
t0, tf = tspan = [0., 1.]
dt = (tf - t0)/args.loading_steps
theta = 1.0
beta = 0.25
gamma = 0.5
save_freq = 1

pressure = dlf.Expression('%f*t' % args.pressure, degree=2, t=t0)

config = {
    'material': {
        'const_eqn': args.material,
        'type': 'elastic',
        'incompressible': args.incompressible,
        'density': 0.0,
        'nu': nu,
        'mu': mu,
        'C': C,
        'bf': bf,
        'bt': bt,
        'bfs': bfs
        },
    'mesh': {
        'mesh_file': args.meshname,
        'mesh_function': args.meshname,
        'fiber_file': args.meshname,
        'element': element_type
        },
    'formulation': {
        'time': {
            'unsteady': True,
            'theta': theta,
            'beta': beta,
            'gamma': gamma,
            'dt': dt,
            'interval': tspan
        },
        'domain': 'lagrangian',
        'inverse': args.inverse,
        'bcs': {
            'dirichlet': {
                'displacement': [dlf.Constant([0.]*3)],
                'regions': [HDBC],
                'velocity': [dlf.Constant([0.]*3)]
                },
            'neumann': {
                'regions': [INBC],
                'types': ['pressure'],
                'values': [pressure]
                }
            }
        }
    }

result_file = args.output_dir + '/ellipsoid-fm.pvd'
problem = fm.SolidMechanicsProblem(config)
solver = fm.SolidMechanicsSolver(problem)
solver.full_solve(save_freq=save_freq, fname_disp=result_file)
