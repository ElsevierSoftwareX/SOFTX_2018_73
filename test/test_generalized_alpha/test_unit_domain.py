import os
import sys
import argparse
import dolfin as dlf

import fenicsmechanics as fm

# Parse through the arguments provided at the command line.
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dim',
                    help='dimension',
                    default=2,
                    type=int)
parser.add_argument('-r', '--refinement',
                    help='mesh refinement',
                    default=12,
                    type=int)
parser.add_argument('-m', '--material',
                    help='constitutive relation',
                    default='lin_elastic',
                    choices=['lin_elastic', 'neo_hookean'])
parser.add_argument('-inv', '--inverse',
                    help='activate inverse elasticity',
                    action='store_true')
parser.add_argument('-inc', '--incompressible',
                    help='active incompressible material',
                    action='store_true')
parser.add_argument('-hdf5',
                    help="use HDF5 files",
                    action='store_true')
args = parser.parse_args()

# Mesh file names based on arguments given
mesh_dims = (args.refinement,)*args.dim
dim_str = 'x'.join(['%i' % i for i in mesh_dims])

if args.incompressible:
    name_dims = ('incomp_' + args.material, dim_str)
    element_type = 'p2-p1'
    kappa = dlf.Constant(1e8)
else:
    name_dims = ('comp_' + args.material, dim_str)
    element_type = 'p2'
    kappa = None

mesh_dir = '../meshfiles/unit_domain/'
if args.inverse:
    disp_file = 'results/inverse-disp-%s-%s.pvd' % name_dims
    vel_file = 'results/inverse-vel-%s-%s.pvd' % name_dims
else:
    disp_file = 'results/forward-disp-%s-%s.pvd' % name_dims
    vel_file = 'results/forward-vel-%s-%s.pvd' % name_dims

mesh_file = mesh_dir + 'unit_domain-mesh-%s' % dim_str
mesh_function = mesh_dir + 'unit_domain-mesh_function-%s' % dim_str
if args.hdf5:
    mesh_file += '.h5'
    mesh_function += '.h5'
else:
    mesh_file += '.xml.gz'
    mesh_function += '.xml.gz'

# Check if the mesh file exists
if not os.path.isfile(mesh_file):
    raise Exception('The mesh file, \'%s\', does not exist. ' % mesh_file
                    + 'Please run the script \'generate_mesh_files.py\''
                    + 'with the same arguments first.')

# Check if the mesh function file exists
if not os.path.isfile(mesh_function):
    raise Exception('The mesh function file, \'%s\', does not exist. ' % mesh_function
                    + 'Please run the script \'generate_mesh_files.py\''
                    + 'with the same arguments first.')

# Optimization options for the form compiler
dlf.parameters['form_compiler']['cpp_optimize'] = True
dlf.parameters['form_compiler']['quadrature_degree'] = 3
ffc_options = {'optimize' : True,
               'eliminate_zeros' : True,
               'precompute_basis_const' : True,
               'precompute_ip_const' : True}

# Elasticity parameters
# E = 500.0 # Young's modulus (TRY A SMALLER VALUE)
E = 20.0 # Young's modulus (TRY A SMALLER VALUE)
nu = 0.3 # Poisson's ratio
la = dlf.Constant(E*nu/((1. + nu)*(1. - 2.*nu))) # 1st Lame parameter
mu = dlf.Constant(E/(2.*(1. + nu))) # 2nd Lame parameter

# Time interval
t0 = 0.0
tf = 1.0
nsteps = 1000
dt = (tf - t0)/nsteps
tspan = [t0, tf]
alpha = 1.0

# Traction on the Neumann boundary region
# traction = dlf.Expression(['3.0*t'] + ['0.0']*(args.dim-1), t=t0, degree=2)
pressure = dlf.Expression('3.0*t', t=t0, degree=2)

# Region IDs
ALL_ELSE = 0
CLIP = 1
TRACTION = 2

# Problem configuration dictionary
config = {'material' : {
              'const_eqn' : args.material,
              'type' : 'elastic',
              'incompressible' : args.incompressible,
              'density' : 10.0,
              'lambda' : la,
              'mu' : mu,
              'kappa' : kappa
              },
          'mesh' : {
              'mesh_file' : mesh_file,
              'mesh_function' : mesh_function,
              'element' : element_type
              },
          'formulation' : {
              'time' : {
                  'unsteady' : True,
                  'integrator' : 'generalized_alpha',
                  'alpha' : alpha,
                  'dt' : dt,
                  'interval' : tspan
                  },
              'domain' : 'lagrangian',
              'inverse' : args.inverse,
              'body_force' : dlf.Constant((0.,)*args.dim),
              'bcs' : {
                  'dirichlet' : {
                      'displacement' : {
                          'regions' : [CLIP],
                          'values' : [dlf.Constant([0.]*args.dim)],
                          },
                      'velocity' : {
                          'regions' : [CLIP],
                          'values' : [dlf.Constant([0.]*args.dim)]
                          }
                      },
                  'neumann' : {
                      'regions' : [TRACTION],
                      # 'types' : ['cauchy'],
                      # 'values' : [traction]
                      'types' : ['pressure'],
                      'values' : [pressure]
                      }
                  }
              }
          }

problem = fm.MechanicsProblem(config, form_compiler_parameters=ffc_options)

############################################################
my_solver = fm.MechanicsSolver(problem)
my_solver.solve(iter_tol=1e-6,
                maxLinIters=250,
                fname_disp=disp_file,
                fname_vel=vel_file,
                save_freq=10, show=0)
