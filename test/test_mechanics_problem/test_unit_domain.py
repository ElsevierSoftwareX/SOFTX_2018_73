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
    kappa = 1e8
else:
    name_dims = ('comp_' + args.material, dim_str)
    element_type = 'p2'
    kappa = None

mesh_dir = '../meshfiles/unit_domain/'

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
                    + 'Please run the script \'generate_mesh_files.py\' '
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
E = 20.0 # Young's modulus
nu = 0.3 # Poisson's ratio
la = E*nu/((1. + nu)*(1. - 2.*nu)) # 1st Lame parameter
mu = E/(2.*(1. + nu)) # 2nd Lame parameter

# Traction on the Neumann boundary region
trac = dlf.Constant([10.0] + [0.0]*(args.dim-1))

# Region IDs
ALL_ELSE = 0
CLIP = 1
TRACTION = 2

if args.material == 'lin_elastic':
    domain = 'eulerian'
else:
    domain = 'lagrangian'

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
                  'unsteady' : False
                  },
              'domain' : domain,
              'inverse' : args.inverse,
              'body_force' : dlf.Constant((0.,)*args.dim),
              'bcs' : {
                  'dirichlet' : {
                      'displacement' : [dlf.Constant([0.]*args.dim)],
                      'regions' : [CLIP],
                      },
                  'neumann' : {
                      'regions' : [TRACTION],
                      'types' : ['cauchy'],
                      'values' : [trac]
                      }
                  }
              }
          }

problem = fm.MechanicsProblem(config)
my_solver = fm.MechanicsSolver(problem)
my_solver.solve(print_norm=True)

# Plot solution if running on one process.
if dlf.MPI.size(dlf.mpi_comm_world()) == 1:
    dlf.plot(problem.displacement, interactive=True, mode='displacement')
