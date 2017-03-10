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
parser.add_argument('-s','--save',
                    help='save solution',
                    action='store_true')
parser.add_argument('-v', '--compute-volume',
                    help='compute deformed volume',
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
if args.save:
    if args.inverse:
        disp_file = 'results/inverse-disp-%s-%s.pvd' % name_dims
        vel_file = 'results/inverse-vel-%s-%s.pvd' % name_dims
    else:
        disp_file = 'results/forward-disp-%s-%s.pvd' % name_dims
        vel_file = 'results/forward-vel-%s-%s.pvd' % name_dims
else:
    disp_file = None
    vel_file = None

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
dlf.parameters['form_compiler']['representation'] = 'uflacs'
ffc_options = {'optimize' : True,
               'eliminate_zeros' : True,
               'precompute_basis_const' : True,
               'precompute_ip_const' : True}

# Elasticity parameters
E = 20.0 # Young's modulus
# nu = 0.499999999 # Poisson's ratio
nu = 0.5 # Poisson's ratio
# la = E*nu/((1. + nu)*(1. - 2.*nu)) # 1st Lame parameter
inv_la = (1. + nu)*(1. - 2.*nu)/(E*nu)
# la = None
mu = E/(2.*(1. + nu)) # 2nd Lame parameter

# Traction on the Neumann boundary region
trac = dlf.Constant([1.0] + [0.0]*(args.dim-1))

# Region IDs
ALL_ELSE = 0
CLIP = 1
TRACTION = 2

if args.material == 'lin_elastic':
    domain = 'eulerian'
else:
    domain = 'lagrangian'

# Problem configuration dictionary
config = {'material' :
          {
              'const_eqn' : args.material,
              'type' : 'elastic',
              'incompressible' : args.incompressible,
              'density' : 10.0,
              # 'lambda' : la,
              'inv_la': inv_la,
              'mu' : mu,
              'kappa' : kappa
              },
          'mesh' :
          {
              'mesh_file' : mesh_file,
              'mesh_function' : mesh_function,
              'element' : element_type
              },
          'formulation' :
          {
              'time' :
              {
                  'unsteady' : False
                  },
              'domain' : domain,
              'inverse' : args.inverse,
              'body_force' : dlf.Constant((0.,)*args.dim),
              'bcs' :
              {
                  'dirichlet' :
                  {
                      'displacement' : [dlf.Constant([0.]*args.dim)],
                      'regions' : [CLIP],
                      },
                  'neumann' :
                  {
                      'regions' : [TRACTION],
                      'types' : ['cauchy'],
                      'values' : [trac]
                      }
                  }
              }
          }

problem = fm.MechanicsProblem(config)
my_solver = fm.MechanicsSolver(problem)
my_solver.solve(print_norm=True,
                iter_tol=1e-6,
                maxLinIters=50,
                fname_disp=disp_file,
                show=2)

# Plot solution if running on one process.
if dlf.MPI.size(dlf.mpi_comm_world()) == 1:
    dlf.plot(problem.displacement, interactive=True, mode='displacement')

# Compute the final volume
if args.compute_volume:
    W1 = dlf.VectorFunctionSpace(problem.mesh, 'CG', 1)
    xi1 = dlf.TestFunction(W1)
    du1 = dlf.TrialFunction(W1)
    u_move = dlf.Function(W1)
    move_bcs = dlf.DirichletBC(W1, dlf.Constant([0.0]*args.dim),
                               problem.mesh_function, CLIP)
    a = dlf.dot(xi1, du1)*dlf.dx
    L = dlf.dot(xi1, problem.displacement)*dlf.dx
    dlf.solve(a == L, u_move, move_bcs)

    ale = dlf.ALE()
    ale.move(problem.mesh, u_move)
    print "Total volume after: ", \
        dlf.assemble(dlf.Constant(1.0)*dlf.dx(domain=problem.mesh))
