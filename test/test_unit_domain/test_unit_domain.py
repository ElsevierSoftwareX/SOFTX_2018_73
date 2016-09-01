import os
import argparse
import dolfin as dlf

from fenicsmechanics.mechanicsproblem import MechanicsProblem

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
parser.add_argument('-i', '--inverse',
                    help='activate inverse elasticity',
                    action='store_true')
parser.add_argument('-inc', '--incompressible',
                    help='active incompressible material',
                    action='store_true')
args = parser.parse_args()

# Mesh file names based on arguments given
mesh_dims = (args.refinement,)*args.dim
dim_str = 'x'.join(['%i' % i for i in mesh_dims])
mesh_dir = '../meshfiles/'
if args.inverse:
    mesh_file = mesh_dir + 'mesh-inverse-%s.xml.gz' % dim_str
else:
    mesh_file = mesh_dir + 'mesh-%s.xml.gz' % dim_str

# Check if the mesh file exists
if not os.path.isfile(mesh_file):
    raise Exception('The mesh file, \'%s\', does not exist. ' % mesh_file
                    + 'Please run the script \'generate_mesh_files.py\''
                    + 'with the same arguments first.')

# Check if the mesh function file exists
mesh_function = mesh_dir + 'mesh_function-%s.xml.gz' % dim_str
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
nu = 0.49 # Poisson's ratio
la = dlf.Constant(E*nu/((1. + nu)*(1. - 2.*nu))) # 2nd Lame parameter
mu = dlf.Constant(E/(2.*(1. + nu))) # 2nd Lame parameter

# Traction vector
trac = dlf.Constant((5.0, 0.0))

# Region IDs
ALL_ELSE = 0
CLIP = 1
TRACTION = 2


class InitialCondition(dlf.Expression):
    def eval(self, values, x):
        values[0] = 0.1 * x[0]
    def value_shape(self):
        return (2,)


u_init = InitialCondition()

config = {'mechanics' : {
              'const_eqn' : 'lin_elastic',
              'material' : {
                  'type' : 'elastic',
                  'incompressible' : args.incompressible,
                  'density' : 10.0,
                  'lambda' : la,
                  'mu' : mu,
                  }
              },
          'mesh' : {
              'mesh_file' : mesh_file,
              'mesh_function' : mesh_function,
              'element' : 'p2'
              },
          'formulation' : {
              'unsteady' : False,
              'initial_condition' : u_init,
              'domain' : 'lagrangian',
              'inverse' : args.inverse,
              'body_force' : dlf.Constant((0.,)*args.dim),
              'bcs' : {
                  'dirichlet' : {
                      'regions' : [CLIP],
                      'values' : [dlf.Constant((0.,)*args.dim)],
                      'unsteady' : [False]
                      },
                  'neumann' : {
                      'regions' : [TRACTION],
                      'types' : ['traction'],
                      'unsteady' : [False],
                      'values' : [trac]
                      }
                  }
              }
          }

problem = MechanicsProblem(config, form_compiler_parameters=ffc_options)
solver = dlf.NonlinearVariationalSolver(problem)
solver.solve()
soln = problem.solution()

# Save solution before mesh is moved.
if args.dim > 1:
    if args.inverse:
        dlf.File('results/inverse-disp-%s.pvd' % dim_str) << soln
    else:
        dlf.File('results/forward-disp-%s.pvd' % dim_str) << soln

mesh = problem.trial_space().mesh()
mesh_func = dlf.MeshFunction('size_t', mesh, mesh_function)

# Extract the displacement
P1_vec = dlf.VectorElement("CG", mesh.ufl_cell(), 1)
W = dlf.FunctionSpace(mesh, P1_vec)
u_func = dlf.TrialFunction(W)
u_test = dlf.TestFunction(W)
a = dlf.dot(u_test, u_func) * dlf.dx
L = dlf.dot(soln, u_test) * dlf.dx
u_func = dlf.Function(W)
bcs = dlf.DirichletBC(W, dlf.Constant((0.,)*2), mesh_func, CLIP)
dlf.solve(a == L, u_func, bcs)

# Move mesh according to solution and save.
# dlf.ALE.move(mesh, soln)
dlf.ALE.move(mesh, u_func)
dlf.File('../meshfiles/mesh-inverse-%s.xml.gz' % dim_str) << mesh
