import dolfin as dlf

from fenicsmechanics.mechanicsproblem import MechanicsProblem

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
                  'incompressible' : False,
                  'density' : 10.0,
                  'lambda' : la,
                  'mu' : mu,
                  }
              },
          'mesh' : {
              'mesh_file' : 'meshfiles/mesh-plate-12x12.xml.gz',
              'mesh_function' : 'meshfiles/mesh_function-plate-12x12.xml.gz',
              'element' : 'p2'
              },
          'formulation' : {
              'unsteady' : False,
              'initial_condition' : u_init,
              'domain' : 'lagrangian',
              'inverse' : False,
              'body_force' : dlf.Constant((0.,)*2),
              'bcs' : {
                  'dirichlet' : {
                      'regions' : [1],
                      'values' : [dlf.Constant((0.,)*2)],
                      'unsteady' : [False]
                      },
                  'neumann' : {
                      'regions' : [2],
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
dlf.File('results/test_forward_mechanicsproblem.pvd') << soln

mesh = problem.trial_space().mesh()

# Extract the displacement
P1_vec = dlf.VectorElement("CG", mesh.ufl_cell(), 1)
W = dlf.FunctionSpace(mesh, P1_vec)
u_func = dlf.TrialFunction(W)
u_test = dlf.TestFunction(W)
a = dlf.dot(u_test, u_func) * dlf.dx
L = dlf.dot(soln, u_test) * dlf.dx
u_func = dlf.Function(W)
dlf.solve(a == L, u_func)

# Move mesh according to solution and save.
# dlf.ALE.move(mesh, soln)
dlf.ALE.move(mesh, u_func)
dlf.File('meshfiles/mesh-inverse-plate-12x12.xml.gz') << mesh

mesh_function = dlf.MeshFunction('size_t', mesh, config['mesh']['mesh_function'])
dlf.File('meshfiles/mesh_function-inverse-plate-12x12.xml.gz') << mesh_function
dlf.File('meshfiles/mesh_function-inverse-plate-12x12.pvd') << mesh_function
