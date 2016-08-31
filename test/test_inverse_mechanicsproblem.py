import dolfin as dlf

from fenicsmechanics.mechanicsproblem import MechanicsProblem

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
              'mesh_file' : 'meshfiles/mesh-inverse-plate-12x12.xml.gz',
              'mesh_function' : 'meshfiles/mesh_function-inverse-plate-12x12.xml.gz',
              'element' : 'p2'
              },
          'formulation' : {
              'unsteady' : False,
              'initial_condition' : u_init,
              'domain' : 'lagrangian',
              'inverse' : True,
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

problem = MechanicsProblem(config)
solver = dlf.NonlinearVariationalSolver(problem)
solver.solve()
soln = problem.solution()

dlf.File('results/test_inverse_mechanicsproblem.pvd') << soln
