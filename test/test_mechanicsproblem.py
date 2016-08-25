import dolfin as dlf

from fenicsmechanics.mechanicsproblem import MechanicsProblem

# Elasticity parameters
E = 20.0 # Young's modulus
nu = 0.49 # Poisson's ratio
lame1 = dlf.Constant(E*nu/((1. + nu)*(1. - 2.*nu))) # 2nd Lame parameter
lame2 = dlf.Constant(E/(2.*(1. + nu))) # 2nd Lame parameter

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
                  'lame1' : lame1,
                  'lame2' : lame2,
                  }
              },
          'mesh' : {
              'mesh_file' : 'meshfiles/mesh-plate-12x12.xml.gz',
              'mesh_function' : 'meshfiles/mesh_function-plate-12x12.xml.gz',
              'element' : 'p1'
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
                      'values' : {
                          'types' : ['traction'],
                          'function' : [trac],
                          'unsteady' : [False]
                          }
                      }
                  }
              }
          }


problem = MechanicsProblem(config)
