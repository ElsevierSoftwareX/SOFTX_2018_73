import dolfin as dlf

import fenicsmechanics_dev.mechanicsproblem as mprob

mesh_file = '../meshfiles/unit_domain-mesh-2x2.xml.gz'
mesh_function = '../meshfiles/unit_domain-mesh_function-2x2.xml.gz'

# Region IDs
ALL_ELSE = 0
CLIP = 1
TRACTION = 2

# Reduce this to just 2 time-varying expressions
trac_top = dlf.Expression(['t', 'pow(t, 2)'],
                          t=0.0, degree=2)

press_right = dlf.Expression('cos(t)',
                             t=0.0, degree=2)

disp_bot = dlf.Expression(['1.0 + 2.0*t', '3.0*t'],
                          t=0.0, degree=2)

disp_left = dlf.Constant([0.0, 0.0])

body_force = dlf.Expression(['log(1.0+t)', 'exp(t)'],
                            t=0.0, degree=2)

# Elasticity parameters
la = dlf.Constant(2.0) # 1st Lame parameter
mu = dlf.Constant(0.5) # 2nd Lame parameter

# Problem configuration dictionary
config = {'material' : {
              'type' : 'elastic',
              'const_eqn' : 'lin_elastic',
              'incompressible' : False,
              'density' : dlf.Constant(10.0),
              'lambda' : la,
              'mu' : mu,
              'kappa' : None
              },
          'mesh' : {
              'mesh_file' : mesh_file,
              'mesh_function' : mesh_function,
              'element' : 'p2'
              },
          'formulation' : {
              'unsteady' : False,
              'domain' : 'lagrangian',
              'inverse' : False,
              'body_force' : body_force,
              'bcs' : {
                  'dirichlet' : {
                      'regions' : [BOTTOM, LEFT], # MORE REGIONS THAN ACTUALLY DEFINED
                      'values' : [disp_bot, disp_left],
                      'unsteady' : [True, False]
                      },
                  'neumann' : {
                      'regions' : [TOP, RIGHT], # MORE REGIONS THAN ACTUALLY DEFINED
                      'types' : ['piola', 'pressure'],
                      'unsteady' : [True, True],
                      'values' : [trac_top, press_right]
                      }
                  }
              }
          }

problem = mprob.MechanicsProblem(config)

t = 0.0
tf = 1.0
dt = 0.2

while t <= tf:

    # print values to check

    t += dt

    problem.update_time(t)
