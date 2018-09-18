from __future__ import print_function

import numpy as np
import dolfin as dlf
import fenicsmechanics as fm
import fenicsmechanics.mechanicsproblem as mprob

mesh_file, boundaries = fm._get_mesh_file_names("unit_domain", ret_facets=True,
                                                refinements=(2, 2))

body_force = dlf.Expression(['log(1.0+t)', 'exp(t)'],
                            t=0.0, degree=2)

# Elasticity parameters
la = 2.0 # 1st Lame parameter
mu = 0.5 # 2nd Lame parameter

# Problem configuration dictionary
config = {'material' : {
              'type' : 'elastic',
              'const_eqn' : 'lin_elastic',
              'incompressible' : False,
              'density' : dlf.Constant(10.0),
              'la' : la,
              'mu' : mu,
              'kappa' : None
              },
          'mesh' : {
              'mesh_file' : mesh_file,
              'boundaries' : boundaries
              },
          'formulation' : {
              'element' : 'p2',
              'domain' : 'lagrangian',
              'inverse' : False,
              'body_force' : body_force,
              'bcs' : {
                  'dirichlet' : None
                  }
              }
          }

problem = mprob.MechanicsProblem(config)

t = 0.0
tf = 1.0
dt = 0.2

zero = np.zeros(2)
vals = np.zeros(2)

while t <= tf:

    # print values to check

    t += dt

    problem.update_time(t)
    problem.config['formulation']['body_force'].eval(vals, zero)

    expected = np.array([np.log(1.0 + t), np.exp(t)])

    print('*'*60)
    print('t = %.2f' % t)
    print('\n')
    print('vals      = ', vals)
    print('expected  = ', expected)
    print('\n')
