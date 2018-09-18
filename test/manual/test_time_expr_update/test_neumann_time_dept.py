from __future__ import print_function

import numpy as np
import dolfin as dlf
import fenicsmechanics as fm
import fenicsmechanics.mechanicsproblem as mprob

mesh_file, boundaries = fm._get_mesh_file_names("unit_domain", ret_facets=True,
                                                refinements=(2, 2))

# Region IDs
ALL_ELSE = 0
CLIP = 1
TRACTION = 2

# Reduce this to just 2 time-varying expressions
trac_clip = dlf.Expression(['t', 'pow(t, 2)'],
                           t=0.0, degree=2)

press_trac = dlf.Expression('10.0*cos(t)',
                            t=0.0, degree=2)

body_force = dlf.Constant([0.0]*2)

# Elasticity parameters
la = 2.0 # 1st Lame parameter
mu = 0.5 # 2nd Lame parameter

# Problem configuration dictionary
config = {'material' : {
              'type' : 'elastic',
              'const_eqn' : 'lin_elastic',
              'incompressible' : False,
              'density' : 10.0,
              'la' : la,
              'mu' : mu,
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
                  'neumann' : {
                      'regions' : [CLIP, TRACTION],
                      'types' : ['piola', 'pressure'],
                      'values' : [trac_clip, press_trac]
                      }
                  }
              }
          }

problem = mprob.MechanicsProblem(config)

t = 0.0
tf = 1.0
dt = 0.2

zero = np.zeros(2)
trac_vals = np.zeros(2)
pres_val = np.zeros(1)

while t <= tf:

    problem.update_time(t)
    problem.config['formulation']['bcs']['neumann']['values'][0].eval(trac_vals, zero)
    problem.config['formulation']['bcs']['neumann']['values'][1].eval(pres_val, zero)

    exp_trac = np.array([t, t**2])
    exp_pres = 10.0*np.cos(t)

    print('*'*60)
    print('t = %.2f' % t)
    print('\n')
    print('trac_vals = ', trac_vals)
    print('exp_trac  = ', exp_trac)
    print('\n')
    print('pres_val  = ', pres_val)
    print('exp_pres  = ', exp_pres)
    print('\n')

    t += dt
