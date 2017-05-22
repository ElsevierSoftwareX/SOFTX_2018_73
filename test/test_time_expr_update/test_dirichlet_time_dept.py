import numpy as np
import dolfin as dlf
import fenicsmechanics.mechanicsproblem as mprob

mesh_file = '../meshfiles/unit_domain/unit_domain-mesh-2x2.xml.gz'
mesh_function = '../meshfiles/unit_domain/unit_domain-mesh_function-2x2.xml.gz'

# Region IDs
ALL_ELSE = 0
CLIP = 1
TRACTION = 2

# Reduce this to just 2 time-varying expressions
disp_clip = dlf.Constant([0.0, 0.0])
disp_trac = dlf.Expression(['1.0 + 2.0*t', '3.0*t'],
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
              'mesh_function' : mesh_function,
              'element' : 'p2'
              },
          'formulation' : {
              'time': {'unsteady' : False},
              'domain' : 'lagrangian',
              'inverse' : False,
              'body_force' : dlf.Constant([0.0]*2),
              'bcs' : {
                  'dirichlet': {
                      'displacement': [disp_clip, disp_trac],
                      'regions': [CLIP, TRACTION] # MORE REGIONS THAN ACTUALLY DEFINED
                      }
                  # 'displacement' : {
                  #     'dirichlet' : {
                  #         'regions' :
                  #         'values' :
                  #         }
                  #     }
                  }
              }
          }

problem = mprob.MechanicsProblem(config)

t = 0.0
tf = 1.0
dt = 0.2

v1 = np.array([1.0, 0.0])
v2 = np.array([2.0, 3.0])

zero = np.zeros(2)
vals = np.zeros(2)

while t <= tf:

    # print values to check

    t += dt

    problem.update_time(t)
    problem.config['formulation']['bcs']['dirichlet']['displacement'][1].eval(vals, zero)

    expected = v1 + t*v2

    print '************************************************************'
    print 't = %.2f' % t
    print 'vals     = ', vals
    print 'expected = ', expected
    print '\n'
