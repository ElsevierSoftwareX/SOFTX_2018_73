import dolfin as dlf
import fenicsmechanics as fm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--save',
                    help='save solution',
                    action='store_true')
args = parser.parse_args()

dlf.parameters['form_compiler']['cpp_optimize'] = True
dlf.parameters['form_compiler']['quadrature_degree'] = 3

# Mesh files
mesh_dir = '../../meshfiles/unit_domain/'
mesh_file = mesh_dir + 'unit_domain-mesh-24x16x16.xml.gz'
boundaries = mesh_dir + 'unit_domain-boundaries-24x16x16.xml.gz'

# Region IDs
ALL_ELSE = 0
CLIP = 1
TWIST = 2

# Dirichlet BCs
twist = dlf.Expression(("scale*0.0",
                        "scale*(y0 + (x[1] - y0)*cos(theta) - (x[2] - z0)*sin(theta) - x[1])",
                        "scale*(z0 + (x[1] - y0)*sin(theta) + (x[2] - z0)*cos(theta) - x[2])"),
                       scale=0.5, y0=0.5, z0=0.5, theta=dlf.pi/3, degree=2)

# Body force
body_force = dlf.Constant([0.0, -0.5, 0.0])
traction = dlf.Constant([0.1, 0.0, 0.0])

config = {'material':
          {
              'const_eqn': 'neo_hookean',
              'type': 'elastic',
              'incompressible': False,
              'density': 1.0,
              'E': 10.0,
              'nu': 0.3
          },
          'mesh':
          {
              'mesh_file': mesh_file,
              'boundaries': boundaries
          },
          'formulation':
          {
              'element': 'p1',
              'domain': 'lagrangian',
              'inverse': False,
              'body_force': body_force,
              'bcs':
              {
                  'dirichlet':
                  {
                      'displacement': [dlf.Constant([0.,0.,0.]),
                                       twist],
                      'regions': [CLIP, TWIST]
                  },
                  'neumann':
                  {
                      'regions': ['everywhere'],
                      'types': ['piola'],
                      'values': [traction]
                  }
              }
          }
}

if args.save:
    result_file = 'results/displacement.pvd'
else:
    result_file = None

problem = fm.SolidMechanicsProblem(config)
solver = fm.SolidMechanicsSolver(problem, fname_disp=result_file)
solver.full_solve()
