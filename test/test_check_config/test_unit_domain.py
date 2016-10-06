import os
import sys
import argparse
import dolfin as dlf

import fenicsmechanics_dev.mechanicsproblem as mprob
import fenicsmechanics_dev.mechanicssolver as msolv

dim = 2
mesh_file = '../meshfiles/unit_domain-mesh-12x12.xml.gz'
mesh_function = '../meshfiles/unit_domain-mesh_function-12x12.xml.gz'

# Check if the mesh file exists
if not os.path.isfile(mesh_file):
    raise Exception('The mesh file, \'%s\', does not exist. ' % mesh_file
                    + 'Please run the script \'generate_mesh_files.py\''
                    + 'with the same arguments first.')

# Check if the mesh function file exists
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
la = dlf.Constant(E*nu/((1. + nu)*(1. - 2.*nu))) # 1st Lame parameter
mu = dlf.Constant(E/(2.*(1. + nu))) # 2nd Lame parameter

# Traction on the Neumann boundary region
trac = dlf.Constant((3.0,) + (0.0,)*(dim-1))

# Region IDs
ALL_ELSE = 0
CLIP = 1
TRACTION = 2

# Problem configuration dictionary
config = {'material' : {
              'type' : 'elastic',
              'const_eqn' : 'lin_elastic',
              'incompressible' : False,
              'density' : 10.0,
              'lambda' : la,
              'mu' : mu,
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
              'body_force' : dlf.Constant((0.,)*dim),
              'bcs' : {
                  'dirichlet' : {
                      'regions' : [CLIP],
                      'values' : [dlf.Constant((0.,)*dim)],
                      },
                  'neumann' : {
                      'regions' : [TRACTION],
                      'types' : ['cauchy'],
                      'values' : [trac]
                      }
                  }
              }
          }

problem = mprob.MechanicsProblem(config, form_compiler_parameters=ffc_options)

############################################################
my_solver = msolv.MechanicsSolver(problem)
print 'Solving linear algebra problem...'
my_solver.solve()
print '...[DONE]'

# Save solution before mesh is moved.
if dim > 1:
    dlf.File('results/disp.pvd') << problem.displacement
