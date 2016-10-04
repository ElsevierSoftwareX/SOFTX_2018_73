import os
import sys
import argparse
import dolfin as dlf

import fenicsmechanics_dev.mechanicsproblem as mprob
import fenicsmechanics_dev.mechanicssolver as msolv

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
parser.add_argument('-inv', '--inverse',
                    help='activate inverse elasticity',
                    action='store_true')
parser.add_argument('-inc', '--incompressible',
                    help='active incompressible material',
                    action='store_true')
parser.add_argument('-hdf5',
                    help="use HDF5 files",
                    action='store_true')
args = parser.parse_args()

# Mesh file names based on arguments given
mesh_dims = (args.refinement,)*args.dim
dim_str = 'x'.join(['%i' % i for i in mesh_dims])

if args.incompressible:
    name_dims = ('incomp_' + args.material, dim_str)
    element_type = 'p2-p1'
    kappa = dlf.Constant(1e8)
else:
    name_dims = ('comp_' + args.material, dim_str)
    element_type = 'p2'
    kappa = None

mesh_dir = '../meshfiles/'
if args.inverse:
    # mesh_file = mesh_dir + 'unit_domain-defm_mesh-%s-%s' % name_dims
    result_file = dlf.File('results/inverse-disp-%s-%s.pvd' % name_dims)
else:
    # mesh_file = mesh_dir + 'unit_domain-mesh-%s' % dim_str
    result_file = dlf.File('results/forward-disp-%s-%s.pvd' % name_dims)

mesh_file = mesh_dir + 'unit_domain-mesh-%s' % dim_str
mesh_function = mesh_dir + 'unit_domain-mesh_function-%s' % dim_str
if args.hdf5:
    mesh_file += '.h5'
    mesh_function += '.h5'
else:
    mesh_file += '.xml.gz'
    mesh_function += '.xml.gz'

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
trac = dlf.Constant((3.0,) + (0.0,)*(args.dim-1))

# Region IDs
ALL_ELSE = 0
CLIP = 1
TRACTION = 2

# Problem configuration dictionary
config = {'material' : {
              'const_eqn' : args.material,
              'type' : 'elastic',
              'incompressible' : args.incompressible,
              'density' : 10.0,
              'lambda' : la,
              'mu' : mu,
              'kappa' : kappa
              },
          'mesh' : {
              'mesh_file' : mesh_file,
              'mesh_function' : mesh_function,
              'element' : element_type
              },
          'formulation' : {
              'unsteady' : False,
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
                      'types' : ['cauchy'],
                      'unsteady' : [False],
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
if args.dim > 1:
    result_file << problem.displacement

# soln = problem.displacement

# ############################################################
# # Save matrices to file
# import numpy as np
# import scipy.sparse as sp

# # Function to save CSR matrix
# def save_sparse_csr(filename, array):
#     np.savez(filename, data=array.data, indices=array.indices,
#              indptr=array.indptr, shape=array.shape)

# np_stiff_name = 'stiffness-%s-%s.npz' % name_dims
# np_load_name = 'load-%s-%s.npz' % name_dims
# np_soln_name = 'soln-%s-%s.npz' % name_dims

# print 'Writing stiffness matrix to file...'
# A = sp.csr_matrix(problem._stressWorkMatrix.array())
# save_sparse_csr(np_stiff_name, A)
# print '...[DONE]'

# print 'Writing total load vector to file...'
# problem._totalLoadVector.array().tofile(np_load_name)
# print '...[DONE]'

# print 'Writing solution vector to file...'
# problem.displacement.vector().array().tofile(np_soln_name)
# print '...[DONE]'

# ############################################################

# #
# mesh = problem.mesh
# mesh_func = problem.mesh_function

# # Extract the displacement
# P1_vec = dlf.VectorElement("CG", mesh.ufl_cell(), 1)
# W = dlf.FunctionSpace(mesh, P1_vec)
# u_func = dlf.TrialFunction(W)
# u_test = dlf.TestFunction(W)
# a = dlf.dot(u_test, u_func) * dlf.dx
# L = dlf.dot(u_test, soln) * dlf.dx
# u_func = dlf.Function(W)
# bcs = dlf.DirichletBC(W, dlf.Constant((0.,)*args.dim), mesh_func, CLIP)
# dlf.solve(a == L, u_func, bcs)

# # Move mesh according to solution and save.
# dlf.ALE.move(mesh, u_func)

# # if not args.inverse:
# #     defm_mesh = '../meshfiles/unit_domain-defm_mesh-%s-%s' % name_dims
# #     if args.hdf5:
# #         defm_mesh += '.h5'
# #         f = dlf.HDF5File(dlf.mpi_comm_world(), defm_mesh, 'w')
# #         f.write(mesh, 'mesh')
# #         f.close()
# #     else:
# #         defm_mesh += '.xml.gz'
# #         dlf.File(defm_mesh) << mesh

# # Compute the total volume
# total_volume = dlf.assemble(dlf.Constant(1.0)*dlf.dx(domain=mesh))
# print 'Total volume (after deformation): %.8f' % total_volume
