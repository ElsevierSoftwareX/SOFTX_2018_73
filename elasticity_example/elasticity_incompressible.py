#!/usr/bin/env python2

from __future__ import print_function
import sys
from dolfin import *
import argparse
import utilities as ut

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dim",
                    help="dimension",
                    default=2,
                    type=int)
parser.add_argument("-r", "--refinement",
                    help="dimension",
                    default=12,
                    type=int)
parser.add_argument("-m", "--material",
                    help="constitutive relation",
                    default='lin-elast',
                    choices=['lin-elast','neo-hooke','aniso'])
parser.add_argument("-i","--inverse",
                    help="activate inverse elasticity",
                    action='store_true')
args = parser.parse_args()

# Optimization options for the form compiler
parameters['form_compiler']['cpp_optimize'] = True
parameters['form_compiler']['quadrature_degree'] = 3
ffc_options = {'optimize' : True,
               'eliminate_zeros' : True,
               'precompute_basis_const' : True,
               'precompute_ip_const' : True}

rank = MPI.rank(mpi_comm_world())

# Region IDs
ALL_ELSE = 0
CLIP = 1
TRACTION = 2

class Clip(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0]) < DOLFIN_EPS \
            and on_boundary

class Traction(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0] - 1.0) < DOLFIN_EPS \
            and on_boundary

mesh_dims = (args.refinement,)*args.dim
dim_str   = 'x'.join(['%i' % i for i in mesh_dims])
name_dims = ('elongation', 'incomp_' + args.material, dim_str)
if not args.inverse:
    if args.dim == 1:
        mesh = UnitIntervalMesh(*mesh_dims)
    elif args.dim == 2:
        mesh = UnitSquareMesh(*mesh_dims)
    else:
        mesh = UnitCubeMesh(*mesh_dims)
    sub_domains = MeshFunction("size_t", mesh, args.dim-1)
    sub_domains.set_all(ALL_ELSE)
    clip = Clip()
    clip.mark(sub_domains, CLIP)
    traction = Traction()
    traction.mark(sub_domains, TRACTION)
else:
    meshname = 'results/%s-%s-forward-%s.h5' % name_dims
    f = HDF5File(mpi_comm_world(),meshname,'r')
    mesh = Mesh()
    f.read(mesh, "mesh", False)
    sub_domains = MeshFunction("size_t", mesh)
    f.read(sub_domains, "subdomains")
    name_dims = ('inverse', 'incomp_' + args.material, dim_str)

P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = P2 * P1
W = FunctionSpace(mesh, TH)

# Define Dirichlet BC
zeroVec = Constant((0.,)*args.dim)
bcs = DirichletBC(W.sub(0), zeroVec, sub_domains, CLIP)

# Define mixed functions
sys_u = Function(W)
sys_v = TestFunction(W)
sys_du = TrialFunction(W)

# Initial guess
class InitialCondition(Expression):
    def eval(self, values, x):
        values[0] = 0.1 * x[0]
    def value_shape(self):
        return (args.dim+1,)

sys_u_init = InitialCondition()
sys_u.interpolate(sys_u_init)

# Obtain pointers to each sub function
u, p = split(sys_u)
v, q = split(sys_v)
du, dp = split(sys_du)

# Elasticity parameters
E      = 20.0                                            #Young's modulus
nu     = 0.5                                #Poisson's ratio
mu     = Constant(E/(2.*(1. + nu)))                  #1st Lame parameter
inv_la = Constant(((1. + nu)*(1. - 2.*nu))/(E*nu))   #reciprocal 2nd Lame

# Jacobian for later use
I = Identity(args.dim)
F = I + grad(u)
invF = inv(F)
J = det(F)

# Applied external surface forces
trac = Constant((5.0,) + (0.0,)*(args.dim-1))
ds_right = ds(TRACTION, domain=mesh, subdomain_data=sub_domains)

# Stress tensor depending on constitutive equation
if args.inverse: #inverse formulation
  if args.material == 'lin-elast':
      # Weak form (momentum)
      stress = ut.inverse_lin_elastic(u, mu, inv_la)
      G1 = inner(grad(v), stress) * dx + div(v)*p*dx\
           - dot(trac, v) * ds_right
      # Weak form (incompressibility)
      G2 = (div(u) - inv_la*p) * q * dx
  else:
      # Weak form (momentum)
      if args.material == 'neo-hooke':
          sigma_isc = ut.incompressible_inverse_neo_hookean(u, mu, True)
      elif args.material == 'aniso':
          sigma_isc = ut.incompressible_inverse_aniso(u, mu, True)
      else:
          sigma_isc = 0
      sigma_vol = ut.inverse_volumetricStress(u, p)
      B     = ut.incompressibilityCondition(u)
      G1 = inner(grad(v), sigma_isc) * dx + inner(grad(v), sigma_vol) * dx\
           - inner(trac, v) * ds_right
      # Weak form (incompressibility)
      G2 = B*q*dx - inv_la*p*q*dx
else: #forward
  if args.material == 'lin-elast':
      # Weak form (momentum)
      stress = ut.forward_lin_elastic(u, mu, inv_la)
      G1 = inner(grad(v), stress) * dx + div(v)*p*dx\
           - dot(trac, v) * ds_right
      # Weak form (incompressibility)
      G2 = (div(u) - inv_la*p) * q * dx
  else:
      if args.material == 'neo-hooke':
      # Weak form (momentum)
        S_isc = ut.incompressible_forward_neo_hookean(u, mu, True)
      elif args.material == 'aniso':
        S_isc = ut.incompressible_forward_aniso(u, mu, True)
      else:
          sigma_isc = 0
      S_vol = ut.volumetricStress(u, p)
      B     = ut.incompressibilityCondition(u)
      G1 = inner(grad(v), S_isc) * dx + inner(grad(v), S_vol) * dx\
           - inner(J * invF.T * trac, v) * ds_right
      # Weak form (incompressibility)
      G2 = B*q*dx - inv_la*p*q*dx

# Overall weak form and its derivative
G = G1 + G2
dG = derivative(G, sys_u, sys_du)

solve(G == 0, sys_u, bcs, J=dG,
          form_compiler_parameters=ffc_options)

# Extract the displacement
begin("extract displacement")
P1_vec = VectorElement("Lagrange", mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, P1_vec)
u_func = TrialFunction(V)
u_test = TestFunction(V)
a = dot(u_test, u_func) * dx
L = dot(u, u_test) * dx
u_func = Function(V)
solve(a == L, u_func)
end()

# Extract the pressure
Q = FunctionSpace(mesh, P1)
p_func = TrialFunction(Q)
p_test = TestFunction(Q)
a = p_func * p_test * dx
L = p * p_test * dx
p_func = Function(Q)
solve(a == L, p_func)

# Extract the Jacobian
J_func = TrialFunction(Q)
J_test = TestFunction(Q)
a = J_func * J_test * dx
L = J * J_test * dx
J_func = Function(Q)
solve(a == L, J_func)
if rank == 0:
  print('J = \n', J_func.vector().array())

# Save displacement, pressure, and Jacobian
if args.dim > 1:
    File('results/%s-%s-disp-%s.pvd' % name_dims) << u_func
#File('results/%s-%s-disp-%s.xml' % name_dims) << u_func
#File('results/%s-%s-pressure-%s.pvd' % name_dims) << p_func
#File('results/%s-%s-pressure-%s.xml' % name_dims) << p_func
#File('results/%s-%s-jacobian-%s.pvd' % name_dims) << J_func
#File('results/%s-%s-jacobian-%s.xml' % name_dims) << J_func

# Move the mesh according to solution
ALE.move(mesh,u_func)
if not args.inverse:
  out_filename ="results/%s-%s-forward-%s.h5" % name_dims
  Hdf = HDF5File(mesh.mpi_comm(), out_filename, "w")
  Hdf.write(mesh, "mesh")
  Hdf.write(sub_domains, "subdomains")
  Hdf.close()

# Compute the volume of each cell
dgSpace = FunctionSpace(mesh, 'DG', 0)
dg_v = TestFunction(dgSpace)
form_volumes = dg_v * dx
volumes = assemble(form_volumes)
volume_func = Function(dgSpace)
volume_func.vector()[:] = volumes
#File('results/%s-%s-volume-%s.pvd' % name_dims) << volume_func

# Compute the total volume
total_volume = assemble(Constant(1.0)*dx(domain=mesh))
if rank == 0:
    print ('Total volume: %.8f' % total_volume)
