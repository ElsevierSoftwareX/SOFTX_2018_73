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
                    default='linear',
                    choices=['linear','neo-hooke','aniso'])
parser.add_argument("-i","--inverse",
                    help="activate inverse elasticity",
                    action='store_true')
parser.add_argument("-ic","--incompressible",
                    help="block formulation for incompressible materials",
                    action='store_true')
parser.add_argument("-pd", "--polynomial_degree",
                    help="polynomial degree of the ansatz functions",
                    default=1,
                    type=int)
parser.add_argument("-p", "--pressure",
                    help="target pressure of inflation test in kPA",
                    default=5,
                    type=float)
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
name_dims = ('elongation', 'comp_' + args.material, dim_str)

if args.inverse:
    meshname = 'results/%s-%s-forward-%s.h5' % name_dims
    f = HDF5File(mpi_comm_world(),meshname,'r')
    mesh = Mesh()
    f.read(mesh, "mesh", False)
    sub_domains = MeshFunction("size_t", mesh)
else:
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

if args.incompressible:
    P2 = VectorElement("Lagrange", mesh.ufl_cell(), args.polynomial_degree)
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    TH = P2 * P1
    W = FunctionSpace(mesh, TH)
    Vvec = W.sub(0)

    # Define mixed functions
    sys_u = Function(W)
    sys_v = TestFunction(W)
    sys_du = TrialFunction(W)
    u, p = split(sys_u)
    v, q = split(sys_v)
    du, dp = split(sys_du)
else:
    Pvec  = VectorElement("Lagrange", mesh.ufl_cell(), args.polynomial_degree)
    Vvec  = FunctionSpace(mesh, Pvec)
    # Define mixed functions
    u  = Function(Vvec)
    v  = TestFunction(Vvec)
    du = TrialFunction(Vvec)

Pscal = FiniteElement("Lagrange", mesh.ufl_cell(), args.polynomial_degree)
Vscal = FunctionSpace(mesh, Pscal)

# Define Dirichlet BC
zeroVec = Constant((0.,)*args.dim)
bcs = DirichletBC(Vvec, zeroVec, sub_domains, CLIP)


# Elasticity parameters
E      = 200.0                                       #Young's modulus
nu     = 0.49                                        #Poisson's ratio
mu     = Constant(E/(2.*(1. + nu)))                  #1st Lame parameter
inv_la = Constant(((1. + nu)*(1. - 2.*nu))/(E*nu))   #reciprocal 2nd Lame

# Jacobian for later use
I = Identity(args.dim)
F = I + grad(u)
invF = inv(F)
J = det(F)

# Applied external surface forces
trac = Constant((5.0,) + (0.0,)*(args.dim-1))
pressure = Function (Vscal)
ds_right = ds(TRACTION, domain=mesh, subdomain_data=sub_domains)

# Stress tensor depending on constitutive equation
stress = ut.computeStressTensorPenalty(u, mu, inv_la, args.material, args.inverse)
if args.inverse or args.material == "linear":
    G = inner(grad(v), stress) * dx - dot(trac, v) * ds_right
else:
    G = inner(grad(v), stress) * dx - inner(J * invF.T * trac, v) * ds_right

# Overall weak form and its derivative
if args.incompressible:
    G = G1 + G2
    dG = derivative(G, sys_u, sys_du)

    solve(G == 0, sys_u, bcs, J=dG,
          form_compiler_parameters=ffc_options)
else:
    dG = derivative(G, u, du)

    solve(G == 0, u, bcs, J=dG,
          form_compiler_parameters=ffc_options)


# Extract the displacement
begin("extract displacement")
P1_vec = VectorElement("Lagrange", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, P1_vec)
u_func = TrialFunction(W)
u_test = TestFunction(W)
a = dot(u_test, u_func) * dx
L = dot(u, u_test) * dx
u_func = Function(W)
solve(a == L, u_func)
end()

# Extract the Jacobian
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
Q = FunctionSpace(mesh, P1)
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
