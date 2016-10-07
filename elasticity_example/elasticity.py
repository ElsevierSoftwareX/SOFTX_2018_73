#!/usr/bin/env python2

from __future__ import print_function
import sys
import argparse
import dolfin    as df
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
df.parameters['form_compiler']['cpp_optimize'] = True
df.parameters['form_compiler']['quadrature_degree'] = 3
ffc_options = {'optimize' : True,
               'eliminate_zeros' : True,
               'precompute_basis_const' : True,
               'precompute_ip_const' : True}

rank = df.MPI.rank(df.mpi_comm_world())
if rank !=0:
    df.set_log_level(df.ERROR)

# Region IDs
ALL_ELSE = 0
CLIP = 1
TRACTION = 2

class Clip(df.SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0]) < df.DOLFIN_EPS \
            and on_boundary

class Traction(df.SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0] - 1.0) < df.DOLFIN_EPS \
            and on_boundary

mesh_dims = (args.refinement,)*args.dim
dim_str   = 'x'.join(['%i' % i for i in mesh_dims])
name_dims = ('elongation', 'comp_' + args.material, dim_str)

if args.inverse:
    meshname = 'results/%s-%s-forward-%s.h5' % name_dims
    f = df.HDF5File(mpi_comm_world(),meshname,'r')
    mesh = Mesh()
    f.read(mesh, "mesh", False)
    sub_domains = df.MeshFunction("size_t", mesh)
else:
    if args.dim == 1:
        mesh = df.UnitIntervalMesh(*mesh_dims)
    elif args.dim == 2:
        mesh = df.UnitSquareMesh(*mesh_dims)
    else:
        mesh = df.UnitCubeMesh(*mesh_dims)
    sub_domains = df.MeshFunction("size_t", mesh, args.dim-1)
    sub_domains.set_all(ALL_ELSE)
    clip = Clip()
    clip.mark(sub_domains, CLIP)
    traction = Traction()
    traction.mark(sub_domains, TRACTION)

if args.incompressible:
    P2 = df.VectorElement("Lagrange", mesh.ufl_cell(), args.polynomial_degree)
    P1 = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    TH = P2 * P1
    W = df.FunctionSpace(mesh, TH)
    Vvec = W.sub(0)

    # Define mixed functions
    sys_u  = df.Function(W)
    sys_v  = df.TestFunction(W)
    sys_du = df.TrialFunction(W)
    u, p   = df.split(sys_u)
    v, q   = df.split(sys_v)
    du, dp = df.split(sys_du)
else:
    Pvec  = df.VectorElement("Lagrange", mesh.ufl_cell(), args.polynomial_degree)
    Vvec  = df.FunctionSpace(mesh, Pvec)
    # Define mixed functions
    u  = df.Function(Vvec)
    v  = df.TestFunction(Vvec)
    du = df.TrialFunction(Vvec)

Pscal = df.FiniteElement("Lagrange", mesh.ufl_cell(), args.polynomial_degree)
Vscal = df.FunctionSpace(mesh, Pscal)

# Define Dirichlet BC
zeroVec = df.Constant((0.,)*args.dim)
bcs     = df.DirichletBC(Vvec, zeroVec, sub_domains, CLIP)


# Elasticity parameters
E      = 200.0                                       #Young's modulus
nu     = 0.49                                        #Poisson's ratio
mu     = df.Constant(E/(2.*(1. + nu)))                  #1st Lame parameter
inv_la = df.Constant(((1. + nu)*(1. - 2.*nu))/(E*nu))   #reciprocal 2nd Lame

# Jacobian for later use
I = df.Identity(args.dim)
F = I + df.grad(u)
invF = df.inv(F)
J = df.det(F)

# Applied external surface forces
trac     = df.Constant((5.0,) + (0.0,)*(args.dim-1))
pressure = df.Function (Vscal)
ds_right = df.ds(TRACTION, domain=mesh, subdomain_data=sub_domains)

# Stress tensor depending on constitutive equation
stress = ut.computeStressTensorPenalty(u, mu, inv_la, args.material, args.inverse)
if args.inverse or args.material == "linear":
    G = df.inner(df.grad(v), stress) * df.dx - df.dot(trac, v) * ds_right
else:
    G = df.inner(df.grad(v), stress) * df.dx - df.inner(J * invF.T * trac, v) * ds_right

# Overall weak form and its derivative
if args.incompressible:
    G = G1 + G2
    dG = df.derivative(G, sys_u, sys_du)

    df.solve(G == 0, sys_u, bcs, J=dG,
          form_compiler_parameters=ffc_options)
else:
    dG = df.derivative(G, u, du)

    df.solve(G == 0, u, bcs, J=dG,
          form_compiler_parameters=ffc_options)


# Extract the displacement
df.begin("extract displacement")
P1_vec = df.VectorElement("Lagrange", mesh.ufl_cell(), 1)
W = df.FunctionSpace(mesh, P1_vec)
u_func = df.TrialFunction(W)
u_test = df.TestFunction(W)
a = df.dot(u_test, u_func) * df.dx
L = df.dot(u, u_test) * df.dx
u_func = df.Function(W)
df.solve(a == L, u_func)
df.end()

# Extract the Jacobian
df.begin("extract jacobian")
P1 = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
Q = df.FunctionSpace(mesh, P1)
J_func = df.TrialFunction(Q)
J_test = df.TestFunction(Q)
a = J_func * J_test * df.dx
L = J * J_test * df.dx
J_func = df.Function(Q)
df.solve(a == L, J_func)
if rank == 0:
  print('J = \n', J_func.vector().array())
df.end()

# Save displacement, pressure, and Jacobian
df.begin("Print mesh")
if args.dim > 1:
    df.File('results/%s-%s-disp-%s.pvd' % name_dims) << u_func
#df.File('results/%s-%s-disp-%s.xml' % name_dims) << u_func
#df.File('results/%s-%s-pressure-%s.pvd' % name_dims) << p_func
#df.File('results/%s-%s-pressure-%s.xml' % name_dims) << p_func
#df.File('results/%s-%s-jacobian-%s.pvd' % name_dims) << J_func
#df.File('results/%s-%s-jacobian-%s.xml' % name_dims) << J_func

# Move the mesh according to solution
df.ALE.move(mesh,u_func)
if not args.inverse:
  out_filename ="results/%s-%s-forward-%s.h5" % name_dims
  Hdf = df.HDF5File(mesh.mpi_comm(), out_filename, "w")
  Hdf.write(mesh, "mesh")
  Hdf.write(sub_domains, "subdomains")
  Hdf.close()
df.end()

# Compute the volume of each cell
dgSpace = df.FunctionSpace(mesh, 'DG', 0)
dg_v = df.TestFunction(dgSpace)
form_volumes = dg_v * df.dx
volumes = df.assemble(form_volumes)
volume_func = df.Function(dgSpace)
volume_func.vector()[:] = volumes
#File('results/%s-%s-volume-%s.pvd' % name_dims) << volume_func

# Compute the total volume
total_volume = df.assemble(df.Constant(1.0)*df.dx(domain=mesh))
if rank == 0:
    print ('Total volume: %.8f' % total_volume)
