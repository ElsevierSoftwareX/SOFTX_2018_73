#!/usr/bin/env python2

from __future__ import print_function
import sys
import argparse
import dolfin    as df
import numpy     as np
from fenicsmechanics import materials

parser = argparse.ArgumentParser()

parser.add_argument("-m", "--material",
                    help="constitutive relation",
                    default='guccione',
                    choices=['linear','neo-hooke',
                             'guccione','fung'])
parser.add_argument("--output_dir",
                    help="output directory",
                    default='problem_2')
parser.add_argument("--meshname",
                    help="mesh file",
                    default='../meshfiles/ellipsoid/ellipsoid_1000um.h5')
parser.add_argument("-i","--inverse",
                    help="activate inverse elasticity",
                    action='store_true')
parser.add_argument("-pd", "--polynomial_degree",
                    help="polynomial degree of the ansatz functions",
                    default=1,
                    type=int)
parser.add_argument("-ic","--incompressible",
                    help="block formulation for incompressible materials",
                    action='store_true')
parser.add_argument("-ai","--anisotropic",
                    help="anisotropic material properties",
                    action='store_true')
parser.add_argument("--pressure",
                    help="target pressure of inflation test in kPA",
                    default=10,
                    type=float)
parser.add_argument("-nu", "--poissons_ratio",
                    help="poissons ratio",
                    default=0.4,
                    type=float)
parser.add_argument("-kappa", "--bulk_modulus",
                    help="bulk modulus",
                    default=1000.0,
                    type=float)
parser.add_argument("-mu", "--shear_modulus",
                    help="shear modulus",
                    default=30.,
                    type=float)
parser.add_argument("-ls", "--loading_steps",
                    help="number of loading steps",
                    default=50,
                    type=int)
parser.add_argument("--solver",
                    help="choose solving method",
                    default='mumps',
                    choices=['umfpack','mumps','pastix','hypre_amg','ml_amg','petsc_amg'])
args = parser.parse_args()

# Optimization options for the form compiler
df.parameters['form_compiler']['cpp_optimize'] = True
df.parameters['form_compiler']['representation'] = "uflacs"
df.parameters['form_compiler']['quadrature_degree'] = 4
df.parameters['form_compiler']['optimize'] = True
#ffc_options = {'optimize' : Truer
               #'eliminate_zeros' : True,
              #'precompute_basis_const' : True,
              # 'precompute_ip_const' : True

rank = df.MPI.rank(df.mpi_comm_world())
if rank !=0:
    df.set_log_level(df.ERROR)

# --- GLOBAL VARIABLES --------------------------------------------------------
output_dir = args.output_dir

# Region IDs
HNBC  = 0  # homogeneous Neumann BC
HDBC  = 10  # homogeneous Dirichlet BC
INBC  = 20  # inhomogeneous Neumann BC

# --- FUNCTIONS ---------------------------------------------------------------
# Code for C++ evaluation of conductivity
fiber_directions_code = """

class FiberDirections : public Expression
{
public:

  // Create expression with 3 components
  FiberDirections() : Expression(3) {}

  // Function for evaluating expression on each cell
  void eval(Array<double>& values, const Array<double>& x, const ufc::cell& cell) const
  {
    const uint D = cell.topological_dimension;
    const uint cell_index = cell.index;
    values[0] = (*f1)[cell_index];
    values[1] = (*f2)[cell_index];
    values[2] = (*f3)[cell_index];
  }

  // The data stored in mesh functions
  std::shared_ptr<MeshFunction<double> > f1;
  std::shared_ptr<MeshFunction<double> > f2;
  std::shared_ptr<MeshFunction<double> > f3;

};
"""

meshname = args.meshname
hdf = df.HDF5File(df.mpi_comm_world(), meshname, 'r')
mesh = df.Mesh()
hdf.read(mesh, 'mesh', False)
boundaries = df.MeshFunction("size_t", mesh)
hdf.read(boundaries, "boundaries")
fib1 = df.MeshFunction("double", mesh)
hdf.read(fib1, "fib1")
fib2 = df.MeshFunction("double", mesh)
hdf.read(fib2, "fib2")
fib3 = df.MeshFunction("double", mesh)
hdf.read(fib3, "fib3")
she1 = df.MeshFunction("double", mesh)
hdf.read(she1, "she1")
she2 = df.MeshFunction("double", mesh)
hdf.read(she2, "she2")
she3 = df.MeshFunction("double", mesh)
hdf.read(she3, "she3")
hdf.close()

cf = df.Expression(cppcode=fiber_directions_code, degree=0)
cs = df.Expression(cppcode=fiber_directions_code, degree=0)
cf.f1 = fib1
cf.f2 = fib2
cf.f3 = fib3
cs.f1 = she1
cs.f2 = she2
cs.f3 = she3

mesh_dim = mesh.topology().dim()

ds = df.ds(INBC, domain=mesh, subdomain_data=boundaries)

node_number = mesh.num_vertices()
element_number = mesh.num_cells()
connectivity_matrix = mesh.cells()

Pvec  = df.VectorElement("Lagrange", mesh.ufl_cell(), args.polynomial_degree)
Pscal = df.FiniteElement("Lagrange", mesh.ufl_cell(), args.polynomial_degree)
Vscal = df.FunctionSpace(mesh, Pscal)

if args.incompressible:
    p_degree = max(args.polynomial_degree-1, 1)
    Qelem = df.FiniteElement("Lagrange", mesh.ufl_cell(), p_degree )
    TH = Pvec * Qelem
    M = df.FunctionSpace(mesh, TH)
    Vvec = M.sub(0)
else:
    Vvec  = df.FunctionSpace(mesh, Pvec)
    M = Vvec

state = df.Function(M)
u = df.split(state)[0] if args.incompressible else state
p = df.split(state)[1] if args.incompressible else None

stest = df.TestFunction(M)
v = df.split(stest)[0] if args.incompressible else stest
q = df.split(stest)[1] if args.incompressible else None

# Define Dirichlet BC
zeroVec = df.Constant((0.,)*mesh_dim)
#fix inlet/outlet in all directions
bcs = df.DirichletBC(Vvec,zeroVec,boundaries,HDBC)

# Applied external surface forces
pressure = df.Function (Vscal)


# define normal directions
normal = df.FacetNormal(mesh)

# Elasticity parameters
nu     = df.Constant(args.poissons_ratio)           #Poisson's ratio nu
mu     = df.Constant(args.shear_modulus)            #shear modulus mu
inv_la = df.Constant((1.-2.*nu)/(2.*mu*nu))         #reciprocal 2nd Lame
E      = df.Constant(2.*mu*(1.+nu))                 #Young's modulus E
kappa  = df.Constant(args.bulk_modulus)  #bulk modulus kappa
if (args.bulk_modulus < 10e6 and args.bulk_modulus > df.DOLFIN_EPS):
    inv_kappa = df.Constant(1./args.bulk_modulus)
else:
    la        = None
    inv_kappa = df.Constant(0.)

# Jacobian for later use
I     = df.Identity(mesh_dim)
F     = I + df.grad(u)
invF  = df.inv(F)
J     = df.det(F)

if (args.material == "linear"):
    mat = materials.solid_materials.LinearIsoMaterial
if (args.material == "neo-hooke"):
    mat = materials.solid_materials.NeoHookeMaterial(mu=args.shear_modulus,
                   kappa=args.bulk_modulus, incompressible=args.incompressible,
                   inverse=args.inverse)
elif (args.material == "guccione"):
    mat = materials.solid_materials.GuccioneMaterial
    params = mat.default_parameters()
    params['incompressible'] = args.incompressible
    if args.anisotropic:
        params['C']     = 2.0
        params['bf']    = 8.0
        params['bt']    = 2.0
        params['bfs']   = 4.0
    else:
        params['C']     = 10.0
        params['bf']    = 1.0
        params['bt']    = 1.0
        params['bfs']   = 1.0
    params['kappa'] = args.bulk_modulus
    params['fibers'] = {'fiber_files': [cf, cs],
                        'fiber_names': ['e1', 'e2'],
                        'element': None}
    mat = mat(mesh, inverse=args.inverse, **params)
else: # fung
    mat = materials.solid_materials.FungMaterial
    params = mat.default_parameters()
    params['incompressible'] = args.incompressible
    params['inverse'] = args.inverse
    params['C'] = 10.0
    params['d'] = [1.0]*3 + [0.0]*3 + [2.0]*3
    params['kappa'] = args.bulk_modulus
    params['fibers'] = {'fiber_files': [cf, cs],
                        'fiber_names': ['e1', 'e2'],
                        'element': None}
    mat = mat(mesh, inverse=args.inverse, **params)

mat.set_active(False)
mat.print_info()

strain_energy_formulation = False

if strain_energy_formulation:
    #strain energy formulation
    G  = df.derivative(mat.strain_energy(u,p) * df.dx, state, stest)
else:
    #stress formulation
    G = df.inner(df.grad(v), mat.stress_tensor(F, J, p)) * df.dx

if args.incompressible:
    B  = mat.incompressibilityCondition(u)
    G += B*q*df.dx + inv_kappa*p*q*df.dx

if args.inverse or args.material == "linear":
    G += pressure*df.inner(normal, v) * ds
else:
    G += pressure*J*df.inner(df.inv(F.T)*normal, v)*ds

# Overall weak form and its derivative
dG = df.derivative(G, state, df.TrialFunction(M))
problem = df.NonlinearVariationalProblem(G, state, bcs, J=dG)

#solver
solver = df.NonlinearVariationalSolver(problem)
prm = solver.parameters
prm['newton_solver']['absolute_tolerance']   = 1E-8
prm['newton_solver']['relative_tolerance']   = 1E-7
prm['newton_solver']['maximum_iterations']   = 25
prm['newton_solver']['relaxation_parameter'] = 1.0
if args.solver == 'umfpack':
    prm['newton_solver']['linear_solver'] = 'umfpack'
if args.solver == 'pastix':
    prm['newton_solver']['linear_solver'] = 'pastix'
if args.solver == 'mumps':
    prm['newton_solver']['linear_solver'] = 'mumps'
if args.solver == 'ml_amg':
    ML_param = {"max levels"           : 3,
                "output"               : 10,
                "smoother: type"       : "ML symmetric Gauss-Seidel",
                "aggregation: type"    : "Uncoupled",
                "ML validate parameter list" : False
    }
    prm['newton_solver']['linear_solver']  =  'gmres'
    prm['newton_solver']['preconditioner'] = 'ml_amg'
if args.solver == 'hypre_amg':
    prm['newton_solver']['linear_solver']  = 'gmres'
    prm['newton_solver']['preconditioner'] = 'hypre_amg'
if args.solver == 'petsc_amg':
    prm['newton_solver']['linear_solver']  = 'gmres'
    prm['newton_solver']['preconditioner'] = 'petsc_amg'

# create file for storing solution
ufile = df.File('%s/ellipsoid.pvd' % output_dir)
if args.incompressible:
    u, p = state.split()
ufile << u # print initial zero solution

scaling = 1./args.loading_steps

P_start = args.pressure*scaling
for P_current in np.linspace(P_start, args.pressure, num=args.loading_steps):
    temp_array = pressure.vector().get_local()
    if rank == 0:
        print(P_current)
    temp_array[:] = P_current
    pressure.vector().set_local(temp_array)
    solver.solve()

    # Overall weak form and its derivative
    if args.incompressible:
        u, p = state.split()

    # print solution
    ufile << u


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

# Move the mesh according to solution
df.ALE.move(mesh,u_func)
#move fiber directions
F     = I + df.grad(u)
if not args.inverse:
  out_filename ="%s/forward-ellipsoid.h5" % output_dir
  Hdf = df.HDF5File(mesh.mpi_comm(), out_filename, "w")
  Hdf.write(mesh, "mesh")
  Hdf.write(boundaries, "boundaries")
  Hdf.write(fib1, "fib1")
  Hdf.write(fib2, "fib2")
  Hdf.write(fib3, "fib3")
  Hdf.write(she1, "she1")
  Hdf.write(she2, "she2")
  Hdf.write(she3, "she3")
  Hdf.close()

# Compute the volume of each cell
dgSpace = df.FunctionSpace(mesh, 'DG', 0)
dg_v = df.TestFunction(dgSpace)
form_volumes = dg_v * df.dx
volumes = df.assemble(form_volumes)
volume_func = df.Function(dgSpace)
volume_func.vector()[:] = volumes

# Compute the total volume
total_volume = df.assemble(df.Constant(1.0)*df.dx(domain=mesh))
if rank == 0:
    print ('Total volume: %.8f' % total_volume)
