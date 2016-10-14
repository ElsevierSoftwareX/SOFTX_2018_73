#!/usr/bin/env python2

from __future__ import print_function
import sys
import argparse
import dolfin    as df
import utilities as ut
import numpy     as np
import materials as mat

parser = argparse.ArgumentParser()

parser.add_argument("-m", "--material",
                    help="constitutive relation",
                    default='neo-hooke',
                    choices=['linear','neo-hooke','aniso'])
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
parser.add_argument("--pressure",
                    help="target pressure of inflation test in kPA",
                    default=5,
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
                    default=100.,
                    type=float)
parser.add_argument("-ls", "--loading_steps",
                    help="number of loading steps",
                    default=5,
                    type=int)
parser.add_argument("--solver",
                    help="choose solving method",
                    default='mumps',
                    choices=['umfpack','mumps','pastix','hypre_amg','ml_amg','petsc_amg'])
args = parser.parse_args()

# Optimization options for the form compiler
df.parameters['form_compiler']['cpp_optimize'] = True
df.parameters['form_compiler']['representation'] = "quadrature"
df.parameters['form_compiler']['quadrature_degree'] = 3
ffc_options = {'optimize' : True,
               'eliminate_zeros' : True,
               'precompute_basis_const' : True,
               'precompute_ip_const' : True}

rank = df.MPI.rank(df.mpi_comm_world())
if rank !=0:
    df.set_log_level(df.ERROR)

# --- GLOBAL VARIABLES --------------------------------------------------------
output_dir = 'results_jiacheng'

mesh_edge_size = 0.0007
radius         = 0.01 # unit: m
length         = 0.02 # unit: m
Z_high         = 0.02

# Region IDs
HNBC  = 0  # homogeneous Neumann BC
HDBC  = 1  # homogeneous Dirichlet BC
INBC  = 2  # inhomogeneous Neumann BC

# --- FUNCTIONS ---------------------------------------------------------------

def direction_45_135_array(sigma1_array, sigma2_array, T_c_eigenvector_1_array,
                           T_c_eigenvector_2_array):
    # size of sigma1_array, sigma2_array = element_number (1D array)
    gamma_array = np.arctan(sigma2_array/sigma1_array)
    p45_array = np.zeros((element_number,3))
    p135_array = np.zeros((element_number,3))

    p45_array[:,0] = T_c_eigenvector_1_array[:,0]*np.cos(gamma_array) + \
                               T_c_eigenvector_2_array[:,0]*np.sin(gamma_array)
    p45_array[:,1] = T_c_eigenvector_1_array[:,1]*np.cos(gamma_array) + \
                               T_c_eigenvector_2_array[:,1]*np.sin(gamma_array)
    p45_array[:,2] = T_c_eigenvector_1_array[:,2]*np.cos(gamma_array) + \
                               T_c_eigenvector_2_array[:,2]*np.sin(gamma_array)

    p135_array[:,0] = T_c_eigenvector_1_array[:,0]*np.cos(gamma_array) - \
                               T_c_eigenvector_2_array[:,0]*np.sin(gamma_array)
    p135_array[:,1] = T_c_eigenvector_1_array[:,1]*np.cos(gamma_array) - \
                               T_c_eigenvector_2_array[:,1]*np.sin(gamma_array)
    p135_array[:,2] = T_c_eigenvector_1_array[:,2]*np.cos(gamma_array) - \
                               T_c_eigenvector_2_array[:,2]*np.sin(gamma_array)

    return p45_array, p135_array

# T_c_symbolic is the symbolic representation for the stress tensor contribution from collagen
# Vb_vector_DG is the function space to store the stress tensor
# n_function_CG is normal vector function
def generate_fiber_direction_45_135_new(T_c_matrix_temp, Vb_vector_DG):
    element_number = T_c_matrix_temp.size/9
    T_c_matrix = T_c_matrix_temp.reshape((element_number,9))
    T_c_matrix_tensor = np.reshape(T_c_matrix,(element_number,3,3))
    T_c_eigenvalue, T_c_eigenvector = np.linalg.eigh(T_c_matrix_tensor)

    # extract two largest eigenvalues and the corresponding eigenvectors
    T_c_eigenvector_1_array_1D = T_c_eigenvector[:,:,-1].flatten()
    T_c_eigenvector_1_array = T_c_eigenvector[:,:,-1]

    T_c_eigenvector_1 = Function(Vb_vector_DG)
    T_c_eigenvector_2 = Function(Vb_vector_DG)

    T_c_eigenvector_1.vector().set_local(T_c_eigenvector_1_array_1D)

    # obtain right-hand consistent fiber directions
    T_c_eigenvector_2 = project(cross(n_function_DG,T_c_eigenvector_1),Vb_vector_DG)
    T_c_eigenvector_2_array = T_c_eigenvector_2.vector().array().reshape((element_number,3))
    ######################################################
    # compute gamma = +- arctan(sigma2/sigma1)
    sigma1_array = T_c_eigenvalue[:,-1]
    sigma2_array = T_c_eigenvalue[:,-2]



    p45_array, p135_array = direction_45_135_array(sigma1_array, sigma2_array,\
                              T_c_eigenvector_1_array, T_c_eigenvector_2_array)

    direction_45_function = Function(Vb_vector_DG)
    direction_45_function.vector()[:] = p45_array.flatten()

    direction_135_function = Function(Vb_vector_DG)
    direction_135_function.vector()[:] = p135_array.flatten()

    # only need to update the element directions coresponding to the "core"
    # region, which does not include the boundary effect
    # (1) I need to extract the coordinates corresponding to each node or element
    # (2) If the coordinates are in the core region, then update the directions, otherwise not!

    return direction_45_function, direction_135_function, T_c_eigenvector_1, T_c_eigenvector_2


# --- functions to specify boundary conditions --------------------------------
def dirichletbc_b(x, on_boundary):
    tol = mesh_edge_size/5.0
    return (abs(x[2])<tol) or (abs(x[2]-length)<tol)

def dirichletbc_inlet(x, on_boundary):
    tol = mesh_edge_size/5.0
    return abs(x[2]-length)<tol

def dirichletbc_outlet(x, on_boundary):
    tol = mesh_edge_size/5.0
    return abs(x[2])<tol

def dirichletbc_x_movement(x, on_boundary):
    tol = mesh_edge_size/5.0
    return ((abs(x[2])<tol) or (abs(x[2]-length)<tol) and (abs(x[1]) < tol))

def dirichletbc_y_movement(x, on_boundary):
    tol = mesh_edge_size/5.0
    return ((abs(x[2])<tol) or (abs(x[2]-length)<tol) and (abs(x[0]) < tol))

def dirichletbc_x_movement_inlet(x, on_boundary):
    tol = mesh_edge_size/5.0
    return (abs(x[2]-length)<tol and abs(x[1]) < tol)

def dirichletbc_y_movement_inlet(x, on_boundary):
    tol = mesh_edge_size/5.0
    return (abs(x[2]-length)<tol and abs(x[0]) < tol)

class inner_wall_surface(df.SubDomain):
    def inside(self, x, on_boundary):
        tol = mesh_edge_size/5.0
        #tol = DOLFIN_EPS
        #return on_boundary and abs(df.sqrt(x[0]*x[0]+x[1]*x[1])-radius)<tol
        return abs(df.sqrt(x[0]*x[0]+x[1]*x[1])-radius)<tol

class inlet_surface(df.SubDomain):
    def inside(self, x, on_boundary):
        tol = mesh_edge_size/5.0
        #tol = DOLFIN_EPS
        #return on_boundary and abs(df.sqrt(x[0]*x[0]+x[1]*x[1])-radius)<tol
        return abs(x[2]-length)<tol

class outlet_surface(df.SubDomain):
    def inside(self, x, on_boundary):
        tol = mesh_edge_size/5.0
        #tol = DOLFIN_EPS
        #return on_boundary and abs(df.sqrt(x[0]*x[0]+x[1]*x[1])-radius)<tol
        return abs(x[2])<tol

TOL = mesh_edge_size/3.0

class Pinpoint(df.SubDomain):

    def __init__(self, coords):
        self.coords = np.array(coords)
        df.SubDomain.__init__(self)
    def move(self, coords):
        self.coords[:] = np.array(coords)
    def inside(self, x, on_boundary):
        return np.linalg.norm(x-self.coords) < TOL

# -----------------------------------------------------------------------------

meshname = 'short_cylinder_mesh_parallel.hdf5'
hdf = df.HDF5File(df.mpi_comm_world(), meshname, 'r')
mesh = df.Mesh()
hdf.read(mesh, 'mesh', False)
hdf.close()
mesh_dim = mesh.topology().dim()

# subdomain dimension = topological dim - 1
sub_boundaries = df.MeshFunction("size_t", mesh, mesh_dim-1)
sub_boundaries.set_all(HNBC)

inletBC = inlet_surface()
inletBC.mark(sub_boundaries, HDBC)
outletBC = outlet_surface()
outletBC.mark(sub_boundaries, HDBC)
traction = inner_wall_surface()
traction.mark(sub_boundaries, INBC)

ds = df.ds(INBC, domain=mesh, subdomain_data=sub_boundaries)

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
bc1    = df.DirichletBC(Vvec, zeroVec, dirichletbc_inlet)
bc2    = df.DirichletBC(Vvec, zeroVec, dirichletbc_outlet)
#fix inlet/outlet only in z-direction
bc1    = df.DirichletBC(Vvec.sub(2), 0.0, dirichletbc_inlet)
bc2    = df.DirichletBC(Vvec.sub(2), 0.0, dirichletbc_outlet)
bc3_pi = df.DirichletBC(Vvec.sub(1), 0.0, Pinpoint([radius, 0.0, Z_high]), 'pointwise')
bc4_pi = df.DirichletBC(Vvec.sub(1), 0.0, Pinpoint([-1*radius, 0.0, Z_high]), 'pointwise')
bc5_pi = df.DirichletBC(Vvec.sub(0), 0.0, Pinpoint([0.0, radius, Z_high]), 'pointwise')
bc6_pi = df.DirichletBC(Vvec.sub(0), 0.0, Pinpoint([0.0, -1*radius, Z_high]), 'pointwise')
bc3_po = df.DirichletBC(Vvec.sub(1), 0.0, Pinpoint([radius, 0.0, 0.0]), 'pointwise')
bc4_po = df.DirichletBC(Vvec.sub(1), 0.0, Pinpoint([-1*radius, 0.0, 0.0]), 'pointwise')
bc5_po = df.DirichletBC(Vvec.sub(0), 0.0, Pinpoint([0.0, radius, 0.0]), 'pointwise')
bc6_po = df.DirichletBC(Vvec.sub(0), 0.0, Pinpoint([0.0, -1*radius, 0.0]), 'pointwise')

bcs = [bc1, bc2, bc3_pi, bc4_pi, bc5_pi, bc6_pi, bc3_po, bc4_po, bc5_po, bc6_po]

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
if args.poissons_ratio < (0.5 - 1e-12) or args.bulk_modulus < 10e6:
    la     = df.Constant(2.*mu*nu/(1.-2*nu))            #Lame's 1st param lambda
    inv_kappa = df.Constant(1./args.bulk_modulus)
else:
    la        = None
    inv_kappa = df.Constant(0.)

# Jacobian for later use
I     = df.Identity(mesh_dim)
F     = I + df.grad(u)
invF  = df.inv(F)
J     = df.det(F)

mat = mat.NeoHookeMaterial(mu=args.shear_modulus, kappa=args.bulk_modulus,
                       incompressible=args.incompressible)

strain_energy_formulation = False

if strain_energy_formulation:
    #strain energy formulation
    G  = df.derivative(mat.strain_energy(u,p) * df.dx, state, stest)
else:
    #stress formulation
    G = df.inner(df.grad(v), mat.stress_tensor(u, p)) * df.dx

if args.incompressible:
    B  = ut.incompressibilityCondition(u)
    G += B*q*df.dx - inv_kappa*p*q*df.dx

if args.inverse or args.material == "linear":
    G += df.inner(normal, v) * ds
else:
    G += pressure*J*df.inner(df.inv(F.T)*normal, v)*ds

# Overall weak form and its derivative
dG = df.derivative(G, state, df.TrialFunction(M))
problem = df.NonlinearVariationalProblem(G, state, bcs, J=dG,
              form_compiler_parameters=ffc_options)

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
ufile = df.File('%s/cylinder.pvd' % output_dir)
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
if not args.inverse:
  out_filename ="%s/forward-cylinder.h5" % output_dir
  Hdf = df.HDF5File(mesh.mpi_comm(), out_filename, "w")
  Hdf.write(mesh, "mesh")
  Hdf.write(sub_boundaries, "subdomains")
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
