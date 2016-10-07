#!/usr/bin/env python2

from __future__ import print_function
import sys
import argparse
import dolfin    as df
import utilities as ut
import numpy     as np

parser = argparse.ArgumentParser()

parser.add_argument("-m", "--material",
                    help="constitutive relation",
                    default='lin-elast',
                    choices=['lin-elast','neo-hooke','aniso'])
parser.add_argument("-i","--inverse",
                    help="activate inverse elasticity",
                    action='store_true')
parser.add_argument("-d", "--polynomial_degree",
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
dx = df.dx

node_number = mesh.num_vertices()
element_number = mesh.num_cells()
connectivity_matrix = mesh.cells()


Pvec  = df.VectorElement("Lagrange", mesh.ufl_cell(), args.polynomial_degree)
Pscal = df.FiniteElement("Lagrange", mesh.ufl_cell(), args.polynomial_degree)
Vvec  = df.FunctionSpace(mesh, Pvec)
Vscal = df.FunctionSpace(mesh, Pscal)

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

# Define mixed functions
u  = df.Function(Vvec)
v  = df.TestFunction(Vvec)
du = df.TrialFunction(Vvec)

# define normal directions
normal = df.FacetNormal(mesh)

# Elasticity parameters
E      = 200.0                                            #Young's modulus
nu     = 0.49                                #Poisson's ratio
mu     = df.Constant(E/(2.*(1. + nu)))                  #1st Lame parameter
inv_la = df.Constant(((1. + nu)*(1. - 2.*nu))/(E*nu))   #reciprocal 2nd Lame

# Jacobian for later use
I     = df.Identity(mesh_dim)
F     = I + df.grad(u)
invF  = df.inv(F)
J     = df.det(F)


# Stress tensor depending on constitutive equation
if args.inverse: #inverse formulation
  if args.material == 'lin-elast':
      # Weak form (momentum)
      stress = ut.inverse_lin_elastic(u, mu, inv_la)
      G = df.inner(df.grad(v), stress) * dx - df.dot(n, v) * ds
  else:
      # Weak form (momentum)
      sigma = ut.inverse_neo_hookean(u, mu, inv_la)
      G = df.inner(df.grad(v), sigma) * dx
  G += pressure * df.inner(n, v) * ds
else: #forward
  if args.material == 'lin-elast':
      # Weak form (momentum)
      stress = ut.forward_lin_elastic(u, mu, inv_la)
      G = df.inner(df.grad(v), stress) * dx + df.dot(normal, v) * ds
  else:
      if args.material == 'neo-hooke':
      # Weak form (momentum)
        FS = ut.forward_neo_hookean(u, mu, inv_la)
      elif args.material == 'aniso':
        FS = ut.forward_aniso(u, mu, inv_la)
      else:
          sigma_isc = 0
      # Weak form (momentum)
      FS = ut.forward_neo_hookean(u, mu, inv_la)
      G  = df.inner(df.grad(v), FS) * dx
      G += pressure*J*df.inner(df.inv(F.T)*normal, v)*ds

# Overall weak form and its derivative
dG = df.derivative(G, u, du)

# create file for storing solution
ufile = df.File('%s/cylinder.pvd' % output_dir)
ufile << u # print initial zero solution

P_start = args.pressure*0.2
for P_current in np.linspace(P_start, args.pressure, num=5):
    temp_array = pressure.vector().get_local()
    if rank == 0:
        print(P_current)
    temp_array[:] = P_current
    pressure.vector().set_local(temp_array)

    df.solve(G == 0, u, bcs, J=dG,
              form_compiler_parameters=ffc_options)
    # print solution
    ufile << u


# Extract the displacement
df.begin("extract displacement")
P1_vec = VectorElement("Lagrange", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, P1_vec)
u_func = TrialFunction(W)
u_test = TestFunction(W)
a = df.dot(u_test, u_func) * dx
L = df.dot(u, u_test) * dx
u_func = Function(W)
solve(a == L, u_func)
df.end()

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

# Move the mesh according to solution
ALE.move(mesh,u_func)
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
total_volume = df.assemble(Constant(1.0)*dx(domain=mesh))
if rank == 0:
    print ('Total volume: %.8f' % total_volume)
