import argparse
import dolfin as dlf

from fenicsmechanics_dev.materials import elastic

# Parse through the arguments provided at the command line.
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dim',
                    help='dimension',
                    default=2,
                    type=int,
                    choices=range(1,4))
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
args = parser.parse_args()

# Mesh file names based on arguments given
mesh_dims = (args.refinement,)*args.dim
dim_str = 'x'.join(['%i' % i for i in mesh_dims])

# Generate mesh
if args.dim == 1:
    mesh = dlf.UnitIntervalMesh(*mesh_dims)
elif args.dim == 2:
    mesh = dlf.UnitSquareMesh(*mesh_dims)
else:
    mesh = dlf.UnitCubeMesh(*mesh_dims)
W = dlf.VectorFunctionSpace(mesh, 'CG', 2)

# Region IDs
ALL_ELSE = 0
CLIP = 1
TRACTION = 2


class Clip(dlf.SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0]) < dlf.DOLFIN_EPS \
            and on_boundary


class Traction(dlf.SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0] - 1.0) < dlf.DOLFIN_EPS \
            and on_boundary


subdomains = dlf.MeshFunction('size_t', mesh, args.dim - 1)
subdomains.set_all(ALL_ELSE)

clip = Clip()
clip.mark(subdomains, CLIP)

traction = Traction()
traction.mark(subdomains, TRACTION)

# Define Dirichlet BCs
zeroVec = dlf.Constant((0.,)*args.dim)
bcs = dlf.DirichletBC(W, zeroVec, subdomains, CLIP)

# Elasticity parameters
E = 20.0 # Young's modulus
nu = 0.49 # Poisson's ratio
la = dlf.Constant(E*nu/((1. + nu)*(1. - 2.*nu))) # 1st Lame parameter
mu = dlf.Constant(E/(2.*(1. + nu))) # 2nd Lame parameter

# Define functions
u = dlf.Function(W)
xi = dlf.TestFunction(W)
du = dlf.TrialFunction(W)

# Tensors
F = dlf.Identity(args.dim) + dlf.grad(u)
if args.inverse:
    stress_function = getattr(elastic, 'inverse_' + args.material)
    result_file_sym = dlf.File('results/sym-inverse-%s-%s.pvd' % (args.material, dim_str))
    result_file_notsym = dlf.File('results/not_sym-inverse-%s-%s.pvd' % (args.material, dim_str))
else:
    stress_function = getattr(elastic, 'forward_' + args.material)
    result_file_sym = dlf.File('results/sym-forward-%s-%s.pvd' % (args.material, dim_str))
    result_file_notsym = dlf.File('results/not_sym-forward-%s-%s.pvd' % (args.material, dim_str))

if args.material == 'lin_elastic':
    stress = stress_function(F, la, mu)
else:
    J = dlf.det(F)
    stress = stress_function(F, J, la, mu)

# Traction on the Neumann boundary region
ds_trac = dlf.ds(TRACTION, domain=mesh, subdomain_data=subdomains)
trac = dlf.Constant((3.0,) + (0.0,)*(args.dim-1))

# Weak form
stress_term = dlf.inner(dlf.grad(xi), stress)*dlf.dx
trac_term = dlf.dot(xi, trac)*ds_trac

############################################################
# Solve using NonlinearVariationalProblem
G = stress_term - trac_term
dG = dlf.derivative(G, u, du)

# Optimization options for the form compiler
dlf.parameters['form_compiler']['cpp_optimize'] = True
dlf.parameters['form_compiler']['quadrature_degree'] = 3
ffc_options = {'optimize' : True,
               'eliminate_zeros' : True,
               'precompute_basis_const' : True,
               'precompute_ip_const' : True}

# Create problem object
problem = dlf.NonlinearVariationalProblem(G, u, bcs, J=dG,
                                          form_compiler_parameters=ffc_options)
solver = dlf.NonlinearVariationalSolver(problem)
solver.solve()

# Save the solution
if args.dim > 1:
    result_file_sym << u

############################################################
# Solve by assembling matrices

u_notsym = dlf.Function(W)

diff_stress_term = dlf.derivative(stress_term, u, du)
A = dlf.assemble(diff_stress_term)
b = dlf.assemble(trac_term)

bcs.apply(A)
bcs.apply(b)

print 'Solving linear algebra problem...'
dlf.solve(A, u_notsym.vector(), b)
print '...[DONE]'

if args.dim > 1:
    result_file_notsym << u_notsym
