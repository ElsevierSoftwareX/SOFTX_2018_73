import dolfin as dlf

mesh = dlf.UnitSquareMesh(12, 12)
P2 = dlf.VectorElement('CG', mesh.ufl_cell(), 2)
P1 = dlf.FiniteElement('CG', mesh.ufl_cell(), 1)
TH = P2 * P1
W = dlf.FunctionSpace(mesh, TH)

sys_u = dlf.Function(W)
sys_v = dlf.TestFunction(W)
sys_du = dlf.TrialFunction(W)

u, p = dlf.split(sys_u)
v, q = dlf.split(sys_v)
du, dp = dlf.split(sys_du)

G1 = dlf.inner(dlf.grad(u), dlf.grad(v))*dlf.dx - p*dlf.div(v)*dlf.ds
G2 = (p - dlf.div(u))*q*dlf.dx
G = G1 + G2
dG = dlf.derivative(G, sys_u, sys_du)

def boundary(x, on_boundary):
    return on_boundary

bc = dlf.DirichletBC(W, dlf.Constant((0.,)*3), boundary)

problem = dlf.NonlinearVariationalProblem(G, sys_u, bc, J=dG)
