import dolfin as dlf

mesh = dlf.UnitSquareMesh(12, 12)
W = dlf.FunctionSpace(mesh, 'CG', 1)

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


subdomains = dlf.MeshFunction('size_t', mesh, 1)
subdomains.set_all(ALL_ELSE)

clip = Clip()
clip.mark(subdomains, CLIP)

traction = Traction()
traction.mark(subdomains, TRACTION)

bc = dlf.DirichletBC(W, expr, subdomains, CLIP)

f = dlf.Constant(1.0)
u = dlf.TrialFunction(W)
xi = dlf.TestFunction(W)

a = dlf.dot(dlf.grad(xi), dlf.grad(u))*dlf.dx

ds_trac = dlf.ds(TRACTION, domain=mesh, subdomain_data=subdomains)
trac = dlf.Expression('1 + x[1]*t', t=1.0, element=W.ufl_element())

L = xi*f*dlf.dx + xi*trac*ds_trac

expr = dlf.Expression('1 + t*pow(x[0],2)', t=1.0, element=W.ufl_element())

A = dlf.PETScMatrix()
b = dlf.PETScVector()

dlf.assemble(a, tensor=A)
dlf.assemble(L, tensor=b)

print 'norm(b) = ', b.norm('l2')
