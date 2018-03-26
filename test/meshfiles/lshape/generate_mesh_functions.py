import dolfin as dlf

mesh = dlf.Mesh('lshape-mesh.xml.gz')

# Region IDs
ALL_ELSE = 0
INFLOW = 1
OUTFLOW = 2
NOSLIP = 3

boundaries = dlf.MeshFunction('size_t', mesh, mesh.geometry().dim() - 1)
boundaries.set_all(ALL_ELSE)


class Inflow(dlf.SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[1] - 1.0) < dlf.DOLFIN_EPS \
            and on_boundary


class Outflow(dlf.SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0] - 1.0) < dlf.DOLFIN_EPS \
            and on_boundary


class NoSlip(dlf.SubDomain):
    def inside(self, x, on_boundary):
        left = x[0] < dlf.DOLFIN_EPS
        bottom = x[1] < dlf.DOLFIN_EPS
        right = x[0] > 0.5 - dlf.DOLFIN_EPS
        top = x[1] > 0.5 - dlf.DOLFIN_EPS
        return (left or bottom or (right and top)) \
            and on_boundary


inflow = Inflow()
inflow.mark(boundaries, INFLOW)

outflow = Outflow()
outflow.mark(boundaries, OUTFLOW)

noslip = NoSlip()
noslip.mark(boundaries, NOSLIP)

# Save file
dlf.File("lshape-boundaries.xml.gz") << boundaries
