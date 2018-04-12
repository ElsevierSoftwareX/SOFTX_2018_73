import mshr
import dolfin as dlf

radius = 0.5
height = 10.0
circ = mshr.Circle(dlf.Point(), radius, 100)
domain = mshr.Extrude2D(circ, height)

mesh = mshr.generate_mesh(domain, 200)
dlf.File('cylinder-mesh.xml.gz') << mesh
dlf.File('cylinder-mesh.pvd') << mesh

# Region IDs
ALL_ELSE = 0
INLET = 1
OUTLET = 2
NOSLIP = 3


class Boundary(dlf.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


class Inlet(dlf.SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[2]) < dlf.DOLFIN_EPS \
            and on_boundary


class Outlet(dlf.SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[2] - height) < dlf.DOLFIN_EPS \
            and on_boundary


boundaries = dlf.MeshFunction('size_t', mesh, 2)
boundaries.set_all(ALL_ELSE)

boundary = Boundary()
boundary.mark(boundaries, NOSLIP)

inlet = Inlet()
inlet.mark(boundaries, INLET)

outlet = Outlet()
outlet.mark(boundaries, OUTLET)

dlf.File('cylinder-mesh_function.xml.gz') << boundaries
dlf.File('cylinder-mesh_function.pvd') << boundaries
