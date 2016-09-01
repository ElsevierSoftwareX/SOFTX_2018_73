import argparse
import dolfin as dlf

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dim",
                    help="dimension",
                    default=2,
                    type=int)
parser.add_argument("-r", "--refinement",
                    help="dimension",
                    default=12,
                    type=int)
args = parser.parse_args()

mesh_dims = (args.refinement,)*args.dim
if args.dim == 1:
    mesh = dlf.UnitIntervalMesh(*mesh_dims)
    # name = 'bar'
elif args.dim == 2:
    mesh = dlf.UnitSquareMesh(*mesh_dims)
    # name = 'plate'
elif args.dim == 3:
    mesh = dlf.UnitCubeMesh(*mesh_dims)
    # name = 'cube'
else:
    raise ValueError('Dimension %i is invalid!' % args.dim)

# Region IDs
ALL_ELSE = 0
CLIP = 1
TRACTION = 2

mesh_function = dlf.MeshFunction('size_t', mesh, args.dim-1)
mesh_function.set_all(ALL_ELSE)


class Clip(dlf.SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0]) < dlf.DOLFIN_EPS \
            and on_boundary


class Traction(dlf.SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0] - 1.0) < dlf.DOLFIN_EPS \
            and on_boundary


clip = Clip()
clip.mark(mesh_function, CLIP)

traction = Traction()
traction.mark(mesh_function, TRACTION)

dim_str   = 'x'.join(['%i' % i for i in mesh_dims])
# name_dims = (name, dim_str)

# Save files
dlf.File('../meshfiles/mesh-%s.xml.gz' % dim_str) << mesh
dlf.File('../meshfiles/mesh_function-%s.xml.gz' % dim_str) << mesh_function
# dlf.File('meshfiles/mesh-%s-%s.xml.gz' % name_dims) << mesh
# dlf.File('meshfiles/mesh_function-%s-%s.xml.gz' % name_dims) << mesh_function
