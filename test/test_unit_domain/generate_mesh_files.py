from __future__ import print_function

import argparse
import dolfin as dlf

print('Parsing through command-line arguments.....', end='')
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
print('[DONE]')

print('Generating mesh.....', end='')
mesh_dims = (args.refinement,)*args.dim
if args.dim == 1:
    mesh = dlf.UnitIntervalMesh(*mesh_dims)
elif args.dim == 2:
    mesh = dlf.UnitSquareMesh(*mesh_dims)
elif args.dim == 3:
    mesh = dlf.UnitCubeMesh(*mesh_dims)
else:
    raise ValueError('Dimension %i is invalid!' % args.dim)
print('[DONE]')

# Region IDs
ALL_ELSE = 0
CLIP = 1
TRACTION = 2

print('Generating mesh function.....', end='')
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
print('[DONE]')

dim_str   = 'x'.join(['%i' % i for i in mesh_dims])

# Save files
print('Writing files.....', end='')

mesh_name = '../meshfiles/unit_domain-mesh-%s.xml.gz' % dim_str
mesh_func_name = '../meshfiles/unit_domain-mesh_function-%s.xml.gz' % dim_str

dlf.File(mesh_name) << mesh
dlf.File(mesh_func_name) << mesh_function

print('[DONE]')
