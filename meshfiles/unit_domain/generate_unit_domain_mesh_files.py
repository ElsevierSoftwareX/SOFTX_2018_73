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
                    default=[12],
                    nargs="+",
                    type=int)
parser.add_argument('-hdf5',
                    help="use HDF5 files",
                    action='store_true')
args = parser.parse_args()
print('[DONE]')

print('Generating mesh.....', end='')
if len(args.refinement) == 1:
    mesh_dims = args.refinement*args.dim
else:
    if len(args.refinement) != args.dim:
        raise ValueError("The number of refinement values must equal the number of dimensions.")
    mesh_dims = args.refinement

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
LEFT = 1
RIGHT = 2
TOP = 3
BOTTOM = 4

print('Generating boundary mesh function.....', end='')
boundaries = dlf.MeshFunction('size_t', mesh, args.dim-1)
boundaries.set_all(ALL_ELSE)


class Left(dlf.SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0]) < dlf.DOLFIN_EPS \
            and on_boundary


class Right(dlf.SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0] - 1.0) < dlf.DOLFIN_EPS \
            and on_boundary


class Top(dlf.SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[1] - 1.0) < dlf.DOLFIN_EPS \
            and on_boundary


class Bottom(dlf.SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[1]) < dlf.DOLFIN_EPS \
            and on_boundary


top = Top()
top.mark(boundaries, TOP)

bottom = Bottom()
bottom.mark(boundaries, BOTTOM)

left = Left()
left.mark(boundaries, LEFT)

right = Right()
right.mark(boundaries, RIGHT)
print('[DONE]')

dim_str   = 'x'.join(['%i' % i for i in mesh_dims])

# Save files
print('Writing files.....', end='')

mesh_name = 'unit_domain-mesh-%s' % dim_str
boundaries_name = 'unit_domain-boundaries-%s' % dim_str

if args.hdf5:
    mesh_name += '.h5'
    boundaries_name += '.h5'

    f1 = dlf.HDF5File(dlf.mpi_comm_world(), mesh_name, 'w')
    f2 = dlf.HDF5File(dlf.mpi_comm_world(), boundaries_name, 'w')

    f1.write(mesh, 'mesh')
    f2.write(boundaries, 'boundaries')

    f1.close()
    f2.close()

else:
    mesh_name += '.xml.gz'
    boundaries_name += '.xml.gz'

    dlf.File(mesh_name) << mesh
    dlf.File(boundaries_name) << boundaries

print('[DONE]')
