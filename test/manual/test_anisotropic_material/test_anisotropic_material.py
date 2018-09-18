import dolfin as dlf

mesh = dlf.UnitSquareMesh(30,30)
pd = 2
W = dlf.VectorFunctionSpace(mesh, "CG", pd)

n1_mag = "sqrt(2.0*(2.0*x[0]*x[0] - 2.0*x[0] + 1.0))"
n1_expr = dlf.Expression(["1.0/%s" % n1_mag, "(1.0 - 2.0*x[0])/%s" % n1_mag],
                         element=W.ufl_element())
n1 = dlf.Function(W)
n1.assign(n1_expr)

n1_file = "fibers/n1.xml.gz"
dlf.File(n1_file) << n1

n2_expr = dlf.Expression(["1.0/%s" % n1_mag, "(2.0*x[0] - 1.0)/%s" % n1_mag],
                         element=W.ufl_element())
n2 = dlf.Function(W)
n2.assign(n2_expr)

n2_file = "fibers/n2.xml.gz"
dlf.File(n2_file) << n2

fiber_files = [n1_file, n2_file, n1, dlf.Constant([1.0,0.0])]
fiber_names = ["n1", "n2", "n3", "n4"]

fiber_dict1 = {'fiber_files': fiber_files,
               'fiber_names': fiber_names,
               'element': 'p%i' % pd}

from fenicsmechanics.materials.solid_materials import AnisotropicMaterial, FungMaterial

mat1 = AnisotropicMaterial(fiber_dict1, mesh)

hdf5_name = "fibers/n_all.h5"
f = dlf.HDF5File(dlf.mpi_comm_world(), hdf5_name, 'w')
f.write(n1, "n1")
f.write(n2, "n2")
f.close()

fiber_dict2 = {'fiber_files': hdf5_name,
               'fiber_names': fiber_names[:2],
               'element': 'p%i' % pd}

mat2 = AnisotropicMaterial(fiber_dict2, mesh)

material_dict = {'kappa': 1e4,
                 'fibers': fiber_dict2}
mat3 = FungMaterial(mesh, inverse=True, incompressible=False, **material_dict)

u = dlf.Function(W)
F = dlf.Identity(2) + dlf.grad(u)
J = dlf.det(F)
P = mat3.stress_tensor(F, J)
