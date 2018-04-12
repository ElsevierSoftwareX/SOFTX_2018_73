import dolfin as dlf
import fenicsmechanics as fm

mesh = dlf.Mesh()
boundaries = dlf.MeshFunction("size_t", mesh)
fib1 = dlf.MeshFunction("size_t", mesh)
fib2 = dlf.MeshFunction("size_t", mesh)
fib3 = dlf.MeshFunction("size_t", mesh)
she1 = dlf.MeshFunction("size_t", mesh)
she2 = dlf.MeshFunction("size_t", mesh)
she3 = dlf.MeshFunction("size_t", mesh)

fname = "new-ellipsoid_1000um.h5"
fname_new = "ellipsoid-mesh_fibers_boundaries-1000um.h5"

fm.utils._read_write_hdf5("r", fname, close=True,
                          mesh=mesh,
                          mesh_function=boundaries,
                          fib1=fib1, fib2=fib2, fib3=fib3,
                          she1=she1, she2=she2, she3=she3)

fm.utils._read_write_hdf5("w", fname_new, close=True,
                          mesh=mesh,
                          boundaries=boundaries,
                          fib1=fib1, fib2=fib2, fib3=fib3,
                          she1=she1, she2=she2, she3=she3)
