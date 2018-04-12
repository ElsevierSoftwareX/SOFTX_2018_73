import dolfin as dlf
import fenicsmechanics as fm

f = "new-ellipsoid_1000um.h5"

mesh = dlf.Mesh()
boundaries = dlf.MeshFunction("size_t", mesh)

# fib1 = dlf.MeshFunction("double", mesh)
# fib2 = dlf.MeshFunction("double", mesh)
# fib3 = dlf.MeshFunction("double", mesh)
# she1 = dlf.MeshFunction("double", mesh)
# she2 = dlf.MeshFunction("double", mesh)
# she3 = dlf.MeshFunction("double", mesh)

kwargs = dict(mesh=mesh, boundaries=boundaries)#,
              # fib1=fib1, fib2=fib2, fib3=fib3,
              # she1=she1, she2=she2, she3=she3)
fm.utils._read_write_hdf5("r", f, t=None, close=True, **kwargs)

fib = fm.materials.solid_materials.define_fiber_dir(f, ["fib1","fib2","fib3"], mesh)
she = fm.materials.solid_materials.define_fiber_dir(f, ["she1","she2","she3"], mesh)

V = dlf.VectorFunctionSpace(mesh, "DG", 0)
fib = dlf.Function(V, "fibers/n1-p0.xml.gz")
