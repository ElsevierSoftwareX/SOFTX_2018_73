from __future__ import print_function

import dolfin as dlf

def load_mesh(mesh_file):
    """
    Load the mesh file specified by the user. The file may be
    xml, or HDF5 (assuming the current dolfin installation
    has the support).


    Parameters
    ----------

    mesh_file : str
        Name of the file that contains the desired mesh information.


    Returns
    -------

    mesh : dolfin.cpp.mesh.Mesh
        This function returns a dolfin mesh object.


    """

    if isinstance(mesh_file, dlf.Mesh):
        return mesh_file

    if mesh_file[-3:] == '.h5':
        mesh = __load_mesh_hdf5(mesh_file)
    else:
        mesh = dlf.Mesh(mesh_file)

    return mesh


def load_mesh_function(mesh_function, mesh):
    """
    Load the mesh function file specified by the user. The file may be
    xml, or HDF5 (assuming the current dolfin installation
    has the support).

    Parameters
    ----------

    mesh_function : str
        Name of the file that contains the desired mesh function information.

    mesh : dolfin.cpp.mesh.Mesh
        The dolfin mesh object that corresponds to the mesh function.


    Returns
    -------

    mesh_func : dolfin.cpp.mesh.MeshFunctionSizet
        This function returns a dolfin mesh function object.


    """

    mesh_func_classes = (dlf.MeshFunctionSizet, dlf.MeshFunctionDouble,
                         dlf.MeshFunctionBool, dlf.MeshFunctionInt)

    if isinstance(mesh_function, mesh_func_classes):
        return mesh_function

    if mesh_function[-3:] == '.h5':
        mesh_func = __load_mesh_function_hdf5(mesh_function, mesh)
    else:
        mesh_func = dlf.MeshFunction('size_t', mesh, mesh_function)

    return mesh_func


def __load_mesh_hdf5(mesh_file):
    """
    Load dolfin mesh from an HDF5 file.

    Parameters
    ----------

    mesh_file : str
        Name of the file containing the mesh information


    Returns
    -------

    mesh : dolfin.cpp.mesh.Mesh
        This function returns a dolfin mesh object.


    """

    # Check dolfin for HDF5 support
    if not dlf.has_hdf5():
        s1 = 'The current installation of dolfin does not support HDF5 files.'
        raise Exception(s1)

    # Check file extension
    if mesh_file[-3:] == '.h5':
        f = dlf.HDF5File(dlf.mpi_comm_world(), mesh_file, 'r')
        mesh = dlf.Mesh()
        f.read(mesh, 'mesh', False)
        f.close()
    else:
        s1 = 'The file extension provided must be \'.h5\'.'
        raise ValueError(s1)

    return mesh


def __load_mesh_function_hdf5(mesh_function, mesh):
    """
    Load a dolfin mesh function from an HDF5 file.

    Parameters
    ----------

    mesh_function : str
        Name of the file containing the mesh function information


    Returns
    -------

    mesh_func : dolfin.cpp.mesh.MeshFunctionSizet
        This function returns a dolfin mesh function object.


    """

    # Check dolfin for HDF5 support
    if not dlf.has_hdf5():
        s1 = 'The current installation of dolfin does not support HDF5 files.'
        raise Exception(s1)

    # Check file extension
    if mesh_function[-3:] == '.h5':
        f = dlf.HDF5File(dlf.mpi_comm_world(), mesh_function, 'r')
        mesh_func = dlf.MeshFunction('size_t', mesh)
        f.read(mesh_func, 'boundaries')
        f.close()
    else:
        s1 = 'The file extension provided must be \'.h5\'.'
        raise ValueError(s1)

    return mesh_func


def petsc_identity(N, dofs=None):
    """
    Create an identity matrix using petsc4py. Note: this currently
    only works in one process.


    """

    from petsc4py import PETSc

    v = PETSc.Vec()
    v.create()
    v.setSizes(N)
    v.setType('standard')
    v.setValues(range(N), [1.0]*N)

    A = PETSc.Mat()
    A.createAIJ([N,N], nnz=N)
    if dofs is not None:
        lgmap = PETSc.LGMap().create(dofs)
        A.setLGMap(lgmap, lgmap)
    A.setDiagonal(v)
    A.assemble()

    return dlf.PETScMatrix(A)


def duplicate_expressions(*args):
    """
    Duplicate dolfin.Expression objects to be used at different time steps.

    Parameters
    ----------

    A set of dolfin.Expression objects to be duplicated.


    Returns
    -------

    retval : list
        List containing duplicated dolfin.Expression objects.


    """

    retval = list()
    import copy

    for arg in args:
        if hasattr(arg, 'cppcode'):
            if hasattr(arg, 't'):
                expr = dlf.Expression(arg.cppcode, t=0.0, element=arg.ufl_element())
            else:
                expr = dlf.Expression(arg.cppcode, element=arg.ufl_element())
        else:
            expr = copy.copy(arg)

        retval.append(expr)

    return retval


def list_implemented_materials():
    """
    Print the material types and corresponding constitutive equations that
    have been implemented in fenicsmechanics.materials.


    """

    from .__CONSTANTS__ import dict_implemented

    string_template = "{:^8} | {:^24}"
    header = string_template.format("Type", "Constitutive Equation")
    string_length = len(header)

    print("\n", header)
    print("-"*string_length)

    for mat_type in dict_implemented['materials']:
        n = len(dict_implemented['materials'][mat_type])
        for i in range(n):
            if i == 0:
                print(string_template.format(mat_type, dict_implemented['materials'][mat_type][i]))
            else:
                print(string_template.format("", dict_implemented['materials'][mat_type][i]))

    return None


def _write_objects(f_objects, t=None, close=False, **kwargs):
    """


    """

    rank = dlf.MPI.rank(dlf.mpi_comm_world())

    if isinstance(f_objects, dlf.HDF5File):
        # Save all objects to same HDF5 file if only one file object is given.
        _read_write_hdf5("a", f_objects, t=t, close=close, **kwargs)
    elif isinstance(f_objects, dlf.XDMFFile):
        # Save all objects to same XDMF file if only one file object is given.
        _write_xdmf(f_objects, kwargs.values(), tspan=t)
    elif isinstance(f_objects, dlf.File):
        # Save through the dolfin.File class. Only one object can be saved.
        if len(kwargs) != 1:
            raise ValueError("A 'dolfin.File' object can only store one object.")

        a = list(kwargs.values())[0]
        if t is not None:
            f_objects << (a, t)
        else:
            f_objects << a
    elif hasattr(f_objects, "__iter__") \
         and hasattr(t, "__iter__"):
        # Save each object to its own separate file and using its own time stamp.

        if len(f_objects) != len(kwargs) != len(t):
            raise ValueError("The length of 'f_objects', " \
                             + "'kwargs', and 't' must be the same.")

        for tval,f,key in zip(t, f_objects, sorted(kwargs.keys())):
            val = kwargs[key]
            if (f is not None) and (val is not None):
                _write_objects(f, t=tval, close=close, **{key: val})
                if rank == 0:
                    print("* '%s' saved *" % key)
    elif hasattr(f_objects, "__iter__"):
        # Save each object to its own separate file using the same time stamp.
        if len(f_objects) != len(kwargs):
            raise ValueError("The length of 'f_objects' and " \
                             + "'kwargs' must be the same.")

        for f,key in zip(f_objects, sorted(kwargs.keys())):
            val = kwargs[key]
            if (f is not None) and (val != 0):
                _write_objects(f, t=t, close=close, **{key: val})
                if rank == 0:
                    print("* '%s' saved *" % key)

    else:
        raise ValueError("'f_objects' must be a file object.")

    return None


def _read_write_hdf5(mode, fname, t=None, close=False, **kwargs):
    """


    """


    if isinstance(fname, str):
        f = dlf.HDF5File(dlf.mpi_comm_world(), fname, mode)
    elif isinstance(fname, dlf.HDF5File):
        f = fname
    else:
        raise ValueError("'fname' provided is not a valid type.")

    if mode == "r":
        func = f.read
    else:
        func = f.write

    # Read/write mesh if in kwargs. This prevents errors when
    # other values in kwargs depend on mesh, e.g. mesh functions.
    try:
        mesh = kwargs.pop("mesh")
        if mode == "r":
            func(mesh, "mesh", False)
        else:
            func(mesh, "mesh")
    except KeyError:
        pass

    for key,val in kwargs.items():
        if (val == 0) or (val is None):
            continue # Nothing to read/write

        if t is not None:
            func(val, key, t)
        else:
            func(val, key)

    if close:
        f.close()

    return None


def _write_xdmf(fname, args, tspan=None):
    """


    """

    if isinstance(fname, str):
        f = dlf.XDMFFile(dlf.mpi_comm_world(), fname)
    elif isinstance(fname, dlf.XDMFFile):
        f = fname
    else:
        raise ValueError("'fname' provided is not a valid type.")

    if tspan is not None:
        if isinstance(tspan, (float, int)):
            len_tspan = 1
        else:
            len_tspan = len(tspan)

        if (len_tspan > 1) and (len_tspan != len(args)):
            raise ValueError("Number of time stamps and objects given to save must be the same.")
        elif len_tspan == 1:
            tspan = [tspan]*len(args)

        for t,a in zip(tspan, args):
            if (a == 0) or (a is None):
                continue # Nothing to write
            f.write(a, t)

    else:
        for a in args:
            f.write(a)

    return None


def _create_file_objects(*fnames):
    """


    """

    splits = map(_splitext, fnames)
    exts = [a[-1] for a in splits]

    dlf_file_objs = [".bin", ".raw", ".svg", ".xd3", ".xml", ".xyz", ".pvd"]

    f_objects = list()
    for name, ext in zip(fnames, exts):
        if ext in dlf_file_objs:
            f = dlf.File(name)
        elif ext == ".h5":
            # Try "append" mode in case file already exists.
            try:
                f = dlf.HDF5File(dlf.mpi_comm_world(), name, "a")
            except RuntimeError:
                f = dlf.HDF5File(dlf.mpi_comm_world(), name, "w")
        elif ext == ".xdmf":
            f = dlf.XDMFFile(dlf.mpi_comm_world(), name)
        elif ext is None:
            f = None
        else:
            raise ValueError("Files with extension '%s' are not supported." % ext)
        f_objects.append(f)

    return f_objects


def _splitext(p):
    """


    """

    from os.path import splitext
    try:
        s = splitext(p)
    except (AttributeError, TypeError): # Added TypeError for python>=3.6
        s = (None, None)

    return s
