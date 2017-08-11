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
        f.read(mesh_func, 'mesh_function')
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

    print ("\n", header)
    print ("-"*string_length)

    for mat_type in dict_implemented['materials']:
        n = len(dict_implemented['materials'][mat_type])
        for i in range(n):
            if i == 0:
                print (string_template.format(mat_type, dict_implemented['materials'][mat_type][i]))
            else:
                print (string_template.format("", dict_implemented['materials'][mat_type][i]))

    return None
