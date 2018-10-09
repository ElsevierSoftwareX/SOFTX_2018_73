import numpy as np

import ufl
import dolfin as dlf

from ..exceptions import *
from ..utils import _splitext
from ..fmdecorators import hdf5_file
from ..dolfincompat import MPI_COMM_WORLD


def define_fiber_direction(fiber_file, fiber_name, mesh,
                           pd=None, elementwise=False):
    """
    Define a single fiber direction by loading (and converting if necessary)
    the corresponding vector field.

    Parameters
    ----------

    fiber_file : str, ufl.Coefficient, list/tuple
        The name of the file containing the vector field function, a ufl.Coefficient
        object approximating the vector field, or a list of file names where the
        separate components of the vector field are stored.
    fiber_name : str, list/tuple
        The name of the fiber vector field, or a list of the component names.
        These names are used to read HDF5 files.
    mesh : dolfin.Mesh
        The computational mesh over which the fiber directions need to be
        defined. This is needed to either create corresponding mesh functions,
        or the necessary function space(s) to read the vector field or its
        components.
    pd : int (default None)
        The polynomial degree used to approximate the vector field describing
        the fiber directions. This should be kept as None if 'elementwise' is
        set to True.
    elementwise : bool (default False)
        Set to True if the vector field is constant in each cell. Furthermore,
        setting this to True assumes that the data is stored as a set of mesh
        functions. These mesh functions are then converted to 'p0' scalar-valued
        functions, and then assigned to the components of a vector-valued
        function.


    Returns
    -------

    fiber_direction : ufl.Coefficient


    """
    geo_dim = mesh.geometry().dim()

    # If a ufl.Coefficient object is given, exit the function.
    if isinstance(fiber_file, ufl.Coefficient):

        # Simplest case. Rename and return the same object. Check the ufl_shape.
        if not isinstance(fiber_name, str):
            raise InvalidCombination()
        fiber_direction = fiber_file
        fiber_direction.rename(fiber_name, "Fiber direction")
        return fiber_direction

    # Make sure that the vector field is specified as either constant over a
    # a cell, or that a polynomial degree to approximate it is given.
    if (pd is None) and (not elementwise):
        msg = "User must either specify the finite element used to approximate" \
              + " the vector field, or set 'elementwise' to True to specify" \
              + " that the vector field is constant over each cell."
        raise InvalidCombination(msg)

    load_meshfunctions = False
    if elementwise:
        load_meshfunctions = True
        pd = 0
    else:
        if not isinstance(pd, int):
            msg = "The polynomial degree used to approximate the vector field" \
                  + " must be specified if 'elementwise' is not set to True.."
            raise TypeError(msg)
    family = "DG" if pd == 0 else "CG"

    # Check to see if the number of files matches the geometric dimension.
    # Also check if the same file was specified multiple times, and collapse
    # into one if that's the case.
    if isinstance(fiber_file, (list, tuple)):
        if len(fiber_file) != geo_dim:
            msg = "The number of fiber files specified must be the same" \
                  + " as the geometric dimension if a list/tuple is provided."
            raise InconsistentCombination(msg)
        # Extract the first entry if they are all the same.
        if len(set(fiber_file)) == 1:
            fiber_file = fiber_file[0]

    if isinstance(fiber_name, (list, tuple)):
        if len(fiber_name) != geo_dim:
            msg = "The number of fiber names specified must be the same" \
                  + " as the geometric dimension if a list/tuple is provided."
            raise InconsistentCombination(msg)

    if isinstance(fiber_file, str):

        # Need to open file and check what is given under "fiber_names". The
        # "fiber_names" type will determine if a vector field is provided, or
        # its components are given.
        if isinstance(fiber_name, str):

            # Load a vector field from the file.
            fiber_direction = load_vector_field(fiber_file, fiber_name,
                                                mesh, pd=pd)

        elif isinstance(fiber_name, (list, tuple)):

            # Load each component of the vector field. Need to make sure all
            # entries given are strings. Also need to check number of components
            # given against the geometric dimension.
            if load_meshfunctions:
                components = load_meshfunction_fibers(fiber_file, fiber_name, mesh)
            else:
                functionspace = dlf.FunctionSpace(mesh, family, pd)
                components = load_scalar_function_fibers(fiber_file, fiber_name,
                                                         mesh, functionspace=functionspace)

            vfs = dlf.VectorFunctionSpace(mesh, family, pd)
            fiber_direction = convert_scalarcomponents_to_vectorfunction(components,
                                                                         functionspace=vfs)

        else:
            msg = "The type given for 'fiber_name' is not valid. It must either" \
                  + " be a string or a list/tuple."
            raise TypeError(msg)

    elif isinstance(fiber_file, (list, tuple)):

        if not isinstance(fiber_name, (list, tuple)):
            msg = "A list of fiber names must be given when a list of file" \
                  + " names is given."
            raise InvalidCombination(msg)

        components = list()
        for i, (fname, name) in enumerate(zip(fiber_file, fiber_name)):
            if load_meshfunctions:
                component = load_meshfunction_fibers(fname, [name], mesh)
            else:
                functionspace = dlf.FunctionSpace(mesh, family, pd)
                component = load_scalar_function_fibers(fname, [name], mesh,
                                                        functionspace=functionspace)
            components.append(component)
        vfs = dlf.VectorFunctionSpace(mesh, family, pd)
        fiber_direction = convert_scalarcomponents_to_vectorfunction(components,)
    else:
        msg = "The type given under 'fiber_file' is not valid. It must either" \
              + " be a string or a list/tuple."
        raise TypeError(msg)

    return fiber_direction


################################################################################
# Load fiber components as either mesh functions, or scalar functions in the
# 'p0' function space.
@hdf5_file(mode="r")
def load_meshfunction_fibers_hdf5(hdf, component_names, mesh):
    """
    Load cell functions specifying the components of a fiber vector field from
    an HDF5 file.

    Parameters
    ----------

    hdf : str, dolfin.HDF5File
        The name of the HDF5 file where the cell functions are stored, or the
        handle to the file.
    component_names : list, tuple
        The names under which the mesh functions are stored in the HDF5 file.
    mesh : dolfin.Mesh
        The computational mesh over which the fiber directions need to be
        defined. This is needed to create the corresponding mesh functions.

    Returns
    -------

    component_meshfunctions : list, dolfin.cpp.mesh.MeshFunctionDouble
        A list of mesh functions loaded from the file given. The mesh function
        is returned directly if there is only one.

    """
    component_meshfunctions = list()
    for i, component in enumerate(component_names):
        mf = dlf.MeshFunction("double", mesh, mesh.geometry().dim())
        hdf.read(mf, component)
        component_meshfunctions.append(mf)
    if len(component_meshfunctions) == 1:
        component_meshfunctions, = component_meshfunctions
    return component_meshfunctions


def load_meshfunction_fibers(fname, component_names, mesh):
    """
    Load cell functions specifying the components of a fiber vector field from
    an HDF5 file.

    Parameters
    ----------

    fname : str, dolfin.HDF5File
        The name of the file where the cell functions are stored, or the
        handle to the an HDF5 file.
    component_names : list, tuple
        The names under which the mesh functions are stored in the HDF5 file.
    mesh : dolfin.Mesh
        The computational mesh over which the fiber directions need to be
        defined. This is needed to create the corresponding mesh functions.

    Returns
    -------

    component_meshfunctions : list, dolfin.cpp.mesh.MeshFunctionDouble
        A list of mesh functions loaded from the file given. The mesh function
        is returned directly if there is only one.

    """
    _, ext = _splitext(fname)
    if ext == ".h5":
        component_meshfunctions = load_meshfunction_fibers_hdf5(fname,
                                                                component_names,
                                                                mesh)
    else:
        if len(component_names) != 1:
            msg ="Only one mesh function can be stored in files that are not" \
                + " HDF5 files."
            raise InconsistentCombination(msg)
        name = component_names[0]
        component_meshfunctions = dlf.MeshFunction("double", mesh, fname)
    return component_meshfunctions


@hdf5_file(mode="r")
def load_scalar_function_fibers_hdf5(hdf, component_names, mesh,
                                     functionspace=None):
    """
    Load scalar functions specifying the components of a fiber vector field from
    an HDF5 file.

    Parameters
    ----------

    hdf : str, dolfin.HDF5File
        The name of the file where the cell functions are stored, or the
        handle to the an HDF5 file.
    component_names : list, tuple
        The names under which the scalar functions are stored in the HDF5 file.
    mesh : dolfin.Mesh
        The computational mesh over which the fiber directions need to be
        defined. This is needed to create the corresponding mesh functions.
    functionspace : dolfin.FunctionSpace (default None)
        The function space used to create the function. If none is specified,
        a 'p0' element is assumed.

    Returns
    -------

    component_functions : list, dolfin.Function
        A list of scalar functions loaded from the file given. The function is
        returned directly if there is only one.

    """
    if functionspace is None:
        functionspace = dlf.FunctionSpace(mesh, "DG", 0)
    component_functions = list()
    for i, name in enumerate(component_names):
        func = dlf.Function(functionspace)
        hdf.read(func, name)
        component_functions.append(func)
    if len(component_functions) == 1:
        component_functions, = component_functions
    return component_functions


def load_scalar_function_fibers(fname, component_names, mesh,
                                functionspace=None):
    """
    Load scalar functions specifying the components of a fiber vector field from
    a file.

    Parameters
    ----------

    fname : str, dolfin.HDF5File
        The name of the file where the cell functions are stored, or the
        handle to the an HDF5 file.
    component_names : list, tuple
        The names under which the scalar functions are stored in the HDF5 file.
    mesh : dolfin.Mesh
        The computational mesh over which the fiber directions need to be
        defined. This is needed to create the corresponding mesh functions.
    functionspace : dolfin.FunctionSpace (default None)
        The function space used to create the function. If none is specified,
        a 'p0' element is assumed.

    Returns
    -------

    component_functions : list, dolfin.Function
        A list of scalar functions loaded from the file given. The function is
        returned directly if there is only one.

    """
    _, ext = _splitext(fname)
    if ext == ".h5":
        component_functions = load_scalar_function_fibers_hdf5(fname, component_names, mesh,
                                                               functionspace=functionspace)
    else:
        if len(component_names) != 1:
            msg ="Only one mesh function can be stored in files that are not" \
                + " HDF5 files."
            raise InconsistentCombination(msg)
        name = component_names[0]
        component_functions = dlf.Function(functionspace, fname)
    return component_functions


@hdf5_file(mode="r")
def load_vector_field_hdf5(hdf, name, mesh, pd):
    """
    Load a vector-valued function from an HDF5 file.


    Parameters
    ----------

    hdf : str, dolfin.HDF5File
        The name of the file where the cell functions are stored, or the
        handle to the an HDF5 file.
    name : str
        The name under which the vector-valued function is stored in the HDF5 file.
    mesh : dolfin.Mesh
        The computational mesh over which the fiber directions need to be
        defined. This is needed to create the corresponding mesh functions.
    pd : int
        The polynomial degree used to approximate the vector field describing
        the fiber directions.


    Returns
    -------

    vector_field : dolfin.Function
        The vector-valued dolfin.Function object describing the fiber directions.


    """
    family = "DG" if pd == 0 else "CG"
    V = dlf.VectorFunctionSpace(mesh, family, pd)
    vector_field = dlf.Function(V)
    hdf.read(vector_field, name)
    return vector_field


def load_vector_field(fname, name, mesh, pd):
    """
    Load a vector-valued function from an HDF5 file.


    Parameters
    ----------

    hdf : str
        The name of the file where the cell functions are stored.
    name : str
        The name of the vector field. This is used to extract it from an HDF5
        file when appropriate.
    mesh : dolfin.Mesh
        The computational mesh over which the fiber directions need to be
        defined. This is needed to create the corresponding mesh functions.
    pd : int
        The polynomial degree used to approximate the vector field describing
        the fiber directions.


    Returns
    -------

    vector_field : dolfin.Function
        The vector-valued dolfin.Function object describing the fiber directions.


    """
    _, ext = _splitext(fname)
    if ext == ".h5":
        vector_field = load_vector_field_hdf5(fname, name, mesh, pd)
    else:
        family = "DG" if pd == 0 else "CG"
        V = dlf.VectorFunctionSpace(mesh, family, pd)
        vector_field = dlf.Function(V, fname)
    return vector_field


################################################################################
# Convert scalar components to a vector function.
def convert_meshfunction_to_function(meshfunction, functionspace=None):
    """
    Convert a meshfunction to a function with 'p0' elements.

    Parameters
    ----------

    meshfunction : dolfin.cpp.mesh.MeshFunctionDouble
        The mesh function specifying a field that is constant over each cell.
    functionspace : dolfin.FunctionSpace (default None)
        The function space to be used for creating the dolfin.Function object.
        A 'p0' function space is used if None is specified.


    Returns
    -------

    func : dolfin.Function
        The scalar-valued function storing the values of the mesh function.

    """
    mesh = meshfunction.mesh()
    if functionspace is None:
        functionspace = dlf.FunctionSpace(mesh, "DG", 0)
    func = dlf.Function(functionspace)
    dofmap = functionspace.dofmap()
    new_values = np.zeros(func.vector().local_size())
    for cell in dlf.cells(mesh):
        new_values[dofmap.cell_dofs(cell.index())] = meshfunction[cell]
    func.vector().set_local(new_values)
    return func


def assign_scalars_to_vectorfunctions(components, vector_func):
    """
    Assign the values of scalar functions to the components of a vector field.

    Parameters
    ----------

    components : list, tuple
        A list/tuple of scalar-valued dolfin.Function objects used to create a
        vector-valued function.
    vector_func : dolfin.Function
        The vector-valued dolfin.Function object that the scalar components
        are to be assigned to.

    """
    for i, component in enumerate(components):
        dlf.assign(vector_func.sub(i), component)
    return None


def convert_meshfunctions_to_vectorfunction(meshfunctions, functionspace=None):
    """
    Convert a list of mesh functions to a vector-valued dolfin.Function object.

    Parameters
    ----------

    meshfunctions : list, tuple
        The list/tuple of mesh function objects used to create a vector-valued
        function. These are first converted to scalar-valued functions, and then
        they are assigned to the components of the vector-valued function.
    functionspace : dolfin.FunctionSpace (default None)
        The function space used to create the dolfin.Function object. A new
        function space with element 'p0' is created if None is provided.


    Returns
    -------

    vector_func : dolfin.Function
        The vector-valued dolfin.Function object created to describe the fiber
        directions specified by the list of mesh functions.

    """
    func_components = list()
    for i, mf in enumerate(meshfunctions):
        func_components.append(convert_meshfunction_to_function(mf))

    mesh = meshfunctions[0].mesh()
    if functionspace is None:
        functionspace = dlf.VectorFunctionSpace(mesh, "DG", 0)
    vector_func = dlf.Function(functionspace)
    assign_scalars_to_vectorfunctions(func_components, vector_func)
    return vector_func


def convert_scalarcomponents_to_vectorfunction(components, functionspace=None):
    """
    Convert a list of components to a vector-valued dolfin.Function object.

    Parameters
    ----------

    components : list, tuple
        The list/tuple of either mesh function or dolfin.Function objects used
        to create a vector-valued function. Mesh functions are first converted to
        scalar-valued functions, and then they are assigned to the components of
        the vector-valued function.
    functionspace : dolfin.FunctionSpace (default None)
        The function space used to create the vector-valued dolfin.Function
        object. A new function space with element 'p0' is created if None is
        provided.


    Returns
    -------

    vector_func : dolfin.Function
        The vector-valued dolfin.Function object created to describe the fiber
        directions specified by the list of mesh functions.

    """
    new_components = [None]*len(components)
    for i, component in enumerate(components):
        if isinstance(component, dlf.cpp.mesh.MeshFunctionDouble):
            new_components[i] = convert_meshfunction_to_function(component)
        elif isinstance(component, dlf.Function):
            new_components[i] = component
        else:
            raise TypeError("I don't recognize the object you are giving " \
                            + "as a component.")

    try:
        mesh = component.mesh()
    except AttributeError:
        mesh = component.function_space().mesh()

    if functionspace is None:
        functionspace = dlf.VectorFunctionSpace(mesh, "DG", 0)
    vector_func = dlf.Function(functionspace)
    assign_scalars_to_vectorfunctions(new_components, vector_func)
    return vector_func
