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
    fiber_name : str, list/tuple
    element : str (default None)
    elementwise : bool (default False)


    Returns
    -------

    fiber_direction : ufl.Coefficient


    """
    geo_dim = mesh.geometry().dim()
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

    if isinstance(fiber_file, ufl.Coefficient):

        # Simplest case. Rename and return the same object. Check the ufl_shape.
        if not isinstance(fiber_name, str):
            raise InvalidCombination()
        fiber_direction = fiber_file
        fiber_direction.rename(fiber_name, "Fiber direction")

    elif isinstance(fiber_file, str):

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
    component_meshfunctions = list()
    for i, component in enumerate(component_names):
        mf = dlf.MeshFunction("double", mesh, mesh.geometry().dim())
        hdf.read(mf, component)
        component_meshfunctions.append(mf)
    if len(component_meshfunctions) == 1:
        component_meshfunctions, = component_meshfunctions
    return component_meshfunctions


def load_meshfunction_fibers(fname, component_names, mesh):
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
    family = "DG" if pd == 0 else "CG"
    V = dlf.VectorFunctionSpace(mesh, family, pd)
    vector_field = dlf.Function(V)
    hdf.read(vector_field, name)
    return vector_field


def load_vector_field(fname, name, mesh, pd):
    _, ext = _splitext(fname)
    if ext == ".h5":
        vector_field = load_vector_field_hdf5(fname, name, mesh, pd)
    else:
        family = "DG" if pd == 0 else "CG"
        V = dlf.VectorFunctionSpace(mesh, family, pd)
        vector_field = dlf.Function(V, fname)
    return vector_field


# def load_elementwise_fibers(hdf, component_names, mesh, as_meshfunctions=True,
#                             functionspace=None):
#     if as_meshfunctions:
#         fiber_components = load_meshfunction_fibers(hdf, component_names, mesh)
#     else:
#         fiber_components = load_scalar_function_fibers(hdf, component_names, mesh,
#                                                        functionspace=functionspace)
#     return fiber_components

################################################################################
# Convert scalar components to a vector function.
def convert_meshfunction_to_function(meshfunction, functionspace=None):
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
    for i, component in enumerate(components):
        dlf.assign(vector_func.sub(i), component)
    return None


def convert_meshfunctions_to_vectorfunction(meshfunctions, functionspace=None):
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


# def define_fiber_directions_from_scalars(hdf, component_names, mesh,
#                                          as_meshfunctions=True, functionspace=None):
#     fiber_components = load_elementwise_fibers(hdf, component_names, mesh,
#                                                as_meshfunctions=as_meshfunctions,
#                                                functionspace=functionspace)
#     fiber_vector = convert_scalarcomponents_to_vectorfunction(fiber_components)
#     return fiber_vector


if __name__ == "__main__":
    fiber_file = '../../meshfiles/ellipsoid/ellipsoid_meshes/ellipsoid_500um.h5'
    fiber_dict = {
        'fiber_file': fiber_file,
        'fiber_name': ['fib1', 'fib2', 'fib3'],
        'elementwise': True
    }

    mesh = dlf.Mesh()
    f = dlf.HDF5File(MPI_COMM_WORLD, fiber_file, "r")
    f.read(mesh, "mesh", False)
    f.close()

    fiber_direction = define_fiber_direction(**fiber_dict, mesh=mesh)
    dlf.File("test/fib_vector_field1.pvd") << fiber_direction

    full_vector_fname = "test/full_vector-fiber_file.h5"
    f = dlf.HDF5File(MPI_COMM_WORLD, full_vector_fname, "w")
    f.write(fiber_direction, "fib")
    f.close()

    fiber_dict['fiber_file'] = full_vector_fname
    fiber_dict['fiber_name'] = "fib"
    fiber_direction = define_fiber_direction(**fiber_dict, mesh=mesh)
    dlf.File("test/fib_vector_field2.pvd") << fiber_direction

    scalar_functions_fname = "test/scalar_functions-fiber_file.h5"
    f = dlf.HDF5File(MPI_COMM_WORLD, scalar_functions_fname, "w")
    for i in range(3):
        f.write(fiber_direction.sub(i), "fib%i" % i)
    f.close()

    fiber_dict = {
        'fiber_file': scalar_functions_fname,
        'fiber_name': ['fib0', 'fib1', 'fib2'],
        'pd': 0
    }

    fiber_direction = define_fiber_direction(**fiber_dict, mesh=mesh)
    dlf.File("test/fib_vector_field3.pvd") << fiber_direction
