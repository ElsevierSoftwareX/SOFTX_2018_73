"""
FEniCS Mechanics is a free python package that is meant to facilitate the
formulation and simulation of computational mechanics problems. It is released
under the BSD 3-clause license found in 'LICENSE.txt'.

This package is built on top of libraries from the <FEniCS Project
`https://fenicsproject.org/`>_. The user is expected to be familiar with basic
python syntax and data types, specifically python dictionaries. This is due to
the fact that FEniCS Mechanics requires that the user specify the problem they
wish to solve through a python dictionary.

The user must provide a python dictionary (referred to as 'config' throughout)
with the keys and values listed below. Actions taken when optional values are
not provided are listed at the very bottom.


'material':
    * 'type' : str
        The class of material that will be used, e.g. elastic, viscous,
        viscoelastic, etc.
    * 'const_eqn' : str, class
        The name of the constitutive equation to be used. User may provide
        their own class which defines a material instead of using those
        implemented in fenicsmechanics.materials. For a list of implemented
        materials, call :code:`fenicsmechanics.list_implemented_materials()`.
    * 'incompressible' : bool
        True if the material is incompressible. An
        additional weak form for the incompressibility
        constraint will be added to the problem.
    * 'density' : float, int
        Scalar specifying the density of the material.

Additional material parameters:
    The additional material parameters depend on the constitutive equation
    being used. Please check the documentation of the specific model used.
    A list of implemented material types and constitutive equations is
    provided by :code:`list_implemented_materials`.


'mesh':
    * 'mesh_file' : str, dolfin.Mesh
        Name of the file containing the mesh information of
        the problem geometry, or a dolfin.Mesh object. Supported
        file formats are *.xml, *.xml.gz, and *.h5.
    * 'boundaries' : str, dolfin.MeshFunction
        Name of the file containing the mesh function to mark different
        boundary regions of the geometry, or a dolfin.MeshFunction object.
        Supported file formats are *.xml, *.xml.gz, and *.h5. This mesh
        function will be used to mark different regions of the domain
        boundary.


'formulation':
    * 'time' (OPTIONAL)
        * 'unsteady' : bool
            True if the problem is time dependent, and False otherwise.
        * 'dt' : float
            Time step used for the numerical integrator.
        * 'interval' : list, tuple
            A list or tuple of length 2 specifying the time interval,
            i.e. [t0, tf].
        * 'theta': float, int (OPTIONAL)
            The weight given to the current time step and subtracted
            from the previous, i.e.

              dy/dt = theta*f(y_{n+1}) + (1 - theta)*f(y_n).

            Note: theta = 1 gives a fully implicit scheme, while
            theta = 0 gives a fully explicit one. The default value is 1.
        * 'beta' : float, int (OPTIONAL)
            The beta parameter used in the Newmark integration scheme.
            Note: the Newmark integration scheme is only used by
            SolidMechanicsProblem. The default value is 0.25.
        * 'gamma' : float, int (OPTIONAL)
            The gamma parameter used in the Newmark integration scheme.
            Note: the Newmark integration scheme is only used by
            SolidMechanicsProblem. The default value is 0.5
    * 'initial_condition' (OPTIONAL)
        * 'displacement' : ufl.Coefficient, list, tuple (OPTIONAL)
            A ufl.Coefficient object specifying the initial value for
            the displacement. If a list or tuple is specified, it is
            converted to a ufl.Coefficient with the same polynomial
            degree specified for the vector field.
        * 'velocity' : ufl.Coefficient, list, tuple (OPTIONAL)
            A ufl.Coefficient object specifying the initial value for
            the velocity. If a list or tuple is specified, it is
            converted to a ufl.Coefficient with the same polynomial
            degree specified for the vector field.
        * 'pressure' : ufl.Coefficient, float, int (OPTIONAL)
            A ufl.Coefficient object specifying the initial value for
            the pressure. If a float or int is specified, it is
            converted to a ufl.Coefficient with the same polynomial
            degree specified for the scalar field. An exception is raised
            if the material is not set as incompressible.
    * 'element' : str
        Name of the finite element to be used for the discrete
        function space. Currently, elements of the form 'p<n>-p<m>'
        are supported, where <n> is the degree used for the vector
        function space, and <m> is the degree used for the scalar
        function space. If the material is not incompressible, only the
        first term should be specified. E.g., 'p2-p1'.
    * 'domain' : str
        String specifying whether the problem is to be formulated
        in terms of Lagrangian, Eulerian, or ALE coordinates. Note:
        ALE is currently not supported.
    * 'inverse' : bool (OPTIONAL)
        True if the problem is an inverse elastostatics problem, and False
        otherwise. For more information, see Govindjee and Mihalic (1996 &
        1998).
    * 'body_force' : ufl.Coefficient, list, tuple (OPTIONAL)
        Value of the body force to be applied throughout the body. If a list
        or tuple is applied, it is converted to a ufl.Coefficient with the
        same polynomial degree specified for the vector field.
    * 'bcs' (OPTIONAL)
        * 'dirichlet' (OPTIONAL)
            * 'velocity' : list, tuple
                List of velocity values (dolfin.Constant or dolfin.Expression)
                for each Dirichlet boundary region specified. The order must
                match the order used in the list of region IDs.
            * 'displacement' : list, tuple
                List of displacement values (dolfin.Constant or dolfin.Expression)
                for each Dirichlet boundary region specified. The order must match
                the order used in the list of region IDs.
            * 'regions' : list, tuple
                List of the region IDs on which Dirichlet boundary conditions for
                displacement and velocity are to be imposed. These IDs must match
                those used by the mesh function provided. The order must match that
                used in the list of values (velocity and displacement).
            * 'components' : list, tuple (OPTIONAL)
                List of vector components to apply the Dirichlet BCs to. If the
                BC is to be applied to the entire vector field, then set it to
                "all". If this list if not provided, it is set to a list with "all"
                repeated for each boundary condition.
            * 'pressure' : list, tuple
                List of pressure values (dolfin.Constant or dolfin.Expression)
                for each Dirichlet boundary region specified. The order must match
                the order used in the list of pressure region IDs.
            * 'p_regions' : list, tuple
                List of the region IDs on which Dirichlet boundary conditions for
                pressure are to be imposed. These IDs must match those used by the
                mesh function provided. The order must also match that used in the
                list of values (pressure).
        * 'neumann' (OPTIONAL)
            * 'regions' : list, tuple
                List of the region IDs on which Neumann boundary conditions are to
                be imposed. These IDs must match those used by the mesh function
                provided. The order must match the order used in the list of types
                and values.
            * 'types' : list, tuple
                List of strings specifying whether a 'pressure', 'piola',
                or 'cauchy' is provided for each region. The order
                must match the order used in the list of region IDs
                and values.
            * 'values' : list, tuple
                List of values (dolfin.Constant or dolfin.Expression)
                for each Dirichlet boundary region specified. The order
                must match the order used in the list of region IDs
                and types.


Below is a list of actions taken if an optional key/value IS NOT PROVIDED:

* 'time': the subdictionary {'unsteady': False} is added under this key.
    * 'theta' : if the 'time' subdictionary is defined without theta, a fully
        implicit scheme is assumed (theta = 1).
    * 'beta' : if the 'time' subdictionary is defined without beta, and the
        SolidMechanicsProblem class is being used, the value of 0.25 will be
        assigned to beta.
    * 'gamma' : if the 'time' subdictionary is defined without gamma, and the
        SolidMechanicsProblem class is being used, the value of 0.5 will be
        assigned to gamma.
* 'initial_condition': the initial condition is assumed to be zero for any
    values that are not provided.
* 'inverse' : a boolean value is assigned to this key if none is provided.
* 'body_force': the body force is set to zero.
* 'bcs': the subdictionary {'dirichlet': None, 'neumann': None} is added
    under this key. A warning is printed alerting the user that no boundary
    conditions were specified.
    * 'dirichlet': if 'bcs' is provided, but 'dirichlet' is not, the value
        of 'dirichlet' is set to None. A warning is printed alerting the user
        that no Dirichlet BC was specified.
    * 'neumann': if 'bcs' is provided, but 'neumann' is not, the value of
        'neumann' is set to None. A warning is printed alerting the user that
        not Neumann BC was specified.



"""
# Import the objects necessary to use package.
from .solidmechanics import SolidMechanicsProblem, SolidMechanicsSolver
from .fluidmechanics import *
from .materials.solid_materials import *
from .materials.fluids import *
from .utils import *
from .dolfincompat import MPI_COMM_WORLD

from .dolfincompat import MPI_COMM_WORLD

import sys as _sys
import dolfin as _dlf
_rank = _dlf.MPI.rank(MPI_COMM_WORLD)
if _sys.version_info[0] < 3:
    try:
        from .mechanicssolver import MechanicsBlockSolver
    except ImportError:
        _s = """
        **********************************************************
        *                        WARNING:                        *
        *  The CBC-Block FEniCS App, which MechanicsBlockSolver  *
        *       depends on, does not seem to be installed.       *
        **********************************************************
        """
        if _rank == 0: print(_s)
        del _s
else:
    _s = """
    **********************************************************
    *                        WARNING:                        *
    *  The CBC-Block FEniCS App, which MechanicsBlockSolver  *
    *      depends on, is not compatible with python 3.      *
    **********************************************************
    """
    if _rank == 0: print(_s)
    del _s

del _sys, _dlf, _rank

# Users can still create a MechanicsProblem object, but will
# not be able to use the MechanicsBlockSolver in version < 3.
from .mechanicsproblem import MechanicsProblem


def init(quad_degree=2):
    import dolfin as dlf
    dlf.parameters['form_compiler']['cpp_optimize'] = True
    dlf.parameters['form_compiler']['representation'] = "uflacs"
    dlf.parameters['form_compiler']['quadrature_degree'] = quad_degree
    dlf.parameters['form_compiler']['optimize'] = True


init()


def _get_mesh_file_names(geometry, ret_facets=False, ret_cells=False,
                         ret_dir=False, ret_mesh=True, ext="xml.gz",
                         refinements=[12, 12]):
    """
    Helper function to get the file names of meshes provided. Will raise
    a FileNotFoundError if files do not exist.


    Parameters
    ----------

    geometry : str
    ret_facets : bool
    ret_cells : bool
    ret_dir : bool
    ret_mesh : bool
    ext: str (Default "xml.gz")
    refinements : list


    Returns
    -------

    mesh_file : str
    facets_file : str
    cells_file : str
    mesh_dir : str

    """
    import os
    from .__CONSTANTS__ import base_mesh_dir
    if geometry not in os.listdir(base_mesh_dir):
        raise FileNotFoundError("A mesh for '%s' is not available." % geometry)

    mesh_dir = os.path.join(base_mesh_dir, geometry)
    base_name = "{geometry}-{name}{refinements}.{ext}"
    if geometry == "unit_domain":
        str_refinements = "-" + "x".join(list(map(str, refinements)))
    else:
        str_refinements = ""
    mesh_file = base_name.format(geometry=geometry, name="mesh",
                                 refinements=str_refinements, ext=ext)
    facets_file = base_name.format(geometry=geometry, name="boundaries",
                                   refinements=str_refinements, ext=ext)
    cells_file = base_name.format(geometry=geometry, name="cells",
                                  refinements=str_refinements, ext=ext)

    mesh_file = os.path.join(mesh_dir, mesh_file)
    facets_file = os.path.join(mesh_dir, facets_file)
    cells_file = os.path.join(mesh_dir, cells_file)

    ret = tuple()
    if ret_mesh:
        ret += (mesh_file,)
    if ret_facets:
        ret += (facets_file,)
    if ret_cells:
        ret += (cells_file,)
    if ret_dir:
        ret += (mesh_dir,)

    # Check if files exist.
    for f in ret:
        if not (os.path.isfile(f) or os.path.isdir(f)):
            raise FileNotFoundError(f)

    # Return string directly if only one is being returned.
    if len(ret) == 1:
        ret = ret[0]
    return ret


__version__ = "1.0.0"
