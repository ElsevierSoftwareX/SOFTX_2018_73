Code Structure and User Interface
=================================

This chapter covers the user interface, as well as the internal design
of FEniCS Mechanics. The following abbreviations are used in figures
below.

   +------------------+-------------------------------------+
   | **Abbreviation** |          **Full Name**              |
   +==================+=====================================+
   |   :code:`BMP`    |  :code:`BaseMechanicsProblem`       |
   +------------------+-------------------------------------+
   |   :code:`IM`     |  :code:`IsotropicMaterial`          |
   +------------------+-------------------------------------+
   |   :code:`MP`     |  :code:`MechanicsProblem`           |
   +------------------+-------------------------------------+
   |   :code:`LIM`    |  :code:`LinearIsoMaterial`          |
   +------------------+-------------------------------------+
   |   :code:`FMP`    |  :code:`FluidMechanicsProblem`      |
   +------------------+-------------------------------------+
   |   :code:`NHM`    |  :code:`NeoHookeMaterial`           |
   +------------------+-------------------------------------+
   |   :code:`SMP`    |  :code:`SolidMechanicsProblem`      |
   +------------------+-------------------------------------+
   |   :code:`AM`     |  :code:`AnisotropicMaterial`        |
   +------------------+-------------------------------------+
   |   :code:`MBS`    |  :code:`MechanicsBlockSolver`       |
   +------------------+-------------------------------------+
   |   :code:`FM`     |  :code:`FungMaterial`               |
   +------------------+-------------------------------------+
   |   :code:`NVS`    |  :code:`NonlinearVariationalSolver` |
   +------------------+-------------------------------------+
   |   :code:`GM`     |  :code:`GuccioneMaterial`           |
   +------------------+-------------------------------------+
   |   :code:`SMS`    |  :code:`SolidMechanicsSolver`       |
   +------------------+-------------------------------------+
   |   :code:`F`      |  :code:`Fluid`                      |
   +------------------+-------------------------------------+
   |   :code:`FMS`    |  :code:`FluidMechanicsSolver`       |
   +------------------+-------------------------------------+
   |   :code:`NF`     |  :code:`NewtonianFluid`             |
   +------------------+-------------------------------------+
   |   :code:`EM`     |  :code:`ElasticMaterial`            |
   +------------------+-------------------------------------+


Code Structure
--------------

The flow of information within FEniCS Mechanics is shown in Figure
:numref:`figure-fm-info_flow`. First, the user defines the mechanics
problem they wish to solve through a python dictionary, which we will
refer to as :code:`config`. FEniCS Mechanics then uses this input to
define the variational form that is to be solved through the Unified
Form Language (UFL) from the FEniCS Project. Note that information
provided in :code:`config` is sent to two separate components: problem
formulation, and material law. This separation is done to maintain the
generality of the governing equation given in
:ref:`continuum-mechanics-label`. In other words, a separate part of
the code is responsible for tailoring the governing equations to
specific material models, providing a modular structure that better
lends itself to the addition of new models.

The forms defined in the problem formulation stage are then used for
matrix assembly in order to obtain the numerical solution to the
specified problem. All of the terms that need to be defined within
:code:`config` are listed and explained in
:ref:`problem-config-label`.

.. _figure-fm-info_flow:

.. figure:: /_static/figures/fm-flow.png
   :scale: 50 %

   The flow of information within FEniCS Mechanics.


Problem Objects
***************

There are three classes that define the variational form of the
computational mechanics problem: :class:`MechanicsProblem`,
:class:`SolidMechanicsProblem`, and :class:`FluidMechanicsProblem`.
The input, :code:`config`, takes the same structure for all three. All
of three classes are derived from :class:`BaseMechanicsProblem`, as is
shown in :numref:`figure-fm-problems`. Functions that parse different
parts of :code:`config` belong to :class:`BaseMechanicsProblem` since
they are common to all mechanics problems. In addition to parsing
methods, terms in the variational form of the governing equation are
defined in the parent class, as well as any functions that update the
state of common attributes for all problems.

.. _figure-fm-problems:

.. figure:: /_static/figures/fm-problem_tree.png
   :scale: 50 %

   A tree of the different problem classes in FEniCS Mechanics showing
   their inheritance.

One difference between all three is the time integration scheme
used. Specifically, :class:`MechanicsProblem` treats the system of
ODEs after FEM discretization as first order. Thus, the system is
reduced to a set of first order ODEs for solid mechanics as shown
at the end of :ref:`fem-label`, and integrated with the method
described in :ref:`first-order-odes-label`. The time integration
scheme in :class:`FluidMechanicsProblem` is currently the same without
the need for the equation :math:`\dot{u} = v`. On the other hand,
:class:`SolidMechanicsProblem` defines the variational form using the
Newmark integration scheme. This is a common integrator used for solid
mechanics problems.

Another difference between :class:`MechanicsProblem` and the other two
problem classes is that :class:`MechanicsProblem` uses separate
function space objects from :code:`dolfin` for vector and scalar
fields. The other two problem classes use a mixed function space
object.

All problem classes are instantiated by providing the python
dictionary, :code:`config`, e.g.

.. code-block:: python

   import fenicsmechanics as fm
   # ...
   # Code defining 'config'
   # ...
   problem = fm.MechanicsProblem(config)

Full demonstrations of the use of FEniCS Mechanics are given in
:ref:`examples-label`.

.. SHOULD MAYBE SHOW THE MEMBER DATA HERE?


Solver Objects
**************

Once the problem object has been created with the :code:`config`
dictionary, it is passed to a solver class for instantiation. Like the
problem classes, there are three solver classes:
:class:`MechanicsBlockSolver`, :class:`SolidMechanicsSolver`, and
:class:`FluidMechanicsSolver`. The inheritance of these classes are
shown in :numref:`figure-fm-solvers`. All three solver classes use the
UFL objects defined by their corresponding problem classes to assemble
the resulting linear algebraic system at each iteration of a nonlinear
solve. This is repeated for all time steps of the problem if it is
time-dependent.

Note that :class:`MechanicsBlockSolver` is a stand-alone class, while
:class:`SolidMechanicsSolver` and :class:`FluidMechanicsSolver` are
both subclasses of the :class:`NonlinearVariationalSolver` in
:code:`dolfin`. This is due to the fact that :class:`MechanicsProblem`
uses separate function spaces for the vector and scalar fields
involved in the problem, and hence uses `CBC-Block
<https://bitbucket.org/fenics-apps/cbc.block>`_ to assemble and solve
the resulting variational form.

.. _figure-fm-solvers:

.. figure:: /_static/figures/fm-solver_tree.png
   :scale: 50 %

   A tree of the different solver classes in FEniCS Mechanics showing
   their inheritance.


Constitutive Equations
**********************

A number of constitutive equations have been implemented in FEniCS
Mechanics. All of them can be found in the :code:`materials`
sub-package. A list of all constitutive equations included can be seen
by executing
:code:`fenicsmechanics.list_implemented_materials()`. The inheritance
for constitutive equations of solid materials is shown in
:numref:`figure-fm-solids`.

.. _figure-fm-solids:

.. figure:: /_static/figures/fm-solids_tree.png
   :scale: 50 %

   A tree of the different constitutive equations implemented for
   solid materials in FEniCS Mechanics.

The inheritance for constitutive equations of fluids is shown in
:numref:`figure-fm-fluids`.

.. _figure-fm-fluids:

.. figure:: /_static/figures/fm-fluids_tree.png
   :scale: 50 %

   A tree of the constitutive equations implemented for fluids in
   FEniCS Mechanics.

It can be seen that the classes defining different constitutive
equations are grouped in such a way that common functions are defined
in parent classes. This is more evident for solid materials. We see in
:numref:`figure-fm-solids` that all classes are derived from the
:class:`ElasticMaterial` class. Then, the second level of inheritance
separates isotropic and anisotropic materials.

Do note that the user is not limited to the constitutive equations
provided in :code:`materials`. An example of providing a user-defined
constitutive equation is given in :ref:`user-consteqn-label`.


.. _problem-config-label:

User Interface
--------------

The mechanics problem of interest is specified using a python
dictionary referred to as :code:`config`. Within this dictionary, the
user provides information regarding the mesh, material properties, and
details to formulate the boundary value problem. Each of these are
defined as subdictionaries within :code:`config`. Further details are
provided below.


Mesh
****

The mesh subdictionary is where the user will provide all of the
information regarding the discretization of the computational domain,
and any tags necessary to identify various regions of the
boundary. We now provide a list of keywords and their descriptions.

* :code:`mesh_file`: the name of the file containing the mesh
  information (nodes and connectivity) in a format supported by
  :code:`dolfin`. If the user is creating a :code:`dolfin.Mesh` object
  within the same script, they can use the mesh object instead of a
  file name.
* :code:`boundaries`: the name of the file containing the mesh
  function to mark different boundaries regions of the mesh. Similarly
  to :code:`mesh_file`, the user can pass a
  :code:`dolfin.MeshFunction` object directly if they are creating it
  within the same script.


Material
********

The user specifies the constitutive equation they wish to use, as well
as any parameters that it requires in the material
subdictionary. Below is a list of keywords and their descriptions.

* :code:`type`: The class of material that will be used,
  e.g. elastic, viscous, viscoelastic, etc.
* :code:`const_eqn`: The name of the constitutive equation to be
  used. User may provide their own class which defines a material
  instead of using those implemented in
  :code:`fenicsmechanics.materials`. For a list of implemented
  materials, call
  :code:`fenicsmechanics.list_implemented_materials()`.
* :code:`incompressible`: True if the material is incompressible. An
  additional weak form for the incompressibility constraint will be
  added to the problem.
* :code:`density`: Scalar specifying the density of the material.

Additional material parameters depend on the constitutive equation
that is used. To see which other values are required, check the
documentary of that specific constitutive equation.


Formulation
***********

Details for the formulation of the boundary value problem are provided
in the formulation subdictionary. This is where the user provides
parameters for the time integration, any initial conditions, the type
of finite element to be used, body force to be applied, boundary
conditions, etc. A list of keywords and their descriptions is provided
below.

* :code:`time`: providing this dictionary is optional. If it is not
  provided, the problem is assumed to be a steady-state problem.

  * :code:`unsteady`: A boolean value specifying if the problem is
    time dependent.
  * :code:`dt`: The time step used for the numerical integrator.
  * :code:`interval`: A list or tuple of length 2 specifying the time
    interval, i.e. :code:`[t0, tf]`.
  * :code:`theta`: The weight given to the current time step and
    subtracted from the previous, i.e.

    .. math::

       \frac{dy}{dt} = \theta f(y_{n+1}) + (1 - \theta)f(y_n).

    Note that :math:`\theta = 1` gives a fully implicit scheme, while
    :math:`theta = 0` gives a fully explicit one. It is optional for
    the user to provide this value. If it is not provided, it is
    assumed that :math:`\theta = 1`.
  * :code:`beta`: The :math:`\beta` parameter used in the Newmark
    integration scheme. Note that the Newmark integration scheme is
    only used by :class:`SolidMechanicsProblem`. Providing this value
    is optional. If it is not provided, it is assumed that
    :math:`\beta = 0.25`.
  * :code:`gamma`: The :math:`\gamma` parameter used in the Newmark
    integration scheme. Note that the Newmark integration scheme is
    only used by :class:`SolidMechanicsProblem`. Providing this value
    is optional. If it is not provided, it is assumed that
    :math:`\gamma = 0.5`.

* :code:`initial_condition`: a subdictionary containing initial
  conditions for the field variables involved in the problem
  defined. If this is not provided, all initial conditions are assumed
  to be zero.

  * :code:`displacement`: A :class:`dolfin.Coefficient` object
    specifying the initial value for the displacement.
  * :code:`velocity`: A :class:`dolfin.Coefficient` object specifying
    the initial value for the velocity.
  * :code:`pressure`: A :class:`dolfin.Coefficient` object specifying
    the initial value for the pressure.

* :code:`element`: Name of the finite element to be used for the
  discrete function space. Currently, elements of the form
  :code:`p<n>-p<m>` are supported, where :code:`<n>` is the degree
  used for the vector function space, and :code:`<m>` is the degree
  used for the scalar function space. If the material is not
  incompressible, only the first term should be specified. E.g.,
  :code:`p2`.
* :code:`domain`: String specifying whether the problem is to be
  formulated in terms of Lagrangian, Eulerian, or ALE
  coordinates. Note that ALE is currently not supported.
* :code:`inverse`: Boolean value specifying if the problem is an
  inverse elastostatics problem. If value is not provided, it is set
  to :code:`False`. For more information, see Govindjee and Mihalic
  :cite:`govindjeemihalic1996`.
* :code:`body_force`: Value of the body force throughout the body.
* :code:`bcs`: A subdictionary of :code:`formulation` where the
  boundary conditions are specified. If this dictionary is not
  provided, no boundary conditions are applied and a warning is
  printed to the screen

  * :code:`dirichlet`: A subdictionary of :code:`bcs` where the
    Dirichlet boundary conditions are specified. If this dictionary is
    not provided, no Dirichlet boundary conditions are applied and a
    warning is printed to the screen.

    * :code:`velocity`: List of velocity values for each Dirichlet
      boundary region specified. The order must match the order used
      in the list of region IDs.
    * :code:`displacement`: List of displacement values for each
      Dirichlet boundary region specified. The order must match the
      order used in the list of region IDs.
    * :code:`pressure`: List of pressure values for each Dirichlet
      boundary region specified. The order must match the order used
      in the list of pressure region IDs.
    * :code:`regions`: List of the region IDs on which Dirichlet
      boundary conditions for displacement and velocity are to be
      imposed. These IDs must match those used by the mesh function
      provided. The order must match that used in the list of values
      (velocity and displacement).
    * :code:`p_regions`: List of the region IDs on which Dirichlet
      boundary conditions for pressure are to be imposed. These IDs
      must match those used by the mesh function provided. The order
      must also match that used in the list of values
      (:code:`pressure`).

  * :code:`neumann`: A subdictionary of :code:`bcs` where the Neumann
    boundary conditions are specified. If this dictionary is not
    provided, no Neumann boundary conditions are applied and a warning
    is printed to the screen.

    * :code:`regions`: List of the region IDs on which Neumann
      boundary conditions are to be imposed. These IDs must match
      those used by the mesh function provided. The order must match
      the order used in the list of types and values.
    * :code:`types`: List of strings specifying whether a
      :code:`'pressure'`, :code:`'piola'`, or :code:`'cauchy'` is
      provided for each region. The order must match the order used in
      the list of region IDs and values.
    * :code:`values`: List of values for each Dirichlet boundary
      region specified. The order must match the order used in the
      list of region IDs and types.
