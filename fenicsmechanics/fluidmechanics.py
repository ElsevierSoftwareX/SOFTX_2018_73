"""
This module provides a problem and solver class to solve fluid mechanics
problems using the mixed function space functionality in FEniCS. Note
that only incompressible fluids are currently supported.

"""
from __future__ import print_function

import dolfin as dlf

from . import materials
from .utils import duplicate_expressions, _create_file_objects, _write_objects
from .basemechanics import BaseMechanicsProblem, BaseMechanicsSolver
from .dolfincompat import MPI_COMM_WORLD

from .dolfincompat import MPI_COMM_WORLD

from inspect import isclass

__all__ = ['FluidMechanicsProblem', 'FluidMechanicsSolver']


class FluidMechanicsProblem(BaseMechanicsProblem):
    """
    This class represents the variational form of a fluid
    mechanics problem. The specific form and boundary conditions
    are generated based on definitions provided by the user in a
    dictionary of sub-dictionaries.

    Refer to the documentation of the FEniCS Mechanics package for
    details on how to define a problem using the :code:`config`
    dictionary.

    >>> import fenicsmechanics as fm
    >>> help(fm)


    """

    def __init__(self, user_config):

        BaseMechanicsProblem.class_name = "FluidMechanicsProblem"
        BaseMechanicsProblem.__init__(self, user_config)

        # Define necessary member data
        self.define_function_spaces()
        self.define_functions()
        self.define_deformation_tensors()
        self.define_material()
        self.define_dirichlet_bcs()
        self.define_forms()
        self.define_ufl_equations()
        self.define_ufl_equations_diff()

        return None


    def define_function_spaces(self):
        """
        Define the function space based on the degree(s) specified
        in config['formulation']['element'] and add it as member data.
        If the problem is specified as incompressible, a mixed function
        space made up of a vector and scalar function spaces is defined.
        Note that this differs from MechanicsProblem since there,
        the vector and scalar function spaces are defined separately.


        """

        cell = self.mesh.ufl_cell()
        vec_degree = int(self.config['formulation']['element'][0][-1])
        if vec_degree == 0:
            vec_family = "DG"
        else:
            vec_family = "CG"
        vec_element = dlf.VectorElement(vec_family, cell, vec_degree)

        if self.config['material']['incompressible']:
            scalar_degree = int(self.config['formulation']['element'][1][-1])
            if scalar_degree == 0:
                scalar_family = "DG"
            else:
                scalar_family = "CG"
            scalar_element = dlf.FiniteElement(scalar_family, cell, scalar_degree)
            element = vec_element*scalar_element
        else:
            element = vec_element

        self.functionSpace = dlf.FunctionSpace(self.mesh, element)

        return None


    def define_functions(self):
        """
        Define the vector and scalar functions necessary to define the
        problem specified in the 'config' dictionary. Functions that are
        not needed are set to 0. This method calls one of two methods to
        define these functions based depending on whether the fluid is
        incompressible or not.


        """

        if self.config['material']['incompressible']:
            self.define_incompressible_functions()
        else:
            self.define_compressible_functions()

        return None


    def define_incompressible_functions(self):
        """
        Define mixed functions necessary to formulate an incompressible
        fluid mechanics problem. The mixed function is also split using
        dolfin.split (for use in UFL variational forms) and u.split(),
        where u is a mixed function (for saving solutions separately).
        The names of the member data added to the instance of the
        SolidMechanicsProblem class are:

        - :code:`sys_v`: mixed function
        - :code:`ufl_velocity`: sub component corresponding to velocity
        - :code:`velocity`: copy of sub component for writing and assigning
          values
        - :code:`ufl_pressure`: sub component corresponding to pressure
        - :code:`pressure`: copy of sub component for writing and assigning values
        - :code:`sys_du`: mixed trial function
        - :code:`trial_vector`: sub component of mixed trial function
        - :code:`trial_scalar`: sub component of mixed trial function
        - :code:`test_vector`: sub component of mixed test function
        - :code:`test_scalar`: sub component of mixed test function

        If problem is unsteady, the following are also added:

        - :code:`sys_v0`: mixed function at previous time step
        - :code:`ufl_velocity0`: sub component corresponding to velocity
        - :code:`velocity0`: copy of sub component for writing and assigning values
        - :code:`ufl_pressure0`: sub component at previous time step
        - :code:`pressure0`: copy of sub component at previous time step


        """

        self.sys_v = dlf.Function(self.functionSpace)
        self.ufl_velocity, self.ufl_pressure = dlf.split(self.sys_v)

        init = self.config['formulation']['initial_condition']
        types_for_assign = (dlf.Constant, dlf.Expression)
        if init['velocity'] is not None \
           and init['pressure'] is not None:
            if isinstance(init['velocity'], types_for_assign):
                self.velocity = dlf.Function(self.functionSpace.sub(0).collapse())
                self.velocity.assign(init['velocity'])
            else:
                self.velocity = dlf.project(init['velocity'],
                                            self.functionSpace.sub(0).collapse())
            if isinstance(init['pressure'], types_for_assign):
                self.pressure = dlf.Function(self.functionSpace.sub(1).collapse())
                self.pressure.assign(init['pressure'])
            else:
                self.pressure = dlf.project(init['pressure'],
                                            self.functionSpace.sub(1).collapse())
        elif init['velocity'] is not None:
            _, self.pressure = self.sys_v.split(deepcopy=True)
            if isinstance(init['velocity'], types_for_assign):
                self.velocity = dlf.Function(self.functionSpace.sub(0).collapse())
                self.velocity.assign(init['velocity'])
            else:
                self.velocity = dlf.project(init['velocity'],
                                            self.functionSpace.sub(0).collapse())
        elif init['pressure'] is not None:
            self.velocity, _ = self.sys_v.split(deepcopy=True)
            if isinstance(init['pressure'], types_for_assign):
                self.pressure = dlf.Function(self.functionSpace.sub(1).collapse())
                self.pressure.assign(init['pressure'])
            else:
                self.pressure = dlf.project(init['pressure'],
                                            self.functionSpace.sub(1).collapse())
        else:
            print("No initial conditions were provided")
            self.velocity, self.pressure = self.sys_v.split(deepcopy=True)

        self.velocity.rename("v", "velocity")
        self.pressure.rename("p", "pressure")

        self.define_function_assigners()
        self.assigner_v2sys.assign(self.sys_v, [self.velocity,
                                                self.pressure])

        self.sys_dv = dlf.TrialFunction(self.functionSpace)
        self.trial_vector, self.trial_scalar = dlf.split(self.sys_dv)
        self.test_vector, self.test_scalar = dlf.TestFunctions(self.functionSpace)

        if self.config['formulation']['time']['unsteady']:
            self.sys_v0 = self.sys_v.copy(deepcopy=True)

            self.ufl_velocity0, self.ufl_pressure0 = dlf.split(self.sys_v0)
            self.velocity0, self.pressure0 = self.sys_v0.split(deepcopy=True)
            self.velocity0.rename("v0", "velocity0")
            self.pressure0.rename("p0", "pressure0")

            self.define_ufl_acceleration()

        return None


    def define_compressible_functions(self):
        """
        **COMPRESSIBLE FLUIDS ARE CURRENTLY NOT SUPPORTED.**


        """

        raise NotImplementedError("Compressible fluids are not yet supported.")

        return None


    def define_ufl_acceleration(self):
        """
        Define the acceleration based on a single step finite difference
        and add as member data under 'ufl_acceleration'.

        ufl_acceleration = (ufl_velocity - ufl_velocity0)/dt,

        where dt is the size of the time step.


        """

        # Should add option to use a different finite
        # difference schemes.
        dt = self.config['formulation']['time']['dt']
        v, v0 = self.ufl_velocity, self.ufl_velocity0
        self.ufl_acceleration = (v - v0)/dt

        return None


    def define_deformation_tensors(self):
        """
        Define kinematic tensors needed for constitutive equations. Secondary
        tensors are added with the suffix "0" if the problem is time-dependent.
        The names of member data added to an instance of FluidMechanicsProblem
        class are:

        - :code:`velocityGradient`
        - :code:`velocityGradient0`


        """

        # Exit function if tensors have already been defined
        if hasattr(self, "velocityGradient"):
            return None

        self.velocityGradient = dlf.grad(self.ufl_velocity)
        if self.config['formulation']['time']['unsteady']:
            self.velocityGradient0 = dlf.grad(self.ufl_velocity0)
        else:
            self.velocityGradient0 = 0

        return None


    def define_material(self):
        """
        Create an instance of the class that defines the constitutive
        equation for the current problem and add it as member data under
        '_material'. All necessary parameters must be included in the
        'material' subdictionary of 'config'. The specific values necessary
        depends on the constitutive equation used. Please check the
        documentation of the material classes provided in
        'fenicsmechanics.materials' if using a built-in material.


        """

        # Only one fluid material right now.
        const_eqn = self.config['material']['const_eqn']
        if isclass(const_eqn):
            mat_class = self.config['material']['const_eqn']
        elif const_eqn in ["newtonian", "stokes"]:
            mat_class = materials.fluids.NewtonianFluid
        else:
            msg = "The material '%s' has not been implemented. A class for such" \
                  + " material must be provided."
            raise NotImplementedError(msg % const_eqn)

        self._material = mat_class(**self.config['material'])

        return None


    def define_dirichlet_bcs(self):
        """
        Define a list of Dirichlet BC objects based on the problem configuration
        provided by the user, and add it as member data under 'dirichlet_bcs'. If
        no Dirichlet BCs are provided, 'dirichlet_bcs' is set to None.


        """

        # Exit function if no Dirichlet BCs were provided.
        if self.config['formulation']['bcs']['dirichlet'] is None:
            self.dirichlet_bcs = None
            return None

        if self.config['material']['incompressible']:
            self._define_incompressible_dirichlet_bcs()
        else:
            self._define_compressible_dirichlet_bcs()

        return None


    def _define_compressible_dirichlet_bcs(self):
        """
        Define Dirichlet BCs for compressible fluid mechanics problems.

        THIS TYPE OF PROBLEM IS CURRENTLY NOT SUPPORTED.


        """

        dirichlet_dict = self.config['formulation']['bcs']['dirichlet']
        if 'velocity' is dirichlet_dict:
            self.dirichlet_bcs = dict()
            dirichlet_bcs = self.__define_velocity_bcs(self.functionSpace,
                                                       dirichlet_dict,
                                                       self.boundaries)
            self.dirichlet_bcs.update(dirichlet_bcs)
        else:
            self.dirichlet_bcs = None

        return None


    def _define_incompressible_dirichlet_bcs(self):
        """
        Define Dirichlet BCs for incompressible fluid mechanics problems.


        """

        self.dirichlet_bcs = dict()
        dirichlet_dict = self.config['formulation']['bcs']['dirichlet']
        if 'velocity' in dirichlet_dict:
            vel_bcs = self.__define_velocity_bcs(self.functionSpace.sub(0),
                                                 dirichlet_dict,
                                                 self.boundaries)
            self.dirichlet_bcs.update(vel_bcs)

        if 'pressure' in dirichlet_dict:
            pressure_bcs = self.__define_pressure_bcs(self.functionSpace.sub(1),
                                                      dirichlet_dict,
                                                      self.boundaries)
            self.dirichlet_bcs.update(pressure_bcs)

        return None


    def define_forms(self):
        """
        Define all of the variational forms necessary for the problem specified
        by the user and add them as member data. The variational forms are those
        corresponding to the order reduction of the time derivative (for unsteady
        solid material simulations), the balance of linear momentum, and the
        incompressibility constraint.


        """

        # Define UFL objects corresponding to the local acceleration
        # if problem is unsteady.
        self.define_ufl_local_inertia()

        # Define UFL objects corresponding to the convective acceleration
        # if problem is formulated with respect to Eulerian coordinates
        self.define_ufl_convec_accel()

        # Define UFL objects corresponding to the stress tensor term.
        # This should always be non-zero for deformable bodies.
        self.define_ufl_stress_work()

        # Define UFL object corresponding to the body force term. Assume
        # it is zero if key was not provided.
        self.define_ufl_body_force()

        # Define UFL object corresponding to the traction force terms. Assume
        # it is zero if key was not provided.
        self.define_ufl_neumann_bcs()

        return None


    def define_ufl_local_inertia(self):
        """
        Define the UFL object corresponding to the local acceleration
        term in the weak form.


        """

        # Set to 0 and exit if problem is steady.
        if not self.config['formulation']['time']['unsteady']:
            self.ufl_local_inertia = 0
            return None

        xi = self.test_vector
        rho = self.config['material']['density']

        self.ufl_local_inertia = dlf.dot(xi, rho*self.ufl_acceleration)*dlf.dx

        return None


    def define_ufl_convec_accel(self):
        """
        Define the UFL object corresponding to the convective acceleration
        term in the weak form.


        """

        # Check if modeling Stokes flow, and set convective acceleration
        # to 0 if we are.
        stokes = self.config['material']['const_eqn'] == 'stokes'
        if stokes:
            self.ufl_convec_accel = 0
            self.ufl_convec_accel0 = 0
        else:
            xi = self.test_vector
            rho = self.config['material']['density']
            a_c = rho*dlf.grad(self.ufl_velocity)*self.ufl_velocity
            self.ufl_convec_accel = dlf.dot(xi, a_c)*dlf.dx

            if self.config['formulation']['time']['unsteady']:
                a_c0 = rho*dlf.grad(self.ufl_velocity0)*self.ufl_velocity0
                self.ufl_convec_accel0 = dlf.dot(xi, a_c0)*dlf.dx
            else:
                self.ufl_convec_accel0 = 0

        return None


    def define_ufl_stress_work(self):
        """
        Define the UFL object corresponding to the stress tensor term
        in the weak form.


        """

        stress_tensor = self._material.stress_tensor(self.velocityGradient,
                                                     self.ufl_pressure)

        xi = self.test_vector
        self.ufl_stress_work = dlf.inner(dlf.grad(xi), stress_tensor)*dlf.dx

        # Define the stress work term for the previous time step.
        if self.config['formulation']['time']['unsteady']:
            stress_tensor0 = self._material.stress_tensor(self.velocityGradient0,
                                                          self.ufl_pressure0)
            self.ufl_stress_work0 = dlf.inner(dlf.grad(xi), stress_tensor0)*dlf.dx
        else:
            self.ufl_stress_work0 = 0

        return None


    def define_ufl_body_force(self):
        """
        Define the UFL object corresponding to the body force term in
        the weak form.


        """

        # Set to 0 and exit if key is not in config dictionary.
        if self.config['formulation']['body_force'] is None:
            self.ufl_body_force = 0
            self.ufl_body_force0 = 0
            return None

        rho = self.config['material']['density']
        b = self.config['formulation']['body_force']
        xi = self.test_vector

        self.ufl_body_force = dlf.dot(xi, rho*b)*dlf.dx

        # Create a copy of the body force term to use at a different time step.
        if self.config['formulation']['time']['unsteady'] and hasattr(b,'t'):
            b0, = duplicate_expressions(b)
            self.ufl_body_force0 = dlf.dot(xi, rho*b0)*dlf.dx
        elif self.config['formulation']['time']['unsteady']:
            self.ufl_body_force0 = dlf.dot(xi, rho*b)*dlf.dx
        else:
            self.ufl_body_force0 = 0

        return None


    def define_ufl_neumann_bcs(self):
        """
        Define the variational forms for all of the Neumann BCs given
        in the 'config' dictionary under "ufl_neumann_bcs". If the problem
        is time-dependent, a secondary variational form is defined at the
        previous time step with the name "ufl_neumann_bcs0".


        """

        # Exit function if no Neumann BCs were provided.
        if self.config['formulation']['bcs']['neumann'] is None:
            self.ufl_neumann_bcs = 0
            self.ufl_neumann_bcs0 = 0
            return None

        # Get handles to the different lists and the domain.
        regions = self.config['formulation']['bcs']['neumann']['regions']
        types = self.config['formulation']['bcs']['neumann']['types']
        values = self.config['formulation']['bcs']['neumann']['values']
        domain = self.config['formulation']['domain']

        define_ufl_neumann = BaseMechanicsProblem.define_ufl_neumann_form

        self.ufl_neumann_bcs = define_ufl_neumann(regions, types,
                                                  values, domain,
                                                  self.mesh,
                                                  self.boundaries,
                                                  0, # Deformation gradient
                                                  0, # Jacobian
                                                  self.test_vector)

        if self.config['formulation']['time']['unsteady']:
            values0 = duplicate_expressions(*values)
            self.ufl_neumann_bcs0 = define_ufl_neumann(regions, types,
                                                       values0, domain,
                                                       self.mesh,
                                                       self.boundaries,
                                                       0, # Deformation gradient
                                                       0, # Jacobian
                                                       self.test_vector)
        else:
            self.ufl_neumann_bcs0 = 0

        return None


    def define_ufl_equations(self):
        """
        Define all of the variational forms necessary for the problem
        specified in the 'config' dictionary, as well as the mixed
        variational form.


        """

        theta = self.config['formulation']['time']['theta']

        # Momentum
        self.G1 = self.ufl_local_inertia \
                  + theta*(self.ufl_convec_accel \
                           + self.ufl_stress_work \
                           - self.ufl_body_force \
                           - self.ufl_neumann_bcs) \
                  + (1.0 - theta)*(self.ufl_convec_accel0 \
                                   + self.ufl_stress_work0 \
                                   - self.ufl_body_force0 \
                                   - self.ufl_neumann_bcs0)

        # Incompressibility
        q = self.test_scalar
        bvol = self._material.incompressibilityCondition(self.ufl_velocity)
        self.G2 = q*bvol*dlf.dx

        # Mixed variational form
        self.G = self.G1 + self.G2

        return None


    def _define_incompressibility_constraint(self):
        """
        THIS SHOULD BE SUPPLIED BY THE CONSTITUTIVE EQUATION CLASS.

        """

        q = self.test_scalar
        n = dlf.FacetNormal(self.mesh)
        G2 = q*dlf.dot(self.ufl_velocity, n)*dlf.ds \
             - dlf.dot(dlf.grad(q), self.ufl_velocity)*dlf.dx

        return G2


    def define_ufl_equations_diff(self):
        """
        Differentiate all of the variational forms with respect to appropriate
        fields variables and add as member data.


        """

        self.dG = dlf.derivative(self.G, self.sys_v, self.sys_dv)

        return None


    def define_function_assigners(self):
        """
        Create function assigners to update the current and previous time
        values of all field variables. This is specific to incompressible
        simulations since the mixed function space formulation requires the
        handling of the mixed functions and the copies of its subcomponents
        in a specific manner.


        """

        W = self.functionSpace
        v = self.velocity
        p = self.pressure

        self.assigner_sys2v = dlf.FunctionAssigner([v.function_space(),
                                                    p.function_space()], W)
        self.assigner_v2sys = dlf.FunctionAssigner(W, [v.function_space(),
                                                       p.function_space()])

        if self.config['formulation']['time']['unsteady']:
            v0 = self.velocity0
            p0 = self.pressure0

            self.assigner_v02sys = dlf.FunctionAssigner(W, [v0.function_space(),
                                                            p0.function_space()])
            self.assigner_sys2v0 = dlf.FunctionAssigner([v0.function_space(),
                                                         p0.function_space()], W)

        return None


    @staticmethod
    def __define_velocity_bcs(W, dirichlet_dict, boundaries):
        """
        Create a dictionary storing the dolfin.DirichletBC objects for
        each displacement BC specified in the 'dirichlet' subdictionary.


        Parameters
        ----------

        W : dolfin.FunctionSpace
            The function space for the displacement.
        dirichlet_dict : dict
            The 'dirichlet' subdictionary of 'config'. Refer to the
            documentation for BaseMechanicsProblem for more information.
        boundaries : dolfin.MeshFunction
            The mesh function used to tag different regions of the
            domain boundary.


        Returns
        -------

        displacement_bcs : dict
            a dictionary of the form {'displacement': [...]}, where
            the value is a list of dolfin.DirichletBC objects.


        """

        velocity_bcs = {'velocity': list()}
        vel_vals = dirichlet_dict['velocity']
        regions = dirichlet_dict['regions']
        components = dirichlet_dict['components']
        for region, value, idx in zip(regions, vel_vals, components):
            if idx == "all":
                bc = dlf.DirichletBC(W, value, boundaries, region)
            else:
                bc = dlf.DirichletBC(W.sub(idx), value, boundaries, region)
            velocity_bcs['velocity'].append(bc)

        return velocity_bcs


    @staticmethod
    def __define_pressure_bcs(W, dirichlet_dict, boundaries):
        """
        Create a dictionary storing the dolfin.DirichletBC objects for
        each pressure BC specified in the 'dirichlet' subdictionary.


        Parameters
        ----------

        W : dolfin.FunctionSpace
            The function space for pressure.
        dirichlet_dict : dict
            The 'dirichlet' subdictionary of 'config'. Refer to the
            documentation for BaseMechanicsProblem for more information.
        boundaries : dolfin.MeshFunction
            The mesh function used to tag different regions of the
            domain boundary.


        Returns
        -------

        pressure_bcs : dict
            a dictionary of the form {'pressure': [...]}, where
            the value is a list of dolfin.DirichletBC objects.


        """

        pressure_bcs = {'pressure': list()}
        p_vals = dirichlet_dict['pressure']
        p_regions = dirichlet_dict['p_regions']
        for region, value in zip(p_regions, p_vals):
            bc = dlf.DirichletBC(W, value, boundaries, region)
            pressure_bcs['pressure'].append(bc)

        return pressure_bcs


class FluidMechanicsSolver(BaseMechanicsSolver):
    """
    This class is derived from the dolfin.NonlinearVariationalSolver to
    solve problems formulated with FluidMechanicsProblem. It passes the
    UFL variational forms to the solver, and loops through all time steps
    if the problem is unsteady. The solvers that are available through
    this class are the same as those available through dolfin. The user
    may use the helper function, 'set_parameters' to set the linear solver
    used, as well as the tolerances for iterative and nonlinear solves, or
    do it directly through the 'parameters' member.


    """


    def __init__(self, problem, fname_vel=None,
                 fname_pressure=None, fname_hdf5=None, fname_xdmf=None):
        """
        Initialize a FluidMechanicsSolver object.

        Parameters
        ----------

        problem : FluidMechanicsProblem
            A SolidMechanicsProblem object that contains the necessary UFL
            forms to define and solve the problem specified in a config
            dictionary.
        fname_vel : str (default None)
            Name of the file series in which the velocity values are
            to be saved.
        fname_pressure : str (default None)
            Name of the file series in which the pressure values are to
            be saved.


        """

        BaseMechanicsSolver.class_name = "FluidMechanicsSolver"
        BaseMechanicsSolver.__init__(self, problem, fname_pressure=fname_pressure,
                                     fname_hdf5=fname_hdf5, fname_xdmf=fname_xdmf)
        self._fnames.update(vel=fname_vel)
        self._file_vel = _create_file_objects(fname_vel)

        return None


    def update_assign(self):
        """
        Update the values of the field variables -- both the current and
        previous time step in preparation for the next step of the simulation.


        """

        problem = self._problem
        unsteady = problem.config['formulation']['time']['unsteady']

        v = problem.velocity
        p = problem.pressure

        if unsteady:
            # Assign current solution to previous time step.
            problem.sys_v0.assign(problem.sys_v)

            # Assign the mixed function values to the split functions.
            problem.assigner_sys2v0.assign([problem.velocity0,
                                           problem.pressure0], problem.sys_v0)

        # Assign the mixed function values to the split functions.
        problem.assigner_sys2v.assign([v, p], problem.sys_v)

        return None
