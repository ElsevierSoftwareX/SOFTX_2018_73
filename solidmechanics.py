import ufl
import dolfin as dlf

from . import materials
from .utils import duplicate_expressions
from .basemechanicsproblem import BaseMechanicsProblem

from inspect import isclass

__all__ = ['SolidMechanicsProblem', 'SolidMechanicsSolver']


class SolidMechanicsProblem(BaseMechanicsProblem):
    """
    This class represents the variational form of a solid
    mechanics problem. The specific form and boundary conditions
    are generated based on definitions provided by the user in a
    dictionary of sub-dictionaries.

    *** Refer to the documentation of BaseMechanicsProblem ***
    ***  for details on how to define a problem using the  ***
    ***                'config' dictionary.                ***


    """


    def __init__(self, user_config):

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
        in config['mesh']['element'] and add it as member data. If
        the problem is specified as incompressible, a mixed function
        space made up of a vector and scalar function spaces is defined.
        Note that this differs from MechanicsProblem since there,
        the vector and scalar function spaces are defined separately.


        """

        cell = self.mesh.ufl_cell()
        vec_degree = int(self.config['mesh']['element'][0][-1])
        if vec_degree == 0:
            vec_family = 'DG'
        else:
            vec_family = 'CG'

        vec_element = dlf.VectorElement(vec_family, cell, vec_degree)

        if self.config['material']['incompressible']:
            scalar_degree = int(self.config['mesh']['element'][1][-1])
            if scalar_degree == 0:
                scalar_family = 'DG'
            else:
                scalar_family = 'CG'
            scalar_element = dlf.FiniteElement(scalar_family, cell, scalar_degree)
            element = vec_element*scalar_element
        else:
            element = vec_element

        self.functionSpace = dlf.FunctionSpace(self.mesh, element)

        return None


    def define_functions(self):
        """
        Define mixed functions necessary to formulate the problem
        specified in the 'config' dictionary. The sub-functions are
        also defined for use in formulating variational forms and saving
        results separately. Functions that are not needed are set to 0.


        """

        if self.config['material']['incompressible']:
            self.define_incompressible_functions()
        else:
            self.define_compressible_functions()

        return None


    def define_incompressible_functions(self):
        """
        Define mixed functions necessary to formulate an incompressible
        solid mechanics problem. The mixed function is also split using
        dolfin.split (for use in UFL variational forms) and u.split(),
        where u is a mixed function (for saving solutions separately).
        The names of the member data added to the instance of the
        SolidMechanicsProblem class are:

        - sys_u : mixed function
        - ufl_displacement : sub component corresponding to displacement
        - displacement : copy of sub component for writing and assigning
                         values
        - ufl_pressure : sub component corresponding to pressure
        - pressure : copy of sub component for writing and assigning values
        - sys_du : mixed trial function
        - trial_vector : sub component of mixed trial function
        - trial_scalar : sub component of mixed trial function
        - test_vector : sub component of mixed test function
        - test_scalar : sub component of mixed test function

        If problem is unsteady, the following are also added:

        - ufl_velocity0 : sub component corresponding to velocity
        - velocity0 : copy of sub component for writing and assigning values
        - ufl_acceleration0 : sub component corresponding to acceleration
        - acceleration0 : copy of sub component for writing and assigning values
        - sys_u0 : mixed function at previous time step
        - ufl_displacement0 : sub component at previous time step
        - displacement0 : copy of sub component at previous time step
        - ufl_pressure0 : sub component at previous time step
        - pressure0 : copy of sub component at previous time step


        """

        self.sys_u = dlf.Function(self.functionSpace)
        self.ufl_displacement, self.ufl_pressure = dlf.split(self.sys_u)
        self.displacement, self.pressure = self.sys_u.split(deepcopy=True)
        self.displacement.rename('u', 'displacement')
        self.pressure.rename('p', 'pressure')

        self.sys_du = dlf.TrialFunction(self.functionSpace)
        self.trial_vector, self.trial_scalar = dlf.split(self.sys_du)

        self.test_vector, self.test_scalar = dlf.TestFunctions(self.functionSpace)

        if self.config['formulation']['time']['unsteady']:
            self.sys_u0 = dlf.Function(self.functionSpace)
            self.ufl_displacement0, self.ufl_pressure0 = dlf.split(self.sys_u0)
            self.displacement0, self.pressure0 = self.sys_u0.split(deepcopy=True)
            self.displacement0.rename('u0', 'displacement0')
            self.pressure0.rename('p0', 'pressure0')

            self.sys_v0 = dlf.Function(self.functionSpace)
            self.ufl_velocity0, _ = dlf.split(self.sys_v0)
            self.velocity0, _ = self.sys_v0.split(deepcopy=True)
            self.velocity0.rename('v0', 'velocity0')

            self.sys_a0 = dlf.Function(self.functionSpace)
            self.ufl_acceleration0, _ = dlf.split(self.sys_a0)
            self.acceleration0, _ = self.sys_a0.split(deepcopy=True)
            self.acceleration0.rename('a0', 'acceleration0')

            self.define_ufl_acceleration()

        initial_condition = self.config['formulation']['initial_condition']
        if initial_condition['displacement'] is not None:
            init_disp = initial_condition['displacement']
            self.apply_initial_conditions(init_disp,
                                          self.displacement,
                                          self.displacement0)
        if initial_condition['pressure'] is not None:
            init_pressure = initial_condition['pressure']
            self.apply_initial_conditions(init_pressure,
                                          self.pressure,
                                          self.pressure0)

        return None


    def define_compressible_functions(self):
        """
        Define functions necessary to formulate a compressible solid
        mechanics problem. The names of the member data added to the
        instance of the SolidMechanicsProblem class are:

        - sys_u = ufl_displacement = displacement : all point to the same
                  displacement function, unlike the incompressible case.
        - sys_du = trial_vector : trial function for vector function space
        - test_vector : sub component of mixed test function
        - ufl_pressure = pressure = None

        If problem is unsteady, the following are also added:

        - sys_v0 = ufl_velocity0 = velocity0
        - sys_a0 = ufl_acceleration0 = acceleration0
        - sys_u0 = ufl_displacement0 = displacement0


        """

        self.sys_u = self.ufl_displacement \
                     = self.displacement = dlf.Function(self.functionSpace)
        self.displacement.rename('u', 'displacement')
        self.ufl_pressure = self.pressure = 0
        self.ufl_pressure0 = self.pressure0 = 0
        self.test_vector = dlf.TestFunction(self.functionSpace)
        self.trial_vector = self.sys_du = dlf.TrialFunction(self.functionSpace)

        if self.config['formulation']['time']['unsteady']:
            self.sys_u0 = self.ufl_displacement0 \
                          = self.displacement0 = dlf.Function(self.functionSpace)
            self.displacement0.rename('u0', 'displacement0')

            self.sys_v0 = self.ufl_velocity0 \
                          = self.velocity0 = dlf.Function(self.functionSpace)
            self.velocity0.rename('v0', 'velocity0')

            self.sys_a0 = self.ufl_acceleration0 \
                          = self.acceleration0 = dlf.Function(self.functionSpace)
            self.acceleration0.rename('a0', 'acceleration0')

            self.define_ufl_acceleration()
        else:
            self.sys_u0 = self.ufl_displacement0 = self.displacement0 = 0
            self.sys_v0 = self.ufl_velocity0 = self.velocity0 = 0
            self.sys_a0 = self.ufl_acceleration0 = self.acceleration0 = 0

        initial_condition = self.config['formulation']['initial_condition']
        if initial_condition['displacement'] is not None:
            init_disp = initial_condition['displacement']
            self.apply_initial_conditions(init_disp,
                                          self.displacement,
                                          self.displacement0)

        return None


    def define_ufl_acceleration(self):
        """
        Define the acceleration based on the Newmark integration scheme
        and add as member data under 'ufl_acceleration'.

        """

        dt = self.config['formulation']['time']['dt']
        beta = self.config['formulation']['time']['beta']

        u = self.ufl_displacement
        u0 = self.ufl_displacement0
        v0 = self.ufl_velocity0
        a0 = self.ufl_acceleration0

        self.ufl_acceleration = 1.0/(beta*dt**2)*(u - u0 - dt*v0) \
                                - (1.0/(2.0*beta) - 1.0)*a0

        return None


    def define_deformation_tensors(self):
        """
        Define kinematic tensors needed for constitutive equations. Secondary
        tensors are added with the suffix "0" if the problem is time-dependent.
        The names of member data added to an instance of the SolidMechanicsProblem
        class are:

        - deformationGradient
        - deformationGradient0
        - jacobian
        - jacobian0


        """

        dim = self.mesh.geometry().dim()
        I = dlf.Identity(dim)
        self.deformationGradient = I + dlf.grad(self.ufl_displacement)
        self.jacobian = dlf.det(self.deformationGradient)

        if self.config['formulation']['time']['unsteady']:
            self.deformationGradient0 = I + dlf.grad(self.ufl_displacement0)
            self.jacobian0 = dlf.det(self.deformationGradient0)

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

        if isclass(self.config['material']['const_eqn']):
            mat_class = self.config['material']['const_eqn']
        elif self.config['material']['const_eqn'] == 'lin_elastic':
            mat_class = materials.solid_materials.LinearMaterial
        elif self.config['material']['const_eqn'] == 'neo_hookean':
            mat_class = materials.solid_materials.NeoHookeMaterial
        elif self.config['material']['const_eqn'] == 'fung':
            mat_class = materials.solid_materials.FungMaterial
        elif self.config['material']['const_eqn'] == 'guccione':
            mat_class = materials.solid_materials.GuccioneMaterial
        else:
            s = "The material '%s' has not been implemented. A class for such" \
                + " material must be provided."
            raise ValueError(s)

        try:
            fiber_file = self.config['mesh']['fiber_file']
        except KeyError:
            fiber_file = None
        self._material = mat_class(mesh=self.mesh,
                                   fiber_file=fiber_file,
                                   inverse=self.config['formulation']['inverse'],
                                   **self.config['material'])

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
        Define Dirichlet BCs for compressible solid mechanics problems.


        """

        if 'displacement' in self.config['formulation']['bcs']['dirichlet']:
            self.dirichlet_bcs = dict()
            dirichlet_dict = self.config['formulation']['bcs']['dirichlet']
            dirichlet_bcs = self.__define_displacement_bcs(self.functionSpace,
                                                           dirichlet_dict,
                                                           self.mesh_function)
            self.dirichlet_bcs.update(dirichlet_bcs)
        else:
            self.dirichlet_bcs = None

        return None


    def _define_incompressible_dirichlet_bcs(self):
        """
        Define Dirichlet BCs for incompressible solid mechanics problems.


        """

        self.dirichlet_bcs = dict()
        dirichlet_dict = self.config['formulation']['bcs']['dirichlet']
        if 'displacement' in self.config['formulation']['bcs']['dirichlet']:
            displacement_bcs = self.__define_displacement_bcs(self.functionSpace.sub(0),
                                                           dirichlet_dict,
                                                           self.mesh_function)
            self.dirichlet_bcs.update(displacement_bcs)

        if 'pressure' in self.config['formulation']['bcs']['dirichlet']:
            pressure_bcs = self.__define_pressure_bcs(self.functionSpace.sub(1),
                                                      dirichlet_dict,
                                                      self.mesh_function)
            self.dirichlet_bcs.update(pressure_bcs)

        if not self.dirichlet_bcs:
            self.dirichlet_bcs = None

        return None


    def define_forms(self):
        """
        Define all of the variational forms necessary for the problem specified
        by the user and add them as member data. The variational forms are those
        corresponding to the balance of linear momentum and the incompressibility
        constraint.


        """

        # Define UFL objects corresponding to the local acceleration
        # if problem is unsteady.
        self.define_ufl_local_inertia()

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
        rho = dlf.Constant(self.config['material']['density'])

        # Will need both of these terms if problem is unsteady
        self.ufl_local_inertia = dlf.dot(xi, rho*self.ufl_acceleration)
        self.ufl_local_inertia *= dlf.dx

        return None


    def define_ufl_stress_work(self):
        """
        Define the UFL object corresponding to the stress tensor term
        in the weak form.


        """

        stress_func = self._material.stress_tensor
        stress_tensor = stress_func(self.deformationGradient,
                                    self.jacobian,
                                    self.ufl_pressure)
        xi = self.test_vector
        self.ufl_stress_work = dlf.inner(dlf.grad(xi), stress_tensor)
        self.ufl_stress_work *= dlf.dx
        if self.config['formulation']['time']['unsteady']:
            stress_tensor0 = stress_func(self.deformationGradient0,
                                         self.jacobian0,
                                         self.ufl_pressure0)
            self.ufl_stress_work0 = dlf.inner(dlf.grad(xi), stress_tensor0)
            self.ufl_stress_work0 *= dlf.dx
        else:
            self.ufl_stress_work0 = 0

        return None


    def define_ufl_body_force(self):
        """
        Define the UFL object corresponding to the body force term in
        the weak form.


        """

        if self.config['formulation']['body_force'] is None:
            self.ufl_body_force = 0
            self.ufl_body_force0 = 0
            return None

        rho = self.config['material']['density']
        b = self.config['formulation']['body_force']
        xi = self.test_vector
        self.ufl_body_force = dlf.dot(xi, rho*b)*dlf.dx

        # Create a copy of the body force term to use at a different time step.
        if self.config['formulation']['time']['unsteady']:
            b0, = duplicate_expressions(b)
            self.ufl_body_force0 = dlf.dot(xi, rho*b0)*dlf.dx
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

        if self.config['formulation']['bcs']['neumann'] is None:
            self.ufl_neumann_bcs = 0
            self.ufl_neumann_bcs0 = 0
            return None

        regions = self.config['formulation']['bcs']['neumann']['regions']
        types = self.config['formulation']['bcs']['neumann']['types']
        values = self.config['formulation']['bcs']['neumann']['values']
        domain = self.config['formulation']['domain']

        define_ufl_neumann = BaseMechanicsProblem.define_ufl_neumann_form

        self.ufl_neumann_bcs = define_ufl_neumann(regions, types,
                                                  values, domain,
                                                  self.mesh,
                                                  self.mesh_function,
                                                  self.deformationGradient,
                                                  self.jacobian,
                                                  self.test_vector)
        if self.config['formulation']['time']['unsteady']:
            values0 = duplicate_expressions(*values)
            self.ufl_neumann_bcs0 = define_ufl_neumann(regions, types,
                                                       values0, domain,
                                                       self.mesh,
                                                       self.mesh_function,
                                                       self.deformationGradient0,
                                                       self.jacobian0,
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

        self.G1 = self.ufl_local_inertia \
                  + theta*(self.ufl_stress_work \
                           - self.ufl_body_force \
                           - self.ufl_neumann_bcs) \
                  + (1.0 - theta)*(self.ufl_stress_work0 \
                                   - self.ufl_body_force0 \
                                   - self.ufl_neumann_bcs0)

        if self.config['material']['incompressible']:
            q = self.test_scalar
            kappa = self._material._parameters['kappa']
            bvol = self._material.incompressibilityCondition(self.ufl_displacement)
            self.G2 = q*(kappa*bvol - self.ufl_pressure)*dlf.dx
        else:
            self.G2 = 0
        self.G = self.G1 + self.G2

        return None


    def define_ufl_equations_diff(self):
        """
        Differentiate all of the variational forms with respect to appropriate
        fields variables and add as member data.


        """

        self.dG = dlf.derivative(self.G, self.sys_u, self.sys_du)

        return None


    @staticmethod
    def __define_displacement_bcs(W, dirichlet_dict, mesh_function):
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
        mesh_function : dolfin.MeshFunction
            The mesh function used to tag different regions of the
            domain boundary.


        Returns
        -------

        displacement_bcs : dict
            a dictionary of the form {'displacement': [...]}, where
            the value is a list of dolfin.DirichletBC objects.


        """

        displacement_bcs = {'displacement': list()}
        disp_vals = dirichlet_dict['displacement']
        regions = dirichlet_dict['regions']
        for region, value in zip(regions, disp_vals):
            bc = dlf.DirichletBC(W, value, mesh_function, region)
            displacement_bcs['displacement'].append(bc)

        return displacement_bcs


    @staticmethod
    def __define_pressure_bcs(W, dirichlet_dict, mesh_function):
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
        mesh_function : dolfin.MeshFunction
            The mesh function used to tag different regions of the
            domain boundary.


        Returns
        -------

        pressure_bcs : dict
            a dictionary of the form {'pressure': [...]}, where
            the value is a list of dolfin.DirichletBC objects.


        """

        pressure_bcs = {'pressure': list()}
        p_vals = dirichlet_bcs['pressure']
        p_regions = dirichlet_dict['p_regions']
        for region, value in zip(p_regions, p_vals):
            bc = dlf.DirichletBC(W, value, mesh_function, region)
            pressure_bcs['pressure'].append(bc)

        return pressure_bcs


class SolidMechanicsSolver(dlf.NonlinearVariationalSolver):
    """
    This class is derived from the dolfin.NonlinearVariationalSolver to
    solve problems formulated with SolidMechanicsProblem. It passes the
    UFL variational forms to the solver, and loops through all time steps
    if the problem is unsteady. The solvers that are available through
    this class are the same as those available through dolfin. The user
    may use the helper function, 'set_parameters' to set the linear solver
    used, as well as the tolerances for iterative and nonlinear solves, or
    do it directly through the 'parameters' member.


    """


    def __init__(self, problem, fname_disp=None, fname_pressure=None):
        """
        Initialize a SolidMechanicsSolver object.

        Parameters
        ----------

        problem : SolidMechanicsProblem
            A SolidMechanicsProblem object that contains the necessary UFL
            forms to define and solve the problem specified in a config
            dictionary.
        fname_disp : str (default None)
            Name of the file series in which the displacement values are
            to be saved.
        fname_pressure : str (default None)
            Name of the file series in which the pressure values are to
            be saved.


        """

        self._problem = problem

        bcs = list()
        for val in problem.dirichlet_bcs.values():
            bcs.extend(val)

        dlf_problem = dlf.NonlinearVariationalProblem(problem.G, problem.sys_u,
                                                      bcs, J=problem.dG)

        dlf.NonlinearVariationalSolver.__init__(self, dlf_problem)

        if problem.config['material']['incompressible']:
            self.define_function_assigners()

        # Create file objects. This keeps the counter from being reset
        # each time the solve function is called.
        if fname_disp is not None:
            self._file_disp = dlf.File(fname_disp, "compressed")
        else:
            self._file_disp = None
        if fname_pressure is not None:
            self._file_pressure = dlf.File(fname_pressure, "compressed")
        else:
            self._file_pressure = None

        return None


    def set_parameters(self, linear_solver='default',
                       newton_abstol=1e-10,
                       newton_reltol=1e-9,
                       newton_maxIters=50,
                       krylov_abstol=1e-8,
                       krylov_reltol=1e-7,
                       krylov_maxIters=50):
        """
        Set the parameters used by the NonlinearVariationalSolver.


        Parameters
        ----------

        linear_solver : str
            The name of linear solver to be used.
        newton_abstol : float (default 1e-10)
            Absolute tolerance used to terminate Newton's method.
        newton_reltol : float (default 1e-9)
            Relative tolerance used to terminate Newton's method.
        newton_maxIters : int (default 50)
            Maximum number of iterations for Newton's method.
        krylov_abstol : float (default 1e-8)
            Absolute tolerance used to terminate Krylov solver methods.
        krylov_reltol : float (default 1e-7)
            Relative tolerance used to terminate Krylov solver methods.
        krylov_maxIters : int (default 50)
            Maximum number of iterations for Krylov solver methods.


        Returns
        -------

        None


        """

        param = self.parameters
        param['newton_solver']['linear_solver'] = linear_solver
        param['newton_solver']['absolute_tolerance'] = newton_abstol
        param['newton_solver']['relative_tolerance'] = newton_reltol
        param['newton_solver']['maximum_iterations'] = newton_maxIters
        param['newton_solver']['krylov_solver']['absolute_tolerance'] = krylov_abstol
        param['newton_solver']['krylov_solver']['relative_tolerance'] = krylov_reltol
        param['newton_solver']['krylov_solver']['maximum_iterations'] = krylov_maxIters

        return None


    def full_solve(self, save_freq=1, save_initial=True):
        """
        Solve the mechanics problem defined by SolidMechanicsProblem. If the
        problem is unsteady, this function will loop through the entire time
        interval using the parameters provided for the Newmark integration
        scheme.


        Parameters
        ----------

        save_freq : int (default 1)
            The frequency at which the solution is to be saved if the problem is
            unsteady. E.g., save_freq = 10 if the user wishes to save the solution
            every 10 time steps.
        save_initial : bool (default True)
            True if the user wishes to save the initial condition and False otherwise.


        Returns
        -------

        None


        """

        problem = self._problem
        rank = dlf.MPI.rank(dlf.mpi_comm_world())

        if problem.config['formulation']['time']['unsteady']:
            t, tf = problem.config['formulation']['time']['interval']
            t0 = t

            dt = problem.config['formulation']['time']['dt']
            count = 0 # Used to check if files should be saved.

            # Save initial condition
            if save_initial:
                if self._file_disp is not None:
                    self._file_disp << (problem.displacement, t)
                    if not rank:
                        print('* Displacement saved *')
                if self._file_pressure is not None:
                    self._file_pressure << (problem.pressure, t)
                    if not rank:
                        print('* Pressure saved *')

            # Hack to avoid rounding errors.
            while t < (tf - dt/10.0):

                # Advance the time.
                t += dt

                # Assign and update all vectors.
                self.update_assign()

                # Update expressions that depend on time.
                problem.update_time(t, t0)

                # Print the current time.
                if not rank:
                    print('*'*30)
                    print('t = %3.6f' % t)

                # Solver current time step.
                self.step()

                t0 = t
                count += 1

                # Save current time step
                if not count % save_freq:
                    if self._file_disp is not None:
                        self._file_disp << (problem.displacement, t)
                        if not rank:
                            print('* Displacement saved *')
                    if self._file_pressure is not None:
                        self._file_pressure << (problem.pressure, t)
                        if not rank:
                            print('* Pressure saved *')

        else:
            self.step()

            self.update_assign()

            if self._file_disp is not None:
                self._file_disp << self._problem.displacement
            if self._file_pressure is not None:
                self._file_pressure << self._problem.pressure

        return None


    def step(self):
        """
        Compute the solution for the next time step in the simulation. Note that
        there is only one "step" if the simulation is steady.


        """

        self.solve()

        return None


    def update_assign(self):
        """
        Update the values of the field variables -- both the current and
        previous time step in preparation for the next step of the simulation.


        """

        problem = self._problem
        incompressible = problem.config['material']['incompressible']
        unsteady = problem.config['formulation']['time']['unsteady']

        u = problem.displacement

        if unsteady:
            u0 = problem.displacement0
            v0 = problem.velocity0
            a0 = problem.acceleration0

            beta = problem.config['formulation']['time']['beta']
            gamma = problem.config['formulation']['time']['gamma']
            dt = problem.config['formulation']['time']['dt']

        if incompressible:
            p = problem.pressure
            self.assigner_sys2u.assign([u, p], problem.sys_u)

        if unsteady:
            self.update(u, u0, v0, a0, beta, gamma, dt)

        if incompressible and unsteady:
            p0 = problem.pressure0
            self.assigner_u02sys.assign(problem.sys_u0, [u0, p0])
            self.assigner_v02sys.assign(problem.sys_v0.sub(0), v0)
            self.assigner_a02sys.assign(problem.sys_a0.sub(0), a0)

        return None


    def define_function_assigners(self):
        """
        Create function assigners to update the current and previous time
        values of all field variables. This is specific to incompressible
        simulations since the mixed function space formulation requires the
        handling of the mixed functions and the copies of its subcomponents
        in a specific manner.


        """

        problem = self._problem
        W = problem.functionSpace
        u = problem.displacement
        p = problem.pressure

        self.assigner_sys2u = dlf.FunctionAssigner([u.function_space(),
                                                    p.function_space()], W)

        if problem.config['formulation']['time']['unsteady']:
            u0 = problem.displacement0
            p0 = problem.pressure0
            v0 = problem.velocity0
            a0 = problem.acceleration0

            self.assigner_u02sys = dlf.FunctionAssigner(W, [u0.function_space(),
                                                            p0.function_space()])
            self.assigner_v02sys = dlf.FunctionAssigner(W.sub(0), v0.function_space())
            self.assigner_a02sys = dlf.FunctionAssigner(W.sub(0), a0.function_space())

        return None


    @staticmethod
    def update(u, u0, v0, a0, beta, gamma, dt):
        """
        Function to update values of field variables at the current and previous
        time steps based on the Newmark integration scheme:

        a = 1.0/(beta*dt^2)*(u - u0 - v0*dt) - (1.0/(2.0*beta) - 1.0)*v0
        v = dt*((1.0 - gamma)*a0 + gamma*a) + v0

        This particular method is to be used when the function objects do not
        derive from a mixed function space.


        Parameters
        ----------

        u : dolfin.Function
            Object storing the displacement at the current time step.
        u0 : dolfin.Function
            Object storing the displacement at the previous time step.
        v0 : dolfin.Function
            Object storing the velocity at the previous time step.
        a0 : dolfin.Function
            Object storing the acceleration at the previous time step.
        beta : float
            Scalar parameter for the family of Newmark integration schemes.
            See equation above.
        gamma : float
            Scalar parameter for the family of Newmark integration schemes.
            See equation above.
        dt : float
            Time step used to advance through the entire time interval.


        Returns
        -------

        None


        """

        # Get vector references
        u_vec, u0_vec = u.vector(), u0.vector()
        v0_vec, a0_vec = v0.vector(), a0.vector()

        # Update acceleration and velocity
        a_vec = 1.0/(beta*dt**2)*(u_vec - u0_vec - v0_vec*dt) \
                                  - (1.0/(2.0*beta) - 1.0)*a0_vec
        v_vec = dt*((1.0 - gamma)*a0_vec + gamma*a_vec) + v0_vec

        v0.vector()[:], a0.vector()[:] = v_vec, a_vec
        u0.vector()[:] = u_vec

        return None
