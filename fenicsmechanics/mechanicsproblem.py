import dolfin as dlf

from . import materials
from .utils import duplicate_expressions
from .basemechanicsproblem import BaseMechanicsProblem
from .exceptions import *

from inspect import isclass

__all__ = ['MechanicsProblem']


class MechanicsProblem(BaseMechanicsProblem):
    """
    This class represents the variational form of a continuum
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

        BaseMechanicsProblem.class_name = "MechanicsProblem"
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
        Define the vector (and scalar if incompressible) function spaces
        based on the degrees specified in config['formulation']['element'],
        and add them to the instance of MechanicsProblem as member data. If
        the material is not incompressible, the scalar function space is
        set to None.


        """

        cell = self.mesh.ufl_cell()
        vec_degree = int(self.config['formulation']['element'][0][-1])
        if vec_degree == 0:
            vec_family = "DG"
        else:
            vec_family = "CG"
        vec_element = dlf.VectorElement(vec_family, cell, vec_degree)
        self.vectorSpace = dlf.FunctionSpace(self.mesh, vec_element)

        if self.config['material']['incompressible']:
            scalar_degree = int(self.config['formulation']['element'][1][-1])
            if scalar_degree == 0:
                scalar_family = "DG"
            else:
                scalar_family = "CG"
            scalar_element = dlf.FiniteElement(scalar_family, cell, scalar_degree)
            self.scalarSpace = dlf.FunctionSpace(self.mesh, scalar_element)
        else:
            self.scalarSpace = None

        return None


    def define_functions(self):
        """
        Define the vector and scalar functions necessary to define the
        problem specified in the 'config' dictionary. Functions that are
        not needed are set to 0. This method calls a method to define
        the vector functions and another for the scalar functions.


        """

        self.define_vector_functions()
        self.define_scalar_functions()

        return None


    def define_vector_functions(self):
        """
        Define the vector functions necessary to define the problem
        specified in the 'config' dictionary. If the material is elastic,
        displacement and velocity functions are defined. Secondary
        displacement and velocity functions are defined with the suffix
        "0" if the problem is time-dependent to store the previous time
        step. Functions that are not needed are set to 0. The trial and
        test functions for the vector function space are also defined by
        this function. The names of the member data added to the instance
        of the MechanicsProblem class are:

        - :code:`test_vector`
        - :code:`trial_vector`
        - :code:`displacement`
        - :code:`displacement0`
        - :code:`velocity`
        - :code:`velocity0`


        """

        # Exit if functions have already been defined.
        # A function decorator might work better here...
        if hasattr(self, 'velocity'):
            return None

        unsteady = self.config['formulation']['time']['unsteady']
        lagrangian = self.config['formulation']['domain'] == 'lagrangian'
        lin_elastic = self.config['material']['const_eqn'] == 'lin_elastic'
        elastic = self.config['material']['type'] == 'elastic'

        init = self.config['formulation']['initial_condition']

        # Trial and test functions
        self.test_vector = dlf.TestFunction(self.vectorSpace)
        self.trial_vector = dlf.TrialFunction(self.vectorSpace)

        if elastic and unsteady:
            if init['displacement'] is not None:
                disp = init['displacement']
                self.displacement = dlf.project(disp, self.vectorSpace)
                self.displacement0 = self.displacement.copy(deepcopy=True)
            else:
                self.displacement = dlf.Function(self.vectorSpace)
                self.displacement0 = dlf.Function(self.vectorSpace)
            self.displacement.rename("u", "displacement")
            self.displacement0.rename("u0", "displacement")

            if init['velocity'] is not None:
                vel = init['velocity']
                self.velocity = dlf.project(vel, self.vectorSpace)
                self.velocity0 = self.velocity.copy(deepcopy=True)
            else:
                self.velocity = dlf.Function(self.vectorSpace)
                self.velocity0 = dlf.Function(self.vectorSpace)
            self.velocity.rename("v", "velocity")
            self.velocity0.rename("v0", "velocity")
        elif unsteady: # Unsteady viscous material.
            self.displacement = 0
            self.displacement0 = 0

            if init['velocity'] is not None:
                vel = init['velocity']
                self.velocity = dlf.project(vel, self.vectorSpace)
                self.velocity0 = self.velocity.copy(deepcopy=True)
            else:
                self.velocity = dlf.Function(self.vectorSpace)
                self.velocity0 = dlf.Function(self.vectorSpace)
            self.velocity.rename("v", "velocity")
            self.velocity0.rename("v0", "velocity")

            # self.velocity = dlf.Function(self.vectorSpace, name="v")
            # self.velocity0 = dlf.Function(self.vectorSpace, name="v0")
        elif elastic: # Steady elastic material.
            if init['displacement'] is not None:
                disp = init['displacement']
                self.displacement = dlf.project(disp, self.vectorSpace)
                # self.displacement0 = self.displacement.copy(deepcopy=True)
            else:
                self.displacement = dlf.Function(self.vectorSpace)
                # self.displacement0 = dlf.Function(self.vectorSpace)
            self.displacement.rename("u", "displacement")
            # self.displacement0.rename("u0", "displacement")

            # self.displacement = dlf.Function(self.vectorSpace, name="u")
            self.displacement0 = 0
            self.velocity = 0
            self.velocity0 = 0
        else: # Steady viscous material
            self.displacement = 0
            self.displacement0 = 0

            if init['velocity'] is not None:
                vel = init['velocity']
                self.velocity = dlf.project(vel, self.vectorSpace)
                # self.velocity0 = self.velocity.copy(deepcopy=True)
            else:
                self.velocity = dlf.Function(self.vectorSpace)
                # self.velocity0 = dlf.Function(self.vectorSpace)
            self.velocity.rename("v", "velocity")
            # self.velocity0.rename("v0", "velocity")

            # self.velocity = dlf.Function(self.vectorSpace, name="v")
            self.velocity0 = 0

        # # Apply initial conditions if provided
        # initial_condition = self.config['formulation']['initial_condition']
        # if initial_condition['displacement'] is not None:
        #     init_disp = initial_condition['displacement']
        #     self.apply_initial_conditions(init_disp,
        #                                   self.displacement,
        #                                   self.displacement0)
        # if initial_condition['velocity'] is not None:
        #     init_vel = initial_condition['velocity']
        #     self.apply_initial_conditions(init_vel,
        #                                   self.velocity,
        #                                   self.velocity0)

        return None


    def define_scalar_functions(self):
        """
        Define the pressure function(s) necessary to define the problem
        specified in the 'config' dictionary. If the problem is not
        specified as incompressible, the scalar functions are set to 0.
        A secondary pressure function with the suffix "0" is also added
        if the problem is time-dependent to store the pressure values at
        the previous time step. The names of the member data added to an
        instance of the MechanicsProblem class are:

        - :code:`test_scalar`
        - :code:`trial_scalar`
        - :code:`pressure`
        - :code:`pressure0`


        """

        # Exit if functions have already been defined.
        # A function decorator might work better here...
        if hasattr(self, 'pressure'):
            return None

        if self.config['material']['incompressible']:
            self.pressure = dlf.Function(self.scalarSpace, name='p')

            if self.config['formulation']['time']['unsteady']:
                self.pressure0 = dlf.Function(self.scalarSpace, name='p0')
            else:
                self.pressure0 = 0

            self.test_scalar = dlf.TestFunction(self.scalarSpace)
            self.trial_scalar = dlf.TrialFunction(self.scalarSpace)
        else:
            self.pressure = 0
            self.pressure0 = 0
            self.test_scalar = None
            self.trial_scalar = None

        # Apply initial conditions if provided
        initial_condition = self.config['formulation']['initial_condition']
        if initial_condition['pressure'] is not None:
            init_pressure = initial_condition['pressure']
            self.apply_initial_conditions(init_pressure,
                                          self.pressure,
                                          self.pressure0)

        return None


    def define_deformation_tensors(self):
        """
        Define kinematic tensors needed for constitutive equations. Tensors
        that are irrelevant to the current problem are set to 0, e.g. the
        deformation gradient is set to 0 when simulating fluid flow. Secondary
        tensors are added with the suffix "0" if the problem is time-dependent.
        The names of member data added to an instance of the MechanicsProblem
        class are:

        - :code:`deformationGradient`
        - :code:`deformationGradient0`
        - :code:`velocityGradient`
        - :code:`velocityGradient0`
        - :code:`jacobian`
        - :code:`jacobian0`


        """

        # Exit function if tensors have already been defined.
        if hasattr(self, 'deformationGradient') \
           or hasattr(self, 'deformationRateGradient'):
            return None

        # Checks the type of material to determine which deformation
        # tensors to define.
        if self.config['material']['type'] == 'elastic':
            I = dlf.Identity(self.mesh.geometry().dim())
            self.deformationGradient = I + dlf.grad(self.displacement)
            self.jacobian = dlf.det(self.deformationGradient)

            if self.config['formulation']['time']['unsteady']:
                self.velocityGradient = dlf.grad(self.velocity)
                self.deformationGradient0 = I + dlf.grad(self.displacement0)
                self.velocityGradient0 = dlf.grad(self.velocity0)
                self.jacobian0 = dlf.det(self.deformationGradient0)
            else:
                self.velocityGradient = 0
                self.deformationGradient0 = 0
                self.velocityGradient0 = 0
                self.jacobian0 = 0
        else:
            self.deformationGradient = 0
            self.deformationGradient0 = 0
            self.jacobian = 0
            self.jacobian0 = 0

            self.velocityGradient = dlf.grad(self.velocity)
            if self.config['formulation']['time']['unsteady']:
                self.velocityGradient0 = dlf.grad(self.velocity0)
            else:
                self.velocityGradient0 = 0

        return None


    def define_material(self):
        """
        Create an instance of the class that defining the constitutive
        equation for the current problem and add it as member data under
        '_material'. All necessary parameters must be included in the
        'material' subdictionary of 'config'. The specific values necessary
        depends on the constitutive equation used. Please check the
        documentation of the material classes provided in
        'fenicsmechanics.materials' if using a built-in material.


        """

        # Check which class should be called.
        const_eqn = self.config['material']['const_eqn']
        if isclass(const_eqn):
            mat_class = self.config['material']['const_eqn']
        elif const_eqn == 'lin_elastic':
            mat_class = materials.solid_materials.LinearIsoMaterial
        elif const_eqn == 'neo_hookean':
            mat_class = materials.solid_materials.NeoHookeMaterial
        elif const_eqn == 'fung':
            mat_class = materials.solid_materials.FungMaterial
        elif const_eqn == 'guccione':
            mat_class = materials.solid_materials.GuccioneMaterial
        elif const_eqn == 'newtonian' or const_eqn == 'stokes':
            mat_class = materials.fluids.NewtonianFluid
        else:
            raise InvalidCombination("Shouldn't be in here...")

        # Create an instance of the material class and store
        # as member data.
        try:
            inverse = self.config['formulation']['inverse']
        except KeyError:
            inverse = False
        self._material = mat_class(inverse=inverse,
                                   **self.config['material'])

        return None


    def define_dirichlet_bcs(self):
        """
        Define a list of Dirichlet BC objects based on the problem configuration
        provided by the user, and add it as member data under 'dirichlet_bcs'. If
        no Dirichlet BCs are provided, 'dirichlet_bcs' is set to None.


        """

        # Don't redefine if object already exists.
        if hasattr(self, 'dirichlet_bcs'):
            return None

        # Exit function if no Dirichlet BCs were provided.
        if self.config['formulation']['bcs']['dirichlet'] is None:
            self.dirichlet_bcs = None
            return None

        V = self.vectorSpace
        S = self.scalarSpace

        if 'velocity' in self.config['formulation']['bcs']['dirichlet']:
            vel_vals = self.config['formulation']['bcs']['dirichlet']['velocity']
        else:
            vel_vals = None

        if 'displacement' in self.config['formulation']['bcs']['dirichlet']:
            disp_vals = self.config['formulation']['bcs']['dirichlet']['displacement']
        else:
            disp_vals = None

        if 'pressure' in self.config['formulation']['bcs']['dirichlet'] \
           and 'p_regions' in self.config['formulation']['bcs']['dirichlet']:
            pressure_vals = self.config['formulation']['bcs']['dirichlet']['pressure']
            p_regions = self.config['formulation']['bcs']['dirichlet']['p_regions']
        elif 'pressure' in self.config['formulation']['bcs']['dirichlet']:
            s = "Values for pressure were specified, but the regions were not."
            raise RequiredParameter(s)
        elif 'p_regions' in self.config['formulation']['bcs']['dirichlet']:
            s = "The regions for pressure were specified, but the values were not."
            raise RequiredParameter(s)
        else:
            pressure_vals = None
            p_regions = None

        # Regions for displacement and velocity
        regions = self.config['formulation']['bcs']['dirichlet']['regions']

        self.dirichlet_bcs = {'displacement': None, 'velocity': None, 'pressure': None}

        # Store the Dirichlet BCs for the velocity vector field.
        if vel_vals is not None:
            self.dirichlet_bcs['velocity'] = list()
            for region, value in zip(regions, vel_vals):
                bc = dlf.DirichletBC(V, value, self.boundaries, region)
                self.dirichlet_bcs['velocity'].append(bc)

        # Store the Dirichlet BCs for the displacement vector field.
        if disp_vals is not None:
            self.dirichlet_bcs['displacement'] = list()
            for region, value in zip(regions, disp_vals):
                bc = dlf.DirichletBC(V, value, self.boundaries, region)
                self.dirichlet_bcs['displacement'].append(bc)

        # Store the Dirichlet BCs for the pressure scalar field.
        if pressure_vals is not None:
            self.dirichlet_bcs['pressure'] = list()
            for region, value in zip(p_regions, pressure_vals):
                bc = dlf.DirichletBC(S, value, self.boundaries, region)
                self.dirichlet_bcs['pressure'].append(bc)

        # Remove pressure item if material is not incompressible.
        if not self.config['material']['incompressible']:
            _ = self.dirichlet_bcs.pop('pressure')

        # Remove displacement item if material is not elastic.
        if self.config['material']['type'] != 'elastic':
            _ = self.dirichlet_bcs.pop('displacement')

        # Remove velocity item if material is steady elastic.
        if not self.config['formulation']['time']['unsteady'] \
           and self.config['material']['type'] == 'elastic':
            _ = self.dirichlet_bcs.pop('velocity')

        # If dictionary is empty, replace with None.
        if self.dirichlet_bcs == {}:
            self.dirichlet_bcs = None

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
        self.define_ufl_local_inertia_diff()

        # Define UFL objects corresponding to the convective acceleration
        # if problem is formulated with respect to Eulerian coordinates
        self.define_ufl_convec_accel()
        self.define_ufl_convec_accel_diff()

        # Define UFL objects corresponding to the stress tensor term.
        # This should always be non-zero for deformable bodies.
        self.define_ufl_stress_work()
        self.define_ufl_stress_work_diff()

        # Define UFL object corresponding to the body force term. Assume
        # it is zero if key was not provided.
        self.define_ufl_body_force()

        # Define UFL object corresponding to the traction force terms. Assume
        # it is zero if key was not provided.
        self.define_ufl_neumann_bcs()
        self.define_ufl_neumann_bcs_diff()

        return None


    def define_ufl_neumann_bcs(self):
        """
        Define the variational forms for all of the Neumann BCs given
        in the 'config' dictionary under "ufl_neumann_bcs". If the problem
        is time-dependent, a secondary variational form is defined at the
        previous time step with the name "ufl_neumann_bcs0".


        """

        # Exit function if already defined.
        if hasattr(self, 'ufl_neumann_bcs'):
            return None

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

        self.ufl_neumann_bcs = self.define_ufl_neumann_form(regions, types,
                                                            values, domain,
                                                            self.mesh,
                                                            self.boundaries,
                                                            self.deformationGradient,
                                                            self.jacobian,
                                                            self.test_vector)
        if self.config['formulation']['time']['unsteady']:
            values0 = duplicate_expressions(*values)
            self.ufl_neumann_bcs0 = self.define_ufl_neumann_form(regions, types,
                                                                 values0, domain,
                                                                 self.mesh,
                                                                 self.boundaries,
                                                                 self.deformationGradient0,
                                                                 self.jacobian0,
                                                                 self.test_vector)
        else:
            self.ufl_neumann_bcs0 = 0

        return None


    def define_ufl_neumann_bcs_diff(self):
        """
        Define the derivative(s) of the variational form of the Neumann BCs
        and add them as member data.


        """

        if hasattr(self, 'ufl_neumann_bcs_diff'):
            return None

        # Note that "is not" only works for integers 0 to 255.
        neum_not_zero = self.ufl_neumann_bcs is not 0
        disp_not_zero = self.displacement is not 0

        if neum_not_zero and disp_not_zero:
            self.ufl_neumann_bcs_du = dlf.derivative(self.ufl_neumann_bcs,
                                                     self.displacement,
                                                     self.trial_vector)
        else:
            self.ufl_neumann_bcs_du = 0

        return None


    def define_ufl_local_inertia(self):
        """
        Define the UFL object corresponding to the local acceleration
        term in the weak form. The function exits if it has already
        been defined.


        """

        # Exit if form is already defined!
        if hasattr(self, 'ufl_local_inertia'):
            return None

        # Set to None and exit if problem is steady.
        if not self.config['formulation']['time']['unsteady']:
            self.ufl_local_inertia = 0
            self.ufl_local_inertia0 = 0
            return None

        xi = self.test_vector
        rho = self.config['material']['density']

        # Will need both of these terms if problem is unsteady
        self.ufl_local_inertia = dlf.dot(xi, rho*self.velocity)*dlf.dx
        self.ufl_local_inertia0 = dlf.dot(xi, rho*self.velocity0)*dlf.dx

        return None


    def define_ufl_local_inertia_diff(self):
        """
        Define the UFL object that describes the matrix that results
        from taking the Gateaux derivative of the local acceleration
        term in the weak form. The function exits if it has already
        been defined.


        """

        if hasattr(self, 'ufl_local_inertia_dv'):
            return None

        if not self.config['formulation']['time']['unsteady']:
            self.ufl_local_inertia_dv = 0
            return None

        xi = self.test_vector
        dv = self.trial_vector
        rho = self.config['material']['density']

        self.ufl_local_inertia_dv = dlf.dot(xi, rho*dv)*dlf.dx

        return None


    def define_ufl_convec_accel(self):
        """
        Define the UFL object corresponding to the convective acceleration
        term in the weak form. The function exits if it has already been
        defined.


        """

        # Exit if attribute has already been defined.
        if hasattr(self, 'ufl_convec_accel'):
            return None

        # Exit if problem is formulated with respect to Eulerian
        # coordinates and is not an elastic material.
        eulerian = self.config['formulation']['domain'] == 'eulerian'
        lin_elastic = self.config['material']['const_eqn'] == 'lin_elastic'
        stokes = self.config['material']['const_eqn'] == 'stokes'

        if (not eulerian) or lin_elastic or stokes:
            self.ufl_convec_accel = 0
            self.ufl_convec_accel0 = 0
            return None

        xi = self.test_vector
        rho = self.config['material']['density']
        self.ufl_convec_accel = dlf.dot(xi, rho*dlf.grad(self.velocity) \
                                        *self.velocity)*dlf.dx
        if self.velocity0 is not 0:
            self.ufl_convec_accel0 = dlf.dot(xi, rho*dlf.grad(self.velocity0) \
                                             *self.velocity0)*dlf.dx
        else:
            self.ufl_convec_accel0 = 0

        return None


    def define_ufl_convec_accel_diff(self):
        """
        Define the UFL object corresponding to the Gateaux derivative of
        the convective acceleration term in the weak form. The function
        exits if it has already been defined.


        """

        if hasattr(self, 'ufl_convec_accel_dv'):
            return None

        # Exit if problem is formulated with respect to Eulerian
        # coordinates and is not an elastic material.
        eulerian = self.config['formulation']['domain'] == 'eulerian'
        lin_elastic = self.config['material']['const_eqn'] == 'lin_elastic'
        stokes = self.config['material']['const_eqn'] == 'stokes'
        if (not eulerian) or lin_elastic or stokes:
            self.ufl_convec_accel_dv = 0
            return None

        self.ufl_convec_accel_dv = dlf.derivative(self.ufl_convec_accel,
                                                  self.velocity,
                                                  self.trial_vector)

        return None


    def define_ufl_stress_work(self):
        """
        Define the UFL object corresponding to the stress tensor term
        in the weak form. The function exits if it has already been
        defined.


        """

        if hasattr(self, 'ufl_stress_work'):
            return None

        # THIS NEEDS TO BE GENERALIZED.
        if self.config['material']['type'] == 'elastic':
            # stress_tensor = self._material.stress_tensor(self.displacement,
            #                                              self.pressure)
            stress_tensor = self._material.stress_tensor(self.deformationGradient,
                                                         self.jacobian,
                                                         self.pressure)
        else:
            stress_tensor = self._material.stress_tensor(self.velocityGradient,
                                                         self.pressure)

        xi = self.test_vector
        self.ufl_stress_work = dlf.inner(dlf.grad(xi), stress_tensor)*dlf.dx
        if self.config['formulation']['time']['unsteady']:
            if self.config['material']['type'] == 'elastic':
                stress_tensor0 = self._material.stress_tensor(self.deformationGradient0,
                                                              self.jacobian0,
                                                              self.pressure0)
            else:
                stress_tensor0 = self._material.stress_tensor(self.velocityGradient,
                                                              self.pressure)
            self.ufl_stress_work0 = dlf.inner(dlf.grad(xi), stress_tensor0)*dlf.dx
        else:
            self.ufl_stress_work0 = 0

        return None


    def define_ufl_stress_work_diff(self):
        """
        Define the UFL object corresponding to the Gateaux derivative of
        the stress tensor term in the weak form. The function exits if it
        has already been defined.


        """

        if hasattr(self, 'ufl_stress_work_diff'):
            return None

        if self.displacement != 0:
            # Derivative of stress term w.r.t. to displacement.
            self.ufl_stress_work_du = dlf.derivative(self.ufl_stress_work,
                                                     self.displacement,
                                                     self.trial_vector)
        else:
            self.ufl_stress_work_du = 0

        if self.velocity != 0:
            self.ufl_stress_work_dv = dlf.derivative(self.ufl_stress_work,
                                                     self.velocity,
                                                     self.trial_vector)
        else:
            self.ufl_stress_work_dv = 0

        if self.pressure != 0:
            self.ufl_stress_work_dp = dlf.derivative(self.ufl_stress_work,
                                                     self.pressure,
                                                     self.trial_scalar)
        else:
            self.ufl_stress_work_dp = 0

        return None


    def define_ufl_body_force(self):
        """
        Define the UFL object corresponding to the body force term in
        the weak form. The function exits if it has already been defined.


        """

        if hasattr(self, 'ufl_body_force'):
            return None

        # Set to None and exit if key is not in config dictionary.
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
            b0 = dlf.Expression(b.cppcode, t=0.0,
                                element=self.vectorSpace.ufl_element())
            self.ufl_body_force0 = dlf.dot(xi, rho*b0)*dlf.dx
        else:
            self.ufl_body_force0 = 0

        return None


    def define_ufl_equations(self):
        """
        Define all of the variational forms necessary for the problem
        specified in the 'config' dictionary. This function calls other
        functions to define the velocity, momentum, and incompressibility
        equations.


        """

        self.define_ufl_velocity_equation()
        self.define_ufl_momentum_equation()
        self.define_ufl_incompressibility_equation()

        return None


    def define_ufl_velocity_equation(self):
        """
        Define the variational form for the reduction of order equation
        (equation that relates the velocity and displacement) and add as
        member data. Note that this is only necessary for time-dependent
        elastic materials. If this form is not necessary, it is set to 0.


        """

        if hasattr(self, 'f1'):
            return None

        if self.config['material']['type'] == 'viscous':
            self.f1 = 0
            return None

        if not self.config['formulation']['time']['unsteady']:
            self.f1 = 0
            return None

        theta = self.config['formulation']['time']['theta']
        dt = self.config['formulation']['time']['dt']
        f1 = self.displacement - self.displacement0 \
             - dt*(theta*self.velocity + (1.0 - theta)*self.velocity0)

        self.f1 = dlf.dot(self.test_vector, f1)*dlf.dx

        return None


    def define_ufl_momentum_equation(self):
        """
        Define the variational form corresponding to the balance of
        linear momentum and add as member data.


        """

        if hasattr(self, 'f2'):
            return None

        theta = self.config['formulation']['time']['theta']
        dt = self.config['formulation']['time']['dt']

        self.f2 = self.ufl_local_inertia - self.ufl_local_inertia0
        self.f2 += dt*theta*(self.ufl_convec_accel + self.ufl_stress_work \
                             - self.ufl_neumann_bcs - self.ufl_body_force)
        self.f2 += dt*(1.0 - theta)*(self.ufl_convec_accel0 + self.ufl_stress_work0 \
                                     - self.ufl_neumann_bcs0 - self.ufl_body_force0)

        return None


    def define_ufl_incompressibility_equation(self):
        """
        Define the variational form corresponding to the incompressibility
        constraint and add as member data.

        """

        if hasattr(self, 'f3'):
            return None

        if not self.config['material']['incompressible']:
            self.f3 = 0
            return None

        if self.config['material']['type'] == 'elastic':
            b_vol = self._material.incompressibilityCondition(self.displacement)
        else:
            b_vol = self._material.incompressibilityCondition(self.velocity)

        if self.config['material']['type'] == 'elastic':
            kappa = self._material._parameters['kappa']
            self.f3 = self.test_scalar*(kappa*b_vol - self.pressure)*dlf.dx
        else:
            self.f3 = self.test_scalar*b_vol*dlf.dx

        return None


    def define_ufl_equations_diff(self):
        """
        Differentiate all of the variational forms with respect to appropriate
        fields variables and add as member data.


        """

        # Derivatives of velocity integration equation.
        if self.f1 != 0:
            self.df1_du = dlf.derivative(self.f1, self.displacement, self.trial_vector)
            self.df1_dv = dlf.derivative(self.f1, self.velocity, self.trial_vector)
        else:
            self.df1_du = 0
            self.df1_dv = 0
        self.df1_dp = 0 # This is always zero.

        # Derivatives of momentum equation.
        if self.displacement != 0:
            self.df2_du = dlf.derivative(self.f2, self.displacement, self.trial_vector)
        else:
            self.df2_du = 0

        if self.velocity != 0:
            self.df2_dv = dlf.derivative(self.f2, self.velocity, self.trial_vector)
        else:
            self.df2_dv = 0

        if self.pressure != 0:
            self.df2_dp = dlf.derivative(self.f2, self.pressure, self.trial_scalar)
        else:
            self.df2_dp = 0

        # Derivatives of incompressibility equation.
        if self.f3 != 0:
            if self.displacement != 0:
                self.df3_du = dlf.derivative(self.f3, self.displacement, self.trial_vector)
            else:
                self.df3_du = 0

            if self.velocity != 0:
                self.df3_dv = dlf.derivative(self.f3, self.velocity, self.trial_vector)
            else:
                self.df3_dv = 0

            self.df3_dp = dlf.derivative(self.f3, self.pressure, self.trial_scalar)
        else:
            self.df3_du = 0
            self.df3_dv = 0
            self.df3_dp = 0

        return None
