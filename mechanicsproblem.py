import dolfin as dlf

from . import materials
from .utils import duplicate_expressions
from .basemechanicsproblem import BaseMechanicsProblem

from inspect import isclass

__all__ = ['MechanicsProblem']


class MechanicsProblem(BaseMechanicsProblem):
    """
    This class represents the variational form of a continuum
    mechanics problem. The user The specific form and boundary
    conditions are generated based on definition provided by the
    user in a dictionary of sub-dictionaries. The keys for the
    dictionary and their descriptions are shown below:

    * 'material'
       * 'type' : str
            The class of material that will be used, e.g.
            elastic, viscous, viscoelastic, etc. The name
            must match the name of a module inside of the
            fenicsmechanics.materials sub-package.
       * 'const_eqn' : str
            The name of the constitutive equation to be
            used. The name must match a function name in
            the fenicsmechanics.materials subpackage, which
            returns the stress tensor.
       * 'incompressible' : bool
            True if the material is incompressible. An
            additional weak form for the incompressibility
            constraint will be added to the problem.
       * 'density' : float, int, dolfin.Constant
            Scalar specifying the density of the material.
       * 'la' : float, int, dolfin.Constant
            Scalar value used in constitutive equations. E.g.,
            it is the first Lame parameter in linear elasticity.
       * 'mu' : float, int, dolfin.Constant
            Scalar value used in constitutive equations. E.g.,
            it is the second Lame parameter in linear elasticity,
            and the dynamic viscosity for a Newtonian fluid.
       * 'kappa' : float, int, dolfin.Constant
            Scalar value of the penalty parameter for incompressibility.

    * 'mesh'
       * 'mesh_file' : str
            Name of the file containing the mesh information of
            the problem geometry.
       * 'mesh_function' : str
            Name of the file containing the mesh function information
            of the geometry. A mesh function is typically used to
            label different regions of the mesh.
       * 'element' : str
            Name of the finite element to be used for the discrete
            function space. E.g., 'p2-p1'.

    * 'formulation'
        * 'time'
            * 'unsteady' : bool
                True if the problem is time dependent.
            * 'integrator' : str
                Name of the time integrating scheme to be used.
            * 'dt' : float, dolfin.Constant
                Time step used for the numerical integrator.
            * 'interval' : list, tuple
                A list or tuple of length 2 specifying the time interval,
                i.e. t0 and tf.
        * 'initial_condition' : subclass of dolfin.Expression
            An expression specifying the initial value of the
            solution to the problem.
        * 'domain' : str
            String specifying whether the problem is to be formulated
            in terms of Lagrangian, Eulerian, or ALE coordinates.
        * 'inverse' : bool
            True if the problem is an inverse elastostatics problem.
        * 'body_force' : dolfin.Expression, dolfin.Constant
            Value of the body force throughout the body.
        * 'bcs'
            * 'dirichlet'
                * 'velocity' : list, tuple
                    List of velocity values (dolfin.Constant or dolfin.Expression)
                    for each Dirichlet boundary region specified. The order must
                    match the order used in the list of region IDs.
                * 'displacement' : list, tuple
                    List of displacement values (dolfin.Constant or dolfin.Expression)
                    for each Dirichlet boundary region specified. The order must match
                    the order used in the list of region IDs.
                * 'regions' : list, tuple
                    List of the region IDs (int) on which Dirichlet
                    boundary conditions are to be imposed. These IDs
                    must match those used by the mesh function provided.
                    The order must match the order used in the list of
                    values.
            * 'neumann'
                * 'regions' : list, tuple
                    List of the region IDs (int) on which Neumann
                    boundary conditions are to be imposed. These IDs
                    must match those used by the mesh function provided.
                    The order must match the order used in the list of
                    types and values.
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


        """

        vec_element = dlf.VectorElement('CG', self.mesh.ufl_cell(),
                                        int(self.config['mesh']['element'][0][-1]))
        self.vectorSpace = dlf.FunctionSpace(self.mesh, vec_element)

        if self.config['material']['incompressible']:
            scalar_element = dlf.FiniteElement('CG', self.mesh.ufl_cell(),
                                               int(self.config['mesh']['element'][1][-1]))
            self.scalarSpace = dlf.FunctionSpace(self.mesh, scalar_element)
        else:
            self.scalarSpace = None

        return None


    def define_functions(self):
        """


        """

        self.define_vector_functions()
        self.define_scalar_functions()

        return None


    def define_vector_functions(self):
        """


        """

        # Exit if functions have already been defined.
        # A function decorator might work better here...
        if hasattr(self, 'velocity'):
            return None

        unsteady = self.config['formulation']['time']['unsteady']
        lagrangian = self.config['formulation']['domain'] == 'lagrangian'
        lin_elastic = self.config['material']['const_eqn'] == 'lin_elastic'
        elastic = self.config['material']['type'] == 'elastic'

        # Trial and test functions
        self.test_vector = dlf.TestFunction(self.vectorSpace)
        self.trial_vector = dlf.TrialFunction(self.vectorSpace)

        if elastic and unsteady:
            self.displacement = dlf.Function(self.vectorSpace, name="u")
            self.displacement0 = dlf.Function(self.vectorSpace, name="u0")
            self.velocity = dlf.Function(self.vectorSpace, name="v")
            self.velocity0 = dlf.Function(self.vectorSpace, name="v0")
        elif unsteady: # Unsteady viscous material.
            self.displacement = 0
            self.displacement0 = 0
            self.velocity = dlf.Function(self.vectorSpace)
            self.velocity0 = dlf.Function(self.vectorSpace)
        elif elastic: # Steady elastic material.
            self.displacement = dlf.Function(self.vectorSpace, name="u")
            self.displacement0 = 0
            self.velocity = 0
            self.velocity0 = 0
        else: # Steady viscous material
            self.displacement = 0
            self.displacement0 = 0
            self.velocity = dlf.Function(self.vectorSpace)
            self.velocity0 = 0

        return None


    def define_scalar_functions(self):
        """


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

        return None


    def define_deformation_tensors(self):
        """


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

            self.velocityGradient = dlf.grad(self.velocity)
            if self.config['formulation']['time']['unsteady']:
                self.velocityGradient0 = dlf.grad(self.velocity0)
            else:
                self.velocityGradient0 = 0

        return None


    def define_material(self):
        """


        """

        # Check which class should be called.
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
            raise NotImplementedError("Shouldn't be in here...")

        # Create an instance of the material class and store
        # as member data.
        self._material = mat_class(inverse=self.config['formulation']['inverse'],
                                   **self.config['material'])

        return None


    def define_dirichlet_bcs(self):
        """
        Define a list of Dirichlet BC objects based on the problem configuration
        provided by the user.


        Parameters
        ----------

        functionSpace : dolfin.functions.functionspace.FunctionSpace
            dolfin object representing the discrete function space that
            will be used to approximate the solution to the weak form.


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
            raise ValueError(s)
        elif 'p_regions' in self.config['formulation']['bcs']['dirichlet']:
            s = "The regions for pressure were specified, but the values were not."
            raise ValueError(s)
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
                bc = dlf.DirichletBC(V, value, self.mesh_function, region)
                self.dirichlet_bcs['velocity'].append(bc)

        # Store the Dirichlet BCs for the displacement vector field.
        if disp_vals is not None:
            self.dirichlet_bcs['displacement'] = list()
            for region, value in zip(regions, disp_vals):
                bc = dlf.DirichletBC(V, value, self.mesh_function, region)
                self.dirichlet_bcs['displacement'].append(bc)

        # Store the Dirichlet BCs for the pressure scalar field.
        if pressure_vals is not None:
            self.dirichlet_bcs['pressure'] = list()
            for region, value in zip(p_regions, pressure_vals):
                bc = dlf.DirichletBC(S, value, self.mesh_function, region)
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
        Define all of the forms necessary for the problem specified
        by the user.

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
                                                            self.mesh_function,
                                                            self.deformationGradient,
                                                            self.jacobian,
                                                            self.test_vector)
        if self.config['formulation']['time']['unsteady']:
            values0 = duplicate_expressions(*values)
            self.ufl_neumann_bcs0 = self.define_ufl_neumann_form(regions, types,
                                                                 values0, domain,
                                                                 self.mesh,
                                                                 self.mesh_function,
                                                                 self.deformationGradient0,
                                                                 self.jacobian0,
                                                                 self.test_vector)
        else:
            self.ufl_neumann_bcs0 = 0

        return None


    def define_ufl_neumann_bcs_diff(self):
        """


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

        if (not eulerian) or lin_elastic:
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
        if (not eulerian) or lin_elastic:
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
            raise NotImplementedError("Shouldn't be in here...")

        xi = self.test_vector
        self.ufl_stress_work = dlf.inner(dlf.grad(xi), stress_tensor)*dlf.dx
        if self.config['formulation']['time']['unsteady']:
            if self.config['material']['type'] == 'elastic':
                stress_tensor0 = self._material.stress_tensor(self.deformationGradient0,
                                                              self.jacobian0,
                                                              self.pressure0)
            else:
                raise NotImplementedError("Shouldn't be in here...")
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

        # Derivative of stress term w.r.t. to displacement.
        self.ufl_stress_work_du = dlf.derivative(self.ufl_stress_work,
                                                 self.displacement,
                                                 self.trial_vector)

        # Derivative of stress term w.r.t. to velocity.
        if self.velocity != 0:
            self.ufl_stress_work_dv = dlf.derivative(self.ufl_stress_work,
                                                     self.velocity,
                                                     self.trial_vector)

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


        """

        self.define_ufl_velocity_equation()
        self.define_ufl_momentum_equation()
        self.define_ufl_incompressibility_equation()

        return None


    def define_ufl_velocity_equation(self):
        """


        """

        if hasattr(self, 'f1'):
            return None

        if not self.config['formulation']['time']['unsteady']:
            self.f1 = 0
            return None

        alpha = self.config['formulation']['time']['alpha']
        dt = self.config['formulation']['time']['dt']
        f1 = self.displacement - self.displacement0 \
             - dt*(alpha*self.velocity + (1.0 - alpha)*self.velocity0)

        self.f1 = dlf.dot(self.test_vector, f1)*dlf.dx

        return None


    def define_ufl_momentum_equation(self):
        """


        """

        if hasattr(self, 'f2'):
            return None

        alpha = self.config['formulation']['time']['alpha']
        dt = self.config['formulation']['time']['dt']

        self.f2 = self.ufl_local_inertia - self.ufl_local_inertia0
        self.f2 += dt*alpha*(self.ufl_convec_accel + self.ufl_stress_work \
                             - self.ufl_neumann_bcs - self.ufl_body_force)
        self.f2 += dt*(1.0 - alpha)*(self.ufl_convec_accel0 + self.ufl_stress_work0 \
                                     - self.ufl_neumann_bcs0 - self.ufl_body_force0)

        return None


    def define_ufl_incompressibility_equation(self):
        """


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

        kappa = self._material._parameters['kappa']
        self.f3 = self.test_scalar*(kappa*b_vol - self.pressure)*dlf.dx

        return None


    def define_ufl_equations_diff(self):
        """


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


    def update_time(self, t, t0=None):
        """
        Update the time parameter in the BCs that depend on time explicitly.
        Also, the body force expression if necessary.

        """

        if self.dirichlet_bcs:
            self.update_dirichlet_time(t)

        if self.ufl_neumann_bcs:
            neumann_updated = self.update_neumann_time(t, t0=t0)

        if self.ufl_body_force:
            bodyforce_updated = self.update_bodyforce_time(t, t0=t0)

        return None


    def update_dirichlet_time(self, t):
        """
        Update the time parameter in the Dirichlet BCs that depend on time
        explicitly.

        """

        if self.config['formulation']['bcs']['dirichlet'] is None:
            print ('No Dirichlet BCs to update!')
            return None

        expr_list = list()

        if 'displacement' in self.config['formulation']['bcs']['dirichlet']:
            expr_list.extend(self.config['formulation']['bcs']['dirichlet']['displacement'])

        if 'velocity' in self.config['formulation']['bcs']['dirichlet']:
            expr_list.extend(self.config['formulation']['bcs']['dirichlet']['velocity'])

        for expr in expr_list:
            if hasattr(expr, 't'):
                expr.t = t

        return None


    def update_neumann_time(self, t, t0=None):
        """
        Update the time parameter in the Neumann BCs that depend on time
        explicitly. The PETSc vector corresponding to this term is assembled
        again if necessary.

        """

        if self.ufl_neumann_bcs is not None:
            self.update_form_time(self.ufl_neumann_bcs, t)

        if self.ufl_neumann_bcs0 and (t0 is not None):
            self.update_form_time(self.ufl_neumann_bcs0, t0)

        return None


    def update_bodyforce_time(self, t, t0=None):
        """
        Update the time parameter in the body force expression if it depends
        on time explicitly. The PETSc vector corresponding to this term is
        assembled again if necessary.

        """

        if self.ufl_body_force is not None:
            self.update_form_time(self.ufl_body_force, t)

        if self.ufl_body_force0 and (t0 is not None):
            self.update_form_time(self.ufl_body_force0, t0)

        return None


    @staticmethod
    def define_ufl_neumann_form(regions, types, values, domain,
                                mesh, mesh_function, F, J, xi):
        """
        Define the UFL object representing the Neumann boundary
        conditions based on the problem configuration given by
        the user. The function exits if the object has already
        been defined.


        """

        # Check if Nanson's formula is necessary
        if domain == 'lagrangian':
            if ('pressure' in types) or ('cauchy' in types):
                Finv = dlf.inv(F)
                N = dlf.FacetNormal(mesh)

            if 'pressure' in types:
                n = J*Finv.T*N # Nanson's formula

            if 'cauchy' in types:
                nanson_mag = J*dlf.sqrt(dlf.dot(Finv.T*N, Finv.T*N))
        else:
            if 'pressure' in types:
                # No need for Nanson's formula in Eulerian coordinates
                n = dlf.FacetNormal(mesh)

        neumann_form = 0
        zipped_vals = zip(regions, types, values)

        for region, tt, value in zipped_vals:

            ds_region = dlf.ds(region, domain=mesh,
                               subdomain_data=mesh_function)

            if tt == 'pressure':
                val = -dlf.dot(xi, value*n)*ds_region
            elif tt == 'cauchy' and domain == 'lagrangian':
                val = nanson_mag*dlf.dot(xi, value)*ds_region
            elif tt == 'cauchy' and domain == 'eulerian':
                val = dlf.dot(xi, value)*ds_region
            else: # piola traction in lagrangian coordinates
                val = dlf.dot(xi, value)*ds_region

            neumann_form += val

        return neumann_form


    @staticmethod
    def update_form_time(form, t):
        """


        """

        coeffs = form.coefficients()
        for expr in coeffs:
            if hasattr(expr, 't'):
                expr.t = t

        return None
