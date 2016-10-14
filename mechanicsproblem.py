import re
import dolfin as dlf

from . import materials
from .utils import load_mesh, load_mesh_function

class MechanicsProblem:
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
       * 'lambda' : float, int, dolfin.Constant
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
                * 'regions' : list, tuple
                    List of the region IDs (int) on which Dirichlet
                    boundary conditions are to be imposed. These IDs
                    must match those used by the mesh function provided.
                    The order must match the order used in the list of
                    values.
                * 'values' : list, tuple
                    List of values (dolfin.Constant or dolfin.Expression)
                    for each region Dirichlet boundary region specified.
                    The order must match the order used in the list of
                    region IDs.
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

    def __init__(self, user_config, **kwargs):

        # Check configuration dictionary
        self.config = self.check_config(user_config)

        # Obtain mesh and mesh function
        self.mesh = load_mesh(self.config['mesh']['mesh_file'])
        self.mesh_function = load_mesh_function(self.config['mesh']['mesh_function'], self.mesh)

        # Define necessary member data
        self.define_function_spaces()
        self.define_functions()
        self.define_deformation_tensors()
        self.define_dirichlet_bcs()
        self.define_forms()

        # Initialize PETSc matrices and vectors
        self.init_all()

        return None


    def check_config(self, user_config):
        """
        Check that all parameters provided in 'user_config' are valid
        based on the current capabilities of the package. If certain
        keys are not provided, they are added with a value of 'None'.
        An exception is raised when a parameter is found to be invalid.


        Parameters
        ----------

        user_config : dict
            Dictionary describing the formulation of the mechanics
            problem to be simulated. Check the documentation of
            MechanicsProblem to see the format of the dictionary.


        Results
        -------

        config : dict
            A copy of user_config with possibly new keys that are needed
            if they were not originally provided.


        """

        # Use a copy to avoid altering the original object
        config = user_config.copy()

        ############################################################
        # Check if the finite element type is provided.
        if 'element' not in config['mesh']:
            raise ValueError('You need to specify the type of element(s) to use.')

        # Make sure at most two element types are specified.
        if isinstance(config['mesh']['element'], str):
            fe_list = re.split('-|_| ', config['mesh']['element'])
        else:
            fe_list = config['mesh']['element']

        len_fe_list = len(fe_list)
        if len_fe_list == 0 or len_fe_list > 2:
            s1 = 'The current formulation allows 1 or 2 fields.\n'
            s2 = 'You provided %i. Check config[\'mesh\'][\'element\'].' % len_fe_list
            raise NotImplementedError(s1 + s2)
        elif len_fe_list == 1 and config['material']['incompressible']:
            s1 = 'Only one element type, \'%s\', was specified ' % config['mesh']['element'] \
                 + 'for an incompressible material.'
            raise ValueError(s1)
        elif len_fe_list == 2 and not config['material']['incompressible']:
            s1 = 'Two element types, \'%s\', were specified ' % config['mesh']['element'] \
                 +'for a compressible material.'
            raise ValueError(s1)
        else:
            # Replace with list in case it was originally a string
            config['mesh']['element'] = fe_list

        # Check to make sure all strings are only 2 characters long.
        str_len = set(map(len, config['mesh']['element']))
        s1 = 'Element types must be of the form \'p<int>\', where <int> ' \
             + 'is the polynomial degree to be used.' # error string

        # All strings should have the same number of characters.
        if not len(str_len) == 1:
            raise ValueError(s1)

        # All strings should consist of two characters.
        str_len_val = str_len.pop()
        if not str_len_val == 2:
            raise ValueError(s1)

        # Check to make sure first character is 'p'
        if not fe_list[0][0] == 'p':
            s1 = 'The finite element family, \'%s\', has not been implemented.' \
                 % fe_list[0][0]
            raise NotImplementedError(s1)

        ############################################################
        # Check domain formulation.
        domain = config['formulation']['domain']
        if not domain in ['lagrangian', 'eulerian']:
            s1 = 'Formulation with respect to \'%s\' coordinates is not supported.' \
                 % config['formulation']['domain']
            raise ValueError(s1)

        ############################################################
        # Make sure that the BC dictionaries have the same
        # number of regions, values, etc., if any were
        # specified. If they are not specified, set them to None.
        if 'bcs' in config['formulation']:
            # Check Dirichlet and Neumann BCs
            self.check_dirichlet(config)
            self.check_neumann(config)
        else:
            print '*** No BCs (Neumann and Dirichlet) were specified. ***'
            config['formulation']['bcs'] = dict()
            config['formulation']['bcs']['dirichlet'] = None
            config['formulation']['bcs']['neumann'] = None

        ############################################################
        # Check if material type has been implemented.
        self.check_material_const_eqn(config)

        # ############################################################
        # # Check for unsteady flag in 'formulation' dictionary. Assume
        # # problem is a steady-state problem if not provided.
        # if 'unsteady' not in config['formulation']:
        #     config['formulation']['time']['unsteady'] = False

        ############################################################
        # Check if body force was provided. Assume zero if not.
        if 'body_force' not in config['formulation']:
            config['formulation']['body_force'] = None

        ############################################################
        # Check the parameters given for time integration.
        if 'time' in config['formulation']:
            self.check_time_params(config)
        else:
            config['formulation']['time'] = dict()
            config['formulation']['time']['unsteady'] = False

        return config


    def check_time_params(self, config):
        """


        """

        # Exit function if problem was specified as steady.
        if not config['formulation']['time']['unsteady']:
            return None

        supported = ['explicit_euler']
        if config['formulation']['time']['integrator'] not in supported:
            s1 = 'The time integrating scheme, \'%s\', has not been implemented.' \
                 % config['formulation']['time']['integrator']
            raise NotImplementedError(s1)

        types = (dlf.Constant, float)
        if not isinstance(config['formulation']['time']['dt'], types):
            s1 = 'The \'dt\' parameter must be a scalar value of type: ' \
                 + 'dolfin.Constant, float'
            raise TypeError(s1)

        return None


    def check_material_const_eqn(self, config):
        """
        Check if the material type and the specific constitutive equation
        specified in the config dictionary are implemented.


        Parameters
        ----------

        config : dict
            Dictionary describing the formulation of the mechanics
            problem to be simulated. Check the documentation of
            MechanicsProblem to see the format of the dictionary.


        """

        try:
            submodule = getattr(materials, config['material']['type'])

            # Check if specific constitutive equation is available.
            try:
                const_eqn = getattr(submodule, config['material']['const_eqn'])
            except AttributeError:
                s1 = 'The constitutive equation, \'%s\', has not been implemented ' \
                     % config['material']['const_eqn'] \
                     + 'within the material type, \'%s\'.' % config['material']['type']
                raise NotImplementedError(s1)
        except AttributeError:
            s1 = 'The class of materials, \'%s\', has not been implemented.' \
                 % config['material']['type']
            raise NotImplementedError(s1)

        return None


    def check_dirichlet(self, config):
        """
        Check if the number of parameters for each key in the dirichlet
        sub-dictionary are equal. If the key 'dirichlet' does not exist,
        it is added to the dictionary with the value None.


        Parameters
        ----------

        config : dict
            Dictionary describing the formulation of the mechanics
            problem to be simulated. Check the documentation of
            MechanicsProblem to see the format of the dictionary.


        """

        try:
            # Recognize if the user already specified None.
            if config['formulation']['bcs']['dirichlet'] is None:
                raise AttributeError

            if not self.__check_bc_params(config['formulation']['bcs']['dirichlet']):
                raise ValueError('The number of Dirichlet boundary regions and ' \
                                 + 'values do not match!')
        except AttributeError:
            print '*** No Dirichlet BCs were specified. ***'
        except KeyError:
            config['formulation']['bcs']['dirichlet'] = None
            print '*** No Dirichlet BCs were specified. ***'

        return None


    def check_neumann(self, config):
        """
        Check if the number of parameters for each key in the neumann
        sub-dictionary are equal. If the key 'neumann' does not exist,
        it is added to the dictionary with the value None.


        Parameters
        ----------

        config : dict
            Dictionary describing the formulation of the mechanics
            problem to be simulated. Check the documentation of
            MechanicsProblem to see the format of the dictionary.


        """

        try:
            # Recognize if the user already specified None.
            if config['formulation']['bcs']['neumann'] is None:
                raise AttributeError

            if not self.__check_bc_params(config['formulation']['bcs']['neumann']):
                raise ValueError('The number of Neumann boundary regions and ' \
                                 + 'values do not match!')

            # Make sure all Neumann BC types are supported with domain specified.
            try:
                neumann_types = config['formulation']['bcs']['neumann']['types']
            except KeyError:
                raise KeyError('The Neumann BC type must be specified for each region.')

            # Check that types are valid
            valid_types = {'pressure', 'cauchy', 'piola'}
            union = valid_types.union(neumann_types)
            if len(union) > 3:
                s1 = 'At least one Neumann BC type is unrecognized. The type string must'
                s2 = ' be one of the three: ' + ', '.join(list(valid_types))
                raise NotImplementedError(s1 + s2)

            domain = config['formulation']['domain']
            if domain == 'eulerian' and 'piola' in neumann_types:
                s1 = 'Piola traction in an Eulerian formulation is not supported.'
                raise NotImplementedError(s1)

        except AttributeError:
            print '*** No Neumann BCs were specified. ***'
        except KeyError:
            config['formulation']['bcs']['neumann'] = None
            print '*** No Neumann BCs were specified. ***'

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
            self.scalarSpace = dlf.functionSpace(self.mesh, scalar_element)
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

        self.test_vector = dlf.TestFunction(self.vectorSpace)
        self.trial_vector = dlf.TrialFunction(self.vectorSpace)

        self.velocity = dlf.Function(self.vectorSpace)
        self.acceleration = dlf.Function(self.vectorSpace)

        unsteady = self.config['formulation']['time']['unsteady']
        lagrangian = self.config['formulation']['domain'] == 'lagrangian'

        if unsteady and lagrangian:
            self.displacement = dlf.Function(self.vectorSpace)

            # Functions for previous time step
            self.displacement0 = dlf.Function(self.vectorSpace)
            self.velocity0 = dlf.Function(self.vectorSpace)
            self.acceleration0 = dlf.Function(self.vectorSpace)
        elif unsteady:
            # Functions for previous time step
            self.velocity0 = dlf.Function(self.vectorSpace)
            self.acceleration0 = dlf.Function(self.vectorSpace)
        elif lagrangian:
            self.displacement = dlf.Function(self.vectorSpace)

            self.displacement0 = None
            self.velocity0 = None
            self.acceleration0 = None
        else:
            self.displacement = None

            self.displacement0 = None
            self.velocity0 = None
            self.acceleration0 = None

        return None


    def define_scalar_functions(self):
        """


        """

        # Exit if functions have already been defined.
        # A function decorator might work better here...
        if hasattr(self, 'pressure'):
            return None

        if self.config['material']['incompressible']:
            self.pressure = dlf.Function(self.scalarSpace)

            if self.config['formulation']['time']['unsteady']:
                self.pressure0 = dlf.Function(self.scalarSpace)
            else:
                self.pressure0 = None

            self.test_scalar = dlf.TestFunction(self.scalarSpace)
            self.trial_scalar = dlf.TrialFunction(self.scalarSpace)
        else:
            self.pressure = None
            self.pressure0 = None
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

        # THIS SHOULD CHECK FOR VISCOUS MATERIALS IN THE FUTURE
        #
        # This should check type to determine which deformation tensors
        # are necessary.
        if self.config['material']['type'] == 'elastic':
            I = dlf.Identity(self.mesh.geometry().dim())
            self.deformationGradient = I + dlf.grad(self.displacement)
            self.deformationRateGradient = None
            self.jacobian = dlf.det(self.deformationGradient)
        else:
            s1 = 'The material type, \'%s\', has not been implemented!' \
                 % self.config['material']['type']
            raise NotImplementedError(s1)

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

        # Exit function if no Dirichlet BCs were provided.
        if self.config['formulation']['bcs']['dirichlet'] is None:
            self.dirichlet_bcs = None
            return None

        V = self.vectorSpace

        self.dirichlet_bcs = list()
        for region, value in zip(self.config['formulation']['bcs']['dirichlet']['regions'],
                                 self.config['formulation']['bcs']['dirichlet']['values']):
            self.dirichlet_bcs.append(dlf.DirichletBC(V, value, self.mesh_function, region))

        return None


    def define_ufl_neumann_bcs(self):
        """
        Define the UFL object representing the Neumann boundary
        conditions based on the problem configuration given by
        the user. The function exits if the object has already
        been defined.


        """

        # Exit function if already defined.
        if hasattr(self, 'ufl_neumann_bcs'):
            return None

        # Exit function if no Neumann BCs were provided.
        if self.config['formulation']['bcs']['neumann'] is None:
            self.ufl_neumann_bcs = None
            return None

        region_list = self.config['formulation']['bcs']['neumann']['regions']
        tt_list = self.config['formulation']['bcs']['neumann']['types']
        value_list = self.config['formulation']['bcs']['neumann']['values']

        domain = self.config['formulation']['domain']

        # Check if Nanson's formula is necessary
        if domain == 'lagrangian':
            if 'pressure' in tt_list or 'cauchy' in tt_list:
                Finv = dlf.inv(self.deformationGradient)
                J = self.jacobian
                N = dlf.FacetNormal(self.mesh)

            if 'pressure' in tt_list:
                n = J*Finv.T*N # Nanson's formula

            if 'cauchy' in tt_list:
                nanson_mag = J*dlf.sqrt(dlf.dot(Finv.T*N, Finv.T*N))
        else:
            if 'pressure' in tt_list:
                n = dlf.FacetNormal(self.mesh) # No need for Nanson's formula

        self.ufl_neumann_bcs = 0
        zipped_vals = zip(region_list, tt_list, value_list)

        for region, tt, value in zipped_vals:

            ds_region = dlf.ds(region, domain=self.mesh,
                               subdomain_data=self.mesh_function)

            if tt == 'pressure':
                val = -dlf.dot(self.test_vector, value*n)*ds_region
            elif tt == 'cauchy' and domain == 'lagrangian':
                val = nanson_mag*dlf.dot(self.test_vector, value)*ds_region
            elif tt == 'cauchy' and domain == 'eulerian':
                val = dlf.dot(self.test_vector, value)*ds_region
            else: # piola traction in lagrangian coordinates
                val = dlf.dot(self.test_vector, value)*ds_region

            self.ufl_neumann_bcs += val

        return None


    def define_ufl_local_accel(self):
        """
        Define the UFL object corresponding to the local acceleration
        term in the weak form. The function exits if it has already
        been defined.


        """

        # Exit if form is already defined!
        if hasattr(self, 'ufl_local_accel'):
            return None

        xi = self.test_vector
        rho = self.config['material']['density']

        self.ufl_local_accel = dlf.dot(xi, rho*self.acceleration)*dlf.dx

        return None


    def define_ufl_local_accel_diff(self):
        """
        Define the UFL object that describes the matrix that results
        from taking the Gateaux derivative of the local acceleration
        term in the weak form. The function exits if it has already
        been defined.

        """

        if hasattr(self, 'ufl_local_accel_diff'):
            return None

        xi = self.test_vector
        du = self.trial_vector
        rho = self.config['material']['density']

        self.ufl_local_accel_diff = dlf.dot(xi, rho*du)*dlf.dx

        return None


    def define_ufl_convec_accel(self):
        """
        Define the UFL object corresponding to the convective acceleration
        term in the weak form. The function exits if it has already been
        defined.

        """

        if hasattr(self, 'ufl_convec_accel'):
            return None

        xi = self.test_vector
        rho = self.config['material']['density']
        self.ufl_convec_accel = dlf.dot(xi, rho*dlf.grad(self.velocity)\
                                        * self.velocity)*dlf.dx

        return None


    def define_ufl_convec_accel_diff(self):
        """
        Define the UFL object corresponding to the Gateaux derivative of
        the convective acceleration term in the weak form. The function
        exits if it has already been defined.

        """

        if hasattr(self, 'ufl_convec_accel_diff'):
            return None

        if not hasattr(self, 'ufl_convec_accel'):
            self.define_ufl_convec_accel()

        self.ufl_convec_accel_diff = dlf.derivative(self.ufl_convec_accel,
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

        # Get access to the function defining the stress tensor
        material_submodule = getattr(materials, self.config['material']['type'])
        stress_function = getattr(material_submodule, self.config['material']['const_eqn'])
        stress_tensor = stress_function(self)

        xi = self.test_vector
        self.ufl_stress_work = dlf.inner(dlf.grad(xi), stress_tensor)*dlf.dx

        return None


    def define_ufl_stress_work_diff(self):
        """
        Define the UFL object corresponding to the Gateaux derivative of
        the stress tensor term in the weak form. The function exits if it
        has already been defined.

        """

        if hasattr(self, 'ufl_stress_work_diff'):
            return None

        if not hasattr(self, 'ufl_stress_work'):
            self.define_ufl_stress_work()

        if self.config['material']['type'] == 'elastic':
            self.ufl_stress_work_diff = dlf.derivative(self.ufl_stress_work,
                                                       self.displacement,
                                                       self.trial_vector)
        else:
            # # TAKE DERIVATIVE AT VELOCITY IF VISCOUS MATERIAL.
            # self.ufl_stress_work_diff = dlf.derivative(self.ufl_stress_work,
            #                                            self.velocity,
            #                                            self.trial_vector)
            raise NotImplementedError('Material type is not implemented.')

        return None


    def define_ufl_body_force(self):
        """
        Define the UFL object corresponding to the body force term in
        the weak form. The function exits if it has already been defined.

        """

        if hasattr(self, 'ufl_body_force'):
            return None

        rho = self.config['material']['density']
        b = self.config['formulation']['body_force']
        xi = self.test_vector

        self.ufl_body_force = dlf.dot(xi, rho*b)*dlf.dx

        return None


    def define_forms(self):
        """
        Define all of the forms necessary for the problem specified
        by the user.

        """

        # Check if problem was specified as unsteady.
        if self.config['formulation']['time']['unsteady']:
            self.define_ufl_local_accel()
            self.define_ufl_local_accel_diff()
        else:
            self.ufl_local_accel = None
            self.ufl_local_accel_diff = None

        # Define UFL objects corresponding to the convective acceleration
        # if problem is formulated with respect to Eulerian coordinates
        if self.config['formulation']['domain'] == 'eulerian':
            self.define_ufl_convec_accel()
            self.define_ufl_convec_accel_diff()
        else:
            self.ufl_convec_accel = None
            self.ufl_convec_accel_diff = None

        # Define UFL objects corresponding to the stress tensor term.
        # This should always be non-zero for deformable bodies.
        self.define_ufl_stress_work()
        self.define_ufl_stress_work_diff()

        # Define UFL object corresponding to the body force term. Assume
        # it is zero if key was not provided.
        if self.config['formulation']['body_force']:
            self.define_ufl_body_force()
        else:
            self.ufl_body_force = None

        # Define UFL object corresponding to the traction force terms. Assume
        # it is zero if key was not provided.
        self.define_ufl_neumann_bcs()

        return None


    def init_all(self):
        """
        Initialize all of the necessary PETSc matrices and vector objects
        based on the problem configuration.

        """

        # Exit if objects have already been initialized
        if hasattr(self, '_localAccelMatrix'):
            return None

        # Check if problem was specified as unsteady.
        if self.ufl_local_accel is not None:
            self._localAccelMatrix = dlf.PETScMatrix()
        else:
            self._localAccelMatrix = None

        # Check if problem was formulated in Eulerian coordinates
        # to include the convective acceleration term.
        if self.ufl_convec_accel is not None:
            self._convectiveAccelMatrix = dlf.PETScMatrix()
            self._convectiveAccelVector = dlf.PETScVector()
        else:
            self._convectiveAccelMatrix = None
            self._convectiveAccelVector = None

        # This should always be non-zero for deformable bodies.
        self._stressWorkMatrix = dlf.PETScMatrix()
        self._stressWorkVector = dlf.PETScVector()

        # Initialize the vector corresponding to the body force term.
        if self.ufl_body_force is not None:
            self._bodyForceWorkVector = dlf.PETScVector()
        else:
            self._bodyForceWorkVector = None

        # Initialize vector corresponding to the traction force terms. Assume
        # it is zero if key was not provided.
        if self.ufl_neumann_bcs is not None:
            self._tractionWorkVector = dlf.PETScVector()
        else:
            self._tractionWorkVector = None

        return None


    def update_all(self, t=None):
        """
        Update all matrices and vectors that depend explicitly on time or
        that depend on the state of the problem (velocity/displacement).
        If no time value is provided, only the values that depend on the
        state are updated.


        Parameters
        ----------

        t : scalar (default None)


        """

        if t is not None:
            self.update_time(t)

        # Update the assembled PETSc matrices and vectors that
        # depend on the current state.

        # Update the matrices/vectors coming from local acceleration
        if self._localAccelMatrix is not None:
            self.assembleLocalAccelMatrix()

        # Update the matrices/vectors coming from convective acceleration
        if self._convectiveAccelMatrix is not None:
            self.assembleConvectiveAccelMatrix()
            self.assembleConvectiveAccelVector()

        # Update the matrices/vectors coming from the stress work term
        self.assembleStressWorkMatrix()
        self.assembleStressWorkVector()

        # Assemble the total load vector (i.e. residual vector)
        self.assembleLoadVector()

        return None


    def update_time(self, t):
        """
        Update the time parameter in the BCs that depend on time explicitly.
        Also, the body force expression if necessary.

        """

        if self.dirichlet_bcs is not None:
            self.update_dirichlet_time(t)

        if self.ufl_neumann_bcs is not None:
            neumann_updated = self.update_neumann_time(t)

        if self.ufl_body_force is not None:
            bodyforce_updated = self.update_bodyforce_time(t)

        return None


    def update_dirichlet_time(self, t):
        """
        Update the time parameter in the Dirichlet BCs that depend on time
        explicitly.

        """

        for expr in self.config['formulation']['bcs']['dirichlet']['values']:
            if hasattr(expr, 't'):
                expr.t = t

        return None


    def update_neumann_time(self, t):
        """
        Update the time parameter in the Neumann BCs that depend on time
        explicitly. The PETSc vector corresponding to this term is assembled
        again if necessary.

        """

        need_to_update = False
        for expr in self.config['formulation']['bcs']['neumann']['values']:
            if hasattr(expr, 't'):
                expr.t = t
                need_to_update = True

        if need_to_update:
            self.assembleTractionVector()

        return need_to_update


    def update_bodyforce_time(self, t):
        """
        Update the time parameter in the body force expression if it depends
        on time explicitly. The PETSc vector corresponding to this term is
        assembled again if necessary.

        """

        need_to_update = False

        expr = self.config['formulation']['body_force']
        if hasattr(expr, 't'):
            expr.t = t
            self.assembleBodyForceVector()
            need_to_update = True

        return need_to_update


    def assemble_all(self):
        """
        Assemble the PETSc matrices and vectors that correspond to the
        necessary terms based on the problem configuration provided by
        the user.

        """

        # Assemble local acceleration matrix is problem is unsteady.
        if self._localAccelMatrix is not None:
            self.assembleLocalAccelMatrix()

        # Assemble convective acceleration matrix and vector.
        if self._convectiveAccelMatrix is not None:
            self.assembleConvectiveAccelMatrix()

        if self._convectiveAccelVector is not None:
            self.assembleConvectiveAccelVector()

        # Assemble the matrix from stress work
        self.assembleStressWorkMatrix()
        self.assembleStressWorkVector()

        # Assemble the vectors corresponding to body force, traction,
        # and the residual (load vector).
        self.assembleLoadVector()

        return None


    def assembleLocalAccelMatrix(self):
        """
        Return the matrix that comes from the differential of the local
        acceleration term in the weak form. I.e., the partial derivative
        of velocity with respect to time. Note that this is idential to
        the 'total acceleration' if the problem is formulated using
        Lagrangian coordinates.


        Parameters
        ----------

        bc_apply : bool (default, True)
            Specify whether the boundary conditions should be applied
            to the rows and columns of the matrix. Note that this method
            does not preserve symmetry.


        Returns
        -------

        M : dolfin.cpp.la.Matrix


        """

        dlf.assemble(self.ufl_local_accel_diff,
                     tensor=self._localAccelMatrix)

        return None


    def assembleConvectiveAccelMatrix(self):
        """
        Return the matrix that comes from the differential of the convective
        acceleration term in the weak form. I.e., grad(v)*v.


        Parameters
        ----------

        self :


        Returns
        -------

        convec_accel_matrix : dolfin.cpp.la.Matrix


        """

        dlf.assemble(self.ufl_convec_accel_diff,
                     tensor=self._convectiveAccelMatrix)

        return None


    def assembleConvectiveAccelVector(self):
        """


        """

        dlf.assemble(self.ufl_convec_accel,
                     tensor=self._convectiveAccelVector)

        return None


    def assembleStressWorkMatrix(self):
        """


        """

        dlf.assemble(self.ufl_stress_work_diff,
                     tensor=self._stressWorkMatrix)

        return None


    def assembleStressWorkVector(self):
        """


        """

        dlf.assemble(self.ufl_stress_work,
                     tensor=self._stressWorkVector)

        return None


    def assembleLoadVector(self):
        """


        """

        body_not_none = self._bodyForceWorkVector is not None
        trac_not_none = self._tractionWorkVector is not None

        if body_not_none and trac_not_none:
            self.assembleBodyForceVector()
            self.assembleTractionVector()
            self._totalLoadVector = self._bodyForceWorkVector \
                                    + self._tractionWorkVector
        elif body_not_none:
            self.assembleBodyForceVector()
            self._totalLoadVector = dlf.PETScVector(self._bodyForceWorkVector)
        elif trac_not_none:
            self.assembleTractionVector()
            self._totalLoadVector = dlf.PETScVector(self._tractionWorkVector)
        else:
            pass

        self._totalLoadVector -= self._stressWorkVector

        if self.config['formulation']['domain'] == 'eulerian':
            self._totalLoadVector -= self._convectiveAccelVector

        return None


    def assembleBodyForceVector(self):
        """


        """

        dlf.assemble(self.ufl_body_force,
                     tensor=self._bodyForceWorkVector)

        return None


    def assembleTractionVector(self):
        """


        """

        dlf.assemble(self.ufl_neumann_bcs,
                     tensor=self._tractionWorkVector)

        return None


    def get_dirichlet_bcs(self):

        return self.dirichlet_bcs


    def get_ufl_neumann_bcs(self):

        return self.ufl_neumann_bcs


    def getLocalAccelMatrix(self):

        return self._localAccelMatrix


    def getUFLLocalAccel(self):
        """
        Return the matrix that comes from the differential of the local
        acceleration term in the weak form. I.e., the partial derivative
        of velocity with respect to time. Note that this is idential to
        the 'total acceleration' if the problem is formulated using
        Lagrangian coordinates.


        Parameters
        ----------

        self :


        Returns
        -------

        local_accel_matrix : dolfin.cpp.la.Matrix


        """

        return self.ufl_local_accel


    def getConvectiveAccelMatrix(self):

        return self._convectiveAccelMatrix


    def getConvectiveAccelVector(self, tensor=None):

        return self._convectiveAccelVector


    def getUFLConvectiveAccel(self):
        """


        """

        return self.ufl_convec_accel


    def getUFLConvectiveAccelDifferential(self):
        """


        """

        return self.ufl_convec_accel_diff


    def getUFLStressWork(self):
        """


        """

        return self.ufl_stress_work


    def getUFLStressWorkDifferential(self):
        """


        """

        return self.ufl_stress_work_diff


    def getUFLBodyWork(self):
        """


        """

        return self.ufl_body_force


    @staticmethod
    def __check_bc_params(bc_dict):
        """
        Make sure that the lengths of all the lists/tuples provided within the
        neumann and dirichlet subdictionaries are the same.

        """

        values = bc_dict.values()
        lengths = map(len, values)
        if len(set(lengths)) == 1:
            return True
        else:
            return False
