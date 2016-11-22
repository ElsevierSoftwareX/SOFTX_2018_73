import re
import dolfin as dlf

from . import materials
from .utils import load_mesh, load_mesh_function, duplicate_expressions

from ufl import Form
from inspect import isfunction

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
                * 'velocity'
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
                * 'displacement'
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
        self.define_ufl_equations()
        self.define_ufl_equations_diff()

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

        # Check domain formulation.
        domain = config['formulation']['domain']
        if not domain in ['lagrangian', 'eulerian']:
            s1 = 'Formulation with respect to \'%s\' coordinates is not supported.' \
                 % config['formulation']['domain']
            raise ValueError(s1)

        # Make sure that the BC dictionaries have the same
        # number of regions, values, etc., if any were
        # specified. If they are not specified, set them to None.
        self.check_bcs(config)

        # Check if material type has been implemented.
        self.check_material_const_eqn(config)

        # Check if body force was provided. Assume zero if not.
        if 'body_force' not in config['formulation']:
            config['formulation']['body_force'] = None

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
            config['formulation']['time']['alpha'] = 1
            config['formulation']['time']['dt'] = 1
            return None

        supported = ['explicit_euler', 'generalized_alpha']
        if config['formulation']['time']['integrator'] not in supported:
            s1 = 'The time integrating scheme, \'%s\', has not been implemented.' \
                 % config['formulation']['time']['integrator']
            raise NotImplementedError(s1)

        if not isinstance(config['formulation']['time']['dt'], float):
            s1 = 'The \'dt\' parameter must be a scalar value of type: ' \
                 + 'dolfin.Constant, float'
            raise TypeError(s1)

        # if config['formulation']['time']['integrator'] == 'generalized_alpha':
        try:
            alpha = config['formulation']['time']['alpha']
            if alpha < 0.0 or alpha > 1.0:
                s1 = 'The value of alpha for the generalized alpha ' \
                     + 'method must be between 0 and 1. The value ' \
                     + 'provided was: %.4f ' % alpha
                raise ValueError(s1)
        except KeyError:
            s1 = 'The value of alpha for the generalized alpha ' \
                 + 'method must be provided.'
            raise KeyError(s1)

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


    def check_bcs(self, config):
        """


        """

        try:
            if config['formulation']['bcs'] is not None:
                # Check Dirichlet and Neumann BCs
                self.check_dirichlet(config)
                self.check_neumann(config)
            else:
                raise ValueError
        # except (KeyError, ValueError):
        except (KeyError):
            config['formulation']['bcs'] = dict()
            config['formulation']['bcs']['neumann'] = None
            config['formulation']['bcs']['dirichlet'] = None
            print '*** No BCs (Neumann and Dirichlet) were specified. ***'

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
                raise ValueError

            vel = 'velocity'
            disp = 'displacement'
            subconfig = config['formulation']['bcs']['dirichlet']

            # User must specify BCs for velocity, or displacement AND velocity
            if vel in subconfig and disp in subconfig:
                flag1 = self.check_subconfig(subconfig, vel, True)
                flag2 = self.check_subconfig(subconfig, disp, True)

                # Set dirichlet to None if both displacement and velocity
                # are None.
                if flag1 and flag2:
                    config['formulation']['bcs']['dirichlet'] = None

            elif vel in subconfig:
                flag = self.check_subconfig(subconfig, vel, True)

                # Set dirichlet to None if displacement is not provided
                # and velocity is None.
                if flag:
                    config['formulation']['bcs']['dirichlet'] = None
            else:
                s1 = 'Dirichlet boundary conditions must be specified for ' \
                     + 'velocity, or velocity AND displacement.'
                raise TypeError(s1)

        except ValueError:
            print '*** No Dirichlet BCs were specified. ***'
        except KeyError:
            config['formulation']['bcs']['dirichlet'] = None
            print '*** No Dirichlet BCs were specified. ***'

        return None


    def check_subconfig(self, subconfig, key, is_bc=False):
        """


        """

        retval = False

        try:
            if subconfig[key] is None:
                raise AttributeError

            if is_bc:
                if not self.__check_bc_params(subconfig[key]):
                    raise ValueError('The number of Dirichlet boundary regions and ' \
                                     + 'values for not match!')
        except AttributeError:
            print 'check_subconfig...AttributeError'
            retval = True
            print '*** The value for %s was specified as None! ***' % key
        except KeyError:
            print 'check_subconfig...KeyError'
            retval = True
            subconfig[key] = None
            print '*** No value for %s was specified! (set to None)' % key

        return retval


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

        unsteady = self.config['formulation']['time']['unsteady']
        lagrangian = self.config['formulation']['domain'] == 'lagrangian'
        lin_elastic = self.config['material']['const_eqn'] == 'lin_elastic'

        # Trial and test functions
        self.test_vector = dlf.TestFunction(self.vectorSpace)
        self.trial_vector = dlf.TrialFunction(self.vectorSpace)

        if lagrangian or lin_elastic:
            self.displacement = dlf.Function(self.vectorSpace, name='u')
        else:
            self.displacement = 0
        self.velocity = dlf.Function(self.vectorSpace, name='v')

        if unsteady:
            # functions for previous time step
            if lagrangian or lin_elastic:
                self.displacement0 = dlf.Function(self.vectorSpace, name='u0')
            else:
                self.displacement0 = 0
            self.velocity0 = dlf.Function(self.vectorSpace, name='v0')
        else:
            self.displacement0 = 0
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
            self.velocityGradient = dlf.grad(self.velocity)

            if self.config['formulation']['time']['unsteady']:
                self.deformationGradient0 = I + dlf.grad(self.displacement0)
                self.velocityGradient0 = dlf.grad(self.velocity0)
                self.jacobian0 = dlf.det(self.deformationGradient0)
            else:
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

        self.dirichlet_bcs = {'displacement': None, 'velocity': None}
        vel_bcs = self.config['formulation']['bcs']['dirichlet']['velocity']
        disp_bcs = self.config['formulation']['bcs']['dirichlet']['displacement']

        # Store the Dirichlet BCs for the velocity vector field
        if vel_bcs is not None:
            self.dirichlet_bcs['velocity'] = list()
            for region, value in zip(vel_bcs['regions'], vel_bcs['values']):
                bc = dlf.DirichletBC(V, value, self.mesh_function, region)
                self.dirichlet_bcs['velocity'].append(bc)

        # Store the Dirichlet BCs for the displacement vector field
        if disp_bcs is not None:
            self.dirichlet_bcs['displacement'] = list()
            for region, value in zip(disp_bcs['regions'], disp_bcs['values']):
                bc = dlf.DirichletBC(V, value, self.mesh_function, region)
                self.dirichlet_bcs['displacement'].append(bc)

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
        if (not eulerian) or (not lin_elastic):
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
        if (not eulerian) or (not lin_elastic):
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

        # Get access to the function defining the stress tensor
        if isfunction(self.config['material']['const_eqn']):
            stress_function = self.config['material']['const_eqn']
        else:
            material_submodule = getattr(materials, self.config['material']['type'])
            stress_function = getattr(material_submodule, self.config['material']['const_eqn'])

        if self.config['formulation']['time']['unsteady']:
            stress_tensor, stress_tensor0 = stress_function(self)
        else:
            stress_tensor = stress_function(self)

        xi = self.test_vector
        self.ufl_stress_work = dlf.inner(dlf.grad(xi), stress_tensor)*dlf.dx
        if self.config['formulation']['time']['unsteady']:
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

        return None


    def define_ufl_velocity_equation(self):
        """


        """

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

        # alpha AND dt CAN'T BE SET TO 0 IF STEADY
        alpha = self.config['formulation']['time']['alpha']
        dt = self.config['formulation']['time']['dt']

        self.f2 = self.ufl_local_inertia - self.ufl_local_inertia0
        self.f2 += dt*alpha*(self.ufl_convec_accel + self.ufl_stress_work \
                             - self.ufl_neumann_bcs - self.ufl_body_force)
        self.f2 += dt*(1.0 - alpha)*(self.ufl_convec_accel0 + self.ufl_stress_work0 \
                                     - self.ufl_neumann_bcs0 - self.ufl_body_force0)

        return None


    def define_ufl_equations_diff(self):
        """


        """

        if self.f1 is not 0:
            self.df1_du = dlf.derivative(self.f1, self.displacement, self.trial_vector)
            self.df1_dv = dlf.derivative(self.f1, self.velocity, self.trial_vector)
        else:
            self.df1_du = 0
            self.df1_dv = 0

        self.df2_du = dlf.derivative(self.f2, self.displacement, self.trial_vector)
        self.df2_dv = dlf.derivative(self.f2, self.velocity, self.trial_vector)

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
            print 'No Dirichlet BCs to update!'
            return None

        expr_list = list()

        disp_dict = self.config['formulation']['bcs']['dirichlet']['displacement']
        vel_dict = self.config['formulation']['bcs']['dirichlet']['velocity']

        if disp_dict is not None:
            expr_list.extend(disp_dict['values'])

        if vel_dict is not None:
            expr_list.extend(vel_dict['values'])

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


    def assemble_all(self):
        """
        Assemble the PETSc matrices and vectors that correspond to the
        necessary terms based on the problem configuration provided by
        the user.

        """

        raise DeprecationWarning

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


    def assemble_constants(self):
        """
        Assemble values that do not depend on the current state and/or
        time.


        """

        raise DeprecationWarning

        if self._localAccelMatrix is not None:
            self.assembleLocalAccelMatrix()

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

        raise DeprecationWarning

        dlf.assemble(self.ufl_local_accel_diff,
                     tensor=self._localAccelMatrix)

        return None


    def assembleConvectiveAccelMatrix(self, v, tensor=None):
        """
        Return the matrix that comes from the differential of the convective
        acceleration term in the weak form. I.e., grad(v)*v.


        Parameters
        ----------

        v : dolfin.Function
            A dolfin.Function storing the values at which this matrix
            is to be evaluated.


        Returns
        -------

        convec_accel_matrix : dolfin.cpp.la.Matrix


        """

        raise DeprecationWarning

        if tensor is None:
            tensor = dlf.PETScMatrix()

        self.vector_function.assign(v)
        dlf.assemble(self.ufl_convec_accel_diff,
                     tensor=tensor)

        return tensor


    def assembleConvectiveAccelVector(self, v, tensor=None):
        """
        Return the vector that comes from the convective acceleration
        term in the weak form. I.e., dot(xi, grad(v)*v)*dx.


        Parameters
        ----------

        v : dolfin.Function
            A dolfin.Function storing the values at which this matrix
            is to be evaluated.


        """

        raise DeprecationWarning

        if tensor is None:
            tensor = dlf.PETScVector()

        self.vector_function.assign(v)
        dlf.assemble(self.ufl_convec_accel,
                     tensor=tensor)

        return tensor


    def assembleStressWorkMatrix(self, u, p=None, tensor=None):
        """
        Return the matrix that comes from the differential of the stress
        work term in the weak form. I.e., inner(grad(xi), stress)*dx.


        Parameters
        ----------

        u : dolfin.Function
            A dolfin.Function storing the values at which this matrix
            is to be evaluated.


        """

        raise DeprecationWarning

        if tensor is None:
            tensor = dlf.PETScMatrix()

        self.vector_function.assign(u)
        if p is not None:
            self.scalar_function.assign(p)

        dlf.assemble(self.ufl_stress_work_diff,
                     tensor=tensor)

        return tensor


    def assembleStressWorkVector(self, u, p=None, tensor=None):
        """
        Return the vector that comes from the stress work term in the
        weak form. I.e., inner(grad(xi), stress)*dx.


        Parameters
        ----------

        u : dolfin.Function
            A dolfin.Function storing the values at which this matrix
            is to be evaluated.


        """

        raise DeprecationWarning

        if tensor is None:
            tensor = dlf.PETScVector()

        self.vector_function.assign(u)
        if p is not None:
            self.scalar_function.assign(p)

        dlf.assemble(self.ufl_stress_work,
                     tensor=tensor)

        return tensor


    def assembleLoadVector(self):
        """



        """

        raise DeprecationWarning

        body_not_none = self._bodyForceWorkVector is not None
        trac_not_none = self._tractionWorkVector is not None

        if body_not_none and trac_not_none:
            self.assembleBodyForceVector()
            self.assembleTractionVector()
            b = self._bodyForceWorkVector \
                + self._tractionWorkVector
        elif body_not_none:
            self.assembleBodyForceVector()
            b = dlf.PETScVector(self._bodyForceWorkVector)
        elif trac_not_none:
            self.assembleTractionVector()
            b = dlf.PETScVector(self._tractionWorkVector)
        else:
            pass

        b -= self._stressWorkVector

        if self.config['formulation']['domain'] == 'eulerian':
            b -= self._convectiveAccelVector

        return b


    def assembleBodyForceVector(self, t, tensor=None):
        """


        """

        raise DeprecationWarning

        self.update_bodyforce_time(t)

        if tensor is None:
            tensor = dlf.PETScVector()

        dlf.assemble(self.ufl_body_force,
                     tensor=tensor)

        return tensor


    def assembleTractionVector(self, u, t, tensor=None):
        """


        """

        raise DeprecationWarning

        self.update_neumann_time(t)

        if tensor is None:
            tensor = dlf.PETScVector()

        self.vector_function.assign(u)
        dlf.assemble(self.ufl_neumann_bcs,
                     tensor=tensor)

        return tensor


    def assembleTractionMatrix(self, u, t, tensor=None):
        """


        """

        raise DeprecationWarning

        self.update_neumann_time(t)

        if tensor is None:
            tensor = dlf.PETScMatrix()

        self.vector_function.assign(u)
        dlf.assemble(self.ufl_neumann_bcs_diff,
                     tensor=tensor)

        return tensor


    def get_dirichlet_bcs(self):

        return self.dirichlet_bcs


    def get_ufl_neumann_bcs(self):

        return self.ufl_neumann_bcs


    def getLocalAccelMatrix(self):

        raise DeprecationWarning

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

        raise DeprecationWarning

        return self.ufl_local_accel


    def getUFLConvectiveAccel(self):
        """


        """

        raise DeprecationWarning

        return self.ufl_convec_accel


    def getUFLConvectiveAccelDifferential(self):
        """


        """

        raise DeprecationWarning

        return self.ufl_convec_accel_diff


    def getUFLStressWork(self):
        """


        """

        raise DeprecationWarning

        return self.ufl_stress_work


    def getUFLStressWorkDifferential(self):
        """


        """

        raise DeprecationWarning

        return self.ufl_stress_work_diff


    def getUFLBodyWork(self):
        """


        """

        raise DeprecationWarning

        return self.ufl_body_force


    @staticmethod
    def define_ufl_neumann_form(regions, types, values, domain,
                                mesh, mesh_function, F, J, xi):
        """
        Define the UFL object representing the Neumann boundary
        conditions based on the problem configuration given by
        the user. The function exits if the object has already
        been defined.


        """

        # # Exit function if already defined.
        # if hasattr(self, 'ufl_neumann_bcs'):
        #     return None

        # # Exit function if no Neumann BCs were provided.
        # if self.config['formulation']['bcs']['neumann'] is None:
        #     self.ufl_neumann_bcs = None
        #     return None

        # region_list = self.config['formulation']['bcs']['neumann']['regions']
        # tt_list = self.config['formulation']['bcs']['neumann']['types']
        # value_list = self.config['formulation']['bcs']['neumann']['values']

        # domain = self.config['formulation']['domain']

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


    @staticmethod
    def update_form_time(form, t):
        """


        """

        coeffs = form.coefficients()
        for expr in coeffs:
            if hasattr(expr, 't'):
                expr.t = t

        return None
