import re
import dolfin as dlf

from .utils import load_mesh, load_mesh_function
from .__CONSTANTS__ import dict_implemented as _implemented
from inspect import isclass

__all__ = ['BaseMechanicsProblem']


class BaseMechanicsProblem(object):
    """
    This is the base class for mechanics problems. Checking validity of the
    'config' dictionary provided by users is done at this level since all
    mechanics problems are derived from this class. The derived classes will
    then define the variational problem using the FEniCS UFL language.

    The user must provide a python dictionary (referred to as 'config' throughout)
    with the keys and values listed below. Actions taken when optional values are
    not provided are listed below.


    * 'material'
       * 'type' : str
            The class of material that will be used, e.g. elastic, viscous,
            viscoelastic, etc.
       * 'const_eqn' : str, class
            The name of the constitutive equation to be used. User may provide
            their own class which defines a material instead of using those
            implemented in fenicsmechanics.materials.
       * 'incompressible' : bool
            True if the material is incompressible. An
            additional weak form for the incompressibility
            constraint will be added to the problem.
       * 'density' : float, int
            Scalar specifying the density of the material.

       *** The additional material parameters depend on the ***
       ***  constitutive equation being used. Please check  ***
       ***   the documentation of the specific model used.  ***
       ***                                                  ***
       ***     A list of implemented material types and     ***
       ***       constitutive equations is provided by      ***
       ***         'list_implemented_materials'.            ***


    * 'mesh'
       * 'mesh_file' : str, dolfin.Mesh
            Name of the file containing the mesh information of
            the problem geometry, or a dolfin.Mesh object. Supported
            file formats are *.xml, *.xml.gz, and *.h5.
       * 'mesh_function' : str, dolfin.MeshFunction
            Name of the file containing the mesh function information
            of the geometry, or a dolfin.MeshFunction object. Supported
            file formats are *.xml, *.xml.gz, and *.h5. This mesh function
            will be used to mark different regions of the domain boundary.
       * 'element' : str
            Name of the finite element to be used for the discrete
            function space. Currently, elements of the form 'p<n>-p<m>'
            are supported, where <n> is the degree used for the vector
            function space, and <m> is the degree used for the scalar
            function space. If the material is not incompressible, only the
            first term should be specified. E.g., 'p2-p1'.


    * 'formulation'
        * 'time' (OPTIONAL)
            * 'unsteady' : bool
                True if the problem is time dependent, and False otherwise.
            * 'dt' : float
                Time step used for the numerical integrator.
            * 'interval' : list, tuple
                A list or tuple of length 2 specifying the time interval,
                i.e. [t0, tf].
            * 'theta': float, int
                The weight given to the current time step and subtracted
                from the previous, i.e.

                  dy/dt = theta*f(y_{n+1}) + (1 - theta)*f(y_n).

                Note: theta = 1 gives a fully implicit scheme, while
                theta = 0 gives a fully explicit one.
            * 'beta' : float, int
                The beta parameter used in the Newmark integration scheme.
                Note: the Newmark integration scheme is only used by
                SolidMechanicsProblem.
            * 'gamma' : float, int
                The gamma parameter used in the Newmark integration scheme.
                Note: the Newmark integration scheme is only used by
                SolidMechanicsProblem.
        * 'initial_condition' (OPTIONAL)
            * 'displacement' : dolfin.Coefficient (OPTIONAL)
                A dolfin.Coefficient object specifying the initial value for
                the displacement.
            * 'velocity' : dolfin.Coefficient (OPTIONAL)
                A dolfin.Coefficient object specifying the initial value for
                the velocity.
            * 'pressure' : dolfin.Coefficient (OPTIONAL)
                A dolfin.Coefficient object specifying the initial value for
                the pressure.
        * 'domain' : str
            String specifying whether the problem is to be formulated
            in terms of Lagrangian, Eulerian, or ALE coordinates. Note:
            ALE is currently not supported. The string must be all lower-case.
        * 'inverse' : bool
            True if the problem is an inverse elastostatics problem, and False
            otherwise. For more information, see Govindjee and Mihalic (1996 &
            1998).
        * 'body_force' : dolfin.Coefficient (OPTIONAL)
            Value of the body force throughout the body.
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
                * 'pressure' : list, tuple
                    List of pressure values (dolfin.Constant or dolfin.Expression)
                    for each Dirichlet boundary region specified. The order must match
                    the order used in the list of pressure region IDs.
                * 'regions' : list, tuple
                    List of the region IDs on which Dirichlet boundary conditions for
                    displacement and velocity are to be imposed. These IDs must match
                    those used by the mesh function provided. The order must match that
                    used in the list of values (velocity and displacement).
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
    * 'initial_condition': the initial condition is assumed to be zero for any
        values that are not provided.
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

    def __init__(self, user_config):

        # Check configuration dictionary
        self.config = self.check_config(user_config)

        # Obtain mesh and mesh function
        self.mesh = load_mesh(self.config['mesh']['mesh_file'])
        self.mesh_function = load_mesh_function(self.config['mesh']['mesh_function'],
                                                self.mesh)

        return None


    def check_config(self, user_config):
        """
        Check that all parameters provided in 'user_config' are valid
        based on the current capabilities of the package. An exception
        is raised when a parameter (or combination of parameters) is
        (are) found to be invalid. Please see the documentation of
        BaseMechanicsProblem for detailed information on the required
        values.


        Parameters
        ----------

        user_config : dict
            Dictionary describing the formulation of the mechanics
            problem to be simulated. Check the documentation of
            BaseMechanicsProblem to see the format of the dictionary.


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

        self.check_initial_condition(config)

        return config


    def check_time_params(self, config):
        """
        Check the time parameters provided by the user in the config dictionary.
        Things this function does:

        - If the problem is steady, the theta and dt values are set to 1.
        - If the problem is unsteady and no value was provided for theta,
          an exception is raised.
        - If the problem is unsteady, and a theta outside of the interval
          [0, 1] is provided, an exception is raised.


        Parameters
        ----------

        config : dict
            Dictionary describing the formulation of the mechanics
            problem to be simulated. Check the documentation of
            BaseMechanicsProblem to see the format of the dictionary.


        Returns
        -------

        None

        """

        # Exit function if problem was specified as steady.
        if not config['formulation']['time']['unsteady']:
            config['formulation']['time']['theta'] = 1
            config['formulation']['time']['dt'] = 1
            return None

        if not isinstance(config['formulation']['time']['dt'], (float,dlf.Constant)):
            s1 = 'The \'dt\' parameter must be a scalar value of type: ' \
                 + 'dolfin.Constant, float'
            raise TypeError(s1)

        # if config['formulation']['time']['integrator'] == 'generalized_theta':
        try:
            theta = config['formulation']['time']['theta']
            if theta < 0.0 or theta > 1.0:
                s1 = 'The value of theta for the generalized theta ' \
                     + 'method must be between 0 and 1. The value ' \
                     + 'provided was: %.4f ' % theta
                raise ValueError(s1)
        except KeyError:
            s1 = 'The value of theta for the generalized theta ' \
                 + 'method must be provided.'
            raise KeyError(s1)

        return None


    def check_material_const_eqn(self, config):
        """
        Check if the material type and the specific constitutive equation
        specified in the config dictionary are implemented, unless a class
        is provided. An exception is raised if an unknown material type
        and/or constitutive equation name is provided.


        Parameters
        ----------

        config : dict
            Dictionary describing the formulation of the mechanics
            problem to be simulated. Check the documentation of
            BaseMechanicsProblem to see the format of the dictionary.


        Returns
        -------

        None


        """

        # Exit if user provided a material class.
        if isclass(config['material']['const_eqn']):
            return None

        # Exit if value is neither a class or string.
        if not isinstance(config['material']['const_eqn'], str):
            s = 'The value of \'const_eqn\' must be a class ' \
                + 'or string.'
            raise TypeError(s)

        # Check if the material type is implemented.
        if config['material']['type'] not in _implemented['materials']:
            s1 = 'The class of materials, \'%s\', has not been implemented.' \
                 % config['material']['type']
            raise NotImplementedError(s1)

        mat_subdict = _implemented['materials'][config['material']['type']]
        const_eqn = config['material']['const_eqn']

        # Check if the constitutive equation is implemented under the
        # type specified.
        if const_eqn not in mat_subdict:
            s1 = 'The constitutive equation, \'%s\', has not been implemented ' \
                 % const_eqn \
                 + 'within the material type, \'%s\'.' % config['material']['type']
            raise NotImplementedError(s1)

        return None


    def check_bcs(self, config):
        """
        Check the boundary conditions provided by the user in the config dictionary.


        Parameters
        ----------

        config : dict
            Dictionary describing the formulation of the mechanics
            problem to be simulated. Check the documentation of
            BaseMechanicsProblem to see the format of the dictionary.


        Returns
        -------

        None

        """

        # Check if 'bcs' key is in config dictionary.
        if 'bcs' not in config['formulation']:
            config['formulation']['bcs'] = None

        # Set 'dirichlet' and 'neumann' to None if values were not provided
        # and exit.
        if config['formulation']['bcs'] is None:
            config['formulation']['bcs']['dirichlet'] = None
            config['formulation']['bcs']['neumann'] = None
            print ('*** No BCs (Neumann and Dirichlet) were specified. ***')
            return None

        self.check_dirichlet(config)
        self.check_neumann(config)

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
            BaseMechanicsProblem to see the format of the dictionary.


        Returns
        -------

        None


        """

        if 'dirichlet' not in config['formulation']['bcs']:
            config['formulation']['bcs']['dirichlet'] = None

        if config['formulation']['bcs']['dirichlet'] is None:
            print ('*** No Dirichlet BCs were specified. ***')
            return None

        vel = 'velocity'
        disp = 'displacement'
        subconfig = config['formulation']['bcs']['dirichlet']

        # Make sure the appropriate Dirichlet BCs were specified for the type
        # of problem:
        # - Velocity & displacement for unsteady elastic
        # - Velocity for steady viscous
        # - Displacement for steady elastic
        if config['formulation']['time']['unsteady'] \
           and config['material']['type'] == 'elastic':
            if (vel not in subconfig) or (disp not in subconfig):
                pass
                # s1 = 'Dirichlet boundary conditions must be specified for ' \
                #      + 'both velocity and displacement when the problem is ' \
                #      + 'unsteady. Only %s BCs were provided.'
                # if vel not in subconfig:
                #     s1 = s1 % disp
                # else:
                #     s1 = s1 % vel
                # raise ValueError(s1)
        elif config['material']['type'] == 'elastic':
            if disp not in subconfig:
                s1 = 'Dirichlet boundary conditions must be specified for ' \
                     + ' displacement when solving a quasi-static elastic problem.'
                raise ValueError(s1)
        elif config['material']['type'] == 'viscous':
            if vel not in subconfig:
                s1 = 'Dirichlet boundary conditions must be specified for ' \
                     + ' velocity when solving a quasi-static viscous problem.'
                raise ValueError(s1)

        # Make sure the length of all the lists match.
        if not self.__check_bc_params(subconfig):
            raise ValueError('The number of Dirichlet boundary regions and ' \
                             + 'values for not match!')

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
            BaseMechanicsProblem to see the format of the dictionary.


        Returns
        -------

        None

        """

        # Set value to None if it was not provided.
        if 'neumann' not in config['formulation']['bcs']:
            config['formulation']['bcs']['neumann'] = None

        # Exit if Neumann BCs were not specified.
        if config['formulation']['bcs']['neumann'] is None:
            print ('*** No Neumann BCs were specified. ***')
            return None

        # Make sure that a list for all keys was provided (types, regions, values).
        for t in ['types','regions','values']:
            if t not in config['formulation']['bcs']['neumann']:
                s1 = 'A list of values for %s must be provided.' % t
                raise ValueError(s1)

        # Make sure the length of all the lists match.
        if not self.__check_bc_params(config['formulation']['bcs']['neumann']):
            raise ValueError('The number of Neumann boundary regions, types ' \
                             + 'and values do not match!')

        # Make sure all Neumann BC types are supported with domain specified.
        # Make sure they're all lower case.
        neumann_types = map(str.lower, config['formulation']['bcs']['neumann']['types'])
        config['formulation']['bcs']['neumann']['types'] = neumann_types

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

        return None


    def check_initial_condition(self, config):
        """
        Check if initial condition values were provided. Insert 'None' for the
        values that were not provided.


        Parameters
        ----------

        config : dict
            Dictionary describing the formulation of the mechanics
            problem to be simulated. Check the documentation of
            BaseMechanicsProblem to see the format of the dictionary.


        Returns
        -------

        None


        """

        if ('initial_condition' not in config['formulation']) \
           or (config['formulation']['initial_condition'] is None):
            config['formulation']['initial_condition'] = {
                'displacement': None,
                'velocity': None,
                'pressure': None
            }
            return None

        initial_condition = config['formulation']['initial_condition']
        if 'displacement' not in initial_condition:
            initial_condition['initial_condition'] = None
        if 'velocity' not in initial_condition:
            initial_condition['velocity'] = None
        if 'pressure' not in initial_condition:
            initial_condition['pressure'] = None

        return None


    def update_time(self, t, t0=None):
        """
        Update the time parameter in the BCs that depend on time explicitly.
        Also, the body force expression if necessary.


        Parameters
        ----------

        t : float
            The value to which the user wishes to update the time to.
        t0 : float (default None)
            The previous time value.


        Returns
        -------

        None

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


        Parameters
        ----------

        t : float
            The value to which the users wishes to update the time to.


        Returns
        -------

        None


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
        explicitly.


        Parameters
        ----------

        t : float
            The value to which the user wishes to update the time to.
        t0 : float (default None)
            The previous time value.


        Returns
        -------

        None

        """

        if self.ufl_neumann_bcs is not None:
            self.update_form_time(self.ufl_neumann_bcs, t)

        if self.ufl_neumann_bcs0 and (t0 is not None):
            self.update_form_time(self.ufl_neumann_bcs0, t0)

        return None


    def update_bodyforce_time(self, t, t0=None):
        """
        Update the time parameter in the body force expression if it depends
        on time explicitly.


        Parameters
        ----------

        t : float
            The value to which the user wishes to update the time to.
        t0 : float (default None)
            The previous time value.


        Returns
        -------

        None


        """

        if self.ufl_body_force is not None:
            self.update_form_time(self.ufl_body_force, t)

        if self.ufl_body_force0 and (t0 is not None):
            self.update_form_time(self.ufl_body_force0, t0)

        return None


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
    def define_ufl_neumann_form(regions, types, values, domain,
                                mesh, mesh_function, F, J, xi):
        """
        Define the UFL object representing the variational form of the
        Neumann boundary.


        Parameters
        ----------

        regions : list, tuple
            List of the region IDs on which Neumann boundary conditions are to
            be imposed. These IDs must match those used by the mesh function
            provided. The order must match the order used in the list of types
            and values.
        types : list, tuple
            List of strings specifying whether a 'pressure', 'piola',
            or 'cauchy' is provided for each region. The order
            must match the order used in the list of region IDs
            and values.
        values : list, tuple
            List of values (dolfin.Constant or dolfin.Expression)
            for each Dirichlet boundary region specified. The order
            must match the order used in the list of region IDs
            and types.
        domain : str
            String specifying whether the problem is to be formulated
            in terms of Lagrangian, Eulerian, or ALE coordinates. Note:
            ALE is currently not supported.
        mesh : dolfin.Mesh
            Mesh object used to define a measure.
        mesh_function : dolfin.MeshFunction
            Mesh function used to tag different regions of the domain
            boundary.
        F : ufl object
            Deformation gradient.
        J : ufl object
            Determinant of the deformation gradient.
        xi : dolfin.Argument
            Test function used in variational formulation.


        Returns
        -------

        neumann_form : ufl.Form
            The UFL Form object defining the variational term(s) corresponding
            to the Neumann boundary conditions.

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

            if region == 'all':
                ds_region = dlf.ds
            else:
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
        Update the time of the coefficient objects that depend on it.


        Parameters
        ----------

        form : ufl.Form
            Variational form for which the time is to be updated.
        t : float
            The value to which the time is to be updated.


        Returns
        -------

        None

        """

        coeffs = form.coefficients()
        for expr in coeffs:
            if hasattr(expr, 't'):
                expr.t = t

        return None


    @staticmethod
    def apply_initial_conditions(init_value, function, function0=0):
        """
        Assign the initial values to field variables.


        Parameters
        ----------

        init_value : dolfin.Coefficient, dolfin.Expression
            A function/expression that approximates the initial condition.
        function : dolfin.Function
            The function approximating a field variable in a mechanics problem.
        function0 : dolfin.Function
            The function approximatinga field variable at the previous time
            step in a mechanics problem.


        Returns
        -------

        None


        """

        function.assign(init_value)
        if function0 != 0:
            function0.assign(init_value)

        return None
