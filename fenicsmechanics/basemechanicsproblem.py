from __future__ import print_function

import re
import dolfin as dlf

from .utils import load_mesh, load_mesh_function, _read_write_hdf5
from .__CONSTANTS__ import dict_implemented as _implemented
from .exceptions import *
from inspect import isclass

__all__ = ['BaseMechanicsProblem']


class BaseMechanicsProblem(object):
    """
    This is the base class for mechanics problems. Checking validity of the
    'config' dictionary provided by users is done at this level since all
    mechanics problems are derived from this class. The derived classes will
    then define the variational problem using the FEniCS UFL language.

    For details on the format of the :code:`config` dictionary, check the
    documentation of the FEniCS Mechanics module by executing

    >>> import fenicsmechanics as fm
    >>> help(fm)


    """

    def __init__(self, user_config):

        # Check configuration dictionary
        self.config = self.check_config(user_config)

        # Obtain mesh and mesh function
        mesh_file = self.config['mesh']['mesh_file']
        boundaries = self.config['mesh']['boundaries']
        if (mesh_file == boundaries) and mesh_file[-3:] == ".h5":
            self.mesh = dlf.Mesh()
            self.boundaries = dlf.MeshFunction("size_t", self.mesh)
            _read_write_hdf5("r", mesh_file, mesh=self.mesh,
                             boundaries=self.boundaries)
        else:
            self.mesh = load_mesh(mesh_file)
            self.boundaries = load_mesh_function(boundaries, self.mesh)

        return None


    @property
    def class_name(self):
        return self._class_name


    @class_name.setter
    def class_name(self, name):
        self._class_name = name


    def check_config(self, user_config):
        """
        Check that all parameters provided in 'user_config' are valid
        based on the current capabilities of the package. An exception
        is raised when a parameter (or combination of parameters) is
        (are) found to be invalid. Please see the documentation of
        fenicsmechanics for detailed information on the required
        values.


        Parameters
        ----------

        user_config : dict
            Dictionary describing the formulation of the mechanics
            problem to be simulated. Check the documentation of
            BaseMechanicsProblem to see the format of the dictionary.


        Returns
        -------

        config : dict
            A copy of user_config with possibly new keys that are needed
            if they were not originally provided.


        """

        # Use a copy to avoid altering the original object
        config = user_config.copy()

        # Check mesh file names given.
        self.check_mesh(config)

        # Check the finite element specified.
        self.check_finite_element(config)

        # Check the material type specified.
        self.check_material_type(config)

        # Check if the domain specified is implemented for the material type.
        self.check_domain(config)

        # Check if material type has been implemented.
        self.check_material_const_eqn(config)

        # Check the parameters given for time integration.
        self.check_time_params(config)

        # Make sure that the BC dictionaries have the same
        # number of regions, values, etc., if any were
        # specified. If they are not specified, set them to None.
        self.check_bcs(config)

        # Check if body force was provided. Assume zero if not.
        if 'body_force' not in config['formulation']:
            config['formulation']['body_force'] = None

        self.check_initial_condition(config)

        return config


    def check_mesh(self, config):
        """


        """

        if 'mesh_file' not in config['mesh']:
            msg = "The name of a mesh file, or a dolfin.Mesh object, " \
                  + "must be provided."
            raise RequiredParameter(msg)

        return None


    def check_finite_element(self, config):
        """
        Check the finite element specified for the numerical formulation of
        the problem.


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

        # Check if the finite element type is provided.
        if 'element' not in config['formulation']:
            raise RequiredParameter("You need to specify the type of finite " \
                                    + "element(s) to use.")
        else:
            _check_type(config['formulation']['element'],
                        (str, list, tuple), "element")

        # Make sure at most two element types are specified.
        if isinstance(config['formulation']['element'], str):
            fe_list = re.split("-|_| ", config['formulation']['element'])
        else:
            fe_list = config['formulation']['element']

        if 'incompressible' not in config['material']:
            msg = "You must specify if the material is incompressible or not."
            raise RequiredParameter(msg)
        else:
            _check_type(config['material']['incompressible'],
                        (bool,), "incompressible")

        len_fe_list = len(fe_list)
        if len_fe_list == 0 or len_fe_list > 2:
            msg = "The current formulation allows 1 or 2 fields.\n"
            msg += "You provided %i. Check config['formulation']['element']." % len_fe_list
            raise NotImplementedError(msg)
        elif len_fe_list == 1 and config['material']['incompressible']:
            msg = "Only one element type, '%s', was specified " \
                  % config['formulation']['element'] \
                  + "for an incompressible material."
            raise InconsistentCombination(msg)
        elif len_fe_list == 2 and not config['material']['incompressible']:
            msg = "Two element types, '%s', were specified " % config['formulation']['element'] \
                  + "for a compressible material."
            raise InconsistentCombination(msg)
        else:
            # Replace with list in case it was originally a string
            config['formulation']['element'] = fe_list

        # Check to make sure all strings are only 2 characters long.
        str_len = set(map(len, config['formulation']['element']))
        msg = "Element types must be of the form 'p<int>', where <int> " \
              + "is the polynomial degree to be used." # error string

        # All strings should have the same number of characters.
        if not len(str_len) == 1:
            raise InvalidCombination(msg)

        # All strings should consist of two characters.
        str_len_val = str_len.pop()
        if not str_len_val == 2:
            raise ValueError(msg)

        # Check to make sure first character is 'p'
        if not fe_list[0][0] == "p":
            msg = "The finite element family, '%s', has not been implemented." \
                  % fe_list[0][0]
            raise NotImplementedError(msg)

        return None


    def check_material_type(self, config):
        """
        Check if the material type specified is supported.


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

        if 'type' not in config['material']:
            num_types = len(_implemented['materials'])
            msg = "The material type must be specified. Types currently " \
                  + "supported are: " + ", ".join(["%s"]*num_types) \
                  % tuple(_implemented['materials'].keys())
            raise RequiredParameter(msg)
        else:
            _check_type(config['material']['type'], (str,), 'type')

        # Check if the material type is implemented.
        if config['material']['type'] not in _implemented['materials']:
            msg = "The class of materials, '%s', has not been implemented." \
                  % config['material']['type']
            raise NotImplementedError(msg)

        return None


    def check_domain(self, config):
        """
        Check if the domain formulation specified is implemented for the
        material type.


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

        # Material type should have already been checked, so there should be
        # no issues here.
        mat_type = config['material']['type']

        if 'domain' not in config['formulation']:
            msg = "The domain formulation must be specified. " \
                  "Formulations currently supported are: eulerian, lagrangian"
            raise RequiredParameter(msg)
        else:
            _check_type(config['formulation']['domain'], (str,), 'domain')

        domain = config['formulation']['domain']
        if domain not in ["lagrangian", "eulerian"]:
            if domain.lower() == "ale":
                msg = 'Formulation with respect to \'%s\' coordinates is not supported.' \
                      % config['formulation']['domain']
                raise NotImplementedError(msg)
            else:
                msg = 'Formulation with respect to \'%s\' coordinates is not recognized.' \
                      % config['formulation']['domain']
                raise InvalidOption(msg)

        if ((mat_type == "elastic") and (domain == "eulerian")) \
           or ((mat_type == "viscous") and (domain == "lagrangian")):
            msg = "%s formulation for %s materials is not supported." \
                  % (domain.capitalize(), mat_type)
            raise InvalidCombination(msg)

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

        if 'const_eqn' not in config['material']:
            msg = "The constitutive equation must be specified under the key " \
                  + "'const_eqn' of the 'material' subdictionary."
            raise RequiredParameter(msg)

        # Exit if user provided a material class.
        if isclass(config['material']['const_eqn']):
            return None

        # Exit if value is neither a class or string.
        if not isinstance(config['material']['const_eqn'], str):
            msg = 'The value of \'const_eqn\' must be a class ' \
                  + 'or string.'
            raise TypeError(msg)

        mat_subdict = _implemented['materials'][config['material']['type']]
        const_eqn = config['material']['const_eqn']

        # Check if the constitutive equation is implemented under the
        # type specified.
        if const_eqn not in mat_subdict:
            msg = 'The constitutive equation, \'%s\', has not been implemented ' \
                  % const_eqn \
                  + 'within the material type, \'%s\'.' % config['material']['type']
            raise InvalidCombination(msg)

        if 'inverse' not in config['formulation']:
            config['formulation']['inverse'] = False

        return None


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

        # Check if the 'time' subdictionary was provided. If it wasn't,
        # We assume the problem is steady since there is no time integration
        # information to use.
        if 'time' not in config['formulation']:
            config['formulation']['time'] = dict()
            steady = True
        else:
            # If the user provided a Boolean value for 'unsteady', use it
            # to determine if we should set values for a steady problem.
            # Else, check that they provided both 'dt' and 'tspan'. If not,
            # we raise an exception.
            if 'unsteady' in config['formulation']['time']:
                steady = not config['formulation']['time']['unsteady']
            else:
                if 'dt' in config['formulation']['time'] \
                   and 'interval' in config['formulation']['time']:
                    config['formulation']['time']['unsteady'] = True
                    steady = False
                else:
                    msg = "Need to specify a time step, 'dt', and a time " \
                          + "interval, 'interval', in the 'time' subdictionary " \
                          + "of 'formulation' in order to run a time-dependent " \
                          + "simulation."
                    raise RequiredParameter(msg)

        # Set theta and dt to 1 if the problem is steady, and exit this
        # function.
        if steady:
            config['formulation']['time']['unsteady'] = False
            config['formulation']['time']['theta'] = 1
            config['formulation']['time']['dt'] = 1
            return None

        # The check for 'dt' and 'interval' should only catch errors when
        # the problem is unsteady since the values for the steady case were
        # already updated above.
        if 'dt' not in config['formulation']['time']:
            msg = "The time step size, 'dt', must be specified."
            raise RequiredParameter(msg)
        if 'interval' not in config['formulation']['time']:
            msg = "The time interval must be specified."
            raise RequiredParameter(msg)

        if not isinstance(config['formulation']['time']['dt'], (float,dlf.Constant)):
            msg = 'The \'dt\' parameter must be a scalar value of type: ' \
                  + 'dolfin.Constant, float'
            raise TypeError(msg)

        # Get rank to only print from process 0
        rank = dlf.MPI.rank(dlf.mpi_comm_world())

        # Check the theta value provided. If none is provided, it is
        # set to 1.0.
        try:
            theta = config['formulation']['time']['theta']
            if theta < 0.0 or theta > 1.0:
                msg = 'The value of theta for the generalized theta ' \
                      + 'method must be between 0 and 1. The value ' \
                      + 'provided was: %.4f ' % theta
                raise ValueError(msg)
        except KeyError:
            if rank == 0:
                print("No value was provided for 'theta'. A value of 1.0 (fully " \
                      + "implicit) was used.")
            config['formulation']['time']['theta'] = 1.0

        # Provide a default value for Newmark scheme parameters
        # if not given. This is only for the SolidMechanicsProblem
        # class.
        if self.class_name == "SolidMechanicsProblem":
            if 'beta' not in config['formulation']['time']:
                if rank == 0:
                    print("No value was provided for 'beta'. A value of 0.25 will be " \
                          + "used for the Newmark integration scheme.")
                config['formulation']['time']['beta'] = 0.25

            if 'gamma' not in config['formulation']['time']:
                if rank == 0:
                    print("No value was provided for 'gamma'. A value of 0.5 will be " \
                          + "used for the Newmark integration scheme.")
                config['formulation']['time']['gamma'] = 0.5

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
            print('*** No BCs (Neumann and Dirichlet) were specified. ***')
            return None
        else:
            if 'boundaries' not in config['mesh']:
                msg = "A facet function must be provided under the 'boundaries'" \
                      + " key of the 'mesh' subdictionary when specifying " \
                      + "boundary conditions for the problem."
                raise RequiredParameter(msg)

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
            # USE THE WARNINGS MODULE HERE.
            print('*** No Dirichlet BCs were specified. ***')
            return None

        vel = 'velocity'
        disp = 'displacement'
        subconfig = config['formulation']['bcs']['dirichlet']

        # Make sure the appropriate Dirichlet BCs were specified for the type
        # of problem:
        # - Velocity & displacement for unsteady elastic (only for MechanicsProblem)
        # - Velocity for steady viscous
        # - Displacement for steady elastic
        if config['formulation']['time']['unsteady'] \
           and config['material']['type'] == 'elastic' \
           and self.class_name == "MechanicsProblem":
            if (vel not in subconfig) or (disp not in subconfig):
                msg = 'Dirichlet boundary conditions must be specified for ' \
                      + 'both velocity and displacement when the problem is ' \
                      + 'unsteady. Only %s BCs were provided.'
                if vel not in subconfig:
                    msg = msg % disp
                else:
                    msg = msg % vel
                raise RequiredParameter(msg)
        elif config['material']['type'] == 'elastic':
            if disp not in subconfig:
                msg = 'Dirichlet boundary conditions must be specified for ' \
                      + ' displacement when solving a quasi-static elastic problem.'
                raise RequiredParameter(msg)
        elif config['material']['type'] == 'viscous':
            if vel not in subconfig:
                msg = 'Dirichlet boundary conditions must be specified for ' \
                      + ' velocity when solving a quasi-static viscous problem.'
                raise RequiredParameter(msg)

        # Make sure the length of all the lists match.
        if not self.__check_bc_params(subconfig):
            raise InconsistentCombination('The number of Dirichlet boundary regions ' \
                                          + 'and values for not match!')

        if config['formulation']['time']['unsteady']:
            t0 = config['formulation']['time']['interval'][0]
        else:
            t0 = 0.0

        if 'displacement' in config['formulation']['bcs']['dirichlet']:
            disps = config['formulation']['bcs']['dirichlet']['displacement']
            vals = self.__convert_bc_values(disps, t0)
            config['formulation']['bcs']['dirichlet']['displacement'] = vals

        if 'velocity' in config['formulation']['bcs']['dirichlet']:
            vels = config['formulation']['bcs']['dirichlet']['velocity']
            vals = self.__convert_bc_values(vels, t0)
            config['formulation']['bcs']['dirichlet']['displacement'] = vals

        if 'pressure' in config['formulation']['bcs']['dirichlet']:
            pressures = config['formulation']['bcs']['dirichlet']['pressure']
            vals = self.__convert_bc_values(pressures, t0)
            config['formulation']['bcs']['dirichlet']['displacement'] = vals

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
            print('*** No Neumann BCs were specified. ***')
            return None

        # Make sure that a list for all keys was provided (types, regions, values).
        keys_not_included = list()
        for t in ['types','regions','values']:
            if t not in config['formulation']['bcs']['neumann']:
                keys_not_included.append(t)

        if keys_not_included:
            msg = 'A list of values for %s must be provided.' % keys_not_included
            raise RequiredParameter(msg)

        # Make sure the length of all the lists match.
        if not self.__check_bc_params(config['formulation']['bcs']['neumann']):
            raise InconsistentCombination('The number of Neumann boundary ' \
                                          + 'regions, types and values do not match!')

        # Make sure all Neumann BC types are supported with domain specified.
        # Make sure they're all lower case. (python 3 does not return a list object here)
        neumann_types = list(map(str.lower, config['formulation']['bcs']['neumann']['types']))
        config['formulation']['bcs']['neumann']['types'] = neumann_types

        # Check that types are valid
        valid_types = {'pressure', 'cauchy', 'piola'}
        union = valid_types.union(neumann_types)
        if len(union) > 3:
            msg = 'At least one Neumann BC type is unrecognized. The type string must'
            msg += ' be one of the three: ' + ', '.join(list(valid_types))
            raise NotImplementedError(msg)

        domain = config['formulation']['domain']
        if domain == 'eulerian' and 'piola' in neumann_types:
            msg = 'Piola traction in an Eulerian formulation is not supported.'
            raise NotImplementedError(msg)

        if config['formulation']['time']['unsteady']:
            t0 = config['formulation']['time']['interval'][0]
        else:
            t0 = 0.0

        orig_values = config['formulation']['bcs']['neumann']['values']
        vals = self.__convert_bc_values(orig_values, t0)
        config['formulation']['bcs']['neumann']['values'] = vals

        return None


    @staticmethod
    def __convert_bc_values(values, t0, degree=1):
        """


        """

        new_values = list()
        for i,val in enumerate(values):
            # No need to convert if already a dolfin.Coefficient type.
            if isinstance(val, dlf.Coefficient):
                new_values.append(val)
                continue

            if isinstance(val, str):
                if "t" in val:
                    expr = dlf.Expression(val, t=t0, degree=degree)
                else:
                    expr = dlf.Expression(val, degree=degree)
                new_values.append(expr)
            elif isinstance(val, (float, int)):
                new_values.append(dlf.Constant(val))
            elif isinstance(val, (list, tuple)):
                # Make sure that all objects in list/tuple are of the same type.
                # (Either all strings, or all scalars)
                component_types = set(list(map(type, val)))
                if len(component_types) != 1:
                    # Raise an error if different types were given in the same
                    # list/tuple.
                    msg = "All components of an expression/constant must " \
                          + "be of the same type."
                    raise TypeError(msg)
                else:
                    # Raise a TypeError if the components are not float,
                    # int, or str.
                    component_type = component_types.pop()
                    valid_types = (float, int, str)
                    if component_type not in valid_types:
                        msg = "The components of each value must be one of the " \
                              + "following types: %s" \
                              % str(obj.__name__ for obj in valid_types)
                        raise TypeError(msg)

                if component_type == str:
                    if "t" in val:
                        expr = dlf.Expression(val, t=t0, degree=degree)
                    else:
                        expr = dlf.Expression(val, t=t0, degree=degree)
                    new_values.append(expr)
                else:
                    new_values.append(dlf.Constant(val))
            else:
                msg = "The type '%s' cannot be used to create a dolfin.Coefficient " \
                      + "object." % val.__class__
                raise TypeError(msg)

        return new_values


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
            print('No Dirichlet BCs to update!')
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

        for key, val in bc_dict.items():
            _check_type(val, (list, tuple), "formulation/bcs/../%s" % key)

        # Make sure the regions list is only ints, floats, bools, or strings.
        # The only valid string is "all".
        for i, r_id in enumerate(bc_dict['regions']):
            if r_id != "everywhere":
                _check_type(r_id, (int, float, bool), "regions[%i]" % i)

        values = bc_dict.values()
        lengths = map(len, values)
        if len(set(lengths)) == 1:
            return True
        else:
            return False


    @staticmethod
    def define_ufl_neumann_form(regions, types, values, domain,
                                mesh, boundaries, F, J, xi):
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
        boundaries : dolfin.MeshFunction
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

            ds_region = dlf.ds(region, domain=mesh, subdomain_data=boundaries)
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


def _check_type(obj, valid_types, key):
    if not isinstance(obj, valid_types):
        msg = "The value given for '%s' must be one of the following " % key \
              + "types: %s" % str(tuple(obj.__name__ for obj in valid_types))
        raise TypeError(msg)
    return None
