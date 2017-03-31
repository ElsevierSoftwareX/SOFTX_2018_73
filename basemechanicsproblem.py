import re
import dolfin as dlf

from .utils import load_mesh, load_mesh_function
from .__CONSTANTS__ import dict_implemented as _implemented
from inspect import isclass

__all__ = ['BaseMechanicsProblem']


class BaseMechanicsProblem(object):


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

        if not isinstance(config['formulation']['time']['dt'], (float,dlf.Constant)):
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
            MechanicsProblem to see the format of the dictionary.


        """

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
                s1 = 'Dirichlet boundary conditions must be specified for ' \
                     + 'both velocity and displacement when the problem is ' \
                     + 'unsteady. Only %s BCs were provided.'
                if vel not in subconfig:
                    s1 = s1 % disp
                else:
                    s1 = s1 % vel
                raise ValueError(s1)
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
            MechanicsProblem to see the format of the dictionary.


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
        neumann_types = map(str.lower, config['formulation']['bcs']['neumann']['types'])
        config['formulation']['bcs']['neumann']['types'] = neumann_types # Make sure they're all lower case.

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
