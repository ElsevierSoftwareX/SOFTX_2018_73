import re
import dolfin as dlf
import materials

class MechanicsProblem(dlf.NonlinearVariationalProblem):
    """
    This class represents the variational form of a continuum
    mechanics problem. The user The specific form and boundary
    conditions are generated based on definition provided by the
    user in a dictionary of sub-dictionaries. The keys for the
    dictionary and their descriptions are shown below:

    -'mechanics'
    ----'const_eqn' : str
            The name of the constitutive equation to be
            used. The name must match a function name in
            the fenicsmechanics.materials subpackage, which
            returns the stress tensor.
    ----'material'
    --------'type' : str
                The class of material that will be used, e.g.
                elastic, viscous, viscoelastic, etc. The name
                must match the name of a module inside of the
                fenicsmechanics.materials sub-package.
    --------'incompressible' : bool
                True if the material is incompressible. An
                additional weak form for the incompressibility
                constraint will be added to the problem.
    --------'density' : float, int, dolfin.Constant
                Scalar specifying the density of the material.
    --------'lambda' : float, int, dolfin.Constant
                Scalar value used in constitutive equations. E.g.,
                it is the first Lame parameter in linear elasticity.
    --------'mu' : float, int, dolfin.Constant
                Scalar value used in constitutive equations. E.g.,
                it is the second Lame parameter in linear elasticity,
                and the dynamic viscosity for a Newtonian fluid.
    --------'kappa' : float, int, dolfin.Constant
                Scalar value of the penalty parameter for incompressibility.

    -'mesh'
    ----'mesh_file' : str
            Name of the file containing the mesh information of
            the problem geometry.
    ----'mesh_function' : str
            Name of the file containing the mesh function information
            of the geometry. A mesh function is typically used to
            label different regions of the mesh.
    ----'element' : str
            Name of the finite element to be used for the discrete
            function space. E.g., 'p2-p1'.

    -'formulation'
    ----'unsteady' : bool
            True if the problem is time dependent.
    ----'initial_condition' : subclass of dolfin.Expression
            An expression specifying the initial value of the
            solution to the problem.
    ----'domain' : str
            String specifying whether the problem is to be formulated
            in terms of Lagrangian, Eulerian, or ALE coordinates.
    ----'inverse' : bool
            True if the problem is an inverse elastostatics problem.
    ----'body_force' : dolfin.Expression, dolfin.Constant
            Value of the body force throughout the body.
    ----'bcs'
    --------'dirichlet'
    ------------'regions' : list, tuple
                    List of the region IDs (int) on which Dirichlet
                    boundary conditions are to be imposed. These IDs
                    must match those used by the mesh function provided.
                    The order must match the order used in the list of
                    unsteady booleans, and values.
    ------------'unsteady' : list, tuple
                    List of booleans specifying whether each region is time
                    dependent (True) or not (False). The order must match
                    the order used in the list of region IDs, and values.
    ------------'values' : list, tuple
                    List of values (dolfin.Constant or dolfin.Expression)
                    for each region Dirichlet boundary region specified.
                    The order must match the order used in the list of
                    region IDs, and unsteady booleans.
    --------'neumann'
    ------------'regions' : list, tuple
                    List of the region IDs (int) on which Neumann
                    boundary conditions are to be imposed. These IDs
                    must match those used by the mesh function provided.
                    The order must match the order used in the list of
                    types, unsteady booleans, and values.
    ------------'types' : list, tuple
                    List of strings specifying whether a 'pressure', 'piola',
                    or 'cauchy' is provided for each region. The order
                    must match the order used in the list of region IDs,
                    unsteady booleans, and values.
    ------------'unsteady' : list, tuple
                    List of booleans specifying whether each region is time
                    dependent (True) or not (False). The order must match
                    the order used in the list of region IDs, types, and
                    values.
    ------------'values' : list, tuple
                    List of values (dolfin.Constant or dolfin.Expression)
                    for each Dirichlet boundary region specified. The order
                    must match the order used in the list of region IDs,
                    types, and unsteady booleans.

    """

    def __init__(self, config, **kwargs):

        self.config = config

        # Obtain mesh and mesh function (should this be member data?)
        self.mesh = self.load_mesh()
        self.mesh_function = self.load_mesh_function()

        # Define the finite element(s) (maybe add
        # functionality to specify different families?)
        if config['mesh']['element'] is None:
            raise ValueError('You need to specify the type of element(s) to use!')

        # SHOULD PROBABLY CHECK THAT INCOMPRESSIBLE IS ALSO SPECIFIED
        fe_list = re.split('-|_| ', config['mesh']['element'])
        if len(fe_list) > 2 or len(fe_list) == 0:
            s1 = 'The current formulation allows 1 or 2 fields.\n'
            s2 = 'You provided %i. Check config[\'mesh\'][\'element\'].' % len(fe_list)
            raise NotImplementedError(s1 + s2)
        elif len(fe_list) == 2:
            # Make sure material was specified as incompressible if two
            # element types are given.
            if not config['mechanics']['material']['incompressible']:
                s1 = 'Two element types, \'%s\', were specified ' % config['mesh']['element'] \
                     +'for a compressible material.'
                raise ValueError(s1)
            P_u = dlf.VectorElement('CG', self.mesh.ufl_cell(), int(fe_list[0][-1]))
            P_p = dlf.FiniteElement('CG', self.mesh.ufl_cell(), int(fe_list[1][-1]))
            element = P_u * P_p
        else:
            # Make sure material was specified as compressible if only
            # one element type was given.
            if config['mechanics']['material']['incompressible']:
                s1 = 'Only one element type, \'%s\', was specified ' % config['mesh']['element'] \
                     + 'for an incompressible material.'
                raise ValueError(s1)
            element = dlf.VectorElement('CG', self.mesh.ufl_cell(), int(fe_list[0][-1]))

        # Define the function space (already data in NonlinearVariationalProblem)
        functionSpace = dlf.FunctionSpace(self.mesh, element)

        # Check that number of BC regions and BC values match
        dirichlet = config['formulation']['bcs']['dirichlet']
        if not self.__check_bcs(dirichlet):
            raise ValueError('The number of Dirichlet boundary regions and ' \
                             + 'values do not match!')
        neumann = config['formulation']['bcs']['neumann']
        if not self.__check_bcs(neumann):
            raise ValueError('The number of Neumann boundary regions and ' \
                             + 'values do not match!')

        # Identity tensor for later use
        I = dlf.Identity(self.mesh.geometry().dim())

        # Material properties
        rho = config['mechanics']['material']['density']
        la = config['mechanics']['material']['lambda']
        mu = config['mechanics']['material']['mu']

        # Define functions that are the same for both cases
        sys_u = dlf.Function(functionSpace)
        sys_du = dlf.TrialFunction(functionSpace)

        # Initial condition if provided
        if config['formulation'].has_key('initial_condition'):
            sys_u.interpolate(config['formulation']['initial_condition'])

        # Initialize the weak form
        weak_form = 0

        # Check if material is incompressible, and then formulate problem
        if config['mechanics']['material']['incompressible']:
            # Define Dirichlet BCs
            bc_list = self.define_dirichlet_bcs(functionSpace.sub(0)) # , self.mesh_function)

            # Define test function
            sys_v = dlf.TestFunction(functionSpace)

            # Obtain pointers to each sub function
            u, p = dlf.split(sys_u)
            xi, q = dlf.split(sys_v)

            # Make the pressure member data
            self.pressure = p

            if config['mechanics']['material']['type'] == 'elastic':
                self.deformationGradient = I + dlf.grad(u)
                self.deformationRateGradient = None
                self.jacobian = dlf.det(self.deformationGradient)
            else:
                s1 = 'The material type, %s, has not been implemented!' \
                     % config['mechanics']['material']['type']
                raise NotImplementedError(s1)

            # Penalty parameter for incompressibility
            kappa = config['mechanics']['material']['kappa']

            # Weak form corresponding to incompressibility
            if config['mechanics']['const_eqn'] == 'lin_elastic':
                # Incompressibility constraint for linear elasticity
                weak_form += (dlf.div(u) - la*p)*q*dlf.dx
            else:
                # Incompressibility constraint for nonlinear
                weak_form += (p - kappa*(self.jacobian - 1.0))*q*dlf.dx

        else:
            # Define Dirichlet BCs
            bc_list = self.define_dirichlet_bcs(functionSpace) #, mesh_function)

            xi = dlf.TestFunction(functionSpace)

            if config['mechanics']['material']['type'] == 'elastic':
                self.deformationGradient = I + dlf.grad(sys_u)
                self.deformationRateGradient = None
                self.jacobian = dlf.det(self.deformationGradient)
            else:
                s1 = 'The material type, \'%s\', has not been implemented!' \
                     % config['mechanics']['material']['type']
                raise NotImplementedError(s1)

        # Get access to the function defining the stress tensor
        material_submodule = getattr(materials, config['mechanics']['material']['type'])
        stress_function = getattr(material_submodule, config['mechanics']['const_eqn'])

        # Define the stress tensor
        stress_tensor = stress_function(self)

        # Add contributions from external forces to the weak form
        weak_form -= self.define_neumann_bcs(xi) # mesh, mesh_function, xi)
        weak_form -= rho*dlf.dot(xi, config['formulation']['body_force']) * dlf.dx

        weak_form += dlf.inner(dlf.grad(xi), stress_tensor)*dlf.dx
        weak_form_deriv = dlf.derivative(weak_form, sys_u, sys_du)

        # Initialize NonlinearVariationalProblem object
        dlf.NonlinearVariationalProblem.__init__(self, weak_form,
                                                 sys_u, bc_list,
                                                 weak_form_deriv,
                                                 **kwargs)

        return None


    def load_mesh(self, mesh_file=None):
        """
        Load the mesh file specified by the call to the function, or in the
        problem config dictionary. If mesh_file is None, and the mesh for the
        current object has already been loaded, the mesh stored as member
        data is returned.


        Parameters
        ----------

        mesh_file : str (default None)
            Name of the file that contains the desired mesh information.


        Returns
        -------
        mesh : dolfin.cpp.mesh.Mesh
            This function returns a dolfin mesh object.

        """

        if mesh_file is None:
            mesh_file = self.config['mesh']['mesh_file']

        if hasattr(self, 'mesh'):
            mesh = self.mesh
        else:
            if mesh_file[-3:] == '.h5':
                mesh = self.__load_mesh_hdf5(mesh_file)
            else:
                mesh = dlf.Mesh(mesh_file)

        return mesh


    def load_mesh_function(self, mesh_function=None):
        """
        Load the mesh function file specified by the call to the function, or
        in the problem config dictionary. If mesh_function is None, and the
        mesh function for the current object has already been loaded, the mesh
        function stored as member data is returned.

        Parameters
        ----------

        mesh_function : str (default None)
            Name of the file that contains the desired mesh function information.


        Returns
        -------

        mesh_func : dolfin.cpp.mesh.MeshFunctionSizet
        This function returns a dolfin mesh function object.

        """

        if mesh_function is None:
            mesh_function = self.config['mesh']['mesh_function']

        if hasattr(self, 'mesh_function'):
            mesh_func = self.mesh_function
        else:
            if mesh_function[-3:] == '.h5':
                mesh_func = self.__load_mesh_function_hdf5(mesh_function, self.mesh)
            else:
                mesh_func = dlf.MeshFunction('size_t', self.mesh,
                                             self.config['mesh']['mesh_function'])

        return mesh_func


    @staticmethod
    def __load_mesh_hdf5(mesh_file):
        """
        Load dolfin mesh from an HDF5 file.

        Parameters
        ----------

        mesh_file : str
            Name of the file containing the mesh information

        """

        # Check dolfin for HDF5 support
        if not dlf.has_hdf5():
            s1 = 'The current installation of dolfin does not support HDF5 files.'
            raise Exception(s1)

        # Check file extension
        if mesh_file[-3:] == '.h5':
            f = dlf.HDF5File(dlf.mpi_comm_world(), mesh_file, 'r')
            mesh = dlf.Mesh()
            f.read(mesh, 'mesh', False)
        else:
            s1 = 'The file extension provided must be \'.h5\'.'
            raise ValueError(s1)

        return mesh


    @staticmethod
    def __load_mesh_function_hdf5(mesh_function, mesh):
        """
        Load a dolfin mesh function from an HDF5 file.

        Parameters
        ----------

        mesh_function : str
            Name of the file containing the mesh function information

        """

        # Check dolfin for HDF5 support
        if not dlf.has_hdf5():
            s1 = 'The current installation of dolfin does not support HDF5 files.'
            raise Exception(s1)

        # Check file extension
        if mesh_function[-3:] == '.h5':
            f = dlf.HDF5File(dlf.mpi_comm_world(), mesh_function, 'r')
            mesh_func = dlf.MeshFunction('size_t', mesh)
            f.read(mesh_func, 'mesh_function')
        else:
            s1 = 'The file extension provided must be \'.h5\'.'
            raise ValueError(s1)

        return mesh_func


    # def define_dirichlet_bcs(self, functionSpace, mesh_function):
    def define_dirichlet_bcs(self, functionSpace):
        """


        """

        bc_list = list()
        for region, value in zip(self.config['formulation']['bcs']['dirichlet']['regions'],
                                 self.config['formulation']['bcs']['dirichlet']['values']):
            bc_list.append(dlf.DirichletBC(functionSpace, value, self.mesh_function, region))
        return bc_list


    # def define_neumann_bcs(self, mesh, mesh_function, xi):
    def define_neumann_bcs(self, xi):
        """


        """

        region_list = self.config['formulation']['bcs']['neumann']['regions']
        tt_list = self.config['formulation']['bcs']['neumann']['types']
        value_list = self.config['formulation']['bcs']['neumann']['values']

        if 'pressure' in tt_list or 'cauchy' in tt_list:
            Finv = dlf.inv(self.deformationGradient)
            J = self.jacobian
            N = dlf.FacetNormal(self.mesh)

        if 'pressure' in tt_list:
            # THIS IS ASSUMING THE PROBLEM IS FORMULATED IN LAGRANGIAN COORDINATES
            n = J*Finv.T*N # Nanson's formula

        if 'cauchy' in tt_list:
            # THIS IS ASSUMING THE PROBLEM IS FORMULATED IN LAGRANGIAN COORDINATES
            nanson_mag = J**2 * dlf.sqrt(dlf.dot(Finv.T*N, Finv.T*N))

        total_neumann_bcs = 0

        for region, tt, value in zip(region_list, tt_list, value_list):
            ds_region = dlf.ds(region, domain=self.mesh,
                               subdomain_data=self.mesh_function)
            if tt == 'pressure':
                # THIS IS ASSUMING THE PROBLEM IS FORMULATED IN LAGRANGIAN COORDINATES
                total_neumann_bcs -= dlf.dot(xi, value*n)*ds_region
            elif tt == 'cauchy':
                # THIS IS ASSUMING THE PROBLEM IS FORMULATED IN LAGRANGIAN COORDINATES
                total_neumann_bcs += nanson_mag*dlf.dot(xi, value)*ds_region
            elif tt == 'piola':
                # THIS IS ASSUMING THE PROBLEM IS FORMULATED IN LAGRANGIAN COORDINATES
                total_neumann_bcs += dlf.dot(xi, value)*ds_region
            else:
                raise NotImplementedError('Neumann BC of type %s is not implemented!' % tt)

        return total_neumann_bcs


    @staticmethod
    def __check_bcs(bc_dict):
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
