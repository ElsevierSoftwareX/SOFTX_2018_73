import re
import dolfin as dlf

from . import materials
from .utils import load_mesh, load_mesh_function

class MechanicsProblem(dlf.NonlinearVariationalProblem):
    """
    This class represents the variational form of a continuum
    mechanics problem. The user The specific form and boundary
    conditions are generated based on definition provided by the
    user in a dictionary of sub-dictionaries. The keys for the
    dictionary and their descriptions are shown below:

    * 'mechanics'
       * 'const_eqn' : str
            The name of the constitutive equation to be
            used. The name must match a function name in
            the fenicsmechanics.materials subpackage, which
            returns the stress tensor.
       * 'material'
           * 'type' : str
                The class of material that will be used, e.g.
                elastic, viscous, viscoelastic, etc. The name
                must match the name of a module inside of the
                fenicsmechanics.materials sub-package.
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
       * 'unsteady' : bool
            True if the problem is time dependent.
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
                    unsteady booleans, and values.
               * 'unsteady' : list, tuple
                    List of booleans specifying whether each region is time
                    dependent (True) or not (False). The order must match
                    the order used in the list of region IDs, and values.
               * 'values' : list, tuple
                    List of values (dolfin.Constant or dolfin.Expression)
                    for each region Dirichlet boundary region specified.
                    The order must match the order used in the list of
                    region IDs, and unsteady booleans.
           * 'neumann'
               * 'regions' : list, tuple
                    List of the region IDs (int) on which Neumann
                    boundary conditions are to be imposed. These IDs
                    must match those used by the mesh function provided.
                    The order must match the order used in the list of
                    types, unsteady booleans, and values.
               * 'types' : list, tuple
                    List of strings specifying whether a 'pressure', 'piola',
                    or 'cauchy' is provided for each region. The order
                    must match the order used in the list of region IDs,
                    unsteady booleans, and values.
               * 'unsteady' : list, tuple
                    List of booleans specifying whether each region is time
                    dependent (True) or not (False). The order must match
                    the order used in the list of region IDs, types, and
                    values.
               * 'values' : list, tuple
                    List of values (dolfin.Constant or dolfin.Expression)
                    for each Dirichlet boundary region specified. The order
                    must match the order used in the list of region IDs,
                    types, and unsteady booleans.

    """

    def __init__(self, config, **kwargs):

        # Obtain mesh and mesh function
        self.mesh = load_mesh(config['mesh']['mesh_file'])
        self.mesh_function = load_mesh_function(config['mesh']['mesh_function'], self.mesh)

        # Define the finite element(s) (maybe add
        # functionality to specify different families?)
        if 'element' not in config['mesh']:
            raise ValueError('You need to specify the type of element(s) to use!')

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
        self.functionSpace = dlf.FunctionSpace(self.mesh, element)

        # Check that number of BC regions and BC values match
        if not self.__check_bcs(config['formulation']['bcs']['dirichlet']):
            raise ValueError('The number of Dirichlet boundary regions and ' \
                             + 'values do not match!')
        if not self.__check_bcs(config['formulation']['bcs']['neumann']):
            raise ValueError('The number of Neumann boundary regions and ' \
                             + 'values do not match!')

        # Passed all the checks, make config member data
        self.config = config

        ## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # MOVE THIS TO A FUNCTION THAT PERFORMS THE ASSEMBLY PROCESS????
        # Identity tensor for later use
        I = dlf.Identity(self.mesh.geometry().dim())

        # Material properties
        rho = config['mechanics']['material']['density']
        la = config['mechanics']['material']['lambda']
        mu = config['mechanics']['material']['mu']

        # Define functions that are the same for both cases
        sys_u = dlf.Function(self.functionSpace)
        sys_du = dlf.TrialFunction(self.functionSpace)

        self._solnFunction = sys_u
        self._trialFunction = sys_du

        # Initial condition if provided
        if config['formulation'].has_key('initial_condition'):
            sys_u.interpolate(config['formulation']['initial_condition'])

        # Initialize the weak form
        weak_form = 0

        # Check if material is incompressible, and then formulate problem
        if config['mechanics']['material']['incompressible']:
            # Define Dirichlet BCs
            self.define_dirichlet_bcs(self.functionSpace.sub(0))

            # Define test function
            sys_v = dlf.TestFunction(self.functionSpace)

            # Obtain pointers to each sub function
            self.displacement, self.pressure = dlf.split(sys_u)
            self.test_vector, self.test_scalar = dlf.split(sys_v)

            if config['mechanics']['material']['type'] == 'elastic':
                self.deformationGradient = I + dlf.grad(self.displacement)
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
                weak_form += (dlf.div(self.displacement) - la*self.pressure)\
                             * self.test_scalar * dlf.dx
            else:
                # Incompressibility constraint for nonlinear
                weak_form += (self.pressure - kappa*(self.jacobian - 1.0))\
                             * self.test_scalar * dlf.dx

        else:
            # Define Dirichlet BCs
            self.define_dirichlet_bcs(self.functionSpace)

            # Make test function (displacement) member data
            self.test_vector, self.test_scalar = dlf.TestFunction(self.functionSpace), None
            self.displacement, self.pressure = sys_u, None

            if config['mechanics']['material']['type'] == 'elastic':
                self.deformationGradient = I + dlf.grad(self.displacement)
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
        self.define_neumann_bcs()
        weak_form -= self.get_neumann_bcs()
        weak_form -= rho*dlf.dot(self.test_vector, config['formulation']['body_force']) * dlf.dx

        weak_form += dlf.inner(dlf.grad(self.test_vector), stress_tensor)*dlf.dx
        weak_form_deriv = dlf.derivative(weak_form, sys_u, sys_du)

        # Initialize NonlinearVariationalProblem object
        dlf.NonlinearVariationalProblem.__init__(self, weak_form,
                                                 sys_u, self.dirichlet_bc_list,
                                                 weak_form_deriv,
                                                 **kwargs)

        self.init_all()

        return None


    def define_dirichlet_bcs(self, functionSpace):
        """
        Return a list of Dirichlet BC objects based on the problem configuration
        provided by the user.


        Parameters
        ----------

        functionSpace : dolfin.functions.functionspace.FunctionSpace
            dolfin object representing the discrete function space that
            will be used to approximate the solution to the weak form.


        Returns
        -------

        bc_list : list
            list of dolfin.fem.bcs.DirichletBC objects.


        """

        if self.config['mechanics']['material']['incompressible']:
            V = self.functionSpace.sub(0)
        else:
            V = self.functionSpace

        bc_list = list()
        for region, value in zip(self.config['formulation']['bcs']['dirichlet']['regions'],
                                 self.config['formulation']['bcs']['dirichlet']['values']):
            bc_list.append(dlf.DirichletBC(V, value, self.mesh_function, region))

        self.dirichlet_bc_list = bc_list

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # SEPARATE THIS INTO TWO LISTS, STEADY AND UNSTEADY.

        return None


    def define_neumann_bcs(self):
        """
        Return the ufl form expressing the Neumann boundary conditions
        based on the problem configuration provided by the user.


        Parameters
        ----------

        xi : dolfin.functions.function.Argument


        Returns
        -------

        total_neumann_bcs :


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
                total_neumann_bcs -= dlf.dot(self.test_vector, value*n)*ds_region
            elif tt == 'cauchy':
                # THIS IS ASSUMING THE PROBLEM IS FORMULATED IN LAGRANGIAN COORDINATES
                total_neumann_bcs += nanson_mag*dlf.dot(self.test_vector, value)*ds_region
            elif tt == 'piola':
                # THIS IS ASSUMING THE PROBLEM IS FORMULATED IN LAGRANGIAN COORDINATES
                total_neumann_bcs += dlf.dot(self.test_vector, value)*ds_region
            else:
                raise NotImplementedError('Neumann BC of type %s is not implemented!' % tt)

        self.neumann_bcs = total_neumann_bcs

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # SEPARATE THIS INTO TWO LISTS, STEADY AND UNSTEADY.

        return None


    def init_all(self):
        """
        Initialize all of the necessary PETSc matrices and vector objects
        based on the problem configuration.

        """

        # Check if problem was specified as unsteady. Assume it is steady
        # if key was not provided.
        try:
            if self.config['formulation']['unsteady']:
                self._localAccelMatrix = dlf.PETScMatrix()
            else:
                self._localAccelMatrix = None
        except KeyError:
            self._localAccelMatrix = None

        # Initialize the vector corresponding to the body force term. Assume
        # it is zero if key was not provided.
        try:
            if self.config['formulation']['body_force']:
                self._bodyForceVector = dlf.PETScVector()
            else:
                self._bodyForceVector = None
        except KeyError:
            self._bodyForceVector = None

        # Initialize vector corresponding to the traction force terms. Assume
        # it is zero if key was not provided.
        try:
            if self.config['formulation']['bcs']['neumann']:
                self._tractionVector = dlf.PETScVector()
            else:
                self._tractionVector = None
        except KeyError:
            self._tractionVector = None

        # Check if problem was formulated in Eulerian coordinates
        # to include the convective acceleration term. Note that
        # this MUST be provided, hence no exception handling.
        if self.config['formulation']['domain'] == 'eulerian':
            self._convectiveAccelMatrix = dlf.PETScMatrix()
            self._convectiveAccelVector = dlf.PETScVector()
        else:
            self._convectiveAccelMatrix = None
            self._convectiveAccelVector = None

        # This should always be non-zero, otherwise there is no
        # problem to solve.
        self._stressWorkMatrix = dlf.PETScMatrix()

        return None


    def update_all(self):
        """


        """

        # UPDATE THE MATRICES AND VECTORS THAT DEPEND ON THE CURRENT STATE.
        # Will have to check:
        #
        # - Dirichlet BCs
        # - Neumann BCs
        # - Body force
        # - Functions storing solutions (displacement, velocity, pressure)

        return None


    def assemble_all(self):
        """


        """

        # Assemble local acceleration matrix is problem is unsteady.
        if self._localAccelMatrix is not None:
            self.assembleLocalAccelMatrix()

        # Assemble convective acceleration matrix and vector.
        if self._convectiveAccelMatrix is not None:
            self.assembleConvectiveAccelMatrix()

        if self._convectiveAccelVector is not None:
            self.assembleConvectiveAccelVector()

        # Assemble the vectors corresponding to body force, traction,
        # and their sum
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

        a = self.getUFLLocalAccel()
        dlf.assemble(a, tensor=self._localAccelMatrix)

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

        convecDiff = self.getUFLConvectiveAccelDifferential()
        dlf.assemble(convecDiff, tensor=self._convectiveAccelMatrix)

        return None


    def assembleConvectiveAccelVector(self):
        """


        """

        convec_accel = self.getUFLConvectiveAccel()
        dlf.assemble(convec_accel, tensor=self._convectiveAccelVector)

        return None


    def assembleStressWorkMatrix(self):
        """


        """

        stress_work_diff = self.getUFLStressWorkDifferential()
        dlf.assemble(stress_work_diff, tensor=self._stressWorkMatrix)

        return None


    def assembleStressWorkVector(self, tensor=None):
        """


        """

        stress_work = self.getUFLStressWork()

        if tensor is None:
            tensor = dlf.PETScVector()

        dlf.assemble(stress_work, tensor=tensor)

        return tensor


    def assembleLoadVector(self):
        """


        """

        body_not_none = self._bodyForceVector is not None
        trac_not_none = self._tractionVector is not None

        if body_not_none and trac_not_none:
            self.assembleBodyForceVector()
            self.assembleTractionVector()
            self._totalLoadVector = self._bodyForceVector \
                                    + self._tractionVector
        elif body_not_none:
            self.assembleBodyForceVector()
            self._totalLoadVector = dlf.PETScVector(self._bodyForceVector)
        elif trac_not_none:
            self.assembleTractionVector()
            self._totalLoadVector = dlf.PETScVector(self._tractionVector)
        else:
            s1 = 'Total load vector is zero. There is no problem to solve! '
            s2 = 'Check the specified body and traction forces.'
            raise ValueError(s1+s2)

        return None


    def assembleBodyForceVector(self):
        """


        """

        body_work = self.getUFLBodyWork()
        dlf.assemble(body_work, tensor=self._bodyForceVector)

        return None


    def assembleTractionVector(self):
        """


        """

        dlf.assemble(self.neumann_bcs, tensor=self._tractionVector)

        return None


    def get_dirichlet_bcs(self):

        return self.dirichlet_bc_list


    def get_neumann_bcs(self):

        return self.neumann_bcs


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

        xi = self._testFunction
        du = self._trialFunction
        rho = self.config['mechanics']['material']['density']

        return dlf.dot(xi, rho*du)*dlf.dx


    def getConvectiveAccelMatrix(self):

        return self._convectiveAccelMatrix


    def getConvectiveAccelVector(self, tensor=None):

        return self._convectiveAccelVector


    def getUFLConvectiveAccel(self):
        """


        """

        xi = self._testFunction
        rho = self.config['mechanics']['material']['density']

        return dlf.dot(xi, rho*dlf.grad(self.velocity)*self.velocity)*dlf.dx


    def getUFLConvectiveAccelDifferential(self):
        """


        """

        convec = self.getUFLConvectiveAccel(self.velocity)

        return dlf.derivative(convec, self.velocity, self._trialFunction)


    def getUFLStressWork(self):
        """


        """

        # Get access fo the function defining the stress tensor
        material_submodule = getattr(materials, self.config['mechanics']['material']['type'])
        stress_function = getattr(material_submodule, self.config['mechanics']['const_eqn'])

        stress_tensor = stress_function(self)
        xi = self._testFunction

        return dlf.inner(dlf.grad(xi), stress_tensor)*dlf.dx


    def getUFLStressWorkDifferential(self):
        """


        """

        stress_work = self.getUFLStressWork()

        return dlf.derivative(stress_work, self._solnFunction, self._trialFunction)


    def getUFLBodyWork(self):
        """


        """

        rho = self.config['mechanics']['material']['density']
        b = self.config['formulation']['body_force']
        xi = self.test_vector

        return dlf.dot(xi, rho*b)*dlf.dx


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
