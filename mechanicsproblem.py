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
                    List of strings specifying whether a 'pressure' or
                    'traction' vector is provided for each region. The order
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
        """


        """

        self.config = config

        # Obtain mesh and mesh function (should this be member data?)
        mesh = dlf.Mesh(config['mesh']['mesh_file'])
        mesh_function = dlf.MeshFunction('size_t', mesh, config['mesh']['mesh_function'])

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
            P_u = dlf.VectorElement('CG', mesh.ufl_cell(), int(fe_list[0][-1]))
            P_p = dlf.FiniteElement('CG', mesh.ufl_cell(), int(fe_list[1][-1]))
            element = P_u * P_p
        else:
            degree = int(fe_list[0][-1])
            element = dlf.VectorElement('CG', mesh.ufl_cell(), int(fe_list[0][-1]))

        # Define the function space (already data in NonlinearVariationalProblem)
        functionSpace = dlf.FunctionSpace(mesh, element)

        # Check that number of BC regions and BC values match
        dirichlet = config['formulation']['bcs']['dirichlet']
        if not self.__check_bcs(dirichlet, 'dirichlet'):
            raise ValueError('The number of Dirichlet boundary regions and ' \
                             + 'values do not match!')
        neumann = config['formulation']['bcs']['neumann']
        if not self.__check_bcs(neumann, 'neumann'):
            raise ValueError('The number of Neumann boundary regions and ' \
                             + 'values do not match!')

        # Identity tensor for later use
        I = dlf.Identity(mesh.geometry().dim())

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
            bc_list = self.__define_dirichlet_bcs(dirichlet, functionSpace.sub(0), mesh_function)

            # Define test function
            sys_v = dlf.TestFunction(functionSpace)

            # Obtain pointers to each sub function
            u, p = dlf.split(sys_u)
            xi, q = dlf.split(sys_v)

            if config['mechanics']['material']['type'] == 'elastic':
                self.deformationGradient = I + dlf.grad(u)
                self.deformationRateGradient = None
                self.jacobian = dlf.det(self.deformationGradient)
            else:
                s1 = 'The material type, %s, has not been implemented!' \
                     % config['mechanics']['material']['viscous']
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
            bc_list = self.__define_dirichlet_bcs(dirichlet, functionSpace, mesh_function)

            xi = dlf.TestFunction(functionSpace)

            if config['mechanics']['material']['type'] == 'elastic':
                self.deformationGradient = I + dlf.grad(sys_u)
                self.deformationRateGradient = None
                self.jacobian = dlf.det(self.deformationGradient)
            else:
                s1 = 'The material type, %s, has not been implemented!' \
                     % config['mechanics']['material']['viscous']
                raise NotImplementedError(s1)

        # Get access to the function defining the stress tensor
        material_submodule = getattr(materials, config['mechanics']['material']['type'])
        stress_function = getattr(material_submodule, config['mechanics']['const_eqn'])

        # Define the stress tensor
        stress_tensor = stress_function(self)

        # Add contributions from external forces to the weak form
        weak_form -= self.__define_neumann_bcs(neumann, mesh, mesh_function, xi)
        # weak_form -= rho*dlf.dot(xi, config['formulation']['body_force']) * dlf.dx

        weak_form += dlf.inner(dlf.grad(xi), stress_tensor)*dlf.dx
        weak_form_deriv = dlf.derivative(weak_form, sys_u, sys_du)

        # Initialize NonlinearVariationalProblem object
        dlf.NonlinearVariationalProblem.__init__(self, weak_form, sys_u, bc_list,
                                                 weak_form_deriv,
                                                 **kwargs)


    @staticmethod
    def __check_bcs(bc_dict, bt):
        """


        """

        if bt == 'dirichlet':
            if len(bc_dict['regions']) == len(bc_dict['values']):
                return True
            else:
                return False
        elif bt == 'neumann':
            if len(bc_dict['regions']) == len(bc_dict['values']):
                return True
            else:
                return False
        else:
            raise NotImplementedError('Boundary type %s is not recognized!' % bt)


    @staticmethod
    def __define_dirichlet_bcs(dirichlet_dict, functionSpace, mesh_function):
        """


        """

        bc_list = list()
        for value, region in zip(dirichlet_dict['values'], dirichlet_dict['regions']):
            bc_list.append(dlf.DirichletBC(functionSpace, value, mesh_function, region))

        return bc_list

    @staticmethod
    def __define_neumann_bcs(neumann_dict, mesh, mesh_function, xi):
        """


        """
        # tt_list = neumann_dict['values']['types'] # type list (traction vs. pressure)
        # function_list = neumann_dict['values']['function']
        region_list = neumann_dict['regions']
        tt_list = neumann_dict['types']
        value_list = neumann_dict['values']

        if 'pressure' in tt_list:
            n = dlf.FacetNormal(mesh)

        # Initiate the Neumann BCs
        total_neumann_bcs = 0

        # tt is the type of neumann boundary condition, either pressure or traction
        # for tt, value, region in zip(tt_list, function_list, region_list):
        for tt, value, region in zip(tt_list, value_list, region_list):
            ds_region = dlf.ds(region, domain=mesh, subdomain_data=mesh_function)
            if tt == 'pressure':
                total_neumann_bcs -= dlf.dot(xi, value*n)*ds_region
            elif tt == 'traction':
                total_neumann_bcs += dlf.dot(xi, value)*ds_region
            else:
                raise NotImplementedError('Neumann BC of type %s is not implemented!' % tt)

        return total_neumann_bcs
