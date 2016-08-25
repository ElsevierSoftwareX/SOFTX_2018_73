import re
import dolfin as dlf
import materials # is this the right way to import?


class MechanicsProblem(dlf.NonlinearVariationalProblem):
    """


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
            P_u = dlf.VectorElement('CG', self.mesh.ufl_cell(), int(fe_list[0][-1]))
            P_p = dlf.FiniteElement('CG', self.mesh.ufl_cell(), int(fe_list[1][-1]))
            element = P_u * P_p
        else:
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

        # Get access to the function defining the stress tensor
        material_submodule = getattr(materials, config['mechanics']['material']['type'])
        stress_function = getattr(material_submodule, config['mechanics']['const_eqn'])

        # Material properties
        rho = config['mechanics']['material']['density']
        lame1 = config['mechanics']['material']['lame1']
        lame2 = config['mechanics']['material']['lame2']

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
                weak_form += (dlf.div(u) - lame1*p)*q*dlf.dx
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

        # Define the stress tensor
        stress_tensor = stress_function(self)

        # Add contributions from external forces to the weak form
        weak_form -= self.__define_neumann_bcs(neumann, mesh, mesh_function, xi)
        weak_form -= rho*dlf.dot(xi, config['formulation']['body_force']) * dlf.dx

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
            if len(bc_dict['regions']) == len(bc_dict['values']['function']):
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

    @staticmethod
    def __define_neumann_bcs(neumann_dict, mesh, mesh_function, xi):
        """


        """
        n = dlf.FacetNormal(mesh)

        total_neumann_bcs = 0
        tt_list = neumann_dict['values']['types'] # type list (traction vs. pressure)
        function_list = neumann_dict['values']['function']
        region_list = neumann_dict['regions']

        for tt, value, region in zip(tt_list, function_list, region_list):
            ds_region = dlf.ds(region, domain=mesh, subdomain_data=mesh_function)
            if tt == 'pressure':
                total_neumann_bcs -= dlf.dot(xi, value*n)*ds_region
            elif tt == 'traction':
                total_neumann_bcs -= dlf.dot(xi, value)*ds_region
            else:
                raise NotImplementedError('Neumann BC of type %s is not implemented!' % tt)

        return total_neumann_bcs
