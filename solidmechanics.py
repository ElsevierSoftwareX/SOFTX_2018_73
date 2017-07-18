import ufl
import dolfin as dlf

from . import materials
from .utils import duplicate_expressions
from .basemechanicsproblem import BaseMechanicsProblem

from inspect import isclass

__all__ = ['SolidMechanicsProblem']


class SolidMechanicsProblem(BaseMechanicsProblem):
    """


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

        cell = self.mesh.ufl_cell()
        vec_degree = int(self.config['mesh']['element'][0][-1])
        if vec_degree == 0:
            vec_family = 'DG'
        else:
            vec_family = 'CG'

        vec_element = dlf.VectorElement(vec_family, cell, vec_degree)

        if self.config['material']['incompressible']:
            scalar_degree = int(self.config['mesh']['element'][1][-1])
            if scalar_degree == 0:
                scalar_family = 'DG'
            else:
                scalar_family = 'CG'
            scalar_element = dlf.FiniteElement(scalar_family, cell, scalar_degree)
            element = vec_element*scalar_element
        else:
            element = vec_element

        self.functionSpace = dlf.FunctionSpace(self.mesh, element)

        return None


    def define_functions(self):
        """


        """

        if self.config['material']['incompressible']:
            self.define_incompressible_functions()
        else:
            self.define_compressible_functions()

        return None


    def define_incompressible_functions(self):
        """


        """

        self.sys_u = dlf.Function(self.functionSpace)
        self.ufl_displacement, self.ufl_pressure = dlf.split(self.sys_u)
        self.displacement, self.pressure = self.sys_u.split(deepcopy=True)
        self.displacement.rename('u', 'displacement')
        self.pressure.rename('p', 'pressure')

        self.sys_du = dlf.TrialFunction(self.functionSpace)
        self.trial_vector, self.trial_scalar = dlf.split(self.sys_du)

        self.test_vector, self.test_scalar = dlf.TestFunctions(self.functionSpace)

        if self.config['formulation']['time']['unsteady']:
            self.sys_u0 = dlf.Function(self.functionSpace)
            self.ufl_displacement0, self.ufl_pressure0 = dlf.split(self.sys_u0)
            self.displacement0, self.pressure0 = self.sys_u0.split(deepcopy=True)
            self.displacement0.rename('u0', 'displacement0')
            self.pressure0.rename('p0', 'pressure0')

            self.sys_v0 = dlf.Function(self.functionSpace)
            self.ufl_velocity0, _ = dlf.split(self.sys_v0)
            self.velocity0, _ = self.sys_v0.split(deepcopy=True)
            self.velocity0.rename('v0', 'velocity0')

            self.sys_a0 = dlf.Function(self.functionSpace)
            self.ufl_acceleration0, _ = dlf.split(self.sys_a0)
            self.acceleration0, _ = self.sys_a0.split(deepcopy=True)
            self.acceleration0.rename('a0', 'acceleration0')

            self.define_ufl_acceleration()

        return None


    def define_compressible_functions(self):
        """


        """

        self.sys_u = self.ufl_displacement \
                     = self.displacement = dlf.Function(self.functionSpace)
        self.displacement.rename('u', 'displacement')
        self.ufl_pressure = self.pressure = None
        self.ufl_pressure0 = self.pressure0 = None
        self.test_vector = dlf.TestFunction(self.functionSpace)
        self.trial_vector = self.sys_du = dlf.TrialFunction(self.functionSpace)

        if self.config['formulation']['time']['unsteady']:
            self.sys_u0 = self.ufl_displacement0 \
                          = self.displacement0 = dlf.Function(self.functionSpace)
            self.displacement0.rename('u0', 'displacement0')

            self.sys_v0 = self.ufl_velocity0 \
                          = self.velocity0 = dlf.Function(self.functionSpace)
            self.velocity0.rename('v0', 'velocity0')

            self.sys_a0 = self.ufl_acceleration0 \
                          = self.acceleration0 = dlf.Function(self.functionSpace)
            self.acceleration0.rename('a0', 'acceleration0')

            self.define_ufl_acceleration()

        return None


    def define_ufl_acceleration(self):
        """


        """

        dt = self.config['formulation']['time']['dt']
        beta = self.config['formulation']['time']['beta']

        u = self.ufl_displacement
        u0 = self.ufl_displacement0
        v0 = self.ufl_velocity0
        a0 = self.ufl_acceleration0

        self.ufl_acceleration = 1.0/(beta*dt**2)*(u - u0 - dt*v0) \
                                - (1.0/(2.0*beta) - 1.0)*a0

        return None


    def define_deformation_tensors(self):
        """


        """

        dim = self.mesh.geometry().dim()
        I = dlf.Identity(dim)
        self.deformationGradient = I + dlf.grad(self.ufl_displacement)
        self.jacobian = dlf.det(self.deformationGradient)

        if self.config['formulation']['time']['unsteady']:
            self.deformationGradient0 = I + dlf.grad(self.ufl_displacement0)
            self.jacobian0 = dlf.det(self.deformationGradient0)

        return None


    def define_material(self):
        """


        """

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
            s = "The material '%s' has not been implemented. A class for such" \
                + " material must be provided."
            raise ValueError(s)

        try:
            fiber_file = self.config['mesh']['fiber_file']
        except KeyError:
            fiber_file = None
        self._material = mat_class(mesh=self.mesh,
                                   fiber_file=fiber_file,
                                   inverse=self.config['formulation']['inverse'],
                                   **self.config['material'])

        return None


    def define_dirichlet_bcs(self):
        """


        """

        # Exit function if no Dirichlet BCs were provided.
        if self.config['formulation']['bcs']['dirichlet'] is None:
            self.dirichlet_bcs = None
            return None

        if self.config['material']['incompressible']:
            self._define_incompressible_dirichlet_bcs()
        else:
            self._define_compressible_dirichlet_bcs()

        return None


    def _define_compressible_dirichlet_bcs(self):
        """


        """

        if 'displacement' in self.config['formulation']['bcs']['dirichlet']:
            self.dirichlet_bcs = dict()
            dirichlet_dict = self.config['formulation']['bcs']['dirichlet']
            dirichlet_bcs = self.__define_displacement_bcs(self.functionSpace,
                                                           dirichlet_dict,
                                                           self.mesh_function)
            self.dirichlet_bcs.update(dirichlet_bcs)
        else:
            self.dirichlet_bcs = None

        return None


    def _define_incompressible_dirichlet_bcs(self):
        """


        """

        self.dirichlet_bcs = dict()
        dirichlet_dict = self.config['formulation']['bcs']['dirichlet']
        if 'displacement' in self.config['formulation']['bcs']['dirichlet']:
            displacement_bcs = self.__define_displacement_bcs(self.functionSpace.sub(0),
                                                           dirichlet_dict,
                                                           self.mesh_function)
            self.dirichlet_bcs.update(displacement_bcs)

        if 'pressure' in self.config['formulation']['bcs']['dirichlet']:
            pressure_bcs = self.__define_pressure_bcs(self.functionSpace.sub(1),
                                                      dirichlet_dict,
                                                      self.mesh_function)
            self.dirichlet_bcs.update(pressure_bcs)

        if not self.dirichlet_bcs:
            self.dirichlet_bcs = None

        return None


    def define_forms(self):
        """


        """

        # Define UFL objects corresponding to the local acceleration
        # if problem is unsteady.
        self.define_ufl_local_inertia()

        # Define UFL objects corresponding to the stress tensor term.
        # This should always be non-zero for deformable bodies.
        self.define_ufl_stress_work()

        # Define UFL object corresponding to the body force term. Assume
        # it is zero if key was not provided.
        self.define_ufl_body_force()

        # Define UFL object corresponding to the traction force terms. Assume
        # it is zero if key was not provided.
        self.define_ufl_neumann_bcs()

        return None


    def define_ufl_local_inertia(self):
        """


        """

        # Set to 0 and exit if problem is steady.
        if not self.config['formulation']['time']['unsteady']:
            self.ufl_local_inertia = 0
            return None

        xi = self.test_vector
        rho = dlf.Constant(self.config['material']['density'])

        # Will need both of these terms if problem is unsteady
        self.ufl_local_inertia = dlf.dot(xi, rho*self.ufl_acceleration)
        self.ufl_local_inertia *= dlf.dx

        return None


    def define_ufl_stress_work(self):
        """


        """

        stress_func = self._material.stress_tensor
        stress_tensor = stress_func(self.deformationGradient,
                                    self.jacobian,
                                    self.ufl_pressure)
        xi = self.test_vector
        self.ufl_stress_work = dlf.inner(dlf.grad(xi), stress_tensor)
        self.ufl_stress_work *= dlf.dx
        if self.config['formulation']['time']['unsteady']:
            stress_tensor0 = stress_func(self.deformationGradient0,
                                         self.jacobian0,
                                         self.ufl_pressure0)
            self.ufl_stress_work0 = dlf.inner(dlf.grad(xi), stress_tensor0)
            self.ufl_stress_work0 *= dlf.dx
        else:
            self.ufl_stress_work0 = 0

        return None


    def define_ufl_body_force(self):
        """


        """

        if self.config['formulation']['body_force'] is None:
            self.ufl_body_force = 0
            self.ufl_body_force0 = 0
            return None

        rho = self.config['material']['density']
        b = self.config['formulation']['body_force']
        xi = self.test_vector
        self.ufl_body_force = dlf.dot(xi, rho*b)*dlf.dx

        # Create a copy of the body force term to use at a different time step.
        if self.config['formulation']['time']['unsteady']:
            b0, = duplicate_expressions(b)
            self.ufl_body_force0 = dlf.dot(xi, rho*b0)*dlf.dx
        else:
            self.ufl_body_force0 = 0

        return None


    def define_ufl_neumann_bcs(self):
        """


        """

        if self.config['formulation']['bcs']['neumann'] is None:
            self.ufl_neumann_bcs = 0
            self.ufl_neumann_bcs0 = 0
            return None

        regions = self.config['formulation']['bcs']['neumann']['regions']
        types = self.config['formulation']['bcs']['neumann']['types']
        values = self.config['formulation']['bcs']['neumann']['values']
        domain = self.config['formulation']['domain']

        define_ufl_neumann = BaseMechanicsProblem.define_ufl_neumann_form

        self.ufl_neumann_bcs = define_ufl_neumann(regions, types,
                                                  values, domain,
                                                  self.mesh,
                                                  self.mesh_function,
                                                  self.deformationGradient,
                                                  self.jacobian,
                                                  self.test_vector)
        if self.config['formulation']['time']['unsteady']:
            values0 = duplicate_expressions(*values)
            self.ufl_neumann_bcs0 = define_ufl_neumann(regions, types,
                                                       values0, domain,
                                                       self.mesh,
                                                       self.mesh_function,
                                                       self.deformationGradient0,
                                                       self.jacobian0,
                                                       self.test_vector)
        else:
            self.ufl_neumann_bcs0 = 0

        return None


    def define_ufl_equations(self):
        """


        """

        theta = self.config['formulation']['time']['theta']

        self.G1 = self.ufl_local_inertia \
                  + theta*(self.ufl_stress_work \
                           - self.ufl_body_force \
                           - self.ufl_neumann_bcs) \
                  + (1.0 - theta)*(self.ufl_stress_work0 \
                                   - self.ufl_body_force0 \
                                   - self.ufl_neumann_bcs0)

        if self.config['material']['incompressible']:
            q = self.test_scalar
            kappa = self._material._parameters['kappa']
            bvol = self._material.incompressibilityCondition(self.ufl_displacement)
            self.G2 = q*(kappa*bvol - self.ufl_pressure)*dlf.dx
        else:
            self.G2 = 0
        self.G = self.G1 + self.G2

        return None


    def define_ufl_equations_diff(self):
        """


        """

        self.dG = dlf.derivative(self.G, self.sys_u, self.sys_du)

        return None


    @staticmethod
    def __define_displacement_bcs(W, dirichlet_dict, mesh_function):
        """


        """

        displacement_bcs = {'displacement': list()}
        disp_vals = dirichlet_dict['displacement']
        regions = dirichlet_dict['regions']
        for region, value in zip(regions, disp_vals):
            bc = dlf.DirichletBC(W, value, mesh_function, region)
            displacement_bcs['displacement'].append(bc)

        return displacement_bcs


    @staticmethod
    def __define_pressure_bcs(W, dirichlet_dict, mesh_function):
        """


        """

        pressure_bcs = {'pressure': list()}
        p_vals = dirichlet_bcs['pressure']
        p_regions = dirichlet_dict['p_regions']
        for region, value in zip(p_regions, p_vals):
            bc = dlf.DirichletBC(W, value, mesh_function, region)
            pressure_bcs['pressure'].append(bc)

        return pressure_bcs


class SolidMechanicsSolver(dlf.NonlinearVariationalSolver):
    """


    """


    def __init__(self, problem):

        self._problem = problem

        bcs = list()
        for val in problem.dirichlet_bcs.values():
            bcs.extend(val)

        dlf_problem = dlf.NonlinearVariationalProblem(problem.G, problem.sys_u,
                                                      bcs, J=problem.dG)

        dlf.NonlinearVariationalSolver.__init__(self, dlf_problem)

        if problem.config['material']['incompressible']:
            self.define_function_assigners()

        return None


    def set_parameters(self, linear_solver='default',
                       newton_abstol=1e-10,
                       newton_reltol=1e-9,
                       newton_maxIters=50,
                       krylov_abstol=1e-8,
                       krylov_reltol=1e-7,
                       krylov_maxIters=50):
        """


        """

        param = self.parameters
        param['newton_solver']['linear_solver'] = linear_solver
        param['newton_solver']['absolute_tolerance'] = newton_abstol
        param['newton_solver']['relative_tolerance'] = newton_reltol
        param['newton_solver']['maximum_iterations'] = newton_maxIters
        param['newton_solver']['krylov_solver']['absolute_tolerance'] = krylov_abstol
        param['newton_solver']['krylov_solver']['relative_tolerance'] = krylov_reltol
        param['newton_solver']['krylov_solver']['maximum_iterations'] = krylov_maxIters

        return None


    def full_solve(self, fname_disp=None, fname_press=None, save_freq=1):
        """


        """

        problem = self._problem
        rank = dlf.MPI.rank(dlf.mpi_comm_world())

        if fname_disp:
            file_disp = dlf.File(fname_disp, 'compressed')
        if fname_press:
            file_press = dlf.File(fname_press, 'compressed')

        if problem.config['formulation']['time']['unsteady']:
            t, tf = problem.config['formulation']['time']['interval']
            t0 = t

            dt = problem.config['formulation']['time']['dt']
            count = 0 # Used to check if files should be saved.

            # Hack to avoid rounding errors.
            while t < (tf - dt/10.0):

                # Save current time step
                if not count % save_freq:
                    if fname_disp:
                        file_disp << (problem.displacement, t)
                        if not rank:
                            print('* Displacement saved *')
                    if fname_press:
                        file_press << (problem.pressure, t)
                        if not rank:
                            print('* Pressure saved *')

                # Advance the time.
                t += dt

                # Update expressions that depend on time.
                problem.update_time(t, t0)

                # Print the current time.
                if not rank:
                    print('*'*30)
                    print('t = %3.6f' % t)

                # Solver current time step.
                self.step()

                # Assign and update all vectors.
                self.update_assign()

                t0 = t
                count += 1

        else:
            self.step()

            self.update_assign()

            if fname_disp:
                file_disp << self._problem.displacement
            if fname_press:
                file_press << self._problem.pressure

        return None


    def step(self):
        """


        """

        self.solve()

        return None


    def update_assign(self):

        problem = self._problem
        incompressible = problem.config['material']['incompressible']
        unsteady = problem.config['formulation']['time']['unsteady']

        u = problem.displacement

        if unsteady:
            u0 = problem.displacement0
            v0 = problem.velocity0
            a0 = problem.acceleration0

            beta = problem.config['formulation']['time']['beta']
            gamma = problem.config['formulation']['time']['gamma']
            dt = problem.config['formulation']['time']['dt']

        if incompressible:
            p = problem.pressure
            self.assigner_sys2u.assign([u, p], problem.sys_u)

        if unsteady:
            self.update(u, u0, v0, a0, beta, gamma, dt)

        if incompressible and unsteady:
            p0 = problem.pressure0
            self.assigner_u02sys.assign(problem.sys_u0, [u0, p0])
            self.assigner_v02sys.assign(problem.sys_v0.sub(0), v0)
            self.assigner_a02sys.assign(problem.sys_a0.sub(0), a0)

        return None


    def define_function_assigners(self):

        problem = self._problem
        W = problem.functionSpace
        u = problem.displacement
        p = problem.pressure

        self.assigner_sys2u = dlf.FunctionAssigner([u.function_space(),
                                                    p.function_space()], W)

        if problem.config['formulation']['time']['unsteady']:
            u0 = problem.displacement0
            p0 = problem.pressure0
            v0 = problem.velocity0
            a0 = problem.acceleration0

            self.assigner_u02sys = dlf.FunctionAssigner(W, [u0.function_space(),
                                                            p0.function_space()])
            self.assigner_v02sys = dlf.FunctionAssigner(W.sub(0), v0.function_space())
            self.assigner_a02sys = dlf.FunctionAssigner(W.sub(0), a0.function_space())

        return None


    @staticmethod
    def update(u, u0, v0, a0, beta, gamma, dt):

        # Get vector references
        u_vec, u0_vec = u.vector(), u0.vector()
        v0_vec, a0_vec = v0.vector(), a0.vector()

        # Update acceleration and velocity
        a_vec = 1.0/(beta*dt**2)*(u_vec - u0_vec - v0_vec*dt) \
                                  - (1.0/(2.0*beta) - 1.0)*a0_vec
        v_vec = dt*((1.0 - gamma)*a0_vec + gamma*a_vec) + v0_vec

        v0.vector()[:], a0.vector()[:] = v_vec, a_vec
        u0.vector()[:] = u_vec

        return None
