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
        # self.vectorSpace = dlf.FunctionSpace(self.mesh, vec_element)

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

        self.sys_du = dlf.TrialFunction(self.functionSpace)
        self.trial_vector, self.trial_scalar = dlf.split(self.sys_du)

        self.test_vector, self.test_scalar = dlf.TestFunctions(self.functionSpace)

        if self.config['formulation']['time']['unsteady']:
            self.sys_u0 = dlf.Function(self.functionSpace)
            self.ufl_displacement0, self.ufl_pressure0 = dlf.split(self.sys_u0)
            self.displacement0, self.pressure0 = self.sys_u0.split(deepcopy=True)

            self.sys_v0 = dlf.Function(self.functionSpace)
            self.ufl_velocity0, _ = dlf.split(self.sys_v0)
            self.velocity0, self._i0 = self.sys_v0.split(deepcopy=True)

            self.sys_a0 = dlf.Function(self.functionSpace)
            self.ufl_acceleration0, _ = dlf.split(self.sys_a0)
            self.acceleration0, self._i1 = self.sys_a0.split(deepcopy=True)

            self.define_ufl_acceleration()

        return None


    def define_compressible_functions(self):
        """


        """

        self.sys_u = self.ufl_displacement \
                     = self.displacement = dlf.Function(self.functionSpace)
        self.test_vector = dlf.TestFunction(self.functionSpace)
        self.trial_vector = dlf.TrialFunction(self.functionSpace)

        if self.config['formulation']['time']['unsteady']:
            self.sys_u0 = self.ufl_displacement0 \
                          = self.displacement0 = dlf.Function(self.functionSpace)

            self.sys_v0 = self.ufl_velocity0 \
                          = self.velocity0 = dlf.Function(self.functionSpace)

            self.sys_a0 = self.ufl_acceleration0 \
                          = self.acceleration0 = dlf.Function(self.functionSpace)

            self.define_ufl_acceleration()

        return None


    def define_ufl_acceleration(self):
        """


        """

        self.ufl_acceleration = 1.0/(beta*dt**2)*(u - u0 - dt*v0) - (1.0/(2.0*beta) - 1.0)*a0

        dt = self.config['formulation']['time']['dt']
        beta = self.config['formulation']['time']['beta']
        self.ufl_acceleration = self.ufl_displacement \
                                - self.ufl_displacement0 \
                                - dt*self.ufl_velocity0
        self.ufl_acceleration *= 1.0/(beta*dt**2)
        self.ufl_acceleration += (1.0/(2.0*beta) - 1.0)*self.ufl_acceleration0

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
        else:
            s = "The material '%s' has not been implemented. A class for such" \
                + " material must be provided."
            raise ValueError(s)

        self._material = mat_class(inverse=self.config['formulation']['inverse'],
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


    def define_ufl_local_inertia(self):
        """


        """

        # Set to 0 and exit if problem is steady.
        if not self.config['formulation']['time']['unsteady']:
            self.ufl_local_inertia = 0
            # self.ufl_local_inertia0 = 0
            return None

        xi = self.test_vector
        rho = self.config['material']['density']

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
