import re
import dolfin as dlf


class MechanicsProblem(dlf.NonlinearVariationalProblem):
    def __init__(self, problem_config):
        """


        """

        self.config = problem_config
        self.mesh = dlf.Mesh(problem_config['formulation']['mesh'])
        self.mesh_function = dlf.MeshFunction('sizet', mesh, problem_config['formulation']['mesh_function'])

        # Define the finite element(s) (maybe add
        # functionality to specify different families?)
        if problem_config['mesh']['element'] is None:
            raise ValueError('You need to specify the type of element(s) to use!')

        fe_list = re.split('-|_| ', problem_config['mesh']['element'])
        if len(fe_list) > 2:
            s1 = 'The current formulation only allows up to 2 fields.\n'
            s2 = 'You provided %i: ' % len(fe_list) + problem_config['mesh']['element']
            raise NotImplementedError(s1 + s2)
        elif len(fe_list) == 2:
            P_u = dlf.VectorElement('CG', self.mesh.ufl_cell(), int(fe_list[0][-1]))
            P_p = dlf.FiniteElement('CG', self.mesh.ufl_cell(), int(fe_list[1][-1]))
            Element = P_u * P_p
        elif len(fe_list) == 1:
            Element = dlf.VectorElement('CG', self.mesh.ufl_cell(), int(fe_list[0][-1]))
        else:
            raise ValueError('What happened?')

        # Define the function space
        self.functionSpace = dlf.FunctionSpace(mesh, Element)

        #

        if problem_config['formulation']['bcs'].has_key('dirichlet'):
            # define dirichlet boundary conditions
            pass

        if problem_config['formulation']['bcs'].has_key('neumann'):
            # define the proper measure for the region
            pass
