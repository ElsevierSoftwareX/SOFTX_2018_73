import dolfin as dlf

# from .mechanicsproblem import MechanicsProblem

class MechanicsSolver:
    """


    """

    def __init__(self, mechanics_problem):

        # Check if unsteady simulation
        # If simulation is unsteady, the following must be
        # updated at each step:
        #
        # - solution at current step
        # - solution at previous step
        # - convective acceleration (if present)
        # - stiffness matrix
        #   - geometric stiffness (might already be taken care of by dlf.derivative)
        #   - material stiffness (might already be taken care of by dlf.derivative)
        # - body force (if unsteady)
        # - traction BC (if unsteady)

        self.mechanics_problem = mechanics_problem

        return None


    def time_step(self):
        """


        """

        A = self.mechanics_problem._stressWorkMatrix
        b = self.mechanics_problem._totalLoadVector
        soln_vec = self.mechanics_problem.displacement.vector()

        for bc in self.mechanics_problem.dirichlet_bcs:
            bc.apply(A)
            bc.apply(b)

        # NEED TO USE NEWTON SOLVER, I.E. ITERATE!!!!!
        dlf.solve(A, soln_vec, b)

        return None


    def solve(self):
        """


        """

        self.mechanics_problem.assemble_all()

        if self.mechanics_problem.config['formulation']['unsteady']:
            # Call the step function for each time step. Make sure all
            # matrices and vectors are updated appropriately.
            # self.mechanics_problem.update_all()
            raise NotImplementedError('Time-dependent problems have not been implemented!')
        else:
            self.time_step()

        return None


    def nonlinear_solve(self):
        """


        """

        # Need to code Newton iteration.
        #
        # Use b.norm('l2') where b is a dlf.PETScVector().
        # It works in parallel!

        return None
