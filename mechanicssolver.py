import dolfin as dlf

import numpy as np

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

        # NEED TIME ITERATION SCHEME HERE

        return None


    def solve(self, maxIters=50, tol=1e-10, lin_solver='mumps'):
        """


        """

        self.mechanics_problem.assemble_all()

        if self.mechanics_problem.config['formulation']['unsteady']:
            # Call the step function for each time step. Make sure all
            # matrices and vectors are updated appropriately.
            # self.mechanics_problem.update_all()
            raise NotImplementedError('Time-dependent problems have not been implemented!')
        else:
            self.nonlinear_solve(maxIters=maxIters, tol=tol, lin_solver=lin_solver)

        return None


    def nonlinear_solve(self, maxIters=50, tol=1e-10, lin_solver='mumps'):
        """


        """

        soln_vec = self.mechanics_problem.displacement.vector()
        du = dlf.PETScVector()
        norm = 1.0
        count = 0

        try:
            rank = dlf.MPI.rank(dlf.mpi_comm_world())
        except:
            rank = 0

        while norm >= tol:

            if count >= maxIters:
                s1 = '*** The nonlinear solver reached the max number of iterations. ***'
                raise StopIteration(s1)

            # Update objects that depend on state.
            self.mechanics_problem.update_all()

            # Solve the linear algebra problem
            A = self.mechanics_problem._stressWorkMatrix
            b = self.mechanics_problem._totalLoadVector

            # Apply Dirichlet BCs
            for bc in self.mechanics_problem.dirichlet_bcs:
                bc.apply(A)
                bc.apply(b)

            dlf.solve(A, du, b, lin_solver)

            # Prepare for the next iteration
            # self.mechanics_problem.displacement.vector()[:] += du
            soln_vec += du
            norm = du.norm('l2')
            count += 1

            if rank == 0:
                print 'Iteration %i: norm = %.6e' % (count, norm)

        return None
