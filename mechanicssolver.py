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


    def solve(self, maxIters=50, tol=1e-10, lin_solver='mumps', fname=None):
        """


        """

        self.mechanics_problem.assemble_all()

        if self.mechanics_problem.config['formulation']['time']['unsteady']:
            self.explicit_euler(fname=fname)
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


    def explicit_euler(self, fname=None):
        """


        """

        if fname:
            result_file = dlf.File(fname)

        mp = self.mechanics_problem
        t, tf = mp.config['formulation']['time']['interval']
        dt = mp.config['formulation']['time']['dt']

        while t <= tf:
            t += dt
            print '****************************************'
            print 't = %.3f' % t
            mp.update_all(t)
            un, vn = self.explicit_euler_step()

            print 'un = \n', un.array()
            print 'vn = \n', vn.array()

            mp.displacement.vector()[:] = un
            mp.velocity.vector()[:] = vn

            if fname:
                result_file << mp.displacement

        return None


    def explicit_euler_step(self):
        """


        """

        mp = self.mechanics_problem
        dt = mp.config['formulation']['time']['dt']
        u0 = mp.displacement.vector()
        v0 = mp.velocity.vector()
        M = mp._localAccelMatrix

        # This should always be non-zero for deformable bodies.
        f0 = -mp._stressWorkVector

        # Check if body force is applied
        if mp._bodyForceWorkVector is not None:
            f0 += mp._bodyForceWorkVector

        # Check if traction boundary condition was applied.
        if mp._tractionWorkVector is not None:
            f0 += mp._tractionWorkVector

        # Check if convective acceleration is non-zero.
        if mp._convectiveAccelVector is not None:
            f0 -= mp._convectiveAccelVector

        un = u0 + dt*v0
        vn = dlf.PETScVector()
        dlf.solve(M, vn, M*v0 + dt*f0)

        return un, vn


    def implicit_euler_step(self):
        """


        """

        raise NotImplementedError('This function has not been implemented yet.')

        return None
