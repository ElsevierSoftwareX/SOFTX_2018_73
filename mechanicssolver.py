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


    def solve(self, maxIters=50, tol=1e-10, lin_solver='mumps',
              fname=None, save_freq=1):
        """


        """

        self.mechanics_problem.assemble_all()

        if self.mechanics_problem.config['formulation']['time']['unsteady']:
            self.explicit_euler(fname=fname, save_freq=save_freq)
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

        # Rank to avoid print multiple times.
        rank = dlf.MPI.rank(dlf.mpi_comm_world())

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
            for bc in self.mechanics_problem.dirichlet_bcs['displacement']:
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


    def explicit_euler(self, fname=None, save_freq=1):
        """


        """

        if fname:
            result_file = dlf.File(fname)

        mp = self.mechanics_problem
        t, tf = mp.config['formulation']['time']['interval']
        dt = mp.config['formulation']['time']['dt']

        count = 0

        rank = dlf.MPI.rank(dlf.mpi_comm_world())

        while t <= tf:
            t += dt
            mp.update_all(t)
            un, vn = self.explicit_euler_step()

            un_norm = un.norm('l2')
            vn_norm = vn.norm('l2')

            mp.displacement.vector()[:] = un
            mp.velocity.vector()[:] = vn

            if fname and not count % save_freq:
                result_file << mp.displacement
                if rank == 0:
                    print '*'*40
                    print 't = %.3f' % t
                    print 'un.norm(\'l2\') = ', un_norm
                    print 'vn.norm(\'l2\') = ', vn_norm

            count += 1

        return None


    def rhs(self):
        """


        """

        # Might need to update state before getting these vectors

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

        return f0


    def explicit_euler_step(self):
        """


        """

        mp = self.mechanics_problem
        dt = mp.config['formulation']['time']['dt']
        u0 = mp.displacement.vector()
        v0 = mp.velocity.vector()
        M = mp._localAccelMatrix
        f0 = self.rhs()

        un = u0 + dt*v0
        mp.bc_apply('displacement', b=un)

        b = M*v0 + dt*f0
        mp.bc_apply('velocity', A=M, b=b)

        vn = dlf.PETScVector()
        dlf.solve(M, vn, b)

        return un, vn


    def implicit_euler_step(self):
        """


        """

        raise NotImplementedError('This function has not been implemented yet.')

        return None


    def generalized_alpha(self, alpha=0.5):
        """


        """

        # Get time-stepping parameters.
        #
        # Loop through all the time steps
        # calling generalized_alpha_step
        # each time.
        #
        # Need to assemble the appropriate matrices and vectors
        # for each time step. Both in the current and previous
        # time step, depending on the value of alpha.
        #
        # Prepare for the next time step: Update the displacement,
        # velocity, and acceleration function objects.

        return None


    def generalized_alpha_step(self, alpha=0.5):
        """


        """

        # Get necessary time-stepping parameters.
        #
        # In general, a call to the nonlinear solver will be required.
        # Might need to reformat the nonlinear_solve function so that
        # it can also be used in this case.

        return None
