from .utils import petsc_identity # This might not be necessary
from ufl import Form

import block
from block import iterative

import dolfin as dlf
import numpy as np

__all__ = ['MechanicsSolver']

class MechanicsSolver(object):
    """


    """

    def __init__(self, mechanics_problem):

        # Make the MechanicsProblem object part of the member data.
        self._mp = mechanics_problem

        return None


    def solve(self, nonlinear_tol=1e-10, iter_tol=1e-8, maxNonlinIters=50,
              maxLinIters=200, show=0, print_norm=True, fname_disp=None,
              fname_vel=None, save_freq=1, lin_solver='mumps'):
        """


        """

        if self._mp.config['formulation']['time']['unsteady']:
            self.time_solve(nonlinear_tol=nonlinear_tol, iter_tol=iter_tol,
                            maxNonlinIters=maxNonlinIters,
                            maxLinIters=maxLinIters, show=show,
                            print_norm=print_norm, fname_disp=fname_disp,
                            fname_vel=fname_vel, save_freq=save_freq)
        else:
            lhs, rhs = self.ufl_lhs_rhs()
            if self._mp.config['material']['type'] == 'elastic':
                bcs = self._mp.dirichlet_bcs['displacement']
                if fname_disp:
                    fout = dlf.File(fname_disp)
                    func_out = self._mp.displacement
                else:
                    fout = None
                    func_out = self._mp.velocity
            else:
                bcs = self.dirichlet_bcs['velocity']
                if fname_vel:
                    fout = dlf.File(fname_vel)
                else:
                    fout = None

            self.nonlinear_solve(lhs, rhs, bcs, nonlinear_tol=nonlinear_tol,
                                 iter_tol=iter_tol, maxNonlinIters=maxNonlinIters,
                                 maxLinIters=maxLinIters, show=show,
                                 print_norm=print_norm)
            if fout:
                fout << func_out

        return None


    def time_solve(self, nonlinear_tol=1e-10, iter_tol=1e-8,
                   maxNonlinIters=50, maxLinIters=200, show=0,
                   print_norm=True, fname_disp=None, fname_vel=None,
                   save_freq=1):
        """
        THIS FUNCTION ASSUMES THAT THE PROBLEM IS AN UNSTEADY ELASTIC
        PROBLEM. UNSTEADY VISCOUS MATERIALS HAVE NOT BEEN IMPLEMENTED!

        """

        t, tf = self._mp.config['formulation']['time']['interval']
        t0 = t
        dt = self._mp.config['formulation']['time']['dt']
        count = 0 # Used to check if file(s) should be saved.

        if fname_disp:
            file_disp = dlf.File(fname_disp, 'compressed')
        if fname_vel:
            file_vel = dlf.File(fname_vel, 'compressed')

        lhs, rhs = self.ufl_lhs_rhs()
        bcs = block.block_bc(self._mp.dirichlet_bcs.values(), False)

        rank = dlf.MPI.rank(dlf.mpi_comm_world())

        while t <= tf:

            # Put this at the beginning so that the initial
            # condition is saved.
            if not count % save_freq:
                if fname_disp:
                    file_disp << (self._mp.displacement, t)
                    if not rank:
                        print '* Displacement saved *'
                if fname_vel:
                    file_vel << (self._mp.velocity, t)
                    if not rank:
                        print '* Velocity saved *'

            # Set to the next time step
            t += dt

            # Update expressions that depend on time
            self._mp.update_time(t, t0)

            # Print the current time
            if not rank:
                print '*'*30
                print 't = %3.5f' % t

            # Solve the nonlinear equation(s) at current time step.
            self.nonlinear_solve(lhs, rhs, bcs, nonlinear_tol=nonlinear_tol,
                                 iter_tol=iter_tol, maxNonlinIters=maxNonlinIters,
                                 maxLinIters=maxLinIters, show=show,
                                 print_norm=print_norm)

            # Prepare for the next time step
            self._mp.displacement0.assign(self._mp.displacement)
            self._mp.velocity0.assign(self._mp.velocity)
            t0 = t
            count += 1

        return None


    def nonlinear_solve(self, lhs, rhs, bcs, nonlinear_tol=1e-10,
                        iter_tol=1e-8, maxNonlinIters=50, maxLinIters=200,
                        show=0, print_norm=True):
        """


        """

        norm = 1.0
        count = 0
        rank = dlf.MPI.rank(dlf.mpi_comm_world())

        # Determine if we can use dolfin's assemble_system function or
        # if we need to assemble a block system.
        if isinstance(lhs, Form):
            assemble_system = dlf.assemble_system
            is_block = False
            du = dlf.PETScVector()
        else:
            assemble_system = block.block_assemble
            is_block = True

        while norm >= nonlinear_tol:

            if count >= maxNonlinIters:
                raise StopIteration('Maximum number of iterations reached.')

            # Assemble system with Dirichlet BCs applied symmetrically.
            A, b = assemble_system(lhs, rhs, bcs)

            # Decide between a dolfin direct solver or a block iterative solver.
            if is_block:
                Ainv = iterative.LGMRES(A, show=show, tolerance=iter_tol,
                                        nonconvergence_is_fatal=True,
                                        maxiter=maxLinIters)
                du = Ainv*b
                self._mp.displacement.vector()[:] += du.blocks[0]
                self._mp.velocity.vector()[:] += du.blocks[1]
            else:
                dlf.solve(A, du, b, 'mumps')
                if self._mp.config['material']['type'] == 'elastic':
                    self._mp.displacement.vector()[:] += du
                else:
                    self._mp.velocity.vector()[:] += du

            norm = du.norm('l2')
            if not rank and print_norm:
                print '(iter %2i) norm %.3e' % (count, norm)

            count += 1

        return None


    def ufl_lhs_rhs(self):
        """


        """

        mp = self._mp

        unsteady = mp.config['formulation']['time']['unsteady']
        incompressible = mp.config['material']['incompressible']
        elastic = mp.config['material']['type'] == 'elastic'

        if unsteady and elastic and incompressible:
            lhs = [[mp.df1_du, mp.df1_dv, mp.df1_dp],
                   [mp.df2_du, mp.df2_dv, mp.df2_dp],
                   [mp.df3_du, mp.df3_dv, mp.df3_dp]]
            rhs = [-mp.f1, -mp.f2, -mp.f3]
        elif unsteady and elastic: # Compressible unsteady elastic
            lhs = [[mp.df1_du, mp.df1_dv],
                   [mp.df2_du, mp.df2_dv]]
            rhs = [-mp.f1, -mp.f2]
        elif unsteady and incompressible: # Incompressible unsteady viscous
            lhs = [[mp.df2_dv, mp.df2_dp],
                   [mp.df3_dv, mp.df3_dp]]
            rhs = [-mp.f2, -mp.f3]
        elif elastic and incompressible: # Steady compressible elastic
            lhs = [[mp.df2_du, mp.df2_dp],
                   [mp.df3_du, mp.df3_dp]]
            rhs = [-mp.f1, -mp.f3]
        elif elastic:
            lhs = mp.df2_du
            rhs = -mp.f2
        else:
            raise NotImplementedError('*** Model is not recognized/supported. ***')

        # # Check if system should be a block system or not.
        # if mp.config['formulation']['time']['unsteady']:
        #     if mp.config['material']['incompressible'] \
        #        and mp.config['material']['type'] == 'elastic':
        #         lhs = [[mp.df1_du, mp.df1_dv, mp.df1_dp],
        #                [mp.df2_du, mp.df2_dv, mp.df2_dp],
        #                [mp.df3_du, mp.df3_dv, mp.df3_dp]]
        #         rhs = [-mp.f1, -mp.f2, -mp.f3]
        #     elif mp.config['material']['incompressible']:
        #         lhs = [[mp.df2_dv, mp.df2_dp],[mp.df3_dv, mp.df3_dp]]
        #         rhs = [-mp.f2, -mp.f3]
        #     else:
        #         lhs = [[mp.df1_du, mp.df1_dv],
        #                [mp.df2_du, mp.df2_dv]]
        #         rhs = [-mp.f1, -mp.f2]
        # else:
        #     rhs = -mp.f2
        #     # ADD INCOMPRESSIBILITY CASE
        #     if mp.config['material']['type'] == 'elastic':
        #         lhs = mp.df2_du
        #     elif mp.config['material']['type'] == 'viscous':
        #         lhs = mp.df2_dv
        #     else:
        #         raise NotImplementedError

        return lhs, rhs
