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

    def __init__(self, mechanics_problem, fname_disp=None,
                 fname_vel=None, fname_pressure=None):

        # Make the MechanicsProblem object part of the member data.
        self._mp = mechanics_problem

        # Create file objects. This keeps the counter from being reset
        # each time the solve function is called.
        if fname_disp is not None:
            self._file_disp = dlf.File(fname_disp, "compressed")
        else:
            self._file_disp = None
        if fname_vel is not None:
            self._file_vel = dlf.File(fname_vel, "compressed")
        else:
            self._file_vel = None
        if fname_pressure is not None:
            self._file_pressure = dlf.File(fname_pressure, "compressed")
        else:
            self._file_pressure = None

        return None


    def solve(self, nonlinear_tol=1e-10, iter_tol=1e-8, maxNonlinIters=50,
              maxLinIters=200, show=0, print_norm=True, save_freq=1,
              lin_solver='mumps'):
        """


        """

        if self._mp.config['formulation']['time']['unsteady']:
            self.time_solve(nonlinear_tol=nonlinear_tol, iter_tol=iter_tol,
                            maxNonlinIters=maxNonlinIters,
                            maxLinIters=maxLinIters, show=show,
                            print_norm=print_norm, fname_disp=fname_disp,
                            fname_vel=fname_vel, fname_pressure=fname_pressure,
                            save_freq=save_freq)
        else:
            lhs, rhs = self.ufl_lhs_rhs()
            bcs = self._mp.dirichlet_bcs.values()
            if len(bcs) == 1:
                bcs = bcs[0]

            self.nonlinear_solve(lhs, rhs, bcs, nonlinear_tol=nonlinear_tol,
                                 iter_tol=iter_tol, maxNonlinIters=maxNonlinIters,
                                 maxLinIters=maxLinIters, show=show,
                                 print_norm=print_norm)

            if self._file_disp is not None:
                self._file_disp << self._mp.displacement
            if self._file_vel is not None:
                self._file_vel << self._mp.velocity
            if self._file_pressure is not None:
                self._file_pressure << self._mp.pressure

        return None


    def time_solve(self, nonlinear_tol=1e-10, iter_tol=1e-8,
                   maxNonlinIters=50, maxLinIters=200, show=0,
                   print_norm=True, save_freq=1, save_initial=True):
        """


        """

        t, tf = self._mp.config['formulation']['time']['interval']
        t0 = t
        dt = self._mp.config['formulation']['time']['dt']
        count = 0 # Used to check if file(s) should be saved.

        lhs, rhs = self.ufl_lhs_rhs()

        # Need to be more specific here.
        bcs = block.block_bc(self._mp.dirichlet_bcs.values(), False)

        # Save initial condition
        if save_initial:
            if self._file_disp is not None:
                self._file_disp << (self._mp.displacement, t)
                if not rank:
                    print('* Displacement saved *')
            if self._file_vel is not None:
                self._file_vel << (self._mp.velocity, t)
                if not rank:
                    print('* Velocity saved *')
            if self._file_pressure:
                self._file_pressure << (self._mp.pressure, t)
                if not rank:
                    print('* Pressure saved *')

        rank = dlf.MPI.rank(dlf.mpi_comm_world())

        while t <= tf:

            # Set to the next time step
            t += dt

            # Update expressions that depend on time
            self._mp.update_time(t, t0)

            # Print the current time
            if not rank:
                print ('*'*30)
                print ('t = %3.5f' % t)

            # Solve the nonlinear equation(s) at current time step.
            self.nonlinear_solve(lhs, rhs, bcs, nonlinear_tol=nonlinear_tol,
                                 iter_tol=iter_tol, maxNonlinIters=maxNonlinIters,
                                 maxLinIters=maxLinIters, show=show,
                                 print_norm=print_norm)

            # Prepare for the next time step
            if self._mp.displacement0 != 0:
                self._mp.displacement0.assign(self._mp.displacement)
            self._mp.velocity0.assign(self._mp.velocity)

            # Save current time step.
            if not count % save_freq:
                if self._file_disp is not None:
                    self._file_disp << (self._mp.displacement, t)
                    if not rank:
                        print('* Displacement saved *')
                if self._file_vel is not None:
                    self._file_vel << (self._mp.velocity, t)
                    if not rank:
                        print('* Velocity saved *')
                if self._file_pressure:
                    self._file_pressure << (self._mp.pressure, t)
                    if not rank:
                        print('* Pressure saved *')

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
                                        # nonconvergence_is_fatal=True,
                                        maxiter=maxLinIters)
                du = Ainv*b
            else:
                dlf.solve(A, du, b, 'mumps')

            self.update_soln(du)

            norm = du.norm('l2')
            if not rank and print_norm:
                print ('(iter %2i) norm %.3e' % (count, norm))

            count += 1

        return None


    def update_soln(self, du):
        """


        """

        incompressible = self._mp.config['material']['incompressible']
        unsteady = self._mp.config['formulation']['time']['unsteady']
        elastic = self._mp.config['material']['type'] == 'elastic'

        if unsteady and elastic and incompressible:
            self._mp.displacement.vector()[:] += du[0]
            self._mp.velocity.vector()[:] += du[1]
            self._mp.pressure.vector()[:] += du[2]
        elif unsteady and elastic: # Compressible unsteady elastic
            self._mp.displacement.vector()[:] += du[0]
            self._mp.velocity.vector()[:] += du[1]
        elif unsteady and incompressible: # Incompressible unsteady viscous
            self._mp.velocity.vector()[:] += du[0]
            self._mp.pressure.vector()[:] += du[1]
        elif elastic and incompressible: # Incompressible steady elastic
            self._mp.displacement.vector()[:] += du[0]
            self._mp.pressure.vector()[:] += du[1]
        elif elastic: # Compressible steady elastic
            self._mp.displacement.vector()[:] += du
        elif incompressible: # Incompressible steady viscous
            self._mp.velocity.vector()[:] += du[0]
            self._mp.pressure.vector()[:] += du[1]
        else:
            raise NotImplementedError('*** Model is not recognized/supported. ***')

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
        elif elastic and incompressible: # Steady incompressible elastic
            lhs = [[mp.df2_du, mp.df2_dp],
                   [mp.df3_du, mp.df3_dp]]
            rhs = [-mp.f2, -mp.f3]
        elif elastic: # Compressible steady elastic
            lhs = mp.df2_du
            rhs = -mp.f2
        elif incompressible: # Incompressible steady viscous
            lhs = [[mp.df2_dv, mp.df2_dp],
                   [mp.df3_dv, mp.df3_dp]]
            rhs = [-mp.f2, -mp.f3]
        else:
            raise NotImplementedError('*** Model is not recognized/supported. ***')

        return lhs, rhs
