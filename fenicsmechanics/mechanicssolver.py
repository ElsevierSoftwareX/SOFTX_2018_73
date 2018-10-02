from __future__ import print_function

import sys
if sys.version_info.major == 2:
    import block
    from block import iterative
else:
    # raise ImportError("The module mechanicssolver is not compatible with Python 3.")
    msg = """
    The module 'mechanicssolver' is not compatible with Python 3. Thus,
    it's functionality is severely limited.
    """
    print(msg)

from .utils import _create_file_objects, _write_objects # This might not be necessary
from .dolfincompat import MPI_COMM_WORLD
from ufl import Form

from mpi4py import MPI

import dolfin as dlf
import numpy as np

__all__ = ['MechanicsBlockSolver']

class MechanicsBlockSolver(object):
    """
    This class assembles the UFL variational forms from a MechanicsProblem
    object, and calls solvers to solve the resulting (nonlinear) algebraic
    equations. If the problem is time-dependent, this class loops through
    the time interval specified in the 'config' dictionary. An LGMRES
    algorithm from CBC-Block is used for time-dependent problems. For
    steady problems, the user may choose from the linear solvers available
    through dolfin. Furthermore, Newton's method is used to solve the
    resulting algebraic equations.


    """

    def __init__(self, mechanics_problem, fname_disp=None,
                 fname_vel=None, fname_pressure=None,
                 fname_hdf5=None, fname_xdmf=None):
        """
        Initialize a MechanicsBlockSolver object.


        Parameters
        ----------

        mechanics_problem : MechanicsProblem
            A MechanicsProblem object that contains the necessary UFL
            forms to define and solve the problem specified in its
            config dictionary.
        fname_disp : str (default None)
            Name of the file series in which the displacement values are
            to be saved.
        fname_vel : str (default None)
            Name of the file series in which the velocity values are to
            be saved.
        fname_pressure : str (default None)
            Name of the file series in which the pressure values are to
            be saved.


        """

        # Make the MechanicsProblem object part of the member data.
        self._mp = mechanics_problem

        # Create file objects. This keeps the counter from being reset
        # each time the solve function is called.
        self._file_disp, self._file_vel, self._file_pressure, \
            self._file_hdf5, self._file_xdmf \
            = _create_file_objects(fname_disp, fname_vel, fname_pressure,
                                   fname_hdf5, fname_xdmf)

        return None


    def solve(self, nonlinear_tol=1e-10, iter_tol=1e-8, maxNonlinIters=50,
              maxLinIters=200, show=0, print_norm=True, save_freq=1,
              save_initial=True, lin_solver='mumps'):
        """
        Solve the mechanics problem defined in the MechanicsProblem object.


        Parameters
        ----------

        nonlinear_tol : float (default 1e-10)
            Tolerance used to terminate Newton's method.
        iter_tol : float (default 1e-8)
            Tolerance used to terminate the iterative linear solver.
        maxNonlinIters : int (default 50)
            Maximum number of iterations for Newton's method.
        maxLinIters : int (default 200)
            Maximum number of iterations for iterative linear solver.
        show : int (default 0)
            Amount of information for iterative.LGMRES to show. See
            documentation of this class for different log levels.
        print_norm : bool (default True)
            True if user wishes to see the norm at every linear iteration
            and False otherwise.
        save_freq : int (default 1)
            The frequency at which the solution is to be saved if the problem is
            unsteady. E.g., save_freq = 10 if the user wishes to save the solution
            every 10 time steps.
        save_initial : bool (default True)
            True if the user wishes to save the initial condition and False otherwise.
        lin_solver : str (default "mumps")
            Name of the linear solver to be used for steady compressible elastic
            problems. See the dolfin.solve documentation for a list of available
            linear solvers.


        Returns
        -------

        None


        """

        if self._mp.config['formulation']['time']['unsteady']:
            self.time_solve(nonlinear_tol=nonlinear_tol, iter_tol=iter_tol,
                            maxNonlinIters=maxNonlinIters,
                            maxLinIters=maxLinIters, show=show,
                            print_norm=print_norm,
                            save_freq=save_freq,
                            save_initial=save_initial)
        else:
            lhs, rhs = self.ufl_lhs_rhs()
            bcs = self._mp.dirichlet_bcs.values()
            if len(bcs) == 1:
                bcs = bcs[0]

            self.nonlinear_solve(lhs, rhs, bcs, nonlinear_tol=nonlinear_tol,
                                 iter_tol=iter_tol, maxNonlinIters=maxNonlinIters,
                                 maxLinIters=maxLinIters, show=show,
                                 print_norm=print_norm, lin_solver=lin_solver)

            u = self._mp.displacement
            v = self._mp.velocity
            p = self._mp.pressure
            f_objs = [self._file_disp, self._file_vel, self._file_pressure]

            _write_objects(f_objs, t=None, close=False, u=u, v=v, p=p)
            if self._file_hdf5 is not None:
                _write_objects(self._file_hdf5, t=t, close=False, u=u, p=p)
            if self._file_xdmf is not None:
                _write_objects(self._file_xdmf, t=t, close=False, u=u, p=p)

        return None


    def time_solve(self, nonlinear_tol=1e-10, iter_tol=1e-8,
                   maxNonlinIters=50, maxLinIters=200, show=0,
                   print_norm=True, save_freq=1, save_initial=True,
                   lin_solver="mumps"):
        """
        Loop through the time interval using the time step specified
        in the MechanicsProblem config dictionary.


        Parameters
        ----------

        nonlinear_tol : float (default 1e-10)
            Tolerance used to terminate Newton's method.
        iter_tol : float (default 1e-8)
            Tolerance used to terminate the iterative linear solver.
        maxNonlinIters : int (default 50)
            Maximum number of iterations for Newton's method.
        maxLinIters : int (default 200)
            Maximum number of iterations for iterative linear solver.
        show : int (default 0)
            Amount of information for iterative.LGMRES to show. See
            documentation of this class for different log levels.
        print_norm : bool (default True)
            True if user wishes to see the norm at every linear iteration
            and False otherwise.
        save_freq : int (default int)
            The frequency at which the solution is to be saved if the problem is
            unsteady. E.g., save_freq = 10 if the user wishes to save the solution
            every 10 time steps.
        save_initial : bool (default True)
            True if the user wishes to save the initial condition and False otherwise.
        lin_solver : str (default "mumps")
            Name of the linear solver to be used for steady compressible elastic
            problems. See the dolfin.solve documentation for a list of available
            linear solvers.


        Returns
        -------

        None


        """

        t, tf = self._mp.config['formulation']['time']['interval']
        t0 = t
        dt = self._mp.config['formulation']['time']['dt']
        count = 0 # Used to check if file(s) should be saved.

        lhs, rhs = self.ufl_lhs_rhs()

        # Need to be more specific here.
        bcs = block.block_bc(self._mp.dirichlet_bcs.values(), False)

        rank = dlf.MPI.rank(MPI_COMM_WORLD)

        p = self._mp.pressure
        u = self._mp.displacement
        v = self._mp.velocity
        f_objs = [self._file_pressure, self._file_disp, self._file_vel]

        # Save initial condition
        if save_initial:
            _write_objects(f_objs, t=t, close=False, u=u, v=v, p=p)
            if self._file_hdf5 is not None:
                _write_objects(self._file_hdf5, t=t, close=False, u=u, p=p)
            if self._file_xdmf is not None:
                _write_objects(self._file_xdmf, t=t, close=False, u=u, p=p)

        rank = dlf.MPI.rank(MPI_COMM_WORLD)

        while t < (tf - dt/10.0):

            # Set to the next time step
            t += dt

            # Update expressions that depend on time
            self._mp.update_time(t, t0)

            # Print the current time
            if not rank:
                print('*'*30)
                print('t = %3.5f' % t)

            # Solve the nonlinear equation(s) at current time step.
            self.nonlinear_solve(lhs, rhs, bcs, nonlinear_tol=nonlinear_tol,
                                 iter_tol=iter_tol, maxNonlinIters=maxNonlinIters,
                                 maxLinIters=maxLinIters, show=show,
                                 print_norm=print_norm, lin_solver=lin_solver)

            # Prepare for the next time step
            if self._mp.displacement0 != 0:
                self._mp.displacement0.assign(self._mp.displacement)
            self._mp.velocity0.assign(self._mp.velocity)

            t0 = t
            count += 1

            MPI.COMM_WORLD.Barrier()

            if not count % save_freq:
                _write_objects(f_objs, t=t, close=False, u=u, v=v, p=p)
                if self._file_hdf5 is not None:
                    _write_objects(self._file_hdf5, t=t, close=False, u=u, p=p)
                if self._file_xdmf is not None:
                    _write_objects(self._file_xdmf, t=t, close=False, u=u, p=p)

        return None


    def nonlinear_solve(self, lhs, rhs, bcs, nonlinear_tol=1e-10,
                        iter_tol=1e-8, maxNonlinIters=50, maxLinIters=200,
                        show=0, print_norm=True, lin_solver="mumps"):
        """
        Solve the nonlinear system of equations using Newton's method.


        Parameters
        ----------

        lhs : ufl.Form, list
            The definition of the left-hand side of the resulting linear
            system of equations.
        rhs : ufl.Form, list
            The definition of the right-hand side of the resulting linear
            system of equations.
        bcs : dolfin.DirichletBC, list
            Object specifying the Dirichlet boundary conditions of the
            system.
        nonlinear_tol : float (default 1e-10)
            Tolerance used to terminate Newton's method.
        iter_tol : float (default 1e-8)
            Tolerance used to terminate the iterative linear solver.
        maxNonlinIters : int (default 50)
            Maximum number of iterations for Newton's method.
        maxLinIters : int (default 200)
            Maximum number of iterations for iterative linear solver.
        show : int (default 0)
            Amount of information for iterative.LGMRES to show. See
            documentation of this class for different log levels.
        print_norm : bool (default True)
            True if user wishes to see the norm at every linear iteration
            and False otherwise.
        lin_solver : str (default "mumps")
            Name of the linear solver to be used for steady compressible elastic
            problems. See the dolfin.solve documentation for a list of available
            linear solvers.


        Returns
        -------

        None

        """

        norm = 1.0
        count = 0
        rank = dlf.MPI.rank(MPI_COMM_WORLD)

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
                dlf.solve(A, du, b, lin_solver)

            self.update_soln(du)

            norm = du.norm('l2')
            if not rank and print_norm:
                print('(iter %2i) norm %.3e' % (count, norm))

            count += 1

        return None


    def update_soln(self, du):
        """
        Update the values of the field variables based on the solution
        to the resulting linear system.


        Parameters
        ----------

        du : block.block_vec, dolfin.Vector
            The solution to the resulting linear system of the variational
            problem defined by the MechanicsProblem object.


        Returns
        -------

        None


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
        Return the UFL objects that define the left and right hand sides
        of the resulting linear system for the variational problem defined
        by the MechanicsProblem object.


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
