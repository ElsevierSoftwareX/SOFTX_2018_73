from .utils import petsc_identity # This might not be necessary
from ufl import Form

import block
from block import iterative

import dolfin as dlf
import numpy as np

class MechanicsSolver(object):
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
                if fname_vel:
                    file_vel << (self._mp.velocity, t)

            # Set to the next time step
            t += dt

            # Update expressions that depend on time
            self._mp.update_time(t, t0)

            # Print the current time
            if not rank:
                print '*'*30
                print 't = %2.3f' % t

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
                if count > 0:
                    initial_guess = du
                else:
                    initial_guess = None

                Ainv = iterative.LGMRES(A, show=show, tolerance=iter_tol,
                                        initial_guess=initial_guess,
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


    # def explicit_euler(self, fname=None, save_freq=1):
    #     """


    #     """

    #     if fname:
    #         result_file = dlf.File(fname)

    #     mp = self._mp
    #     t, tf = mp.config['formulation']['time']['interval']
    #     dt = mp.config['formulation']['time']['dt']

    #     rank = dlf.MPI.rank(dlf.mpi_comm_world())
    #     count = 0

    #     while t <= tf:
    #         mp.update_time(t)
    #         un, vn = self.explicit_euler_step()

    #         un_norm = un.norm('l2')
    #         vn_norm = vn.norm('l2')

    #         mp.displacement.vector()[:] = un
    #         mp.velocity.vector()[:] = vn

    #         if fname and not count % save_freq:
    #             result_file << mp.displacement
    #             if rank == 0:
    #                 print '*'*40
    #                 print 't = %.3f' % t
    #                 print 'un.norm(\'l2\') = ', un_norm
    #                 print 'vn.norm(\'l2\') = ', vn_norm

    #         # Prepare for next step
    #         t += dt
    #         count += 1

    #     return None


    # def explicit_euler_step(self):
    #     """


    #     """

    #     mp = self._mp
    #     dt = mp.config['formulation']['time']['dt']
    #     u0 = mp.displacement.vector()
    #     v0 = mp.velocity.vector()
    #     M = mp._localAccelMatrix
    #     f0 = self.rhs()

    #     un = u0 + dt*v0
    #     mp.bc_apply('displacement', b=un)

    #     b = M*v0 + dt*f0
    #     mp.bc_apply('velocity', A=M, b=b)

    #     vn = dlf.PETScVector()
    #     dlf.solve(M, vn, b)

    #     return un, vn


    # def implicit_euler_step(self):
    #     """


    #     """

    #     raise NotImplementedError('This function has not been implemented yet.')

    #     return None


    def generalized_alpha(self, fname=None, save_freq=1):
        """


        """

        if fname:
            result_file = dlf.File(fname)

        mp = self._mp
        t, tf = mp.config['formulation']['time']['interval']
        dt = mp.config['formulation']['time']['dt']
        alpha = mp.config['formulation']['time']['alpha']

        rank = dlf.MPI.rank(dlf.mpi_comm_world())
        count = 0

        while t <= tf:
            # mp.update_time(t)
            un, vn = self.generalized_alpha_step(t, 50, 1e-10, 'mumps')

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

            t += dt
            count += 1

        return None


    def generalized_alpha_step(self, t, maxIters=50, tol=1e-10, lin_solver='mumps'):
        """


        """

        # mp = self._mp
        self.nonlinear_solve(t, maxIters, tol, lin_solver)

        # ########################################
        # # NEED TO CALL NONLINEAR_SOLVE

        # A = self.coefficient_matrix(t, alpha, u=u, v=v, p=p)
        # b = self.rhs(t, alpha, u, u0=u0, v=v, p=p, p0=p0)

        # bcs = block.block_bc(mp.dirichlet_bcs.values(), False)
        # rhs_bc = bcs.apply(A)
        # rhs_bcs.apply(b)

        # Ainv = iterative.LGMRES(A, tolerance=tol)
        # x = Ainv*b

        ########################################

        # Get necessary time-stepping parameters.
        #
        # In general, a call to the nonlinear solver will be required.
        # Might need to reformat the nonlinear_solve function so that
        # it can also be used in this case.

        return x


    def ufl_lhs_rhs(self):
        """


        """

        mp = self._mp

        # Check if system should be a block system or not.
        if self._mp.config['formulation']['time']['unsteady']:
            lhs = [[mp.df1_du, mp.df1_dv],
                   [mp.df2_du, mp.df2_dv]]
            rhs = [-mp.f1, -mp.f2]
        else:
            rhs = -mp.f2
            if mp.config['material']['type'] == 'elastic':
                lhs = mp.df2_du
            elif mp.config['material']['type'] == 'viscous':
                lhs = mp.df2_dv
            else:
                raise NotImplementedError

        return lhs, rhs


    def rhs(self, t0, alpha, u, u0=None,
            v=None, v0=None, p=None, p0=None):
        """


        """

        mp = self._mp
        if mp.config['formulation']['time']['unsteady']:
            b_velocity = self.rhs_velocity(alpha, u, u0, v, v0)
        else:
            b_velocity = None

        b_momentum = self.rhs_momentum(t0, alpha, u, u0,
                                       v, v0, p, p0)

        if b_velocity is not None:
            b = block.block_vec([b_velocity, b_momentum])
        else:
            b = b_momentum

        return b


    def rhs_momentum(self, t0, alpha, u, u0=None,
                     v=None, v0=None, p=None, p0=None):
        """


        """

        if (p is not None) or (p0 is not None):
            raise NotImplementedError

        elastic = self._mp.config['material']['type'] == 'elastic'
        if elastic:
            b_momentum = self.rhs_elastic_momentum(t0, alpha, u, u0,
                                                   v, v0, p, p0)
        else:
            b_momentum = self.rhs_viscous_momentum()

        return b_momentum


    def rhs_elastic_momentum(self, t0, alpha, u, u0=None,
                             v=None, v0=None, p=None, p0=None):
        """


        """

        if (p is not None) or (p0 is not None):
            raise NotImplementedError

        mp = self._mp
        u_vec = u.vector()
        b_temp = dlf.PETScVector()
        b = dlf.PETScVector()

        if mp.config['formulation']['time']['unsteady']:
            dt = mp.config['formulation']['time']['dt']
        else:
            dt = 0.0

        # Compute the contributions at the current iteration.
        # Note that this is the final result for a steady simulation.
        b = self.rhs_steady_elastic_momentum(t0+dt, u, p=None)

        # Need to add the contributions from the previous time
        # step if the problem is unsteady and weigh each according
        # to the value of alpha.
        if mp.config['formulation']['time']['unsteady']:
            # Weigh current time step by alpha.
            b *= alpha*dt

            v_vec = v.vector()
            v0_vec = v0.vector()

            b_temp = self.rhs_steady_elastic_momentum(t0, u0, p=p0)
            b += dt*(1.0 - alpha)*b_temp
            print mp._localAccelMatrix
            b -= mp._localAccelMatrix*(v_vec - v0_vec)

        return b


    # Should change the name of this function
    def rhs_steady_elastic_momentum(self, t, u, p=None):
        """


        """

        if p is not None:
            raise NotImplementedError

        mp = self._mp

        b = dlf.PETScVector()
        b_temp = dlf.PETScVector()

        # Assemble stress work vector at the current iteration.
        mp.assembleStressWorkVector(u, p=p, tensor=b)
        b *= -1.0

        # Add vector due to the body force
        if mp.ufl_body_force is not None:
            mp.assembleBodyForceVector(t, tensor=b_temp)
            b += b_temp

        # Add the vector due to the traction boundary condition
        if mp.ufl_neumann_bcs is not None:
            mp.assembleTractionVector(u, t, tensor=b_temp)
            b += b_temp

        return b


    def rhs_viscous_momentum(self):

        raise NotImplementedError


    def rhs_velocity(self, alpha, u, u0, v, v0):
        """


        """

        u_vec = u.vector()
        v_vec = v.vector()
        u0_vec = u0.vector()
        v0_vec = v0.vector()

        dt = self._mp.config['formulation']['time']['dt']
        b = dt*(alpha*v_vec + (1.0 - alpha)*v0_vec)
        b += u0_vec - u_vec

        return b


    def coefficient_matrix(self, t0, alpha, u, v=None, p=None):
        """


        """

        mp = self._mp
        if mp.config['formulation']['time']['unsteady']:
            n = u.geometric_dimension()
            A_vel1, A_vel2 = self.coefficient_matrices_velocity(alpha, n, scalar=False)
            print 'A_vel1.shape = ', A_vel1.array().shape
            print 'A_vel2.shape = ', A_vel2.array().shape
        else:
            A_vel1 = A_vel2 = None

        A_momentum1, A_momentum2 = self.coefficient_matrices_momentum(t0, alpha, u, v, p)

        all_matrices = [A_vel1, A_vel2, A_momentum1, A_momentum2]
        if None not in all_matrices:
            A = block.block_mat([all_matrices[:2], all_matrices[2:]])
        else:
            block_set = set(all_matrices)

            # Raise exception if more than one, but less than 4 matrices are
            # non-zero.
            if not len(block_set) == 2:
                raise ValueError('Expecting only one block matrix to be nonzero, got %i.' \
                                 % (len(block_set)-1))

            for mat in all_matrices:
                if mat is not None:
                    A = mat
                    break
        print 'A.blocks = ', A.blocks

        return A


    def coefficient_matrices_momentum(self, t0, alpha, u, v=None, p=None):
        """


        """

        if p is not None:
            raise NotImplementedError

        elastic = self._mp.config['material']['type'] == 'elastic'

        if elastic:
            K1, K2 = self.coefficient_matrices_elastic_momentum(t0, alpha, u, v, p)
        else:
            self.coefficient_matrices_viscous_momentum()

        return K1, K2


    def coefficient_matrices_elastic_momentum(self, t0, alpha, u, v=None, p=None):
        """


        """

        if p is not None:
            raise NotImplementedError

        mp = self._mp

        K_temp = dlf.PETScMatrix()
        K1 = dlf.PETScMatrix()

        mp.assembleStressWorkMatrix(u, p=p, tensor=K1)

        if mp.ufl_neumann_bcs_diff is not None:
            mp.assembleTractionMatrix(u, 0.0, tensor=K_temp)
            K1 -= K_temp

        if mp.config['formulation']['time']['unsteady']:
            dt = mp.config['formulation']['time']['dt']
            K2 = dlf.PETScMatrix(mp._localAccelMatrix)
            K1 *= alpha*dt
        else:
            K2 = None

        return K1, K2


    def coefficient_matrices_viscous_momentum(self):
        """


        """

        raise NotImplementedError

        return None


    def coefficient_matrices_velocity(self, alpha, n, scalar=False):
        """


        """

        mp = self._mp
        dt = mp.config['formulation']['time']['dt']

        if scalar:
            ret_val = 1.0, -dt*alpha
        else:
            dofs = mp.vectorSpace.dofmap().dofs()
            I = petsc_identity(n, dofs)
            I_dt_a = petsc_identity(n, dofs)
            I_dt_a *= -dt*alpha
            # ret_val = (I,)
            # I *= -dt*alpha
            # ret_val += (I,)

        return I, I_dt_a
