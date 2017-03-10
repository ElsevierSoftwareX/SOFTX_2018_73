import dolfin as dlf
import ufl

__all__ = ['LinearMaterial', 'NeoHookeMaterial', 'GuccioneMaterial']

class ElasticMaterial(object):
    """
    Base class defining constitutive equations for elastic materials.

    """

    def __init__(self):
        self._material_name  = ''
        self._parameters = {}
        self._incompressible = False
        self._inverse = False
        self._active = False
        self._material_class = 'isotropic'


    @staticmethod
    def default_parameters() :
        return {}


    def set_material_name(self, material_name):
        """
        Set material name.

        """

        self._material_name = material_name


    def get_material_class(self):
        """
        Return material class, i.e. isotropic, transversely isotropic,
        orthotropic or fully anisotropic.

        """

        return self._material_class


    def set_material_class(self, material_class):
        """
        Set material class, i.e. isotropic, transversely isotropic,
        orthotropic or fully anisotropic.

        """

        self._material_class = material_class


    def is_incompressible(self) :
        """
        Return True if the material is incompressible.
        """

        return self._incompressible


    def set_incompressible(self, boolIncompressible):
        """
        Set material to incompressible formulation.

        """

        self._incompressible = boolIncompressible


    def set_inverse(self, boolInverse):
        """
        Set material to inverse formulation.

        """

        self._inverse = boolInverse


    def is_inverse(self):
        """
        Return True if the material formulation is inverse.

        """

        return self._inverse


    def set_active(self, boolActive):
        """
        Set material to inverse formulation.

        """

        self._inverse = boolActive


    def is_active(self) :
        """
        Return True if the material supports active contraction
        """
        return self._active


    def print_info(self) :
        """
        Print material information.

        """

        print('-'*80)
        print('Material: %s' % self._material_name)
        print('Parameters: %s' % self._parameters)
        print('Material class: %s, ' % self._material_class)
        print('Properties: incompressible (%s), inverse (%s), active (%s)'
                % (self._incompressible, self._inverse, self._active))
        print('-'*80)


    def incompressibilityCondition(self, u):
        """
        Return the incompressibility condition for the specific material. The
        default is

        p = ln(J)/J


        Parameters
        ----------

        u :
            The displacement vector.


        Returns
        -------

        UFL object defining the incompressibility condition.

        """

        I    = dlf.Identity(ufl.domain.find_geometric_dimension(u))
        F    = I + dlf.grad(u)
        Finv = dlf.inv(F)
        J    = dlf.det(F)

        Bvol = dlf.ln(J)*dlf.inv(J)
        return Bvol


class LinearMaterial(ElasticMaterial):
    """
    Return the stress tensor based on the linear elasticity, i.e. infinitesimal
    deformations.

    """


    def __init__(self, inverse=False, **params):
        ElasticMaterial.__init__(self)
        ElasticMaterial.set_material_class(self, 'isotropic')
        ElasticMaterial.set_material_name(self, 'Linear material')
        ElasticMaterial.set_inverse(self, inverse)
        ElasticMaterial.set_incompressible(self, params['incompressible'])
        params = params or {}
        self._parameters = self.default_parameters()
        self._parameters.update(params)
        convert_elastic_moduli (self._parameters)


    @staticmethod
    def default_parameters():
        params = { 'mu': None,
                   'kappa': None,
                   'lambda': None,
                   'inv_la': None,
                   'E': None,
                   'nu': None }
        return params


    def stress_tensor(self, F, J, p=None, formulation=None):
        """
        Return the Cauchy stress tensor for a linear material, namely

        T = la*tr(e)*I + 2*mu*e,

        where e = sym(grad(u)), I is the identity tensor, and la & mu are
        the Lame parameters.


        Parameters
        ----------

        F :
            The deformation gradient.
        J :
            The jacobian, i.e. determinant of the deformation gradient. Note
            that this is not used for this material. It is solely a place holder
            to conform to the format of other materials.
        p : (default, None)
            The UFL pressure function for incompressible materials.
        formulation : (default, None)
            This input is not used for this material. It is solely a place holder
            to conform to the format of other materials.


        Returns
        -------

        T defined above.

        """

        params = self._parameters
        dim = ufl.domain.find_geometric_dimension(F)
        mu = dlf.Constant(params['mu'], name='mu')
        la = dlf.Constant(params['lambda'], name='lambda')

        I = dlf.Identity(dim)
        if self._inverse:
            epsilon = dlf.sym(dlf.inv(F)) - I
        else:
            epsilon = dlf.sym(F) - I

        if self._incompressible:
            T = -p*I + 2.0*mu*epsilon
        else:
            T = la*dlf.tr(epsilon)*I + 2.0*mu*epsilon

        return T


    def incompressibilityCondition(self, u):
        """
        Return the incompressibility condition for a linear material,
        p = div(u).


        Parameters
        ----------

        u :
            The displacement vector.


        Returns
        -------

        UFL object defining the incompressibility condition.

        """

        return dlf.div(u)


class NeoHookeMaterial(ElasticMaterial):
    """
    Return the first Piola-Kirchhoff stress tensor based on the strain
    energy function

    standard:    psi(C) = mu/2*(tr(C) - 3) - mu*ln(J) + la/2*(ln(J))**2.
    nearly-inc.: psi(C) = mu/2*(tr(C) - 3) + kappa/2*(ln(J))**2.

    Parameters
    ----------

    F :
        Deformation gradient for the problem.
    J :
        Determinant of the deformation gradient.
    la :
        First parameter for a neo-Hookean material.
    mu :
        Second parameter for a neo-Hookean material.

    """


    def __init__(self, inverse=False, **params):
        ElasticMaterial.__init__(self)
        ElasticMaterial.set_material_class(self, 'isotropic')
        ElasticMaterial.set_material_name(self, 'Neo-Hooke material')
        ElasticMaterial.set_inverse(self, inverse)
        ElasticMaterial.set_incompressible(self, params['incompressible'])
        params = params or {}
        self._parameters = self.default_parameters()
        self._parameters.update(params)
        convert_elastic_moduli(self._parameters)


    @staticmethod
    def default_parameters():
        params = {'mu' : None,
                  'kappa' : None,
                  'lambda' : None,
                  'E' : None,
                  'nu' : None}
        return params


    def strain_energy(self, F, J, formulation=None):
        """
        Define the total strain energy based on the incompressibility of
        the material (fully or nearly incompressible, or compressible), defined
        with respect to the deformation gradient, or by its inverse if the
        objective is to find the inverse displacement. The strain energy for a
        compressible material is defined by

        W = 0.5*mu*(I1 - dim) + 0.5*la*(ln(J))**2 - mu*ln(J)
          = 0.5*mu*(i2/i3 - dim) + 0.5*la*(ln(j))**2 - mu*ln(j),

        where I1 is the first invariant of C = F.T*F, while i2 and i3 are the
        second and third invariants of c = f.T*f, with f = inv(F). For a
        (nearly-)incompressible material, the strain energy is defined by

        W = U(J) + 0.5*mu*(I1 - dim)
          = U(j) + 0.5*mu*(i2/i3 - dim),

        where the invariants are now those of Cbar = J**(-2.0/dim)*C or cbar =
        j**(-2.0/dim)*c, and dU/dJ = p for fully incompressible material, while
        U(J) = kappa*phi(J), where the particular form of phi is given below.


        Parameters
        ----------

        F :
            The (forward or inverse) deformation gradient.
        J :
            The jacobian, i.e. determinant of the deformation gradient given
            above.
        formulation : (default, None)
            The formulation used for the nearly-incompressible materials.
            The accepted values are:

            * square:     phi(J) = 0.5*kappa*(J - 1)**2
            * log:        phi(J) = 0.5*kappa*(ln(J))**2


        Returns
        -------

        The strain energy, W, defined above.

        """

        if self._inverse:
            W = self._inverse_strain_energy(F, J, formulation)
        else:
            W = self._forward_strain_energy(F, J, formulation)

        return W


    def _forward_strain_energy(self, F, J, formulation=None):
        """
        Define the strain energy function for the Neo-Hookean material
        based on the forward deformation gradient, dx/dX.

        """

        mu = self._parameters['mu']

        if self._parameters['incompressible']:
            dim = ufl.domain.find_geometric_dimension(F)
            Fbar = J**(-1.0/dim)*F
            Cbar = Fbar.T*Fbar
            I1 = dlf.tr(Cbar)
            W = self._basic_strain_energy(I1, mu)

            # Note that the strain energy is the same for fully incompressible
            # and penalty formulations.
            kappa = self._parameters['kappa']
            W += self._penalty_strain_energy(J, kappa, formulation=formulation)
        else:
            la = self._parameters['lambda']
            C = F.T*F
            I1 = dlf.tr(C)
            W = self._basic_strain_energy(I1, mu)
            W += self._compressible_strain_energy(J, la, mu)

        return W


    def _inverse_strain_energy(self, f, j, formulation=None):
        """
        Define the strain energy function for the Neo-Hookean material
        based on the inverse deformation gradient, dX/dx.

        """

        mu = self._parameters['mu']

        if self._parameters['incompressible']:

            dim = ufl.domain.find_geometric_dimension(f)
            fbar = j**(-1.0/dim)*f
            cbar = fbar.T*fbar
            i1 = dlf.tr(cbar)
            i2 = dlf.Constant(0.5)*(i1**2 - dlf.tr(cbar**2))
            w = self._basic_strain_energy(i2, mu)

            if formulation is not None: # Nearly incompressible
                kappa = self._parameters['kappa']
                w += self._penalty_strain_energy(1.0/j, kappa,
                                                 formulation=formulation)
            else: # Fully incompressible
                # Need to figure out what to do here.
                pass

        else:
            la = self._parameters['lambda']
            c = f.T*f
            i1 = dlf.tr(c)
            i2 = dlf.Constant(0.5)*(i1**2 - dlf.tr(c**2))
            i3 = dlf.det(c)
            w = self._basic_strain_energy(i2/i3, mu)
            w += self._compressible_strain_energy(1.0/j, kappa,
                                                 formulation=formulation)

        return w


    def stress_tensor(self, F, J, p=None, formulation=None):
        """


        """

        if self._inverse:
            P = self._inverse_stress_tensor(F, J, p, formulation)
        else:
            P = self._forward_stress_tensor(F, J, p, formulation)

        return P


    def _forward_stress_tensor(self, F, J, p=None, formulation=None):
        """
        Define the (first Piola-Kirchhoff or Cauchy) stress tensor based on the
        incompressibility of the material (fully or nearly incompressible, or
        compressible), defined with respect to the deformation gradient, or by
        its inverse if the objective is to find the inverse displacement. The
        first Piola-Kirchhoff stress tensor for a compressible material is defined
        by

        P = [la*ln(J) - mu]*inv(F).T + mu*F
          = -[la*ln(j) + mu]*f.T + mu*inv(f),

        where f = inv(F), and j = det(f) = 1.0/det(F). For a (nearly-)
        incompressible material, the first Piola-Kirchhoff stress tensor is given
        by

        P = [U'(J)]*inv(F).T

        W = U(J) + 0.5*mu*(I1 - dim)
          = U(j) + 0.5*mu*(i2/i3 - dim),

        where the invariants are now those of Cbar = J**(-2.0/dim)*C or cbar =
        j**(-2.0/dim)*c, and dU/dJ = p for fully incompressible material, while
        U(J) = kappa*phi(J), where the particular form of phi is given below.


        Parameters
        ----------

        F :
            The (forward or inverse) deformation gradient.
        J :
            The jacobian, i.e. determinant of the deformation gradient given
            above.
        p : (default, None)
            The pressure scalar field. If it is set to None, the penalty method
            formulation will be used.
        formulation : (default, None)
            The formulation used for the nearly-incompressible materials.
            The accepted values are:

            * square:     phi(J) = 0.5*kappa*(J - 1)**2
            * log:        phi(J) = 0.5*kappa*(ln(J))**2


        Returns
        -------

        The strain energy, W, defined above.


        """

        mu = self._parameters['mu']

        if self._parameters['incompressible']:
            dim = ufl.domain.find_geometric_dimension(F)
            Fbar = J**(-1.0/dim)*F
            Fbar_inv = dlf.inv(Fbar)
            Cbar = Fbar.T*Fbar
            I1 = dlf.tr(Cbar)
            P = J**(-1.0/dim)*self._basic_stress_tensor(Fbar, mu)
            b_vol = (-1.0/dim)*mu*I1
            if p is None:
                kappa = self._parameters['kappa']
                b_vol += J*self._volumetric_strain_energy_diff(J, kappa, formulation)
            else:
                b_vol -= J*p
            P += b_vol*J**(-1.0/dim)*Fbar_inv.T
        else:
            la = self._parameters['lambda']
            C = F.T*F
            I1 = dlf.tr(C)
            P = self._basic_stress_tensor(F, mu)
            P += self._compressible_stress_tensor(F, J, la, mu)

        return P


    def _inverse_stress_tensor(self, f, j, p=None, formulation=None):
        """


        """

        mu = self._parameters['mu']
        finv = dlf.inv(f)
        c = f.T*f
        i1 = dlf.tr(c)
        i2 = dlf.Constant(0.5)*(i1**2 - dlf.tr(c*c))
        T = self._basic_stress_tensor(dlf.inv(c), mu)
        dim = ufl.domain.find_geometric_dimension(f)
        I = dlf.Identity(dim)

        if self._parameters['incompressible']:

            T *= j**(-5.0/dim)
            b_vol = (-1.0/dim)*mu*(-1.0/dim)*i2
            if p is None:
                kappa = self._parameters['kappa']
                b_vol += self._volumetric_strain_energy_diff(1.0/j, kappa,
                                                             formulation)
            else:
                b_vol -= p
            T += b_vol*I
        else:
            la = self._parameters['lambda']
            T = self._basic_stress_tensor(dlf.inv(c), mu)
            T += self._compressible_strain_energy_diff(1.0/j, la, mu)*I

        return T


    @staticmethod
    def _basic_strain_energy(I1, mu):
        """
        Define the strain energy function for the neo-Hookean model:

        psi(C) = 0.5*mu*(tr(C) - 3)


        Parameters
        ----------

        I1 :
            Trace of the right/left Cauchy-Green strain tensor.
        mu :
            Material constant.


        Returns
        -------

        UFL object defining the strain energy given above.

        """

        dim = ufl.domain.find_geometric_dimension(I1)

        return dlf.Constant(0.5)*mu*(I1 - dlf.Constant(dim))


    @staticmethod
    def _compressible_strain_energy(J, la, mu):
        """
        Define additional terms for the strain energy of a compressible
        neo-Hookean model:

        psi_hat(C) = 0.5*la*(ln(J))**2 - mu*ln(J)


        Parameters
        ----------

        J :
            Determinant of the deformation gradient.
        la :
            Material constant.
        mu :
            Material constant.


        Returns
        -------

        UFL object defining the component of the strain energy function given
        above.

        """

        return dlf.Constant(0.5)*la*(dlf.ln(J))**2 \
            - mu*dlf.ln(J)


    @staticmethod
    def _compressible_strain_energy_diff(J, la, mu):
        """


        """

        return (la*dlf.ln(J) - mu)/J


    @staticmethod
    def _volumetric_strain_energy(J, kappa, formulation='square'):
        """
        Define the additional penalty component for the strain energy function
        a nearly incompressible material:

        square: phi(C) = 0.5*kappa*(J - 1)**2
        log:    phi(C) = 0.5*kappa*(ln(J))**2


        Parameters
        ----------

        J :
            Determinant of the deformation gradient.
        kappa :
            Penalty constant.
        formulation : "square", "log"
            String specifying which of the two above formulations to use.


        Returns
        -------

        UFL object defining the penalty component of the strain energy function
        given above.

        """

        if formulation == 'square':
            f =  (J - dlf.Constant(1.0))**2
        elif formulation == 'log':
            f = (dlf.ln(J))**2
        else:
            s = "Formulation, \"%s\" of the volumetric strain energy" % formulation \
                + " function is not recognized."
            raise ValueError(s)

        return dlf.Constant(0.5)*kappa*f


    @staticmethod
    def _volumetric_strain_energy_diff(J, kappa, formulation='square'):
        """


        """

        if formulation == 'square':
            dfdJ = J - dlf.Constant(1.0)
        elif formulation == 'log':
            dfdJ = dlf.ln(J)/J
        else:
            s = "Formulation, \"%s\" of the volumetric strain energy" % formulation \
                + " function is not recognized."
            raise ValueError(s)

        return kappa*dfdJ


    @staticmethod
    def _basic_stress_tensor(F, mu):
        """
        Define the first Piola-Kirchhoff stress tensor that corresponds to a
        basic neo-Hookean strain energy function,

        psi(C) = 0.5*mu*(tr(C) - 3),

        namely, P = mu*F.


        Parameters
        ----------

        F :
            Deformation gradient.
        mu :
            Material constant.


        Returns
        -------

        UFL object defining the above tensor, P.

        """

        return mu*F


    @staticmethod
    def _compressible_stress_tensor(F, J, la, mu):
        """
        Define the additional terms of the first Piola-Kirchhoff stress tensor
        resulting from the strain energy component,

        psi_hat(C) = 0.5*la*(ln(J))**2 - mu*ln(J),

        namely, P = (la*ln(J) - mu)*Finv.T.


        Parameters
        ----------

        F :
            Deformation gradient.
        J :
            Determinant of the deformation gradient.
        la :
            Material constant.
        mu :
            Material constant.


        Returns
        -------

        UFL object defining the above tensor, P.

        """

        Finv = dlf.inv(F)

        return (la*dlf.ln(J) - mu)*Finv.T


    @staticmethod
    def _penalty_stress_tensor(F, J, kappa, formulation='square'):
        """
        Define the additional terms of the first Piola-Kirchhoff stress tensor
        from the strain energy component given by one of the two formulations,

        square: phi(C) = 0.5*kappa*(J - 1)**2
        log:    phi(C) = 0.5*kappa*(ln(J))**2

        namely,

        square: P = kappa*J*(J - 1)*Finv.T
        log:    P = kappa*ln(J)*Finv.T


        Parameters
        ----------

        F :
            Deformation gradient.
        J :
            Determinant of the deformation gradient.
        kappa :
            Penalty constant.
        formulation : "square" (default), "log"
            String specifying which of the two above formulations to use.


        Returns
        -------

        UFL object defining the above tensor, P.

        """

        Finv = dlf.inv(F)
        if formulation == 'square':
            g = J*(J - dlf.Constant(1.0))
        elif formulation == 'log':
            g = dlf.ln(J)
        else:
            s = "Formulation, \"%s\" of the penalty function is not recognized." \
                % formulation
            raise ValueError(s)

        return kappa*g*Finv.T


class GuccioneMaterial(ElasticMaterial):

    def __init__(self, parameters={}, fibers={}, **kwargs):
        ElasticMaterial.__init__(self)
        ElasticMaterial.set_material_class(self, 'transversely isotropic')
        ElasticMaterial.set_material_name(self, 'Guccione material')
        self._parameters = self.default_parameters()
        if parameters == {}:
            prms = kwargs or {}
            self._parameters.update(prms)
        else:
            self._parameters.update(parameters)
        if self._parameters['bt'] == 1.0 and self._parameters['bf'] == 1.0 \
           and self._parameters['bfs'] == 1.0:
            ElasticMaterial.set_material_class(self, 'isotropic')
        fbrs = kwargs or {}
        self._fiber_directions = self.default_fiber_directions()
        self._fiber_directions.update(fibers)
        self._fiber_directions.update(fbrs)

    @staticmethod
    def default_parameters():
        param = {'C' : 2.0,
                 'bf' : 8.0,
                 'bt' : 2.0,
                 'bfs' : 4.0,
                 'kappa' : 1000.0}
        return param

    @staticmethod
    def default_fiber_directions():
        fibers = {'e1' : None,
                  'e2' : None,
                  'e3' : None}
        return fibers

    def strain_energy(self, u, p=None):
        """
        UFL form of the strain energy.
        """
        params = self._parameters
        dim = ufl.domain.find_geometric_dimension(u)

        I = dlf.Identity(dim)
        F = I + dlf.grad(u)
        J = dlf.det(F)
        C = pow(J, -float(2)/dim) * F.T*F
        E = 0.5*(C - I)

        CC = dlf.Constant(params['C'], name='C')
        if self.get_material_class(self) == 'isotropic':
            # isotropic case
            Q = dlf.inner(E, E)
        else:
            # fully anisotropic
            fibers = self._fiber_directions
            bt  = dlf.Constant(params['bt'], name='bt')
            bf  = dlf.Constant(params['bf'], name='bf')
            bfs = dlf.Constant(params['bfs'], name='bfs')

            e1 = fibers['e1']
            e2 = fibers['e2']
            if e1 is None or e2 is None:
                if dim == 2:
                    e1 = dlf.Constant((1.0,0.0))
                    e2 = dlf.Constant((0.0,1.0))
                    e3 = dlf.Constant((0.0,0.0))
                elif dim == 3:
                    e1 = dlf.Constant((1.0,0.0,0.0))
                    e2 = dlf.Constant((0.0,1.0,0.0))
                    e3 = dlf.Constant((0.0,0.0,1.0))
            else:
                e3 = dlf.cross(e1,e2) #params['e3'] #

            E11,E12,E13 = dlf.inner(E*e1,e1), dlf.inner(E*e1,e2), dlf.inner(E*e1,e3)
            E21,E22,E23 = dlf.inner(E*e2,e1), dlf.inner(E*e2,e2), dlf.inner(E*e2,e3)
            E31,E32,E33 = dlf.inner(E*e3,e1), dlf.inner(E*e3,e2), dlf.inner(E*e3,e3)

            Q = bf*E11**2 + bt*(E22**2 + E33**2 + E23**2 + E32**2) \
              + bfs*(E12**2 + E21**2 + E13**2 + E31**2)

        # passive strain energy
        Wpassive = CC/2.0*(dlf.exp(Q) - 1)

        # incompressibility
        if self._incompressible:
            Winc = - p*(J - 1)
        else :
            kappa = dlf.Constant(params['kappa'], name='kappa')
            Winc = kappa*(J**2 - 1 - 2*dlf.ln(J))

        return Wpassive + Winc

    def stress_tensor(self, u, p=None):
        """
        UFL form of the stress tensor.
        """
        dim = ufl.domain.find_geometric_dimension(u)
        params = self._parameters
        kappa = dlf.Constant(params['kappa'], name='kappa')
        CC = dlf.Constant(params['C'], name='C')
        I = dlf.Identity(dim)
        F = I + dlf.grad(u)
        J = dlf.det(F)
        Jm2d = pow(J, -float(2)/dim)
        C = F.T*F
        E_ = 0.5*(Jm2d*C - I)
        Finv = dlf.inv(F)

        # fully anisotropic
        bt = dlf.Constant(params['bt'], name='bt')
        bf = dlf.Constant(params['bf'], name='bf')
        bfs = dlf.Constant(params['bfs'], name='bfs')

        e1 = self._fiber_directions['e1']
        e2 = self._fiber_directions['e2']
        if e1 == None or e2 == None:
            if dim == 2:
                e1 = dlf.Constant((1.0,0.0))
                e2 = dlf.Constant((0.0,1.0))
                e3 = dlf.Constant((0.0,0.0))
            elif dim == 3:
                e1 = dlf.Constant((1.0,0.0,0.0))
                e2 = dlf.Constant((0.0,1.0,0.0))
                e3 = dlf.Constant((0.0,0.0,1.0))
        else:
            e3 = dlf.cross(e1,e2)

        E11,E12,E13 = dlf.inner(E_*e1,e1), dlf.inner(E_*e1,e2), dlf.inner(E_*e1,e3)
        E21,E22,E23 = dlf.inner(E_*e2,e1), dlf.inner(E_*e2,e2), dlf.inner(E_*e2,e3)
        E31,E32,E33 = dlf.inner(E_*e3,e1), dlf.inner(E_*e3,e2), dlf.inner(E_*e3,e3)

        Q = bf*E11**2 + bt*(E22**2 + E33**2 + E23**2 + E32**2) \
          + bfs*(E12**2 + E21**2 + E13**2 + E31**2)
        S_ = CC*dlf.exp(Q) \
            *(bf*E11*dlf.outer(e1,e1) + bt*( E22*dlf.outer(e2,e2) \
            + E33*dlf.outer(e3,e3) + E23*dlf.outer(e2,e3) \
            + E32*dlf.outer(e3,e2)) + bfs*(E12*dlf.outer(e1,e2) \
            + E21*dlf.outer(e2,e1) + E13*dlf.outer(e1,e3) \
            + E31*dlf.outer(e3,e1)))
        FS_isc = Jm2d*F*S_ - 1./dim*Jm2d*dlf.tr(C*S_)*Finv.T

        # incompressibility
        if self._incompressible:
            FS_vol = J*p*Finv.T
        else:
            FS_vol = J*2.*kappa*(J-1./J)*Finv.T

        return FS_vol + FS_isc

def convert_elastic_moduli(param, tol=1e8):
        # original parameters
        nu = param['nu']         # Poisson's ratio [-]
        E = param['E']           # Young's modulus [kPa]
        kappa = param['kappa']   # bulk modulus [kPa]
        mu = param['mu']         # shear modulus (Lame's second parameter) [kPa]
        la = param['lambda']     # Lame's first parameter [kPa]
        inv_la = param['inv_la'] # Inverse of Lame's first parameter [kPa]

        if (la is not None) and (inv_la is not None):
            raise ValueError("The user must provide either 'lambda' or "\
                             + "'inv_la', but not both.")

        if la is not None:
            try:
                inv_la = 1.0/la
                if inv_la > tol:
                    raise ZeroDivisionError
            except ZeroDivisionError:
                inv_la = float('inf')
        else:
            try:
                la = 1.0/inv_la
                if la > tol:
                    raise ZeroDivisionError
            except ZeroDivisionError:
                la = float('inf')

        if (kappa > 0) and (mu > 0):
            E = 9.*kappa*mu / (3.*kappa + mu)
            la = kappa - 2.*mu/3.
            nu = (3.*kappa - 2.*mu) / (2.*(3.*kappa+mu))
        elif (la > 0) and (mu > 0):
            E = mu*(3.*la + 2.*mu) / (la + mu)
            kappa = la + 2.*mu / 3.
            nu = la / (2.*(la + mu))
        elif (inv_la > 0) and (mu > 0):
            E = mu*(3.0 + 2.0*mu*inv_la)/(1.0 + mu/inv_la)
            kappa = 1.0/inv_la + 2.0*mu/3.0
            nu = 1.0/(2.0*(1.0 + mu*inv_la))
        elif (0 < nu <= 0.5) and (E > 0):
            kappa = E / (3.*(1 - 2.*nu))
            la = E*nu / ((1. + nu)*(1. - 2.*nu))
            mu = E / (2.*(1. + nu))
        else:
            raise ValueError('Two material parameters must be specified.')

        s = 'Parameter %s was changed due to contradictory settings.'
        if (param['E'] is not None) and (param['E'] != E):
            print s % 'E'
        if (param['kappa'] is not None) and (param['kappa'] != kappa):
            print s % 'kappa'
        if (param['lambda'] is not None) and (param['lambda'] != la):
            print s % 'lambda'
        if (param['mu'] is not None) and (param['mu'] != mu):
            print s % 'mu'
        if (param['nu'] is not None) and (param['nu'] != nu):
            print s % 'nu'

        param['nu'] = dlf.Constant(nu)       # Poisson's ratio [-]
        param['E'] = dlf.Constant(E)         # Young's modulus [kPa]
        param['kappa'] = dlf.Constant(kappa) # bulk modulus [kPa]
        param['mu'] = dlf.Constant(mu)       # shear modulus (Lame's second parameter) [kPa]
        param['lambda'] = dlf.Constant(la)  # Lame's first parameter [kPa]
