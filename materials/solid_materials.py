import dolfin as dlf
import ufl

__all__ = ['ElasticMaterial', 'LinearMaterial', 'NeoHookeMaterial',
           'FiberMaterial', 'FungMaterial', 'GuccioneMaterial']

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
    def default_parameters():
        return {}


    def set_material_name(self, material_name):
        """
        Set material name.


        Parameters
        ----------

        material_name : str
            Name of the constitutive equation used.


        Returns
        -------

        None


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


        Parameters
        ----------

        material_class : str
            Name of the type of material.


        Returns
        -------

        None


        """

        self._material_class = material_class


    def is_incompressible(self) :
        """
        Return True if the material is incompressible and False otherwise.


        """

        return self._incompressible


    def set_incompressible(self, boolIncompressible):
        """
        Set material to incompressible formulation.


        Parameters
        ----------

        boolIncompressible : bool
            True if the material is incompressible and False otherwise.


        Returns
        -------

        None


        """

        self._incompressible = boolIncompressible


    def set_inverse(self, boolInverse):
        """
        Set material to inverse elastostatics formulation.

        Parameters
        ----------

        boolInverse : bool
            True if the problem is an inverse elastostatics problem and False
            otherwise.


        Returns
        -------

        None


        """

        self._inverse = boolInverse


    def is_inverse(self):
        """
        Return True if the material is formulated for an inverse elastostatics
        problem and False otherwise.


        """

        return self._inverse


    def set_active(self, boolActive):
        """
        Set material to active formulation.


        Parameters
        ----------

        boolActive : bool
            True if the material is active and False if it is passive.


        Returns
        -------

        None


        """

        self._inverse = boolActive


    def is_active(self) :
        """
        Return True if the material supports active contraction and False
        otherwise.


        """
        return self._active


    def print_info(self) :
        """
        Print material parameters, class, and name.


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
    Return the Cauchy stress tensor based on the linear elasticity, i.e.
    infinitesimal deformations, both compressible and incompressible. The
    stress tensor is given by

    (Compressible)    T = la*tr(e)*I + 2*mu*e,
    (Incompressible)  T = -p*I + 2*mu*e,

    where la and mu are the Lame material parameters, e = sym(grad(u))
    where u is the displacement, and p is the pressure in the case of an
    incompressible material.

    The inverse elastostatics formulation is also supported for this material
    model. In that case, the only change that must be accounted for is the
    fact that

    e = sym(F^{-1}) - I.

    At least two of the material constants from the list below must be provided
    in the 'material' subdictionary of 'config' in addition to the values
    already listed in the documentation of BaseMechanicsProblem.

    * 'la' : float
        The first Lame parameter used as shown in the equations above. Note:
        providing la and inv_la does not qualify as providing two material
        parameters.
    * 'mu' : float
        The second Lame parameter used as shown in the equations above.
    * 'kappa' : float
        The bulk modulus of the material
    * 'inv_la' : float
        The reciprocal of the first Lame parameter, la. Note: providing la and
        inv_la does not qualify as providing two material parameters.
    * 'E' : float
        The Young's modulus of the material.
    * 'nu' : float
        The Poisson's ratio of the material.

    The remaining constants will be computed based on the parameters provided
    by the user.


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
        convert_elastic_moduli(self._parameters)


    @staticmethod
    def default_parameters():
        params = { 'mu': None,
                   'kappa': None,
                   'la': None,
                   'inv_la': None,
                   'E': None,
                   'nu': None }
        return params


    def stress_tensor(self, F, J, p=None, formulation=None):
        """
        Return the Cauchy stress tensor for a linear material, namely

        (Compressible)    T = la*tr(e)*I + 2*mu*e,
        (Incompressible)  T = -p*I + 2*mu*e,

        where e = sym(grad(u)), I is the identity tensor, and la & mu are
        the Lame parameters.


        Parameters
        ----------

        F :
            The deformation gradient.
        J :
            The Jacobian, i.e. determinant of the deformation gradient. Note
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
        la = dlf.Constant(params['la'], name='la')

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
    Return the first Piola-Kirchhoff stress tensor based on variations of the
    strain energy function

    psi(C) = 0.5*mu*(tr(C) - n),

    where C = F.T*F, and n is the geometric dimension. For nearly incompressible
    materials, the total strain energy function is given by

    W = U(J) + psi(C),

    where U(J) corresponds to the strain energy in response to dilatation of
    the material with J = det(F). The two forms of U(J) supported here are

    (square)  U(J) = 0.5*kappa*(J - 1)**2,
    (log)     U(J) = 0.5*kappa*(ln(J))**2,

    where kappa is the bulk modulus of the material. This results in the first
    Piola-Kirchhoff stress tensor given by

    P = J*(dU/dJ)*F^{-T} + mu*F.

    In the case of an incompressible material, the total strain energy is
    assumed to be of the form

    W = U(J) + psi(Cbar),

    where Cbar = J^{-2/n}. Furthermore, the pressure scalar field is defined
    such that p = -dU/dJ. The resulting first Piola-Kirchhoff stress tensor
    is then

    P = [J*(dU/dJ) - (1/n)*mu*J^{-2/n}*tr(C)]*F^{-T} + mu*J^{-2/n}*F.

    The inverse elastostatics formulation is also supported for this material
    model. In that case, the Cauchy stress tensor for compressible material is

    T = -j^2*(dU/dj)*I + mu*c^{-1},

    and

    T = -[p + (1/n)*mu*j^{-1/n}*i2]*I + mu*j^{5/n}*c^{-1},

    for incompressible material where j = det(f) = det(F^{-1}) = 1/J, c = f^T*f,
    i2 is the second invariant of c, and p is the pressure in the latter case.
    Note that f is the deformation gradient from the current configuration to
    the reference configuration.

    At least two of the material constants from the list below must be provided
    in the 'material' subdictionary of 'config' in addition to the values
    already listed in the documentation of BaseMechanicsProblem.

    * 'la' : float
        The first parameter used as shown in the equations above. Note: providing
        la and inv_la does not qualify as providing two material parameters.
    * 'mu' : float
        The second material parameter used as shown in the equations above.
    * 'kappa' : float
        The bulk modulus of the material
    * 'inv_la' : float
        The reciprocal of the first parameter, la. Note: providing la and inv_la
        does not qualify as providing two material parameters.
    * 'E' : float
        The Young's modulus of the material. Note: this is not entirely consistent
        with the neo-Hookean formulation, but la and mu will be computed based
        on the relation between the Young's modulus and the Lame parameters if
        E is given.
    * 'nu' : float
        The Poisson's ratio of the material. Note: this is not entirely consistent
        with the neo-Hookean formulation, but la and mu will be computed based
        on the relation between the Poisson's ratio and the Lame parameters if
        nu is given.


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
        params = {'mu': None,
                  'kappa': None,
                  'la': None,
                  'inv_la': None,
                  'E': None,
                  'nu': None}
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
            la = self._parameters['la']
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
            la = self._parameters['la']
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
        Return the first Piola-Kirchhoff stress tensor for a neo-Hookean
        material. The stress tensor is given by

        (Compressible)    P = J*(dU/dJ)*F^{-T} + mu*F,
        (Incompressible)  P = [-J*p - mu*J^(-2/n)/n*tr(C)]*F^{-T} + mu*J^{-2/n}*F,

        where F is the deformation gradient, J = det(F), C = F^T*F, and mu is
        a material constant. If the problem is an inverse elastostatics problem,
        the Cauchy stress tensor is given by

        (Compressible)   T = [-j^2*(dU/dj) - mu*j^{-1/n}/n*i2]*I + mu*j^{-5/n}*c^{-1}
        (Incompressible) T = -[p + mu*j^{-1/n}/n*i2]*I + mu*j^{-5/n}*c^{-1}

        where f = F^{-1}, j = det(f), c = f^T*f, and n is the geometric dimension.


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

        The stress tensor, P, defined above.


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
            la = self._parameters['la']
            C = F.T*F
            I1 = dlf.tr(C)
            P = self._basic_stress_tensor(F, mu)
            P += self._compressible_stress_tensor(F, J, la, mu)

        return P


    def _inverse_stress_tensor(self, f, j, p=None, formulation=None):
        """
        Return the Cauchy stress tensor for an inverse elastostatics problem.
        The Cauchy stress tensor is given by

        (Compressible)   T = [-j^2*(dU/dj) - mu*j^{-1/n}/n*i2]*I + mu*j^{-5/n}*c^{-1}
        (Incompressible) T = -[p + mu*j^{-1/n}/n*i2]*I + mu*j^{-5/n}*c^{-1}

        where f = F^{-1}, j = det(f), c = f^T*f, and n is the geometric dimension.


        Parameters
        -------

        f :
            The deformation gradient from the current to the reference
            configuration.
        j :
            The determinant of f.
        p :
            The pressure field variable.
        formulation : str (default None)
            Choose between quadratic and log formulations.


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
            la = self._parameters['la']
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
        The derivative of the isochoric component of the strain energy,

        dU/dJ = (la*dlf.ln(J) - mu)/J.


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
        Return the derivative of the volumetric component of strain energy,

        (Square)  dU/dJ = kappa*(J - 1.0),
        (Log)     dU/dJ = kappa*ln(J)/J.


        Parameters
        ----------

        J :
            Determinant of the deformation gradient.
        kappa :
            Bulk modulus.
        formulation : str (default "square")
            Choose between square and log formulation above.


        Returns
        -------

        dU/dJ


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


class FiberMaterial(ElasticMaterial):
    """
    Base class for fiber reinforced materials. This base class contains
    methods for loading and saving vector fields that represent the directions
    tangent to the fibers throughout the domain. Note that this class does
    not provide any constitutive equation and is merely a utility for common
    operations with fiber reinforced materials.

    In addition to the values listed in the documentation of BaseMechanicsProblem
    for the 'material' subdictionary of 'config', the user must provide a
    subdictionary within 'material' named 'fibers' with the following values:

    * 'fiber_files' : str, list, tuple, dolfin.Coefficient
        The name(s) of the files containing the vector field functions, or
        dolfin.Coefficient objects approximating the vector field.
    * 'fiber_names' : str, list, tuple
        A name, or list of names, of all of the fiber direction fields.
    * 'function_space' : dolfin.FunctionSpace
        The function space used to approximate the vector fields tangent
        to the fiber directions.

    Note: all classes that are derived from this one require the 'fibers'
    subdictionary.


    """


    def __init__(self, fiber_dict, mesh):
        ElasticMaterial.__init__(self)
        ElasticMaterial.set_material_class(self, 'transversely isotropic')

        # Extract fiber file information
        s = "A value must be given for '%s' within the 'fibers' " \
            + "sub-dictionary of the material dictionary."
        try:
            fiber_files = fiber_dict['fiber_files']
        except KeyError as err:
            err.args += (s % 'fiber_files',)
            raise

        # Extract fiber names
        try:
            fiber_names = fiber_dict['fiber_names']
        except KeyError as err:
            err.args += (s % 'fiber_names',)
            raise

        try:
            element_type = fiber_dict['element']
        except KeyError as err:
            err.args += (s % 'element',)
            raise

        if element_type is not None:
            pd = int(element_type[-1])
            if pd == 0:
                function_space = dlf.VectorFunctionSpace(mesh, "DG", pd)
            else:
                function_space = dlf.VectorFunctionSpace(mesh, "CG", pd)
        else:
            function_space = None

        self.define_fiber_directions(fiber_files, fiber_names,
                                     function_space=function_space)


    def define_fiber_directions(self, fiber_files, fiber_names, function_space=None):
        """
        Load the fiber tangent vector fields from a given list of file names and
        add the function objects are member data under "_fiber_directions".


        Parameters
        ----------

        fiber_files : str, list, tuple, dolfin.Coefficient
            The name(s) of the file(s) containing the vector field functions, or
            dolfin.Coefficient objects approximating the vector field.
        fiber_names : str, list, tuple
            A name, or list of names, of all of the fiber direction fields.
        function_space : dolfin.FunctionSpace
            The function space used to approximate the vector fields tangent
            to the fiber directions.


        Returns
        -------

        None


        """

        self._fiber_directions = dict()
        key = 'e%i'

        if isinstance(fiber_files, str):
            if fiber_files[-3:] == '.h5':
                fbr = self.__load_fibers_hdf5(fiber_files, fiber_names, function_space)
                self._fiber_directions.update(fbr)
            else:
                self._fiber_directions[key % 1] = dlf.Function(function_space, fiber_files,
                                                               name=fiber_names)
        elif isinstance(fiber_files, dlf.Coefficient):
            fiber_files.rename(fiber_names, "Fiber direction")
            self._fiber_directions[key % 1] = fiber_files
        else:
            if len(fiber_files) != len(fiber_names):
                s = "The number of files and fiber family names must be the same."
                raise ValueError(s)

            self._fiber_directions = dict()
            for i,f in enumerate(fiber_files):
                if isinstance(f, dlf.Coefficient):
                    f.rename(fiber_names[i], "Fiber direction")
                    self._fiber_directions[key % (i+1)] = f
                    continue

                if f[-3:] == '.h5':
                    fbr = self.__load_fibers_hdf5(f, fiber_names[i],
                                                  function_space, key=key%(i+1))
                    self._fiber_directions.update(fbr)
                else:
                    self._fiber_directions[key % (i+1)] = dlf.Function(function_space, f,
                                                                       name=fiber_names[i])


    @staticmethod
    def __load_fibers_hdf5(fiber_file, fiber_names, function_space, key='e1'):
        f = dlf.HDF5File(dlf.mpi_comm_world(), fiber_file, 'r')
        fiber_directions = dict()
        if isinstance(fiber_names, str):
            n = dlf.Function(function_space)
            f.read(n, fiber_names)
            fiber_directions[key] = n
        else:
            key = 'e%i'
            for i,name in enumerate(fiber_names):
                n = dlf.Function(function_space)
                f.read(n, name)
                fiber_directions[key % (i+1)] = n
        return fiber_directions


class FungMaterial(FiberMaterial):
    """
    This class defines the stress tensor for Fung type materials which are
    based on the strain energy function given by

    W = C*exp(Q),

    where

    Q = d1*E11^2 + d2*E22^2 + d3*E33^2
        + 2*(d4*E11*E22 + d5*E22*E33 + d6*E11*E33)
        + d7*E12^2 + d8*E23^2 + d9*E13^2,

    and C and di, i = 1,...,9 are material constants.

    The Eij components are the components of the Lagrangian strain tensor,
    E = 0.5*(F^T*F - I), with respect to the orthonormal set {e1, e2, e3},
    where e1 and e2 are two fiber directions, and e3 = e1 x e2.

    The resulting first Piola-Kirchhoff stress tensor is then

    P = P_vol + P_iso

    with

    P_iso = J^{-2/n}*F*S_ - (1/n)*J^{-2/n}*tr(C*S_)*F^{-T},

    and

    S_ = C*exp(Q)*((d1*E11 + d4*E22 + d6*E33)*outer(e1, e1)
                 + (d4*E11 + d2*E22 + d5*E33)*outer(e2, e2)
                 + (d6*E11 + d5*E22 + d3*E33)*outer(e3, e3)
                 + d7*E12*(outer(e1, e2) + outer(e2, e1))
                 + d9*E13*(outer(e1, e3) + outer(e3, e1))
                 + d8*E23*(outer(e2, e3) + outer(e3, e2))).

    For compressible materials,

    P_vol = 2*J*kappa*(J - 1/J)*F^{-T},

    and

    P_vol = -J*p*F^{-T}

    for incompressible, where kappa is the bulk modulus, and p is the pressure.

    The inverse elastostatics formulation for this material is supported. The
    constitutive equation remains the same, but the Lagrangian strain is
    defined in terms of the f = F^{-1}, i.e.

    E = 0.5*((f*f^T)^{-1} - I).

    Furthermore, F = f^{-1} is substituted, and the stress tensor returned
    is the Cauchy stress tensor given by T = j*P*f^{-T}, where j = det(f).

    In addition to the values listed in the documentation for BaseMechanicsProblem
    for the 'material' subdictionary of 'config', the user must provide the
    following values:

    * 'fibers': dict
        See FiberMaterial for details.
    * 'C' : float
        Material constant that can be thought of as "stiffness" in some
        limiting cases.
    * 'd' : list, tuple
        A list containing the coefficients in the exponent, Q, defined above.
    * 'kappa' : float
        The bulk modulus of the material.


    """


    def __init__(self, mesh, inverse=False, **params):
        params_cp = dict(params)
        fiber_dict = params_cp.pop('fibers')

        self._fiber_directions = self.default_fiber_directions()
        FiberMaterial.__init__(self, fiber_dict, mesh)

        FiberMaterial.set_material_name(self, 'Fung material')
        FiberMaterial.set_inverse(self, inverse)
        FiberMaterial.set_incompressible(self, params['incompressible'])

        self._parameters = self.default_parameters()
        self._parameters.update(params_cp)

        # Check if material is isotropic to change the name
        d = list(self._parameters['d'])
        d_iso = [1.0]*3 + [0.0]*3 + [2.0]*3
        if d == d_iso:
            FiberMaterial.set_material_class(self, 'isotropic')


    @staticmethod
    def default_parameters():
        param = {'C': 2.0,
                 'd': [1.0]*3 + [0.0]*3 + [0.5]*3,
                 'mu': None,
                 'kappa': 1000.0,
                 'la': None,
                 'inv_la': None,
                 'E': None,
                 'nu': None}
        return param


    @staticmethod
    def default_fiber_directions():
        fibers = {'e1': None,
                  'e2': None,
                  'e3': None}
        return fibers


    def stress_tensor(self, F, J, p=None, formulation=None):
        """
        UFL form of the stress tensor.

        """

        if self._inverse:
            stress = self._inverse_stress_tensor(F, J, p, formulation)
        else:
            stress = self._forward_stress_tensor(F, J, p, formulation)

        return stress


    def _forward_stress_tensor(self, F, J, p=None, formulation=None):
        """


        """

        CC = self._parameters['C']
        dd = self._parameters['d']
        kappa = self._parameters['kappa']
        dim = ufl.domain.find_geometric_dimension(F)
        I = dlf.Identity(dim)
        C = F.T*F
        Finv = dlf.inv(F)
        Jm2d = pow(J, -float(2)/dim)
        E = dlf.Constant(0.5)*(Jm2d*C - I)

        e1 = self._fiber_directions['e1']
        e2 = self._fiber_directions['e2']
        if (e1 is None) or (e2 is None):
            if dim == 2:
                e1 = dlf.Constant([1.0, 0.0])
                e2 = dlf.Constant([0.0, 1.0])
                e3 = dlf.Constant([0.0, 0.0])
            elif dim == 3:
                e1 = dlf.Constant([1.0, 0.0, 0.0])
                e2 = dlf.Constant([0.0, 1.0, 0.0])
                e3 = dlf.Constant([0.0, 0.0, 1.0])
        else:
            if dim == 2:
                e3 = dlf.Constant([0.0, 0.0])
            elif dim == 3:
                e3 = dlf.cross(e1, e2)

        E11,E12,E13 = dlf.inner(e1, E*e1), dlf.inner(e1, E*e2), dlf.inner(e1, E*e3)
        E22,E23 = dlf.inner(e2, E*e2), dlf.inner(e2, E*e3)
        E33 = dlf.inner(e3, E*e3)

        Q = dd[0]*E11**2 + dd[1]*E22**2 + dd[2]*E33**2 \
            + 2.0*dd[3]*E11*E22 + 2.0*dd[4]*E22*E33 + 2.0*dd[5]*E11*E33 \
            + dd[6]*E12**2 + dd[7]*E23**2 + dd[8]*E13**2
        S_ = CC*dlf.exp(Q) \
             *((dd[0]*E11 + dd[3]*E22 + dd[5]*E33)*dlf.outer(e1, e1) \
               + (dd[3]*E11 + dd[1]*E22 + dd[4]*E33)*dlf.outer(e2, e2) \
               + (dd[5]*E11 + dd[4]*E22 + dd[2]*E33)*dlf.outer(e3, e3) \
               + dd[6]*E12*(dlf.outer(e1, e2) + dlf.outer(e2, e1)) \
               + dd[8]*E13*(dlf.outer(e1, e3) + dlf.outer(e3, e1)) \
               + dd[7]*E23*(dlf.outer(e2, e3) + dlf.outer(e3, e2)))
        FS_isc = Jm2d*F*S_ - 1./dim*Jm2d*dlf.tr(C*S_)*Finv.T

        # incompressibility
        if self._incompressible:
            FS_vol = -J*p*Finv.T
        else:
            FS_vol = J*2.*kappa*(J-1./J)*Finv.T

        return FS_vol + FS_isc


    def _inverse_stress_tensor(self, f, j, p=None, formulation=None):
        """


        """

        CC = self._parameters['C']
        dd = self._parameters['d']
        kappa = self._parameters['kappa']
        dim = ufl.domain.find_geometric_dimension(f)
        I = dlf.Identity(dim)
        finv = dlf.inv(f)
        b = f*f.T
        binv = dlf.inv(b)
        jm2d = pow(j, 2.0/dim)
        E = dlf.Constant(0.5)*(jm2d*binv - I)

        e1 = self._fiber_directions['e1']
        e2 = self._fiber_directions['e2']
        if (e1 is None) or (e2 is None):
            if dim == 2:
                e1 = dlf.Constant([1.0, 0.0])
                e2 = dlf.Constant([0.0, 1.0])
                e3 = dlf.Constant([0.0, 0.0])
            elif dim == 3:
                e1 = dlf.Constant([1.0, 0.0, 0.0])
                e2 = dlf.Constant([0.0, 1.0, 0.0])
                e3 = dlf.Constant([0.0, 0.0, 1.0])
        else:
            if dim == 2:
                e3 = dlf.Constant([0.0, 0.0])
            elif dim == 3:
                e3 = dlf.cross(e1, e2)

        E11,E12,E13 = dlf.inner(e1, E*e1), dlf.inner(e1, E*e2), dlf.inner(e1, E*e3)
        E22,E23 = dlf.inner(e2, E*e2), dlf.inner(e2, E*e3)
        E33 = dlf.inner(e3, E*e3)

        Q = dd[0]*E11**2 + dd[1]*E22**2 + dd[2]*E33**2 \
            + 2.0*dd[3]*E11*E22 + 2.0*dd[4]*E22*E33 + 2.0*dd[5]*E11*E33 \
            + dd[6]*E12**2 + dd[7]*E23**2 + dd[8]*E13**2
        S_ = CC*dlf.exp(Q) \
             *((dd[0]*E11 + dd[3]*E22 + dd[5]*E33)*dlf.outer(e1, e1) \
               + (dd[3]*E11 + dd[1]*E22 + dd[4]*E33)*dlf.outer(e2, e2) \
               + (dd[5]*E11 + dd[4]*E22 + dd[2]*E33)*dlf.outer(e3, e3) \
               + dd[6]*E12*(dlf.outer(e1, e2) + dlf.outer(e2, e1)) \
               + dd[8]*E13*(dlf.outer(e1, e3) + dlf.outer(e3, e1)) \
               + dd[7]*E23*(dlf.outer(e2, e3) + dlf.outer(e3, e2)))
        T_iso = j**(-5.0/dim)*finv*S_*finv.T \
                - (1.0/dim)*dlf.inner(S_, binv)*I

        # Incompressibility
        if self._incompressible:
            T_vol = -p*I
        else:
            T_vol = 2.0*kappa*(1.0/j - j)*I

        return T_vol + T_iso


class GuccioneMaterial(FungMaterial):
    """
    This class defines the stress tensor for Guccione type materials, which are
    a subclass of Fung-type materials. The constitutive equation is the same
    as that found in the FungMaterial documentation, but with

    Q = bf*E11^2 + bt*(E22^2 + E33^2 + 2*E23^2)
        + 2*bfs*(E12^2 + E13^2),

    where bf, bt, and bfs are the material parameters. The relation between
    these material constants and the di, i = 1,...,9 can be obtained in a
    straight-forward manner.

    Note that the inverse elastostatics formulation is also supported since
    this class is derived from FungMaterial.

    In addition to the values listed in the documentation for BaseMechanicsProblem
    for the 'material' subdictionary of 'config', the user must provide the
    following values:

    * 'fibers' : dict
        See FiberMaterial for details.
    * 'C' : float
        Material constant that can be thought of as "stiffness" in some
        limiting cases.
    * 'bf' : float
        Material "stiffness" in the fiber direction.
    * 'bt' : float
        Material "stiffness" in transverse directions.
    * 'bfs' : float
        Material rigidity under shear.


    """


    def __init__(self, mesh, inverse=False, **params):
        bt = params['bt']
        bf = params['bf']
        bfs = params['bfs']
        params['d'] = [bf, bt, bt,
                       0.0, 0.0, 0.0,
                       2.0*bfs, 2.0*bt, 2.0*bfs]
        FungMaterial.__init__(self, mesh, inverse=inverse, **params)
        ElasticMaterial.set_material_name(self, 'Guccione material')

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
        if self._material_class == 'isotropic':
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


def convert_elastic_moduli(param, tol=1e8):
        # original parameters
        nu = param['nu']         # Poisson's ratio [-]
        E = param['E']           # Young's modulus [kPa]
        kappa = param['kappa']   # bulk modulus [kPa]
        mu = param['mu']         # shear modulus (Lame's second parameter) [kPa]
        la = param['la']         # Lame's first parameter [kPa]
        inv_la = param['inv_la'] # Inverse of Lame's first parameter [kPa]

        inf = float('inf')
        if (kappa > 0) and (mu > 0):
            E = 9.*kappa*mu / (3.*kappa + mu)
            nu = (3.*kappa - 2.*mu) / (2.*(3.*kappa+mu))
            la = kappa - 2.*mu/3.
            inv_la = 1.0/la
        elif (la == inf or inv_la == 0.0) and (mu > 0):
            kappa = inf
            E = 3.0*mu
            nu = 0.5
            if la == inf:
                inv_la = 0.0
            else:
                la = inf
        elif (la > 0) and (mu > 0):
            E = mu*(3.*la + 2.*mu) / (la + mu)
            kappa = la + 2.*mu / 3.
            nu = la / (2.*(la + mu))
            inv_la = 1.0/la
        elif (inv_la > 0) and (mu > 0):
            E = mu*(3.0 + 2.0*mu*inv_la)/(1.0 + mu/inv_la)
            kappa = 1.0/inv_la + 2.0*mu/3.0
            nu = 1.0/(2.0*(1.0 + mu*inv_la))
            la = 1.0/inv_la
        elif (0 < nu < 0.5) and (E > 0):
            kappa = E / (3.*(1 - 2.*nu))
            mu = E / (2.*(1. + nu))
            la = E*nu / ((1. + nu)*(1. - 2.*nu))
            inv_la = 1.0/la
        elif (nu == 0.5) and (E > 0):
            kappa = inf
            mu = E/3.0
            la = inf
            inv_la = 0.0
        else:
            raise ValueError('Two material parameters must be specified.')

        s = 'Parameter %s was changed due to contradictory settings.'
        if (param['E'] is not None) and (param['E'] != E):
            print(s % 'E')
        if (param['kappa'] is not None) and (param['kappa'] != kappa):
            print(s % 'kappa')
        if (param['la'] is not None) and (param['la'] != la):
            print(s % 'la')
        if (param['inv_la'] is not None) and (param['inv_la'] != inv_la):
            print(s % 'inv_la')
        if (param['mu'] is not None) and (param['mu'] != mu):
            print(s % 'mu')
        if (param['nu'] is not None) and (param['nu'] != nu):
            print(s % 'nu')

        param['nu'] = dlf.Constant(nu)         # Poisson's ratio [-]
        param['E'] = dlf.Constant(E)           # Young's modulus [kPa]
        param['kappa'] = dlf.Constant(kappa)   # bulk modulus [kPa]
        param['mu'] = dlf.Constant(mu)         # shear modulus (Lame's second parameter) [kPa]
        param['la'] = dlf.Constant(la)         # Lame's first parameter [kPa]
        param['inv_la'] = dlf.Constant(inv_la) # Inverse of Lame's first parameters [kPa]


__fiber_directions_code__ = """

class FiberDirections : public Expression
{
public:

  // Create expression with 3 components
  FiberDirections() : Expression(3) {}

  // Function for evaluating expression on each cell
  void eval(Array<double>& values, const Array<double>& x, const ufc::cell& cell) const
  {
    const uint D = cell.topological_dimension;
    const uint cell_index = cell.index;
    values[0] = (*f1)[cell_index];
    values[1] = (*f2)[cell_index];
    values[2] = (*f3)[cell_index];
  }

  // The data stored in mesh functions
  std::shared_ptr<MeshFunction<double> > f1;
  std::shared_ptr<MeshFunction<double> > f2;
  std::shared_ptr<MeshFunction<double> > f3;

};
"""


def load_fibers(fname, fiber_names, mesh):

    fiber_mesh_functions = list()
    f = dlf.HDF5File(dlf.mpi_comm_world(), fname, 'r')

    for name in fiber_names:
        fib = dlf.MeshFunction('double', mesh)
        f.read(fib, name)
        fiber_mesh_functions.append(fib)
    f.close()

    return fiber_mesh_functions


def define_fiber_dir(fname, fiber_names, mesh, degree=0):

    fibers = load_fibers(fname, fiber_names, mesh)
    c = dlf.Expression(cppcode=__fiber_directions_code__,
                       degree=degree)
    c.f1, c.f2, c.f3 = fibers

    return dlf.as_vector((c[0], c[1], c[2]))
