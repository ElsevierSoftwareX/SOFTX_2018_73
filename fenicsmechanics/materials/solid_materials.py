"""
This module contains classes that define the stress tensor for various
solid materials. The parent class :class:`ElasticMaterial` is used for
all class definitions. Then materials are separated into isotropic and
anisotropic using :class:`IsotropicMaterial` and
:class:`AnisotropicMaterial`.

Parameters required by each constitutive equation is case-dependent.
Thus, the user should check that particular documentation.


"""
import dolfin as dlf
import ufl

from ..dolfincompat import MPI_COMM_WORLD

from ..exceptions import *

__all__ = ['ElasticMaterial', 'LinearIsoMaterial', 'NeoHookeMaterial',
           'AnisotropicMaterial', 'FungMaterial', 'GuccioneMaterial']

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

        self._active = boolActive


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

        Bvol : ufl.algebra.Product
            UFL object defining the incompressibility condition.

        """

        I    = dlf.Identity(ufl.domain.find_geometric_dimension(u))
        F    = I + dlf.grad(u)
        Finv = dlf.inv(F)
        J    = dlf.det(F)

        Bvol = dlf.ln(J)*dlf.inv(J)
        return Bvol


class IsotropicMaterial(ElasticMaterial):
    """
    Base class for isotropic materials. This provides methods common to all
    isotropic materials.


    """


    def __init__(self):
        ElasticMaterial.__init__(self)
        ElasticMaterial.set_material_class(self, "isotropic")
        return None


class LinearIsoMaterial(IsotropicMaterial):
    """
    Return the Cauchy stress tensor based on linear elasticity, i.e.
    infinitesimal deformations, both compressible and incompressible. The
    stress tensor is given by

    * Compressible: :math:`\mathbf{T} = \lambda\\text{tr}(\mathbf{e})\mathbf{I} \\
      + 2\mu\mathbf{e}`,
    * Incompressible: :math:`\mathbf{T} = -p\mathbf{I} + 2\mu\mathbf{e}`,

    where :math:`\lambda` and :math:`\mu` are the Lame material parameters,
    :math:`\mathbf{e} = \\text{sym}(\\text{grad}(\mathbf{u}))` where
    :math:`\mathbf{u}` is the displacement, and :math:`p` is the pressure in
    the case of an incompressible material.

    The inverse elastostatics formulation is also supported for this material
    model. In that case, the only change that must be accounted for is the
    fact that

    .. math::

       \mathbf{e} = \\text{sym}(\mathbf{f}^{-1}) - \mathbf{I},

    where :math:`\mathbf{f} = \mathbf{F}^{-1}` is the deformation gradient from
    the current to the reference configuration.

    At least two of the material constants from the list below must be provided
    in the :code:`'material'` subdictionary of :code:`config` in addition to the
    values already listed in the docstring of :code:`fenicsmechanics`. The remaining
    constants will be computed based on the parameters provided by the user.


    Parameters
    ----------

    'la' : float
        The first Lame parameter used as shown in the equations above. Note:
        providing la and inv_la does not qualify as providing two material
        constants.
    'mu' : float
        The second Lame parameter used as shown in the equations above.
    'kappa' : float
        The bulk modulus of the material.
    'inv_la' : float
        The reciprocal of the first Lame parameter, la. Note: providing la and
        inv_la does not qualify as providing two material constants.
    'E' : float
        The Young's modulus of the material.
    'nu' : float
        The Poisson's ratio of the material.


    """


    def __init__(self, inverse=False, **params):
        IsotropicMaterial.__init__(self)
        IsotropicMaterial.set_material_name(self, 'Linear material')
        IsotropicMaterial.set_inverse(self, inverse)
        IsotropicMaterial.set_incompressible(self, params['incompressible'])
        params = params or {}
        self._parameters = self.default_parameters()
        self._parameters.update(params)
        convert_elastic_moduli(self._parameters)


    @staticmethod
    def default_parameters():
        params = { 'mu': None,
                   'kappa': 1e6,
                   'la': None,
                   'inv_la': None,
                   'E': None,
                   'nu': None }
        return params


    def stress_tensor(self, F, J, p=None, formulation=None):
        """
        Return the Cauchy stress tensor for a linear material, namely:

        * Compressible: :math:`\mathbf{T} = \lambda\\text{tr}(\mathbf{e})\mathbf{I} \\
          + 2\mu\mathbf{e}`,
        * Incompressible: :math:`\mathbf{T} = -p\mathbf{I} + 2\mu\mathbf{e}`,

        Parameters
        ----------

        F : ufl.algebra.Sum
            The deformation gradient.
        J : ufl.tensoralgebra.Determinant
            The Jacobian, i.e. determinant of the deformation gradient. Note
            that this is not used for this material. It is solely a place holder
            to conform to the format of other materials.
        p : ufl.Coefficient (default, None)
            The UFL pressure function for incompressible materials.
        formulation : str (default, None)
            This input is not used for this material. It is solely a place holder
            to conform to the format of other materials.


        Returns
        -------

        T : ufl.algebra.Sum
            The Cauchy stress tensor given by the equation above.

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
        :math:`p = \kappa\\text{div}(\mathbf{u})`.


        Parameters
        ----------

        u : dolfin.Function, ufl.tensors.ListTensor
            The displacement vector.


        Returns
        -------

        Bvol : ufl.algebra.Product
            UFL object defining the incompressibility condition.


        """

        return dlf.div(u)


class NeoHookeMaterial(IsotropicMaterial):
    """
    Return the first Piola-Kirchhoff stress tensor based on variations of the
    strain energy function

    .. math::

       \psi(\mathbf{C}) = \\frac{1}{2}\mu(\\text{tr}(\mathbf{C}) - n),

    where :math:`\mathbf{C} = \mathbf{F}^T\mathbf{F}`, and :math:`n` is the
    geometric dimension. For nearly incompressible materials, the total strain
    energy function is given by

    .. math::

       W = U(J) + \psi(\mathbf{C}),

    where :math:`U(J)` corresponds to the strain energy in response to dilatation
    of the material with :math:`J = \det(\mathbf{F})`. The two forms of :math:`U(J)`
    supported here are

    * Square: :math:`U(J) = \\frac{1}{2}\kappa(J - 1)^2`,
    * Log: :math:`U(J) = \\frac{1}{2}\kappa(\ln(J))^2`,

    where :math:`\kappa` is the bulk modulus of the material. This results in the
    first Piola-Kirchhoff stress tensor given by

    .. math::

       \mathbf{P} = J\\frac{dU}{dJ}\mathbf{F}^{-T} + \mu\mathbf{F}.

    In the case of an incompressible material, the total strain energy is
    assumed to be of the form

    .. math::

       W = U(J) + \psi(\\bar{\mathbf{C}}),

    where :math:`\\bar{\mathbf{C}} = J^{-2/n}\mathbf{C}`. Furthermore, the
    pressure scalar field is defined such that :math:`p = -\\frac{dU}{dJ}`.
    The resulting first Piola-Kirchhoff stress tensor is then

    .. math::

       \mathbf{P} = -\left[Jp + \\frac{1}{n}\mu J^{-2/n}\\text{tr}(\mathbf{C})
             \\right]\mathbf{F}^{-T} + \mu J^{-2/n}\mathbf{F}.

    The inverse elastostatics formulation is also supported for this material
    model. In that case, the Cauchy stress tensor for compressible material is

    .. math::

       \mathbf{T} = -j^2\\frac{dU}{dj}\mathbf{I} + \mu\mathbf{c}^{-1},

    and

    .. math::

       \mathbf{T} = -\left[p + \\frac{1}{n}\mu j^{-1/n}i_2\\right]\mathbf{I}
             + \mu j^{5/n}\mathbf{c}^{-1},

    for incompressible material where :math:`j = \det(\mathbf{f}) = \\
    \det(\mathbf{F}^{-1}) = \\frac{1}{J}, \mathbf{c} = \mathbf{f}^T\mathbf{f}`,
    :math:`i_2` is the second invariant of c, and p is the pressure in the latter
    case. Note that :math:`\mathbf{f}` is the deformation gradient from the
    current configuration to the reference configuration.

    At least two of the material constants from the list below must be provided
    in the :code:`'material'` subdictionary of :code:`config` in addition to the
    values already listed in the docstring of :code:`fenicsmechanics`.

    Parameters
    ----------

    'la' : float
        The first parameter used as shown in the equations above. Note: providing
        :code:`la` and :code:`inv_la` does not qualify as providing two material
        parameters.
    'mu' : float
        The second material parameter used as shown in the equations above.
    'kappa' : float
        The bulk modulus of the material.
    'inv_la' : float
        The reciprocal of the first parameter, :code:`la`. Note: providing
        :code:`la` and :code:`inv_la` does not qualify as providing two material
        parameters.
    'E' : float
        The Young's modulus of the material. Note: this is not entirely consistent
        with the neo-Hookean formulation, but :code:`la` and :code:`mu` will be
        computed based on the relation between the Young's modulus and the Lame
        parameters if :code:`E` is given.
    'nu' : float
        The Poisson's ratio of the material. Note: this is not entirely consistent
        with the neo-Hookean formulation, but :code:`la` and :code:`mu` will be
        computed based on the relation between the Poisson's ratio and the Lame
        parameters if :code:`nu` is given.


    """


    def __init__(self, inverse=False, **params):
        IsotropicMaterial.__init__(self)
        IsotropicMaterial.set_material_class(self, 'isotropic')
        IsotropicMaterial.set_material_name(self, 'Neo-Hooke material')
        IsotropicMaterial.set_inverse(self, inverse)
        IsotropicMaterial.set_incompressible(self, params['incompressible'])
        params = params or {}
        self._parameters = self.default_parameters()
        self._parameters.update(params)
        self.__check_la(self._parameters)

        return None


    @staticmethod
    def __check_la(params):

        la = params['la']
        inv_la = params['inv_la']

        # Exit if these are already dolfin objects
        if isinstance(la, ufl.Coefficient) or isinstance(inv_la, ufl.Coefficient):
            return None

        inf = float("inf")
        if la is not None:
            if la == inf:
                inv_la = 0.0
            elif la == 0.0:
                inv_la = inf
            else:
                inv_la = 1.0/la
        elif inv_la is not None:
            if inv_la == 0.0:
                la = inf
            else:
                la = 1.0/inv_la
        else:
            convert_elastic_moduli(params)

        if (la is not None) or (inv_la is not None):
            params['la'] = dlf.Constant(la)
            params['inv_la'] = dlf.Constant(inv_la)

        return None


    @staticmethod
    def default_parameters():
        params = {'mu': None,
                  'kappa': 1e6,
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

        .. math::

           W = \\frac{1}{2}\mu(I_1 - n) + \\frac{1}{2}\lambda(\ln(J))^2
                 - \mu\ln(J) \\\\
             = \\frac{1}{2}\mu\left(\\frac{i_2}{i_3} - n\\right)
                 + \\frac{1}{2}\lambda(\ln(j))^2 - \mu\ln(j),

        where :math:`I_1` is the first invariant of :math:`\mathbf{C} = \\
        \mathbf{F}^T\mathbf{F}`, while :math:`i_2` and :math:`i_3` are the
        second and third invariants of :math:`\mathbf{c} = \mathbf{f}^T \\
        \mathbf{f}`, with :math:`\mathbf{f} = \mathbf{F}^{-1}`. For a
        (nearly-)incompressible material, the strain energy is defined by

        .. math::

           W = U(J) + \\frac{1}{2}\mu(I_1 - n) \\\\
             = U(j) + \\frac{1}{2}\mu(\\frac{i_2}{i_3} - n),

        where the invariants are now those of :math:`\\bar{\mathbf{C}} = \\
        J^{-2/n}\mathbf{C}` or :math:`\\bar{\mathbf{c}} = j^{-2/n}\mathbf{c}`,
        and :math:`\\frac{dU}{dJ} = p` for fully incompressible material, while
        :math:`U(J) = \kappa\phi(J)`. The two forms of :math:`\phi(J)` supported
        here are

        * Square: :math:`\phi(J) = \\frac{1}{2}(J - 1)^2`,
        * Log: :math:`\phi(J) = \\frac{1}{2}(\ln(J))^2`.


        Parameters
        ----------

        F : ufl.algebra.Sum
            The (forward or inverse) deformation gradient.
        J : ufl.tensoralgebra.Determinant
            The jacobian, i.e. determinant of the deformation gradient given
            above.
        formulation : str (default, None)
            The formulation used for the nearly-incompressible materials.
            Value must either be :code:`'square'` or :code:`'log'`.


        Returns
        -------

        W : ufl.algebra.Sum
            The strain energy defined above.


        """

        if self._inverse:
            W = self._inverse_strain_energy(F, J, formulation)
        else:
            W = self._forward_strain_energy(F, J, formulation)

        return W


    def _forward_strain_energy(self, F, J, formulation=None):
        """
        Define the strain energy function for the Neo-Hookean material
        based on the forward deformation gradient,
        :math:`\partial\mathbf{x}/\partial\mathbf{X}`.


        Parameters
        ----------

        F : ufl.algebra.Sum
            The deformation gradient.
        J : ufl.tensoralgebra.Determinant
            The Jacobian, i.e. determinant of the deformation gradient.
        formulation : str (default, None)
            The formulation used for the nearly-incompressible materials.
            Value must either be :code:`'square'` or :code:`'log'`.


        Returns
        -------

        W : ufl.algebra.Sum
            The strain energy of the forward problem.


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
        based on the inverse deformation gradient,
        :math:`\partial\mathbf{X}/\partial\mathbf{x}`.


        Parameters
        ----------

        f : ufl.algebra.Sum
            The deformation gradient from the current to the reference
            configuration.
        j : ufl.tensoralgebra.Determinant
            The Jacobian of the inverse deformation gradient.
        formulation : str (default, None)
            The formulation used for the strain energy due to dilatation
            of the material.


        Returns
        -------

        W : ufl.algebra.Sum
            The strain energy of the forward problem.


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

        * Compressible:

          .. math::

             \mathbf{P} = J\\frac{dU}{dJ}\mathbf{F}^{-T} + \mu\mathbf{F}

        * Incompressible:

          .. math::

             \mathbf{P} = \left[-Jp - \\frac{\mu J^{-2/n}}{n}\\text{tr}(\mathbf{C}) \\
                   \\right]\mathbf{F}^{-T} + \mu J^{-2/n}\mathbf{F},

        where :math:`\mathbf{F}` is the deformation gradient, :math:`J = \\
        \det(\mathbf{F})`, :math:`\mathbf{C} = \mathbf{F}^T\mathbf{F}`, and
        :math:`\mu` is a material constant. If the problem is an inverse
        elastostatics problem, the Cauchy stress tensor is given by

        * Compressible:

          .. math::

             \mathbf{T} = \left[-j^2\\frac{dU}{dj} - \\frac{\mu j^{-1/n}}{n}i_2 \\
                   \\right]\mathbf{I} + \mu j^{-5/n}\mathbf{c}^{-1}

        * Incompressible:

          .. math::

             \mathbf{T} = -\left[p + \\frac{\mu j^{-1/n}}{n}i_2 \\
                   \\right]\mathbf{I} + \mu j^{-5/n}\mathbf{c}^{-1}

        where :math:`\mathbf{f} = \mathbf{F}^{-1}`, :math:`j = \det(\mathbf{f})`,
        :math:`\mathbf{c} = \mathbf{f}^T\mathbf{f}`, and :math:`n` is the geometric
        dimension.


        Parameters
        ----------

        F : ufl.algebra.Sum
            The deformation gradient.
        J : ufl.tensoralgebra.Determinant
            The Jacobian, i.e. determinant of the deformation gradient.
        p : dolfin.Function, ufl.indexed.Indexed (default, None)
            The pressure for incompressible materials.
        formulation : str (default, None)
            The formulation used for the strain energy due to dilatation
            of the material.

        Returns
        -------

        P : ufl.algebra.Sum
            The first Piola-Kirchhoff stress tensor for forward problems and
            the Cauchy stress tensor for inverse elastostatics problems.


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
        compressible), defined with respect to the deformation gradient. The first
        Piola-Kirchhoff stress tensor for a compressible material is defined by

        .. math::

           \mathbf{P} = \left[\lambda\ln(J) - \mu\\right]\mathbf{F}^{-T}
                 + \mu\mathbf{F}.

        For a (nearly-)incompressible material, the first Piola-Kirchhoff stress
        tensor is given by

        .. math::

           \mathbf{P} = J\\frac{dU}{dJ}\mathbf{F}^{-T} + \mu\mathbf{F},

        where there are two formulations of :math:`U(J)` implemented. Namely,
        :math:`U(J) = \kappa\phi(J)`, where the two forms of :math:`\phi(J)`
        supported here are

        * Square: :math:`\phi(J) = \\frac{1}{2}(J - 1)^2`,
        * Log: :math:`\phi(J) = \\frac{1}{2}(\ln(J))^2`.


        Parameters
        ----------

        F : ufl.algebra.Sum
            The deformation gradient.
        J : ufl.tensoralgebra.Determinant
            The jacobian, i.e. determinant of the deformation gradient given
            above.
        p : dolfin.Function, ufl.indexed.Indexed (default, None)
            The pressure scalar field. If it is set to None, the penalty method
            formulation will be used.
        formulation : str (default, None)
            The formulation used for the nearly-incompressible materials.
            Value must either be :code:`'square'` or :code:`'log'`.


        Returns
        -------

        P : ufl.algebra.Sum
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

        (Compressible)     T = -j^2*(dU/dj)*I + mu*c^{-1}
        (Incompressible)   T = -[p + mu*j^{-1/n}/n*i2]*I + mu*j^{-5/n}*c^{-1}

        where f = F^{-1}, j = det(f), c = f^T*f, and n is the geometric dimension.


        Parameters
        ----------

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

        .. math::

           \psi(\mathbf{C}) = \\frac{1}{2}\mu\left(\\text{tr}(\mathbf{C})
                 - 3\\right)


        Parameters
        ----------

        I1 :
            Trace of the right/left Cauchy-Green strain tensor.
        mu :
            Material constant.


        Returns
        -------

        W : ufl.algebra.Sum
            UFL object defining the strain energy given above.

        """

        dim = ufl.domain.find_geometric_dimension(I1)

        return dlf.Constant(0.5)*mu*(I1 - dlf.Constant(dim))


    @staticmethod
    def _compressible_strain_energy(J, la, mu):
        """
        Define additional terms for the strain energy of a compressible
        neo-Hookean model:

        .. math::

            \hat{\psi}(\mathbf{C}) = \\frac{1}{2}\lambda(\ln(J))^2
                 - \mu\ln(J)


        Parameters
        ----------

        J : ufl.tensoralgebra.Determinant
            Determinant of the deformation gradient.
        la : float
            Material constant.
        mu : float
            Material constant.


        Returns
        -------

        UFL object defining the component of the strain energy function given
        above.

        """

        return dlf.Constant(0.5)*la*(dlf.ln(J))**2 - mu*dlf.ln(J)


    @staticmethod
    def _compressible_strain_energy_diff(J, la, mu):
        """
        The derivative of the isochoric component of the strain energy,

        .. math::

           \\frac{dU}{dJ} = \\frac{1}{J}(\lambda\ln(J) - \mu).


        """

        return (la*dlf.ln(J) - mu)/J


    @staticmethod
    def _volumetric_strain_energy(J, kappa, formulation='square'):
        """
        Define the additional penalty component for the strain energy function
        a nearly incompressible material:

        * Square: :math:`U(J) = \\frac{1}{2}\kappa(J - 1)^2`
        * Log: :math:`U(J) = \\frac{1}{2}\kappa(\ln(J))^2`


        Parameters
        ----------

        J : ufl.tensoralgebra.Determinant
            Determinant of the deformation gradient.
        kappa : float
            Bulk modulus of the material. Can also be interpreted as a penalty
            constant.
        formulation : str (default, 'square')
            String specifying which of the two above formulations to use.


        Returns
        -------

        U : ufl.algebra.Sum
            UFL object defining the penalty component of the strain energy
            function given above.


        """

        if formulation == 'square':
            f =  (J - dlf.Constant(1.0))**2
        elif formulation == 'log':
            f = (dlf.ln(J))**2
        else:
            msg = "Formulation, \"%s\" of the volumetric strain energy" % formulation \
                  + " function is not recognized."
            raise NotImplementedError(msg)

        return dlf.Constant(0.5)*kappa*f


    @staticmethod
    def _volumetric_strain_energy_diff(J, kappa, formulation='square'):
        """
        Return the derivative of the volumetric component of strain energy,

        * Square:

        .. math::

           \\frac{dU}{dJ} = \kappa\left(J - 1\\right)

        * Log:

        .. math::

           \\frac{dU}{dJ} = \\frac{\kappa\ln(J)}{J}


        Parameters
        ----------

        J : ufl.tensoralgebra.Determinant
            Determinant of the deformation gradient.
        kappa : float
            Bulk modulus.
        formulation : str (default 'square')
            Choose between square and log formulation above.


        Returns
        -------

        dUdJ : ufl.algebra.Sum
            The derivative of the volumetric component of strain energy.


        """

        if formulation == 'square':
            dfdJ = J - dlf.Constant(1.0)
        elif formulation == 'log':
            dfdJ = dlf.ln(J)/J
        else:
            msg = "Formulation, \"%s\" of the volumetric strain energy" % formulation \
                  + " function is not recognized."
            raise NotImplementedError(msg)

        return kappa*dfdJ


    @staticmethod
    def _basic_stress_tensor(F, mu):
        """
        Define the first Piola-Kirchhoff stress tensor that corresponds to a
        basic neo-Hookean strain energy function,

        .. math::

           \psi(\mathbf{C}) = \\frac{1}{2}\mu(\\text{tr}(\mathbf{C}) - n),

        namely, :math:`\mathbf{P} = \mu\mathbf{F}`.


        Parameters
        ----------

        F : ufl.algebra.Sum
            Deformation gradient.
        mu : float
            Material constant.


        Returns
        -------

        P : ufl.algebra.Sum
            UFL object defining the first Piola-Kirchhoff stress tensor.


        """

        return mu*F


    @staticmethod
    def _compressible_stress_tensor(F, J, la, mu):
        """
        Define the additional terms of the first Piola-Kirchhoff stress tensor
        resulting from the strain energy component,

        .. math::

           \hat{\psi}(\mathbf{C}) = \\frac{1}{2}\lambda(\ln(J))^2 - \mu\ln(J),

        namely, :math:`\mathbf{P} = (\lambda\ln(J) - \mu)\mathbf{F}^{-T}`.


        Parameters
        ----------

        F : ufl.algebra.Sum
            Deformation gradient.
        J : ufl.tensoralgebra.Determinant
            Determinant of the deformation gradient.
        la : float
            Material constant.
        mu : float
            Material constant.


        Returns
        -------

        P : ufl.algebra.Sum
            UFL object defining the above tensor, P.

        """

        Finv = dlf.inv(F)

        return (la*dlf.ln(J) - mu)*Finv.T


    @staticmethod
    def _penalty_stress_tensor(F, J, kappa, formulation='square'):
        """
        Define the additional terms of the first Piola-Kirchhoff stress tensor
        from the strain energy component given by one of the two formulations,

        * Square:

        .. math::

           U(J) = \\frac{1}{2}\kappa(J - 1)^2

        * Log:

        ..  math::

           U(J) = \\frac{1}{2}\kappa(\ln(J))^2

        namely,

        * Square:

        .. math::

           \mathbf{P} = \kappa J(J - 1)\mathbf{F}^{-T}

        * Log:

        .. math::

           \mathbf{P} = \kappa\ln(J)\mathbf{F}^{-T}


        Parameters
        ----------

        F : ufl.algebra.Sum
            Deformation gradient.
        J : ufl.tensoralgebra.Determinant
            Determinant of the deformation gradient.
        kappa : float
            Penalty constant.
        formulation : str (default, 'square')
            String specifying which of the two above formulations to use.


        Returns
        -------

        P : ufl.algebra.Sum
            UFL object defining the above tensor, P.


        """

        Finv = dlf.inv(F)
        if formulation == 'square':
            g = J*(J - dlf.Constant(1.0))
        elif formulation == 'log':
            g = dlf.ln(J)
        else:
            msg = "Formulation, \"%s\" of the penalty function is not recognized." \
                  % formulation
            raise NotImplementedError(msg)

        return kappa*g*Finv.T


class AnisotropicMaterial(ElasticMaterial):
    """
    Base class for fiber reinforced materials. This base class contains
    methods for loading and saving vector fields that represent the directions
    tangent to the fibers throughout the domain. Note that this class does
    not provide any constitutive equation and is merely a utility for common
    operations with fiber reinforced materials.

    In addition to the values listed in the documentation of BaseMechanicsProblem
    for the 'material' subdictionary of 'config', the user must provide a
    subdictionary within 'material' named 'fibers' with the values listed below.

    Note: all classes that are derived from this one require the :code:`'fibers'`
    subdictionary.


    Parameters
    ----------

    'fiber_files' : str, list, tuple, ufl.Coefficient
        The name(s) of the files containing the vector field functions, or
        ufl.Coefficient objects approximating the vector field.
    'fiber_names' : str, list, tuple
        A name, or list of names, of all of the fiber direction fields.
    'function_space' : dolfin.FunctionSpace
        The function space used to approximate the vector fields tangent
        to the fiber directions.


    """


    def __init__(self, fiber_dict, mesh):
        ElasticMaterial.__init__(self)
        ElasticMaterial.set_material_class(self, 'transversely isotropic')

        # Extract fiber file information
        msg = "A value must be given for '%s' within the 'fibers' " \
            + "sub-dictionary of the material dictionary."
        try:
            fiber_files = fiber_dict['fiber_files']
        except KeyError as err:
            err.args += (msg % 'fiber_files',)
            raise

        # Extract fiber names
        try:
            fiber_names = fiber_dict['fiber_names']
        except KeyError as err:
            err.args += (msg % 'fiber_names',)
            raise

        if 'element' in fiber_dict:
            element_type = fiber_dict['element']
        elif 'element-wise' in fiber_dict:
            # Check if fibers are given element-wise (p0) or
            # node-wise (p1).
            if fiber_dict['element-wise']:
                element_type = 'p0'
            else:
                element_type = 'p1'
        else:
            raise KeyError(msg % '{element-wise,element}')

        # Element type should only be None if ufl.Coefficient objects
        # were already provided.
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
        add the function objects as member data under "_fiber_directions".


        Parameters
        ----------

        fiber_files : str, list, tuple, ufl.Coefficient
            The name(s) of the file(s) containing the vector field functions, or
            ufl.Coefficient objects approximating the vector field.
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
        elif isinstance(fiber_files, ufl.Coefficient):
            fiber_files.rename(fiber_names, "Fiber direction")
            self._fiber_directions[key % 1] = fiber_files
        else:
            if len(fiber_files) != len(fiber_names):
                msg = "The number of files and fiber family names must be the same."
                raise InconsistentCombination(msg)

            self._fiber_directions = dict()
            for i,f in enumerate(fiber_files):
                if isinstance(f, ufl.Coefficient):
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
        f = dlf.HDF5File(MPI_COMM_WORLD, fiber_file, 'r')
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


class FungMaterial(AnisotropicMaterial):
    """
    This class defines the stress tensor for Fung type materials which are
    based on the strain energy function given by

    .. math::

       W = C\exp(Q),

    where

    .. math::

       Q = d_1 E_{11}^2 + d_2 E_{22}^2 + d_3 E_{33}^2
           + 2(d_4 E_{11}E_{22} + d_5 E_{22}E_{33} + d_6 E_{11}E_{33})
           + d_7 E_{12}^2 + d_8 E_{23}^2 + d_9 E_{13}^2,

    and :math:`C` and :math:`d_i, i = 1,...,9` are material constants. The
    :math:`E_{ij}` components here are the components of the Lagrangian
    strain tensor,

    .. math::

       \mathbf{E} = \\frac{1}{2}(\mathbf{F}^T\mathbf{F} - \mathbf{I}),

    with respect to the orthonormal set :math:`\{\mathbf{e}_1, \mathbf{e}_2, \\
    \mathbf{e}_3\}`, where :math:`\mathbf{e}_1` is a fiber direction,
    :math:`\mathbf{e}_2` the direction normal to the fiber, and
    :math:`\mathbf{e}_3 = \mathbf{e}_1 \\times \mathbf{e}_2`.

    The resulting first Piola-Kirchhoff stress tensor is then

    .. math::

       \mathbf{P} = \mathbf{P}_{vol} + \mathbf{P}_{iso}

    with

    .. math::

       \mathbf{P}_{iso} = J^{-2/n}\mathbf{F}\hat{\mathbf{S}}
             - \\frac{1}{n}J^{-2/n}\\text{tr}(\mathbf{C}\hat{\mathbf{S}})
             \mathbf{F}^{-T},

    and

    .. math::

       \hat{\mathbf{S}} = C\exp(Q)\left(
             (d_1 E_{11} + d_4 E_{22} + d_6 E_{33})
                \\text{outer}(\mathbf{e}_1, \mathbf{e}_1)
           + (d_4 E_{11} + d_2 E_{22} + d_5 E_{33})
                \\text{outer}(\mathbf{e}_2, \mathbf{e}_2)
           + (d_6 E_{11} + d_5 E_{22} + d_3 E_{33})
                \\text{outer}(\mathbf{e}_3, \mathbf{e}_3)
           + d_7 E_{12}(\\text{outer}(\mathbf{e}_1, \mathbf{e}_2)
                + \\text{outer}(\mathbf{e}_2, \mathbf{e}_1))
           + d_9 E_{13}(\\text{outer}(\mathbf{e}_1, \mathbf{e}_3)
                + \\text{outer}(\mathbf{e}_3, \mathbf{e}_1))
           + d_8 E_{23}(\ttext{outer}(\mathbf{e}_2, \mathbf{e}_3)
                + \\text{outer}(\mathbf{e}_3, \mathbf{e}_2))\\right).

    For compressible materials,

    .. math::

       \mathbf{P}_{vol} = 2J\kappa(J - \\frac{1}{J})\mathbf{F}^{-T},

    and

    .. math::

       \mathbf{P}_{vol} = -Jp\mathbf{F}^{-T}

    for incompressible, where :math:`\kappa` is the bulk modulus, and
    :math:`p` is the pressure.

    The inverse elastostatics formulation for this material is supported. The
    constitutive equation remains the same, but the Lagrangian strain is
    defined in terms of the :math:`\mathbf{f} = \mathbf{F}^{-1}`, i.e.

    .. math::

       \mathbf{E} = \\frac{1}{2}((\mathbf{ff}^T)^{-1} - \mathbf{I}).

    Furthermore, :math:`\mathbf{F} = \mathbf{f}^{-1}` is substituted, and the
    stress tensor returned is the Cauchy stress tensor given by :math:`\mathbf{T} \\
    = j\mathbf{Pf}^{-T}`, where :math:`j = \det(\mathbf{f})`.

    In addition to the values listed in the docstring of :code:`fenicsmechanics`
    for the 'material' subdictionary of 'config', the user must provide the
    values listed below.


    Parameters
    ----------


    'fibers': dict
        See AnisotropicMaterial for details.
    'C' : float
        Material constant that can be thought of as "stiffness" in some
        limiting cases.
    'd' : list, tuple
        A list containing the coefficients in the exponent, Q, defined above.
    'kappa' : float
        The bulk modulus of the material.


    """


    def __init__(self, mesh, inverse=False, **params):
        params_cp = dict(params)
        fiber_dict = params_cp.pop('fibers')

        self._fiber_directions = self.default_fiber_directions()
        AnisotropicMaterial.__init__(self, fiber_dict, mesh)

        AnisotropicMaterial.set_material_name(self, 'Fung material')
        AnisotropicMaterial.set_inverse(self, inverse)
        AnisotropicMaterial.set_incompressible(self, params['incompressible'])

        self._parameters = self.default_parameters()
        self._parameters.update(params_cp)

        # Check if material is isotropic to change the name
        d = list(self._parameters['d'])
        d_iso = [1.0]*3 + [0.0]*3 + [2.0]*3
        if d == d_iso:
            AnisotropicMaterial.set_material_class(self, 'isotropic')


    @staticmethod
    def default_parameters():
        param = {'C': 2.0,
                 'd': [1.0]*3 + [0.0]*3 + [0.5]*3,
                 'mu': None,
                 'kappa': 1e6,
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
        Return the first Piola-Kirchhoff stress tensor for forward problems
        and the Cauchy stress tensor for inverse elastostatics problems. The
        constitutive equation is shown in the documentation of FungMaterial.


        Parameters
        ----------

        F : ufl.algebra.Sum
            The deformation gradient.
        J : ufl.tensoralgebra.Determinant
            The Jacobian, i.e. the determinant of the deformation gradient.
        p : dolfin.Function, ufl.indexed.Indexed (default, None)
            The pressure function for incompressible materials.
        formulation : str (default, None)
            This input is not used for this material. It is solely a place holder
            to conform to the format of other materials.


        Returns
        -------

        P : ufl.algebra.Sum
            The stress tensor defined in the FungMaterial documentation.


        """

        if self._inverse:
            stress = self._inverse_stress_tensor(F, J, p, formulation)
        else:
            stress = self._forward_stress_tensor(F, J, p, formulation)

        return stress


    def _forward_stress_tensor(self, F, J, p=None, formulation=None):
        """
        Return the first Piola-Kirchhoff stress tensor for forward problems.
        The constitutive equation is shown in the documentation of FungMaterial.


        Parameters
        ----------

        F : ufl.algebra.Sum
            The deformation gradient.
        J : ufl.tensoralgebra.Determinant
            The Jacobian, i.e. the determinant of the deformation gradient.
        p : dolfin.Function, ufl.indexed.Indexed (default, None)
            The pressure function for incompressible materials.
        formulation : str (default, None)
            This input is not used for this material. It is solely a place holder
            to conform to the format of other materials.


        Returns
        -------

        P : ufl.algebra.Sum
            The stress tensor defined in the FungMaterial documentation.


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
        Return the Cauchy stress tensor for inverse elastostatics problems.
        The constitutive equation is shown in the documentation of FungMaterial.


        Parameters
        ----------

        f : ufl.algebra.Sum
            The deformation gradient from the current to the reference
            configuration.
        j : ufl.tensoralgebra.Determinant
            The determinant of f.
        p : dolfin.Function, ufl.indexed.Indexed (default, None)
            The pressure function for incompressible materials.
        formulation : str (default, None)
            This input is not used for this material. It is solely a place holder
            to conform to the format of other materials.


        Returns
        -------

        P : ufl.algebra.Sum
            The stress tensor defined in the FungMaterial documentation.


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

    .. math::

       Q = b_f E_{11}^2 + b_t (E_{22}^2 + E_{33}^2 + 2 E_{23}^2)
           + 2b_{fs}(E_{12}^2 + E_{13}^2),

    where :math:`b_f`, :math:`b_t`, and :math:`b_{fs}` are the material parameters.
    The relation between these material constants and the :math:`d_i, i = 1,...,9`
    can be obtained in a straight-forward manner.

    Note that the inverse elastostatics formulation is also supported since
    this class is derived from FungMaterial.

    In addition to the values listed in the documentation for :code:`fenicsmechanics`
    for the :code:`'material'` subdictionary of 'config', the user must provide the
    following values:


    Parameters
    ----------

    'fibers' : dict
        See AnisotropicMaterial for details.
    'C' : float
        Material constant that can be thought of as "stiffness" in some
        limiting cases.
    'bf' : float
        Material "stiffness" in the fiber direction.
    'bt' : float
        Material "stiffness" in transverse directions.
    'bfs' : float
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
        Return the strain energy of Guccione material defined in the
        documentation of GuccioneMaterial.


        Parameters
        ----------

        u : dolfin.Funcion, ufl.tensors.ListTensor
            The displacement of the material.
        p : dolfin.Function, ufl.indexed.Indexed (default, None)
            The pressure for incompressible materials.


        Returns
        -------

        W : ufl.algebra.Sum
            The strain energy defined in the documentation of GuccioneMaterial.


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
                e3 = dlf.cross(e1,e2)

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
        if (kappa is not None and kappa > 0) \
           and (mu is not None and mu > 0):
            E = 9.*kappa*mu / (3.*kappa + mu)
            nu = (3.*kappa - 2.*mu) / (2.*(3.*kappa+mu))
            la = kappa - 2.*mu/3.
            inv_la = 1.0/la
        elif (la == inf or inv_la == 0.0) \
             and (mu is not None and mu > 0):
            kappa = inf
            E = 3.0*mu
            nu = 0.5
            if la == inf:
                inv_la = 0.0
            else:
                la = inf
        elif (la is not None and la > 0) \
             and (mu is not None and mu > 0):
            E = mu*(3.*la + 2.*mu) / (la + mu)
            kappa = la + 2.*mu / 3.
            nu = la / (2.*(la + mu))
            inv_la = 1.0/la
        elif (inv_la is not None and inv_la > 0) \
             and (mu is not None and mu > 0):
            E = mu*(3.0 + 2.0*mu*inv_la)/(1.0 + mu/inv_la)
            kappa = 1.0/inv_la + 2.0*mu/3.0
            nu = 1.0/(2.0*(1.0 + mu*inv_la))
            la = 1.0/inv_la
        elif (nu is not None and 0 < nu < 0.5) \
             and (E is not None and E > 0):
            kappa = E / (3.*(1 - 2.*nu))
            mu = E / (2.*(1. + nu))
            la = E*nu / ((1. + nu)*(1. - 2.*nu))
            inv_la = 1.0/la
        elif (nu is not None and nu == 0.5) \
             and (E is not None and E > 0):
            kappa = inf
            mu = E/3.0
            la = inf
            inv_la = 0.0
        else:
            raise RequiredParameter('Two material parameters must be specified.')

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
    """
    Load the fiber vector fields from a given HDF5 file and create a
    dolfin.MeshFunction.


    Parameters
    ----------

    fname : str
        The name of the file.
    fiber_names : list, tuple, str
        The names under which each fiber vector field was stored in
        the given HDF5 file.
    mesh : dolfin.Mesh
        Mesh of the computational domain.


    Returns
    -------

    fiber_mesh_function : dolfin.MeshFunction
        Mesh function defining fiber directions object.


    """

    fiber_mesh_functions = list()
    f = dlf.HDF5File(MPI_COMM_WORLD, fname, 'r')

    for name in fiber_names:
        fib = dlf.MeshFunction('double', mesh)
        f.read(fib, name)
        fiber_mesh_functions.append(fib)
    f.close()

    return fiber_mesh_functions


def define_fiber_dir(fname, fiber_names, mesh, degree=0):
    """
    Define the fiber directions given the file in which they are stored,
    their names, and the mesh of the computational domain.


    Parameters
    ----------

    fname : str
        The name of the file.
    fiber_names : list, tuple, str
        The names under which each fiber vector field was stored in
        the given HDF5 file.
    mesh : dolfin.Mesh
        Mesh of the computational domain.
    degree : int
        The polynomial degree used to approximate the fiber vector fields.


    Returns
    -------

    c : ufl.Coefficient
        Representation of the fiber vector fields.


    """

    fibers = load_fibers(fname, fiber_names, mesh)
    c = dlf.Expression(cppcode=__fiber_directions_code__,
                       degree=degree)
    c.f1, c.f2, c.f3 = fibers

    return dlf.as_vector((c[0], c[1], c[2]))
