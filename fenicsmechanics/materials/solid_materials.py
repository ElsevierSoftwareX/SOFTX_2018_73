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

from ..exceptions import *
from ..dolfincompat import MPI_COMM_WORLD

from .fiberutils import define_fiber_direction

__all__ = ['ElasticMaterial', 'LinearIsoMaterial', 'NeoHookeMaterial',
           'DemirayMaterial', 'AnisotropicMaterial', 'FungMaterial',
           'GuccioneMaterial', 'HolzapfelOgdenMaterial']

# -----------------------------------------------------------------------------
def max_ufl(a_const, b_const):
    """
    compute maximum of a and b such that it can be handled by FEniCS
    """
    return (a_const+b_const+abs(a_const-b_const))/dlf.Constant(2)

# -----------------------------------------------------------------------------
def min_ufl(a_const, b_const):
    """
    compute minimum of a and b such that it can be handled by FEniCS
    """
    return (a_const+b_const-abs(a_const-b_const))/dlf.Constant(2)

# -----------------------------------------------------------------------------
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




# -----------------------------------------------------------------------------
class IsotropicMaterial(ElasticMaterial):
    """
    Base class for isotropic materials. This provides methods common to all
    isotropic materials.


    """


    def __init__(self):
        ElasticMaterial.__init__(self)
        ElasticMaterial.set_material_class(self, "isotropic")
        return None


# -----------------------------------------------------------------------------
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
        params_cp = dict(params)
        IsotropicMaterial.__init__(self)
        IsotropicMaterial.set_material_name(self, 'Linear material')
        IsotropicMaterial.set_inverse(self, inverse)

        incompressible = params_cp.pop('incompressible', False)
        IsotropicMaterial.set_incompressible(self, incompressible)
        self._parameters = self.default_parameters()
        for k, v in self._parameters.items():
            self._parameters[k] = params_cp.pop(k, self._parameters[k])
        convert_elastic_moduli(self._parameters,
                               material_name=self._material_name)

        # Saving this for debugging.
        self._unused_parameters = params_cp

        return None


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

# -----------------------------------------------------------------------------
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
        params_cp = dict(params)
        IsotropicMaterial.__init__(self)
        IsotropicMaterial.set_material_class(self, 'isotropic')
        IsotropicMaterial.set_material_name(self, 'Neo-Hooke material')
        IsotropicMaterial.set_inverse(self, inverse)

        incompressible = params_cp.pop('incompressible', False)
        IsotropicMaterial.set_incompressible(self, incompressible)
        self._parameters = self.default_parameters()
        for k, v in self._parameters.items():
            self._parameters[k] = params_cp.pop(k, self._parameters[k])
        convert_elastic_moduli(self._parameters,
                               material_name=self._material_name)

        # Saving this for debugging.
        self._unused_parameters = params_cp

        return None


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

        if self._incompressible:
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

        if self._incompressible:

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

        if self._incompressible:
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

        if self._incompressible:

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

# -----------------------------------------------------------------------------
class DemirayMaterial(IsotropicMaterial):
    """
    This class provides the constitutive equation presented by Demiray (1972).
    The strain energy function is given by

    .. math::
        W = W_{\text{iso}} + W_\text{vol},

    where

    .. math::

        W_{\text{iso}} = \\frac{a}{2b}(\exp(b(\bar{I}_1 - n)) - 1),

    and

    .. math::

        W_{\text{vol}} = -p(J - 1)

    for incompressible materials. While

    .. math::

        W_{\text{vol}} = \\frac{1}{2}\kappa(\ln(J))^2

    for compressible materials.

    See Demiray, H. (1972). "A note on the elasticity of soft biological
    tissues." Journal of Biomechanics, 5(3), 309-311.
    http://doi.org/10.1016/0021-9290(72)90047-4a
    """
    def __init__(self, mesh, inverse=False, **params):
        params_cp = dict(params)
        IsotropicMaterial.__init__(self)
        IsotropicMaterial.set_material_class(self, 'isotropic')
        IsotropicMaterial.set_material_name(self, 'Demiray material')
        IsotropicMaterial.set_inverse(self, inverse)

        incompressible = params_cp.pop('incompressible', False)
        IsotropicMaterial.set_incompressible(self, incompressible)
        self._parameters = self.default_parameters()
        for k, v in self._parameters.items():
            self._parameters[k] = params_cp.pop(k, self._parameters[k])

        # Saving this for debugging.
        self._unused_parameters = params_cp

        return None

    @staticmethod
    def default_parameters():
        """
        set default parameters for Demiray material
        """
        params = {'kappa': 1e3,
                 'a': 20.,
                 'b': 5.}
        return params

    def strain_energy(self, u, p=None):
        """
        UFL form of the strain energy.

        Args:
            u: deformation of the solid domain
            p: hydrostatic pressure in the solid domain
        """
        params = self._parameters
        dim = ufl.domain.find_geometric_dimension(u)

        # material parameters
        a_c = dlf.Constant(params['a'], name='a')
        b_c = dlf.Constant(params['b'], name='b')

        eye = dlf.Identity(dim)
        f__ = eye + dlf.grad(u)
        jac = dlf.det(f__)
        j_m23 = pow(jac, -float(2)/dim)
        c_bar = j_m23 * f__.T*f__
        i_1 = dlf.tr(c_bar)

        w_isc = 0.5*a_c/b_c*(dlf.exp(b_c*(i_1-dim)) - 1)

        # incompressibility
        if self._incompressible:
            w_vol = (-1.)*p * (jac - 1)
        else:
            kappa = dlf.Constant(params['kappa'], name='kappa')
            w_vol = self._volumetric_strain_energy(jac, kappa, 'log')

        return w_vol + w_isc

    def stress_tensor(self, f__, jac, p=None, formulation=None):
        """
        UFL form of the stress tensor.

        Args:
        f__ : ufl.algebra.Sum
            The deformation gradient.
        jac : ufl.tensoralgebra.Determinant
            The Jacobian, i.e. the determinant of the deformation gradient.
        p : dolfin.Function, ufl.indexed.Indexed (default, None)
            The pressure function for incompressible materials.
        formulation : str (default, None)
            This input is not used for this material. It is solely a place holder
            to conform to the format of other materials.

        """
        params = self._parameters
        dim = ufl.domain.find_geometric_dimension(f__)

        # material parameters
        a_c = dlf.Constant(params['a'], name='a')
        b_c = dlf.Constant(params['b'], name='b')

        eye = dlf.Identity(dim)
        f_inv = dlf.inv(f__)
        c__ = f__.T*f__
        j_m23 = pow(jac, -float(2)/dim)
        c_bar = j_m23 * f__.T*f__
        i_1 = dlf.tr(c_bar)

        d_i1 = 0.5*a_c*dlf.exp(b_c*(i_1 - dim))

        s_bar = 2*d_i1*eye

        fs_isc = j_m23*f__*s_bar - 1./dim*j_m23*dlf.tr(c__*s_bar)*f_inv.T

        # incompressibility
        if self._incompressible:
            fs_vol = jac*p*f_inv.T
        else:
            kappa = self._parameters['kappa']
            du_dj = self._volumetric_strain_energy_diff(jac, kappa, 'log')
            fs_vol = jac*du_dj*f_inv.T

        return fs_vol + fs_isc

# -----------------------------------------------------------------------------
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
    subdictionary. The following parameters should be included in this dictionary:


    Parameters
    ----------

    'fiber_files' : str, list, tuple, ufl.Coefficient
        The name(s) of the files containing the vector field functions, or
        ufl.Coefficient objects approximating the vector field.
    'fiber_names' : str, list, tuple
        A name, or list of names, of all of the fiber direction fields.
    'elementwise' : bool (default False)
        Set to True if the vector field is constant in each cell. Furthermore,
        setting this to True assumes that the data is stored as a set of mesh
        functions. These mesh functions are then converted to 'p0' scalar-valued
        functions, and then assigned to the components of a vector-valued
        function.
    'element': str (default None)
        Specify the finite element used to approximate the vector field that
        describes the fiber directions.


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
            fiber_dict['elementwise'] = False
        elif 'elementwise' in fiber_dict:
            # Check if fibers are given element-wise (p0) or
            # node-wise (p1).
            if fiber_dict['elementwise']:
                element_type = 'p0'
            else:
                element_type = 'p1'
        else:
            fiber_dict['element'] = element_type = None
            fiber_dict['elementwise'] = False

        # Element type should only be None if ufl.Coefficient objects
        # were already provided.
        pd = None
        if element_type is not None:
            pd = int(element_type[-1])

        self.define_fiber_directions(fiber_files, fiber_names, mesh, pd=pd,
                                     elementwise=fiber_dict['elementwise'])

        return None


    def define_fiber_directions(self, fiber_files, fiber_names, mesh,
                                pd=None, elementwise=False):
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
        mesh : dolfin.Mesh
            The computational mesh over which the fiber directions need to be
            defined. This is needed to either create corresponding mesh functions,
            or the necessary function space(s) to read the vector field or its
            components.
        pd : int (default None)
            The polynomial degree used to approximate the vector field describing
            the fiber directions. This should be kept as None if 'elementwise' is
            set to True.
        'elementwise' : bool (default False)
            Set to True if the vector field is constant in each cell. Furthermore,
            setting this to True assumes that the data is stored as a set of mesh
            functions. These mesh functions are then converted to 'p0' scalar-valued
            functions, and then assigned to the components of a vector-valued
            function.


        Returns
        -------

        None


        """

        self._fiber_directions = dict()
        key = 'e%i'
        if isinstance(fiber_files, str):
            fiber_files = [fiber_files]*len(fiber_names)

        for i, (fname, fib_name) in enumerate(zip(fiber_files, fiber_names)):
            fiber_direction = define_fiber_direction(fname, fib_name, mesh, pd=pd,
                                                     elementwise=elementwise)
            self._fiber_directions[key % (i+1)] = fiber_direction

        return None


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

        incompressible = params_cp.pop('incompressible', False)
        AnisotropicMaterial.set_incompressible(self, incompressible)

        self._parameters = self.default_parameters()
        for k, v in self._parameters.items():
            self._parameters[k] = params_cp.pop(k, self._parameters[k])

        # Saving this for debugging.
        self._unused_parameters = params_cp

        # Check if material is isotropic to change the name
        d = list(self._parameters['d'])
        d_iso = [1.0]*3 + [0.0]*3 + [2.0]*3
        if d == d_iso:
            AnisotropicMaterial.set_material_class(self, 'isotropic')
        return None


    @staticmethod
    def default_parameters():
        params = {
            'C': 2.0,
            'd': [1.0]*3 + [0.0]*3 + [0.5]*3,
            'kappa': 1e6
        }
        return params


    @staticmethod
    def default_fiber_directions():
        fibers = {
            'e1': None,
            'e2': None,
            'e3': None
        }
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

        half = dlf.Constant(0.5)
        Q = dd[0]*E11**2 + dd[1]*E22**2 + dd[2]*E33**2 \
            + 2.0*dd[3]*E11*E22 + 2.0*dd[4]*E22*E33 + 2.0*dd[5]*E11*E33 \
            + dd[6]*E12**2 + dd[7]*E23**2 + dd[8]*E13**2
        S_ = CC*dlf.exp(Q) \
             *((dd[0]*E11 + dd[3]*E22 + dd[5]*E33)*dlf.outer(e1, e1) \
               + (dd[3]*E11 + dd[1]*E22 + dd[4]*E33)*dlf.outer(e2, e2) \
               + (dd[5]*E11 + dd[4]*E22 + dd[2]*E33)*dlf.outer(e3, e3) \
               + half*dd[6]*E12*(dlf.outer(e1, e2) + dlf.outer(e2, e1)) \
               + half*dd[8]*E13*(dlf.outer(e1, e3) + dlf.outer(e3, e1)) \
               + half*dd[7]*E23*(dlf.outer(e2, e3) + dlf.outer(e3, e2)))
        FS_isc = Jm2d*F*S_ - 1./dim*Jm2d*dlf.tr(C*S_)*Finv.T

        # incompressibility
        if self._incompressible:
            FS_vol = -J*p*Finv.T
        else:
            kappa = self._parameters['kappa']
            dU_dJ = self._volumetric_strain_energy_diff(J, kappa, 'log')
            FS_vol = J*dU_dJ*Finv.T

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

        half = dlf.Constant(0.5)
        Q = dd[0]*E11**2 + dd[1]*E22**2 + dd[2]*E33**2 \
            + 2.0*dd[3]*E11*E22 + 2.0*dd[4]*E22*E33 + 2.0*dd[5]*E11*E33 \
            + dd[6]*E12**2 + dd[7]*E23**2 + dd[8]*E13**2
        S_ = CC*dlf.exp(Q) \
             *((dd[0]*E11 + dd[3]*E22 + dd[5]*E33)*dlf.outer(e1, e1) \
               + (dd[3]*E11 + dd[1]*E22 + dd[4]*E33)*dlf.outer(e2, e2) \
               + (dd[5]*E11 + dd[4]*E22 + dd[2]*E33)*dlf.outer(e3, e3) \
               + half*dd[6]*E12*(dlf.outer(e1, e2) + dlf.outer(e2, e1)) \
               + half*dd[8]*E13*(dlf.outer(e1, e3) + dlf.outer(e3, e1)) \
               + half*dd[7]*E23*(dlf.outer(e2, e3) + dlf.outer(e3, e2)))
        T_iso = j**(-5.0/dim)*finv*S_*finv.T \
                - (1.0/dim)*dlf.inner(S_, binv)*I

        # Incompressibility
        if self._incompressible:
            T_vol = -p*I
        else:
            T_vol = 2.0*kappa*(1.0/j - j)*I

        return T_vol + T_iso

# -----------------------------------------------------------------------------
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
        else:
            kappa = dlf.Constant(params['kappa'], name='kappa')
            Winc = self._volumetric_strain_energy(jac, kappa, 'log')

        return Wpassive + Winc

# -----------------------------------------------------------------------------
class HolzapfelOgdenMaterial(AnisotropicMaterial):
    """
    This class provides the constitutive equation proposed by Holzapfel and
    Ogden (2009). The strain energy function is given by

    .. math::

        W = A + \sum_{i=f,s}B_i + D_{fs},

    where

    .. math::

        A = \\frac{a}{2b}\exp(b(I_1 - 3)), \\\\
        B_i = \\frac{a_i}{2b_i}(\exp(b_i(I_{4i} - 1)^2) - 1),

    and

    .. math::

        D_{fs} = \\frac{a_{fs}}{2b_{fs}}(\exp{b_{fs}I^2_{8fs}} - 1).

    """

    def __init__(self, mesh, inverse=False, **params):
        params_cp = dict(params)
        fiber_dict = params_cp.pop('fibers')

        self._fiber_directions = self.default_fiber_directions()
        AnisotropicMaterial.__init__(self, fiber_dict, mesh)
        AnisotropicMaterial.set_material_name(self, 'Holzapfel-Ogden (2009) material')
        AnisotropicMaterial.set_inverse(self, inverse)

        incompressible = params_cp.pop('incompressible', False)
        AnisotropicMaterial.set_incompressible(self, incompressible)
        AnisotropicMaterial.set_material_class(self, 'orthotropic')

        self._parameters = self.default_parameters()
        for k, v in self._parameters.items():
            self._parameters[k] = params_cp.pop(k, self._parameters[k])

        # Saving this for debugging.
        self._unused_parameters = params_cp

        return None

    @staticmethod
    def default_parameters():
        """
        set default parameters for Holzapfel-Ogden material
        """
        params = {'kappa': 1000.0,
                  'a': 0.345,
                  'b': 9.242,
                  'af': 18.535,
                  'bf': 15.972,
                  'as': 2.564,
                  'bs': 10.446,
                  'afs': 0.417,
                  'bfs': 11.602}
        return params

    @staticmethod
    def default_fiber_directions():
        fibers = {'e1': None,
                  'e2': None,
                  'e3': None}
        return fibers

    def strain_energy(self, u, p=None):
        """
        UFL form of the strain energy.

        Args:
            u: deformation of the solid domain
            p: hydrostatic pressure in the solid domain
        """
        params = self._parameters
        dim = ufl.domain.find_geometric_dimension(u)

        # material parameters
        a_c = dlf.Constant(params['a'], name='a')
        b_c = dlf.Constant(params['b'], name='b')
        a_f = dlf.Constant(params['af'], name='af')
        b_f = dlf.Constant(params['bf'], name='bf')
        a_s = dlf.Constant(params['as'], name='as')
        b_s = dlf.Constant(params['bs'], name='bs')
        a_fs = dlf.Constant(params['afs'], name='afs')
        b_fs = dlf.Constant(params['bfs'], name='bfs')
        # fiber directions
        f_0 = self._fiber_directions['e1']
        s_0 = self._fiber_directions['e2']

        eye = dlf.Identity(dim)
        f__ = eye + dlf.grad(u)
        jac = dlf.det(f__)
        j_m23 = pow(jac, -float(2)/dim)
        c_bar = j_m23 * f__.T*f__
        i_1 = dlf.tr(c_bar)
        i_f = dlf.inner(f_0, c_bar*f_0)
        i_s = dlf.inner(s_0, c_bar*s_0)
        i_fs = dlf.inner(f_0, c_bar*s_0)
        i_fg1 = max_ufl(i_f, 1)  # st. fiber terms cancel out for If < 1
        i_sg1 = max_ufl(i_s, 1)  # st. sheet terms cancel out for Is < 1

        w_isc = 0.5*a_c/b_c*(dlf.exp(b_c*(i_1-dim)) - 1) \
                + 0.5*a_f/b_f*(dlf.exp(b_f*(i_fg1-1)**2) - 1) \
                + 0.5*a_s/b_s*(dlf.exp(b_s*(i_sg1-1)**2) - 1) \
                + 0.5*a_fs/b_fs*(dlf.exp(b_fs*i_fs**2) - 1)

        # incompressibility
        if self._parameters['incompressible']:
            w_vol = (-1.)*p * (jac - 1)
        else:
            kappa = dlf.Constant(params['kappa'], name='kappa')
            w_vol = self._volumetric_strain_energy(jac, kappa, 'log')

        return w_vol + w_isc

    def stress_tensor(self, f__, jac, p=None, formulation=None):
        """
        UFL form of the stress tensor.

        Args:
        f__ : ufl.algebra.Sum
            The deformation gradient.
        jac : ufl.tensoralgebra.Determinant
            The Jacobian, i.e. the determinant of the deformation gradient.
        p : dolfin.Function, ufl.indexed.Indexed (default, None)
            The pressure function for incompressible materials.
        formulation : str (default, None)
            This input is not used for this material. It is solely a place holder
            to conform to the format of other materials.
        """
        params = self._parameters
        dim = ufl.domain.find_geometric_dimension(f__)

        # material parameters
        a_c = dlf.Constant(params['a'], name='a')
        b_c = dlf.Constant(params['b'], name='b')
        a_f = dlf.Constant(params['af'], name='af')
        b_f = dlf.Constant(params['bf'], name='bf')
        a_s = dlf.Constant(params['as'], name='as')
        b_s = dlf.Constant(params['bs'], name='bs')
        a_fs = dlf.Constant(params['afs'], name='afs')
        b_fs = dlf.Constant(params['bfs'], name='bfs')
        # fiber directions
        f_0 = self._fiber_directions['e1']
        s_0 = self._fiber_directions['e2']

        eye = dlf.Identity(dim)
        f_inv = dlf.inv(f__)
        c__ = f__.T*f__
        j_m23 = pow(jac, -float(2)/dim)
        c_bar = j_m23 * f__.T*f__
        i_1 = dlf.tr(c_bar)
        i_f = dlf.inner(f_0, c_bar*f_0)
        i_s = dlf.inner(s_0, c_bar*s_0)
        i_fs = dlf.inner(f_0, c_bar*s_0)

        d_i1 = 0.5*a_c*dlf.exp(b_c*(i_1 - dim))
        i_fg1 = max_ufl(i_f, 1)  # st. fiber terms cancel out for If < 1
        i_sg1 = max_ufl(i_s, 1)  # st. sheet terms cancel out for Is < 1
        d_if = a_f*(i_fg1 - 1)*dlf.exp(b_f*(i_fg1 - 1)**2)
        d_is = a_s*(i_sg1 - 1)*dlf.exp(b_s*(i_sg1 - 1)**2)
        d_ifs = a_fs*(i_fs)*dlf.exp(b_fs*(i_fs)**2)

        s_bar = 2*d_i1*eye \
                + 2*d_if*dlf.outer(f_0, f_0) \
                + 2*d_is*dlf.outer(s_0, s_0) \
                + d_ifs*(dlf.outer(f_0, s_0) + dlf.outer(s_0, f_0))

        fs_isc = j_m23*f__*s_bar - 1./dim*j_m23*dlf.tr(c__*s_bar)*f_inv.T

        # incompressibility
        if self._parameters['incompressible']:
            fs_vol = jac*p*f_inv.T
        else:
            kappa = self._parameters['kappa']
            du_dj = self._volumetric_strain_energy_diff(jac, kappa, 'log')
            fs_vol = jac*du_dj*f_inv.T

        return fs_vol + fs_isc


# -----------------------------------------------------------------------------
def convert_elastic_moduli(params, material_name="lin_elastic", tol=1e-12):
    """
    Compute values of missing material coefficients for linear elastic and
    neo-Hookean models. Two, and only two, of the following parameters
    must be provided:

    * 'E': Young's modulus
    * 'nu': Poisson's ratio
    * 'la': First Lame parameter
    * 'mu': Second Lame parameter (shear modulus)
    * 'kappa': bulk modulus
    * 'inv_la': The reciprocal of the first Lame parameter

    Two of the above values must be provided within 'params', and the other 4
    must be set to None.

    Parameters
    ----------

    params : dict
        A dictionary storing the material coefficients listed above.
    tol : float (default 1e-12)
        Tolerance used to check if two values are considered to be equal.

    """
    from numpy import sqrt
    num_vals = 0
    for k,v in params.items():
        if v is not None:

            if not isinstance(v, (float, int, dlf.Constant)):
                msg = "*** Parameters given do not appear to be constant." \
                      + " Will not try converting parameters. ***"
                print(msg)
                return None

            params[k] = float(v)
            num_vals += 1

            if (params[k] <= 0.0) and (k != "inv_la"):
                msg = "Parameters for a '%s' material must be positive." \
                      % material_name + "The following value was given: "\
                      + "%s = %f" % (k, params[k])
                raise ValueError(msg)

    if num_vals != 2:
        msg = "Exactly 2 parameters must be given to define a "\
              + "'%s' material." % material_name \
              + " User provided %i parameters." % num_vals
        raise InconsistentCombination(msg)

    # original parameters
    nu = params['nu']         # Poisson's ratio [-]
    E = params['E']           # Young's modulus [kPa]
    kappa = params['kappa']   # bulk modulus [kPa]
    mu = params['mu']         # shear modulus (Lame's second parameter) [kPa]
    la = params['la']         # Lame's first parameter [kPa]
    inv_la = params['inv_la'] # Inverse of Lame's first parameter [kPa]

    inf = float('inf')
    if (mu is not None) and (kappa is not None):
        ratio = mu/kappa
        E = 9.*mu/(3. + ratio)
        la = kappa - 2.*mu/3.
        inv_la = 1.0/la
        nu = (3. - 2.*ratio)/(2.*(3. + ratio))
    elif (mu is not None) and (la is not None):
        ratio = mu/la
        kappa = la + 2.*mu/3.
        E = mu*(3. + 2.*ratio)/(1. + ratio)
        nu = 1./(2.*(1. + ratio))
        inv_la = 1.0/la
    elif (mu is not None) and (inv_la is not None):
        la = inf
        if inv_la != 0.0:
            la = 1.0/inv_la
        ratio = mu/la
        kappa = la + 2.*mu/3.
        E = mu*(3. + 2.*ratio)/(1. + ratio)
        nu = 1./(2.*(1. + ratio))
    elif (mu is not None) and (E is not None):
        denominator = 3.*mu - E
        kappa = la = inf; nu = 0.5
        if abs(denominator) > tol:
            kappa = E*mu/(3.*denominator)
            la = mu*(E - 2.*mu)/denominator
            nu = E/(2.*mu) - 1.
        inv_la = 1.0/la
    elif (mu is not None) and (nu is not None):
        denominator = 1. - 2.*nu
        kappa = la = inf
        if abs(denominator) > tol:
            kappa = 2.*mu*(1. + nu)/(3.*denominator)
            la = 2.*mu*nu/denominator
        E = 2.*mu*(1. + nu)
        inv_la = 1.0/la
    elif (E is not None) and (nu is not None):
        denominator = 1. - 2.*nu
        kappa = la = inf
        if abs(denominator) > tol:
            kappa = E/(3.*denominator)
            la = E*nu/((1. + nu)*denominator)
        mu = E/(2.*(1. + nu))
        inv_la = 1.0/la
    elif (E is not None) and (kappa is not None):
        ratio = E/kappa
        la = inf
        if kappa < inf:
            la = 3.*kappa*(3.*kappa - E)/(9.*kappa - E)
        mu = 3.*E/(9. - ratio)
        nu = (3. - ratio)/6.
        inv_la = 1.0/la
    elif (E is not None) and (la is not None):
        r = sqrt(E**2 + 9.*la**2 + 2.*E*la)
        kappa = (E + 3.*la + r)/6.
        mu = 3.*E/(9. - E/kappa)
        nu = (3. - E/kappa)/6.
        inv_la = 1.0/la
    elif (E is not None) and (inv_la is not None):
        la = inf
        if inv_la > 0.0:
            la = 1.0/inv_la
        r = sqrt(E**2 + 9.*la**2 + 2.*E*la)
        kappa = (E + 3.*la + r)/6.
        mu = 3.*E/(9. - E/kappa)
        nu = (3. - E/kappa)/6.
    elif (nu is not None) and (kappa is not None):
        if (nu == 0.5) or (kappa == inf):
            msg = "A fully incompressible material can't be well defined using " \
                  + "'nu' and 'kappa'."
            raise InvalidCombination(msg)
        E = 3.*kappa*(1. - 2.*nu)
        la = 3.*kappa*nu/(1. + nu)
        mu = 3.*kappa*(1. - 2.*nu)/(2.*(1. + nu))
        inv_la = 1.0/la
    elif (nu is not None) and (la is not None):
        if (nu == 0.5) or (kappa == inf):
            msg = "A fully incompressible material can't be well defined using " \
                  + "'nu' and 'la'."
            raise InvalidCombination(msg)
        kappa = la*(1. + nu)/(3.*nu)
        E = la*(1. + nu)*(1. - 2.*nu)/nu
        mu = la*(1. - 2.*nu)/(2.*nu)
        inv_la = 1.0/la
    elif (nu is not None) and (inv_la is not None):
        if (nu == 0.5) or (inv_la == 0.0):
            msg = "A fully incompressible material can't be well defined using " \
                  + "'nu' and 'inv_la'."
            raise InvalidCombination(msg)
        la = 1.0/inv_la
        kappa = la*(1. + nu)/(3.*nu)
        E = la*(1. + nu)*(1. - 2.*nu)/nu
        mu = la*(1. - 2.*nu)/(2.*nu)
    elif (kappa is not None) and (la is not None):
        if (kappa == inf) or (la == inf):
            msg = "A fully incompressible material can't be well defined using " \
                  + "'kappa' and 'la'."
            raise InvalidCombination(msg)
        E = 9.*kappa*(kappa - la)/(3.*kappa - la)
        mu = 3.*(kappa - la)/2.
        nu = la/(3.*kappa - la)
        inv_la = 1.0/la
    elif (kappa is not None) and (inv_la is not None):
        if (kappa == inf) or (inv_la == 0.0):
            msg = "A fully incompressible material can't be well defined using " \
                  + "'kappa' and 'inv_la'."
            raise InvalidCombination(msg)
        la = 1.0/inv_la
        E = 9.*kappa*(kappa - la)/(3.*kappa - la)
        mu = 3.*(kappa - la)/2.
        nu = la/(3.*kappa - la)
    else:
        raise RequiredParameter('Two material parameters must be specified.')

    params['nu'] = dlf.Constant(nu)         # Poisson's ratio [-]
    params['E'] = dlf.Constant(E)           # Young's modulus [kPa]
    params['kappa'] = dlf.Constant(kappa)   # bulk modulus [kPa]
    params['mu'] = dlf.Constant(mu)         # shear modulus (Lame's second parameter) [kPa]
    params['la'] = dlf.Constant(la)         # Lame's first parameter [kPa]
    params['inv_la'] = dlf.Constant(inv_la) # Inverse of Lame's first parameters [kPa]

    return None
