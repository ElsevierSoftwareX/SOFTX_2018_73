import ufl
import dolfin as dlf

from ..exceptions import *

__all__ = ['Fluid', 'NewtonianFluid']


class Fluid(object):
    """
    Base class for fluids.

    """


    def __init__(self):
        self._material_name = ''
        self._parameters = {}
        self._incompressible = True
        self._material_class = 'Newtonian fluid'


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
        orthotropic, fully anisotropic, viscous fluid, etc.


        """

        return self._material_class


    def set_material_class(self, material_class):
        """
        Set material class.


        Parameters
        ----------

        material_class : str
            Name of the type of material.


        Returns
        -------

        None


        """

        self._material_class = material_class


    def is_incompressible(self):
        """
        Return True if material is incompressible and False otherwise.


        """

        return self._incompressible


    def set_incompressible(self, boolIncompressible):
        """
        Set fluid to incompressible formulation. Note that compressible fluids
        are currently not supported.


        Parameters
        ----------

        boolIncompressible : bool
            True if the material is incompressible and False otherwise.


        Returns
        -------

        None


        """

        self._incompressible = boolIncompressible


    def print_info(self):
        """
        Print material parameters, class, and name.


        """

        print('-'*80)
        print('Material: %s' % self._material_name)
        print('Parameters: %s' % self._parameters)
        print('Material class: %s' % self._material_class)
        print('Incompressible: %s' % self._incompressible)
        print('-'*80)


    def incompressibilityCondition(self, v):
        """
        Return the incompressibility function, :math:`f(\mathbf{v})`,
        for fluids such that :math:`f(\mathbf{v}) = 0`. The default is

        .. math::

           f(\mathbf{v}) = \\text{div}(\mathbf{v}) = 0,

        where :math:`\mathbf{v}` is the velocity vector field. This can
        be redefined by subclasses if a different constraint function is
        desired.


        Parameters
        ----------

        v : dolfin.Function, ufl.tensors.ListTensor
            The velocity vector.


        Returns
        -------

        f : ufl.differentiation.Div
            UFL object defining the incompressibility condition.

        """

        return dlf.div(v)


class NewtonianFluid(Fluid):
    """
    Class defining the stress tensor for an incompressible Newtonian
    fluid. Currently, only incompressible fluids are supported. The
    Cauchy stress tensor is given by

    .. math::

       \mathbf{T} = -p\mathbf{I} + 2\mu\\text{Sym}(\mathbf{L}),

    where :math:`p` is the pressure, :math:`\mathbf{I}` is the identity
    tensor, :math:`\mu` is the dynamic viscosity of the fluid, and
    :math:`\mathbf{L} = \\text{grad}(\mathbf{v})`, where :math:`\mathbf{v}`
    is the velocity vector field.

    In addition to the values listed in the documentation of :code:`fenicsmechanics`
    for the :code:`'material'` subdictionary of 'config', the user must provide at
    least one of the values listed below:


    Parameters
    ----------

    'mu' : float
        Dynamic viscosity of the fluid.
    'nu' : float
        Kinematic viscosity of the fluid.


    """


    def __init__(self, **params):
        params_cp = dict(params)
        Fluid.__init__(self)
        Fluid.set_material_class(self, 'Fluid')
        Fluid.set_material_name(self, 'Incompressible Newtonian fluid')

        # Assume True here (unlike solids) since compressible is not
        # yet supported.
        incompressible = params_cp.pop('incompressible', True)
        Fluid.set_incompressible(self, incompressible)

        self._parameters = self.default_parameters()
        for k, v in self._parameters.items():
            self._parameters[k] = params_cp.pop(k, self._parameters[k])

        # self._parameters.update(params)
        convert_viscosity(self._parameters)

        # Saving this for debugging.
        self._unused_parameters = params_cp

        if not self._incompressible:
            msg = "Compressible flows have not been implemented."
            raise NotImplementedError(msg)
        return None


    @staticmethod
    def default_parameters():
        params = {
            'nu': None,
            'mu': None, # Pa*s
            'density': None # kg/m^3
        }
        return params


    def stress_tensor(self, L, p):
        """
        Return the Cauchy stress tensor for an incompressible Newtonian fluid,
        namely

        .. math::

           \mathbf{T} = -p\mathbf{I} + 2\mu\\text{Sym}(\mathbf{L}),

        where :math:`\mathbf{L} = \\text{grad}(\mathbf{v})`, :math:`\mathbf{I}`
        is the identity tensor, :math:`p` is the hydrostatic pressure, and
        :math:`\mu` is the dynamic viscosity.


        Parameters
        ----------

        L : ufl.differentiation.Grad
            The velocity gradient.
        p : dolfin.Function, ufl.indexed.Indexed
            The hydrostatic pressure.


        Returns
        -------

        T : ufl.algebra.Sum
            The Cauchy stress tensor defined above.


        """

        params = self._parameters
        dim = ufl.domain.find_geometric_dimension(L)
        mu = dlf.Constant(params['mu'], name='mu')

        I = dlf.Identity(dim)
        D = dlf.sym(L)

        return -p*I + dlf.Constant(2.0)*mu*D


def convert_viscosity(param, material_name="newtonian"):
    """
    Ensure that the dynamic and kinematic viscosity values are
    consistent. If the density and kinematic viscosity are both
    available, the kinematic viscosity is (re)calculated. Note
    that the three material properties are related by

    .. math::

       \mu = \rho\nu,

    where :math:`\mu` is the dynamic viscosity, :math:`\rho` is
    the density, and :math:`\nu` is the kinematic viscosity.

    Note: the material parameters are recalculated in-place.


    Parameters
    ----------

    param : dict
        The material subdictionary in :code:`'config'`, i.e. the
        dictionary passed into material classes.


    """
    num_vals = 0
    for k, v in param.items():
        if v is not None:
            if not isinstance(v, (float, int, dlf.Constant)):
                msg = "*** Parameters given do not appear to be constant." \
                      + " Will not try converting parameters. ***"
                print(msg)
                return None

            param[k] = float(v)
            num_vals += 1
            if (param[k] < 0.0):
                msg = "Parameters for a '%s' material must be positive." \
                      % material_name + "The following value was given: "\
                      + "%s = %f" % (k, param[k])
                raise ValueError(msg)

    if num_vals != 2:
        msg = "Exactly 2 parameters must be given to define a "\
              + "'%s' material." % material_name \
              + " User provided %i parameters." % num_vals
        raise InconsistentCombination(msg)

    # Original parameters
    rho = param['density']
    mu = param['mu'] # Dynamic viscosity
    nu = param['nu'] # Kinematic viscosity

    if rho == 0.0:
        if (mu is None) or (mu <= 0.0):
            msg = "A non-zero dynamic viscosity must be provided when the" \
                  + " density is set to zero."
            raise InconsistentCombination(msg)
        msg = "The density was set to zero. We'll assume this was done on" \
              + " purpose and will not attempt any parameter conversions."
        print(msg)
        return None

    if (rho is not None) and (mu is not None):
        nu = mu/rho
    elif (rho is not None) and (nu is not None):
        mu = rho*nu
    elif (mu is not None) and (nu is not None):
        rho = mu/nu
    else:
        raise RequiredParameter('Two material parameters must be specified.')

    param['nu'] = dlf.Constant(nu, name='nu')
    param['mu'] = dlf.Constant(mu, name='mu')
    param['density'] = dlf.Constant(rho, name='density')

    return None
