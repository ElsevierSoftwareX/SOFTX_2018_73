import ufl
import dolfin as dlf

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
        Fluid.__init__(self)
        Fluid.set_material_class(self, 'Fluid')
        Fluid.set_material_name(self, 'Incompressible Newtonian fluid')
        Fluid.set_incompressible(self, params['incompressible'])
        self._parameters = self.default_parameters()
        self._parameters.update(params)
        convert_viscosity(self._parameters)

        if not self._incompressible:
            s = "Compressible flows have not been implemented."
            raise NotImplementedError(s)


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


def convert_viscosity(param):
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

    # Original parameters
    rho = param['density']
    mu = param['mu'] # Dynamic viscosity
    nu = param['nu'] # Kinematic viscosity

    if (rho is not None) and (mu is not None):
        if (rho > 0) and (mu > 0):
            nu = mu/rho
        elif rho > 0:
            raise ValueError("A positive value must be provided for 'mu'.")
        else:
            raise ValueError("A positive value must be provided for the density.")
    elif (rho is not None) and (nu is not None):
        if (rho > 0) and (nu > 0):
            mu = rho*nu
        elif (rho > 0):
            raise ValueError("A positive value must be provided for 'nu'.")
        else:
            raise ValueError("A positive value must be provided for the density.")
    elif (mu is not None) and (nu is not None):
        if (mu > 0) and (nu > 0):
            rho = mu/nu
        elif mu > 0:
            raise ValueError("A positive value must be provided for 'nu'.")
        else:
            raise ValueError("A positive value must be provided for 'mu'.")
    else:
        raise ValueError('Two material parameters must be specified.')

    s = 'Parameter \'%s\' was changed due to contradictory settings.'
    if (param['nu'] is not None) and (param['nu'] != nu):
        print(s % 'nu')
    if (param['mu'] is not None) and (param['mu'] != mu):
        print(s % 'mu')
    if (param['density'] is not None) and (param['density'] != rho):
        print(s % 'density')

    param['nu'] = dlf.Constant(nu, name='nu')
    param['mu'] = dlf.Constant(mu, name='mu')
    param['density'] = dlf.Constant(rho, name='density')
