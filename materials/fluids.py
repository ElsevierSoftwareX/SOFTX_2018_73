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
        Return the incompressibility function, f(v), for fluids such that
        f(v) = 0. The default is

        f(v) = div(v) = 0,

        where v is the velocity vector field. This can be redefined by
        subclasses if a different constraint function is desired.


        Parameters
        ----------

        v :
            The velocity vector.


        Returns
        -------

        UFL object defining the incompressibility condition, f(v).

        """

        return dlf.div(v)


class NewtonianFluid(Fluid):
    """
    Class defining the stress tensor for an incompressible Newtonian
    fluid. Currently, only incompressible fluids are supported. The
    Cauchy stress tensor is given by

    T = -p*I + 2*mu*Sym(L),

    where p is the pressure, I is the identity tensor, mu is the
    dynamic viscosity of the fluid, and L = grad(v), where v is the
    velocity vector field.

    In addition to the values listed in the documentation of BaseMechanicsProblem
    for the 'material' 'subdictionary' of 'config', the user must provide at
    least one of the values listed below:

    * 'mu' : float
        Dynamic viscosity of the fluid.
    * 'nu' : float
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

        T = -p*I + 2*mu*Sym(L),

        where L = grad(v), I is the identity tensor, p is the hydrostatic pressure,
        and mu is the dynamic viscosity.


        Parameters
        ----------

        L :
            The velocity gradient.
        p :
            The hydrostatic pressure.


        Returns
        -------

        T :
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

    mu = rho*nu,

    where mu is the dynamic viscosity, rho is the density, and nu
    is the kinematic viscosity.


    Parameters
    ----------

    param : dict
        The material subdictionary in 'config', i.e. the dictionary
        passed into material classes.


    Returns
    -------

    None


    Note: the material parameters are recalculated in-place.


    """

    # Original parameters
    rho = param['density']
    mu = param['mu'] # Dynamic viscosity
    nu = param['nu'] # Kinematic viscosity

    if (rho > 0) and (mu > 0):
        nu = mu/rho
    elif (rho > 0) and (nu > 0):
        mu = rho*nu
    elif (mu > 0) and (nu > 0):
        rho = mu/nu
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
