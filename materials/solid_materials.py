import dolfin as dlf
import ufl

__all__ = ['LinearMaterial', 'NeoHookeMaterial', 'GuccioneMaterial']

class ElasticMaterial(object):

    def __init__(self):
        self._material_name  = ''
        self._parameters     = {}
        self._incompressible = False
        self._inverse        = False
        self._active         = False
        self._material_class = 'isotropic'

    @staticmethod
    def default_parameters() :
        return {}

    def set_material_name(self, material_name):
        """
        Set material name
        """
        self._material_name = material_name

    def get_material_class(self):
        """
        Return material class, i.e. isotropic, transversely isotropic,
        orthotropic or fully anisotropic
        """
        return self._material_class

    def set_material_class(self, material_class):
        """
        Set material class, i.e. isotropic, transversely isotropic,
        orthotropic or fully anisotropic
        """
        self._material_class = material_class

    def is_incompressible(self) :
        """
        Return True if the material is incompressible.
        """
        return self._incompressible

    def set_incompressible(self, boolIncompressible):
        """
        Set material to incompressible formulation
        """
        self._incompressible = boolIncompressible

    def set_inverse(self, boolInverse):
        """
        Set material to inverse formulation
        """
        self._inverse = boolInverse

    def is_inverse(self):
        """
        Return True if the material formulation is inverse
        """
        return self._inverse

    def set_active(self, boolActive):
        """
        Set material to inverse formulation
        """
        self._inverse = boolActive

    def is_active(self) :
        """
        Return True if the material supports active contraction
        """
        return self._active

    def print_info(self) :
        """
        Print material information
        """
        print('-'*80)
        print('Material: %s' % self._material_name)
        print('Parameters: %s' % self._parameters)
        print('Material class: %s, ' % self._material_class)
        print('Properties: incompressible (%s), inverse (%s), active (%s)'
                % (self._incompressible, self._inverse, self._active))
        print('-'*80)

    def incompressibilityCondition(self, u):
        I    = dlf.Identity(ufl.domain.find_geometric_dimension(u))
        F    = I + dlf.grad(u)
        Finv = dlf.inv(F)
        J    = dlf.det(F)

        Bvol = dlf.ln(J)*dlf.inv(J)
        return Bvol

class LinearMaterial(ElasticMaterial) :
    """
    Return the stress tensor based on the linear elasticity
    """

    def __init__(self, **params) :
        ElasticMaterial.__init__(self)
        ElasticMaterial.set_material_class (self, 'isotropic')
        ElasticMaterial.set_material_name (self, 'Linear material')
        params = params or {}
        self._parameters = self.default_parameters()
        self._parameters.update(params)
        convert_elastic_moduli (self._parameters)

    @staticmethod
    def default_parameters() :
        params = { 'mu'             : None,
                   'kappa'          : None,
                   'lambda'         : None,
                   'E'              : None,
                   'nu'             : None }
        return params

    def stress_tensor(self, u, p=None):

        params = self._parameters
        dim   = ufl.domain.find_geometric_dimension(u)
        mu    = dlf.Constant(params['mu'], name='mu')
        la    = dlf.Constant(params['lambda'], name='lambda')

        I       = dlf.Identity(dim)
        F       = I + dlf.grad(u)
        if self._inverse:
            epsilon = dlf.sym(dlf.inv(F)) - I
        else:
            epsilon = dlf.sym(F) - I

        if self._incompressible:
            T = 2.0*mu*epsilon
        else:
            T = la*dlf.tr(epsilon)*I + 2.0*mu*epsilon

        return T

    def incompressibilityCondition(self, u):
        return dlf.div(u)

class NeoHookeMaterial(ElasticMaterial) :
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

    def __init__(self, **params) :
        ElasticMaterial.__init__(self)
        ElasticMaterial.set_material_class (self, 'isotropic')
        ElasticMaterial.set_material_name (self, 'Neo-Hooke material')
        params = params or {}
        self._parameters = self.default_parameters()
        self._parameters.update(params)
        convert_elastic_moduli (self._parameters)

    @staticmethod
    def default_parameters() :
        params = { 'mu'             : None,
                   'kappa'          : None,
                   'lambda'         : None,
                   'E'              : None,
                   'nu'             : None }
        return params

    def strain_energy(self, u, p=None) :
        """
        UFL form of the strain energy.
        """
        params = self._parameters
        dim = ufl.domain.find_geometric_dimension(u)
        mu    = dlf.Constant(params['mu'], name='mu')
        kappa = dlf.Constant(params['kappa'], name='kappa')
        la    = dlf.Constant(params['lambda'], name='lambda')

        I     = dlf.Identity(dim)
        F     = I + dlf.grad(u)
        if params['inverse'] is True:
            F=dlf.inv(F)
        J     = dlf.det(F)
        C     = F.T*F
        Jm2d  = pow(J, -float(2)/dim)

        # incompressibility
        if self._incompressible:
            I1bar = dlf.tr(Jm2d*C)
            W_isc = 0.5*mu * (I1bar - dim)
            if p == 0 or p == None:
                W_vol = kappa * (J**2 - 1 - 2*dlf.ln(J))
            else:
                W_vol = - p * (J - 1)
        else:
            W_isc = 0.5*mu * (dlf.tr(C) - dim)
            W_vol = (la*dlf.ln(J) - mu)

        return W_vol + W_isc

    def stress_tensor(self, u, p=None):
        """
        UFL form of the stress tensor.
        """
        #parameters
        dim    = ufl.domain.find_geometric_dimension(u)
        params = self._parameters
        mu     = dlf.Constant(params['mu'], name='mu')
        kappa  = dlf.Constant(params['kappa'], name='kappa')
        la     = dlf.Constant(params['lambda'], name='lambda')

        I      = dlf.Identity(dim)
        F      = I + dlf.grad(u)
        Finv   = dlf.inv(F)
        Cinv   = dlf.inv(F.T*F)
        J      = dlf.det(F)
        Jm2d   = pow(J, -float(2)/dim)
        I1     = dlf.tr(F.T*F)

        # incompressibility
        if self._incompressible:
            FS_isc = mu*Jm2d*F - 1./dim*Jm2d*mu*I1*Finv.T
            # nearly incompressible penalty formulation
            if p == 0 or p == None:
                FS_vol = J*2.*kappa*(J-1./J)*Finv.T
            # (nearly) incompressible block system formulation
            else:
                FS_vol = J*p*Finv.T
        # standard compressible formulation
        else:
            FS_isc = mu*F
            FS_vol = (la*dlf.ln(J) - mu)*Finv.T

        return FS_vol + FS_isc

    def elasticity_tensor(self, u, p=None):
        #parameters
        dim    = ufl.domain.find_geometric_dimension(u)
        mu     = dlf.Constant(params['mu'], name='mu')
        kappa  = dlf.Constant(params['kappa'], name='kappa')
        la     = dlf.Constant(params['lambda'], name='lambda')

        I      = dlf.Identity(dim)
        F      = I + dlf.grad(u)
        C      = F.T*F
        Finv   = dlf.inv(F)
        Cinv   = dlf.inv(F.T*F)
        J      = dlf.det(F)
        Jm2d   = pow(J, -float(2)/dim)
        I1     = dlf.tr(F.T*F)
        S_isc  = mu*Jm2d*I - 1./dim*Jm2d*mu*I1*Cinv
        ET_isc = - 2./dim*(dlf.outer(Siso,Cinv)+dlf.outer(Cinv,Siso))\
                 + 2./dim*J2d*mu*dlf.tr(C)*(ut.sym_product(Cinv, Cinv)\
                                      -1./dim*dlf.outer(Cinv,Cinv))
        # incompressibility
        if self._incompressible:
            if p == 0 or p == None:
                dhyd_p  = p
            else:
                hyd_p   = 2.*kappa*(J-1./J)
                dhyd_p  = hyd_p + 2.*kappa*(J+1./J)
        ETM_vol = J*dhyd_p*dlf.outer(Cinv,Cinv) \
                      - 2.*J*hyd_p*ut.sym_product(Cinv, Cinv)




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

def convert_elastic_moduli(param):
        # original parameters
        nu = param['nu']       # Poisson's ratio [-]
        E = param['E']         # Young's modulus [kPa]
        kappa = param['kappa'] # bulk modulus [kPa]
        mu = param['mu']       # shear modulus (Lame's second parameter) [kPa]
        lam = param['lambda']  # Lame's first parameter [kPa]

        if (kappa is not None) and (mu is not None) \
           and (kappa > 0) and (mu > 0):
            E = 9.*kappa*mu / (3.*kappa + mu)
            lam = kappa - 2.*mu/3.
            nu = (3.*kappa - 2.*mu) / (2.*(3.*kappa+mu))
        if (lam is not None) and (mu is not None) \
           and (lam > 0) and (mu > 0):
            E = mu*(3.*lam + 2.*mu) / (lam + mu)
            kappa = lam + 2.*mu / 3.
            nu = lam / (2.*(lam + mu))
        if nu > 0 and nu <= 0.5 and E > 0:
            kappa = E / (3.*(1 - 2.*nu))
            lam = E*nu / ((1. + nu)*(1. - 2.*nu))
            mu = E / (2.*(1. + nu))

        s = 'Parameter %s was changed due to contradictory settings.'
        if (param['E'] is not None) and (param['E'] != E):
            print s % 'E'
        if (param['kappa'] is not None) and (param['kappa'] != kappa):
            print s % 'kappa'
        if (param['lambda'] is not None) and (param['lambda'] != lam):
            print s % 'lambda'
        if (param['mu'] is not None) and (param['mu'] != mu):
            print s % 'mu'
        if (param['nu'] is not None) and (param['nu'] != nu):
            print s % 'nu'

        param['nu'] = dlf.Constant(nu)       # Poisson's ratio [-]
        param['E'] = dlf.Constant(E)         # Young's modulus [kPa]
        param['kappa'] = dlf.Constant(kappa) # bulk modulus [kPa]
        param['mu'] = dlf.Constant(mu)       # shear modulus (Lame's second parameter) [kPa]
        param['lambda'] = dlf.Constant(lam)  # Lame's first parameter [kPa]
