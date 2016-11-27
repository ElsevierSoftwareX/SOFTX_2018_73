import dolfin    as df
import utilities as ut
import ufl

__all__ = [ 'NeoHookeMaterial', 'GuccioneMaterial' ]

class NeoHookeMaterial(object) :

    def __init__(self, **params) :
        params = params or {}
        self._parameters = self.default_parameters()
        self._parameters.update(params)

    @staticmethod
    def default_parameters() :
        params = { 'mu'             : 10.0,
                   'kappa'          : 1000.0,
                   'incompressible' : False,
                   'inverse'        : False }
        return params

    def is_isotropic(self) :
        """
        Return True if the material is isotropic.
        """
        return True

    def is_incompressible(self) :
        """
        Return True if the material is incompressible.
        """
        return self._parameters['kappa'] is None

    def strain_energy(self, u, p=None) :
        """
        UFL form of the strain energy.
        """
        params = self._parameters
        dim = ufl.domain.find_geometric_dimension(u)

        I     = df.Identity(dim)
        F     = I + df.grad(u)
        if params['inverse'] is True:
            F=df.inv(F)
        J     = df.det(F)
        Jm23  = pow(J, -float(2)/3)
        Cbar  = Jm23 * F.T*F
        I1bar = df.tr(Cbar)
        mu    = df.Constant(params['mu'], name='mu')
        W_isc = 0.5*mu * (I1bar - 3)

        # incompressibility
        if params['incompressible'] is True:
            W_vol = - p * (J - 1)
        else :
            kappa = df.Constant(params['kappa'], name='kappa')
            W_vol = kappa * (J**2 - 1 - 2*df.ln(J))

        return W_vol + W_isc

    def stress_tensor(self, u, p=None):
        """
        UFL form of the stress tensor.
        """
        params = self._parameters
        dim    = ufl.domain.find_geometric_dimension(u)
        I      = df.Identity(dim)
        F      = I + df.grad(u)
        Finv   = df.inv(F)
        Cinv   = df.inv(F.T*F)
        J      = df.det(F)
        Jm23   = pow(J, -float(2)/3)
        I1     = df.tr(F.T*F)
        mu     = df.Constant(params['mu'], name='mu')
        FS_isc = mu*Jm23*F - 1./dim*Jm23*mu*I1*Finv.T

        # incompressibility
        if params['incompressible'] is True:
            FS_vol = J*p*Finv.T
        else:
            kappa  = df.Constant(params['kappa'], name='kappa')
            FS_vol = J*2.*kappa*(J-1./J)*Finv.T

        return FS_vol + FS_isc

    def elasticity_tensor(self, u, p=None):
        dim    = ufl.domain.find_geometric_dimension(u)
        mu     = df.Constant(params['mu'], name='mu')
        I      = df.Identity(dim)
        F      = I + df.grad(u)
        C      = F.T*F
        Finv   = df.inv(F)
        Cinv   = df.inv(F.T*F)
        J      = df.det(F)
        Jm23   = pow(J, -float(2)/3)
        I1     = df.tr(F.T*F)
        S_isc  = mu*Jm23*I - 1./dim*Jm23*mu*I1*Cinv
        ET_isc = - 2./3.*(df.outer(Siso,Cinv)+df.outer(Cinv,Siso))\
                 + 2./3.*J23*mu*df.tr(C)*(ut.sym_product(Cinv, Cinv)\
                                      -1./3.*df.outer(Cinv,Cinv))
        # incompressibility
        if params['incompressible'] is True:
            dhyd_p  = p
        else:
            kappa   = df.Constant(params['kappa'], name='kappa')
            hyd_p   = 2.*kappa*(J-1./J)
            dhyd_p  = hyd_p + 2.*kappa*(J+1./J)
        ETM_vol = J*dhyd_p*df.outer(Cinv,Cinv) \
                      - 2.*J*hyd_p*ut.sym_product(Cinv, Cinv)




class GuccioneMaterial(object) :

    def __init__(self, **params) :
        params = params or {}
        self._parameters = self.default_parameters()
        self._parameters.update(params)

    @staticmethod
    def default_parameters() :
        p = { 'C' : 2.0,
              'bf' : 8.0,
              'bt' : 2.0,
              'bfs' : 4.0,
              'e1' : None,
              'e2' : None,
              'kappa' : None,
              'incompressible' : False,
              'Tactive' : None }
        return p

    def is_isotropic(self) :
        """
        Return True if the material is isotropic.
        """
        p = self._parameters
        return p['bt'] == 1.0 and p['bf'] == 1.0 and p['bfs'] == 1.0

    def is_incompressible(self) :
        """
        Return True if the material is incompressible.
        """
        return self._parameters['kappa'] is None

    def strain_energy(self, u, p=None) :
        """
        UFL form of the strain energy.
        """
        params = self._parameters
        dim = ufl.domain.find_geometric_dimension(u)

        I     = df.Identity(dim)
        F     = I + df.grad(u)
        J = df.det(F)
        C = pow(J, -float(2)/dim) * F.T*F
        E = 0.5*(C - I)

        CC  = df.Constant(params['C'], name='C')
        if self.is_isotropic() :
            # isotropic case
            Q = df.inner(E, E)
        else :
            # fully anisotropic
            bt  = df.Constant(params['bt'], name='bt')
            bf  = df.Constant(params['bf'], name='bf')
            bfs = df.Constant(params['bfs'], name='bfs')

            e1 = params['e1']
            e2 = params['e2']
            e3 = df.cross(e1,e2) #params['e3'] #

            E11, E12, E13 = df.inner(E*e1, e1), df.inner(E*e1, e2), df.inner(E*e1, e3)
            E21, E22, E23 = df.inner(E*e2, e1), df.inner(E*e2, e2), df.inner(E*e2, e3)
            E31, E32, E33 = df.inner(E*e3, e1), df.inner(E*e3, e2), df.inner(E*e3, e3)

            Q = bf*E11**2 + bt*(E22**2 + E33**2 + E23**2 + E32**2) \
              + bfs*(E12**2 + E21**2 + E13**2 + E31**2)

        # passive strain energy
        Wpassive = CC/2.0 * (df.exp(Q) - 1)

        # incompressibility
        if params['incompressible'] is True:
            Winc = - p * (J - 1)
        else :
            kappa = df.Constant(params['kappa'], name='kappa')
            Winc = kappa * (J**2 - 1 - 2*df.ln(J))

        return Wpassive + Winc

    def stress_tensor(self, u, p=None):
        """
        UFL form of the stress tensor.
        """
        params = self._parameters
        dim    = ufl.domain.find_geometric_dimension(u)
        I      = df.Identity(3)
        F      = I + df.grad(u)
        J      = df.det(F)
        Jm23   = pow(J, -float(2)/3)
        C      = F.T*F
        Ebar   = 0.5*(Jm23*C - I)
        Finv   = df.inv(F)

        CC  = df.Constant(params['C'], name='C')
        # fully anisotropic
        bt  = df.Constant(params['bt'], name='bt')
        bf  = df.Constant(params['bf'], name='bf')
        bfs = df.Constant(params['bfs'], name='bfs')

        e1 = params['e1']
        e2 = params['e2']
        e3 = df.cross(e1,e2)#params['e3']

        E11, E12, E13 = df.inner(Ebar*e1, e1), df.inner(Ebar*e1, e2), df.inner(Ebar*e1, e3)
        E21, E22, E23 = df.inner(Ebar*e2, e1), df.inner(Ebar*e2, e2), df.inner(Ebar*e2, e3)
        E31, E32, E33 = df.inner(Ebar*e3, e1), df.inner(Ebar*e3, e2), df.inner(Ebar*e3, e3)

        Q = bf*E11**2 + bt*(E22**2 + E33**2 + E23**2 + E32**2) \
          + bfs*(E12**2 + E21**2 + E13**2 + E31**2)
        Sbar = CC * df.exp(Q)*\
           ( bf*E11*df.outer(e1,e1)   + bt*( E22*df.outer(e2,e2) + \
                E33*df.outer(e3,e3)   +      E23*df.outer(e2,e3) + \
                E32*df.outer(e3,e2) ) + bfs*(E12*df.outer(e1,e2) + \
                E21*df.outer(e2,e1)   +      E13*df.outer(e1,e3) + \
                E31*df.outer(e3,e1) ))
        FS_isc = Jm23*F*Sbar - 1./3.*Jm23*df.tr(C*Sbar)*Finv.T

        # incompressibility
        if params['incompressible'] is True:
            FS_vol = J*p*Finv.T
        else:
            kappa  = df.Constant(params['kappa'], name='kappa')
            FS_vol = J*2.*kappa*(J-1./J)*Finv.T

        return FS_vol + FS_isc
