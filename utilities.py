from dolfin import *

def forward_lin_elastic(u, mu, inv_la):

    I = Identity(u.geometric_dimension())
    epsilon = sym(grad(u))

    if inv_la.values() < 1e-8:
        la = 0.
    else:
        la = 1./inv_la

    return la*tr(epsilon)*I + 2*mu*epsilon

def inverse_lin_elastic(u, mu, inv_la):

    I = Identity(u.geometric_dimension())
    f = I + grad(u)
    finv = inv(f)
    epsilon = sym(finv) - I  # is forward lin elast with F replaced by inv(F)

    if inv_la.values() < 1e-8:
        la = 0.
    else:
        la = 1./inv_la

    return la*tr(epsilon)*I + 2.*mu*epsilon


def forward_neo_hookean(u, mu, inv_la):

    # Kinematics
    dim = u.geometric_dimension()
    I    = Identity(dim)
    F    = I + grad(u)
    Finv = inv(F)
    Cinv = inv(F.T*F)
    J    = det(F)
    J23  = J**(-2.0/dim)
    kap  = 1./inv_la
    I1   = tr(F.T*F)
    FS   = kap*ln(J)*Finv.T + mu*J23*F - 1./dim*J23*mu*I1*Finv.T
    #FS   = 2*F*mu
    return FS

def incompressible_forward_neo_hookean(u, mu):

    # Kinematics
    dim = u.geometric_dimension()
    I    = Identity(dim)
    F    = I + grad(u)
    Finv = inv(F)
    J    = det(F)
    J23  = J**(-2.0/dim)
    I1   = tr(F.T*F)

    # First Piola-Kirchhoff stress tensor
    FS = mu*J23*F - 1./dim*J23*mu*I1*Finv.T
    return FS

def forward_aniso(u, mu, inv_la):
    dim   = u.geometric_dimension()
    I     = Identity(dim)
    F     = I + grad(u)
    C     = F.T*F
    Finv  = inv(F)
    J     = det(F)
    Fbar  = J**(-1.0/dim)*F
    J23   = J**(-2.0/dim)
    Cbar  = J23*C
    length= sqrt(0.5*0.5*dim)
    value = 0.5/length
    f_0   = Constant((value,)*(dim))
    f_t   = Fbar*f_0
    I4bar = inner(f_t,Cbar*f_t)
    af    = Constant(433.345)
    bf    = Constant(9.242)
    Sbar  = 2.*J23*(mu*I+af*(I4bar-1.)*exp(bf*(I4bar-1.)*(I4bar-1.))*outer(f_0,f_0))
    kap   = Constant(33334.)#1./inv_la
    FS    = kap*ln(J)*Finv.T +F*Sbar - 1./dim*inner(C,Sbar)*Finv.T
    return FS


def incompressible_forward_aniso(u, mu):
    dim   = u.geometric_dimension()
    I     = Identity(dim)
    F     = I + grad(u)
    C     = F.T*F
    Finv  = inv(F)
    J     = det(F)
    Fbar  = J**(-1.0/dim)*F
    J23   = J**(-2.0/dim)
    Cbar  = J23*C
    length= sqrt(0.5*0.5*dim)
    value = 0.5/length
    f_0   = Constant((value,)*(dim))
    f_t   = Fbar*f_0
    I4bar = inner(f_t,Cbar*f_t)
    af    = Constant(433.345)
    bf    = Constant(9.242)
    Sbar  = 2.*J23*(mu*I+af*(I4bar-1.)*exp(bf*(I4bar-1.)*(I4bar-1.))*outer(f_0,f_0))
    FS    = F*Sbar - 1./dim*inner(C,Sbar)*Finv.T
    return FS


def volumetricStress(u, p):

    # Kinematics
    I    = Identity(u.geometric_dimension())
    F    = I + grad(u)
    Finv = inv(F)
    J    = det(F)

    # First Piola-Kirchhoff stress tensor
    Svol = J*p*Finv.T
    return Svol

def incompressibilityCondition(u):

    # Kinematics
    I    = Identity(u.geometric_dimension())
    F    = I + grad(u)
    Finv = inv(F)
    J    = det(F)

    # First Piola-Kirchhoff stress tensor
    Bvol = ln(J)*inv(J)
    return Bvol

def inverse_neo_hookean(u, mu, inv_la):

    # Kinematics
    dim = u.geometric_dimension()
    I   = Identity(dim)
    F   = I + grad(u)
    J   = det(F)
    Fbar= J**(-1.0/dim)*F
    f   = inv(F)
    fbar= inv(Fbar)
    j   = det(f)
    kap = 1./inv_la

    # Cauchy stress tensor
    sigBar = fbar*(mu/2.)*fbar.T
    sigma  = kap*ln(j)*I + 2*j*(sigBar - 1./dim*tr(sigBar)*I)
    return sigma

def incompressible_inverse_neo_hookean(u, mu):

    # Kinematics
    dim   = u.geometric_dimension()
    I     = Identity(dim)
    F     = I + grad(u)
    J     = det(F)
    Fbar  = J**(-1.0/dim)*F
    f     = inv(F)
    fbar  = inv(Fbar)
    j     = inv(J)               # j = J^{-1}

    # First Piola-Kirchhoff stress tensor
    sigBar = fbar*(mu/2.)*fbar.T
    sigma_isc = 2*j*(sigBar - 1./dim*tr(sigBar)*I)
    return sigma_isc

def incompressible_inverse_aniso(u, mu):
    dim   = u.geometric_dimension()
    I     = Identity(dim)
    F     = I + grad(u)
    C     = F.T*F
    Finv  = inv(F)
    J     = det(F)
    Fbar  = J**(-1.0/dim)*F
    J23   = J**(-2.0/dim)
    Cbar  = J23*C
    length= sqrt(0.5*0.5*dim)
    value = 0.5/length
    f_0   = Constant((value,)*(dim))
    f_t   = Fbar*f_0
    I4bar = inner(f_t,Cbar*f_t)
    af    = Constant(33.345)
    bf    = Constant(9.242)
    sigbar  = fbar*(mu/2.*I+af*(i4bar-1.)*exp(bf*(i4bar-1.)*(i4bar-1.))*outer(f_0,f_0))
    return sigbar


def inverse_volumetricStress(u, p):
    dim   = u.geometric_dimension()
    I     = Identity(dim)

    sigma_vol = p*I
    return sigma_vol
