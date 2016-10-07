import dolfin as df
import ufl    as ufl

def forward_lin_elastic(u, mu, inv_la):

    I = df.Identity(ufl.domain.find_geometric_dimension(u))
    epsilon = df.sym(df.grad(u))

    if inv_la.values() < 1e-8:
        la = 0.
    else:
        la = 1./inv_la

    return la*df.tr(epsilon)*I + 2*mu*epsilon

def inverse_lin_elastic(u, mu, inv_la):

    I = df.Identity(ufl.domain.find_geometric_dimension(u))
    f = I + df.grad(u)
    finv = df.inv(f)
    epsilon = df.sym(finv) - I  # is forward lin elast with F replaced by df.inv(F)

    if inv_la.values() < 1e-8:
        la = 0.
    else:
        la = 1./inv_la

    return la*df.tr(epsilon)*I + 2.*mu*epsilon


def forward_neo_hookean(u, mu, inv_la):

    # Kinematics
    dim = ufl.domain.find_geometric_dimension(u)
    I    = df.Identity(dim)
    F    = I + df.grad(u)
    Finv = df.inv(F)
    Cinv = df.inv(F.T*F)
    J    = df.det(F)
    J23  = J**(-2.0/dim)
    kap  = 1./inv_la
    I1   = df.tr(F.T*F)
    FS   = kap*df.ln(J)*Finv.T + mu*J23*F - 1./dim*J23*mu*I1*Finv.T
    #FS   = 2*F*mu
    return FS

def incompressible_forward_neo_hookean(u, mu):

    # Kinematics
    dim = ufl.domain.find_geometric_dimension(u)
    I    = df.Identity(dim)
    F    = I + df.grad(u)
    Finv = df.inv(F)
    J    = df.det(F)
    J23  = J**(-2.0/dim)
    I1   = df.tr(F.T*F)

    # First Piola-Kirchhoff stress tensor
    FS = mu*J23*F - 1./dim*J23*mu*I1*Finv.T
    return FS

def forward_aniso(u, mu, inv_la):
    dim   = ufl.domain.find_geometric_dimension(u)
    I     = df.Identity(dim)
    F     = I + df.grad(u)
    C     = F.T*F
    Finv  = df.inv(F)
    J     = df.det(F)
    Fbar  = J**(-1.0/dim)*F
    J23   = J**(-2.0/dim)
    Cbar  = J23*C
    length= sqrt(0.5*0.5*dim)
    value = 0.5/length
    f_0   = df.Constant((value,)*(dim))
    f_t   = Fbar*f_0
    I4bar = df.inner(f_t,Cbar*f_t)
    af    = df.Constant(433.345)
    bf    = df.Constant(9.242)
    Sbar  = 2.*J23*(mu*I+af*(I4bar-1.)*exp(bf*(I4bar-1.)*(I4bar-1.))*df.outer(f_0,f_0))
    kap   = df.Constant(33334.)#1./inv_la
    FS    = kap*df.ln(J)*Finv.T +F*Sbar - 1./dim*df.inner(C,Sbar)*Finv.T
    return FS


def incompressible_forward_aniso(u, mu):
    dim   = ufl.domain.find_geometric_dimension(u)
    I     = df.Identity(dim)
    F     = I + df.grad(u)
    C     = F.T*F
    Finv  = df.inv(F)
    J     = df.det(F)
    Fbar  = J**(-1.0/dim)*F
    J23   = J**(-2.0/dim)
    Cbar  = J23*C
    length= sqrt(0.5*0.5*dim)
    value = 0.5/length
    f_0   = df.Constant((value,)*(dim))
    f_t   = Fbar*f_0
    I4bar = df.inner(f_t,Cbar*f_t)
    af    = df.Constant(433.345)
    bf    = df.Constant(9.242)
    Sbar  = 2.*J23*(mu*I+af*(I4bar-1.)*exp(bf*(I4bar-1.)*(I4bar-1.))*df.outer(f_0,f_0))
    FS    = F*Sbar - 1./dim*df.inner(C,Sbar)*Finv.T
    return FS


def volumetricStress(u, p):

    # Kinematics
    I    = df.Identity(ufl.domain.find_geometric_dimension(u))
    F    = I + df.grad(u)
    Finv = df.inv(F)
    J    = df.det(F)

    # First Piola-Kirchhoff stress tensor
    Svol = J*p*Finv.T
    return Svol

def incompressibilityCondition(u):

    # Kinematics
    I    = df.Identity(ufl.domain.find_geometric_dimension(u))
    F    = I + df.grad(u)
    Finv = df.inv(F)
    J    = df.det(F)

    # First Piola-Kirchhoff stress tensor
    Bvol = ln(J)*inv(J)
    return Bvol

def inverse_neo_hookean(u, mu, inv_la):

    # Kinematics
    dim = ufl.domain.find_geometric_dimension(u)
    I   = df.Identity(dim)
    F   = I + df.grad(u)
    J   = df.det(F)
    Fbar= J**(-1.0/dim)*F
    f   = df.inv(F)
    fbar= df.inv(Fbar)
    j   = df.det(f)
    kap = 1./inv_la

    # Cauchy stress tensor
    sigBar = fbar*(mu/2.)*fbar.T
    sigma  = kap*ln(j)*I + 2*j*(sigBar - 1./dim*df.tr(sigBar)*I)
    return sigma

def incompressible_inverse_neo_hookean(u, mu):

    # Kinematics
    dim   = ufl.domain.find_geometric_dimension(u)
    I     = df.Identity(dim)
    F     = I + df.grad(u)
    J     = df.det(F)
    Fbar  = J**(-1.0/dim)*F
    f     = df.inv(F)
    fbar  = df.inv(Fbar)
    j     = df.inv(J)               # j = J^{-1}

    # First Piola-Kirchhoff stress tensor
    sigBar = fbar*(mu/2.)*fbar.T
    sigma_isc = 2*j*(sigBar - 1./dim*df.tr(sigBar)*I)
    return sigma_isc

def incompressible_inverse_aniso(u, mu):
    dim   = ufl.domain.find_geometric_dimension(u)
    I     = df.Identity(dim)
    F     = I + df.grad(u)
    C     = F.T*F
    Finv  = df.inv(F)
    J     = df.det(F)
    Fbar  = J**(-1.0/dim)*F
    J23   = J**(-2.0/dim)
    Cbar  = J23*C
    length= sqrt(0.5*0.5*dim)
    value = 0.5/length
    f_0   = df.Constant((value,)*(dim))
    f_t   = Fbar*f_0
    I4bar = df.inner(f_t,Cbar*f_t)
    af    = df.Constant(33.345)
    bf    = df.Constant(9.242)
    sigbar  = fbar*(mu/2.*I+af*(i4bar-1.)*exp(bf*(i4bar-1.)*(i4bar-1.))*df.outer(f_0,f_0))
    return sigbar


def inverse_volumetricStress(u, p):
    dim   = ufl.domain.find_geometric_dimension(u)
    I     = df.Identity(dim)

    sigma_vol = p*I
    return sigma_vol

# compute Stress tensor depending on constitutive equation
def computeStressTensorPenalty(u, mu, inv_la, material, inverse):

  if inverse: #inverse formulation
    if material == 'linear':
        stress = inverse_lin_elastic(u, mu, inv_la)
    elif material == 'neo-hooke':
        stress = inverse_neo_hookean(u, mu, inv_la)
    elif material == 'aniso':
        stress = inverse_neo_hookean(u, mu, inv_la)
    else:
        stress = 0
  else: #forward
    if   material == 'linear':
        stress = forward_lin_elastic(u, mu, inv_la)
    elif material == 'neo-hooke':
        stress = forward_neo_hookean(u, mu, inv_la)
    elif material == 'aniso':
        stress = forward_aniso(u, mu, inv_la)
    else:
        stress = 0

  return stress
