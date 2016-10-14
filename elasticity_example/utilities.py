import dolfin   as df
import ufl
import warnings

def sym_product(A, B):
    D=0.5*(df.as_tensor(A[df.i,df.k]*B[df.j,df.l],(df.i,df.j,df.k,df.l))+\
           df.as_tensor(A[df.i,df.l]*B[df.j,df.k],(df.i,df.j,df.k,df.l)))
    return D

def contraction(A, D):
    B=1.0*df.as_tensor(A[df.k,df.l]*D[df.k,df.l,df.i,df.j],(df.i,df.j))
    return B

def forward_lin_elastic(u, mu, inv_la, incompressible):

    I = df.Identity(ufl.domain.find_geometric_dimension(u))
    epsilon = df.sym(df.grad(u))

    if incompressible:
        return 2*mu*epsilon
    else:
        if inv_la.values() < 1e-8:
            warnings.warn("Value of lambda is very small: better use block\
                           formulation for nearly incompressible materials")
        la = 1./inv_la

        return la*df.tr(epsilon)*I + 2*mu*epsilon

def inverse_lin_elastic(u, mu, inv_la, incompressible):

    I = df.Identity(ufl.domain.find_geometric_dimension(u))
    f = I + df.grad(u)
    finv = df.inv(f)
    epsilon = df.sym(finv) - I  # is forward lin elast with F replaced by df.inv(F)

    if incompressible:
        sigma = 2.*mu*epsilon
    else:
        if inv_la.values() < 1e-8:
            warnings.warn("Value of lambda is very small: better use block\
                           formulation for nearly incompressible materials")
        la = 1./inv_la

        sigma = la*df.tr(epsilon)*I + 2.*mu*epsilon

    return sigma


def forward_neo_hookean(u, mu, inv_la, incompressible):

    # Kinematics
    dim = ufl.domain.find_geometric_dimension(u)
    I    = df.Identity(dim)
    F    = I + df.grad(u)
    Finv = df.inv(F)
    Cinv = df.inv(F.T*F)
    J    = df.det(F)
    J23  = J**(-2.0/dim)
    I1   = df.tr(F.T*F)
    if incompressible:
        FS = mu*J23*F - 1./dim*J23*mu*I1*Finv.T
    else:
        kap  = 1./inv_la
        FS   = kap*df.ln(J)*Finv.T + mu*J23*F - 1./dim*J23*mu*I1*Finv.T

    return FS

def inverse_neo_hookean(u, mu, inv_la, incompressible):

    # Kinematics
    dim    = ufl.domain.find_geometric_dimension(u)
    I      = df.Identity(dim)
    F      = I + df.grad(u)
    J      = df.det(F)
    Fbar   = J**(-1.0/dim)*F
    f      = df.inv(F)
    fbar   = df.inv(Fbar)
    j      = df.det(f)
    sigBar = fbar*(mu/2.)*fbar.T

    if incompressible:
        sigma = 2.*j*(sigBar - 1./dim*df.tr(sigBar)*I)
    else:
        kap = 1./inv_la
        sigma  = kap*df.ln(j)*I + 2.*j*(sigBar - 1./dim*df.tr(sigBar)*I)

    return sigma

def forward_aniso(u, mu, inv_la, incompressible):
    dim     = ufl.domain.find_geometric_dimension(u)
    I       = df.Identity(dim)
    F       = I + df.grad(u)
    C       = F.T*F
    Finv    = df.inv(F)
    J       = df.det(F)
    Fbar    = J**(-1.0/dim)*F
    J23     = J**(-2.0/dim)
    Cbar    = J23*C
    length  = df.sqrt(0.5*0.5*dim)
    value   = 0.5/length
    f_0     = df.Constant((value,)*(dim))
    f_t     = Fbar*f_0
    I4bar   = df.inner(f_t,Cbar*f_t)
    af      = df.Constant(433.345)
    bf      = df.Constant(9.242)
    exp_fun = df.exp(bf*(I4bar-1.)*(I4bar-1.))
    Sbar    = 2.*J23*(mu*I+af*(I4bar-1.)*exp_fun*df.outer(f_0,f_0))
    if incompressible:
        FS    = F*Sbar - 1./dim*df.inner(C,Sbar)*Finv.T
    else:
        kap   = 1./inv_la
        FS    = kap*df.ln(J)*Finv.T +F*Sbar - 1./dim*df.inner(C,Sbar)*Finv.T
    return FS

def inverse_aniso(u, mu, inv_la, incompressible):
    dim     = ufl.domain.find_geometric_dimension(u)
    I       = df.Identity(dim)
    F       = I + df.grad(u)
    C       = F.T*F
    Finv    = df.inv(F)
    J       = df.det(F)
    Fbar    = J**(-1.0/dim)*F
    J23     = J**(-2.0/dim)
    Cbar    = J23*C
    length  = df.sqrt(0.5*0.5*dim)
    value   = 0.5/length
    f_0     = df.Constant((value,)*(dim))
    f_t     = Fbar*f_0
    I4bar   = df.inner(f_t,Cbar*f_t)
    af      = df.Constant(33.345)
    bf      = df.Constant(9.242)
    exp_fun = df.exp(bf*(i4bar-1.)*(i4bar-1.))
    sigbar  = (mu/2.*I+af*(i4bar-1.)*exp_fun*df.outer(f_0,f_0))
    if incompressible:
        sigma = 2.*j*fbar*(sigBar - 1./dim*df.tr(sigBar)*I)*fbar.T
    else:
        kap   = 1./inv_la
        sigma = kap*df.ln(j)*I + 2.*j*(sigBar - 1./dim*df.tr(sigBar)*I)
    return sigma

# compute Stress tensor depending on constitutive equation
def computeIsochoricStressTensor(u, mu, inv_la, material, inverse, incompressible):

  if inverse: #inverse formulation
    if material == 'linear':
        stress = inverse_lin_elastic(u, mu, inv_la, incompressible)
    elif material == 'neo-hooke':
        stress = inverse_neo_hookean(u, mu, inv_la, incompressible)
    elif material == 'aniso':
        stress = inverse_neo_hookean(u, mu, inv_la, incompressible)
    else:
        stress = 0
  else: #forward
    if   material == 'linear':
        stress = forward_lin_elastic(u, mu, inv_la, incompressible)
    elif material == 'neo-hooke':
        stress = forward_neo_hookean(u, mu, inv_la, incompressible)
    elif material == 'aniso':
        stress = forward_aniso(u, mu, inv_la, incompressible)
    else:
        stress = 0

  return stress

# compute volumetric Stress tensor
def computeVolumetricStressTensor(u, p, inverse):
    dim  = ufl.domain.find_geometric_dimension(u)
    I    = df.Identity(dim)
    if inverse:
        sigma_vol = p*I
        return sigma_vol
    else:
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
    Bvol = df.ln(J)*df.inv(J)
    return Bvol


