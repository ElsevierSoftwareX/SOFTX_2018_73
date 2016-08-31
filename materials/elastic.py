import dolfin as dlf

from ufl.domain import find_geometric_dimension as find_dim


# This function might not be necessary!!!!!
def elasticMaterial(problem, name='lin-elastic', strain=False,
            incompressible=False, inverse=False):
    """
    Return the stress tensor of the specified constitutive equation.
    # Return either the strain energy function or the stress tensor of the
    # specified constitutive equation.

    Parameters
    ----------

    problem : MechanicsProblem
        This object must be an instance of the MechanicsProblem class,
        which contains necessary data to formulate the variational form,
        such as material parameters.
    name : string
        The name of the constitutive equation used. To see a list of
        implemented constitutive equations, print the list by the
        name 'implemented'.
    strain : bool
        Should be set to True if the strain energy function is to be
        returned as opposed

    """

    dim = find_dim(problem.deformationGradient)
    I = dlf.Identity(dim)

    if problem.const_eqn.lower() == 'lin-elastic':
        stress = lin_elastic(problem)
    elif problem.const_eqn.lower() == 'neo-hookean':
        stress = neo_hookean(problem)
    else:
        s1 = "The constitutive equation, '%s', has not been implemented."
        raise NotImplementedError(s1 % problem.const_eqn.lower())

    return stress


def lin_elastic(problem):
    """
    Return the Cauchy stress tensor of a linear elastic material.

    """

    dim = find_dim(problem.deformationGradient)
    I = dlf.Identity(dim)

    # Check if the problem is an inverse problem.
    if not problem.config['formulation']['inverse']:
        epsilon = dlf.sym(problem.deformationGradient) - I
    else:
        Finv = dlf.inv(problem.deformationGradient)
        epsilon = dlf.sym(Finv) - I

    # Check if the first Lame parameter is large.
    if problem.config['mechanics']['material']['lambda'].values() > 1e8:
        la = 0.0
    else:
        la = problem.config['mechanics']['material']['lambda']
    mu = problem.config['mechanics']['material']['mu']

    return la*dlf.tr(epsilon)*I + 2.0*mu*epsilon


def neo_hookean(problem):
    """
    Return the first Piola-Kirchhoff stress tensor for a neo-Hookean
    material. The material may be compressible or incompressible.

    Parameters
    ----------
    problem : MechanicsProblem
        This object must be an instance of the MechanicsProblem class,
        which contains necessary data to formulate the variational form,
        such as material parameters.

    """

    F = dlf.deformationGradient
    Finv = dlf.inv(F)
    J = dlf.jacobian
    dim = find_dim(F)
    I = dlf.Identity(dim)

    la = problem.config['mechanics']['material']['lambda']
    mu = problem.config['mechanics']['material']['mu']

    if problem.config['formulation']['inverse']:
        P = inverse_neo_hookean(problem)
    else:
        if problem.config['mechanics']['material']['incompressible']:
            Fbar = J**(-1.0/dim)*F
            Jbar = dlf.det(Fbar)
            P = forward_neo_hookean(Fbar, Jbar, la, mu)
            # Turn this off to test isochoric component above
            P += problem.pressure*J*Finv.T
        else:
            P = forward_neo_hookean(F, J, la, mu, Finv=Finv)

    return P


def forward_neo_hookean(F, J, la, mu, **kwargs):
    """
    Return the first Piola-Kirchhoff stress tensor based on the strain
    energy function

    psi(C) = mu/2*(tr(C) - 3) - mu*ln(J) + la/2*(ln(J))**2.

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

    if not kwargs.has_key('Finv'):
        Finv = dlf.inv(F)

    return mu*F + (la*dlf.ln(J) - mu)*Finv.T


def forward_neo_hookean(problem):
    """
    Return the first Piola-Kirchhoff stress tensor for a compressible
    or incompressible neo-Hookean material.

    Parameters
    ----------
    problem : MechanicsProblem
        This object must be an instance of the MechanicsProblem class,
        which contains necessary data to formulate the variational form,
        such as material parameters.

    """

    dim = find_dim(problem.deformationGradient)
    I = dlf.Identity(dim)

    F = problem.deformationGradient
    Finv = dlf.inv(F)
    J = problem.jacobian
    J23 = J**(-2.0/dim)
    I1 = dlf.tr(F.T * F)

    la = problem.config['mechanics']['material']['lambda']
    mu = problem.config['mechanics']['material']['mu']

    if problem.config['mechanics']['material']['incompressible']:
        p = problem.pressure
        P_vol = J*p*Finv.T
        P_isc = mu*J23*F - 1.0/dim*J23*mu*I1*Finv.T
        P = P_vol + P_isc
    else:
        P = la*dlf.ln(J)*Finv.T + mu*J23*F \
            - 1.0/dim*J23*mu*I1*Finv.T

    return P


def inverse_neo_hookean(problem):
    """
    Return the Cauchy stress tensor for  compressible or incompressible
    material for the inverse elastostatics problem.

    Parameters
    ----------
    problem : MechanicsProblem
        This object must be an instance of the MechanicsProblem class,
        which contains necessary data to formulate the variational form,
        such as material parameters.

    """
    dim = find_dim(problem.deformationGradient)
    I = dlf.Identity(dim)

    # f = problem.deformationGradient
    # j = problem.jacobian
    # j23 = j**(-2.0/dim)
    # fbar = j**(-1.0/dim)*f
    # i1 = dlf.tr(f.T * f)
    la = problem.config['mechanics']['material']['lambda']
    mu = problem.config['mechanics']['material']['mu']
    # sigBar = fbar*(mu/2.0)*fbar.T
    # sigma = 2.0*j*(sigBar - 1.0/dim*dlf.tr(sigBar)*I)

    if problem.config['mechanics']['material']['incompressible']:
        # p = problem.pressure
        # sigma += p*I

        s1 = 'The inverse, incompressible, neo-hookean material has' \
             + ' not been implemented!'
        raise NotImplementedError(s1)
    else:
        # sigma += la*dlf.ln(j)*I

        f = problem.deformationGradient
        c = f.T * f
        cinv = dlf.inv(c)
        j = dlf.det(f)

        sigma = j*mu*(cinv - I) - la*j*dlf.ln(j)*I

    return sigma
