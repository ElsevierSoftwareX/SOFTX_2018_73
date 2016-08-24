import dolfin as dlf

from ufl.domain import find_geometric_dimension as find_dim

__IMPLEMENTED__ = {'linear elastic' :
                   {
                       'stress tensor' :
                       {
                           'compressible' : True,
                           'incompressible' : True
                           },
                       'strain energy' :
                       {
                           'compressible' : False,
                           'incompressible' : False
                           }
                       },
                   'neo-hookean' :
                   {
                       'strain energy' :
                       {
                           'incompressible' : False,
                           'compressible' : False
                           },
                       'stress tensor' :
                       {
                           'incompressible' : True,
                           'compressible' : True
                           }
                       }
                   }


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

    dim = find_dim(problem.u)
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


    if not problem._inverse:
        epsilon = dlf.sym(problem.deformationGradient) - I
    else:
        Finv = dlf.inv(problem.deformationGradient)
        epsilon = dlf.sym(Finv) - I

    if problem._lame1.values() > 1e8:
        lame1 = 0.0
    else:
        lame1 = problem._lame1
    lame2 = problem._lame2

    return lame1*dlf.tr(epsilon)*I + 2.0*lame2*epsilon


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

    if problem._inverse:
        P = inverse_neo_hookean(problem)
    else:
        P = forward_neo_hookean(problem)

    return P


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
    F = problem.deformationGradient
    Finv = dlf.inv(F)
    J = dlf.det(problem.deformationGradient)
    J23 = J**(-2.0/dim)
    I1 = dlf.tr(F.T * F)

    if problem._incompressible:
        p = problem.pressure
        P_vol = J*p*Finv.T
        P_isc = problem._lame2*J23*F - 1.0/dim*J23*mu*I1*Finv.T
        P = P_vol + P_isc
    else:
        P = problem._lame1*dlf.ln(J)*Finv.T + mu*J23*F \
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

    f = problem.deformationGradient
    j = problem.jacobian
    j23 = j**(-2.0/dim)
    fbar = j**(-1.0/dim)*f
    i1 = dlf.tr(f.T * f)
    lame1 = problem._lame1
    lame2 = problem._lame2
    sigBar = fbar*(lame2/2.0)*fbar.T
    sigma = 2.0*j*(sigBar - 1.0/dim*dlf.tr(sigBar)*I)

    if problem._incompressible:
        p = problem.pressure
        sigma += p*I
    else:
        sigma += lame1*dlf.ln(j)*I

    return sigma
