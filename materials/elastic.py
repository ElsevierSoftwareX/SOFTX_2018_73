import dolfin as dlf

from ufl.domain import find_geometric_dimension as find_dim


def lin_elastic(problem):
    """
    Return the Cauchy stress tensor of a linear elastic material. The
    stress tensor is formulated based on the parameters set in the config
    dictionary of the 'problem' object.

    Parameters
    ----------

    problem : MechanicsProblem
        This object must be an instance of the MechanicsProblem class,
        which contains necessary data to formulate the variational form,
        such as material parameters.

    """

    if problem.config['formulation']['inverse']:
        T = inverse_lin_elastic(problem.deformationGradient,
                                problem.config['material']['lambda'],
                                problem.config['material']['mu'])
    else:
        T = forward_lin_elastic(problem.deformationGradient,
                                problem.config['material']['lambda'],
                                problem.config['material']['mu'])

    return T


def forward_lin_elastic(F, la, mu):
    """


    """

    dim = find_dim(F)
    I = dlf.Identity(dim)
    epsilon = dlf.sym(F) - I
    if la.values() > 1e8:
        T = 2.0*mu*epsilon
    else:
        T = la*dlf.tr(epsilon)*I + 2.0*mu*epsilon

    return T


def inverse_lin_elastic(F, la, mu):
    """


    """

    dim = find_dim(F)
    I = dlf.Identity(dim)
    epsilon = dlf.sym(dlf.inv(F)) - I
    if la.values() > 1e8:
        T = 2.0*mu*epsilon
    else:
        T = la*dlf.tr(epsilon)*I + 2.0*mu*epsilon

    return T


def neo_hookean(problem):
    """
    Return the first Piola-Kirchhoff or the Cauchy stress tensor for
    a neo-Hookean material. If an inverse problem is given, the Cauchy
    stress is returned, while the first Piola-Kirchhoff stress is returned
    otherwise. Note that the material may be compressible or incompressible.

    Parameters
    ----------
    problem : MechanicsProblem
        This object must be an instance of the MechanicsProblem class,
        which contains necessary data to formulate the variational form,
        such as material parameters.

    """

    F = problem.deformationGradient
    Finv = dlf.inv(F)
    J = problem.jacobian
    dim = find_dim(F)
    I = dlf.Identity(dim)

    la = problem.config['material']['lambda']
    mu = problem.config['material']['mu']

    if problem.config['formulation']['inverse']:
        if problem.config['material']['incompressible']:
            fbar = J**(-1.0/dim)*F
            jbar = dlf.det(fbar)
            stress = inverse_neo_hookean(fbar, jbar, la, mu)
            # Turn this off to test isochoric component above
            stress += problem.pressure*I
        else:
            # Note that F here is the deformation gradient from
            # the current to the reference configuration.
            stress = inverse_neo_hookean(F, J, la, mu)
    else:
        if problem.config['material']['incompressible']:
            Fbar = J**(-1.0/dim)*F
            Jbar = dlf.det(Fbar)
            stress = forward_neo_hookean(Fbar, Jbar, la, mu)
            # Turn this off to test isochoric component above
            stress += problem.pressure*J*Finv.T
        else:
            stress = forward_neo_hookean(F, J, la, mu, Finv=Finv)

    return stress


def forward_neo_hookean(F, J, la, mu, Finv=None):
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

    if Finv is None:
        Finv = dlf.inv(F)

    return mu*F  + (la*dlf.ln(J) - mu)*Finv.T


def inverse_neo_hookean(f, j, la, mu):
    """
    Return the Caucy stress tensor based on the strain energy function

    psi(c) = mu/2*(i2/i3 - 3) + mu*ln(j) + la/2*(ln(j))**2.

    Parameters
    ----------

    f :
        Deformation gradient from the current to the
        reference configuration.
    j :
        Determinant of the deformation gradient from
        the current to the reference configuration.
    la :
        First parameter for the neo-Hookean material.
    mu:
        Second parameter for the neo-Hookean material.

    """

    dim = find_dim(f)
    I = dlf.Identity(dim)
    c = f.T * f
    cinv = dlf.inv(c)

    return -j*(mu + la*dlf.ln(j))*I + j*mu*cinv
