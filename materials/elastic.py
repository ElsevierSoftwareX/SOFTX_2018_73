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

    la = problem.config['mechanics']['material']['lambda']
    mu = problem.config['mechanics']['material']['mu']

    if problem.config['formulation']['inverse']:
        if problem.config['mechanics']['material']['incompressible']:
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
        if problem.config['mechanics']['material']['incompressible']:
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

    return mu*F + (la*dlf.ln(J) - mu)*Finv.T


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
