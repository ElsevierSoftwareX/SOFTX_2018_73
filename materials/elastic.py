import dolfin as dlf
import solid_materials as mts

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

    la = problem.config['material']['lambda']
    mu = problem.config['material']['mu']
    mat = mts.LinearMaterial(mu=mu, lmbda=la)
    mat.set_active(False)
    mat.set_inverse(problem.config['formulation']['inverse'])
    mat.set_incompressible(problem.config['material']['incompressible'])
    mat.print_info()
    u  = problem.displacement
    p  = problem.pressure

    stress = mat.stress_tensor(u, p)

    if problem.config['formulation']['time']['unsteady']:
        u0  = problem.displacement0
        stress0 = mat.stress_tensor(u0, p)

    if problem.config['formulation']['time']['unsteady']:
        return stress, stress0
    else:
        return stress

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

    lam = problem.config['material']['lambda']
    mu  = problem.config['material']['mu']
    mat = mts.NeoHookeMaterial(mu=mu, lmbda=lam)
    mat.set_active(False)
    mat.set_inverse(problem.config['formulation']['inverse'])
    mat.set_incompressible(problem.config['material']['incompressible'])
    mat.print_info()
    u  = problem.displacement
    p  = problem.pressure

    stress = mat.stress_tensor(u, p)

    if problem.config['formulation']['time']['unsteady']:
        u0  = problem.displacement0
        stress0 = mat.stress_tensor(u0, p)
        return stress, stress0
    else:
        return stress

def guccione(problem):
    """
    """

    lam = problem.config['material']['lambda']
    mu  = problem.config['material']['mu']
    mat = mts.GuccioneMaterial(mu=mu, lmbda=lam)
    mat.set_active(False)
    mat.set_inverse(problem.config['formulation']['inverse'])
    mat.set_incompressible(problem.config['material']['incompressible'])
    mat.print_info()
    u  = problem.displacement
    p  = problem.pressure

    stress = mat.stress_tensor(u, p)

    if problem.config['formulation']['time']['unsteady']:
        u0  = problem.displacement0
        stress0 = mat.stress_tensor(u0, p)
        return stress, stress0
    else:
        return stress
