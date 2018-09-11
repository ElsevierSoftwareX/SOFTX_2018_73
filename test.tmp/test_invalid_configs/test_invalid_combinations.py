import os
import pytest

import fenicsmechanics as fm


# Testing that fm catches when user provides a different number objects
# under the BCs sub dictionary.
@pytest.mark.parametrize("class_name", ("MechanicsProblem",
                                        "SolidMechanicsProblem",
                                        "FluidMechanicsProblem"))
@pytest.mark.parametrize("bc_type, key",
                         (("dirichlet", "regions"),
                          ("dirichlet", "displacement"),
                          ("dirichlet", "velocity"),
                          ("neumann", "regions"),
                          ("neumann", "values"),
                          ("neumann", "types")))
def test_nonmatching_bc_lengths(default_config, class_name, bc_type, key):
    # Adding another object to the list to provoke an exception.
    config = default_config(class_name)
    config['formulation']['bcs'][bc_type][key] += [None]

    # Extracting the specific problem class.
    problem_class = getattr(fm, class_name)
    with pytest.raises(fm.exceptions.InconsistentCombination) as e:
        problem_class(config)
    return None


# This should not raise any exceptions.
@pytest.mark.parametrize("class_name", ("MechanicsProblem",
                                        "SolidMechanicsProblem",
                                        "FluidMechanicsProblem"))
def test_matching_bc_lengths(default_config, class_name):
    config = default_config(class_name)
    problem_class = getattr(fm, class_name)
    problem = problem_class(config)
    return None

# Testing that fm catches when the user specifies the wrong number
# of elements based on whether the material is incompressible or not.
@pytest.mark.parametrize("class_name", ("MechanicsProblem",
                                        "SolidMechanicsProblem",
                                        "FluidMechanicsProblem"))
@pytest.mark.parametrize("incompressible, element",
                         ((True, "p2"), (False, "p2-p1")))
def test_invalid_incompressible_elements(default_config, class_name,
                                         incompressible, element):
    config = default_config(class_name)
    config['formulation']['element'] = element
    config['material']['incompressible'] = incompressible

    # Exiting in this case since compressible fluids have not been
    # implemented, and hence a different exception will be raised.
    if (class_name == "FluidMechanicsProblem") and (not incompressible):
        return None

    problem_class = getattr(fm, class_name)
    with pytest.raises(fm.exceptions.InconsistentCombination) as e:
        problem = problem_class(config)
    return None


# This should not raise any exceptions.
@pytest.mark.parametrize("class_name", ("MechanicsProblem",
                                        "SolidMechanicsProblem",
                                        "FluidMechanicsProblem"))
@pytest.mark.parametrize("incompressible, element",
                         ((True, "p2-p1"), (False, "p2")))
def test_valid_incompressible_elements(default_config, class_name,
                                       incompressible, element):
    config = default_config(class_name)
    config['formulation']['element'] = element
    config['material']['incompressible'] = incompressible

    # Exiting in this case since compressible fluids have not been
    # implemented, and hence a different exception will be raised.
    if (class_name == "FluidMechanicsProblem") and (not incompressible):
        return None

    problem_class = getattr(fm, class_name)
    problem = problem_class(config)
    return None


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# THIS TEST IS NOT PERFORMING AS EXPECTED...
#
# THIS IS CAUSING THE DEFORMATION GRADIENT TO BE SET TO ZERO, WHICH CAUSES A
# 'ZeroDivisionError' DOWN THE LINE WHEN THE INVERSE OF THE GRADIENT IS DEFINED.
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
@pytest.mark.parametrize("class_name", ("MechanicsProblem",
                                        "SolidMechanicsProblem",
                                        "FluidMechanicsProblem"))
def _test_wrong_domain(default_config, class_name):
    config = default_config(class_name)
    if config['material']['type'] == "elastic":
        config['formulation']['domain'] = "eulerian"
    else:
        config['formulation']['domain'] = "lagrangian"

    problem_class = getattr(fm, class_name)
    # with pytest.raises(fm.exceptions.InvalidCombination) as e:
    #     problem = problem_class(config)
    problem = problem_class(config)
    return None
