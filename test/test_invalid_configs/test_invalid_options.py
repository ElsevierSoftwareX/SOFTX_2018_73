import pytest
import fenicsmechanics as fm

# Testing UI problem classes since the user will likely not use
# BaseMechanicsProblem directly.
# problem_classes = ("BaseMechanicsProblem",)
problem_classes = ("MechanicsProblem",
                   "SolidMechanicsProblem",
                   "FluidMechanicsProblem")

@pytest.mark.parametrize("class_name", problem_classes)
@pytest.mark.parametrize("subdict, rm_key", (("material", "type"),
                                             ("material", "const_eqn"),
                                             ("material", "incompressible"),
                                             ("mesh", "mesh_file"),
                                             ("mesh", "boundaries"),
                                             ("formulation", "element"),
                                             ("formulation", "domain"),
                                             ("formulation/time", "interval"),
                                             ("formulation/time", "dt")))
def test_required_parameters(default_config, class_name, subdict, rm_key):
    if "time" in subdict:
        sub_config = config = default_config(class_name, unsteady=True)
    else:
        sub_config = config = default_config(class_name, unsteady=False)
    sub_config = _get_subdict(subdict, config)
    sub_config.pop(rm_key)

    problem_class = getattr(fm, class_name)
    with pytest.raises(fm.exceptions.RequiredParameter) as e:
        problem_class(config)
    return None


# Test for unrecognized values for different keys. I.e., "my_domain"
# for the 'domain' key (only "lagrangian", "eulerian", and "ale")
# are explicitly recognized. This will test values that should be strings
# since they are specific names that are being used.
@pytest.mark.parametrize("class_name", problem_classes)
@pytest.mark.parametrize("key, new_value", (("material/type", "plastic"),
                                            ("material/const_eqn", "rivlin"),
                                            ("formulation/element", "d1-d2"),
                                            ("formulation/element", "p2-p1-p0"),
                                            ("formulation/domain", "taylor"),
                                            ("formulation/domain", "ALE"),
                                            ("formulation/bcs/neumann/types", ["force", "load"]),
                                            ("formulation/bcs/dirichlet/components", ["x1"])))
def test_unrecognized_parameters(default_config, class_name, key, new_value):
    config = default_config(class_name, unsteady=False)
    subconfig, last_key = _get_subdict(key, config, ret_last_key=True)
    subconfig[last_key] = new_value

    problem_class = getattr(fm, class_name)
    expected_exceptions = (NotImplementedError,
                           fm.exceptions.InvalidCombination,
                           fm.exceptions.InvalidOption)
    with pytest.raises(expected_exceptions) as e:
        problem_class(config)
    return None


# Purposely passing in invalid types to make sure they are caught.
# Should include more tests for values within deeper subdictionaries.
@pytest.mark.parametrize("class_name", problem_classes)
@pytest.mark.parametrize("key, new_value",
                         (("material/type", True),
                          ("material/const_eqn", 1),
                          ("material/incompressible", "yes"),
                          ("mesh/mesh_file", [0.3]),
                          ("mesh/mesh_file", 1),
                          ("mesh/boundaries", False),
                          ("mesh/boundaries", 2),
                          ("formulation/element", 1.1),
                          ("formulation/domain", set(range(10))),
                          ("formulation/body_force", [1.0, "x[0]", 4.0]),
                          ("formulation/bcs/neumann/types", "my_bcs"),
                          ("formulation/bcs/neumann/values", 3.0),
                          ("formulation/bcs/neumann/regions", 3),
                          ("formulation/bcs/neumann/types", [True, "piola"]),
                          ("formulation/bcs/neumann/values", [["x[0]", 3.0, 0.0],
                                                              ["x[0]", "1.0", "0.0"]]),
                          ("formulation/bcs/neumann/regions", ["right", "top"]),
                          ("formulation/bcs/dirichlet/displacement", 0.0),
                          ("formulation/bcs/dirichlet/velocity", "fast"),
                          ("formulation/bcs/dirichlet/regions", "left"),
                          ("formulation/bcs/dirichlet/displacement", [["x[0]", 1.0, 0.0]]),
                          ("formulation/bcs/dirichlet/velocity", [["x[0]", 0.0, 1.0]]),
                          ("formulation/bcs/dirichlet/regions", ["left"]),
                          ("formulation/bcs/dirichlet/components", [[1]]),
                          ("formulation/time/dt", "0.1")))
def test_invalid_types(default_config, class_name, key, new_value):
    if 'time' in key:
        config = default_config(class_name, unsteady=True)
    else:
        config = default_config(class_name, unsteady=False)

    subconfig, last_key = _get_subdict(key, config, ret_last_key=True)
    subconfig[last_key] = new_value

    problem_class = getattr(fm, class_name)
    with pytest.raises(TypeError) as e:
        problem_class(config)
    return None


def _get_subdict(key, my_dict, ret_last_key=False):
    subdict = my_dict
    key_list = key.split("/")
    keys_used = list()
    for sub in key_list:
        old_subdict = subdict

        # Check if key is in subdictionary, else exit loop.
        if sub in subdict:
            subdict = subdict[sub]
        else:
            break

        # Check if last object obtained is a dictionary, else exit loop.
        if not isinstance(subdict, dict):
            subdict = old_subdict
            break
        keys_used.append(sub)

    if keys_used != key_list:
        print("Returning '%s' since '%s' is not a dictionary." \
              % ("/".join(keys_used), key))

    if len(set(key_list).difference(keys_used)) > 1:
        msg = "There are multiple keys left, but the last object is " \
              + "not a dictionary."
        raise KeyError(msg)

    if ret_last_key:
        ret = subdict, key_list[-1]
    else:
        ret = subdict

    return ret
