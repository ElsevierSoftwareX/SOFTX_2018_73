import pytest
import fenicsmechanics as fm


problem_classes = ("MechanicsProblem",
                   "SolidMechanicsProblem",
                   "FluidMechanicsProblem")

_EXPRESSIONS = {
    'body_force': ["np.log(1.0 + t)", "np.exp(t)", "1.0 - t"],
    'displacement': ["1.0 + 2.0*t", "3.0*t", "1.0"],
    'velocity': ["np.tanh(t)", "np.exp(-t)*np.cos(2.0*np.pi*t)", "0.5"],
    'values': ["t", "t*t", "10.0*np.cos(t)"]
}


@pytest.mark.parametrize("class_name, field_name",
                         (("MechanicsProblem", "formulation/body_force"),
                          ("MechanicsProblem", "formulation/bcs/dirichlet/displacement"),
                          ("MechanicsProblem", "formulation/bcs/dirichlet/velocity"),
                          ("MechanicsProblem", "formulation/bcs/neumann/values"),
                          ("SolidMechanicsProblem", "formulation/body_force"),
                          ("SolidMechanicsProblem", "formulation/bcs/dirichlet/displacement"),
                          ("SolidMechanicsProblem", "formulation/bcs/neumann/values"),
                          ("FluidMechanicsProblem", "formulation/body_force"),
                          ("FluidMechanicsProblem", "formulation/bcs/dirichlet/velocity"),
                          ("FluidMechanicsProblem", "formulation/bcs/neumann/values")))
def test_single_time_update_tmp(default_config, class_name, field_name):
    import numpy as np
    # import dolfin as dlf
    config = default_config(class_name, unsteady=True)
    if "body_force" in field_name:
        config['formulation']['body_force'] = None

    t, tf = config['formulation']['time']['interval']
    dt = config['formulation']['time']['dt']
    subconfig, last_key = _get_subdict(field_name, config, ret_last_key=True)
    fm_expr = _get_expressions(_EXPRESSIONS[last_key])
    _update_subconfig(last_key, subconfig, fm_expr, t)

    problem_class = getattr(fm, class_name)
    problem = problem_class(config)
    subconfig, last_key = _get_subdict(field_name, problem.config, ret_last_key=True)

    tspan = np.arange(t, tf + dt/10.0, dt)
    expected_values = _get_expected_values(tspan, *_EXPRESSIONS[last_key])

    actual_values = np.zeros(expected_values.shape)
    for i, t in enumerate(tspan):
        problem.update_time(t)
        if last_key == "body_force":
            subconfig[last_key].eval(actual_values[i, :], np.zeros(3))
        else:
            subconfig[last_key][0].eval(actual_values[i, :], np.zeros(3))
    assert np.all(expected_values == actual_values)

    return None


@pytest.mark.parametrize("class_name", problem_classes)
def test_all_time_updates_tmp(default_config, class_name):
    import numpy as np
    config = default_config(class_name, unsteady=True)
    config['formulation']['body_force'] = None
    t, tf = config['formulation']['time']['interval']
    dt = config['formulation']['time']['dt']
    tspan = np.arange(t, tf + dt/10.0, dt)

    all_expr = list()
    all_keys = ("formulation/body_force",)
    if class_name in ["MechanicsProblem", "SolidMechanicsProblem"]:
        all_keys += ("formulation/bcs/dirichlet/displacement",)
    if class_name in ["MechanicsProblem", "FluidMechanicsProblem"]:
        all_keys += ("formulation/bcs/dirichlet/velocity",)
    all_keys += ("formulation/bcs/neumann/values",)

    for key in all_keys:
        subconfig, last_key = _get_subdict(key, config, ret_last_key=True)
        expr = _EXPRESSIONS[last_key]
        all_expr.extend(expr)
        fm_expr = _get_expressions(expr)
        _update_subconfig(last_key, subconfig, fm_expr, t)

    problem_class = getattr(fm, class_name)
    problem = problem_class(config)

    all_expected = _get_expected_values(tspan, *all_expr)
    all_actual = np.zeros(all_expected.shape)
    for i, t in enumerate(tspan):
        problem.update_time(t)
        for key in all_keys:
            subconfig, last_key = _get_subdict(key, problem.config,
                                               ret_last_key=True)
            if last_key == "body_force":
                subconfig[last_key].eval(all_actual[i, 0:3], np.zeros(3))
            elif last_key == "displacement":
                subconfig[last_key][0].eval(all_actual[i, 3:6], np.zeros(3))
            elif last_key == "velocity":
                if all_expected.shape[1] == 9:
                    subconfig[last_key][0].eval(all_actual[i, 3:6], np.zeros(3))
                else:
                    subconfig[last_key][0].eval(all_actual[i, 6:9], np.zeros(3))
            else:
                subconfig[last_key][0].eval(all_actual[i, -3:], np.zeros(3))

    assert np.all(all_actual == all_expected)

    return None


def _update_subconfig(last_key, subconfig, fm_expr, t):
    import dolfin as dlf
    if last_key == "body_force":
        fm_expr = dlf.Expression(fm_expr, degree=1, t=t)
    elif last_key == "displacement":
        subconfig['regions'] = [1]
    elif last_key == "velocity":
        subconfig['regions'] = [1]
    elif last_key == "values":
        subconfig['types'] = ['cauchy']
        subconfig['regions'] = [2]
    subconfig[last_key] = fm_expr if last_key == "body_force" else [fm_expr]
    return None



def _get_expected_values(t, *expr_list):
    import numpy as np
    expected_values = list()
    # print("expr_list = ", expr_list)
    for expr in expr_list:
        # print("expr = ", expr)
        eval_expr = eval(expr)
        if isinstance(eval_expr, float):
            eval_expr = eval_expr*np.ones(t.shape)
        expected_values.append(eval_expr)
    return np.array(expected_values).T


def _get_expressions(expr_list):
    import re
    expr_fm_list = list()
    for expr in expr_list:
        expr_fm_list.append(re.sub("np.", "", expr))
    return expr_fm_list


def _get_subdict(key, my_dict, ret_last_key=False):
    subdict = my_dict
    key_list = key.split("/")
    keys_used = list()
    for sub in key_list:
        old_subdict = subdict
        subdict = subdict[sub]
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


if __name__ == "__main__":
    import sys
    sys.path.append("../")
    from conftest import _default_config

    # Trying specific values
    _ = test_single_time_update_tmp(_default_config, "MechanicsProblem",
                                    "formulation/body_force")
    _ = test_single_time_update_tmp(_default_config, "MechanicsProblem",
                                    "formulation/bcs/dirichlet/displacement")
    _ = test_single_time_update_tmp(_default_config, "MechanicsProblem",
                                    "formulation/bcs/neumann/values")
    _ = test_single_time_update_tmp(_default_config, "FluidMechanicsProblem",
                                    "formulation/body_force")

    _ = test_all_time_updates_tmp(_default_config, "SolidMechanicsProblem")
