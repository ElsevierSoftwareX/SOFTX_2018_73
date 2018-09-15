import pytest
import numpy as np
import fenicsmechanics as fm

problem_classes = ("MechanicsProblem",
                   "SolidMechanicsProblem",
                   "FluidMechanicsProblem")

@pytest.mark.skip(reason="Test is not complete.")
@pytest.mark.parametrize("class_name", problem_classes)
@pytest.mark.parametrize("test_field", ("formulation/body_force",
                                        "formulation/bcs/dirichlet/displacement",
                                        "formulation/bcs/dirichlet/velocity",
                                        "formulation/bcs/dirichlet/pressure",
                                        "formulation/bcs/neumann/values"))
def test_time_expr_update(default_config, class_name, test_field):
    config = default_config(class_name, unsteady=True)
    interval = list(config['formulation']['time']['interval'])
    dt = config['formulation']['time']['dt']
    interval[1] += dt/10.0
    tspan = np.arange(*interval, dt)

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # SHOULD PARAMETRIZE THE FIELD WE ARE UPDATING THAT WAY THEY
    # ARE DIFFERENT TESTS.
    #
    # THIS WILL PROBABLY REQUIRE THAT I ADAPT '__convert_bc_values'
    # SO THAT IT CAN BE USED WITH FIELDS OTHER THAN THOSE IN THE
    # 'bcs' SUBDICTIONARY. THIS IS PROBABLY SOMETHING THAT SHOULD
    # BE DONE ANYWAY.
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    raise Exception("FUNCTION NOT COMPLETE")
    bf_expr, bf_fm_expr = _get_eval_and_fm_expr("np.log(1.0 + t)",
                                                "np.exp(t)",
                                                "3.0*np.pi*np.cos(2.0*np.pi*t)")
    bf_expected = _get_expected_vals(tspan, bf_expr)

    return None


def _get_eval_and_fm_expr(*expressions):
    import re
    orig_expr = list()
    fm_expr = list()
    for expr in expressions:
        orig_expr.append(expr)
        fm_expr(re.sub("np.", "", expr))
    return orig_expr, fm_expr


def _get_expected_vals(t, expr_list):
    if isinstance(expr_list, str):
        expr_list = [expr_list]
    all_vals = list()
    for expr in expr_list:
        all_vals.append(eval(expr))
    return np.array(all_vals).T
