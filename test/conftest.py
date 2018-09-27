import pytest

def _default_config(class_name, unsteady=False):
    import fenicsmechanics as fm

    mat_dict = {
        'incompressible': True,
        'density': 10.0,
    }

    mesh_file, boundaries_file = fm._get_mesh_file_names("unit_domain",
                                                         ret_facets=True,
                                                         refinements=[12]*3)
    mesh_dict = {
        'mesh_file': mesh_file,
        'boundaries': boundaries_file
    }

    formulation_dict = {
        'element': "p2-p0",
        'bcs': {
            'dirichlet': {
                'displacement': [[0.0]*3],
                'velocity': [[0.0]*3],
                'regions': [1],
                'components': ["all"]
            },
            'neumann': {
                'values': ["30.0*t", [0.0, 1.0, 0.0]],
                'regions': [2, 2],
                'types': ["pressure", "cauchy"]
            }
        }
    }

    if class_name == "FluidMechanicsProblem":
        mat_dict['const_eqn'] = "newtonian"
        mat_dict['type'] = "viscous"
        mat_dict['mu'] = 1.0

        formulation_dict['domain'] = "eulerian"
    else:
        mat_dict['const_eqn'] = "lin_elastic"
        mat_dict['type'] = "elastic"
        mat_dict['la'] = 15.0
        mat_dict['mu'] = 5.0

        formulation_dict['domain'] = "lagrangian"

    if unsteady:
        formulation_dict['time'] = {
            'unsteady': True,
            'interval': [0., 1.],
            'dt': 0.1
        }
    else:
        formulation_dict['time'] = {'unsteady': False}

    config = {'material': mat_dict, 'mesh': mesh_dict,
              'formulation': formulation_dict}

    return config


# Returns the function above used to generate a config dictionary based
# on the problem class that will be used.
@pytest.fixture
def default_config():
    return _default_config
