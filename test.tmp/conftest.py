import pytest

# Need to figure out how to generate a default config dictionary based
# on the class that it will be used for.

def _default_config(class_name):
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
        'time': {'unsteady': False},
        'element': "p2-p0",
        # 'domain': "lagrangian",
        'bcs': {
            'dirichlet': {
                'displacement': [[0.0]*3],
                'velocity': [[0.0]*3],
                'regions': [1],
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

    config = {'material': mat_dict, 'mesh': mesh_dict,
              'formulation': formulation_dict}

    return config


@pytest.fixture
def default_config():
    return _default_config

# @pytest.fixture
# def default_config():
#     import fenicsmechanics as fm

#     mat_dict = {
#         'const_eqn': "lin_elastic",
#         'type': "elastic",
#         'incompressible': True,
#         'density': 10.0,
#         'la': 15.0,
#         'mu': 5.0
#     }

#     mesh_file, boundaries_file = fm._get_mesh_file_names("unit_domain",
#                                                          ret_facets=True,
#                                                          refinements=[12]*3)
#     mesh_dict = {
#         'mesh_file': mesh_file,
#         'boundaries': boundaries_file
#     }

#     formulation_dict = {
#         'time': {'unsteady': False},
#         'element': "p2-p0",
#         'domain': "lagrangian",
#         'bcs': {
#             'dirichlet': {
#                 'displacement': [[0.0]*3],
#                 'velocity': [[0.0]*3],
#                 'regions': [1],
#                 # 'pressure': [0.0],
#                 # 'p_regions': [1]
#             },
#             'neumann': {
#                 'values': ["30.0*t", [0.0, 1.0, 0.0]],
#                 'regions': [2, 2],
#                 'types': ["pressure", "piola"]
#             }
#         }
#     }

#     config = {'material': mat_dict, 'mesh': mesh_dict,
#               'formulation': formulation_dict}

#     return config
