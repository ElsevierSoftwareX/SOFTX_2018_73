import fenicsmechanics as fm

material = {
    'type': 'elastic',
    'const_eqn': 'neo_hookean',
    'incompressible': True,
    'kappa': 10e9, # Pa
    'mu': 1.5e6 # Pa
}

mesh_file, boundaries = fm.get_mesh_file_names("unit_domain", ret_facets=True,
                                               refinements=[20, 20])
mesh = {
    'mesh_file': mesh_file,
    'boundaries': boundaries
}

formulation = {
    'element': 'p2-p1',
    'domain': 'lagrangian',
    'bcs': {
        'dirichlet': {
            'displacement': [0.0, 0.0],
            'regions': [1, 3],
            'components': ["x", "y"]
        },
        'neumann': {
            'values': [[1e6, 0.]],
            'regions': [2],
            'types': ['piola']
        }
    }
}

config = {
    'material': material,
    'mesh': mesh,
    'formulation': formulation
}

problem = fm.SolidMechanicsProblem(config)
solver = fm.SolidMechanicsSolver(problem, fname_disp="results/displacement.pvd")
solver.full_solve()
