import fenicsmechanics as fm

material = {
    'type': 'elastic',
    'const_eqn': 'neo_hookean',
    'incompressible': True,
    'kappa': 10e9, # Pa
    'mu': 1.5e6 # Pa
}

mesh_dir = '../../meshfiles/unit_domain/'
mesh = {
    'mesh_file': mesh_dir + 'unit_domain-mesh-20x20.xml.gz',
    'boundaries': mesh_dir + 'unit_domain-boundaries-20x20.xml.gz'
}

formulation = {
    'element': 'p2-p1',
    'domain': 'lagrangian',
    'bcs': {
        'dirichlet': {
            'displacement': [[0, 0]],
            'regions': [1]
        },
        'neumann': {
            'values': [[1e6,0]],
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
