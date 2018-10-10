import fenicsmechanics as fm

mesh_file, boundaries = fm.get_mesh_file_names("lshape", ret_facets=True, refinements="fine")
config = {
    'material':
    {
        'type': 'elastic',
        'const_eqn': 'neo_hookean',
        'incompressible': True,
        'kappa': 10e9,
        'mu': 1.5e6
    },
    'mesh': {
        'mesh_file': mesh_file,
        'boundaries': boundaries
    },
    'formulation': {
        'element': 'p1-p1',
        'domain': 'lagrangian',
        'inverse': True,
        'bcs': {
            'dirichlet': {
                'displacement': [[0, 0]],
                'regions': [1]
            },
            'neumann': {
                'values': [[0., -1e5]],
                'regions': [2],
                'types': ['cauchy']
            }
        }
    }
}

# First solve the inverse elastostatics problem.
problem = fm.SolidMechanicsProblem(config)
solver = fm.SolidMechanicsSolver(problem, fname_disp="results/unloaded_config.pvd")
solver.full_solve()

# Move the mesh using dolfin's ALE functionality
from dolfin import ALE, Mesh
ALE.move(problem.mesh, problem.displacement)
mesh_copy = Mesh(problem.mesh)

# Only need to change relevant entries in config
config['mesh']['mesh_file'] = mesh_copy
config['formulation']['inverse'] = False

# Solve a 'forward' problem.
problem = fm.SolidMechanicsProblem(config)
solver = fm.SolidMechanicsSolver(problem, fname_disp="results/loaded_config.pvd")
solver.full_solve()
