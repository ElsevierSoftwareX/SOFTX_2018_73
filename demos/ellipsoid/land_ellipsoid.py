import fenicsmechanics as fm

# Specify material model and parameters
mesh_file = fm.get_mesh_file_names("ellipsoid", refinements="1000um", ext="h5")
mat_dict = {'const_eqn': 'guccione', 'type': 'elastic',
    'incompressible': True, 'density': 0.0,
    'bt': 1.0, 'bf': 1.0, 'bfs': 1.0, 'C': 10.0,
    'fibers': {
        'fiber_files': mesh_file,
        'fiber_names': [['fib1', 'fib2', 'fib3'],
                        ['she1', 'she2', 'she3']],
        'elementwise': True}}

# Provide mesh file names
mesh_dict = {'mesh_file': mesh_file,
             'boundaries': mesh_file}

# Specify time integration parameters, BCs, and polynomial degree.
formulation_dict = {
    'time':{'dt': 0.01, 'interval': [0., 1.]},
    'element': 'p2-p1',
    'domain': 'lagrangian',
    'bcs':{
        'dirichlet': {
            'displacement': [[0., 0., 0.]],
            'regions': [10], # Integer ID for base plane
            },
        'neumann': {
            'regions': [20], # Integer ID for inner surface
            'types': ['pressure'],
            'values': ['10.0*t']}}}

# Combine above dictionaries into one.
config = {'material': mat_dict, 'mesh': mesh_dict, 'formulation': formulation_dict}

# Create problem and solver objects.
problem = fm.SolidMechanicsProblem(config)
solver = fm.SolidMechanicsSolver(problem, fname_disp='results/displacement_output.pvd')
solver.set_parameters(linear_solver="mumps")

# Numerically solve the problem.
solver.full_solve()
