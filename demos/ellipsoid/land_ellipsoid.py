import fenicsmechanics as fm

# Specify material model and parameters
mesh_dir = '../../meshfiles/ellipsoid/'
mat_dict = {'const_eqn': 'guccione', 'type': 'elastic',
    'incompressible': True, 'density': 0.0,
    'bt': 1.0, 'bf': 1.0, 'bfs': 1.0, 'C': 10.0,
    'fibers': {
        'fiber_files': [mesh_dir + 'fibers/n1-p0-1000um.xml.gz',
                        mesh_dir + 'fibers/n2-p0-1000um.xml.gz'],
        'fiber_names': ['n1', 'n2'], 'element-wise': True}}

# Provide mesh file names
mesh_dict = {'mesh_file': mesh_dir + 'ellipsoid-mesh_fibers_boundaries-1000um.h5',
             'boundaries': mesh_dir + 'ellipsoid-mesh_fibers_boundaries-1000um.h5'}

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
