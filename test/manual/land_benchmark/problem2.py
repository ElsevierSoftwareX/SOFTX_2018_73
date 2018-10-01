import dolfin as dlf
import fenicsmechanics as fm

# Optimization options for the form compiler
dlf.parameters['form_compiler']['cpp_optimize'] = True
dlf.parameters['form_compiler']['representation'] = "uflacs"
dlf.parameters['form_compiler']['quadrature_degree'] = 3
dlf.parameters['form_compiler']['optimize'] = True

rank = dlf.MPI.rank(dlf.mpi_comm_world())
if rank !=0:
    dlf.set_log_level(dlf.ERROR)

# --- GLOBAL VARIABLES --------------------------------------------------------
# output_dir = args.output_dir

# Region IDs
HNBC  = 0  # homogeneous Neumann BC
HDBC  = 10  # homogeneous Dirichlet BC
INBC  = 20  # inhomogeneous Neumann BC

# Time parameters
loading_steps = 100
t0, tf = tspan = [0., 1.]
dt = (tf - t0)/loading_steps
theta = 1.0
beta = 0.25
gamma = 0.5
save_freq = 1

pressure = dlf.Expression('10.0*t', degree=2, t=t0)

# Material subdictionary
mat_dict = {
    'const_eqn': 'guccione',
    'type': 'elastic',
    'incompressible': False,
    'density': 0.0,
    'kappa': 1e3,
    'bt': 1.0,
    'bf': 1.0,
    'bfs': 1.0,
    'C': 10.0,
    'fibers': {
        'fiber_files': ['fibers/n1-p0.xml.gz',
                        'fibers/n2-p0.xml.gz'],
        'fiber_names': ['n1', 'n2'],
        'element': 'p0'
        }
    }

# Mesh subdictionary
# mesh_dict = {'mesh_file': 'new-ellipsoid_300um.h5',
#              'boundaries': 'new-ellipsoid_300um.h5'}
mesh_dict = {'mesh_file': 'ellipsoid_300um.h5',
             'boundaries': 'ellipsoid_300um.h5'}

# Formulation subdictionary
formulation_dict = {
    'time':{
        'unsteady': True,
        'theta': theta,
        'beta': beta,
        'gamma': gamma,
        'dt': dt,
        'interval': tspan
        },
    'domain': 'lagrangian',
    'element': 'p1',
    'inverse': False,
    'bcs':{
        'dirichlet': {
            'displacement': [[0.]*3],
            'regions': [HDBC],
            'velocity': [[0.]*3]
            },
        'neumann': {
            'regions': [INBC],
            'types': ['pressure'],
            'values': [pressure]
            }
        }
    }

# Overall configuration
config = {'material': mat_dict,
          'mesh': mesh_dict,
          'formulation': formulation_dict}

result_file = 'results-guccione-tmp/ellipsoid-fm.pvd'
problem = fm.SolidMechanicsProblem(config)
solver = fm.SolidMechanicsSolver(problem, fname_disp=result_file)
solver.full_solve(save_freq=save_freq)
