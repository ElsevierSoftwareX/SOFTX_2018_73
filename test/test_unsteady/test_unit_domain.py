import os
import sys
import argparse
import dolfin as dlf

import fenicsmechanics as fm

# Parse through the arguments provided at the command line.
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dim',
                    help='dimension',
                    default=2,
                    type=int)
parser.add_argument('-r', '--refinement',
                    help='mesh refinement',
                    default=12,
                    type=int)
parser.add_argument('-m', '--material',
                    help='constitutive relation',
                    default='lin_elastic',
                    choices=['lin_elastic', 'neo_hookean',
                             'guccione', 'fung'])
parser.add_argument('-inv', '--inverse',
                    help='activate inverse elasticity',
                    action='store_true')
parser.add_argument('-inc', '--incompressible',
                    help='active incompressible material',
                    action='store_true')
parser.add_argument('-hdf5',
                    help="use HDF5 files",
                    action='store_true')
parser.add_argument('-s','--save',
                    help='save solution',
                    action='store_true')
parser.add_argument('-v', '--compute-volume',
                    help='compute deformed volume',
                    action='store_true')
parser.add_argument('-bs', '--block-solver',
                    help="Use MechanicsBlockSolver",
                    action='store_true')
args = parser.parse_args()

# Mesh file names based on arguments given
mesh_dims = (args.refinement,)*args.dim
dim_str = 'x'.join(['%i' % i for i in mesh_dims])

if args.incompressible:
    name_dims = ('incomp_' + args.material, dim_str)
    element_type = 'p2-p1'
else:
    name_dims = ('comp_' + args.material, dim_str)
    element_type = 'p2'

mesh_dir = '../../meshfiles/unit_domain/'
if args.save:
    if args.inverse:
        disp_file = 'results/inverse-disp-%s-%s.pvd' % name_dims
        vel_file = 'results/inverse-vel-%s-%s.pvd' % name_dims
        hdf5_file = 'results/inverse-hdf5-%s-%s.h5' % name_dims
        xdmf_file = 'results/inverse-xdmf-%s-%s.xdmf' % name_dims
        if args.incompressible:
            p_file = 'results/inverse-pressure-%s-%s.pvd' % name_dims
        else:
            p_file = None
    else:
        disp_file = 'results/forward-disp-%s-%s.pvd' % name_dims
        vel_file = 'results/forward-vel-%s-%s.pvd' % name_dims
        hdf5_file = 'results/forward-hdf5-%s-%s.h5' % name_dims
        xdmf_file = 'results/forward-xdmf-%s-%s.xdmf' % name_dims
        if args.incompressible:
            p_file = 'results/forward-pressure-%s-%s.pvd' % name_dims
        else:
            p_file = None
else:
    disp_file = None
    vel_file = None
    hdf5_file = None
    xdmf_file = None
    p_file = None

mesh_file = mesh_dir + 'unit_domain-mesh-%s' % dim_str
boundaries = mesh_dir + 'unit_domain-boundaries-%s' % dim_str
if args.hdf5:
    mesh_file += '.h5'
    boundaries += '.h5'
else:
    mesh_file += '.xml.gz'
    boundaries += '.xml.gz'

# Check if the mesh file exists
if not os.path.isfile(mesh_file):
    raise Exception('The mesh file, \'%s\', does not exist. ' % mesh_file
                    + 'Please run the script \'generate_mesh_files.py\''
                    + 'with the same arguments first.')

# Check if the mesh function file exists
if not os.path.isfile(boundaries):
    raise Exception('The mesh function file, \'%s\', does not exist. ' % boundaries
                    + 'Please run the script \'generate_mesh_files.py\''
                    + 'with the same arguments first.')

# Optimization options for the form compiler
dlf.parameters['form_compiler']['cpp_optimize'] = True
dlf.parameters['form_compiler']['quadrature_degree'] = 3
ffc_options = {'optimize' : True,
               'eliminate_zeros' : True,
               'precompute_basis_const' : True,
               'precompute_ip_const' : True}

# Elasticity parameters
E = 20.0 # Young's modulus (TRY A SMALLER VALUE)
if args.incompressible:
    nu = 0.5 # Poisson's ratio
else:
    nu = 0.3 # Poisson's ratio
inv_la = (1. + nu)*(1. - 2.*nu)/(E*nu) # Reciprocal of 1st Lame parameter
mu = E/(2.*(1. + nu)) # 2nd Lame parameter

# Time interval
t0 = 0.0
tf = 1.0
nsteps = 200
dt = (tf - t0)/nsteps
tspan = [t0, tf]
theta = 1.0
beta = 0.25
gamma = 0.5
save_freq = int(0.01/dt)

# Traction on the Neumann boundary region
expr_subclass = False
if expr_subclass:
    class Pressure(dlf.Expression):
        def __init__(self, t=0.0, **kwargs):
            self.t = t
        def eval(self, vals, x):
            vals[0] = 3.0*self.t
        def value_shape(self):
            return tuple()
    pressure = Pressure(t=t0, degree=2)
else:
    pressure = dlf.Expression('3.0*t', t=t0, degree=2)

# Region IDs
ALL_ELSE = 0
CLIP = 1
TRACTION = 2

# Material subdictionary
material_dict = {'const_eqn': args.material,
                 'type': 'elastic',
                 'incompressible': args.incompressible,
                 'density': 10.0}
if args.material in ['fung', 'guccione']:
    if args.material == 'fung':
        material_dict['d'] = [10.0]*3 + [0.0]*3 + [5.0]*3
    else:
        material_dict['bt'] = 10.0
        material_dict['bf'] = 1.0
        material_dict['bfs'] = 5.0

    from numpy import sqrt
    material_dict['C'] = 20.0
    material_dict['kappa'] = 1e4
    if args.dim == 2:
        e1 = dlf.Constant([1.0/sqrt(2.0)]*2)
        e2 = dlf.Constant([1.0/sqrt(2.0), -1.0/sqrt(2.0)])
    else:
        e1 = dlf.Constant([1.0/sqrt(2.0)]*2 + [0.0])
        e2 = dlf.Constant([1.0/sqrt(2.0), -1.0/sqrt(2.0), 0.0])
    material_dict['fibers'] = {'fiber_files': [e1, e2],
                               'fiber_names': ['e1', 'e2'],
                               'element': None}
else:
    material_dict.update({'inv_la': inv_la, 'mu': mu})

# Mesh subdictionary
mesh_dict = {'mesh_file': mesh_file,
             'boundaries': boundaries}

# Formulation subdictionary
formulation_dict = {
    'time' : {
        'theta' : theta,
        'beta': beta,
        'gamma': gamma,
        'dt' : dt,
        'interval' : tspan
    },
    'element': element_type,
    'domain' : 'lagrangian',
    'inverse' : args.inverse,
    'body_force' : dlf.Constant([0.]*args.dim),
    'bcs' : {
        'dirichlet' : {
            'displacement' : [dlf.Constant([0.]*args.dim)],
            'regions' : [CLIP],
        },
        'neumann' : {
            'regions' : [TRACTION],
            'types' : ['pressure'],
            'values' : [pressure]
        }
    }
}

# Problem configuration dictionary
config = {'material': material_dict,
          'mesh': mesh_dict,
          'formulation': formulation_dict}

if args.block_solver:
    config['formulation']['bcs']['dirichlet']['velocity'] = [dlf.Constant([0.]*args.dim)]
    problem = fm.MechanicsProblem(config)
    solver = fm.MechanicsBlockSolver(problem, fname_disp=disp_file, fname_vel=vel_file,
                                     fname_pressure=p_file, fname_hdf5=hdf5_file,
                                     fname_xdmf=xdmf_file)
    solver.solve(save_freq=save_freq)
else:
    problem = fm.SolidMechanicsProblem(config)
    solver = fm.SolidMechanicsSolver(problem, fname_disp=disp_file,
                                     fname_pressure=p_file, fname_hdf5=hdf5_file,
                                     fname_xdmf=xdmf_file)
    solver.full_solve(save_freq=save_freq)

# Compute the final volume
if args.compute_volume:
    W1 = dlf.VectorFunctionSpace(problem.mesh, 'CG', 1)
    xi1 = dlf.TestFunction(W1)
    du1 = dlf.TrialFunction(W1)
    u_move = dlf.Function(W1)
    move_bcs = dlf.DirichletBC(W1, dlf.Constant([0.0]*args.dim),
                               problem.boundaries, CLIP)
    a = dlf.dot(xi1, du1)*dlf.dx
    L = dlf.dot(xi1, problem.displacement)*dlf.dx
    dlf.solve(a == L, u_move, move_bcs)

    ale = dlf.ALE()
    ale.move(problem.mesh, u_move)
    print ("Total volume after: ", \
           dlf.assemble(dlf.Constant(1.0)*dlf.dx(domain=problem.mesh)))
