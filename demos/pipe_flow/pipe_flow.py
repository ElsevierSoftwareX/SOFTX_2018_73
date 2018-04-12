import fenicsmechanics as fm

# Generate a 10mx1m rectangle mesh.
import dolfin as dlf
mesh = dlf.RectangleMesh(dlf.Point(), dlf.Point(10, 1), 100, 10)
boundaries = dlf.MeshFunction("size_t", mesh, 1)
boundaries.set_all(0)

# Define different regions of the boundary.
inlet = dlf.CompiledSubDomain("near(x[0], 0.0)")
outlet = dlf.CompiledSubDomain("near(x[0], 10.0)")
no_slip = dlf.CompiledSubDomain("near(x[1], 0.0) || near(x[1], 1.0)")

# Mark the different regions with integers.
inlet.mark(boundaries, 1)
outlet.mark(boundaries, 2)
no_slip.mark(boundaries, 3)

mesh_dict = {
    'mesh_file': mesh,
    'boundaries': boundaries
}

material_dict = {
    'type': 'viscous',
    'const_eqn': 'newtonian',
    'incompressible': True,
    'density': 1, # kg/m^3
    'mu': 0.01   # Pa*s
}

formulation_dict = {
    'element': 'p2-p1',
    'domain': 'eulerian',
    'time': {
        'unsteady': True,
        'interval': [0.0, 4.0],
        'dt': 0.01
    },
    'bcs': {
        'dirichlet': {
            'velocity': [[0.0, 0.0]],
            'regions': [3],
            'pressure': [0.0],
            'p_regions': [2]
        },
        'neumann': {
            'values': ["1.0 + sin(2.0*pi*t)"],
            'regions': [1],
            'types': ['pressure']
        }
    }
}

config = {
    'mesh': mesh_dict,
    'material': material_dict,
    'formulation': formulation_dict
}

problem = fm.FluidMechanicsProblem(config)
solver = fm.FluidMechanicsSolver(problem, fname_vel="results/v.pvd",
                                 fname_pressure="results/p.pvd")
solver.full_solve()
