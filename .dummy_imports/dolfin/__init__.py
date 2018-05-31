class Dummy_mpi:
    def __init__(self):
        pass
    def rank(*args):
        return 0

MPI = Dummy_mpi()
parameters = {
    'form_compiler': {
        'cpp_optimize': None,
        'representation': None,
        'quadrature_degree': None,
        'optimize': None
    }
}
