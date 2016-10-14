import numpy as np
import scipy.sparse as sp

# Function to load sparse matrix from file
def load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape'])

def csr_equal(A1, A2, tol=1e-16):

    # Make sure all zero entries are removed
    A1.eliminate_zeros()
    A2.eliminate_zeros()

    if not A1.nnz == A2.nnz:
        flag = False
    elif not A1.shape == A2.shape:
        flag = False
    elif not np.all(A1.indptr == A2.indptr):
        flag = False
    elif not np.all(A1.indices == A2.indices):
        flag = False
    else:
        try:
            flag = np.all(np.abs(A1.data - A2.data) < tol)
        except ValueError:
            flag = False

    return flag

print 'Loading A_neo...'
A_neo = load_sparse_csr('numpy_data/stiffness-comp_neo_hookean-12x12x12.npz')
print '...[DONE]'

print 'Loading b_neo...'
b_neo = np.fromfile('numpy_data/load-comp_neo_hookean-12x12x12.npz')
print '...[DONE]'

print 'Loading A_lin...'
A_lin = load_sparse_csr('numpy_data/stiffness-comp_lin_elastic-12x12x12.npz')
print '...[DONE]'

print 'Loading b_lin...'
b_lin = np.fromfile('numpy_data/load-comp_lin_elastic-12x12x12.npz')
print '...[DONE]'

tol = 1e-16

print 'Stiffness matrices the same? ', csr_equal(A_neo, A_lin, tol=tol)
print 'Load vectors the same? ', np.all(np.abs(b_neo - b_lin) < tol)

############################################################
from scipy.sparse.linalg import spsolve

x_neo = spsolve(A_neo, b_neo)
x_lin = spsolve(A_lin, b_lin)
