from numpy.random import randn
import numpy as np


def make_sym_pos(n):
    u = np.random.randn(n, n)
    return u @ u.T
    

def make_diag_orth(dd):
    q, _ = np.linalg.qr(randn(dd, dd))
    return q


def check_zero(mat):
    return np.sum(np.abs(mat))


def random_orthogonal(k):
    """Generate a random orthogonal matrix of size (k, k)
    real matrix based on the paper
    How to generate random matrices from the classical compact groups
    example:
    O = random_orthogonal(3)
    print O
    [[ 0.25452591 -0.92275001  0.28939416]
    [ 0.96115429  0.27441539  0.0296417 ]
    [-0.10676609  0.27060785  0.95675096]]

    np.dot(O, O.T)
    array([[ 1.00000000e+00, -1.21314803e-16, -2.59623113e-18],
       [-1.21314803e-16,  1.00000000e+00,  3.20641969e-17],
       [-2.59623113e-18,  3.20641969e-17,  1.00000000e+00]])
    """
    z = randn(k, k) / np.sqrt(2.)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.abs(d)
    q = np.multiply(q, ph, q)
    return q
