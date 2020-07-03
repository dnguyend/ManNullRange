import numpy as np
from numpy import sqrt, zeros, arange
from numpy.random import randn


sqrt2 = sqrt(2)


def sym(A):
    return 0.5*(A+A.T)


def asym(A):
    return 0.5*(A-A.T)


def hsym(A):
    return 0.5*(A+A.T.conjugate())


def ahsym(A):
    return 0.5*(A-A.T.conjugate())


def check_zero(mat):
    return np.sum(np.abs(mat))


def vech(mat):
    """ dealing with real matrices
    """
    p = len(mat)
    ret = mat.take(_triu_indices(p))
    ret2 = ret*sqrt2
    ret2[triu_diag_vech_indices(p)] = ret[triu_diag_vech_indices(p)]
    return ret2


def _tril_indices(n):
    rows, cols = np.tril_indices(n)
    return rows * n + cols


def _triu_indices(n):
    rows, cols = np.triu_indices(n)
    return rows * n + cols


def _diag_indices(n):
    rows, cols = np.diag_indices(n)
    return rows * n + cols


def unvech(v):
    # quadratic formula, correct fp error
    rows = .5 * (-1 + sqrt(1 + 8 * len(v)))
    rows = int(np.round(rows))

    result = zeros((rows, rows))
    result[np.triu_indices(rows)] = v
    result = (result + result.T)/sqrt2

    # divide diagonal elements by 2
    result[np.diag_indices(rows)] /= sqrt2

    return result


def triu_diag_vech_indices(p):
    rp = arange(p)
    return rp * p - (rp * (rp - 1)) // 2


def unvecah(v, hermitian=False):
    rows = .5 * (1 + sqrt(1 + 8 * len(v)))
    rows = int(np.round(rows))

    result = np.zeros((rows, rows))
    result[np.triu_indices(rows, 1)] = v
    if hermitian:
        result = (result - result.T.conjugate())/sqrt2
    else:
        result = (result - result.T)/sqrt2
    return result


def vecah(mat):
    p = len(mat)
    rows, cols = np.triu_indices(p, 1)
    ret = mat.take(rows*p+cols)
    return ret*sqrt2


def cvecah(mat):
    p = len(mat)
    rows, cols = np.triu_indices(p, 1)
    retc = mat.take(rows*p+cols)*sqrt2
    ret = np.concatenate([retc.real, retc.imag, np.diagonal(mat).imag], axis=0)
    return ret


def cvech(mat):
    """ dealing with complex matrices
    lay out is real of upper, complex of upper
    then real of diagonal
    """
    p = len(mat)
    rows, cols = np.triu_indices(p, 1)
    retc = mat.take(rows*p+cols)*sqrt2
    ret = np.concatenate([retc.real, retc.imag, np.diagonal(mat).real], axis=0)
    # ret2[triu_diag_vech_indices(p)] = ret[triu_diag_vech_indices(p)]
    return ret


def cunvech(v):
    """ realvector of dimenion p^2 - make to a complex hermitian matrix
    shape (p, p)
    """
    # quadratic formula, correct fp error
    rows = sqrt(v.shape[0])
    rows = int(np.round(rows))

    result = zeros((rows, rows), dtype=np.complex)
    usize = rows*(rows-1)//2
    result[np.triu_indices(rows, 1)] = v[:usize] + 1j*v[usize:2*usize]
    result = (result + result.T.conjugate())/sqrt2

    # diagonal
    result[np.diag_indices(rows)] = v[-rows:]
    return result


def cunvecah(v):
    """ realvector of dimenion p^2 - make to a complex hermitian matrix
    shape (p, p)
    """
    # quadratic formula, correct fp error
    rows = sqrt(v.shape[0])
    rows = int(np.round(rows))

    result = zeros((rows, rows), dtype=np.complex)
    usize = rows*(rows-1)//2
    result[np.triu_indices(rows, 1)] = v[:usize] + 1j*v[usize:2*usize]
    result = (result - result.T.conjugate())/sqrt2

    # diagonal
    result[np.diag_indices(rows)] = v[-rows:]*1j
    return result


def crandn(*args):
    return randn(*args) +\
        1.j*randn(*args)


def cunvec(avec, ashape):
    """unpack a real vector to a complex matrix of shape ashape
    """
    t_size = avec.shape[0] // 2
    if t_size != np.prod(ashape):
        print("bad size")
        raise(ValueError('bad size %d != ! %d' % (t_size != np.prod(ashape))))
    return avec[:t_size].reshape(ashape) + 1j*avec[t_size:].reshape(ashape)


def cvec(mat):
    t_size = np.prod(mat.shape)
    vec = zeros(t_size*2, dtype=np.float)
    vec[:t_size] = mat.reshape(-1).real
    vec[t_size:] = mat.reshape(-1).imag
    return vec
