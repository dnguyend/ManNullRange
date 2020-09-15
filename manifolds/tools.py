import numpy as np
import numpy.linalg as la
from numpy import sqrt, zeros, arange, trace
from numpy.random import randn


sqrt2 = sqrt(2)


def sym(A):
    return 0.5*(A+A.T)


def asym(A):
    return 0.5*(A-A.T)


def hsym(A):
    return 0.5*(A+A.T.conj())


def ahsym(A):
    return 0.5*(A-A.T.conj())


def check_zero(mat):
    return np.sum(np.abs(mat))


def vech(mat):
    """ Vectorize a Symmetric Matrix to a real vector
    sqrt2*upper triangular part concatenate with diagonal
    This is compatible with the trace(a@b) metric

    Parameters
    ----------
    mat  : A hermitian matrix
    Returns
    ----------
    a real vector
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
    """ Unvvectorize a symmetric matrix to a real vector
    Undoing the vech operation.
    sqrt2*upper triangular part concatenate with diagonal
    This is compatible with the trace(a@b) metric

    Parameters
    ----------
    v  : A vector
    Returns
    ----------
    the symmetric matrix undoing the vech operation
    """
    
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
    """ Unvectorize an antisymmetric matrix to a real vector
    Undoing the vecah operation.
    sqrt2*upper triangular part
    This is compatible with the trace(-a@b) metric

    Parameters
    ----------
    v  : A vector
    Returns
    ----------
    the antisymmetric matrix undoing the vecah operation

    """
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
    """ Vectorize an antisymmetric matrix to a real vector
    sqrt2*upper triangular part
    This is compatible with the trace(-a@b) metric

    Parameters
    ----------
    mat  : An antisymmetric matrix

    Returns
    ----------
    A vector vectorizing the upper half of mat

    """
    
    p = len(mat)
    rows, cols = np.triu_indices(p, 1)
    ret = mat.take(rows*p+cols)
    return ret*sqrt2


def cvecah(mat):
    """Vectorization of an anti-Hermitian matrix to a real vector.
    with scaling to be compatible with the tr(0ab^H) metric

    Parameters
    ----------
    mat  : An anti-Hermitian matrix

    Returns
    ----------
    A result vector
    """
    p = len(mat)
    rows, cols = np.triu_indices(p, 1)
    retc = mat.take(rows*p+cols)*sqrt2
    ret = np.concatenate([retc.real, retc.imag, np.diagonal(mat).imag], axis=0)
    return ret


def cvech(mat):
    """ Vectorizing a Hermitian matrix
    lay out is real of upper, complex of upper
    then real of diagonal

    Parameters
    ----------
    mat  : A Hermitian matrix

    Returns
    ----------
    The result real vector

    """
    p = len(mat)
    rows, cols = np.triu_indices(p, 1)
    retc = mat.take(rows*p+cols)*sqrt2
    ret = np.concatenate([retc.real, retc.imag, np.diagonal(mat).real], axis=0)
    # ret2[triu_diag_vech_indices(p)] = ret[triu_diag_vech_indices(p)]
    return ret


def cunvech(v):
    """ realvector of dimenion p^2 - Undo cvech, form a complex hermitian matrix
    shape (p, p)

    Parameters
    ----------
    v    : a real vector

    Returns
    ----------
    The original matrix

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
    shape (p, p). Undo the scaling to be compatible with trace(-ab^H)

    Parameters
    ----------
    v     : a real vector

    Returns
    ----------
    The original matrix
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
    """ Complex random number. Take *args like randn. Return
    a random complex array or number

    Parameters
    ----------
    *args: args for randn.

    Returns
    ----------
    A random complex array or number
    """
    return randn(*args) + 1.j*randn(*args)


def cunvec(avec, ashape):
    """unpack a real vector to a complex matrix of shape ashape
    Undo the operation in cvec
    """
    t_size = avec.shape[0] // 2
    if t_size != np.prod(ashape):
        print("bad size")
        raise(ValueError('bad size %d != ! %d' % (t_size != np.prod(ashape))))
    return avec[:t_size].reshape(ashape) + 1j*avec[t_size:].reshape(ashape)


def cvec(mat):
    """
    Vectorize a complex matrix to a real vector. First half of the
    vector is the real part of the matrix, second part in the imaginary part

    Parameters
    ----------
    *args: args for randn.

    Returns
    ----------
    The result vector
    """
    t_size = np.prod(mat.shape)
    vec = zeros(t_size*2, dtype=np.float)
    vec[:t_size] = mat.reshape(-1).real
    vec[t_size:] = mat.reshape(-1).imag
    return vec


def complex_base_change(z, lbd):
    """convert the complex number z = a+bi to a new base
    z = f1 lbd +f0
    """
    return np.array([0, 1/lbd.imag, 1, -lbd.real/lbd.imag]).reshape(2, 2) @\
        np.array([z.real, z.imag])


def QL(M0_phi):
    """ The QL (as opposed to QR decomposition.
    This is not the most efficient way but will do for experimental purpose
    
    """
    from numpy.linalg import qr
    lrk = M0_phi.shape[1]
    P = zeros((lrk, lrk))
    P[arange(lrk), lrk-arange(lrk)-1] = 1
    if M0_phi.dtype is np.float:
        q, r = qr(M0_phi @ P)
        S = P @ la.inv(r) @ P
        W = q @ P
        return S, W, W.T, None
    else:
        q, r = qr(M0_phi @ P)
        S = P @ la.inv(r) @ P
        W = q @ P
        return S, W, W.T.conj(), None
    

def extended_lyapunov(alpha1, beta, P, B, Peig=None, Pevec=None):
    """ solve the Lyapunov-type equation:
    (alpha-2beta)X + \beta(PXP^{-1} + P^{-1}XP = B
    Peig and Pevec are precomputed eigenvalue decomposition of P

    Parameters
    ----------
    alpha, beta: scalar coefficients of the equation, positive numbers
    P   : Positive definite matrix coefficient
    B   : the right hand side

    Returns
    ----------
    P

    """

    if Peig is None:
        Peig, Pevec = la.eigh(P)
    evli = 1/Peig
    return Pevec @ ((Pevec.T@B@Pevec) / (beta*(Peig[:, None] * evli[None, :] +
                                         evli[:, None] * Peig[None, :]) +
                                         alpha1-2*beta)) @ Pevec.T


def complex_extended_lyapunov(alpha, beta, P, B, Peig=None, Pevec=None):
    """ solve the Lyapunov-type equation:
    (alpha-2beta)X + \beta(PXP^{-1} + P^{-1}XP = B
    Peig and Pevec are precomputed eigenvalue decomposition of P

    Parameters
    ----------
    alpha, beta: scalar coefficients of the equation, positive numbers
    P   : Positive definite matrix coefficient
    B   : the right hand side

    Returns
    ----------
    P

    """

    if Peig is None:
        Peig, Pevec = la.eigh(P)
    evli = 1/Peig
    return Pevec @ ((Pevec.T.conjugate() @ B @ Pevec) /
                    (beta*(Peig[:, None] * evli[None, :] +
                           evli[:, None] * Peig[None, :]) +
                     alpha-2*beta)) @ Pevec.T.conjugate()


def SMW_inv(X, Y):
    """ Inverse of (I-XY) by Sherman Morrision Woodbury
    With X = [x_1...X_k], Y[y_1^T..y_k^T]^T
    Recursive formula:
    B_0 = I
    B_i = (B_{i-1}^[-1} -(1/(y_iB_{i-1}^{-1}x_i-1)B_{i-1}^{-1}x_iy_iB_{i-1})

    Parameters
    ----------
    X, Y to matrices such that XY is defined

    Returns
    ----------
    (I - XY)^{-1}
    """
    Biv = np.eye(X.shape[0], dtype=X.dtype)
    for i in range(X.shape[1]):
        ft = 1/(np.sum(Y[[i], :] @ (Biv@X[:, [i]]))-1)
        Biv -= ft*(Biv@X[:, i]).reshape(-1, 1)@(Y[i, :]@Biv).reshape(1, -1)
    return Biv


def C0_xy(t, X, Y):
    """Do the base cayley transform

    Parameters
    ----------
    X, Y to matrices such that XY is defined

    Returns
    ----------
    (I - 0.5*t*X@Y)^{-1} (I + 0.5t*X@Y)
    """
    iv = SMW_inv(0.5*t*X, Y)
    return iv + 0.5*t*(iv @ X) @ Y


def Cayley_A_opt_xy(W, eta):
    """Compute A(eta) returning an anti-symmetric
    matrix to evaluate Cayley. A is given as a pair
    of matrices X, Y of lower rank with XY = A(eta)

    Parameters
    ----------
    W     : an orthogonal matrix in row format WW.T = I
            W is of size d m
    eta   : a matrix   satisfies W@eta.T + eta@W.T = 0

    Returns
    ----------
    X, Y such that XY = A(eta)
    X size m d
    """

    ew = eta + 0.5*(W@eta.T)@W
    X = np.concatenate([W.T, -ew.T], axis=1)
    Y = np.concatenate([ew, W], axis=0)
    return X, Y


def Cayley_A_opt_xy_complex(W, eta):
    """Compute A(eta) returning an anti-symmetric
    matrix to evaluate Cayley. A is given as a pair
    of matrices X, Y of lower rank with XY = A(eta)

    Parameters
    ----------
    W     : an orthogonal matrix in row format WW.T = I
            W is of size d m
    eta   : a matrix   satisfies W@eta.T + eta@W.T = 0

    Returns
    ----------
    X, Y such that XY = A(eta)
    X size m d
    """

    ew = eta + 0.5*(W@eta.T.conj())@W
    X = np.concatenate([W.T.conj(), -ew.T.conj()], axis=1)
    Y = np.concatenate([ew, W], axis=0)
    return X, Y


def rtrace(A):
    """ Real part of trace
    """
    return trace(A).real
