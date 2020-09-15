from __future__ import division
from .NullRangeManifold import NullRangeManifold
import numpy.linalg as la
import numpy as np
from numpy import zeros_like, bmat, zeros
from scipy.linalg import expm
from .tools import cvech, cunvech, crandn, rtrace, cvec, cunvec


if not hasattr(__builtins__, "xrange"):
    xrange = range


def _calc_dim(n, d):
    dm = 2*d * (n-d) + d*d
    cdm = d*d
    tdim = 2*n*d
    return dm, cdm, tdim

    
class ComplexStiefel(NullRangeManifold):
    """Class for a Complex Stiefel manifold
    Block matrix Y with Y.T.conj() @ Y = I
    Y of dimension n*d

    Metric is defined by 2 parameters in the array alpha

    Parameters
    ----------
    n, d     : # of rows and columns of the manifold point
    alpha    : array of size 2, alpha  > 0

    """
    
    def __init__(self, n, d, alpha=None):
        self._point_layout = 1
        self.n = n
        self.d = d
        self._dimension, self._codim, _ = _calc_dim(n, d)
        if alpha is None:
            self.alpha = np.array([1, .5])
        else:
            self.alpha = alpha

    def inner(self, X, eta1, eta2=None):
        """ Inner product (Riemannian metric) on the tangent space.
        The tangent space is given as a matrix of size n*d
        """
        alf = self.alpha
        if eta2 is None:
            eta2 = eta1
        return alf[0]*rtrace(eta1.T.conj() @ eta2) + (alf[1]-alf[0]) *\
            rtrace((eta1.T.conj() @ X) @ (X.T.conj() @ eta2))
    
    def __str__(self):
        self._name = "Complex stiefel manifold n=%d d=%d alpha=%s" % (
            self.n, self.d, str(self.alpha))

        return self._name

    def base_inner_ambient(self, eta1, eta2):
        return rtrace(eta1.T.conj() @ eta2)

    def base_inner_E_J(self, a1, a2):
        return rtrace(a1 @ a2.T.conj())
    
    def g(self, X, eta):
        alf = self.alpha
        return alf[0]*eta + (alf[1]-alf[0]) *\
            X @ (X.T.conj() @ eta)

    def g_inv(self, X, ambient):
        ialp = 1/self.alpha
        return ialp[0]*ambient + (ialp[1]-ialp[0]) * X @ (X.T.conj() @ ambient)
    
    def J(self, X, eta):
        return eta.T.conj() @ X + X.T.conj() @ eta

    def Jst(self, X, a):
        return 2*X@a

    def g_inv_Jst(self, X, a):
        return (2/self.alpha[1])*X@a

    def D_g(self, X, xi, eta):
        alf = self.alpha
        return (alf[1]-alf[0]) * (xi @ (X.T.conj() @ eta) +
                                  X @ (xi.T.conj() @ eta))

    def christoffel_form(self, X, xi, eta):
        ret = xi @ X.T.conj() @ eta + eta @ X.T.conj() @ xi
        ret += X @ (xi.T.conj() @ eta + eta.T.conj() @ xi)
        ret -= (xi @ eta.T.conj() + eta @ xi.T.conj()) @ X
        return 0.5*(self.alpha[1]-self.alpha[0]) * ret

    def D_J(self, X, xi, eta):
        return eta.T.conj() @ xi + xi.T.conj() @ eta
    
    def D_Jst(self, X, xi, a):
        return 2*xi@a

    def D_g_inv_Jst(self, X, xi, a):
        return (2/self.alpha[1])*xi@a
    
    def contract_D_g(self, X, xi, eta):
        alf = self.alpha
        return (alf[1] - alf[0])*(eta @ xi.T.conj() + xi @ eta.T.conj()) @ X
    
    def st(self, mat):
        """The split_transpose. transpose if real, hermitian transpose if complex
        """
        return mat.T.conj()

    def J_g_inv_Jst(self, X, a):
        return 4/self.alpha[1]*a

    def solve_J_g_inv_Jst(self, X, b):
        """ base is use CG. Unlikely to use
        """
        return self.alpha[1]/4*b
    
    def proj(self, X, U):
        """projection. U is in ambient
        return one in tangent
        """
        UTX = U.T.conj() @ X
        return U - 0.5*X @ (UTX + UTX.T.conj())

    def proj_g_inv(self, X, U):
        ret = zeros_like(X, dtype=np.complex)
        ialp = 1/self.alpha
        ret = ialp[0] * U
        ret += 0.5*(ialp[1]-2*ialp[0]) * X @ (X.T.conj() @ U)
        ret -= 0.5*ialp[1]*X @ (U.T.conj() @ X)
        return ret

    def zerovec(self, X):
        return zeros_like(X, dtype=np.complex)

    def egrad2rgrad(self, X, U):
        return self.proj_g_inv(X, U)

    def rhess02_alt(self, X, xi, eta, egrad, ehess_val):
        """ Ehess is the Hessian Vector Product
        """
        alpha = self.alpha
        etaxiy = xi @ (eta.T.conj()@X) + eta@(xi.T.conj()@X)
        egcoef = 0.5*X @ (xi.T.conj()@eta + eta.T.conj()@xi)
        ft = (alpha[0]-alpha[1])/alpha[0]
        egcoef += ft*(etaxiy - X@(X.T.conj()@etaxiy))
        return ehess_val - rtrace(egcoef @ egrad.T.conj())
    
    def ehess2rhess(self, X, egrad, ehess, H):
        """ Convert Euclidean into Riemannian Hessian.
        ehess is the Hessian product on the ambient space
        egrad is the gradient on the ambient space
        Formula would be:
        project of ehess -\
          (gradient (self.st(JJ)) H) @ ((JJ @ self.st(JJ))^{-1}) @ JJ @ egrad)
        """
        alp = self.alpha
        egrady = egrad.T.conj() @ X
        grad_part = 0.5*H@(egrady+egrady.T.conj())
        egyxi = egrad@X.T.conj()@H
        xiproj = H - X@(X.T.conj()@H)
        grad_part += (1-alp[1]/alp[0])*(egyxi-X@(X.T.conj()@egyxi))
        grad_part += (1-alp[1]/alp[0])*X@(egrad.T.conj()@xiproj)
        return self.proj_g_inv(X, ehess-grad_part)

    def retr(self, X, eta):
        """ svd
        """
        u, _, vh = la.svd(X+eta, full_matrices=False)
        return u @ vh

    def norm(self, X, eta):
        # Norm on the tangent space is simply the Euclidean norm.
        return np.sqrt(self.inner(X, eta, eta))

    def rand(self):
        # Generate random  point using qr of random normally distributed
        # matrix.
        O, _ = la.qr(crandn(
            self.n, self.d))
        return O
    
    def randvec(self, X):
        U = self.proj(X, crandn(*X.shape))
        U = U / self.norm(X, U)
        return U

    def _rand_ambient(self):
        return crandn(self.n, self.d)

    def _rand_range_J(self):
        u = crandn(self.d, self.d)
        return u + u.T.conj()

    def _vec(self, E):
        return cvec(E)

    def _unvec(self, vec):
        return cunvec(vec, (self.n, self.d))

    def _vec_range_J(self, a):
        return cvech(a)

    def _unvec_range_J(self, vec):
        return cunvech(vec)

    def exp(self, Y, eta):
        """Geodesic from Y in direction eta

        Parameters
        ----------
        Y    : a manifold point
        eta  : tangent vector
        
        Returns
        ----------
        gamma(1), where gamma(t) is the geodesics at Y in direction eta

        """

        K = eta - Y @ (Y.T.conj() @ eta)
        Yp, R = la.qr(K)
        alf = self.alpha[1]/self.alpha[0]
        A = Y.T.conj() @eta
        x_mat = bmat([[2*alf*A, -R.T.conj()],
                      [R, zeros((self.d, self.d), dtype=np.complex)]])
        return bmat([Y, Yp]) @ expm(x_mat)[:, :self.d] @ \
            expm((1-2*alf)*A)

    def exp_alt(self, Y, eta):
        """ Alternative geodesics formula
        """
        alf = self.alpha[1]/self.alpha[0]
        A = Y.T.conj() @ eta
        e_mat = bmat([[(2*alf-1)*A, -eta.T.conj()@eta - 2*(1-alf)*A@A],
                      [np.eye(self.d), A]])
        return (bmat([Y, eta]) @ expm(e_mat))[:, :self.d] @ expm((1-2*alf)*A)
