from __future__ import division
from .NullRangeManifold import NullRangeManifold
import numpy.linalg as la
import numpy as np
from numpy import trace, zeros_like, bmat, zeros
from numpy.random import randn
from scipy.linalg import expm
from . import tools


if not hasattr(__builtins__, "xrange"):
    xrange = range


def _calc_dim(n, d):
    dm = d * (n-d) + d*(d-1)//2
    cdm = d*(d+1) // 2
    tdim = n*d
    return dm, cdm, tdim

    
class RealStiefel(NullRangeManifold):
    def __init__(self, n, d, alpha=None):
        self._point_layout = 1
        self.n = n
        self.d = d
        self._dim, self._codim, _ = _calc_dim(n, d)
        if alpha is None:
            self.alpha = np.array([1, .5])
        else:
            self.alpha = alpha

    def inner_product_amb(self, X, eta1, eta2=None):
        alf = self.alpha
        if eta2 is None:
            eta2 = eta1
        return alf[0]*trace(eta1.T @ eta2) + (alf[1]-alf[0]) *\
            trace((eta1.T @ X) @ (X.T @ eta2))
    
    def __str__(self):
        return "real_stiefel manifold n=%d d=%d alpha=str(alpha)" % (
            self.n, self.d, str(self.alpha))

    def base_inner_ambient(X, eta1, eta2):
        return trace(eta1.T @ eta2)

    def base_inner_E_J(X, a1, a2):
        return trace(a1 @ a2.T)
    
    def g(self, X, eta):
        alf = self.alpha
        return alf[0]*eta + (alf[1]-alf[0]) *\
            X @ (X.T @ eta)

    def g_inv(self, X, ambient):
        ialp = 1/self.alpha
        return ialp[0]*ambient + (ialp[1]-ialp[0]) * X @ (X.T @ ambient)
    
    def J(self, X, eta):
        return eta.T @ X + X.T @ eta

    def Jst(self, X, a):
        return 2*X@a

    def g_inv_Jst(self, X, a):
        return (2/self.alpha[1])*X@a

    def D_g(self, X, xi, eta):
        alf = self.alpha
        return (alf[1]-alf[0]) * (xi @ (X.T @ eta) + X @ (xi.T @ eta))

    def christoffel_form(self, X, xi, eta):
        ret = xi @ X.T @ eta + eta @ X.T @ xi
        ret += X @ (xi.T @ eta + eta.T @ xi)
        ret -= (xi @ eta.T + eta @ xi.T) @ X
        return 0.5*(self.alpha[1]-self.alpha[0]) * ret

    def D_J(self, X, xi, eta):
        return eta.T @ xi + xi.T @ eta
    
    def D_Jst(self, X, xi, a):
        return 2*xi@a

    def D_g_inv_Jst(self, X, xi, a):
        return (2/self.alpha[1])*xi@a
    
    def contract_D_g(self, X, xi, eta):
        alf = self.alpha
        return (alf[1] - alf[0])*(eta @ xi.T + xi @ eta.T) @ X
    
    def inner(self, X, G, H):
        """ Inner product (Riemannian metric) on the tangent space.
        The tangent space is given as a matrix of size mm_degree * m
        """
        # return inner_product_tangent
        return self.base_inner_ambient(self.g(X, G), H)

    def st(self, mat):
        """The split_transpose. transpose if real, hermitian transpose if complex
        """
        return mat.T

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
        UTX = U.T @ X
        return U - 0.5*X @ (UTX + UTX.T)

    def proj_g_inv(self, X, U):
        ret = zeros_like(X)
        ialp = 1/self.alpha
        ret = ialp[0] * U
        ret += 0.5*(ialp[1]-2*ialp[0]) * X @ (X.T @ U)
        ret -= 0.5*ialp[1]*X @ (U.T @ X)
        return ret

    def zerovec(self, X):
        return zeros_like(X)

    def egrad2rgrad(self, X, U):
        return self.proj_g_inv(X, U)

    def rhess02_alt(self, X, xi, eta, egrad, ehess_val):
        """ Ehess is the Hessian Vector Product
        """
        alpha = self.alpha
        etaxiy = xi @ (eta.T@X) + eta@(xi.T@X)
        egcoef = 0.5*X @ (xi.T@eta + eta.T@xi)
        ft = (alpha[0]-alpha[1])/alpha[0]
        egcoef += ft*(etaxiy - X@(X.T@etaxiy))
        return ehess_val - trace(egcoef @ egrad.T)
    
    def ehess2rhess(self, X, egrad, ehess, H):
        """ Convert Euclidean into Riemannian Hessian.
        ehess is the Hessian product on the ambient space
        egrad is the gradient on the ambient space
        Formula would be:
        project of ehess -\
          (gradient (self.st(JJ)) H) @ ((JJ @ self.st(JJ))^{-1}) @ JJ @ egrad)
        """
        alp = self.alpha
        egrady = egrad.T @ X
        grad_part = 0.5*H@(egrady+egrady.T)
        egyxi = egrad@X.T@H
        xiproj = H - X@(X.T@H)
        grad_part += (1-alp[1]/alp[0])*(egyxi-X@(X.T@egyxi))
        grad_part += (1-alp[1]/alp[0])*X@(egrad.T@xiproj)
        return self.proj_g_inv(X, ehess-grad_part)

    def retr(self, X, eta):
        """ Calculate 'thin' qr decomposition of X + G
        then add point X
        then do thin lq decomposition
        """
        u, _, vh = la.svd(X+eta, full_matrices=False)
        return u @ vh

    def norm(self, X, eta):
        # Norm on the tangent space is simply the Euclidean norm.
        return np.sqrt(self.inner(X, eta, eta))

    def rand(self):
        # Generate random  point using qr of random normally distributed
        # matrix.
        O, _ = la.qr(randn(
            self.n, self.d))
        return O
    
    def randvec(self, X):
        U = self.proj(X, randn(*X.shape))
        U = U / self.norm(X, U)
        return U

    def _rand_ambient(self):
        return randn(self.n, self.d)

    def _rand_range_J(self):
        u = randn(self.d, self.d)
        return u + u.T

    def _vec(self, E):
        return E.reshape(-1)

    def _unvec(self, vec):
        return vec.reshape(self.n, self.d)

    def _vec_range_J(self, a):
        return tools.vech(a)

    def _unvec_range_J(self, vec):
        return tools.unvech(vec)

    def exp(self, Y, eta):
        # geodesic
        K = eta - Y @ (Y.T @ eta)
        Yp, R = la.qr(K)
        alf = self.alpha[1]/self.alpha[0]
        A = Y.T @eta
        x_mat = bmat([[2*alf*A, -R.T],
                      [R, zeros((self.d, self.d))]])
        return bmat([Y, Yp]) @ expm(x_mat)[:, :self.d] @ \
            expm((1-2*alf)*A)

    def exp_alt(self, Y, eta):
        alf = self.alpha[1]/self.alpha[0]
        A = Y.T @ eta
        e_mat = bmat([[(2*alf-1)*A, -eta.T@eta - 2*(1-alf)*A@A],
                      [np.eye(self.d), A]])
        return (bmat([Y, eta]) @ expm(e_mat))[:, :self.d] @ expm((1-2*alf)*A)
