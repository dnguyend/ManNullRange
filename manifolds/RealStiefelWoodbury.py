from __future__ import division
# from pymanopt.manifolds.manifold import Manifold
from .NullRangeManifold import NullRangeManifold
import numpy.linalg as la
import numpy as np
from numpy import trace, zeros_like
from numpy.random import randn
from . import tools


if not hasattr(__builtins__, "xrange"):
    xrange = range


def _calc_dim(n, d):
    dm = d * (n-d) + d*(d-1)//2
    cdm = d*(d+1) // 2
    tdim = n*d
    return dm, cdm, tdim

    
class RealStiefelWoodbury(NullRangeManifold):
    """Stiefel with metric Trace(eta.T@A@eta) + Trace(X@B@X.T@eta@eta.T)
    use CG. Not the most efficient but to show the framework handles it

    Parameters
    ----------
    n, d     : # of rows and columns of the manifold point
    A        : matrix of size n*n
    B        : matrix of size d*d
    """
    def __init__(self, n, d, A, B):
        self._point_layout = 1
        self.n = n
        self.d = d
        self._dim, self._codim, _ = _calc_dim(n, d)
        self.A = A
        self.B = B
        self.retr_method = 'geo'
        
    def inner(self, X, eta1, eta2=None):
        A, B = (self.A, self.B)
        if eta2 is None:
            eta2 = eta1
        return trace(eta1.T @ A @ eta2) +\
            trace(X @ B @ X.T @ eta1 @ eta2.T)
    
    def __str__(self):
        return "real_stiefel manifold with Woodbury metric n=%d d=%d" % (
            self.n, self.d)

    def zerovec(self, X):
        return zeros_like(X)

    def base_inner_ambient(X, eta1, eta2):
        return trace(eta1.T @ eta2)

    def base_inner_E_J(X, a1, a2):
        return trace(a1 @ a2.T)
    
    def g(self, X, eta):
        A, B = (self.A, self.B)
        return (A + X @ B @ X.T) @ eta

    def g_inv(self, X, ambient):
        A, B = (self.A, self.B)
        asol = la.solve(A, ambient)
        xsol = la.solve(A, X)
        second = xsol @ la.solve(la.inv(B) + X.T @ xsol, X.T @ asol)
        return asol - second
    
    def J(self, X, eta):
        return eta.T @ X + X.T @ eta

    def Jst(self, X, a):
        return 2*X@a

    def D_g(self, X, xi, eta):
        B = self.B
        return xi @ B @ (X.T @ eta) + X @ B @ (xi.T @ eta)

    def D_J(self, X, xi, eta):
        return eta.T @ xi + xi.T @ eta
    
    def D_Jst(self, X, xi, a):
        return 2*xi@a
    
    def contract_D_g(self, X, xi, eta):
        B = self.B
        return xi @ (eta.T@X) @ B + eta @ (xi.T@X) @ B
    
    def _inner_(self, X, G, H):
        """ Inner product (Riemannian metric) on the tangent space.
        The tangent space is given as a matrix of size mm_degree * m
        """
        # return inner_product_tangent
        return self.base_inner_ambient(self.g(X, G), H)

    def st(self, mat):
        """The split_transpose. transpose if real, hermitian transpose if complex
        """
        return mat.T
        
    def rhess02_alt(self, X, xi, eta, egrad, ehess_val):
        """ optional
        """
        try:
            g_inv_solve_J_g_in_Jst_DJ = self.g_inv(
                X, self.Jst(
                    X, self.solve_J_g_inv_Jst(X, self.D_J(X, xi, eta))))
            proj_christoffel = self.proj_g_inv(
                X, self.christoffel_form(X, xi, eta))
            return ehess_val - self.base_inner_ambient(
                g_inv_solve_J_g_in_Jst_DJ + proj_christoffel, egrad)
        except Exception as e:
            raise(RuntimeError("%s if D_J is not implemeted try rhess02" % e))
    
    def retr_svd(self, X, eta):
        """ Calculate 'thin' qr decomposition of X + G
        then add point X
        then do thin lq decomposition
        """
        u, _, vh = la.svd(X+eta, full_matrices=False)
        return u @ vh

    def Gamma(self, X, xi, eta):
        g_inv_solve_J_g_in_Jst_DJ = self.g_inv(
            X, self.Jst(
                X, self.solve_J_g_inv_Jst(X, self.D_J(X, xi, eta))))
        proj_christoffel = self.proj_g_inv(
            X, self.christoffel_form(X, xi, eta))
        return g_inv_solve_J_g_in_Jst_DJ + proj_christoffel
    
    def retr_geo(self, X, eta):
        """ SVD using geodesics exponential
        equation is X + eta - 0.5 *gamma(eta, eta)
        """
        G = self.Gamma(X, eta, eta)
        t = 1
        nn = self.norm(X, eta)
        if nn > 1:
            eta = eta / nn
        while True:
            # import pdb
            # pdb.set_trace()
            ret = X + t*eta - 0.5*t*t*G
            rtio = np.mean(np.abs(ret.T @ ret - X.T@X))
            if rtio < 1e-6:
                break
            else:
                t *= min(1e-6/rtio, .9)
                print('t=%f rtio=%f norm=%f' % (t, rtio, nn))
        return ret

    def retr(self, X, eta):
        if self.retr_method.lower() == 'svd':
            return self.retr_svd(X, eta)
        else:
            return self.retr_geo(X, eta)
    
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
