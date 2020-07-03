from __future__ import division
# from pymanopt.manifolds.manifold import Manifold
from .NullRangeManifold import NullRangeManifold
import numpy.linalg as la
import numpy as np
from numpy import trace
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
    """Stiefel with metric Trace((AH+XBX.TH)G)
    """
    def __init__(self, n, d, A, B):
        self.n = n
        self.d = d
        self._dim, self._codim, _ = _calc_dim(n, d)
        self.A = A
        self.B = B
        
    def inner_product_amb(self, X, eta1, eta2=None):
        A, B = (self.A, self.B)
        if eta2 is None:
            eta2 = eta1
        return trace(eta1.T @ A @ eta2) +\
            trace(X @ B @ X.T @ eta1 @ eta2.T)
    
    def __str__(self):
        return "real_stiefel manifold n=%d d=%d alpha=str(alpha)" % (
            self.n, self.d, str(self.alpha))

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

    def g_inv_Jst(self, X, eta):
        return self.g_inv(X, self.Jst(X, eta))

    def D_g(self, X, xi, eta):
        B = self.B
        return xi @ B @ (X.T @ eta) + X @ B @ (xi.T @ eta)

    def christoffel_form(self, X, xi, eta):
        ret = 0.5*self.D_g(X, xi, eta)
        ret += 0.5*self.D_g(X, eta, xi)
        ret -= 0.5*self.contract_D_g(X, xi, eta)
        return ret

    def D_J(self, X, xi, eta):
        return eta.T @ xi + xi.T @ eta
    
    def D_Jst(self, X, xi, a):
        return 2*xi@a

    def D_g_inv_Jst(self, X, xi, a):
        djst = self.D_Jst(X, xi, a)
        return self.g_inv(
            X, -self.D_g(X, xi, self.g_inv(X, self.Jst(X, a))) + djst)
    
    def contract_D_g(self, X, xi, eta):
        B = self.B
        return xi @ (eta.T@X) @ B + eta @ (xi.T@X) @ B
    
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
        return self.J(X, self.g_inv_Jst(X, a))

    def solve_J_g_inv_Jst(self, X, b):
        """ base is use CG. Unlikely to use
        """
        from scipy.sparse.linalg import cg, LinearOperator
        
        def Afunc(a):
            return self._vec_range_J(
                self.J_g_inv_Jst(X, self._unvec_range_J(a)))
        A = LinearOperator(
            dtype=float, shape=(self._codim, self._codim), matvec=Afunc)
        res = cg(A, self._vec_range_J(b))
        return self._unvec_range_J(res[0])
        
    def proj(self, X, U):
        """projection. U is in ambient
        return one in tangent
        """
        return U - self.g_inv_Jst(
            X, self.solve_J_g_inv_Jst(X, self.J(X, U)))

    def proj_g_inv(self, X, U):
        return self.proj(X, self.g_inv(X, U))

    def egrad2rgrad(self, X, U):
        return self.proj_g_inv(X, U)

    def rhess02(self, X, xi, eta, egrad, ehess):
        """ Ehess is the Hessian Vector Product
        """
        return self.base_inner_ambient(
            self.ehess2rhess(X, egrad, ehess, xi))
    
    def rhess02_alt(self, X, xi, eta, egrad, ehess_val):
        """ optional
        """
        try:
            g_inv_solve_J_g_in_Jst_DJ = self.g_inv(
                X, self.solve_J_g_inv_Jst(
                    X, self.D_J(X, xi, eta)))
            proj_christoffel = self.proj_g_inv(
                self.christofel_form(X, xi, eta))
            return ehess_val - self.base_inner_ambient(
                (g_inv_solve_J_g_in_Jst_DJ + proj_christoffel, egrad))
        except Exception as e:
            raise(RuntimeError("%s if D_J is not implemeted try rhess02" % e))
    
    def ehess2rhess(self, X, egrad, ehess, H):
        """ Convert Euclidean into Riemannian Hessian.
        ehess is the Hessian product on the ambient space
        egrad is the gradient on the ambient space
        Formula would be:
        project of ehess -\
          (gradient (self.st(JJ)) H) @ ((JJ @ self.st(JJ))^{-1}) @ JJ @ egrad)
        """
        first = ehess
        a = self.J(X, self.g_inv(X, egrad))
        rgrad = self.proj_g_inv(X, egrad)
        second = self.D_g(X, H, self.g_inv(X, egrad))
        aout = self.solve_J_g_inv_Jst(X, a)
        third = self.proj(X, self.D_g_inv_Jst(X, H, aout))
        fourth = self.christoffel_form(X, H, rgrad)
        return self.proj_g_inv(X, (first - second) + fourth) - third

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

    def _vec(self, X):
        return X.reshape(-1)

    def _unvec(self, vec):
        return vec.reshape(self.n, self.d)

    def _vec_range_J(self, a):
        return tools.vech(a)

    def _unvec_range_J(self, vec):
        return tools.unvech(vec)
