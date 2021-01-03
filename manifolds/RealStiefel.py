from __future__ import division
from .NullRangeManifold import NullRangeManifold
import numpy.linalg as la
import numpy as np
from numpy import trace, zeros_like, bmat, zeros
from numpy.random import randn
from scipy.linalg import expm, expm_frechet, logm, null_space
from scipy.optimize import minimize
from .tools import vech, unvech, vecah, unvecah, asym


if not hasattr(__builtins__, "xrange"):
    xrange = range


def _calc_dim(n, d):
    dm = d * (n-d) + d*(d-1)//2
    cdm = d*(d+1) // 2
    tdim = n*d
    return dm, cdm, tdim

    
class RealStiefel(NullRangeManifold):
    """Class for a Real Stiefel manifold
    Block matrix Y with Y.T @ Y = I
    Y of dimension n*d

    Metric is defined by 2 parameters in the array alpha

    Parameters
    ----------
    n, d         : # of rows and columns of the manifold point
    alpha        : array of size 2, alpha  > 0
    log_callback : print out progress when running log
    log_method   : None is trust-krylov. Otherwise
                   one of trust-ncg or trust-krylov
    """
    
    def __init__(self, n, d, alpha=None, log_stats=False,
                 log_method=None):
        self._point_layout = 1
        self.n = n
        self.d = d
        self._name = "Real_stiefel manifold n=%d d=%d alpha=%s" % (
            self.n, self.d, str(alpha))
        self._dimension, self._codim, _ = _calc_dim(n, d)
        if alpha is None:
            self.alpha = np.array([1, .5])
        else:
            self.alpha = alpha
        self.log_stats = log_stats
        if log_method is None:
            self.log_method = 'trust-krylov'
        elif log_method.lower() in ['trust-ncg', 'trust-krylov', 'l-bfgs-b']:
            self.log_method = log_method.lower()
        else:
            raise(ValueError(
                'log method must be one of trust-ncg or trust-krylov'))
        self.log_gtol = None

    def inner(self, X, eta1, eta2=None):
        """ Inner product (Riemannian metric) on the tangent space.
        The tangent space is given as a matrix of size n*d
        """
        alf = self.alpha
        if eta2 is None:
            eta2 = eta1
        return alf[0]*trace(eta1.T @ eta2) + (alf[1]-alf[0]) *\
            trace((eta1.T @ X) @ (X.T @ eta2))
    
    def __str__(self):
        return self._name

    def base_inner_ambient(self, eta1, eta2):
        return trace(eta1.T @ eta2)

    def base_inner_E_J(self, a1, a2):
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
        return vech(a)

    def _unvec_range_J(self, vec):
        return unvech(vec)

    def exp(self, Y, eta):
        """ Geodesics, the formula involves matrices of size 2d

        Parameters
        ----------
        Y    : a manifold point
        eta  : tangent vector
        
        Returns
        ----------
        gamma(1), where gamma(t) is the geodesics at Y in direction eta

        """
        K = eta - Y @ (Y.T @ eta)
        Yp, R = la.qr(K)
        alf = self.alpha[1]/self.alpha[0]
        A = Y.T @eta
        x_mat = bmat([[2*alf*A, -R.T],
                      [R, zeros((self.d, self.d))]])
        return np.array(
            bmat([Y, Yp]) @ expm(x_mat)[:, :self.d] @ expm((1-2*alf)*A))

    def exp_alt(self, Y, eta):
        """ Geodesics, alternative formula
        """
        alf = self.alpha[1]/self.alpha[0]
        A = Y.T @ eta
        e_mat = bmat([[(2*alf-1)*A, -eta.T@eta - 2*(1-alf)*A@A],
                      [np.eye(self.d), A]])
        return np.array(
            (bmat([Y, eta]) @ expm(e_mat))[:, :self.d] @ expm((1-2*alf)*A))

    def dist(self, X, Y):
        lg = self.log(X, Y, show_steps=False, init_type=1)
        return self.norm(X, lg)

    def log(self, Y, Y1, show_steps=False, init_type=1):
        """Inverse of exp

        Parameters
        ----------
        Y    : a manifold point
        Y1  : tangent vector

        Returns
        ----------
        eta such that self.exp(X, eta) = Y1

        Algorithm: use the scipy.optimize trust region method
        to minimize in eta ||self.exp(Y, eta) - Y1||_F^2
        _F is the Frobenius norm in R^{n\times d}
        The jacobian could be computed by the expm_frechet function
        """
        alf = self.alpha[1]/self.alpha[0]
        d = self.d
        adim = (d*(d-1))//2

        def getQ():
            """ algorithm: find a basis in linear span of Y Y1
            orthogonal to Y
            """
            u, s, v = np.linalg.svd(
                np.concatenate([Y, Y1], axis=1), full_matrices=False)
            k = (s > 1e-14).sum()
            good = u[:, :k]@v[:k, :k]
            qs = null_space(Y.T@good)
            Q, _ = np.linalg.qr(good@qs)
            return Q
        
        # Q, s, _ = la.svd(Y1 - Y@Y.T@Y1, full_matrices=False)
        # Q = Q[:, :np.sum(np.abs(s) > 1e-14)]
        Q = getQ()
        k = Q.shape[1]
        if k == 0:
            # Y1 and Y has the same linear span
            A = logm(Y.T @ Y1)

            if self.log_stats:
                return Y@A, [('success', True), ('message', 'aligment')]
            return Y@A
        
        def vec(A, R):
            return np.concatenate(
                [vecah(A), R.reshape(-1)])

        def unvec(avec):
            return unvecah(avec[:adim]), avec[adim:].reshape(k, d)

        def dist(v):
            A, R = unvec(v)
            ex2 = expm(
                np.array(
                    bmat([[2*alf*A, -R.T], [R, np.zeros((k, k))]])))
            M = ex2[:d, :d]
            N = ex2[d:, :d]

            return -np.trace(Y1.T@(Y@M+Q@N)@expm((1-2*alf)*A))

        def jac(v):
            A, R = unvec(v)
            ex1 = expm((1-2*alf)*A)

            mat = np.array(bmat([[2*alf*A, -R.T], [R, np.zeros((k, k))]]))
            E = np.array(bmat(
                [[ex1@Y1.T@Y, ex1@Y1.T@Q],
                 [np.zeros_like(R), np.zeros((k, k))]]))

            ex2, fe2 = expm_frechet(mat, E)
            M = ex2[:d, :d]
            N = ex2[d:, :d]
            YMQN = (Y@M+Q@N)

            partA = asym(
                (1-2*alf)*expm_frechet((1-2*alf)*A, Y1.T@YMQN)[1])

            partA += 2*alf*asym(fe2[:d, :d])
            partR = -(fe2[:d, d:].T - fe2[d:, :d])

            return vec(partA, partR)
    
        def hessp(v, xi):
            dlt = 1e-8
            return (jac(v+dlt*xi) - jac(v))/dlt

        def conv_to_tan(A, R):
            return Y@A + Q@R
        
        eta0 = self.proj(Y, Y1-Y)
        A0 = asym(Y.T@eta0)
        R0 = Q.T@eta0 - (Q.T@Y)@(Y.T@eta0)
        
        if init_type != 0:
            x0 = vec(A0, R0)
        else:
            x0 = np.zeros(adim + self.d*k)
                    
        def printxk(xk):
            print(la.norm(jac(xk)), dist(xk))

        if show_steps:
            callback = printxk
        else:
            callback = None

        res = {'fun': np.nan, 'x': np.zeros_like(x0),
               'success': False,
               'message': 'minimizer exception'}
        try:
            if self.log_gtol is None:
                if self.log_method.startswith('trust'):
                    res = minimize(dist, x0, method=self.log_method,
                                   jac=jac, hessp=hessp, callback=callback)
                else:
                    res = minimize(dist, x0, method=self.log_method,
                                   jac=jac, callback=callback)                
            else:
                if self.log_method.startswith('trust'):            
                    res = minimize(dist, x0, method=self.log_method,
                                   jac=jac, hessp=hessp, callback=callback,            
                                   options={'gtol': self.log_gtol})
                else:
                    res = minimize(dist, x0, method=self.log_method,
                                   jac=jac, callback=callback,
                                   options={'gtol': self.log_gtol})
        except Exception:
            pass
                
        stat = [(a, res[a]) for a in res.keys() if a not in ['x', 'jac']]
        A1, R1 = unvec(res['x'])
        if self.log_stats:
            return conv_to_tan(A1, R1), stat
        else:
            return conv_to_tan(A1, R1)            

