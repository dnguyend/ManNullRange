from __future__ import division
from .NullRangeManifold import NullRangeManifold
import numpy as np
import numpy.linalg as la
from scipy.linalg import null_space, expm, expm_frechet, logm
from scipy.optimize import minimize
from numpy import trace, zeros, allclose
from numpy.random import randn
from .tools import (sym, asym, vecah, unvecah, extended_lyapunov,
                    vech, unvech)


if not hasattr(__builtins__, "xrange"):
    xrange = range


def _calc_dim(n, p):
    # ambient E has 2 pieces
    # n*p for Stiefel
    # p*p for Hermitian dim E = n*p + p*p

    # E_J has 2 pieces: JP, (Antisymmetric, JYP: ful p*p
    # J should be of size (antisym + full) * (full(p*p) + full(n*p))
    # dim: dim_St + dim symmetric - dim antisym= dim_st + p
    # codim = dimE - dim = dim E_J

    # dm_St = p * (n-p) + p*(p-1)//2
    # dm_P = p*(p+1) // 2
    dm = p * (n-p) + p*(p+1)//2
    cdm_YP = p*p
    cdm_P = p*(p-1) // 2
    tdim_St = n*p
    tdim_P = p*p
    return dm, cdm_P, cdm_YP, tdim_St, tdim_P


class psd_point(object):
    """a psd_point consists of
    pair of Y and P
    such that S = YPY.T

    Parameters
    ----------
    Y, P   : matrices
    
    Results
    ----------
    A psd_point representing the matrix YPY.T

    """
    def __init__(self, Y, P):
        self._Y = Y
        self._P = P
        if not allclose(P, P.T):
            raise(ValueError('P not symmetric'))
        evl, evec = np.linalg.eigh(P)
        self.evl = np.abs(evl)
        self.evec = evec.real
        self.Pinv = self.evec @ np.diag(1/self.evl) @ self.evec.T
        self._Y0 = None

    @property
    def Y(self):
        return self._Y
    
    @property
    def P(self):
        return self._P

    @property
    def Y0(self):
        """This could be expensive. Cache it. Only computed if called
        """
        if self._Y0 is None:
            Y0 = null_space(self._Y.T)
            self._Y0 = Y0
        return self._Y0

    def _add_point(self, other):
        """ not safe, a cheap retraction. Only to test
        derivatives (other is small so the new point is close enough to
        the manifold)
        """
        return psd_point(self.Y + other.tY, self.P + other.tP)

    def copy(self):
        return psd_point(self.Y, self.P)
    

class psd_ambient(object):
    """ Representing an ambient vector.
    Has 2 components corresponding to Y, P
    Cache for the D matrix which is expensive
    to compute.

    Parameters
    ----------
    tY, tP: the components

    """
    
    def __init__(self, tY, tP):
        self.tY = tY
        self.tP = tP
        self.__D = None

    @property
    def D(self):
        if self.__D is None:
            raise ValueError("Need to set D before access")
        return self.__D

    @D.setter
    def D(self, Din):
        self.__D = Din

    def __neg__(self):
        return self.__class__(-self.tY, -self.tP)

    def __add__(self, other):
        return self.__class__(
            self.tY + other.tY, self.tP + other.tP)

    def __sub__(self, other):
        return self.__class__(
            self.tY - other.tY, self.tP - other.tP)
    
    def scalar_mul(self, other):
        return self.__class__(
             other*self.tY, other*self.tP)

    def __rmul__(self, other: float):
        return self.__class__(
            other*self.tY, other*self.tP)
        
    def mul_and_add(self, other, factor):
        return self.__class__(
            self.tY + factor*other.tY, self.tP + factor*other.tP)
    

class RealPositiveSemidefinite(NullRangeManifold):
    """Class for a Real Positive Semidefinite manifold
    A manifold point is a triple (Y, P) with
       Y.T@Y = I
       P.T = P, P >> 0

    Y of dimension n*p
    P of dimension p*p
    Metric is defined by two sets of parameters alpha, beta

    Parameters
    ----------
    n, p     : # of rows of Y and P respectively
    alpha    : array of size 2 defining a metric on the Stiefel manifold.
               alpha  > 0
    beta     : positive number. Metric scale on P
    """
    
    def __init__(self, n, p, alpha=None, beta=1, log_gtol=1e-4,
                 log_stats=False):
        self._point_layout = 1
        self.n = n
        self.p = p
        # dm_St, dm_P, cdm_St, cdm_P, tdim_St, tdim_P
        dm, cdm_P, cdm_YP, tdim_St, tdim_P = _calc_dim(n, p)
        self._dimension = dm
        self._codim = cdm_YP + cdm_P
        self._codim_YP = cdm_YP
        self._codim_P = cdm_P
        self.tdim_St = tdim_St
        self.tdim_P = tdim_P

        if alpha is None:
            self.alpha = np.array([1, .5])
        else:
            self.alpha = alpha
        self.beta = beta
        self.log_gtol = log_gtol
        self.log_stats = log_stats

    def inner(self, S, Ba, Bb=None):
        """Inner product (Riemannian metric, as an inner product on
        the tangent ambient
        """
        alf = self.alpha
        Y = S.Y
        Pinv = S.Pinv
        if Bb is None:
            Bb = Ba
        return alf[0]*trace(Ba.tY.T @ Bb.tY) + (alf[1]-alf[0]) *\
            trace((Ba.tY.T @ Y) @ (Y.T @ Bb.tY)) +\
            self.beta*trace(Pinv @ Ba.tP @ Pinv @ Bb.tP.T)
    
    @property
    def dim(self):
        return self._dimension
    
    @property
    def codim(self):
        return self._codim

    def __str__(self):
        self._name = "Real Positive Semidefinite manifold n=%d p=%d %s %s" % (
            self.n, self.p, str(self.alpha), (self.beta))

        return self._name

    @property
    def typicaldist(self):
        return np.sqrt(sum(self._dimension))

    def dist(self, X, Y):
        """ Geodesic distance. Not implemented
        """
        raise NotImplementedError

    def base_inner_ambient(self, E1, E2):
        return trace(E1.tP.T @ E2.tP) + trace(E1.tY.T @ E2.tY)

    def base_inner_E_J(self, a1, a2):
        return trace(a1['P'].T @ a2['P']) + trace(a1['YP'].T @ a2['YP'])

    def zerovec(self, S):
        return psd_ambient(
            zeros((self.n, self.p)), zeros((self.p, self.p)))
    
    def g(self, S, E):
        al0, al1 = self.alpha
        Y = S.Y
        return psd_ambient(
            al0*E.tY + (al1-al0) * Y @ (Y.T @ E.tY),
            self.beta * S.Pinv @ E.tP @ S.Pinv)

    def g_inv(self, S, E):
        ialp = 1/self.alpha
        Y = S.Y
        return psd_ambient(
            ialp[0]*E.tY + (ialp[1]-ialp[0]) * Y @ (Y.T @ E.tY),
            1/self.beta * S.P @ E.tP @ S.P)
    
    def J(self, S, E):
        alpha = self.alpha
        beta = self.beta
        a = {}
        a['P'] = E.tP - E.tP.T
        a['YP'] = alpha[1]*S.Y.T @ E.tY +\
            beta*(E.tP@S.Pinv - S.Pinv@E.tP)
        return a

    def Jst(self, S, a):
        return psd_ambient(
            self.alpha[1]*S.Y@a['YP'],
            2*a['P'] + self.beta*(a['YP'] @ S.Pinv - S.Pinv @ a['YP']))

    def g_inv_Jst(self, S, a):
        return psd_ambient(
            S.Y@a['YP'],
            (2/self.beta)*S.P@a['P']@S.P + S.P @ a['YP'] - a['YP'] @ S.P)

    def D_g(self, S, xi, E):
        alf = self.alpha
        beta = self.beta
        Y = S.Y
        Piv = S.Pinv
        return psd_ambient(
            (alf[1]-alf[0]) * (xi.tY @ (Y.T @ E.tY) + Y @ (xi.tY.T @ E.tY)),
            -beta*(Piv @ (xi.tP@Piv@E.tP + E.tP@Piv@xi.tP) @ Piv))

    def D_J(self, S, xi, E):
        alf1 = self.alpha[1]
        beta = self.beta
        Piv = S.Pinv
        a = {}
        a['P'] = zeros(S.P.shape)
        a['YP'] = alf1*xi.tY.T@E.tY-beta*(
            E.tP@Piv@xi.tP@Piv - Piv@xi.tP@Piv@E.tP)
        return a
    
    def D_Jst(self, S, xi, a):
        return psd_ambient(
            self.alpha[1]*xi.tY @ a['YP'],
            self.beta*(S.Pinv@xi.tP@S.Pinv@a['YP'] -
                       a['YP']@S.Pinv@xi.tP@S.Pinv))

    def D_g_inv_Jst(self, X, xi, a):
        djst = self.D_Jst(X, xi, a)
        return self.g_inv(
            X, -self.D_g(X, xi, self.g_inv(X, self.Jst(X, a))) + djst)
    
    def contract_D_g(self, S, xi, E):
        alpha = self.alpha
        Piv = S.Pinv
        return psd_ambient(
            (alpha[1] - alpha[0])*(E.tY @ xi.tY.T + xi.tY @ E.tY.T) @ S.Y,
            -self.beta*(Piv@xi.tP@Piv@E.tP@Piv + Piv@E.tP@Piv@xi.tP@Piv))
    
    def st(self, mat):
        """The split_transpose. transpose if real, hermitian transpose if complex
        """
        return mat.T

    def J_g_inv(self, S, E):
        a = {}
        a['P'] = 1/self.beta*S.P@(E.tP - E.tP.T)@S.P
        a['YP'] = S.Y.T@E.tY + S.P@E.tP - E.tP@S.P
        return a

    def J_g_inv_Jst(self, S, a):
        beta = self.beta
        alf = self.alpha
        anew = {}
        saYP = a['YP'] + a['YP'].T

        anew['P'] = 4/beta * S.P @ a['P'] @ S.P + S.P @ saYP - saYP @ S.P
        anew['YP'] = (alf[1]-2*beta)*a['YP'] + (
            (2*S.P@a['P'] - 2*a['P']@S.P + beta*S.P @ a['YP'] @ S.Pinv +
             beta*S.Pinv@a['YP'] @ S.P))

        return anew

    def solve_J_g_inv_Jst(self, S, b):
        """ base is use CG. Unlikely to use
        """
        beta = self.beta
        alf = self.alpha
        anew = {}
        ayp_even = 1/alf[1]*sym(b['YP']) + beta/2/alf[1]*(S.Pinv@ b['P'] -
                                                          b['P'] @ S.Pinv)
        odd_rhs = S.evec.T @ asym(b['YP']) @ S.evec
        evli = 1/S.evl
        ayp_odd = S.evec @ (odd_rhs / (beta*(S.evl[:, None] * evli[None, :]) +
                                       beta*(evli[:, None] * S.evl[None, :]) +
                                       alf[1]-2*beta)) @ S.evec.T
        anew['YP'] = ayp_even + ayp_odd
        anew['P'] = beta*(.25*S.Pinv@b['P']@S.Pinv + asym(S.Pinv @ ayp_even))
        return anew

    def _calc_D(self, S, U):
        YTU = S.Y.T@U.tY
        D0 = sym(U.tP + YTU@S.P - S.P@YTU)
        D = extended_lyapunov(
            self.alpha[1], self.beta, S.P, D0, S.evl, S.evec)
        return D
    
    def proj(self, S, U):
        """projection. U is in ambient
        return one in tangent
        """
        al1 = self.alpha[1]
        beta = self.beta
        D = self._calc_D(S, U)
        return psd_ambient(
            beta*S.Y@(S.Pinv@D-D@S.Pinv) + U.tY - S.Y@(S.Y.T@U.tY), al1*D)

    def D_proj(self, S, xi, U):
        D = self._calc_D(S, U)
        al1 = self.alpha[1]
        bt = self.beta
        
        ddin = xi.tY.T @ U.tY @ S.P - S.P @ xi.tY.T @ U.tY + \
            S.Y.T @ U.tY @ xi.tP - xi.tP @ S.Y.T @ U.tY - \
            self.beta * (xi.tP @ D @ S.Pinv + S.Pinv @ D @ xi.tP -
                         S.P @ D @ S.Pinv @ xi.tP @ S.Pinv -
                         S.Pinv @ xi.tP @ S.Pinv @ D @ S.P)
        dd = extended_lyapunov(al1, bt, S.P,  sym(ddin), S.evl, S.evec)
        t2 = bt * xi.tY @ (S.Pinv @ D - D @ S.Pinv) +\
            bt * S.Y @ (S.Pinv @ dd - dd @ S.Pinv +
                        D @ S.Pinv @ xi.tP @ S.Pinv -
                        S.Pinv @ xi.tP @ S.Pinv @ D) -\
            (xi.tY @ S.Y.T + S.Y @ xi.tY.T) @ U.tY
        return psd_ambient(t2, al1*dd)

    def ehess2rhess_alt(self, S, egrad, ehess, H):
        return self.proj_g_inv(S, ehess + self.g(S, self.D_proj(
            S, H, self.g_inv(S, egrad))) - self.D_g(
                S, H, self.g_inv(S, egrad)) +
                self.christoffel_form(S, H, self.proj_g_inv(S, egrad)))

    def proj_range_alt(self, S, U):
        """projection. U is in ambient
        return one in tangent
        """
        def N(man, S, B, D):
            al0, al1 = man.alpha
            bt = man.beta

            Y0 = S.Y0
            SPinvD = S.Pinv@D
            return psd_ambient(bt*S.Y@(SPinvD - SPinvD.T)+Y0@B, al1*D)

        def solveNTgN(man, S, Bin, Din):
            al0, al1 = man.alpha
            bt = man.beta

            Bout = 1/al0*Bin
            D1 = 1/(al1*bt)*S.evec.T@S.P@Din@S.P@S.evec
            evli = 1/S.evl
            Dout = D1/(bt*(S.evl[:, None] * evli[None, :]) +
                       bt*(evli[:, None] * S.evl[None, :]) + al1 - 2*bt)

            return Bout, S.evec@Dout@S.evec.T
        
        def NTg(man, S, U):
            al0, al1 = man.alpha
            bt = man.beta
            Y0 = S.Y0
            NTg_B = al0*Y0.T@U.tY
            NTg_D = S.Pinv@U.tP@S.Pinv
            NTg_D += S.Pinv@S.Y.T@U.tY - S.Y.T@U.tY@S.Pinv
            NTg_D = al1*bt*sym(NTg_D)
            return NTg_B, NTg_D
        
        return N(self, S, *solveNTgN(self, S, *NTg(self, S, U)))

    def retr(self, S, E):
        """ Calculate 'thin' qr decomposition of X + G
        then add point X
        then do thin lq decomposition
        """
        x1 = S.Y + E.tY
        u, s, vh = np.linalg.svd(x1, full_matrices=False)
        return psd_point(
            u @ vh,
            sym(S.P + E.tP + 0.5*E.tP @ S.Pinv @ E.tP))

    def rand(self):
        # Generate random  point using qr of random normally distributed
        # matrix.
        Y, _ = la.qr(randn(
            self.n, self.p))
        P = randn(self.p, self.p)
        return psd_point(Y, P@P.T)
    
    def randvec(self, X):
        """Random tangent vector at point X
        """

        U = self.proj(X, self._rand_ambient())
        nrm = self.norm(X, U)
        return psd_ambient(U.tY/nrm, U.tP/nrm)

    def _rand_ambient(self):
        return psd_ambient(randn(self.n, self.p), randn(self.p, self.p))

    def _rand_range_J(self):
        a = {}
        w1 = randn(self.p, self.p)
        a['P'] = w1 - w1.T
        a['YP'] = randn(self.p, self.p)
        return a

    def _vec(self, E):
        """vectorize. This is usually used for sanity test in low dimension
        typically X.reshape(-1). For exampe, we can test J, g by representing
        them as matrices.
        Convenient for testing but dont expect much actual use
        """
        return np.concatenate(
            [E.tY.reshape(-1), E.tP.reshape(-1)])

    def _unvec(self, vec):
        """reshape
        """
        return psd_ambient(vec[:self.tdim_St].reshape(self.n, self.p),
                           vec[self.tdim_St:].reshape(self.p, self.p))

    def _vec_range_J(self, a):
        """vectorize an elememt of rangeJ
        a.reshape(-1)
        """
        ret = zeros(self.codim)
        start = 0
        tp = vecah(a['P'])

        ret[start:start+tp.shape[0]] = tp
        start += tp.shape[0]
        ret[start:] = a['YP'].reshape(-1)
        return ret

    def _unvec_range_J(self, vec):
        a = {}
        a['P'] = unvecah(vec[:self._codim_P])
        a['YP'] = vec[:self.codim_P].reshape(self.n, self.d)
        return a

    def exp(self, X, eta):
        """Geodesic from X in direction eta

        Parameters
        ----------
        X    : a manifold point
        eta  : tangent vector
        
        Returns
        ----------
        gamma(1), where gamma(t) is the geodesics at X in direction eta

        """
        K = eta.tY - X.Y @ (X.Y.T @ eta.tY)
        Yp, R = la.qr(K)
        alf = self.alpha[1]/self.alpha[0]
        A = X.Y.T @eta.tY
        x_mat = np.bmat([[2*alf*A, -R.T],
                         [R, zeros((self.p, self.p))]])
        Yt = np.array(np.bmat([X.Y, Yp]) @ expm(x_mat)[:, :self.p] @
                      expm((1-2*alf)*A))

        sqrtP = X.evec @ np.diag(np.sqrt(X.evl)) @ X.evec.T
        isqrtP = X.evec @ np.diag(1/np.sqrt(X.evl)) @ X.evec.T
        Pinn = isqrtP @ eta.tP @ isqrtP
        ePinn = expm(Pinn)
        Pt = np.array(sqrtP @ ePinn@sqrtP)
        return psd_point(Yt, Pt)

    def log(self, X, X1, show_steps=False):
        """Inverse of exp

        Parameters
        ----------
        X    : a manifold point
        X1  : tangent vector

        Returns
        ----------
        eta such that self.exp(X, eta) = X1

        Algorithm: use the scipy.optimize trust region method
        to minimize in eta ||full(self.exp(X, eta)) - full(X1)||_F^2
        where full(X) = X.Y@X.P@X.Y.T
        _F is the Frobenius norm in R^{n\times n}
        The jacobian could be computed by the expm_frechet function
        """
        beta, alpha = self.beta, self.alpha
        alf = self.alpha[1]/self.alpha[0]    
        n, p = self.n, self.p
        Q, s, _ = la.svd(X1.Y - X.Y@X.Y.T@X1.Y, full_matrices=False)
        Q = Q[:, :np.sum(np.abs(s) > 1e-15)]
        k = Q.shape[1]
        if k == 0:
            # X1.Y and X.Y has the same linear span
            U = X1.Y.T @ X.Y
            # U is orthogonal.
            eta_P = X.P@logm(X.Pinv@U.T@X1.P@U)
            if self.log_stats:
                return psd_ambient(np.zeros_like(X.Y), eta_P), [('success', True),
                                                                ('message', 'aligment')]
            return psd_ambient(np.zeros_like(X.Y), eta_P)
        
        pdim = p*(p+1) // 2
    
        def vec(eta_P, R):
            return np.concatenate(
                [vech(eta_P), R.reshape(-1)])

        def unvec(avec):
            return unvech(avec[:pdim]), avec[pdim:].reshape(k, p)
    
        tp12 = trace(X1.P@X1.P)
        YPY_l = np.bmat([X1.P@X1.Y.T@X.Y, X1.P@X1.Y.T@ Q])
        YPY_r = np.bmat([[X.Y.T@X1.Y], [Q.T@X1.Y]])
        YPY_rl = YPY_r@YPY_l
    
        def dist(v):
            eta_P, R = unvec(v)
            A = beta/alpha[1]*(X.Pinv@eta_P - eta_P@X.Pinv)
            x_mat = np.array(
                np.bmat([[2*alf*A, -R.T], [R, zeros((k, k))]]))
            trYPY2 =  np.trace(
                X.P @ expm(X.Pinv@eta_P) @ X.P @ expm(X.Pinv@eta_P))
            trYPY = trace(
                expm(x_mat)[:, :p] @
                expm((1-2*alf)*A) @ X.P @ expm(X.Pinv@eta_P) @
                expm(-(1-2*alf)*A) @ expm(x_mat)[:, :p].T @ YPY_rl)

            return trYPY2 - 2*trYPY + tp12
    
        def jac(v):
            eta_P, R = unvec(v)
            A = beta/alpha[1]*(X.Pinv@eta_P - eta_P@X.Pinv)
            x_mat = np.array(
                np.bmat([[2*alf*A, -R.T], [R, zeros((k, k))]]))
            ex1 = expm(X.Pinv@eta_P)
            ex3 = expm((1-2*alf)*A)
            exmat = expm(x_mat)
            efPinvD = expm_frechet(
                X.Pinv@eta_P,
                X.P @ ex1 @ X.P - ex3.T @ exmat[:, :p].T @ YPY_rl @
                exmat[:, :p] @ ex3 @ X.P)
            efxmat = expm_frechet(
                x_mat,
                np.bmat([[
                    ex3 @ X.P @ ex1 @
                    ex3.T @ exmat[:, :p].T @ YPY_rl +
                    ex3 @ X.P @ ex1 @ ex3.T @ exmat[:, :p].T @ YPY_rl],
                         [np.zeros((k, p+k))]]))

            efA1 = expm_frechet(
                (1-2*alf)*A,
                (1-2*alf)*beta/alpha[1]*(
                    X.P @ ex1 @
                    ex3.T @ exmat[:, :p].T @ YPY_rl @ exmat[:, :p]))

            efA2 = expm_frechet(
                (1-2*alf)*A, exmat[:, :p].T @ YPY_rl @
                exmat[:, :p] @ ex3 @ X.P @ ex1)

            grP = 2*efPinvD[1] @ X.Pinv
            grP += -2*2*alf*beta/alpha[1]*(efxmat[1][:p, :p] @ X.Pinv -
                                           X.Pinv @ efxmat[1][:p, :p])
            grP += -2*efA1[1] @ X.Pinv + 2*X.Pinv @ efA1[1]
            grP += 2*(1-2*alf)*beta/alpha[1] * (
                ex3.T @ efA2[1]@ ex3.T @ X.Pinv - X.Pinv @ ex3.T @ efA2[1]@ ex3.T)
            grR = -2*(-efxmat[1][p:, :p] + efxmat[1][:p, p:].T)
            return vec(sym(grP), grR)

        def hessp(v, xi):
            dlt = 1e-6
            return (jac(v+dlt*xi) - jac(v))/dlt

        def conv_to_tan(eta_P, R):
            A = beta/alpha[1]*(X.Pinv@eta_P - eta_P@X.Pinv)
            return psd_ambient(X.Y@A + Q@R, eta_P)

        Cmat = np.bmat([X1.Y.T@X.Y, X1.Y.T@Q])

        def calc_U(eta_P, R):
            A = beta/alpha[1]*(X.Pinv@eta_P - eta_P@X.Pinv)
            x_mat = np.array(
                np.bmat([[2*alf*A, -R.T], [R, zeros((k, k))]]))
            MN = expm(x_mat)[:, :p]
            ex1 = expm((1-2*alf)*A)

            U = Cmat @ MN @ ex1            
            return U

        def make_init():
            # idea: good direction is Y1P1Y1.T - YPY.T
            # convert this to Stiefel coordnate
            if True:
                R0 = Q.T @ (X1.Y@X1.P@X1.Y.T - X.Y@X.P@X.Y.T)@X.Y@X.Pinv
                RHS = alpha[1]*((X.Y.T@X1.Y)@X1.P@(X1.Y.T@X.Y) - X.P)
                etaP0 = extended_lyapunov(
                    alpha[1], beta, X.P, RHS, X.evl, X.evec)
                return vec(etaP0, R0)
            else:
                xir = self.randvec(X)
                xirR = Q.T@xir.tY - (Q.T@X.Y)@(X.Y.T@xir.tY)    
                return vec(xir.tP, xirR)        
    
        x0 = make_init()
        
        def printxk(xk):
            print(la.norm(jac(xk)), dist(xk))

        if show_steps:
            callback = printxk
        else:
            callback = None
        res = minimize(dist, np.zeros_like(x0), method='trust-ncg',
                       jac=jac, hessp=hessp, callback=callback,
                       options={'gtol': self.log_gtol})
            
        A1, R1 = unvec(res['x'])
        if self.log_stats:
            return conv_to_tan(A1, R1), [(a, res[a]) for a in res.keys() if a not in ['x', 'jac']]
        else:
            return conv_to_tan(A1, R1)
    
        

