from __future__ import division
from .NullRangeManifold import NullRangeManifold
import numpy as np
import numpy.linalg as la
from scipy.linalg import expm
from numpy import trace, zeros, allclose
from numpy.random import randn
from .tools import sym, asym, extended_lyapunov, calc_stiefel_geodesics


if not hasattr(__builtins__, "xrange"):
    xrange = range


def _calc_dim(m, n, p):
    # ambient E has 3 pieces
    # m*p, n*p for Stiefel
    # p*p for Hermitian dim E = n*p + p*p

    # E_N has 2 pieces: JP, (Antisymmetric, JYP: ful p*p
    # J should be of size (antisym + full) * (full(p*p) + full(n*p))
    # dim: dim_St + dim symmetric - dim antisym= dim_st + p
    # codim = dimE - dim = dim E_J

    dm = (m+n-p)*p
    tdim = n*p + m*p + p*p
    return dm, 2*p*p, tdim


def calc_D(man, X, omg):
    al, bt, gm = (man.alpha, man.beta, man.gamma)
    U = X.U
    V = X.V
    P = X.P

    UTomgU = U.T @omg.tU
    VTomgV = V.T @omg.tV

    D0 = sym(omg.tP + 1/(al[1]+gm[1])*(
        al[1]*(UTomgU@P-P@UTomgU) +
        gm[1]*(VTomgV@P-P@VTomgV)))
    Dp = extended_lyapunov(1/bt, 1/(al[1]+gm[1]), P, D0, X.evl, X.evec)
    Dm = 1/(al[1]+gm[1])*asym(VTomgV - UTomgU)
    return Dp, Dm


class fr_point(object):
    """a psd_point consists of a triple
    U, V and P
    such that S = UPV.T
    Parameters
    ----------
    U, V, P   : matrices, U, V Stiefel, P >> 0

    Results
    ----------
    A fr_point representing the matrix UPV.T

    """
    def __init__(self, U, V, P):
        self._U = U
        self._V = V        
        self._P = P
        if not allclose(P, P.T):
            raise(ValueError('P not symmetric'))
        evl, evec = np.linalg.eigh(P)
        self.evl = np.abs(evl)
        self.evec = evec.real
        self.Pinv = self.evec @ np.diag(1/self.evl) @ self.evec.T

    @property
    def U(self):
        return self._U

    @property
    def V(self):
        return self._V
    
    @property
    def P(self):
        return self._P

    
class fr_ambient(object):
    """ Representing an ambient vector.
    Has 3 components corresponding to U, P, V
    Cache for the D matrix which is expensive
    to compute.

    Parameters
    ----------
    tU, tV, tP: the three components

    """
    
    def __init__(self, tU, tV, tP):
        self.tU = tU
        self.tV = tV
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
        return self.__class__(-self.tU, -self.tV, -self.tP)

    def __add__(self, other):
        return self.__class__(
            self.tU + other.tU,
            self.tV + other.tV,
            self.tP + other.tP)

    def __sub__(self, other):
        return self.__class__(
            self.tU - other.tU,
            self.tV - other.tV,
            self.tP - other.tP)
    
    def scalar_mul(self, other):
        return self.__class__(
            other*self.tU,
            other*self.tV,
            other*self.tP)

    def __rmul__(self, other: float):
        return self.__class__(
            other*self.tU,
            other*self.tV,
            other*self.tP)
        
    def mul_and_add(self, other, factor):
        return self.__class__(
            self.tU + factor*other.tU,
            self.tV + factor*other.tV,
            self.tP + factor*other.tP)
    

class RealFixedRank(NullRangeManifold):
    """Class for a Real Fixed rank manifold
    A manifold point is a triple (U, P, V) with
       U.T@U = I
       V.T@V = I
       P.T = P, P >> 0
    Matrix has decomposition UPV.H
    U of dimension m*p
    P of dimension p*p
    V of dimension n*p
    Metric is defined by three sets of parameters alpha, beta, gamma
    

    Parameters
    ----------
    m, n, p     : # of rows of U, P and V respectively
    alpha       : array size 2, alpha >> 0.
    beta        : positive number
    gamma       : array size 2, gamma >> 0.
    
    """
    def __init__(self, m, n, p, alpha=None, beta=1, gamma=None):
        self._point_layout = 1
        self._name = 'RealFixedRank %d %d %d' % (m, n, p)
        self.n = n
        self.m = m
        self.p = p
        # dm_St, dm_P, cdm_St, cdm_P, tdim_St, tdim_P
        dm, cdm, tdim = _calc_dim(m, n, p)
        self._dimension = dm
        self._codim = cdm
        self.tdim = tdim

        if alpha is None:
            self.alpha = np.array([1, 1])
        else:
            self.alpha = alpha
        if gamma is None:
            self.gamma = np.array([1, 1])
        else:
            self.gamma = gamma
        self.beta = beta

    def inner(self, X, Ba, Bb=None):
        alf = self.alpha
        gmm = self.gamma
        U = X.U
        V = X.V
        Pinv = X.Pinv
        if Bb is None:
            Bb = Ba
        return alf[0]*trace(Ba.tU.T @ Bb.tU) + (alf[1]-alf[0]) *\
            trace((Ba.tU.T @ U) @ (U.T @ Bb.tU)) +\
            self.beta*trace(Pinv @ Ba.tP @ Pinv @ Bb.tP.T) +\
            gmm[0]*trace(Ba.tV.T @ Bb.tV) + (gmm[1]-gmm[0]) *\
            trace((Ba.tV.T @ V) @ (V.T @ Bb.tV))
    
    @property
    def dim(self):
        return self._dimension
    
    @property
    def codim(self):
        return self._codim

    def __str__(self):
        return self._name

    @property
    def typicaldist(self):
        return np.sqrt(sum(self._dimension))

    def dist(self, X, Y):
        """ Geodesic distance. Not implemented
        """
        raise NotImplementedError

    def base_inner_ambient(self, E1, E2):
        return trace(E1.tP.T @ E2.tP) + trace(E1.tU.T @ E2.tU) +\
            trace(E1.tV.T @ E2.tV)

    def zerovec(self, S):
        return fr_ambient(
            zeros((self.m, self.p)),
            zeros((self.n, self.p)),
            zeros((self.p, self.p)))
    
    def g(self, X, E):
        al0, al1 = self.alpha
        gmm0, gmm1 = self.gamma
        U = X.U
        V = X.V
        return fr_ambient(
            al0*E.tU + (al1-al0) * U @ (U.T @ E.tU),
            gmm0*E.tV + (gmm1-gmm0) * V @ (V.T @ E.tV),
            self.beta * X.Pinv @ E.tP @ X.Pinv)

    def g_inv(self, X, E):
        ialp = 1/self.alpha
        igmm = 1/self.gamma
        U = X.U
        V = X.V
        return fr_ambient(
            ialp[0]*E.tU + (ialp[1]-ialp[0]) * U @ (U.T @ E.tU),
            igmm[0]*E.tV + (igmm[1]-igmm[0]) * V @ (V.T @ E.tV),
            1/self.beta * X.P @ E.tP @ X.P)
    
    def D_g(self, X, xi, E):
        al, gm, bt = (self.alpha, self.gamma, self.beta)
        U, V, Piv = (X.U, X.V, X.Pinv)        
        return fr_ambient(
            (al[1]-al[0]) * (xi.tU @ (U.T @ E.tU) +
                             U @ (xi.tU.T @ E.tU)),
            (gm[1]-gm[0]) * (xi.tV @ (V.T @ E.tV) +
                             V @ (xi.tV.T @ E.tV)),
            -bt*(Piv @ (xi.tP@Piv@E.tP + E.tP@Piv@xi.tP) @ Piv))
        
    def contract_D_g(self, X, xi, E):
        alp = self.alpha
        gmm = self.gamma
        Piv = X.Pinv
        return fr_ambient(
            (alp[1] - alp[0])*(E.tU @ xi.tU.T + xi.tU @ E.tU.T) @ X.U,
            (gmm[1] - gmm[0])*(E.tV @ xi.tV.T + xi.tV @ E.tV.T) @ X.V,
            -self.beta*(Piv@xi.tP@Piv@E.tP@Piv + Piv@E.tP@Piv@xi.tP@Piv))

    def christoffel_form(self, X, xi, eta):
        al, bt, gm = (self.alpha, self.beta, self.gamma)
        return fr_ambient(
            (al[1]-al[0])*(X.U@sym(eta.tU.T@xi.tU)-eta.tU@(xi.tU.T@X.U) -
                           xi.tU@(eta.tU.T@X.U)),
            (gm[1]-gm[0])*(X.V@sym(eta.tV.T@xi.tV)-eta.tV@(xi.tV.T@X.V) -
                           xi.tV@(eta.tV.T@X.V)),
            -bt*sym(X.Pinv@eta.tP@X.Pinv@xi.tP@X.Pinv))
    
    def st(self, mat):
        """The split_transpose. transpose if real, hermitian transpose if complex
        """
        return mat.T

    def D_proj(self, X, xi, omg):
        """ Derivatives of the projection
        """
        Dp, Dm = calc_D(self, X, omg)
        al1 = self.alpha[1]
        gm1 = self.gamma[1]
        agi = 1/(al1+gm1)
        bt = self.beta
        U, V, P, Piv = (X.U, X.V, X.P, X.Pinv)
        bkPivDp = Piv@Dp - Dp@Piv
        DxiLDp = agi*(xi.tP @ Dp @ Piv + Piv @ Dp @ xi.tP -
                      P @ Dp @ Piv @ xi.tP @ Piv -
                      Piv @ xi.tP @ Piv @ Dp @ P)
        ddin = agi*(
            al1*(xi.tU.T@omg.tU@P - P@xi.tU.T@omg.tU +
                 U.T @omg.tU@xi.tP - xi.tP@U.T@omg.tU) +
            gm1*(xi.tV.T@omg.tV@P - P@xi.tV.T@omg.tV +
                 V.T @omg.tV@xi.tP - xi.tP@V.T@omg.tV)) - DxiLDp

        Ddp = extended_lyapunov(1/bt, agi, P, sym(ddin), X.evl, X.evec)
        Ddm = agi*asym(xi.tV.T@omg.tV - xi.tU.T@omg.tU)

        PxPP = Piv@xi.tP@Piv
        DbkPivDp = -PxPP@Dp + Piv@Ddp - Ddp@Piv + Dp@PxPP
        dtU = xi.tU@(-gm1*Dm + agi*bkPivDp) +\
            U@(-gm1*Ddm + agi*DbkPivDp) -\
            xi.tU @ (U.T @ omg.tU) - U @ (xi.tU.T @omg.tU)
        dtV = xi.tV@(al1*Dm + agi*bkPivDp) +\
            V@(al1*Ddm + agi*DbkPivDp) -\
            xi.tV @ (V.T @ omg.tV) - V @ (xi.tV.T @omg.tV)
        return fr_ambient(dtU, dtV, 1/bt*Ddp)

    def ehess2rhess(self, S, egrad, ehess, H):
        return self.proj_g_inv(S, ehess + self.g(S, self.D_proj(
            S, H, self.g_inv(S, egrad))) - self.D_g(
                S, H, self.g_inv(S, egrad)) +
                self.christoffel_form(S, H, self.proj_g_inv(S, egrad)))

    def proj(self, X, omg):
        """projection. omg is in ambient
        return one in tangent.
        This is the idempotent projection, not the Riemannian gradient.
        """
        U, V, Piv = (X.U, X.V, X.Pinv)
        al, bt, gm = (self.alpha, self.beta, self.gamma)
        Dp, Dm = calc_D(self, X, omg)
        bkPivDp = Piv@Dp - Dp@Piv
        return fr_ambient(
            U @ (-gm[1]*Dm + 1/(al[1]+gm[1])*bkPivDp)+omg.tU - U@(U.T@omg.tU),
            V @(al[1]*Dm + 1/(al[1]+gm[1])*bkPivDp)+omg.tV - V@(V.T@omg.tV),
            1/bt*Dp)

    def christoffel_gamma(self, X, xi, eta):
        dprj = self.D_proj(X, xi, eta)
        proj_christoffel = self.proj_g_inv(
            X, self.christoffel_form(X, xi, eta))
        return proj_christoffel - dprj

    def rhess02_alt(self, X, xi, eta, egrad, ehess_val):
        return ehess_val - self.base_inner_ambient(
            self.christoffel_gamma(X, xi, eta), egrad)

    def retr(self, X, E):
        """ retract by SVD on the U+E.tU and V+E.tV components
        and by geodesic approximation for the positive definite part

        Parameters
        ----------
        X   : fr_point
        E   : fr_ambient

        Returns:
        ----------
        A new manifold point
        """
        x1 = X.U + E.tU
        x2 = X.V + E.tV
        u1, _, vh1 = np.linalg.svd(x1, full_matrices=False)
        u2, _, vh2 = np.linalg.svd(x2, full_matrices=False)
        return fr_point(
            u1 @ vh1,
            u2 @ vh2,
            sym(X.P + E.tP + 0.5*E.tP @ X.Pinv @ E.tP))

    def rand(self):
        # Generate random  point using qr of random normally distributed
        # matrix.
        U, _ = la.qr(randn(
            self.m, self.p))
        V, _ = la.qr(randn(
            self.n, self.p))        
        P = randn(self.p, self.p)
        return fr_point(U, V, P@P.T)
    
    def randvec(self, X):
        """Random tangent vector at point X
        """

        """
        U = np.random.randn(self._dim)
        U = U / self.norm(X, U)
        return U
        """
        amb = self.proj(X, self._rand_ambient())
        nrm = self.norm(X, amb)
        return fr_ambient(
            amb.tU/nrm,
            amb.tV/nrm,
            amb.tP/nrm)

    def _rand_ambient(self):
        return fr_ambient(
            randn(self.m, self.p),
            randn(self.n, self.p),
            randn(self.p, self.p))

    def _vec(self, E):
        """vectorize. This is usually used for sanity test in low dimension
        typically X.reshape(-1). For exampe, we can test J, g by representing
        them as matrices.
        Convenient for testing but dont expect much actual use
        """
        return np.concatenate(
            [E.tU.reshape(-1),
             E.tV.reshape(-1),
             E.tP.reshape(-1)])

    def _unvec(self, vec):
        """reshape
        """
        d1 = self.m*self.p
        d2 = self.n*self.p
        return fr_ambient(vec[:d1].reshape(self.m, self.p),
                          vec[d1:d1+d2].reshape(self.n, self.p),
                          vec[d1+d2:].reshape(self.p, self.p))
    
    def exp(self, X, eta):
        """ Geodesic exponent. gamma(1) for a geodesics gamma with
            gamma(0) = X
            dot{gamma}(0) = eta

        Parameters
        ----------
        X   : fr_point
        eta : tangent at X, fr_ambient

        Results
        ----------
        gamma(1)
        """
        
        Ut = calc_stiefel_geodesics(X.U, eta.tU, self.alpha[1]/self.alpha[0])
        Vt = calc_stiefel_geodesics(X.V, eta.tV, self.gamma[1]/self.gamma[0])
            
        sqrtP = X.evec @ np.diag(np.sqrt(X.evl)) @ X.evec.T
        isqrtP = X.evec @ np.diag(1/np.sqrt(X.evl)) @ X.evec.T
        Pinn = isqrtP@eta.tP@isqrtP
        ePinn = expm(Pinn)
        Pt = sqrtP@ePinn@sqrtP
    
        return fr_point(np.array(Ut),
                        np.array(Vt),
                        np.array(Pt))
