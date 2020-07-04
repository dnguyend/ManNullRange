from __future__ import division
from .NullRangeManifold import NullRangeManifold
import numpy as np
import numpy.linalg as la
from scipy.linalg import null_space
from numpy import trace, zeros, allclose
from numpy.random import randn
from .tools import hsym, ahsym, cvecah, cunvecah, cvec, cunvec, crandn


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
    dm = 2*p * (n-p) + p*p
    cdm_YP = 2*p*p
    cdm_P = p*p
    tdim_St = 2*n*p
    tdim_P = 2*p*p
    return dm, cdm_P, cdm_YP, tdim_St, tdim_P


def _extended_lyapunov(alpha1, beta, P, B, Peig=None, Pevec=None):
    """ solve the Lyapunov-type equation:
    (alpha-2beta)X + \beta(PXP^{-1} + P^{-1}XP = B
    Peig and Pevec are precomputed eigenvalue decomposition of P
    """

    if Peig is None:
        Peig, Pevec = la.eigh(P)
    evli = 1/Peig
    return Pevec @ ((Pevec.T.conjugate() @ B @ Pevec) /
                    (beta*(Peig[:, None] * evli[None, :] +
                           evli[:, None] * Peig[None, :]) +
                     alpha1-2*beta)) @ Pevec.T.conjugate()


class psd_point(object):
    """a psd_point consists of
    pair of Y and P
    such that S = YR^2Y
    """
    def __init__(self, Y, P):
        self._Y = Y
        self._P = P
        if not allclose(P, P.T.conjugate()):
            raise(ValueError('P not symmetric'))
        evl, evec = np.linalg.eigh(P)
        self.evl = np.abs(evl)
        self.evec = evec
        self.Pinv = self.evec @ np.diag(1/self.evl) @ self.evec.T.conjugate()
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
            Y0 = null_space(self._Y.T.conjugate())
            self._Y0 = Y0
        return self._Y0
    

class psd_ambient(object):
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
    

class ComplexPositiveSemidefinite(NullRangeManifold):
    def __init__(self, n, p, alpha=None, beta=1):
        self.n = n
        self.p = p
        # dm_St, dm_P, cdm_St, cdm_P, tdim_St, tdim_P
        dm, cdm_P, cdm_YP, tdim_St, tdim_P = _calc_dim(n, p)
        self._dim = dm
        self._codim = cdm_YP + cdm_P
        self._codim_YP = cdm_YP
        self._codim_P = cdm_P
        self.tdim_St = tdim_St
        self.tdim_P = tdim_P
        self._point_layout = 1
        if alpha is None:
            self.alpha = np.array([1, .5])
        else:
            self.alpha = alpha
        self.beta = beta

    def inner_product_amb(self, S, Ba, Bb=None):
        alf = self.alpha
        Y = S.Y
        Pinv = S.Pinv
        if Bb is None:
            Bb = Ba
        inn = alf[0]*trace(Ba.tY.T.conjugate() @ Bb.tY) + (alf[1]-alf[0]) *\
            trace((Ba.tY.T.conjugate() @ Y) @ (Y.T.conjugate() @ Bb.tY)) +\
            self.beta*trace(Pinv @ Ba.tP @ Pinv @ Bb.tP.T.conjugate())
        return inn.real
    
    @property
    def dim(self):
        return self._dim
    
    @property
    def codim(self):
        return self._codim

    def __str__(self):
        return "ComplexPositiveSemidefinite=(n=%d p=%d)" % (self.n, self.p)

    @property
    def typicaldist(self):
        return np.sqrt(sum(self._dim))

    def dist(self, X, Y):
        """ Geodesic distance. Not implemented
        """
        raise NotImplementedError

    def base_inner_ambient(X, E1, E2):
        return trace(E1.tP.T.conjugate() @ E2.tP).real +\
            trace(E1.tY.T.conjugate() @ E2.tY).real

    def base_inner_E_J(X, a1, a2):
        return trace(a1['P'].T.conjugate() @ a2['P']).real +\
            trace(a1['YP'].T.conjugate() @ a2['YP']).real

    def zerovec(self, S):
        return psd_ambient(
            zeros((self.n, self.p), dtype=np.complex),
            zeros((self.p, self.p), dtype=np.complex))
    
    def g(self, S, E):
        al0, al1 = self.alpha
        Y = S.Y
        return psd_ambient(
            al0*E.tY + (al1-al0) * Y @ (Y.T.conjugate() @ E.tY),
            self.beta * S.Pinv @ E.tP @ S.Pinv)

    def g_inv(self, S, E):
        ialp = 1/self.alpha
        Y = S.Y
        return psd_ambient(
            ialp[0]*E.tY + (ialp[1]-ialp[0]) * Y @ (Y.T.conjugate() @ E.tY),
            1/self.beta * S.P @ E.tP @ S.P)
    
    def J(self, S, E):
        alpha = self.alpha
        beta = self.beta
        a = {}
        a['P'] = E.tP - E.tP.T.conjugate()
        a['YP'] = alpha[1]*S.Y.T.conjugate() @ E.tY +\
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
            (alf[1]-alf[0]) * (xi.tY @ (Y.T.conjugate() @ E.tY) +
                               Y @ (xi.tY.T.conjugate() @ E.tY)),
            -beta*(Piv @ (xi.tP@Piv@E.tP + E.tP@Piv@xi.tP) @ Piv))

    def D_J(self, S, xi, E):
        alf1 = self.alpha[1]
        beta = self.beta
        Piv = S.Pinv
        a = {}
        a['P'] = zeros(S.P.shape, dtype=np.complex)
        a['YP'] = alf1*xi.tY.T.conjugate()@E.tY-beta*(
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
            (alpha[1] - alpha[0])*(E.tY @ xi.tY.T.conjugate() +
                                   xi.tY @ E.tY.T.conjugate()) @ S.Y,
            -self.beta*(Piv@xi.tP@Piv@E.tP@Piv + Piv@E.tP@Piv@xi.tP@Piv))

        raise NotImplementedError
    
    def inner(self, X, G, H):
        """ Inner product (Riemannian metric) on the tangent space.
        The tangent space is given as a matrix of size mm_degree * m
        """
        # return inner_product_tangent
        return self.inner_product_amb(X, G, H)

    def st(self, mat):
        """The split_transpose. transpose if real, hermitian transpose if complex
        """
        return mat.T.conjugate()

    def J_g_inv(self, S, E):
        a = {}
        a['P'] = 1/self.beta*S.P@(E.tP - E.tP.T.conjugate())@S.P
        a['YP'] = S.Y.T.conjugate()@E.tY + S.P@E.tP - E.tP@S.P
        return a

    def J_g_inv_Jst(self, S, a):
        beta = self.beta
        alf = self.alpha
        anew = {}
        saYP = a['YP'] + a['YP'].T.conjugate()

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
        ayp_even = 1/alf[1]*hsym(b['YP']) + beta/2/alf[1]*(S.Pinv@ b['P'] -
                                                           b['P'] @ S.Pinv)
        odd_rhs = S.evec.T.conjugate() @ ahsym(b['YP']) @ S.evec
        evli = 1/S.evl
        ayp_odd = S.evec @ (odd_rhs / (beta*(S.evl[:, None] * evli[None, :]) +
                                       beta*(evli[:, None] * S.evl[None, :]) +
                                       alf[1]-2*beta)) @ S.evec.T.conjugate()
        anew['YP'] = ayp_even + ayp_odd
        anew['P'] = beta*(.25*S.Pinv@b['P']@S.Pinv + ahsym(S.Pinv @ ayp_even))
        return anew

    def _calc_D(self, S, U):
        YTU = S.Y.T.conjugate()@U.tY
        D0 = hsym(U.tP + YTU@S.P - S.P@YTU)
        D = _extended_lyapunov(
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
            beta*S.Y@(S.Pinv@D-D@S.Pinv) + U.tY -
            S.Y@(S.Y.T.conjugate()@U.tY), al1*D)

    def D_proj(self, S, xi, U):
        D = self._calc_D(S, U)
        al1 = self.alpha[1]
        bt = self.beta
        
        ddin = xi.tY.T.conjugate() @ U.tY @ S.P -\
            S.P @ xi.tY.T.conjugate() @ U.tY + \
            S.Y.T.conjugate() @ U.tY @ xi.tP -\
            xi.tP @ S.Y.T.conjugate() @ U.tY - \
            self.beta * (xi.tP @ D @ S.Pinv + S.Pinv @ D @ xi.tP -
                         S.P @ D @ S.Pinv @ xi.tP @ S.Pinv -
                         S.Pinv @ xi.tP @ S.Pinv @ D @ S.P)
        dd = _extended_lyapunov(al1, bt, S.P,  hsym(ddin), S.evl, S.evec)
        t2 = bt * xi.tY @ (S.Pinv @ D - D @ S.Pinv) +\
            bt * S.Y @ (S.Pinv @ dd - dd @ S.Pinv +
                        D @ S.Pinv @ xi.tP @ S.Pinv -
                        S.Pinv @ xi.tP @ S.Pinv @ D) -\
            (xi.tY @ S.Y.T.conjugate() + S.Y @ xi.tY.T.conjugate()) @ U.tY
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
            return psd_ambient(bt*S.Y@(SPinvD -
                                       SPinvD.T.conjugate())+Y0@B, al1*D)

        def solveNTgN(man, S, Bin, Din):
            al0, al1 = man.alpha
            bt = man.beta

            Bout = 1/al0*Bin
            D1 = 1/(al1*bt)*S.evec.T.conjugate()@S.P@Din@S.P@S.evec
            evli = 1/S.evl
            Dout = D1/(bt*(S.evl[:, None] * evli[None, :]) +
                       bt*(evli[:, None] * S.evl[None, :]) + al1 - 2*bt)

            return Bout, S.evec@Dout@S.evec.T.conjugate()
        
        def NTg(man, S, U):
            al0, al1 = man.alpha
            bt = man.beta
            Y0 = S.Y0
            NTg_B = al0*Y0.T.conjugate()@U.tY
            NTg_D = S.Pinv@U.tP@S.Pinv
            NTg_D += S.Pinv@S.Y.T.conjugate()@U.tY -\
                S.Y.T.conjugate()@U.tY@S.Pinv
            NTg_D = al1*bt*hsym(NTg_D)
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
            hsym(S.P + E.tP + 0.5*E.tP @ S.Pinv @ E.tP))

    def rand(self):
        # Generate random  point using qr of random normally distributed
        # matrix.
        Y, _ = la.qr(crandn(
            self.n, self.p))
        P = crandn(self.p, self.p)
        return psd_point(Y, P@P.T.conjugate())
    
    def randvec(self, X):
        """Random tangent vector at point X
        """

        """
        U = np.random.randn(self._dim)
        U = U / self.norm(X, U)
        return U
        """
        U = self.proj(X, self._rand_ambient())
        nrm = self.norm(X, U)
        return psd_ambient(U.tY/nrm, U.tP/nrm)

    def _rand_ambient(self):
        return psd_ambient(crandn(self.n, self.p), crandn(self.p, self.p))

    def _rand_range_J(self):
        a = {}
        w1 = crandn(self.p, self.p)
        a['P'] = w1 - w1.T.conjugate()
        a['YP'] = crandn(self.p, self.p)
        return a

    def _vec(self, E):
        """vectorize. This is usually used for sanity test in low dimension
        typically X.reshape(-1). For exampe, we can test J, g by representing
        them as matrices.
        Convenient for testing but dont expect much actual use
        """
        return np.concatenate(
            [cvec(E.tY), cvec(E.tP)])

    def _unvec(self, vec):
        """reshape
        """
        return psd_ambient(cunvec(vec[:self.tdim_St], (self.n, self.p)),
                           cunvec(vec[self.tdim_St:], (self.p, self.p)))

    def _vec_range_J(self, a):
        """vectorize an elememt of rangeJ
        a.reshape(-1) to a vector
        """
        ret = zeros(self.codim)
        start = 0
        tp = cvecah(a['P'])

        ret[start:start+tp.shape[0]] = tp
        start += tp.shape[0]
        ret[start:] = cvec(a['YP'])
        return ret

    def _unvec_range_J(self, vec):
        a = {}
        a['P'] = cunvecah(vec[:self._codim_P])
        a['YP'] = cunvec(vec[:self.codim_P], (self.n, self.d))
        return a
    
