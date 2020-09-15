from __future__ import division
from ManNullRange import NullRangeManifold
import numpy as np
import numpy.linalg as la
from numpy import zeros, zeros_like
from .tools import crandn, rtrace


if not hasattr(__builtins__, "xrange"):
    xrange = range


def _calc_dim(dvec):
    """ Calculate the real dimension
    """
    s = 0
    for i in range(1, len(dvec)):
        for j in range(i):
            s += dvec[i]*dvec[j]
    return s*2
    

class ComplexFlag(NullRangeManifold):
    """Class for a Complex Flag manifold
    Block matrix Y with Y.T.conj() @ Y = I
    dvec is a vector defining the blocks of Y
    dvec of size p
    Y of dimension n*d

    n = dvec.sum()
    d = dvec[1:].sum()

    Metric is defined by a sets of parameters alpha of size (p-1)p

    Parameters
    ----------
    dvec     : vector defining the block size
    alpha    : array of size (p-1)p, p = dvec.shape[0]
               Defining a metric on the Flag manifold
               alpha  > 0

    """
    
    def __init__(self, dvec, alpha=None):
        self._point_layout = 1
        self.dvec = np.array(dvec)
        self.n = dvec.sum()
        self.d = dvec[1:].sum()
        self._name = "Complex flag manifold dvec=(%s)" % self.dvec
        self._dimension = _calc_dim(dvec)
        self._codim = 2*self.d * self.n - self._dimension
        cs = dvec[:].cumsum() - dvec[0]
        self._g_idx = dict((i+1, (cs[i], cs[i+1]))
                           for i in range(cs.shape[0]-1))
        p = self.dvec.shape[0]-1
        self.p = p
        if alpha is None:
            self.alpha = np.full((p, p+1), fill_value=1/2)
            self.alpha[:, 0] = 1
        else:
            self.alpha = alpha

    def inner(self, X, Ba, Bb=None):
        """ Inner product (Riemannian metric) on the tangent space.
        The tangent space is given as a matrix of size mm_degree * m
        """
        gdc = self._g_idx
        alpha = self.alpha
        p = self.dvec.shape[0]-1
        s2 = 0
        if Bb is None:
            Bb = Ba
        for rr in range(p, 0, -1):
            br, er = gdc[rr]
            ss = rtrace(alpha[rr-1, 0] * Ba[:, br:er] @
                        Bb[:, br:er].T.conj())
            s2 += ss

            for jj in range(1, p+1):
                bj, ej = gdc[jj]
                ss = rtrace(
                    (alpha[rr-1, jj] - alpha[rr-1, 0]) * (
                        (Ba[:, br:er].T.conj() @ X[:, bj:ej]) @
                        (X[:, bj:ej].T.conj() @ Bb[:, br:er])))
                s2 += ss
        return s2
    
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

    def base_inner_ambient(X, eta1, eta2):
        return rtrace(eta1.T.conj() @ eta2)

    def base_inner_E_J(X, a1, a2):
        raise rtrace(a1.T.conj() @ a2)
    
    def g(self, X, omg):
        gdc = self._g_idx
        alpha = self.alpha
        ret = zeros_like(omg, dtype=np.complex)
        p = self.p
        for rr in range(1, p+1):
            br, er = gdc[rr]
            ret[:, br:er] = alpha[rr-1, 0]*omg[:, br:er]
            for jj in range(1, p+1):
                bj, ej = gdc[jj]
                ret[:, br:er] += (alpha[rr-1, jj]-alpha[rr-1, 0]) *\
                    (X[:, bj:ej] @ (X[:, bj:ej].T.conj() @ omg[:, br:er]))
        
        return ret
        
    def g_inv(self, X, omg):
        gdc = self._g_idx
        alpha = self.alpha
        ret = zeros_like(omg, dtype=np.complex)
        p = self.p
        for rr in range(1, p+1):
            br, er = gdc[rr]
            ret[:, br:er] = 1/alpha[rr-1, 0]*omg[:, br:er]
            for jj in range(1, p+1):
                bj, ej = gdc[jj]
                ret[:, br:er] += (1/alpha[rr-1, jj]-1/alpha[rr-1, 0]) *\
                    (X[:, bj:ej] @ (X[:, bj:ej].T.conj() @ omg[:, br:er]))
        
        return ret
    
    def J(self, X, eta):
        ret = {}
        ph = self.dvec
        gidx = self._g_idx
        alpha = self.alpha
        p = self.dvec.shape[0]-1
        for r in range(p, 0, -1):
            if ph[r] == 0:
                continue
            r_g_beg, r_g_end = gidx[r]
            for s in range(p, 0, -1):
                if ph[s] == 0:
                    continue
                s_g_beg, s_g_end = gidx[s]
                if r == s:
                    ret[r, r] = alpha[r-1, r] *\
                                X[:, r_g_beg:r_g_end].T.conj() @\
                                eta[:, r_g_beg:r_g_end]

                elif s > r:
                    ret[r, s] = eta[:, r_g_beg:r_g_end].T.conj() @\
                        X[:, s_g_beg:s_g_end]

                    ret[r, s] += X[:, r_g_beg:r_g_end].T.conj() @\
                        eta[:, s_g_beg:s_g_end]
        return ret

    def Jst(self, X, a):
        ret = zeros_like(X, dtype=np.complex)
        alpha = self.alpha
        for r, s in a:
            br, er = self._g_idx[r]
            if r == s:
                ret[:, br:er] += alpha[r-1, r] * X[:, br:er] @ a[r, r]
            else:
                bs, es = self._g_idx[s]
                ret[:, br:er] += X[:, bs:es] @ a[r, s].T.conj()
                ret[:, bs:es] += X[:, br:er] @ a[r, s]
        return ret

    def g_inv_Jst(self, X, a):
        ret = np.zeros_like(X, dtype=np.complex)
        alpha = self.alpha
        for r, s in a:
            br, er = self._g_idx[r]
            if r == s:
                ret[:, br:er] += X[:, br:er] @ a[r, r]
            else:
                bs, es = self._g_idx[s]
                ret[:, br:er] += 1/alpha[r-1, s] * X[:, bs:es] @\
                    a[r, s].T.conj()
                ret[:, bs:es] += 1/alpha[s-1, r] * X[:, br:er] @ a[r, s]
        return ret

    def D_g(self, X, xi, eta):
        gdc = self._g_idx
        alpha = self.alpha
        ret = zeros_like(eta)
        p = self.p
        for rr in range(1, p+1):
            br, er = gdc[rr]
            for jj in range(1, p+1):
                bj, ej = gdc[jj]
                ret[:, br:er] += (alpha[rr-1, jj]-alpha[rr-1, 0]) *\
                    xi[:, bj:ej] @ (X[:, bj:ej].T.conj() @ eta[:, br:er])
                ret[:, br:er] += (alpha[rr-1, jj]-alpha[rr-1, 0]) *\
                    X[:, bj:ej] @ (xi[:, bj:ej].T.conj() @ eta[:, br:er])
        return ret

    def christoffel_form(self, X, xi, eta):
        ret = 0.5*self.D_g(X, xi, eta)
        ret += 0.5*self.D_g(X, eta, xi)
        ret -= 0.5*self.contract_D_g(X, xi, eta)
        return ret

    def D_J(self, X, xi, eta):
        """ Derivatives of J
        """
        ret = {}
        ph = self.dvec
        gidx = self._g_idx
        alpha = self.alpha
        p = self.p
        for r in range(p, 0, -1):
            if ph[r] == 0:
                continue
            r_g_beg, r_g_end = gidx[r]
            for s in range(p, 0, -1):
                if ph[s] == 0:
                    continue
                s_g_beg, s_g_end = gidx[s]
                if r == s:
                    ret[r, r] = alpha[r-1, r] *\
                        xi[:, r_g_beg:r_g_end].T.conj() @\
                        eta[:, r_g_beg:r_g_end]

                elif s > r:
                    ret[r, s] = eta[:, r_g_beg:r_g_end].T.conj() @\
                        xi[:, s_g_beg:s_g_end]

                    ret[r, s] += xi[:, r_g_beg:r_g_end].T.conj() @\
                        eta[:, s_g_beg:s_g_end]
        return ret
    
    def D_Jst(self, X, xi, a):
        ret = np.zeros_like(xi)

        alpha = self.alpha
        for r, s in a:
            br, er = self._g_idx[r]
            if r == s:
                ret[:, br:er] += alpha[r-1, r] * xi[:, br:er] @ a[r, r]
            else:
                bs, es = self._g_idx[s]
                ret[:, br:er] += xi[:, bs:es] @ a[r, s].T.conj()
                ret[:, bs:es] += xi[:, br:er] @  a[r, s]
        return ret

    def D_g_inv_Jst(self, Y, xi, a):
        ret = zeros_like(Y)
        alpha = self.alpha
        for r, s in a:
            br, er = self._g_idx[r]
            if r == s:
                ret[:, br:er] += xi[:, br:er] @ a[r, r]
            else:
                bs, es = self._g_idx[s]
                ret[:, br:er] += 1/alpha[r-1, s] * xi[:, bs:es] @\
                    a[r, s].T.conj()
                ret[:, bs:es] += 1/alpha[s-1, r] * xi[:, br:er] @ a[r, s]
        return ret
    
    def contract_D_g(self, X, xi, eta):
        ret = zeros_like(eta)
        alpha = self.alpha
        gidx = self._g_idx
        p = self.p
        for r in range(1, p+1):
            br, er = gidx[r]
            for jj in range(1, p+1):
                bj, ej = gidx[jj]
                ret[:, br:er] += (alpha[jj-1, r] - alpha[jj-1, 0])*(
                    eta[:, bj:ej] @ (xi[:, bj:ej].T.conj()@X[:, br:er]) +
                    xi[:, bj:ej] @ (eta[:, bj:ej].T.conj()@X[:, br:er]))
        return ret
    
    def st(self, mat):
        """The split_transpose. transpose if real, hermitian transpose if complex
        """
        return mat.T.conj()

    def J_g_inv_Jst(self, X, a):
        raise NotImplementedError

    def solve_J_g_inv_Jst(self, X, b):
        alf = 1/self.alpha
        a = dict()
        for r in range(1, alf.shape[1]):
            a[r, r] = alf[r-1, r] * b[r, r]
            for s in range(r+1, alf.shape[1]):
                a[r, s] = 1/(alf[r-1, s] + alf[s-1, r])*b[r, s]
        return a
    
    def proj(self, X, U):
        """projection. U is in ambient
        return one in tangent
        """
        ret = zeros_like(U)
        alpha = self.alpha
        p = self.p
        for tt in range(1, p+1):
            bt, et = self._g_idx[tt]
            ret[:, bt:et] = U[:, bt:et] -\
                X[:, bt:et] @ (X[:, bt:et].T.conj() @ U[:, bt:et])
            for uu in range(1, p+1):
                if uu == tt:
                    continue
                bu, eu = self._g_idx[uu]
                ft = alpha[uu-1, tt] / (alpha[uu-1, tt] + alpha[tt-1, uu])
                ret[:, bt:et] -= ft*X[:, bu:eu:] @ (
                    U[:, bu:eu:].T.conj() @ X[:, bt:et] +
                    X[:, bu:eu:].T.conj() @ U[:, bt:et])
        return ret

    def proj_g_inv(self, X, U):
        ret = zeros_like(U)
        alpha = self.alpha
        p = self.p
        for tt in range(1, p+1):
            bt, et = self._g_idx[tt]
            ret[:, bt:et] = 1/alpha[tt-1, 0] *\
                (U[:, bt:et] -
                 X @ (X.T.conj() @ U[:, bt:et]))
            for uu in range(1, p+1):
                if uu == tt:
                    continue
                bu, eu = self._g_idx[uu]
                ft = 1 / (alpha[uu-1, tt] + alpha[tt-1, uu])
                ret[:, bt:et] += ft*X[:, bu:eu:] @ (
                    X[:, bu:eu:].T.conj() @ U[:, bt:et] -
                    U[:, bu:eu:].T.conj() @ X[:, bt:et])
        return ret

    def egrad2rgrad(self, X, U):
        return self.proj_g_inv(X, U)

    def rhess02_alt(self, X, xi, eta, egrad, ehess):
        raise NotImplementedError
    
    def rhess02(self, X, xi, eta, egrad, ehess_val):
        egcoef = np.zeros_like(eta)
        ph = self.dvec
        alpha = self.alpha
        gidx = self._g_idx
        p = ph.shape[0]-1

        for tt in range(1, p+1):
            bt, et = gidx[tt]
            egcoef[:, bt:et] += X[:, bt:et] @ (xi[:, bt:et].T.conj() @
                                               eta[:, bt:et])
                
            for uu in range(1, p+1):
                if uu != tt:
                    bu, eu = gidx[uu]
                    """
                    ft = alpha[uu-1, tt]/(alpha[uu-1, tt] + alpha[tt-1, uu])
                    egcoef[bt:et, :] += ft*(
                        xi[bt:et, :] @ eta[bu:eu, :].T.conj() +
                        eta[bt:et, :] @ xi[bu:eu, :].T.conj()) @\
                        W[bu:eu, :]
                    """
                    ft2 = 0.5*(alpha[tt-1, uu]+alpha[uu-1, tt] -
                               alpha[tt-1, 0]+alpha[uu-1, 0]) /\
                        (alpha[tt-1, uu]+alpha[uu-1, tt])
                    egcoef[:, bt:et] += ft2*X[:, bu:eu] @ (
                        eta[:, bu:eu].T.conj() @ xi[:, bt:et] +
                        xi[:, bu:eu].T.conj() @ eta[:, bt:et])

        for tt in range(1, p+1):
            bt, et = self._g_idx[tt]
            for jj in range(1, p+1):
                bj, ej = self._g_idx[jj]
                ftt = 0.5*(alpha[jj-1, 0]+alpha[tt-1, 0] -
                           alpha[jj-1, tt]-alpha[tt-1, jj])

                omg_t_j = ftt*(
                    eta[:, bj:ej] @ xi[:, bj:ej].T.conj() +
                    xi[:, bj:ej] @ eta[:, bj:ej].T.conj()) @ X[:, bt:et]

                egcoef[:, bt:et] += 1/alpha[tt-1, 0] *\
                    (omg_t_j - X @ (X.T.conj() @ omg_t_j))
                for uu in range(1, p+1):
                    if uu == tt:
                        continue
                    bu, eu = self._g_idx[uu]
                    ft = 1 / (alpha[uu-1, tt] + alpha[tt-1, uu])
                    # omg_u = np.zeros_like(W[bu:eu, :])
                    ftu = 0.5*(alpha[jj-1, 0]+alpha[uu-1, 0] -
                               alpha[jj-1, uu]-alpha[uu-1, jj])
                    egcoef[:, bt:et] += ft*(ftt-ftu) *\
                        X[:, bu:eu] @ X[:, bu:eu:].T.conj() @ (
                            eta[:, bj:ej] @ xi[:, bj:ej].T.conj() @
                            X[:, bt:et] +
                            xi[:, bj:ej] @ eta[:, bj:ej].T.conj() @
                            X[:, bt:et])
                    
        return ehess_val - rtrace(egrad.T.conj() @ egcoef)
    
    def ehess2rhess(self, X, egrad, ehess, H):
        """ Convert Euclidean into Riemannian Hessian.
        ehess is the Hessian product on the ambient space
        egrad is the gradient on the ambient space
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
        """ Do svd decomposition
        """
        u, _, vh = la.svd(X+eta, full_matrices=False)
        return u @ vh

    def norm(self, X, eta):
        # Norm on the tangent space is simply the Euclidean norm.
        return np.sqrt(self.inner(X, eta, eta))

    def rand(self):
        # Generate random  point using qr of random normally distributed
        # matrix.
        O, _ = np.linalg.qr(crandn(
            self.n, self.d))
        return O
    
    def randvec(self, X):
        """Random tangent vector at point X
        """
        U = self.proj(X, self._rand_ambient())
        return U / self.norm(X, U)

    def zerovec(self, X):
        return zeros(X.shape)

    def _rand_ambient(self):
        return crandn(self.n, self.d)

    def _rand_range_J(self):
        p = self.p
        out = {}
        dv = self.dvec
        for r in range(p, 0, -1):
            for s in range(p, r-1, -1):
                out[r, s] = crandn(dv[r], dv[s])
        return out

        return crandn(self.d, self.d)

    def _vec(self, E):
        """vectorize. This is usually used for sanity test in low dimension
        typically X.reshape(-1). For exampe, we can test J, g by representing
        them as matrices.
        Convenient for testing but dont expect much actual use
        """
        return np.concatenate([E.reshape(-1).real, E.reshape(-1).imag], axis=0)

    def _unvec(self, vec):
        """reshape to shape of matrix - use unvech if hermitian,
        unvecah if anti hermitian. For testing, don't expect actual use
        """
        nd = self.n*self.d
        return vec[:nd].reshape(self.n, self.d) +\
            1j*vec[nd:].reshape(self.n, self.d)

    def _vec_range_J(self, a):
        """vectorize an elememt of rangeJ
        a.reshape(-1). Vectorize to real vector
        # first half is real second is complex
        """
        ret = zeros(self.codim)
        hdim = self.codim // 2
        start = 0
        for r, s in sorted(a, reverse=True):
            tp = a[r, s].reshape(-1)
            ret[start:start+tp.shape[0]] = tp.real
            ret[start+hdim:start+hdim+tp.shape[0]] = tp.imag
            start += tp.shape[0]
        return ret

    def _unvec_range_J(self, vec):
        """ unvectorize from real vector
        """
        dout = {}
        start = 0
        p = self.p
        dv = self.dvec
        hdim = vec.shape[0] // 2
        for r in range(p, 0, -1):
            for s in range(p, r-1, 0):
                vlen = dv[r]*dv[s]
                dout[r, s] = vec[start:start+vlen].reshape(
                    dv[r], dv[s]) + 1j*vec[start+hdim:start+hdim+vlen].reshape(
                    dv[r], dv[s])
                start += vlen
        return dout

    
