import numpy as np
from numpy.random import (randint, randn)
from numpy import zeros, zeros_like, trace, allclose

from ManNullRange.manifolds.RealFlag import RealFlag
from test_tools import check_zero, make_sym_pos, random_orthogonal


def test_inner(man, Y):
    for i in range(10):
        eta1 = man._rand_ambient()
        eta2 = man._rand_ambient()
        """
        alpha = man.alpha
        inn1 = alpha[0]*trace(eta1.T @ eta2) +\
            (alpha[1]-alpha[0])*trace((eta1.T@Y) @ (Y.T@eta2))
        assert(allclose(man.inner(Y, eta1, eta2), inn1))
        """
        eta2a = man.g_inv(Y, eta2)
        v1 = man.inner(Y, eta1, eta2a)
        v2 = trace(eta1 @ eta2.T)
        assert(allclose(v1, v2))
    print(True)


def test_J(man, Y):
    alpha = man.alpha
    
    def diff_i_j(i, j):
        U = zeros_like(Y)
        U[ii, jj] = 1
        ju = man.J(Y, U)
        asum = 0
        gdc = man._g_idx
        for r, s in ju:
            br, er = gdc[r]
            if r == s:
                asum += check_zero(
                    ju[r, r] - alpha[r-1, r]*Y[:, br:er].T @ U[:, br:er])
            else:
                bs, es = gdc[s]
                asum += check_zero(ju[r, s] -
                                   Y[:, br:er].T @ U[:, bs:es] -
                                   U[:, br:er].T @ Y[:, bs:es])
        return asum / len(ju)

    diffs = zeros_like(Y)
    for ii in range(Y.shape[0]):
        for jj in range(Y.shape[1]):
            diffs[ii, jj] = diff_i_j(ii, jj)
    try:
        assert(allclose(diffs, 0))
        print(True)
    except Exception:
        print(False)


def test_Jst(man, Y, jmat):
    print("test JST")
    for ii in range(10):
        a = man._rand_range_J()
        avec = man._vec_range_J(a)
        jtout = man._unvec(jmat.T @ avec)
        jtout2 = man.Jst(Y, a)
        # print(np.where(np.abs(jtout - jtout2) > 1e-9))
        diff = check_zero(jtout-jtout2)
        print(diff)


def misc_test_cg(Y, b):
    from scipy.sparse.linalg import LinearOperator, cg
    Amat = np.eye(Y.shape[0]) - .5*Y @ Y.T
    
    def Afunc(x):
        return (Amat @ x.reshape(Y.shape)).reshape(-1)
    tdim = np.prod(Y.shape)
    A = LinearOperator(dtype=float, shape=(tdim, tdim), matvec=Afunc)
    print(cg(A, b.reshape(-1)))
        

def make_j_mat(man, Y):
    codim = man.codim
    ret = zeros((codim, np.prod(Y.shape)))
    for ii in range(ret.shape[1]):
        ee = zeros(ret.shape[1])
        ee[ii] = 1
        ret[:, ii] = man._vec_range_J(
            man.J(Y, man._unvec(ee)))
    return ret


def make_g_inv_mat(man, Y):
    nn = np.prod(Y.shape)
    ret = zeros((nn, nn))
    for ii in range(ret.shape[1]):
        ee = zeros(ret.shape[1])
        ee[ii] = 1
        ret[:, ii] = man.g_inv(Y, ee.reshape(Y.shape)).reshape(-1)
    return ret
    

def test_projection(man, Y):
    print("test projection")
    N = 10
    ret = np.zeros(N)

    for i in range(N):
        U = man._rand_ambient()
        Upr = man.proj(Y, U)
        ret[i] = check_zero(man._vec_range_J(
            man.J(Y, Upr)))
    # print(ret)
    
    if check_zero(ret) > 1e-9:
        print("not all works")
    else:
        print("All good")
    print("test randvec")
    ret_pr = np.zeros(N)
    for i in range(N):
        XX = man.rand()
        H = man.randvec(XX)
        U = man._rand_ambient()
        Upr = man.proj(XX, U)
        ret_pr[i] = man.inner(XX, U, H) - man.inner(XX, Upr, H)
        ret[i] = (check_zero(XX.T @ H + H.T @ XX))
    print(ret)
    if check_zero(ret) > 1e-9:
        print("not all works")
    else:
        print("All good")
    print("check inner of projection = inner of original")
    print(ret_pr)
    if check_zero(ret_pr) > 1e-9:
        print("not all works")
    else:
        print("All good")


def test_all_projections():
    dvec = np.array([10, 3, 2, 3])
    p = dvec.shape[0]-1
    alpha = randint(1, 10, (p, p+1)) * .1
    man = RealFlag(dvec, alpha=alpha)
    print(man)
    Y = man.rand()
    U = man._rand_ambient()
    Upr = man.proj(Y, U)
    
    test_inner(man, Y)
    test_J(man, Y)
            
    # now check metric, Jst etc
    # check Jst: vectorize the operator J then compare Jst with jmat.T
    jmat = make_j_mat(man, Y)
    test_Jst(man, Y, jmat)
    ginv_mat = make_g_inv_mat(man, Y)
    # test g_inv_Jst
    for ii in range(10):
        a = man._rand_range_J()
        avec = man._vec_range_J(a)
        jtout = (ginv_mat @ jmat.T @ avec).reshape(Y.shape)
        
        jtout2 = man.g_inv_Jst(Y, a)
        diff = check_zero(jtout-jtout2)
        print(diff)
    # test projection
    test_projection(man, Y)

    for i in range(20):
        Uran = man._rand_ambient()
        Upr = man.proj(Y, man.g_inv(Y, Uran))
        Upr2 = man.proj_g_inv(Y, Uran)
        print(check_zero(Upr-Upr2))

    for ii in range(10):
        a = man._rand_range_J()
        xi = man._rand_ambient()
        jtout2 = man.Jst(Y, a)
        dlt = 1e-7
        Ynew = Y + dlt*xi
        jtout2a = man.Jst(Ynew, a)
        d1 = (jtout2a - jtout2)/dlt
        d2 = man.D_Jst(Y, xi, a)
        print(check_zero(d2-d1))

    for ii in range(10):
        Y = man.rand()
        eta = man._rand_ambient()
        xi = man.randvec(Y)
        a1 = man.J(Y, eta)
        dlt = 1e-7
        Ynew = Y + dlt*xi
        a2 = man.J(Ynew, eta)
        d1 = (man._vec_range_J(a2) -
              man._vec_range_J(a1))/dlt
        d2 = man._vec_range_J(man.D_J(Y, xi, eta))
        print(check_zero(d2-d1))

    for ii in range(10):
        a = man._rand_range_J()
        xi = man._rand_ambient()
        jtout2 = man.g_inv_Jst(Y, a)
        dlt = 1e-7
        Ynew = Y + dlt*xi
        jtout2a = man.g_inv_Jst(Ynew, a)
        d1 = (jtout2a - jtout2)/dlt
        d2 = man.D_g_inv_Jst(Y, xi, a)
        print(check_zero(d2-d1))
        
    for ii in range(10):
        arand = man._rand_range_J()
        a2 = man.solve_J_g_inv_Jst(Y, arand)
        a1 = man.J(Y, man.g_inv_Jst(Y, a2))
        print(check_zero(man._vec_range_J(a1) -
                         man._vec_range_J(arand)))

    # derives
    for ii in range(10):
        Y1 = man.rand()
        xi = man.randvec(Y1)
        omg1 = man._rand_ambient()
        omg2 = man._rand_ambient()
        dlt = 1e-7
        Y2 = Y1 + dlt*xi
        p1 = man.inner(Y1, omg1, omg2)
        p2 = man.inner(Y2, omg1, omg2)
        der1 = (p2-p1)/dlt
        der2 = man.base_inner_ambient(
            man.D_g(Y1, xi, omg2), omg1)
        print(check_zero(der1-der2))

    # cross term for christofel
    for i in range(10):
        Y1 = man.rand()
        xi = man.randvec(Y1)
        omg1 = man._rand_ambient()
        omg2 = man._rand_ambient()
        dr1 = man.D_g(Y1, xi, omg1)
        x12 = man.contract_D_g(Y1, omg1, omg2)

        p1 = trace(dr1 @ omg2.T)
        p2 = trace(x12 @ xi.T)
        print(p1, p2, p1-p2)

    # now test christofel:
    # two things: symmetric on vector fields
    # and christofel relation
    # in the case metric
    for i in range(10):
        Y1 = man.rand()
        xi = man.randvec(Y1)
        eta1 = man.randvec(Y1)
        eta2 = man.randvec(Y1)
        p1 = man.proj_g_inv(Y1, man.christoffel_form(Y1, xi, eta1))
        p2 = man.proj_g_inv(Y1, man.christoffel_form(Y1, eta1, xi))
        print(check_zero(p1-p2))
        v1 = man.base_inner_ambient(
            man.christoffel_form(Y1, eta1, eta2), xi)
        v2 = man.base_inner_ambient(man.D_g(Y1, eta1, eta2), xi)
        v3 = man.base_inner_ambient(man.D_g(Y1, eta2, eta1), xi)
        v4 = man.base_inner_ambient(man.D_g(Y1, xi, eta1), eta2)
        print(v1, 0.5*(v2+v3-v4), v1-0.5*(v2+v3-v4))
        """
        p2 = man.christoffel_form_explicit(
            Y1, xi, eta)
        """
        

def test_christ_flat():
    """now test that christofel preserve metrics:
    on the flat space
    d_xi <v M v> = 2 <v M nabla_xi v>
     v = proj(W) @ (aa W + b)
    """
    dvec = np.array([10, 3, 2, 3])
    p = dvec.shape[0]-1
    alpha = randint(1, 10, (p, p+1)) * .1
    man = RealFlag(dvec, alpha=alpha)
    Y = man.rand()
    n = man.n
    d = man.d
    
    xi = man.randvec(Y)
    aa = np.random.randn(n*d, n*d)
    bb = np.random.randn(n*d)
    
    def v_func_flat(Y):
        return (aa @ Y.reshape(-1) + bb).reshape(n, d)

    vv = v_func_flat(Y)
    dlt = 1e-7
    Ynew = Y + dlt * xi
    vnew = v_func_flat(Ynew)

    val = man.inner(Y, vv, vv)
    valnew = man.inner(Ynew, vnew, vnew)
    d1 = (valnew - val)/dlt
    dv = (vnew - vv) / dlt
    nabla_xi_v = dv + man.g_inv(
        Y, man.christoffel_form(Y, xi, vv))
    d2 = man.inner(Y, vv, nabla_xi_v)

    print(d1)
    print(2*d2)


def calc_covar_numeric(man, Y, xi, v_func):
    """ compute nabla on E dont do the metric
    lower index. So basically
    Nabla (Pi e).
    Thus, if we want to do Nabla Pi g_inv df
    We need to send g_inv df
    """

    def vv_func(W):
        return man.proj(W, v_func(W))
    
    vv = vv_func(Y)

    dlt = 1e-7
    Wnew = Y + dlt * xi
    vnew = vv_func(Wnew)

    val = man.inner_product_amb(Y, vv)
    valnew = man.inner_product_amb(
        Wnew, vnew)
    d1 = (valnew - val)/dlt
    dv = (vnew - vv) / dlt
    cx = man.christoffel_form(Y, xi, vv)
    nabla_xi_v_up = dv + man.g_inv(Y, cx)
    nabla_xi_v = man.proj(Y, nabla_xi_v_up)
    if False:
        d2 = man.inner_product_amb(Y, vv, nabla_xi_v)
        d2up = man.inner_product_amb(
            Y, vv, nabla_xi_v_up)

        print(d1)
        print(2*d2up)
        print(2*d2)
    return nabla_xi_v, dv, cx


def test_chris_vectorfields():
    # now test that it works on embedded metrics
    # we test that D_xi (eta g eta) = 2(eta g nabla_xi eta)
    dvec = np.array([10, 3, 2, 3])
    p = dvec.shape[0]-1
    alpha = randint(1, 10, (p, p+1)) * .1
    man = RealFlag(dvec, alpha=alpha)
    n = man.n
    d = man.d
    
    slp = randn(n*d)
    Y0 = man.rand()
    slpxi = randn(n*d)

    aa = randn(n*d, n*d)
    aaxi = randn(n*d, n*d)

    def v_func(Y):
        return man.proj(Y, (aa @ (Y-Y0).reshape(-1)
                            + slp).reshape(n, d))

    YY = Y0.copy()
    xi = man.proj(YY, slpxi.reshape(n, d))

    nabla_xi_v, dv, cxv = calc_covar_numeric(
        man, YY, xi, v_func)

    def xi_func(Y):
        return man.proj(Y, (aaxi @ (Y-Y0).reshape(-1)
                            + slpxi).reshape(n, d))

    vv = v_func(YY)

    nabla_v_xi, dxi, cxxi = calc_covar_numeric(
        man, YY, vv, xi_func)
    diff = nabla_xi_v - nabla_v_xi
    # print(diff)
    # now do Lie bracket:
    dlt = 1e-7
    YnewXi = YY + dlt * xi
    Ynewvv = YY + dlt * vv
    vnewxi = v_func(YnewXi)
    xnewv = xi_func(Ynewvv)
    dxiv = (vnewxi - vv)/dlt
    dvxi = (xnewv - xi)/dlt
    diff2 = man.proj(YY, dxiv-dvxi)
    print(check_zero(diff - diff2))
                            
            
def test_covariance_deriv():
    # now test full:
    # do covariant derivatives
    # check that it works, preseving everything
    dvec = np.array([10, 3, 2, 3])
    p = dvec.shape[0]-1
    alpha = randint(1, 10, (p, p+1)) * .1
    man = RealFlag(dvec, alpha=alpha)
    n = man.n
    d = man.d
    Y = man.rand()

    slp = np.random.randn(n*d)
    aa = np.random.randn(n*d, n*d)

    def omg_func(Y):
        return (aa @ Y.reshape(-1) + slp).reshape(n, d)

    xi = man.randvec(Y)

    egrad = omg_func(Y)
    ehess = (aa @ xi.reshape(-1)).reshape(n, d)

    val1 = man.ehess2rhess(Y, egrad, ehess, xi)
    
    def rgrad_func(W):
        return man.proj_g_inv(W, omg_func(W))
    
    if False:
        d_xi_rgrad = num_deriv(man, Y, xi, rgrad_func)
        rgrad = man.proj_g_inv(Y, egrad)
        fourth = man.christoffel_form(Y, xi, rgrad)
        val1c = man.proj(Y, d_xi_rgrad) + man.proj_g_inv(Y, fourth)

    if False:
        first = ehess
        a = man.J(Y, man.g_inv(Y, egrad))
        rgrad = man.proj_g_inv(Y, egrad)
        second = - man.D_g(Y, xi, man.g_inv(Y, egrad))
        aout = man.solve_J_g_inv_Jst(Y, a)
        third = - man.proj(Y, man.D_g_inv_Jst(Y, xi, aout))
        fourth = man.christoffel_form(Y, xi, rgrad)
        val1a = man.proj_g_inv(Y, first + second + fourth) + third

    d_xi_rgrad = num_deriv(man, Y, xi, rgrad_func)
    rgrad = man.proj_g_inv(Y, egrad)
    fourth = man.christoffel_form(Y, xi, rgrad)
    val1b = man.proj(Y, d_xi_rgrad) + man.proj_g_inv(Y, fourth)
    print(check_zero(val1-val1b))
    # nabla_v_xi, dxi, cxxi
    # val2a, _, _ = calc_covar_numeric(man, Y, xi, omg_func)
    val2, _, _ = calc_covar_numeric(man, Y, xi, rgrad_func)
    # val2_p = project(prj, val2)
    val2_p = man.proj(Y, val2)
    # print(val1)
    # print(val2_p)
    print(val1-val2_p)

    
def num_deriv(man, W, xi, func, dlt=1e-7):
    Wnew = W + dlt*xi
    return (func(Wnew) - func(W))/dlt


def test_rhess_02():
    np.random.seed(0)
    dvec = np.array([10, 3, 2, 3])
    p = dvec.shape[0]-1
    alpha = randint(1, 10, (p, p+1)) * .1
    man = RealFlag(dvec, alpha=alpha)
    n = man.n
    d = man.d

    Y = man.rand()
    UU = {}
    p = alpha.shape[0]
    VV = {}
    gidx = man._g_idx

    for rr in range(p):
        UU[rr] = make_sym_pos(n)
        VV[rr] = randn(n, dvec[rr+1])

    def f(Y):
        ss = 0
        for rr in range(p):
            br, er = gidx[rr+1]
            wr = Y[:, br:er]
            ss += trace(UU[rr] @ wr @ wr.T)
        return ss

    def df(W):
        ret = np.zeros_like(W)
        for rr in range(p):
            br, er = gidx[rr+1]
            wr = W[:, br:er]
            ret[:, br:er] += 2 * UU[rr] @ wr
        return ret

    def ehess_form(W, xi, eta):
        ss = 0
        for rr in range(p):
            br, er = gidx[rr+1]
            ss += 2 * trace(UU[rr] @ xi[:, br:er] @ eta[:, br:er].T)
        return ss

    def ehess_vec(W, xi):
        ret = np.zeros_like(W)
        for rr in range(p):
            br, er = gidx[rr+1]
            ret[:, br:er] += 2 * UU[rr] @ xi[:, br:er]
        return ret    

    xxi = randn(n, d)
    dlt = 1e-8
    Ynew = Y + dlt*xxi
    d1 = (f(Ynew) - f(Y))/dlt
    d2 = df(Y)
    print(d1 - trace(d2 @ xxi.T))

    eeta = randn(n, d)

    d1 = trace((df(Ynew) - df(Y)) @ eeta.T) / dlt
    ehess_val = ehess_form(Y, xxi, eeta)
    # ehess_val2 = ehess_form(Y, eeta, xxi)
    dv2 = ehess_vec(Y, xxi)
    print(trace(dv2 @ eeta.T))
    print(d1, ehess_val, d1-ehess_val)

    # now check the formula: ehess = xi (eta_func(f)) - <D_xi eta, df(Y)>
    # promote eta to a vector field.

    m1 = randn(n, n)
    m2 = randn(d, d)

    def eta_field(Yin):
        return m1 @ (Yin - Y) @ m2 + eeta

    # xietaf: should go to ehess(xi, eta) + df(Y) @ etafield)
    xietaf = trace(df(Ynew) @ eta_field(Ynew).T -
                   df(Y) @ eta_field(Y).T) / dlt
    # appy eta_func to f: should go to tr(m1 @ xxi @ m2 @ df(Y).T)
    Dxietaf = trace((eta_field(Ynew) - eta_field(Y)) @ df(Y).T)/dlt
    # this is ehess. should be same as d1 or ehess_val
    print(xietaf-Dxietaf)
    print(xietaf-Dxietaf-ehess_val)

    # now check: rhess. Need to make sure xi, eta in the tangent space.
    # first compare this with numerical differentiation
    xi1 = man.proj(Y, xxi)
    eta1 = man.proj(Y, eeta)
    egvec = df(Y)
    ehvec = ehess_vec(Y, xi1)
    rhessvec = man.ehess2rhess(Y, egvec, ehvec, xi1)

    # check it numerically:
    def rgrad_func(Y):
        return man.proj_g_inv(Y, df(Y))
    
    val2, _, _ = calc_covar_numeric(man, Y, xi1, rgrad_func)
    val2_p = man.proj(Y, val2)
    # print(rhessvec)
    # print(val2_p)
    print(check_zero(rhessvec-val2_p))
    rhessval = man.inner(Y, rhessvec, eta1)
    print(man.inner(Y, val2, eta1))
    print(rhessval)

    # check symmetric:
    ehess_valp = ehess_form(Y, xi1, eta1)
    ehvec_e = ehess_vec(Y, eta1)
    rhessvec_e = man.ehess2rhess(Y, egvec, ehvec_e, eta1)
    rhessval_e = man.inner(Y, rhessvec_e, xi1)
    rhessval_e1 = man.rhess02(Y, xi1, eta1, egvec, ehess_valp)
    # rhessval_e2 = man.rhess02_alt(Y, xi1, eta1, egvec, trace(ehvec@eta1.T))
    # print(rhessval_e, rhessval_e1, rhessval_e2)
    print(rhessval_e, rhessval_e1)
    
    print('rhessval_e %f ' % rhessval_e)
    # the above computed inner_prod(Nabla_xi Pi * df, eta)
    # in the following check. Extend eta1 to eta_proj
    # (Pi Nabla_hat Pi g_inv df, g eta)
    # = D_xi (Pi g_inv df, g eta) - (Pi g_inv df g Pi Nabla_hat eta)
    
    def eta_proj(Y):
        return man.proj(Y, eta_field(Y))
    print(check_zero(eta1-eta_proj(Y)))
    
    e1 = man.inner(Y, man.proj_g_inv(Y, df(Y)), eta_proj(Y))
    e1a = trace(df(Y) @ eta_proj(Y).T)
    print(e1, e1a, e1-e1a)
    Ynew = Y + xi1*dlt
    e2 = man.inner(Ynew, man.proj_g_inv(Ynew, df(Ynew)), eta_proj(Ynew))
    e2a = trace(df(Ynew) @ eta_proj(Ynew).T)
    print(e2, e2a, e2-e2a)
    
    first = (e2 - e1)/dlt
    first1 = trace(df(Ynew) @ eta_proj(Ynew).T - df(Y) @ eta_proj(Y).T)/dlt
    print(first-first1)
    
    val3, _, _ = calc_covar_numeric(man, Y, xi1, eta_proj)
    second = man.inner(Y, man.proj_g_inv(Y, df(Y)), man.proj(Y, val3))
    second2 = man.inner(Y, man.proj_g_inv(Y, df(Y)), val3)
    print(second, second2, second-second2)
    print('same as rhess_val %f' % (first-second))


def make_simple_alpha(alpha0, beta):
    """Make the alpha_{rl} dependent on alpha_{t0} and beta only
    """
    p = alpha0.shape[0]
    alpha = zeros((p, p+1))
    alpha[:, 0] = alpha0
    for i in range(p):
        for j in range(i, p):
            alpha[i, j+1] = alpha0[i]/2 + alpha0[j]/2 + beta/2
            alpha[j, i+1] = alpha0[i]/2 + alpha0[j]/2 + beta/2
    return alpha

    
def optim_test():
    from pymanopt import Problem
    from pymanopt.solvers import TrustRegions
    from pymanopt.function import Callable

    n = 1000

    # problem Tr(AXBX^T)
    for i in range(1):
        dvec = np.array([0, 30, 2, 1])
        dvec[0] = 1000 - dvec[1:].sum()
        d = dvec[1:].sum()
        D = randint(1, 10, n) * 0.02 + 1
        OO = random_orthogonal(n)
        A = OO @ np.diag(D) @ OO.T
        
        B = make_sym_pos(d)
        p = dvec.shape[0]-1
        alpha = randint(1, 10, (p, p+1)) * .1
        alpha0 = randint(1, 10, (p))
        # alpha0 = randint(1, 2, (p))
        alpha = make_simple_alpha(alpha0, 0)
        man = RealFlag(dvec, alpha=alpha)

        @Callable
        def cost(X):
            return trace(A @ X @ B @ X.T)
        
        @Callable
        def egrad(X):
            return 2*A @ X @ B

        @Callable
        def ehess(X, H):
            return 2*A @ H @ B

        if False:
            X = man.rand()
            xi = man.randvec(X)
            d1 = num_deriv(man, X, xi, cost)
            d2 = trace(egrad(X) @ xi.T)
            print(check_zero(d1-d2))
        
        prob = Problem(
            man, cost, egrad=egrad)
        XInit = man.rand()

        prob = Problem(
            man, cost, egrad=egrad, ehess=ehess)

        solver = TrustRegions(maxtime=100000, maxiter=100)
        opt = solver.solve(prob, x=XInit, Delta_bar=250)
        print(cost(opt))

        if False:
            min_val = 1e190
            # min_X = None
            for i in range(100):
                Xi = man.rand()
                c = cost(Xi)
                if c < min_val:
                    # min_X = Xi
                    min_val = c
                if i % 1000 == 0:
                    print('i=%d min=%f' % (i, min_val))
            print(min_val)
        alpha_c = alpha.copy()
        alpha_c[:] = 1
        man1 = RealFlag(dvec, alpha=alpha_c)
        prob = Problem(
            man1, cost, egrad=egrad, ehess=ehess)

        solver = TrustRegions(maxtime=100000, maxiter=100)
        opt = solver.solve(prob, x=XInit, Delta_bar=250)
        alpha_c5 = alpha_c.copy()
        alpha_c5[:, 1:] = .5
        man1 = RealFlag(dvec, alpha=alpha_c5)
        prob = Problem(
            man1, cost, egrad=egrad, ehess=ehess)

        solver = TrustRegions(maxtime=100000, maxiter=100)
        opt = solver.solve(prob, x=XInit, Delta_bar=250)


if __name__ == '__main__':
    optim_test()
        
