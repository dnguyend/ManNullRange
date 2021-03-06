import numpy as np
from numpy.random import (randint)
from numpy import zeros, zeros_like, allclose

from ManNullRange.manifolds.ComplexStiefel import ComplexStiefel
from test_tools import check_zero, make_sym_pos, random_orthogonal
from ManNullRange.manifolds.tools import rtrace, crandn


def test_inner(man, Y):
    for i in range(10):
        alpha = man.alpha
        eta1 = man._rand_ambient()
        eta2 = man._rand_ambient()
        inn1 = alpha[0]*rtrace(eta1.T.conj() @ eta2) +\
            (alpha[1]-alpha[0])*rtrace((eta1.T.conj()@Y) @ (Y.T.conj()@eta2))
        assert(allclose(man.inner(Y, eta1, eta2), inn1))
        
        eta2a = man.g_inv(Y, eta2)
        v1 = man.inner(Y, eta1, eta2a)
        v2 = rtrace(eta1 @ eta2.T.conj())
        assert(allclose(v1, v2))
    print(True)


def test_J(man, Y):
    def diff_i_j(i, j):
        U = zeros_like(Y, dtype=np.complex)
        U[ii, jj] = 1
        return np.mean(np.abs(man.J(Y, U) - (Y.T.conj() @ U + U.T.conj() @Y)))

    diffs = zeros_like(Y, dtype=np.complex)
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
        jtout = man._unvec(jmat.T.conj() @ avec)
        jtout2 = man.Jst(Y, a)
        # print(np.where(np.abs(jtout - jtout2) > 1e-9))
        diff = check_zero(jtout-jtout2)
        print(diff)


def misc_test_cg(Y, b):
    from scipy.sparse.linalg import LinearOperator, cg
    Amat = np.eye(Y.shape[0]) - .5*Y @ Y.T.conj()
    
    def Afunc(x):
        return (Amat @ x.reshape(Y.shape)).reshape(-1)
    tdim = np.prod(Y.shape)
    A = LinearOperator(dtype=float, shape=(tdim, tdim), matvec=Afunc)
    print(cg(A, b.reshape(-1)))
        

def make_j_mat(man, Y):
    codim = man.codim
    ret = zeros((codim, 2*np.prod(Y.shape)))
    for ii in range(ret.shape[1]):
        ee = zeros(ret.shape[1], dtype=np.complex)
        ee[ii] = 1
        ret[:, ii] = man._vec_range_J(
            man.J(Y, man._unvec(ee)))
    return ret


def make_g_inv_mat(man, Y):
    nn = 2*np.prod(Y.shape)
    ret = zeros((nn, nn))
    for ii in range(ret.shape[1]):
        ee = zeros(ret.shape[1])
        ee[ii] = 1
        ret[:, ii] = man._vec(man.g_inv(Y, man._unvec(ee)))
    return ret
    

def test_projection(man, Y):
    print("test projection")
    N = 10
    ret = np.zeros(N)

    for i in range(N):
        U = man._rand_ambient()
        Upr = man.proj(Y, U)
        ret[i] = check_zero(man.J(Y, Upr))
    print(ret)
    
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
        ret[i] = (check_zero(XX.T.conj() @ H + H.T.conj() @ XX))
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
    alpha = randint(1, 10, 2) * .1
    n = 5
    d = 3
    man = ComplexStiefel(n, d, alpha=alpha)
    Y = man.rand()

    test_inner(man, Y)
    test_J(man, Y)
            
    # now check metric, Jst etc
    # check Jst: vectorize the operator J then compare Jst with jmat.T.conj()
    jmat = make_j_mat(man, Y)
    test_Jst(man, Y, jmat)
    ginv_mat = make_g_inv_mat(man, Y)
    # test g_inv_Jst
    for ii in range(10):
        a = man._rand_range_J()
        avec = man._vec_range_J(a)
        jtout = man._unvec(ginv_mat @ jmat.T @ avec)
        
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
        d1 = (a2-a1)/dlt
        d2 = man.D_J(Y, xi, eta)
        print(check_zero(d2-d1))

    if False:
        for ii in range(10):
            omg = man._rand_ambient()
            a2 = man.J_g_inv(Y, omg)
            a1 = man.J(Y, man.g_inv(man, Y, omg))
            print(check_zero(a1-a2))

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

        p1 = rtrace(dr1 @ omg2.T.conj())
        p2 = rtrace(x12 @ xi.T.conj())
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
    alpha = randint(1, 10, 2) * .1
    n = 5
    d = 3
    man = ComplexStiefel(n, d, alpha=alpha)
    Y = man.rand()
    
    xi = man.randvec(Y)
    aa = crandn(n*d, n*d)
    bb = crandn(n*d)
    
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

    val = man.inner(Y, vv)
    valnew = man.inner(
        Wnew, vnew)
    d1 = (valnew - val)/dlt
    dv = (vnew - vv) / dlt
    cx = man.christoffel_form(Y, xi, vv)
    nabla_xi_v_up = dv + man.g_inv(Y, cx)
    nabla_xi_v = man.proj(Y, nabla_xi_v_up)
    if False:
        d2 = man.inner(Y, vv, nabla_xi_v)
        d2up = man.inner(
            Y, vv, nabla_xi_v_up)

        print(d1)
        print(2*d2up)
        print(2*d2)
    return nabla_xi_v, dv, cx


def test_chris_vectorfields():
    # now test that it works on embedded metrics
    # we test that D_xi (eta g eta) = 2(eta g nabla_xi eta)
    n, d = (5, 3)
    alpha = randint(1, 10, 2) * .1
    man = ComplexStiefel(n, d, alpha=alpha)

    slp = crandn(n*d)
    Y0 = man.rand()
    slpxi = crandn(n*d)

    aa = crandn(n*d, n*d)
    aaxi = crandn(n*d, n*d)

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
    print(diff)
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
    n, d = (5, 3)
    alpha = randint(1, 10, 2) * .1
    man = ComplexStiefel(n, d, alpha=alpha)

    Y = man.rand()

    slp = crandn(n*d)
    aa = crandn(n*d, n*d)

    def omg_func(Y):
        return (aa @ Y.reshape(-1) + slp).reshape(n, d)

    xi = man.randvec(Y)

    egrad = omg_func(Y)
    ehess = (aa @ xi.reshape(-1)).reshape(n, d)

    val1 = man.ehess2rhess(Y, egrad, ehess, xi)
    if False:
        val1a = man.ehess2rhess_alt(Y, egrad, ehess, xi)
        print(check_zero(val1-val1a))

    def rgrad_func(W):
        return man.proj_g_inv(W, omg_func(W))

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
    val2a, _, _ = calc_covar_numeric(man, Y, xi, omg_func)
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
    n, d = (5, 3)
    alpha = randint(1, 10, 2) * .1
    man = ComplexStiefel(n, d, alpha=alpha)

    Y = man.rand()
    UU = make_sym_pos(n)

    def f(Y):
        return rtrace(UU @ Y @ Y.T.conj())

    def df(Y):
        return 2*UU @ Y

    def ehess_form(Y, xi, eta):
        return 2 * rtrace(UU @ xi@eta.T.conj())

    def ehess_vec(Y, xi):
        return 2 * UU @ xi

    xxi = crandn(n, d)
    dlt = 1e-8
    Ynew = Y + dlt*xxi
    d1 = (f(Ynew) - f(Y))/dlt
    d2 = df(Y)
    print(d1 - rtrace(d2 @ xxi.T.conj()))

    eeta = crandn(n, d)

    d1 = rtrace((df(Ynew) - df(Y)) @ eeta.T.conj()) / dlt
    ehess_val = ehess_form(Y, xxi, eeta)
    dv2 = ehess_vec(Y, xxi)
    print(rtrace(dv2 @ eeta.T.conj()))
    print(d1, ehess_val, d1-ehess_val)

    # now check the formula: ehess = xi (eta_func(f)) - <D_xi eta, df(Y)>
    # promote eta to a vector field.

    m1 = crandn(n, n)
    m2 = crandn(d, d)

    def eta_field(Yin):
        return m1 @ (Yin - Y) @ m2 + eeta

    # xietaf: should go to ehess(xi, eta) + df(Y) @ etafield)
    xietaf = rtrace(df(Ynew) @ eta_field(Ynew).T.conj() -
                    df(Y) @ eta_field(Y).T.conj()) / dlt
    # appy eta_func to f: should go to tr(m1 @ xxi @ m2 @ df(Y).T.conj())
    Dxietaf = rtrace((eta_field(Ynew) - eta_field(Y)) @ df(Y).T.conj())/dlt
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
    ehvec_e = ehess_vec(Y, eta1)
    rhessvec_e = man.ehess2rhess(Y, egvec, ehvec_e, eta1)
    rhessval_e = man.inner(Y, rhessvec_e, xi1)
    rhessval_e1 = man.rhess02(Y, xi1, eta1, egvec, ehvec)
    rhessval_e2 = man.rhess02_alt(
        Y, xi1, eta1, egvec, rtrace(ehvec@eta1.T.conj()))
    print(rhessval_e, rhessval_e1, rhessval_e2)
    
    print('rhessval_e %f ' % rhessval_e)
    # the above computed inner_prod(Nabla_xi Pi * df, eta)
    # in the following check. Extend eta1 to eta_proj
    # (Pi Nabla_hat Pi g_inv df, g eta)
    # = D_xi (Pi g_inv df, g eta) - (Pi g_inv df g Pi Nabla_hat eta)
    
    def eta_proj(Y):
        return man.proj(Y, eta_field(Y))
    print(check_zero(eta1-eta_proj(Y)))
    
    e1 = man.inner(Y, man.proj_g_inv(Y, df(Y)), eta_proj(Y))
    e1a = rtrace(df(Y) @ eta_proj(Y).T.conj())
    print(e1, e1a, e1-e1a)
    Ynew = Y + xi1*dlt
    e2 = man.inner(Ynew, man.proj_g_inv(Ynew, df(Ynew)), eta_proj(Ynew))
    e2a = rtrace(df(Ynew) @ eta_proj(Ynew).T.conj())
    print(e2, e2a, e2-e2a)
    
    first = (e2 - e1)/dlt
    first1 = rtrace(df(Ynew) @ eta_proj(Ynew).T.conj() -
                    df(Y) @ eta_proj(Y).T.conj())/dlt
    print(first-first1)
    
    val3, _, _ = calc_covar_numeric(man, Y, xi1, eta_proj)
    second = man.inner(Y, man.proj_g_inv(Y, df(Y)), man.proj(Y, val3))
    second2 = man.inner(Y, man.proj_g_inv(Y, df(Y)), val3)
    print(second, second2, second-second2)
    print('same as rhess_val %f' % (first-second))


def optim_test():
    from pymanopt import Problem
    from pymanopt.solvers import TrustRegions
    from pymanopt.function import Callable
    
    n = 1000
    d = 50
    # problem Tr(AXBX^T)
    for i in range(1):
        D = randint(1, 10, n) * 0.02 + 1
        OO = random_orthogonal(n)
        A = OO @ np.diag(D) @ OO.T.conj()
        B = make_sym_pos(d)
        B = np.diag(randint(1, 10, d) * .2)
        
        alpha = randint(1, 10, 2) * .1
        alpha = alpha/alpha[0]
        alpha = np.array([1, .6])
        print(alpha)
        man = ComplexStiefel(n, d, alpha)

        @Callable
        def cost(X):
            return rtrace(A @ X @ B @ X.T.conj())

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
            d2 = rtrace(egrad(X) @ xi.T.conj())
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
            # print(opt)
            # double check:
            # print(cost(opt))
            min_val = 1e190
            # min_X = None
            for i in range(10000):
                Xi = man.rand()
                c = cost(Xi)
                if c < min_val:
                    # min_X = Xi
                    min_val = c
                if i % 1000 == 0:
                    print('i=%d min=%f' % (i, min_val))
            print(min_val)
        man1 = ComplexStiefel(n, d, alpha=np.array([1, 1]))
        prob = Problem(
            man1, cost, egrad=egrad, ehess=ehess)

        solver = TrustRegions(maxtime=100000, maxiter=100)
        opt = solver.solve(prob, x=XInit, Delta_bar=250)

        man1 = ComplexStiefel(n, d, alpha=np.array([1, .5]))
        prob = Problem(
            man1, cost, egrad=egrad, ehess=ehess)

        solver = TrustRegions(maxtime=100000, maxiter=100)
        opt = solver.solve(prob, x=XInit, Delta_bar=250)


def optim_test2():
    from pymanopt import Problem
    from pymanopt.solvers import TrustRegions
    from pymanopt.function import Callable

    n = 100
    d = 20
    # problem Tr(AXBX^T)
    for i in range(1):
        D = randint(1, 10, n) * 0.02 + 1
        OO = random_orthogonal(n)
        A = OO @ np.diag(D) @ OO.T.conj()
        B = make_sym_pos(d)
        
        alpha = randint(1, 10, 2) * .1
        alpha = alpha/alpha[0]
        print(alpha)
        man = ComplexStiefel(n, d, alpha)
        A2 = A @ A
        @Callable
        def cost(X):
            return rtrace(A @ X @ B @ X.T.conj() @ A2 @ X @ B @ X.T.conj() @ A)

        @Callable
        def egrad(X):
            R = 4*A2 @ X @ B @ X.T.conj() @ A2 @ X @ B
            return R

        @Callable
        def ehess(X, H):
            return 4*A2 @ H @ B @ X.T.conj() @ A2 @ X @ B +\
                4*A2 @ X @ B @ H.T.conj() @ A2 @ X @ B +\
                4*A2 @ X @ B @ X.T.conj() @ A2 @ H @ B

        if False:
            X = man.rand()
            xi = man.randvec(X)
            d1 = num_deriv(man, X, xi, cost)
            d2 = rtrace(egrad(X) @ xi.T.conj())
            print(check_zero(d1-d2))
            d3 = num_deriv(man, X, xi, egrad)
            d4 = ehess(X, xi)
            print(check_zero(d3-d4))
            
        prob = Problem(
            man, cost, egrad=egrad)
        XInit = man.rand()

        prob = Problem(
            man, cost, egrad=egrad, ehess=ehess)

        solver = TrustRegions(maxtime=100000, maxiter=100)
        opt = solver.solve(prob, x=XInit, Delta_bar=2500)
        print(cost(opt))
        if False:
            # print(opt)
            # double check:
            # print(cost(opt))
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
        man1 = ComplexStiefel(n, d, alpha=np.array([1, 1]))
        prob = Problem(
            man1, cost, egrad=egrad, ehess=ehess)

        solver = TrustRegions(maxtime=100000, maxiter=100)
        opt = solver.solve(prob, x=XInit, Delta_bar=250)

        man1 = ComplexStiefel(n, d, alpha=np.array([1, .5]))
        prob = Problem(
            man1, cost, egrad=egrad, ehess=ehess)

        solver = TrustRegions(maxtime=100000, maxiter=100)
        opt = solver.solve(prob, x=XInit, Delta_bar=250)
        

def optim_test3():
    from pymanopt import Problem
    from pymanopt.solvers import TrustRegions
    from pymanopt.function import Callable
    n = 200
    d = 20
    # problem Tr(AXBX^T)
    for i in range(1):
        B = np.diag(
            np.concatenate([randint(1, 10, d), np.zeros(n-d)]))
        D = randint(1, 10, n) * 0.02 + 1
        OO = random_orthogonal(n)
        A = OO @ np.diag(D) @ OO.T.conj()

        alpha = randint(1, 10, 2)
        alpha = alpha/alpha[0]
        print(alpha)
        man = ComplexStiefel(n, d, alpha)
        cf = 10
        B2 = B @ B

        @Callable
        def cost(X):
            return cf * rtrace(
                B @ X @ X.T.conj() @ B2 @ X @ X.T.conj() @ B) +\
                rtrace(X.T.conj() @ A @ X)
        
        @Callable
        def egrad(X):
            R = cf*4*B2 @ X @ X.T.conj() @ B2 @ X + 2*A @ X
            return R

        @Callable
        def ehess(X, H):
            return 4*cf*B2 @ H @ X.T.conj() @ B2 @ X +\
                4*cf*B2 @ X @ H.T.conj() @ B2 @ X +\
                4*cf*B2 @ X @ X.T.conj() @ B2 @ H + 2*A @ H
        
        if False:
            X = man.rand()
            xi = man.randvec(X)
            d1 = num_deriv(man, X, xi, cost)
            d2 = rtrace(egrad(X) @ xi.T.conj())
            print(check_zero(d1-d2))
            d3 = num_deriv(man, X, xi, egrad)
            d4 = ehess(X, xi)
            print(check_zero(d3-d4))
            
        XInit = man.rand()
        prob = Problem(
            man, cost, egrad=egrad, ehess=ehess)

        solver = TrustRegions(maxtime=100000, maxiter=100)
        opt = solver.solve(prob, x=XInit, Delta_bar=2500)
        print(cost(opt))
        if False:
            # print(opt)
            # double check:
            # print(cost(opt))
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
        man1 = ComplexStiefel(n, d, alpha=np.array([1, 1]))
        prob = Problem(
            man1, cost, egrad=egrad, ehess=ehess)

        solver = TrustRegions(maxtime=100000, maxiter=100)
        opt = solver.solve(prob, x=XInit, Delta_bar=250)

        man1 = ComplexStiefel(n, d, alpha=np.array([1, .5]))
        # man1 = ComplexStiefel(n, d, alpha=np.array([1, 1]))
        prob = Problem(
            man1, cost, egrad=egrad, ehess=ehess)

        solver = TrustRegions(maxtime=100000, maxiter=100)
        opt = solver.solve(prob, x=XInit, Delta_bar=250)


def test_geodesics():
    from scipy.linalg import expm
    alpha = np.random.randint(1, 10, (2)) * .1
    # alpha = np.array([1, .5])
    m, d = (5, 3)
    man = ComplexStiefel(m, d, alpha=alpha)
    Y = man.rand()

    alf = alpha[1]/alpha[0]
    
    def calc_gamma(man, Y, eta):
        etaxiy = 2*eta @ (eta.T.conj()@Y)
        egcoef = Y @ (eta.T.conj()@eta)
        ft = 1 - alf
        egcoef += ft*(etaxiy - Y@(Y.T.conj()@etaxiy))
        return egcoef
    eta = man.randvec(Y)
    g1 = calc_gamma(man, Y, eta)
    g2 = man.christoffel_gamma(Y, eta, eta)
    print(g1-g2)
    egrad = crandn(m, d)
    print(rtrace(g1 @ egrad.T.conj()))
    print(man.rhess02_alt(Y, eta, eta, egrad, 0))

    # try to see if the solution is good:
    A = Y.T.conj() @ eta
    S0 = eta.T.conj() @ eta

    e_mat = np.bmat([[(2*alf-1)*A, -S0 - 2*(1-alf)*A@A],
                     [np.eye(d), A]])
    init_c = np.bmat([Y, eta])
    
    def ff(t):
        v1 = init_c @ expm(t*e_mat)
        v2 = expm(t*(1-2*alf)*A)
        return v1[:, :d] @ v2, v1[:, d:] @ v2, None

    t = 100
    dlt = 1e-8

    fval, fd, fdd = ff(t)
    fval1, fd1, fdd1 = ff(t+dlt)
    fval2, fd2, fdd2 = ff(t-dlt)
    ffdot = (fval1 - fval)/dlt
    print(check_zero(ffdot - fd))
    ffddot_0 = (fval1 - 2*fval + fval2)/(dlt*dlt)
    ffddot = (fd1 - fd)/dlt
    print(ffddot_0)
    print(ffddot)
    print(ffddot + calc_gamma(man, fval, ffdot))

    # second solution:
    K = eta - Y @ (Y.T.conj() @ eta)
    Yp, R = np.linalg.qr(K)

    x_mat = np.bmat([[2*alf*A, -R.T.conj()], [R, zeros((d, d))]])
    Yt = np.bmat([Y, Yp]) @ expm(t*x_mat)[:, :d] @ \
        expm(t*(1-2*alf)*A)
    x_d_mat = x_mat[:, :d].copy()
    x_d_mat[:d, :] += (1-2*alf) * A
    Ydt = np.bmat([Y, Yp]) @ expm(t*x_mat) @ x_d_mat @\
        expm(t*(1-2*alf)*A)
    x_dd_mat = x_mat @ x_d_mat + x_d_mat @ ((1-2*alf)*A)
    Yddt = np.bmat([Y, Yp]) @ expm(t*x_mat) @ x_dd_mat @\
        expm(t*(1-2*alf)*A)
    print(Yddt + calc_gamma(man, Yt, Ydt))
    print(man.exp(Y, t*eta) - Yt)
    
        
if __name__ == '__main__':
    optim_test()
    test_geodesics()
    test_all_projections()
