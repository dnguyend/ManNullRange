import numpy as np
from numpy.random import (randint, randn)
from numpy import zeros, trace, allclose
import numpy.linalg as la

from ManNullRange.manifolds.RealPositiveSemidefinite import (
    RealPositiveSemidefinite, psd_ambient, psd_point)
from ManNullRange.manifolds.tools import (sym, extended_lyapunov)
from test_tools import check_zero, make_sym_pos, random_orthogonal


def test_inner(man, S):
    for i in range(10):
        alpha = man.alpha
        bt = man.beta
        eta1 = man._rand_ambient()
        eta2 = man._rand_ambient()
        Y = S.Y
        Pinv = S.Pinv
        inn1 = alpha[0]*trace(eta1.tY.T @ eta2.tY) +\
            (alpha[1]-alpha[0])*trace((eta1.tY.T@Y) @ (Y.T@eta2.tY)) +\
            bt*trace(Pinv@eta1.tP@Pinv@eta2.tP.T)
        assert(allclose(man.inner(S, eta1, eta2), inn1))
        
        eta2a = man.g_inv(S, eta2)
        v1 = man.inner(S, eta1, eta2a)
        v2 = trace(eta1.tY @ eta2.tY.T) + trace(eta1.tP @ eta2.tP.T)
        assert(allclose(v1, v2))
    print(True)


def test_J(man, S):
    al = man.alpha
    bt = man.beta
    
    def diff_i(ii):
        U = zeros(man.tdim_St+man.tdim_P)
        U[ii] = 1
        E = man._unvec(U)
        jE = man.J(S, E)
        jE1 = {'P': E.tP - E.tP.T,
               'YP': al[1]*S.Y.T@E.tY + bt*(E.tP@S.Pinv-S.Pinv@E.tP)}
        return np.mean(np.abs(man._vec_range_J(jE)-man._vec_range_J(jE1)))

    diffs = zeros(man.tdim_St + man.tdim_P)
    for ii in range(diffs.shape[0]):
        diffs[ii] = diff_i(ii)
    try:
        assert(allclose(diffs, 0))
        print(True)
    except Exception:
        print(False)


def test_Jst(man, S, jmat):
    print("test JST")
    for ii in range(10):
        a = man._rand_range_J()
        avec = man._vec_range_J(a)
        jtout = jmat.T @ avec
        jtout2 = man._vec(man.Jst(S, a))
        # print(np.where(np.abs(jtout - jtout2) > 1e-9))
        diff = check_zero(jtout-jtout2)
        print(diff)

        
def make_j_mat(man, S):
    codim = man.codim
    ret = zeros((codim, man.tdim_P + man.tdim_St))
    for ii in range(ret.shape[1]):
        eSt = zeros(man.tdim_St)
        eP = zeros((man.tdim_P))
        if ii < man.tdim_St:
            eSt[ii] = 1
        else:
            eP[ii-man.tdim_St] = 1
        ret[:, ii] = man._vec_range_J(
            man.J(S, psd_ambient(
                eSt.reshape(man.n, man.p), eP.reshape(man.p, man.p))))
    return ret
        

def make_g_inv_mat(man, S):
    nn = man.tdim_P + man.tdim_St
    ret = zeros((nn, nn))
    for ii in range(ret.shape[1]):
        ee = zeros(nn)
        ee[ii] = 1
        ret[:, ii] = man._vec(
            man.g_inv(S, man._unvec(
                ee)))
    return ret


def test_projection(man, S):
    print("test projection")
    N = 10
    ret = np.zeros(N)

    for i in range(N):
        U = man._rand_ambient()
        Upr = man.proj(S, U)
        ret[i] = check_zero(man._vec_range_J(man.J(S, Upr)))
    print(ret)
    
    if check_zero(ret) > 1e-8:
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
        ret[i] = (check_zero(XX.Y.T @ H.tY + H.tY.T @ XX.Y))
    print(ret)
    if check_zero(ret) > 1e-9:
        print("not all works")
    else:
        print("All good")
    print("check inner of projection = inner of original")
    print(ret_pr)
    if check_zero(ret_pr) > 1e-8:
        print("not all works")
    else:
        print("All good")


def test_lyapunov():
    alpha = randint(1, 10, 2) * .1
    beta = randint(1, 10, 1)[0] * .02
    n = 5
    d = 3
    man = RealPositiveSemidefinite(n, d, alpha=alpha, beta=beta)
    S = man.rand()

    P = S.P
    B = randn(d, d)
    alpha1 = alpha[1]
    
    def L(X, P):
        Piv = la.inv(P)
        return (alpha1 - 2*beta)*X + beta*(P@X@Piv + Piv@X@P)
    X = extended_lyapunov(alpha1, beta, P, B)
    # L(X, P)
    print(check_zero(B-L(X, P)))


def test_N_proj():
    alpha = randint(1, 10, 2) * .1
    beta = randint(1, 10, 1)[0] * .02
    n = 10
    d = 6
    man = RealPositiveSemidefinite(n, d, alpha=alpha, beta=beta)
    S = man.rand()
    """
    def proj_range_alt(man, S, U):
        # projection. U is in ambient
        # return one in tangent
        al1 = man.alpha[1]
        beta = man.beta
        YTU = S.Y.T@U.tY
        D0 = sym(U.tP + YTU@S.P - S.P@YTU)
        D = _extended_lyapunov(al1, beta, S.P, D0, S.evl, S.evec)
        return psd_ambient(
            beta*S.Y@(S.Pinv@D-D@S.Pinv) + U.tY - S.Y@(S.Y.T@U.tY), al1*D)
    """
    U = man.randvec(S)
    Upr1 = super(RealPositiveSemidefinite, man).proj(S, U)
    Upr2 = man.proj(S, U)
    Upr3 = man.proj_range_alt(S, U)
    print(check_zero(Upr1.tP-Upr2.tP))
    print(check_zero(Upr1.tY-Upr2.tY))
    print(check_zero(Upr1.tY-Upr3.tY))
    print(check_zero(Upr1.tP-Upr3.tP))
    
        
def test_all_projections():
    alpha = randint(1, 10, 2) * .1
    beta = randint(1, 10, 1)[0] * .02
    n = 5
    d = 3
    man = RealPositiveSemidefinite(n, d, alpha=alpha, beta=beta)
    print(man)
    S = man.rand()

    test_inner(man, S)
    test_J(man, S)
            
    # now check metric, Jst etc
    # check Jst: vectorize the operator J then compare Jst with jmat.T
    jmat = make_j_mat(man, S)
    test_Jst(man, S, jmat)
    ginv_mat = make_g_inv_mat(man, S)
    # test g_inv_Jst
    for ii in range(10):
        a = man._rand_range_J()
        avec = man._vec_range_J(a)
        jtout = ginv_mat @ jmat.T @ avec
        jtout2 = man._vec(man.g_inv_Jst(S, a))
        diff = check_zero(jtout-jtout2)
        print(diff)
    # test projection
    test_projection(man, S)
    # now diff projection

    for i in range(1):
        e = man._rand_ambient()
        S1 = man.rand()
        xi = man.randvec(S1)
        dlt = 1e-7
        S2 = psd_point(S1.Y + dlt*xi.tY, S1.P+dlt*xi.tP)

        # S = psd_point(S1.Y, S1.P)
        """
        U = e
        D = man._calc_D(S1, U)
        print((man.alpha[1] - 2*man.beta)*D + 2*man.beta*(sym(S.P@D@S.Pinv)))
        print(sym(U.tP + S.Y.T@U.tY@S.P - S.P@S.Y.T@U.tY))
        Dnew = man._calc_D(S2, U)
        print((Dnew - D)/dlt)
        ddin = xi.tY.T @ U.tY @ S.P - S.P @ xi.tY.T @ U.tY + \
            S.Y.T @ U.tY @ xi.tP - xi.tP @ S.Y.T @ U.tY - man.beta * (
                xi.tP @ D @ S.Pinv + S.Pinv @ D @ xi.tP -
                S.P @ D @ S.Pinv @ xi.tP @ S.Pinv -
                S.Pinv @ xi.tP @ S.Pinv @ D @ S.P)
        DD = _extended_lyapunov(
            man.alpha[1], man.beta, S.P,  sym(ddin), S.evl, S.evec)
        """
        d1P = (man.proj(S2, e).tP - man.proj(S1, e).tP)/dlt
        d1Y = (man.proj(S2, e).tY - man.proj(S1, e).tY)/dlt
        d2 = man.D_proj(S1, xi, e)
        print(check_zero(d1P-d2.tP) + check_zero(d1Y-d2.tY))
    
    for i in range(10):
        a = man._rand_range_J()
        eta = man._rand_ambient()
        print(man.base_inner_ambient(eta, man.Jst(S, a)))
        print(trace((eta.tP.T - eta.tP) @ a['P'] + (
            man.alpha[1]*eta.tY.T @ S.Y + man.beta*(
                S.Pinv @ eta.tP.T - eta.tP.T @ S.Pinv)) @ a['YP']))

        print(trace(2 * eta.tP.T @ a['P']) + trace((
            man.alpha[1]*eta.tY.T @ S.Y + man.beta*(
                S.Pinv @ eta.tP.T - eta.tP.T @ S.Pinv)) @ a['YP']))

        print(trace(eta.tP.T @ (
            2*a['P'] + man.beta*(a['YP'] @ S.Pinv - S.Pinv @ a['YP'])))
              + trace(eta.tY.T @ (man.alpha[1] * S.Y @ a['YP'])))

        print(man.base_inner_E_J(man.J(S, eta), a))
        print(trace((eta.tP - eta.tP.T).T @ a['P'] + (
            man.alpha[1]*S.Y.T @ eta.tY + man.beta*(
                eta.tP @ S.Pinv - S.Pinv @ eta.tP)).T @ a['YP']))

    for i in range(10):
        a = man._rand_range_J()
        beta = man.beta
        alf = man.alpha
        anew1 = man.J(S, man.g_inv_Jst(S, a))
        
        anew = {}
        saYP = a['YP'] + a['YP'].T
        anew['P'] = 4/beta * S.P @ a['P'] @ S.P + S.P @ saYP - saYP @ S.P
        anew['YP'] = alf[1]*a['YP'] + beta*(
            ((2/man.beta)*S.P@a['P']@S.P + S.P @ a['YP'] - a['YP'] @ S.P) @
            S.Pinv - S.Pinv @ ((2/man.beta)*S.P@a['P']@S.P + S.P @ a['YP'] -
                               a['YP'] @ S.P))
        
        anew['YP'] = alf[1]*a['YP'] + (
            (2*S.P@a['P'] + beta*S.P @ a['YP'] @ S.Pinv - beta*a['YP'])
            - (2*a['P']@S.P + beta*a['YP'] - beta*S.Pinv@a['YP'] @ S.P))

        anew['YP'] = (alf[1]-2*beta)*a['YP'] + (
            (2*S.P@a['P'] + beta*S.P @ a['YP'] @ S.Pinv)
            - (2*a['P']@S.P - beta*S.Pinv@a['YP'] @ S.P))

        anew['YP'] = (alf[1]-2*beta)*a['YP'] + (
            (2*S.P@a['P'] - 2*a['P']@S.P + beta*S.P @ a['YP'] @ S.Pinv +
             beta*S.Pinv@a['YP'] @ S.P))
        print(check_zero(man._vec_range_J(anew1)-man._vec_range_J(anew)))

    for i in range(10):
        a = man._rand_range_J()
        b1 = man.J(S, man.g_inv_Jst(S, a))
        b2 = man.J_g_inv_Jst(S, a)
        print(
            check_zero(man._vec_range_J(b1)-man._vec_range_J(b2)))
        a1 = man.solve_J_g_inv_Jst(S, b1)
        print(check_zero(
            man._vec_range_J(a)-man._vec_range_J(a1)))
                
    for ii in range(10):
        E = man._rand_ambient()
        a2 = man.J_g_inv(S, E)
        a1 = man.J(S, man.g_inv(S, E))
        print(check_zero(man._vec_range_J(a1)-man._vec_range_J(a2)))
    
    for i in range(20):
        Uran = man._rand_ambient()
        Upr = man.proj(S, man.g_inv(S, Uran))
        Upr2 = man.proj_g_inv(S, Uran)
        print(check_zero(man._vec(Upr)-man._vec(Upr2)))

    for ii in range(10):
        a = man._rand_range_J()
        xi = man.randvec(S)
        jtout2 = man.Jst(S, a)
        dlt = 1e-7
        Snew = psd_point(S.Y + dlt*xi.tY, S.P+dlt*xi.tP)
        jtout2a = man.Jst(Snew, a)
        d1 = (jtout2a - jtout2).scalar_mul(1/dlt)
        d2 = man.D_Jst(S, xi, a)
        print(check_zero(man._vec(d2)-man._vec(d1)))

    for ii in range(10):
        S1 = man.rand()
        eta = man._rand_ambient()
        xi = man.randvec(S1)
        a1 = man.J(S1, eta)
        dlt = 1e-8
        Snew = psd_point(S1.Y + dlt*xi.tY, S1.P+dlt*xi.tP)
        a2 = man.J(Snew, eta)
        d1 = {'P': (a2['P']-a1['P'])/dlt,
              'YP': (a2['YP']-a1['YP'])/dlt}
        d2 = man.D_J(S1, xi, eta)
        print(check_zero(man._vec_range_J(d2) - man._vec_range_J(d1)))
                
    # derives metrics
    for ii in range(10):
        S1 = man.rand()
        xi = man.randvec(S1)
        omg1 = man._rand_ambient()
        omg2 = man._rand_ambient()
        dlt = 1e-7
        S2 = psd_point(S1.Y + dlt*xi.tY, S1.P+dlt*xi.tP)
        p1 = man.inner(S1, omg1, omg2)
        p2 = man.inner(S2, omg1, omg2)
        der1 = (p2-p1)/dlt
        der2 = man.base_inner_ambient(
            man.D_g(S1, xi, omg2), omg1)
        print(der1-der2)

    # cross term for christofel
    for i in range(10):
        S1 = man.rand()
        xi = man.randvec(S1)
        eta1 = man.randvec(S1)
        eta2 = man.randvec(S1)
        dr1 = man.D_g(S1, xi, eta1)
        x12 = man.contract_D_g(S1, eta1, eta2)

        p1 = man.base_inner_ambient(dr1, eta2)
        p2 = man.base_inner_ambient(x12, xi)
        print(p1, p2, p1-p2)

    # now test christofel:
    # two things: symmetric on vector fields
    # and christofel relation
    # in the case metric
    for i in range(10):
        S1 = man.rand()
        xi = man.randvec(S1)
        eta1 = man.randvec(S1)
        eta2 = man.randvec(S1)
        p1 = man.proj_g_inv(S1, man.christoffel_form(S1, xi, eta1))
        p2 = man.proj_g_inv(S1, man.christoffel_form(S1, eta1, xi))
        print(check_zero(man._vec(p1)-man._vec(p2)))
        v1 = man.base_inner_ambient(
            man.christoffel_form(S1, eta1, eta2), xi)
        v2 = man.base_inner_ambient(man.D_g(S1, eta1, eta2), xi)
        v3 = man.base_inner_ambient(man.D_g(S1, eta2, eta1), xi)
        v4 = man.base_inner_ambient(man.D_g(S1, xi, eta1), eta2)
        print(v1, 0.5*(v2+v3-v4), v1-0.5*(v2+v3-v4))

        
def test_christ_flat():
    """now test that christofel preserve metrics:
    on the flat space
    d_xi <v M v> = 2 <v M nabla_xi v>
     v = proj(W) @ (aa W + b)
    """
    alpha = randint(1, 10, 2) * .1
    beta = randint(1, 10, 2)[0] * .1
    n = 5
    d = 3
    man = RealPositiveSemidefinite(n, d, alpha=alpha, beta=beta)
    S = man.rand()
    
    xi = man.randvec(S)
    xi = man.randvec(S)
    aa = randn(n*d, n*d)
    bb = randn(n*d)
    cc = randn(d*d, d*d)
    dd = sym(randn(d, d))
        
    def v_func_flat(S):
        # a function from the manifold
        # to ambient
        csp = sym((cc @ S.P.reshape(-1)).reshape(d, d))
        
        return psd_ambient(
            (aa @ S.Y.reshape(-1) + bb).reshape(n, d),
            csp + dd)

    vv = v_func_flat(S)
    dlt = 1e-7
    Snew = psd_point(
        S.Y + dlt * xi.tY,
        S.P + dlt * xi.tP)
    vnew = v_func_flat(Snew)

    val = man.inner_product_amb(S, vv)
    valnew = man.inner_product_amb(Snew, vnew)
    d1 = (valnew - val)/dlt
    dv = (vnew - vv).scalar_mul(1/dlt)
    nabla_xi_v = dv + man.g_inv(
        S, man.christoffel_form(S, xi, vv))
    nabla_xi_va = dv + man.g_inv(
        S, super(RealPositiveSemidefinite, man).christoffel_form(S, xi, vv))
    print(check_zero(man._vec(nabla_xi_v) - man._vec(nabla_xi_va)))
    d2 = man.inner(S, vv, nabla_xi_v)

    print(d1)
    print(2*d2)


def calc_covar_numeric(man, S, xi, v_func):
    """ compute nabla on E dont do the metric
    lower index. So basically
    Nabla (Pi e).
    Thus, if we want to do Nabla Pi g_inv df
    We need to send g_inv df
    """

    def vv_func(W):
        return man.proj(W, v_func(W))
    
    vv = vv_func(S)

    dlt = 1e-7
    Snew = psd_point(S.Y + dlt*xi.tY, S.P + dlt * xi.tP)
    vnew = vv_func(Snew)

    val = man.inner_product_amb(S, vv)
    valnew = man.inner_product_amb(
        Snew, vnew)
    d1 = (valnew - val)/dlt
    dv = (vnew - vv).scalar_mul(1/dlt)
    cx = man.christoffel_form(S, xi, vv)
    nabla_xi_v_up = dv + man.g_inv(S, cx)
    nabla_xi_v = man.proj(S, nabla_xi_v_up)
    
    if False:
        d2 = man.inner_product_amb(S, vv, nabla_xi_v)
        d2up = man.inner_product_amb(
            S, vv, nabla_xi_v_up)

        print(d1)
        print(2*d2up)
        print(2*d2)
    return nabla_xi_v, dv, cx


def test_chris_vectorfields():
    # now test that it works on embedded metrics
    # we test that D_xi (eta g eta) = 2(eta g nabla_xi eta)
    n, d = (5, 3)
    alpha = randint(1, 10, 2) * .1
    beta = randint(1, 10, 1)[0] * .1
    man = RealPositiveSemidefinite(n, d, alpha=alpha, beta=beta)

    S0 = man.rand()
    aa = randn(n*d, n*d)
    intc = randn(n*d)
    cc = randn(d*d, d*d)
    p_intc = sym(randn(d, d))

    inct_xi = man._rand_ambient()
    aa_xi = randn(n*d, n*d)
    cc_xi = randn(d*d, d*d)
    
    def v_func(S):
        # a function from the manifold
        # to ambient
        csp = sym((cc @ (S.P-S0.P).reshape(-1)).reshape(d, d))
        
        return man.proj(S, psd_ambient(
            (aa @ (S.Y-S0.Y).reshape(-1) + intc).reshape(n, d),
            csp + p_intc))

    SS = psd_point(S0.Y, S0.P)
    xi = man.proj(SS, inct_xi)

    nabla_xi_v, dv, cxv = calc_covar_numeric(
        man, SS, xi, v_func)

    def xi_func(S):
        csp_xi = sym((cc_xi @ (S.P-S0.P).reshape(-1)).reshape(d, d))
        xi_amb = psd_ambient(
            (aa_xi @ (S.Y-S0.Y).reshape(-1) +
             inct_xi.tY.reshape(-1)).reshape(n, d),
            csp_xi + inct_xi.tP)
        return man.proj(S, xi_amb)

    vv = v_func(SS)

    nabla_v_xi, dxi, cxxi = calc_covar_numeric(
        man, SS, vv, xi_func)
    diff = nabla_xi_v - nabla_v_xi
    print(diff.tY, diff.tP)
    # now do Lie bracket:
    dlt = 1e-7
    SnewXi = psd_point(SS.Y+dlt*xi.tY, SS.P+dlt*xi.tP)
    Snewvv = psd_point(SS.Y+dlt*vv.tY, SS.P+dlt*vv.tP)
    vnewxi = v_func(SnewXi)
    xnewv = xi_func(Snewvv)
    dxiv = (vnewxi - vv).scalar_mul(1/dlt)
    dvxi = (xnewv - xi).scalar_mul(1/dlt)
    diff2 = man.proj(SS, dxiv-dvxi)
    print(check_zero(man._vec(diff) - man._vec(diff2)))
                            

def num_deriv_amb(man, S, xi, func, dlt=1e-7):
    Snew = psd_point(S.Y + dlt*xi.tY,
                     S.P + dlt*xi.tP)
    return (func(Snew) - func(S)).scalar_mul(1/dlt)

    
def test_covariance_deriv():
    # now test full:
    # do covariant derivatives
    # check that it works, preseving everything
    n, d = (5, 3)
    alpha = randint(1, 10, 2) * .1
    beta = randint(1, 10, 2)[0] * .01
    man = RealPositiveSemidefinite(n, d, alpha=alpha, beta=beta)
    S = man.rand()
    
    aa = randn(n*d, n*d)
    cc = randn(d*d, d*d)
    icpt = man._rand_ambient()

    def omg_func(S):
        csp = sym((cc @ S.P.reshape(-1)).reshape(d, d))
        return psd_ambient(
            (aa @ S.Y.reshape(-1) + icpt.tY.reshape(-1)).reshape(n, d),
            csp + icpt.tP)

    xi = man.randvec(S)
    egrad = omg_func(S)
    ecsp = sym((cc @ xi.tP.reshape(-1)).reshape(d, d))
    ehess = psd_ambient(
        (aa @ xi.tY.reshape(-1)).reshape(n, d),
        ecsp)

    val1 = man.ehess2rhess(S, egrad, ehess, xi)

    def rgrad_func(W):
        return man.proj_g_inv(W, omg_func(W))

    if False:
        first = ehess
        a = man.J_g_inv(S, egrad)
        rgrad = man.proj_g_inv(S, egrad)
        second = man.D_g(
            S, xi, man.g_inv(S, egrad)).scalar_mul(-1)
        aout = man.solve_J_g_inv_Jst(S, a)
        third = man.proj(S, man.D_g_inv_Jst(S, xi, aout)).scalar_mul(-1)
        fourth = man.christoffel_form(S, xi, rgrad)
        val1a1 = man.proj_g_inv(S, first + second + fourth) + third
        print(check_zero(man._vec(val1-val1a1)))
    elif True:
        d_xi_rgrad = num_deriv_amb(man, S, xi, rgrad_func)
        rgrad = man.proj_g_inv(S, egrad)
        fourth = man.christoffel_form(S, xi, rgrad)
        val1a = man.proj(S, d_xi_rgrad) + man.proj_g_inv(S, fourth)
        print(check_zero(man._vec(val1-val1a)))

    # nabla_v_xi, dxi, cxxi
    val2a, _, _ = calc_covar_numeric(man, S, xi, omg_func)
    val2, _, _ = calc_covar_numeric(man, S, xi, rgrad_func)
    # val2_p = project(prj, val2)
    val2_p = man.proj(S, val2)
    # print(val1)
    # print(val2_p)
    print(check_zero(man._vec(val1)-man._vec(val2_p)))
    if True:
        H = xi
        valrangeA_ = ehess + man.g(S, man.D_proj(
            S, H, man.g_inv(S, egrad))) - man.D_g(
                S, H, man.g_inv(S, egrad)) +\
            man.christoffel_form(S, H, man.proj_g_inv(S, egrad))
        valrangeB = man.proj_g_inv(S, valrangeA_)
    valrange = man.ehess2rhess_alt(S, egrad, ehess, xi)
    print(check_zero(man._vec(valrange)-man._vec(val2_p)))
    print(check_zero(man._vec(valrange)-man._vec(val1)))
    print(check_zero(man._vec(valrange)-man._vec(valrangeB)))

    
def test_rhess_02():
    n, d = (5, 3)
    alpha = randint(1, 10, 2) * .1
    beta = randint(1, 10, 2)[0] * .1
    man = RealPositiveSemidefinite(n, d, alpha=alpha, beta=beta)

    S = man.rand()
    # simple function. Distance to a given matrix
    # || S - A||_F^2
    A = sym(randn(n, n))

    def f(S):
        diff = (A - S.Y @ S.P @ S.Y.T)
        return trace(diff @ diff.T)

    def df(S):
        return psd_ambient(-4*A @ S.Y @ S.P,
                           2*(S.P-S.Y.T @ A @ S.Y))

    def ehess_form(S, xi, eta):
        return trace(-4*A @ (xi.tY @ S.P + S.Y @ xi.tP) @ eta.tY.T) +\
            2*trace((xi.tP - xi.tY.T@A@S.Y - S.Y.T@A@xi.tY) @ eta.tP.T)

    def ehess_vec(S, xi):
        return psd_ambient(-4*A @ (xi.tY @ S.P + S.Y @ xi.tP),
                           2*(xi.tP - xi.tY.T @ A @ S.Y - S.Y.T @ A @ xi.tY))

    xxi = man.randvec(S)
    dlt = 1e-8
    Snew = psd_point(
        S.Y+dlt*xxi.tY, S.P + dlt*xxi.tP)
    d1 = (f(Snew) - f(S))/dlt
    d2 = df(S)
    print(d1 - man.base_inner_ambient(d2,  xxi))

    eeta = man.randvec(S)

    d1 = man.base_inner_ambient((df(Snew) - df(S)), eeta) / dlt
    ehess_val = ehess_form(S, xxi, eeta)
    dv2 = ehess_vec(S, xxi)
    print(man.base_inner_ambient(dv2, eeta))
    print(d1, ehess_val, d1-ehess_val)

    # now check the formula: ehess = xi (eta_func(f)) - <D_xi eta, df(Y)>
    # promote eta to a vector field.

    m1 = randn(n, n)
    m2 = randn(d, d)
    m_p = randn(d*d, d*d)

    def eta_field(Sin):
        return man.proj(S, psd_ambient(
            m1 @ (Sin.Y - S.Y) @ m2,
            sym((m_p @ (Sin.P - S.P).reshape(-1)).reshape(d, d)))) + eeta

    # xietaf: should go to ehess(xi, eta) + df(Y) @ etafield)
    xietaf = (man.base_inner_ambient(df(Snew), eta_field(Snew)) -
              man.base_inner_ambient(df(S), eta_field(S))) / dlt
    # appy eta_func to f: should go to tr(m1 @ xxi @ m2 @ df(Y).T)
    Dxietaf = man.base_inner_ambient(
        (eta_field(Snew) - eta_field(S)), df(S))/dlt
    # this is ehess. should be same as d1 or ehess_val
    print(xietaf-Dxietaf)
    print(xietaf-Dxietaf-ehess_val)

    # now check: rhess. Need to make sure xi, eta in the tangent space.
    # first compare this with numerical differentiation
    xi1 = man.proj(S, xxi)
    eta1 = man.proj(S, eeta)
    egvec = df(S)
    ehvec = ehess_vec(S, xi1)
    rhessvec = man.ehess2rhess(S, egvec, ehvec, xi1)

    # check it numerically:
    def rgrad_func(Y):
        return man.proj_g_inv(Y, df(Y))
    
    # val2a, _, _ = calc_covar_numeric_raw(man, W, xi1, df)
    val2, _, _ = calc_covar_numeric(man, S, xi1, rgrad_func)
    val2_p = man.proj(S, val2)
    # print(rhessvec)
    # print(val2_p)
    print(man._vec(rhessvec-val2_p))
    rhessval = man.inner_product_amb(S, rhessvec, eta1)
    print(man.inner_product_amb(S, val2, eta1))
    print(rhessval)

    # check symmetric:
    ehvec_e = ehess_vec(S, eta1)
    rhessvec_e = man.ehess2rhess(S, egvec, ehvec_e, eta1)
    rhessval_e = man.inner_product_amb(S, rhessvec_e, xi1)
    print(rhessval_e)
    # the above computed inner_prod(Nabla_xi Pi * df, eta)
    # in the following check. Extend eta1 to eta_proj
    # (Pi Nabla_hat Pi g_inv df, g eta)
    # = D_xi (Pi g_inv df, g eta) - (Pi g_inv df g Pi Nabla_hat eta)
    
    def eta_proj(S):
        return man.proj(S, eta_field(S))
    print(check_zero(man._vec(eta1-eta_proj(S))))
    
    e1 = man.inner(S, man.proj_g_inv(S, df(S)), eta_proj(S))
    e1a = man.base_inner_ambient(df(S), eta_proj(S))
    print(e1, e1a, e1-e1a)
    Snew = psd_point(S.Y + dlt*xi1.tY, S.P + dlt*xi1.tP)
    e2 = man.inner(Snew, man.proj_g_inv(Snew, df(Snew)), eta_proj(Snew))
    e2a = man.base_inner_ambient(df(Snew), eta_proj(Snew))
    print(e2, e2a, e2-e2a)
    
    first = (e2 - e1)/dlt
    first1 = (man.base_inner_ambient(df(Snew), eta_proj(Snew)) -
              man.base_inner_ambient(df(S), eta_proj(S)))/dlt
    print(first-first1)
    
    val3, _, _ = calc_covar_numeric(man, S, xi1, eta_proj)
    second = man.inner(S, man.proj_g_inv(S, df(S)), man.proj(S, val3))
    second2 = man.inner(S, man.proj_g_inv(S, df(S)), val3)
    print(second, second2, second-second2)
    print('same as rhess_val %f' % (first-second))
    

def solve_dist_with_man(man, A, X0, maxiter):
    from pymanopt import Problem
    from pymanopt.solvers import TrustRegions
    from pymanopt.function import Callable

    @Callable
    def cost(S):
        if not(S.P.dtype == np.float):
            raise(ValueError("Non real"))
        diff = (A - S.Y @ S.P @ S.Y.T)
        val = trace(diff @ diff.T)
        # print('val=%f' % val)
        return val

    @Callable
    def egrad(S):
        return psd_ambient(-4*A @ S.Y @ S.P,
                           2*(S.P-S.Y.T @ A @ S.Y))
    
    @Callable
    def ehess(S, xi):
        return psd_ambient(
            -4*A @ (xi.tY @ S.P + S.Y @ xi.tP),
            2*(xi.tP - xi.tY.T @ A @ S.Y - S.Y.T @ A @ xi.tY))

    prob = Problem(
        man, cost, egrad=egrad, ehess=ehess)

    solver = TrustRegions(maxtime=100000, maxiter=maxiter, use_rand=False)
    opt = solver.solve(prob, x=X0, Delta_bar=250)
    return opt

    
def optim_test():
    n, d = (1000, 50)
    # simple function. Distance to a given matrix
    # || S - A||_F^2
    Y0, _ = np.linalg.qr(randn(n, d))
    P0 = np.diag(randint(1, 1000, d)*.001)
    A0 = sym(Y0 @ P0 @ Y0.T)
    A = sym(randn(n, n))*1e-2 + A0

    alpha = np.array([1, 1])
    print("alpha %s" % str(alpha))

    beta = alpha[1] * .1
    man = RealPositiveSemidefinite(n, d, alpha=alpha, beta=beta)
    XInit = man.rand()
    opt_pre = solve_dist_with_man(man, A, X0=XInit, maxiter=20)

    beta = alpha[1] * 1
    man = RealPositiveSemidefinite(n, d, alpha=alpha, beta=beta)
    opt_mid = solve_dist_with_man(man, A, X0=opt_pre, maxiter=20)
    # opt_mid = opt_pre

    beta = alpha[1] * 30
    man = RealPositiveSemidefinite(n, d, alpha=alpha, beta=beta)
    opt = solve_dist_with_man(man, A, X0=opt_mid, maxiter=50)
    opt_mat = opt.Y @ opt.P @ opt.Y.T
    if False:
        print(A0)
        print(opt_mat)
    print(np.max(np.abs(A0-opt_mat)))


def test_geodesics():
    from scipy.linalg import expm
    alpha = np.random.randint(1, 10, (2)) * .1
    beta = alpha[1] * .1
    m, d = (5, 3)
    man = RealPositiveSemidefinite(m, d, alpha=alpha, beta=beta)
    X = man.rand()

    alf = alpha[1]/alpha[0]
    
    def calc_gamma(man, X, xi, eta):
        g_inv_Jst_solve_J_g_in_Jst_DJ = man.g_inv(
            X, man.Jst(X, man.solve_J_g_inv_Jst(
                X, man.D_J(X, xi, eta))))
        proj_christoffel = man.proj_g_inv(
            X, man.christoffel_form(X, xi, eta))
        return g_inv_Jst_solve_J_g_in_Jst_DJ + proj_christoffel
        
    eta = man.randvec(X)
    g1 = calc_gamma(man, X, eta, eta)
    g2 = man.christoffel_gamma(X, eta, eta)
    print(man._vec(g1-g2))

    egrad = man._rand_ambient()
    print(man.base_inner_ambient(g1, egrad))
    print(man.rhess02_alt(X, eta, eta, egrad, 0))
    print(man.rhess02(X, eta, eta, egrad, man.zerovec(X)))
    # second solution:
    A = X.Y.T @ eta.tY
    t = 2
    K = eta.tY - X.Y @ (X.Y.T @ eta.tY)
    Yp, R = np.linalg.qr(K)

    x_mat = np.bmat([[2*alf*A, -R.T], [R, zeros((d, d))]])
    Yt = np.bmat([X.Y, Yp]) @ expm(t*x_mat)[:, :d] @ \
        expm(t*(1-2*alf)*A)
    x_d_mat = x_mat[:, :d].copy()
    x_d_mat[:d, :] += (1-2*alf) * A
    Ydt = np.bmat([X.Y, Yp]) @ expm(t*x_mat) @ x_d_mat @\
        expm(t*(1-2*alf)*A)
    x_dd_mat = x_mat @ x_d_mat + x_d_mat @ ((1-2*alf)*A)
    Yddt = np.bmat([X.Y, Yp]) @ expm(t*x_mat) @ x_dd_mat @\
        expm(t*(1-2*alf)*A)
    
    sqrtP = X.evec @ np.diag(np.sqrt(X.evl)) @ X.evec.T
    isqrtP = X.evec @ np.diag(1/np.sqrt(X.evl)) @ X.evec.T
    Pinn = t*isqrtP@eta.tP@isqrtP
    ePinn = expm(Pinn)
    Pt = sqrtP@ePinn@sqrtP
    Pdt = eta.tP@isqrtP@ePinn@sqrtP
    Pddt = eta.tP@isqrtP@ ePinn@isqrtP@eta.tP
    
    Xt = psd_point(np.array(Yt), np.array(Pt))
    Xdt = psd_ambient(np.array(Ydt), np.array(Pdt))
    Xddt = psd_ambient(np.array(Yddt), np.array(Pddt))
    gcheck = Xddt + calc_gamma(man, Xt, Xdt, Xdt)
    
    print(man._vec(gcheck))
    Xt1 = man.exp(X, t*eta)
    print((Xt1.Y - Xt.Y))
    print((Xt1.P - Xt.P))

    
if __name__ == '__main__':
    test_geodesics()
    optim_test()
    test_all_projections()
