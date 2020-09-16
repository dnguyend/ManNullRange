import numpy as np
import numpy.linalg as la
from numpy.random import (randint)
from numpy import zeros, allclose

from scipy.linalg import null_space
from ManNullRange.manifolds.ComplexFixedRank import (
    ComplexFixedRank, fr_ambient, fr_point, calc_D,
    complex_extended_lyapunov)
from ManNullRange.manifolds.tools import (
    rtrace, cvec, cunvec, ahsym, hsym, crandn)
from test_tools import check_zero, make_sym_pos, random_orthogonal, pprint


def test_inner(man, X):
    for i in range(10):
        al = man.alpha
        bt = man.beta
        gm = man.gamma
        eta1 = man._rand_ambient()
        eta2 = man._rand_ambient()
        U = X.U
        V = X.V
        Pinv = X.Pinv
        inn1 = al[0]*rtrace(eta1.tU.T.conj() @ eta2.tU) +\
            (al[1]-al[0])*rtrace(
                (eta1.tU.T.conj()@U) @ (U.T.conj()@eta2.tU)) +\
            gm[0]*rtrace(eta1.tV.T.conj() @ eta2.tV) +\
            (gm[1]-gm[0])*rtrace(
                (eta1.tV.T.conj()@V) @ (V.T.conj()@eta2.tV)) +\
            bt*rtrace(Pinv@eta1.tP@Pinv@eta2.tP.T.conj())
        assert(allclose(man.inner(X, eta1, eta2), inn1))
        eta2b = man.g(X, eta2)
        inn2 = man.base_inner_ambient(eta1, eta2b)
        inn3 = rtrace(eta1.tU @ eta2b.tU.T.conj()) +\
            rtrace(eta1.tP @ eta2b.tP.T.conj()) +\
            rtrace(eta1.tV @ eta2b.tV.T.conj())
        print(inn1, inn2, inn3)
        
        eta2a = man.g_inv(X, eta2)
        v1 = man.inner(X, eta1, eta2a)
        v2 = rtrace(eta1.tU @ eta2.tU.T.conj()) +\
            rtrace(eta1.tP @ eta2.tP.T.conj()) +\
            rtrace(eta1.tV @ eta2.tV.T.conj())
        assert(allclose(v1, v2))
    print(True)


def N(man, X, B, C, D):
    al, bt, gm = (man.alpha, man.beta, man.gamma)
    U, V, Piv = (X.U, X.V, X.Pinv)
    Dm = ahsym(D)
    Dp = hsym(D)
    bkPivDp = Piv @ Dp - Dp @ Piv
    U0 = null_space(X._U.T.conj())
    V0 = null_space(X._V.T.conj())
    U0B = U0 @ B
    V0C = V0 @ C
    return fr_ambient(
        U @ (-gm[1]*Dm + 1/(al[1]+gm[1])*bkPivDp)+U0B,
        V @(al[1]*Dm + 1/(al[1]+gm[1])*bkPivDp)+V0C,
        1/bt*Dp)


def make_N_mat(man, X):
    m, n, p = (man.m, man.n, man.p)
    dB = (m-p)*p*2
    dC = (n-p)*p*2
    nmat = np.zeros((man.tdim, man.dim))
    for i in range(man.dim):
        Ux = zeros(man.dim)
        Ux[i] = 1

        nmat[:, i] = man._vec(
            N(man, X,
              cunvec(Ux[:dB], (m-p, p)),
              cunvec(Ux[dB:dB+dC], (n-p, p)),
              cunvec(Ux[dB+dC:], (p, p))))
    return nmat

        
def make_g_mat(man, S):
    nn = man.tdim
    ret = zeros((nn, nn))
    for ii in range(ret.shape[1]):
        ee = zeros(nn)
        ee[ii] = 1
        ret[:, ii] = man._vec(
            man.g(S, man._unvec(
                ee)))
    return ret


def testN(man, X):
    m, n, p = (man.m, man.n, man.p)
    B = crandn(m-p, p)
    C = crandn(n-p, p)
    D = crandn(p, p)
    ee = N(man, X, B, C, D)
    print(check_zero(stU(X, ee)))
    print(check_zero(stV(X, ee)))
    print(check_zero(symP(X, ee)))
    print(check_zero(Hz(man, X, ee)))
    nmat = make_N_mat(man, X)
    ee2 = man._unvec(nmat @ np.concatenate(
        [cvec(B), cvec(C), cvec(D)]))
    print(check_zero(man._vec(ee - ee2)))
    

def stU(X, omg):
    U1 = X.U.T.conj()@omg.tU
    return U1 + U1.T.conj()


def stV(X, omg):
    V1 = X.V.T.conj()@omg.tV
    return V1 + V1.T.conj()


def symP(X, omg):
    return omg.tP - omg.tP.T.conj()


def Hz(man, X, omg):
    al = man.alpha
    bt = man.beta
    gm = man.gamma

    Piv = X.Pinv
    return al[1]*X.U.T.conj()@omg.tU + gm[1]*X.V.T.conj()@omg.tV +\
        bt*(omg.tP@Piv - Piv@omg.tP)
    
        
def test_J(man, X):
    from scipy.linalg import null_space
    al = man.alpha
    bt = man.beta
    gm = man.gamma
    p = man.p
    # U, V, P, Piv = (X.U, X.V, X.P, X.Pinv)
    
    jjmat = np.zeros((8*p*p, man.tdim))
    # this map is not full onto - but just check tangent maps to zero
    for i in range(man.tdim):
        Ux = zeros(man.tdim)
        Ux[i] = 1
        omg = man._unvec(Ux)
        jjmat[:, i] = np.concatenate(
            [cvec(stU(X, omg)),
             cvec(stV(X, omg)),
             cvec(symP(X, omg)),
             cvec(Hz(man, X, omg))])

    # nsp = null_space(jjmat)
    # prj = nsp @ la.solve(nsp.T.conj() @ nsp, nsp.T.conj())
    
    omg = man._rand_ambient()
    eta = man.proj(X, omg)

    def rand_vertical():
        oo = crandn(man.p, man.p)
        oo = oo - oo.T.conj()
        return fr_ambient(
            X.U @ oo,
            X.V @ oo,
            -oo @ X.P + X.P @oo)

    vv = rand_vertical()
    print(man.inner(X, vv, eta))
    nmat = make_N_mat(man, X)
    gmat = make_g_mat(man, X)
    NTgN = nmat.T @ gmat @ nmat
    bcd = nmat.T @ gmat @ man._vec(omg)
    m, n, p = (man.m, man.n, man.p)
    B = cunvec(bcd[:2*(m-p)*p], (m-p, p))
    C = cunvec(bcd[2*(m-p)*p:2*(m+n-2*p)*p], (n-p, p))
    D = cunvec(bcd[2*(m+n-2*p)*p:], (p, p))
    Dp = hsym(D)
    Dm = ahsym(D)
    Dm2 = al[1]*gm[1]*ahsym(X.V.T.conj()@omg.tV - X.U.T.conj()@omg.tU)
    pprint(Dm - Dm2)
    U0 = null_space(X._U.T.conj())
    V0 = null_space(X._V.T.conj())
    B2 = al[0]*U0.T.conj()@omg.tU
    pprint(B-B2)
    C2 = gm[0]*V0.T.conj()@omg.tV
    pprint(C-C2)
    Dmsolve = Dm2/(al[1]+gm[1])/al[1]/gm[1]
    pprint(Dmsolve - (1/(al[1]+gm[1]))*(
        ahsym(X.V.T.conj()@omg.tV - X.U.T.conj()@omg.tU)))
    Dpsolve = complex_extended_lyapunov(1/bt, 1/(al[1]+gm[1]), X.P, X.P@Dp@X.P)
    Drecv = (1/bt-2/(al[1]+gm[1]))*X.Pinv@Dpsolve@X.Pinv + 1/(al[1]+gm[1])*(
        X.Pinv@X.Pinv@Dpsolve+Dpsolve@X.Pinv@X.Pinv)
    pprint(Drecv - Dp)
    
    def get_D2(omg):
        UTomg = X.U.T.conj()@omg.tU
        VTomg = X.V.T.conj()@omg.tV
        Piv = X.Pinv
        D2 = hsym(X.Pinv@omg.tP@X.Pinv + 1/(al[1]+gm[1])*(
            al[1]*(Piv@UTomg - UTomg@Piv) +
            gm[1]*(Piv@VTomg - VTomg@Piv)))
        return D2
                                                
    Dp2 = get_D2(omg)
    print(check_zero(Dp - Dp2))

    def NTgN_opt(X, B, C, D):
        Piv = X.Pinv
        Dp, Dm = hsym(D), ahsym(D)
        Dp_ = (1/bt-2/(al[1]+gm[1]))*Piv@Dp@Piv + 1/(al[1]+gm[1])*(
            X.Pinv@X.Pinv@Dp+Dp@X.Pinv@X.Pinv)
        Dm_ = al[1]*gm[1]*(al[1]+gm[1])*Dm

        return al[0]*B, gm[0]*C, Dp_ + Dm_

    def solveNTgN(X, Bo, Co, Do):
        Dp, Dm = hsym(Do), ahsym(Do)
        Dm_ = 1/(al[1]*gm[1]*(al[1]+gm[1]))*Dm
        Dp_ = complex_extended_lyapunov(1/bt, 1/(al[1]+gm[1]), X.P, X.P@Dp@X.P)
        
        return 1/al[0]*Bo, 1/gm[0]*Co, Dp_ + Dm_

    def testNTgN(man, X):
        m, n, p = (man.m, man.n, man.p)
        B0 = crandn(m-p, p)
        C0 = crandn(n-p, p)
        D0 = crandn(p, p)
        out1 = NTgN @ np.concatenate(
            [cvec(B0), cvec(C0), cvec(D0)])
        out2a = NTgN_opt(X, B0, C0, D0)
        out2 = np.concatenate(
            [cvec(out2a[0]), cvec(out2a[1]), cvec(out2a[2])])
        print(check_zero(out1-out2))
        out2b = solveNTgN(X, *out2a)
        print(check_zero(out2b[2]-D0))
        print(check_zero(out2b[1]-C0))
        print(check_zero(out2b[0]-B0))
    
    Bs, Cs, Ds = solveNTgN(X, B, C, D)
    # the formulas Using the orthogonal complements of U and V
    
    # Dsm = ahsym(Ds)
    # Dsp = hsym(Ds)
    Bf = U0.T.conj() @ omg.tU
    Cf = V0.T.conj() @ omg.tV
    print(check_zero(Bf-Bs))
    print(check_zero(Cf-Cs))
    Dfm = 1/(al[1]+gm[1])*ahsym(X.V.T.conj()@omg.tV - X.U.T.conj()@omg.tU)
    UTomg = X.U.T.conj()@omg.tU
    VTomg = X.V.T.conj()@omg.tV
    Dfp = complex_extended_lyapunov(
        1/bt, 1/(al[1]+gm[1]), X.P,
        hsym(omg.tP + 1/(al[1]+gm[1])*(
            al[1]*(UTomg@X.P - X.P@UTomg) +
            gm[1]*(VTomg@X.P - X.P@VTomg))), X.evl, X.evec)
    ee1 = N(man, X, Bf, Cf, Dfp+Dfm)
    print(check_zero(Ds - Dfm - Dfp))
    print(check_zero(man._vec(ee1-eta)))
    

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


def test_projection(man, X):
    print("test projection")
    NN = 10
    ret = np.zeros(NN)

    for i in range(NN):
        omg = man._rand_ambient()
        Upr = man.proj(X, omg)
        ret[i] = check_zero(
            np.concatenate(
                [stU(X, Upr).reshape(-1),
                 stV(X, Upr).reshape(-1),
                 symP(X, Upr).reshape(-1),
                 Hz(man, X, Upr).reshape(-1)]))
    print(ret)
    
    if check_zero(ret) > 1e-8:
        print("not all works")
    else:
        print("All good")
    print("test randvec")
    ret_pr = np.zeros(NN)
    for i in range(NN):
        XX = man.rand()
        H = man.randvec(XX)
        Ue = man._rand_ambient()
        Upr = man.proj(XX, Ue)
        ret_pr[i] = man.inner(XX, Ue, H) - man.inner(XX, Upr, H)
        ret[i] = check_zero(stU(XX, H)) + check_zero(stV(XX, H)) +\
            check_zero(symP(XX, H)) + check_zero(Hz(man, XX, H))
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


def test_all_projections():
    alpha = randint(1, 10, 2) * .1
    gamma = randint(1, 10, 2) * .1
    beta = randint(1, 10, 1)[0] * .02
    m = 4
    n = 5
    d = 3
    man = ComplexFixedRank(m, n, d, alpha=alpha, beta=beta, gamma=gamma)
    X = man.rand()

    test_inner(man, X)
    test_J(man, X)
            
    # now check metric, Jst etc
    # check Jst: vectorize the operator J then compare Jst with jmat.T.conj()
    # test projection
    test_projection(man, X)
    # now diff projection

    for i in range(1):
        e = man._rand_ambient()
        X1 = man.rand()
        xi = man.randvec(X1)
        dlt = 1e-7
        X2 = fr_point(
            X1.U + dlt*xi.tU,
            X1.V + dlt*xi.tV,
            X1.P+dlt*xi.tP)

        # S = psd_point(S1.Y, S1.P)
        """
        Dp, Dm = calc_D(man, X1, e)
        Dp2, Dm2 = calc_D(man, X2, e)
        omg = e
        al1 = man.alpha[1]
        gm1 = man.gamma[1]
        U, V, P, Piv = (X1.U, X1.V, X1.P, X1.Pinv)
        agi = 1/(al1+gm1)
        DxiLDp = agi*(xi.tP @ Dp @ Piv + Piv @ Dp @ xi.tP -
                      P @ Dp @ Piv @ xi.tP @ Piv -
                      Piv @ xi.tP @ Piv @ Dp @ P)
        
        def LP(X, Dp):
            return (1/man.beta - 2*agi)*Dp +\
                agi*(X.Pinv @Dp @ X.P + X.P@ Dp @ X.Pinv)
        print((LP(X2, Dp) - LP(X1, Dp))/dlt)

        ddin = agi*(
            al1*(xi.tU.T.conj()@omg.tU@P - P@xi.tU.T.conj()@omg.tU +
                 U.T.conj() @omg.tU@xi.tP - xi.tP@U.T.conj()@omg.tU) +
            gm1*(xi.tV.T.conj()@omg.tV@P - P@xi.tV.T.conj()@omg.tV +
                 V.T.conj() @omg.tV@xi.tP - xi.tP@V.T.conj()@omg.tV)) - DxiLDp

        Ddp = extended_lyapunov(1/man.beta, agi, P, sym(ddin), X1.evl, X1.evec)
        Ddm = agi*asym(xi.tV.T.conj()@omg.tV - xi.tU.T.conj()@omg.tU)
        print(check_zero(Ddm - (Dm2-Dm)/dlt))
        print(check_zero(Ddp - (Dp2-Dp)/dlt))
        """
        d1 = (man.proj(X2, e) - man.proj(X1, e)).scalar_mul(1/dlt)
        d2 = man.D_proj(X1, xi, e)
        print(check_zero(man._vec(d1-d2)))
    
    for i in range(20):
        Uran = man._rand_ambient()
        Upr = man.proj(X, man.g_inv(X, Uran))
        Upr2 = man.proj_g_inv(X, Uran)
        print(check_zero(man._vec(Upr)-man._vec(Upr2)))
                
    # derives metrics
    for ii in range(10):
        X1 = man.rand()
        xi = man.randvec(X1)
        omg1 = man._rand_ambient()
        omg2 = man._rand_ambient()
        dlt = 1e-7
        X2 = fr_point(
            X1.U + dlt*xi.tU,
            X1.V + dlt*xi.tV,
            X1.P+dlt*xi.tP)
        p1 = man.inner(X1, omg1, omg2)
        p2 = man.inner(X2, omg1, omg2)
        der1 = (p2-p1)/dlt
        der2 = man.base_inner_ambient(
            man.D_g(X1, xi, omg2), omg1)
        print(der1-der2)

    # cross term for christofel
    for i in range(10):
        X1 = man.rand()
        xi = man.randvec(X1)
        eta1 = man.randvec(X1)
        eta2 = man.randvec(X1)
        dr1 = man.D_g(X1, xi, eta1)
        x12 = man.contract_D_g(X1, eta1, eta2)

        p1 = man.base_inner_ambient(dr1, eta2)
        p2 = man.base_inner_ambient(x12, xi)
        print(p1, p2, p1-p2)

    # now test christofel:
    # two things: symmetric on vector fields
    # and christofel relation
    # in the base metric
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
        v5 = man.base_inner_ambient(man.contract_D_g(S1, eta1, eta2), xi)
        print(v1, 0.5*(v2+v3-v4), v1-0.5*(v2+v3-v4))


def test_christ_flat():
    """now test that christofel preserve metrics:
    on the flat space
    d_xi <v M v> = 2 <v M nabla_xi v>
     v = proj(W) @ (aa W + b)
    """
    alpha = randint(1, 10, 2) * .1
    gamma = randint(1, 10, 2) * .1
    beta = randint(1, 10, 2)[0] * .1

    m = 4
    n = 5
    p = 3
    man = ComplexFixedRank(m, n, p, alpha=alpha, beta=beta, gamma=gamma)
    
    S = man.rand()
    
    xi = man.randvec(S)
    aaU = crandn(m*p, m*p)
    bbU = crandn(m*p)

    aaV = crandn(n*p, n*p)
    bbV = crandn(n*p)
    
    cc = crandn(p*p, p*p)
    dd = hsym(crandn(p, p))
        
    def v_func_flat(S):
        # a function from the manifold
        # to ambient
        csp = hsym((cc @ S.P.reshape(-1)).reshape(p, p))
        
        return fr_ambient(
            (aaU @ S.U.reshape(-1) + bbU).reshape(m, p),
            (aaV @ S.V.reshape(-1) + bbV).reshape(n, p),
            csp + dd)

    vv = v_func_flat(S)  # vv is not horizontal
    dlt = 1e-7
    Snew = fr_point(
        S.U + dlt * xi.tU,
        S.V + dlt * xi.tV,
        S.P + dlt * xi.tP)
    vnew = v_func_flat(Snew)

    val = man.inner(S, vv)
    valnew = man.inner(Snew, vnew)
    d1 = (valnew - val)/dlt
    dv = (vnew - vv).scalar_mul(1/dlt)
    nabla_xi_v = dv + man.g_inv(
        S, man.christoffel_form(S, xi, vv))
    # not equal bc vv is not horizontal:
    nabla_xi_va = dv + man.g_inv(
        S, super(ComplexFixedRank, man).christoffel_form(S, xi, vv))
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
    Snew = fr_point(S.U + dlt*xi.tU,
                    S.V + dlt*xi.tV,
                    S.P + dlt * xi.tP)
    vnew = vv_func(Snew)

    val = man.inner(S, vv)
    valnew = man.inner(
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
    alpha = randint(1, 10, 2) * .1
    gamma = randint(1, 10, 2) * .1
    beta = randint(1, 10, 2)[0] * .1

    m = 4
    n = 5
    p = 3
    man = ComplexFixedRank(m, n, p, alpha=alpha, beta=beta, gamma=gamma)

    S0 = man.rand()
    aaU = crandn(m*p, m*p)
    intcU = crandn(m*p)
    aaV = crandn(n*p, n*p)
    intcV = crandn(n*p)
    
    cc = crandn(p*p, p*p)
    p_intc = hsym(crandn(p, p))

    inct_xi = man._rand_ambient()
    aa_xiU = crandn(m*p, m*p)
    aa_xiV = crandn(n*p, n*p)
    cc_xi = crandn(p*p, p*p)
    
    def v_func(S):
        # a function from the manifold
        # to ambient
        csp = hsym((cc @ (S.P-S0.P).reshape(-1)).reshape(p, p))
        
        return man.proj(S, fr_ambient(
            (aaU @ (S.U-S0.U).reshape(-1) + intcU).reshape(m, p),
            (aaV @ (S.V-S0.V).reshape(-1) + intcV).reshape(n, p),
            csp + p_intc))

    SS = fr_point(S0.U, S0.V, S0.P)
    xi = man.proj(SS, inct_xi)

    nabla_xi_v, dv, cxv = calc_covar_numeric(
        man, SS, xi, v_func)

    def xi_func(S):
        csp_xi = hsym((cc_xi @ (S.P-S0.P).reshape(-1)).reshape(p, p))
        xi_amb = fr_ambient(
            (aa_xiU @ (S.U-S0.U).reshape(-1) +
             inct_xi.tU.reshape(-1)).reshape(m, p),
            (aa_xiV @ (S.V-S0.V).reshape(-1) +
             inct_xi.tV.reshape(-1)).reshape(n, p),
            csp_xi + inct_xi.tP)
        return man.proj(S, xi_amb)

    vv = v_func(SS)

    nabla_v_xi, dxi, cxxi = calc_covar_numeric(
        man, SS, vv, xi_func)
    diff = nabla_xi_v - nabla_v_xi
    print(diff.tU, diff.tV, diff.tP)
    # now do Lie bracket:
    dlt = 1e-7
    SnewXi = fr_point(SS.U+dlt*xi.tU,
                      SS.V+dlt*xi.tV,
                      SS.P+dlt*xi.tP)
    Snewvv = fr_point(SS.U+dlt*vv.tU,
                      SS.V+dlt*vv.tV,
                      SS.P+dlt*vv.tP)
    vnewxi = v_func(SnewXi)
    xnewv = xi_func(Snewvv)
    dxiv = (vnewxi - vv).scalar_mul(1/dlt)
    dvxi = (xnewv - xi).scalar_mul(1/dlt)
    diff2 = man.proj(SS, dxiv-dvxi)
    print(check_zero(man._vec(diff) - man._vec(diff2)))
                            

def num_deriv_amb(man, S, xi, func, dlt=1e-7):
    Snew = fr_point(S.U + dlt*xi.tU,
                    S.V + dlt*xi.tV,
                    S.P + dlt*xi.tP)
    return (func(Snew) - func(S)).scalar_mul(1/dlt)

    
def test_covariance_deriv():
    # now test full:
    # do covariant derivatives
    alpha = randint(1, 10, 2) * .1
    gamma = randint(1, 10, 2) * .1
    beta = randint(1, 10, 2)[0] * .1

    m = 4
    n = 5
    p = 3
    man = ComplexFixedRank(m, n, p, alpha=alpha, beta=beta, gamma=gamma)
    
    S = man.rand()
    
    aaU = crandn(m*p, m*p)
    aaV = crandn(n*p, n*p)
    cc = crandn(p*p, p*p)
    icpt = man._rand_ambient()

    def omg_func(S):
        csp = hsym((cc @ S.P.reshape(-1)).reshape(p, p))
        return fr_ambient(
            (aaU @ S.U.reshape(-1) + icpt.tU.reshape(-1)).reshape(m, p),
            (aaV @ S.V.reshape(-1) + icpt.tV.reshape(-1)).reshape(n, p),
            csp + icpt.tP)

    xi = man.randvec(S)
    egrad = omg_func(S)
    ecsp = hsym((cc @ xi.tP.reshape(-1)).reshape(p, p))
    ehess = fr_ambient(
        (aaU @ xi.tU.reshape(-1)).reshape(m, p),
        (aaV @ xi.tV.reshape(-1)).reshape(n, p),
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
    valrange = man.ehess2rhess(S, egrad, ehess, xi)
    print(check_zero(man._vec(valrange)-man._vec(val2_p)))
    print(check_zero(man._vec(valrange)-man._vec(val1)))
    print(check_zero(man._vec(valrange)-man._vec(valrangeB)))

    
def test_rhess_02():
    alpha = randint(1, 10, 2) * .1
    gamma = randint(1, 10, 2) * .1
    beta = randint(1, 10, 2)[0] * .1

    m = 4
    n = 5
    p = 3
    man = ComplexFixedRank(m, n, p, alpha=alpha, beta=beta, gamma=gamma)

    S = man.rand()
    # simple function. Distance to a given matrix
    # || S - A||_F^2 Basically SVD
    A = crandn(m, n)

    def f(S):
        diff = (A - S.U @ S.P @ S.V.T.conj())
        return rtrace(diff @ diff.T.conj())

    def df(S):
        return fr_ambient(-2*A @ S.V @ S.P,
                          -2*A.T.conj() @ S.U @S.P,
                          2*(S.P-S.U.T.conj() @ A @ S.V))
    
    def ehess_vec(S, xi):
        return fr_ambient(-2*A @ (xi.tV @ S.P + S.V @ xi.tP),
                          -2*A.T.conj() @ (xi.tU @S.P + S.U@xi.tP),
                          2*(xi.tP - xi.tU.T.conj()@A@S.V -
                             S.U.T.conj()@A@xi.tV))
    
    def ehess_form(S, xi, eta):
        ev = ehess_vec(S, xi)
        return rtrace(ev.tU.T.conj() @ eta.tU) +\
            rtrace(ev.tV.T.conj() @ eta.tV) +\
            rtrace(ev.tP.T.conj() @ eta.tP)
    
    xxi = man.randvec(S)
    dlt = 1e-8
    Snew = fr_point(
        S.U+dlt*xxi.tU,
        S.V+dlt*xxi.tV,
        S.P + dlt*xxi.tP)
    d1 = (f(Snew) - f(S))/dlt
    d2 = df(S)
    print(d1 - man.base_inner_ambient(d2,  xxi))

    dv1 = (df(Snew) - df(S)).scalar_mul(1/dlt)
    dv2 = ehess_vec(S, xxi)
    print(man._vec(dv1-dv2))
    
    eeta = man.randvec(S)
    d1 = man.base_inner_ambient((df(Snew) - df(S)), eeta) / dlt
    ehess_val = ehess_form(S, xxi, eeta)
    
    print(man.base_inner_ambient(dv2, eeta))
    print(d1, ehess_val, d1-ehess_val)

    # now check the formula: ehess = xi (eta_func(f)) - <D_xi eta, df(Y)>
    # promote eta to a vector field.

    mU1 = crandn(m, m)
    mV1 = crandn(n, n)
    m2 = crandn(p, p)
    m_p = crandn(p*p, p*p)

    def eta_field(Sin):
        return man.proj(S, fr_ambient(
            mU1 @ (Sin.U - S.U) @ m2,
            mV1 @ (Sin.V - S.V) @ m2,
            hsym((m_p @ (Sin.P - S.P).reshape(-1)).reshape(p, p)))) + eeta

    # xietaf: should go to ehess(xi, eta) + df(Y) @ etafield)
    xietaf = (man.base_inner_ambient(df(Snew), eta_field(Snew)) -
              man.base_inner_ambient(df(S), eta_field(S))) / dlt
    # appy eta_func to f: should go to tr(m1 @ xxi @ m2 @ df(Y).T.conj())
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
    rhessval = man.inner(S, rhessvec, eta1)
    print(man.inner(S, val2, eta1))
    print(rhessval)

    # check symmetric:
    ehvec_e = ehess_vec(S, eta1)
    rhessvec_e = man.ehess2rhess(S, egvec, ehvec_e, eta1)
    rhessval_e = man.inner(S, rhessvec_e, xi1)
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
    Snew = fr_point(
        S.U + dlt*xi1.tU,
        S.V + dlt*xi1.tV,
        S.P + dlt*xi1.tP)
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
        # if not(S.P.dtype == np.float):
        #    raise(ValueError("Non real"))
        diff = (A - S.U @ S.P @ S.V.T.conj())
        val = rtrace(diff @ diff.T.conj())
        # print('val=%f' % val)
        return val

    @Callable
    def egrad(S):
        return fr_ambient(-2*A @ S.V @ S.P,
                          -2*A.T.conj() @ S.U @S.P,
                          2*(S.P-S.U.T.conj() @ A @ S.V))
    
    @Callable
    def ehess(S, xi):
        return fr_ambient(-2*A @ (xi.tV @ S.P + S.V @ xi.tP),
                          -2*A.T.conj() @ (xi.tU @S.P + S.U@xi.tP),
                          2*(xi.tP - xi.tU.T.conj()@A@S.V -
                             S.U.T.conj()@A@xi.tV))

    prob = Problem(
        man, cost, egrad=egrad, ehess=ehess)

    solver = TrustRegions(maxtime=100000, maxiter=maxiter, use_rand=False)
    opt = solver.solve(prob, x=X0, Delta_bar=250)
    return opt

    
def optim_test():
    m, n, p = (1000, 500, 50)
    # m, n, p = (10, 3, 2)
    # simple function. Distance to a given matrix
    # || S - A||_F^2
    U0, _ = la.qr(crandn(m, p))
    V0, _ = la.qr(crandn(n, p))
    P0 = np.diag(randint(1, 1000, p)*.001)
    A0 = U0 @ P0 @ V0.T.conj()
    A = crandn(m, n)*1e-2 + A0

    alpha = np.array([1, 1])
    gamma = np.array([1, 1])
    print("alpha %s" % str(alpha))

    beta = alpha[1] * .1
    man = ComplexFixedRank(m, n, p, alpha=alpha, beta=beta, gamma=gamma)
    XInit = man.rand()
    opt_pre = solve_dist_with_man(man, A, X0=XInit, maxiter=20)

    beta = alpha[1] * 1
    man = ComplexFixedRank(m, n, p, alpha=alpha, beta=beta, gamma=gamma)
    opt_mid = solve_dist_with_man(man, A, X0=opt_pre, maxiter=20)
    # opt_mid = opt_pre

    beta = alpha[1] * 30
    man = ComplexFixedRank(m, n, p, alpha=alpha, beta=beta, gamma=gamma)
    opt = solve_dist_with_man(man, A, X0=opt_mid, maxiter=50)
    opt_mat = opt.U @ opt.P @ opt.V.T.conj()
    if False:
        print(A0)
        print(opt_mat)
    print(np.max(np.abs(A0-opt_mat)))


def test_geodesics():
    from scipy.linalg import expm
    alpha = randint(1, 10, 2) * .1
    gamma = randint(1, 10, 2) * .1
    beta = randint(1, 10, 2)[0] * .1

    m = 5
    n = 6
    p = 3
    man = ComplexFixedRank(m, n, p, alpha=alpha, beta=beta, gamma=gamma)
    X = man.rand()

    alf = alpha[1]/alpha[0]
    gmm = gamma[1]/gamma[0]
    
    def calc_Christoffel_Gamma(man, X, xi, eta):
        dprj = man.D_proj(X, xi, eta)
        proj_christoffel = man.proj_g_inv(
            X, man.christoffel_form(X, xi, eta))
        return proj_christoffel - dprj
        
    eta = man.randvec(X)
    g1 = calc_Christoffel_Gamma(man, X, eta, eta)
    g2 = man.christoffel_gamma(X, eta, eta)
    print(man._vec(g1-g2))

    egrad = man._rand_ambient()
    print(man.base_inner_ambient(g1, egrad))
    print(man.rhess02_alt(X, eta, eta, egrad, 0))
    print(man.rhess02(X, eta, eta, egrad, man.zerovec(X)))

    t = 2
    AU = X.U.T.conj() @ eta.tU
    KU = eta.tU - X.U @ (X.U.T.conj() @ eta.tU)
    Up, RU = np.linalg.qr(KU)

    xU_mat = np.bmat([[2*alf*AU, -RU.T.conj()], [RU, zeros((p, p))]])
    Ut = np.bmat([X.U, Up]) @ expm(t*xU_mat)[:, :p] @ \
        expm(t*(1-2*alf)*AU)
    xU_d_mat = xU_mat[:, :p].copy()
    xU_d_mat[:p, :] += (1-2*alf) * AU
    Udt = np.bmat([X.U, Up]) @ expm(t*xU_mat) @ xU_d_mat @\
        expm(t*(1-2*alf)*AU)
    xU_dd_mat = xU_mat @ xU_d_mat + xU_d_mat @ ((1-2*alf)*AU)
    Uddt = np.bmat([X.U, Up]) @ expm(t*xU_mat) @ xU_dd_mat @\
        expm(t*(1-2*alf)*AU)

    AV = X.V.T.conj() @ eta.tV
    KV = eta.tV - X.V @ (X.V.T.conj() @ eta.tV)
    Vp, RV = np.linalg.qr(KV)

    xV_mat = np.bmat([[2*gmm*AV, -RV.T.conj()], [RV, zeros((p, p))]])
    Vt = np.bmat([X.V, Vp]) @ expm(t*xV_mat)[:, :p] @ \
        expm(t*(1-2*gmm)*AV)
    xV_d_mat = xV_mat[:, :p].copy()
    xV_d_mat[:p, :] += (1-2*gmm) * AV
    Vdt = np.bmat([X.V, Vp]) @ expm(t*xV_mat) @ xV_d_mat @\
        expm(t*(1-2*gmm)*AV)
    xV_dd_mat = xV_mat @ xV_d_mat + xV_d_mat @ ((1-2*gmm)*AV)
    Vddt = np.bmat([X.V, Vp]) @ expm(t*xV_mat) @ xV_dd_mat @\
        expm(t*(1-2*gmm)*AV)
    
    sqrtP = X.evec @ np.diag(np.sqrt(X.evl)) @ X.evec.T.conj()
    isqrtP = X.evec @ np.diag(1/np.sqrt(X.evl)) @ X.evec.T.conj()
    Pinn = t*isqrtP@eta.tP@isqrtP
    ePinn = expm(Pinn)
    Pt = sqrtP@ePinn@sqrtP
    Pdt = eta.tP@isqrtP@ePinn@sqrtP
    Pddt = eta.tP@isqrtP@ ePinn@isqrtP@eta.tP
    
    Xt = fr_point(np.array(Ut),
                  np.array(Vt),
                  np.array(Pt))
    Xdt = fr_ambient(np.array(Udt),
                     np.array(Vdt),
                     np.array(Pdt))
    Xddt = fr_ambient(np.array(Uddt),
                      np.array(Vddt),
                      np.array(Pddt))
    gcheck = Xddt + calc_Christoffel_Gamma(man, Xt, Xdt, Xdt)
    
    print(man._vec(gcheck))
    Xt1 = man.exp(X, t*eta)
    print((Xt1.U - Xt.U))
    print((Xt1.V - Xt.V))
    print((Xt1.P - Xt.P))

    
if __name__ == '__main__':
    optim_test()
    test_all_projections()
