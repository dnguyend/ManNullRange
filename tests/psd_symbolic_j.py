from collections import OrderedDict
from sympy import Integer
from ManNullRange.symbolic.SymMat import (
    matrices, stiefels, scalars, t, mat_spfy,
    xtrace, trace, DDR, sym_symb, asym_symb, sym,
    latex_map, mat_latex, simplify_pd_tangent, inv,
    simplify_stiefel_tangent)


def pprint(expr):
    print(latex_map(mat_latex(expr), OrderedDict(
        [('fYY', r'f_{YY}'), ('fY', 'f_Y'), ('al', r'\alpha')])))


def calc_psd():
    # manifold is on pair (Y, P)
    Y = stiefels('Y')    
    P = sym_symb('P')
    a_P = asym_symb('a_P')
    a_YP, eta_Y, eta_P = matrices('a_YP eta_Y eta_P')

    al0, al1, bt = scalars('al0 al1 bt')

    def g(Y, P, omg_Y, omg_P):
        return al0*omg_Y+(al1-al0)*Y*t(Y)*omg_Y, bt*inv(P)*omg_P*inv(P)

    def ginv(Y, P, omg_Y, omg_P):
        return 1/al0*omg_Y+(1/al1-1/al0)*Y*t(Y)*omg_Y, 1/bt*P*omg_P*P

    # check that ginv \circ g is id
    e1, e2 = ginv(Y, P, *(g(Y, P, eta_Y, eta_P)))
    e1 = mat_spfy(e1)
    e2 = mat_spfy(e2)
    print(e1, e2)

    def ambient_inner(Y, P, omg_Y, omg_P, xi_Y, xi_P):
        return mat_spfy(
            trace(
                mat_spfy(
                    (al0*omg_Y+(al1-al0)*Y*t(Y)*omg_Y) * t(xi_Y))) +
            trace(mat_spfy(
                bt*inv(P)*omg_P*inv(P)*t(xi_P))))

    def base_ambient_inner(Y, P, omg_Y, omg_P, xi_Y, xi_P):
        return mat_spfy(
            trace(mat_spfy(
                omg_Y * t(xi_Y))) +
            trace(mat_spfy(
                omg_P*t(xi_P))))

    def EJ_inner(Y, P, a_P, a_YP, b_P, b_YP):
        return trace(
            mat_spfy(-a_P * b_P) + mat_spfy(
                a_YP*t(b_YP)))

    qat = matrices('qat')
    ipr = ambient_inner(Y, P, eta_Y, eta_P, Y * qat,  P*qat - qat*P)
    dqat = mat_spfy(xtrace(ipr, qat))
    print(dqat)

    def J(Y, P, omg_Y, omg_P):
        J_P = mat_spfy(omg_P - t(omg_P))
        J_YP = mat_spfy(
            al1*t(Y)*omg_Y + bt*omg_P*inv(P) - bt*inv(P)*omg_P)

        return J_P, J_YP
    
    def JT(Y, P, a_P, a_YP):
        dY, dP = matrices('dY dP')
        ipt = mat_spfy(
            EJ_inner(Y, P, *J(Y, P, dY, dP), a_P, a_YP))
        jty = mat_spfy(xtrace(ipt, dY))
        jtp = mat_spfy(xtrace(ipt, dP))
        return jty, jtp

    JTa_Y, JTa_P = JT(Y, P, a_P, a_YP)
    
    ginvJT = ginv(Y, P, JTa_Y, JTa_P)
    ginvJT_Y = mat_spfy(ginvJT[0])
    ginvJT_P = mat_spfy(ginvJT[1])
    pprint(ginvJT_Y)
    pprint(ginvJT_P)

    Jginv_P, Jginv_YP = J(Y, P, *ginv(Y, P, eta_Y, eta_P))
    pprint(Jginv_P)
    pprint(Jginv_YP)
    
    b_P, b_YP = J(Y, P, *ginvJT)
    pprint(b_P)
    pprint(b_YP)

    # even part of a_YP
    aYPev = sym(a_YP)

    exp1 = bt * (Integer(1)/Integer(2)*b_P - P*aYPev + aYPev*P)
    exp1 = mat_spfy(exp1)
    pprint(exp1)
    print('check the formula for even part of a_YP recover b_YP')
    exp1 = al1*aYPev + bt/2*(b_P*inv(P)-inv(P)*b_P) - sym(b_YP)
    pprint(mat_spfy(exp1))

    aYPodd = mat_spfy(a_YP - aYPev)
    exp2 = (al1-2*bt)*aYPodd + bt*P*aYPodd*inv(P) + bt*inv(P)*aYPodd*P -\
        Integer(1)/Integer(2)*(b_YP-t(b_YP))
    # this is the equation to solve a_YPodd
    print('this is the equation to solve a_YPodd')
    pprint(exp2)
        
    def even_solve(Y, P, b_P, b_YP):
        return Integer(1)/al1*sym(b_YP) +\
            bt/(Integer(2)*al1)*(inv(P)*b_P - b_P*inv(P))

    print(mat_spfy(even_solve(Y, P, b_P, b_YP)))
    xi_Y, xi_P, phi_Y, phi_P = matrices('xi_Y xi_P phi_Y phi_P')
    gYPeta = g(Y, P, eta_Y, eta_P)
    Dgxieta_Y = DDR(gYPeta[0], Y, xi_Y)
    Dgxieta_P = DDR(gYPeta[1], P, xi_P)

    gYPxi = g(Y, P, xi_Y, xi_P)
    Dgetaxi_Y = DDR(gYPxi[0], Y, eta_Y)
    Dgetaxi_P = DDR(gYPxi[1], P, eta_P)

    Dgxiphi_Y = DDR(gYPeta[0], Y, phi_Y)
    Dgxiphi_P = DDR(gYPeta[1], P, phi_P)

    tr3 = mat_spfy(
        base_ambient_inner(Y, P, Dgxiphi_Y, Dgxiphi_P, xi_Y, xi_P))
    xcross_Y = xtrace(tr3, phi_Y)
    xcross_P = xtrace(tr3, phi_P)
        
    K_Y = (Integer(1)/Integer(2))*(Dgxieta_Y + Dgetaxi_Y - xcross_Y)
    K_P = (Integer(1)/Integer(2))*(Dgxieta_P + Dgetaxi_P - xcross_P)

    K_Y = simplify_stiefel_tangent(K_Y, Y, (eta_Y, xi_Y))
    K_P = simplify_pd_tangent(K_P, P, (eta_P, xi_P))
    pprint(K_Y)
    pprint(K_P)
    
    giKY, giKP = ginv(Y, P, K_Y, K_P)

    giKY1 = simplify_stiefel_tangent(giKY, Y, (eta_Y, xi_Y))
    giKP1 = simplify_pd_tangent(giKP, P, (eta_P, xi_P))
    jK1 = J(Y, P, giKY1, giKP1)
    jKP = simplify_pd_tangent(jK1[0], P, (eta_P, xi_P))
    jKY = simplify_stiefel_tangent(jK1[1], Y, (eta_Y, xi_Y))
    jKY1 = simplify_pd_tangent(jKY, P, (eta_P, xi_P))
    pprint(jKP)
    pprint(jKY1)
        
    def DJ(Y, P, xi_Y, xi_P, eta_Y, eta_P):
        expr_P, expr_YP = J(Y, P, eta_Y, eta_P)
        der_P = DDR(expr_P, Y, xi_Y)+DDR(expr_P, P, xi_P)
        der_YP = DDR(expr_YP, Y, xi_Y)+DDR(expr_YP, P, xi_P)
        return mat_spfy(der_P), mat_spfy(der_YP)

    pprint(DJ(Y, P, xi_Y, xi_P, eta_Y, eta_P))
