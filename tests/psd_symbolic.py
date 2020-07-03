from collections import OrderedDict
from sympy import Integer
from ManNullRange.symbolic import SymMat as sm
from ManNullRange.symbolic.SymMat import (
    matrices, stiefels, scalars, t, mat_spfy,
    xtrace, trace, DDR, sym_symb, asym_symb,
    latex_map, mat_latex, simplify_pd_tangent, inv)


def pprint(expr):
    print(latex_map(mat_latex(expr), OrderedDict(
        [('fYY', r'f_{YY}'), ('fY', 'f_Y'), ('al', r'\alpha')])))


def calc_psd_range():
    # in this method, the ambient is still
    # R(n times p) + Symmetric(p)
    # Horizontal is still
    # al1*t(Y)*omg_Y + bt*omg_P*inv(R*R) - bt*inv(R*R)*omg_P)
    # manifold is on pair (Y, P)
    # Use pair B, D. B size n*(n-p), D size p(p+1)/2
    # Use the embedding
    # omg_Y = bt(Y*(inv(P)*D - D*inv(P))
    # omg_P = al1*D
    # so need to set up a new variable Y_0 and relation YY_0=0
    
    Y, Y0 = stiefels('Y Y0')
    sm.g_cstiefels[Y] = Y0
    sm.g_cstiefels[Y0] = Y
    
    B, eta_Y, eta_P = matrices('B eta_Y eta_P')
    P, D = sym_symb('P D')
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

    def base_ambient_inner(omg_Y, omg_P, xi_Y, xi_P):
        return mat_spfy(
            trace(
                mat_spfy(omg_Y * t(xi_Y))) +
            trace(mat_spfy(
                omg_P*t(xi_P))))

    def ambient_inner(Y, P, omg_Y, omg_P, xi_Y, xi_P):
        return mat_spfy(
            trace(
                mat_spfy(
                    (al0*omg_Y+(al1-al0)*Y*t(Y)*omg_Y) * t(xi_Y))) +
            trace(mat_spfy(
                bt*inv(P)*omg_P*inv(P)*t(xi_P))))

    def EN_inner(Y, P, Ba, Da, Bb, Db):
        return trace(
            mat_spfy(Da * Db) + mat_spfy(
                Ba*t(Bb)))

    qat = asym_symb('qat')

    ipr = ambient_inner(Y, P, eta_Y, eta_P, Y * qat,  P*qat - qat*P)
    dqat = mat_spfy(xtrace(ipr, qat))
    print(dqat)

    def N(Y, P, B, D):
        N_Y = mat_spfy(bt*Y*(inv(P)*D - D*inv(P))) + Y0*B
        N_P = mat_spfy(
            al1*D)

        return N_Y, N_P
    
    def NT(Y, P, omg_Y, omg_P):
        nB, nD = matrices('nB nD')
        ipt = mat_spfy(
            base_ambient_inner(*N(Y, P, nB, nD), omg_Y, omg_P))
        ntB = mat_spfy(xtrace(ipt, nB))
        ntD1 = mat_spfy(xtrace(ipt, nD))
        ntD = mat_spfy(Integer(1)/Integer(2)*(ntD1 + t(ntD1)))
        return ntB, ntD
    
    # check that image of N is horizontal:
    print(mat_spfy(xtrace(
        mat_spfy(
            ambient_inner(Y, P, *N(Y, P, B, D), Y*qat, P*qat - qat*P)), qat)))
    NTe_B, NTe_D = NT(Y, P, eta_Y, eta_P)
    mdict = {'bt': r'\beta', 'al': r'\alpha'}
    print(latex_map(mat_latex(NTe_B), mdict))
    print(latex_map(mat_latex(NTe_D), mdict))

    gN = g(Y, P, *N(Y, P, B, D))
    gN_B = mat_spfy(gN[0])
    gN_D = mat_spfy(gN[1])
    print(latex_map(mat_latex(gN_B), mdict))
    print(latex_map(mat_latex(gN_D), mdict))

    NTgN_B, NTgN_D = NT(Y, P, *gN)
    
    print(latex_map(mat_latex(NTgN_B), mdict))
    print(latex_map(mat_latex(NTgN_D), mdict))
    
    NTg_B, NTg_D = NT(Y, P, *g(Y, P, eta_Y, eta_P))
    print(latex_map(mat_latex(NTg_B), mdict))
    # print(latex_map(sp.latex(NTgN_P), mdict))
    print(latex_map(mat_latex(NTg_D), mdict))

    def sym(x):
        return mat_spfy(
            Integer(1)/Integer(2)*(x + t(x)))
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
        ambient_inner(Y, P, Dgxiphi_Y, Dgxiphi_P, eta_Y, eta_P))
    xcross_Y = xtrace(tr3, phi_Y)
    xcross_P = xtrace(tr3, phi_P)
        
    K_Y = (Integer(1)/Integer(2))*(Dgxieta_Y + Dgetaxi_Y - xcross_Y)
    K_P = (Integer(1)/Integer(2))*(Dgxieta_P + Dgetaxi_P - xcross_P)

    pprint(K_Y)
    pprint(K_P)


def J_method():
    aYPev = sym(a_YP)
    
    exp1 = bt * (Integer(1)/Integer(2)*b_P - P*aYPev + aYPev*P)
    exp1 = mat_spfy(exp1)
    print(exp1)
    """
    exp1 = bt * (Integer(1)/Integer(2)*(b_P*inv(P) - inv(P)* b_P)
                 - P*aYPev*inv(P) + 2*aYPev - inv(P)* aYPev*P)
    """
    exp1 = al1*aYPev + bt/2*(b_P*inv(P)-inv(P)*b_P) - sym(b_YP)
    print(mat_spfy(exp1))
    aYPodd = mat_spfy(a_YP - aYPev)
    exp2 = (al1-2*bt)*aYPodd + bt*P*aYPodd*inv(P) + bt*inv(P)*aYPodd*P -\
        Integer(1)/Integer(2)*(b_YP-t(b_YP))
    print(exp2)
        
    def even_solve(Y, P, b_P, b_YP):
        return Integer(1)/al1*sym(b_YP) +\
            bt/(Integer(2)*al1)*(inv(P)*b_P - b_P*inv(P))

    print(mat_spfy(even_solve(Y, P, b_P, b_YP)))
    giKY, giKP = ginv(Y, P, K_Y, K_P)

    giKY1 = simplify_stiefel_tangent(giKY, Y, (eta_Y, xi_Y))
    giKP1 = simplify_pd_tangent(giKP, P, (eta_P, xi_P))
    jK1 = J(Y, P, giKY1, giKP1)
    jKP = simplify_pd_tangent(jK1[0], P, (eta_P, xi_P))
    jKY = simplify_stiefel_tangent(jK1[1], Y, (eta_Y, xi_Y))
    jKY1 = simplify_pd_tangent(jKY, P, (eta_P, xi_P))
    print(jKP)
    print(jKY1)
        
    def DJ(Y, P, xi_Y, xi_P, eta_Y, eta_P):
        expr_P, expr_YP = J(Y, P, eta_Y, eta_P)
        der_P = DDR(expr_P, Y, xi_Y)+DDR(expr_P, P, xi_P)
        der_YP = DDR(expr_YP, Y, xi_Y)+DDR(expr_YP, P, xi_P)
        return mat_spfy(der_P), mat_spfy(der_YP)

    dj1 = DJ(Y, P, xi_Y, xi_P, eta_Y, eta_P)
    print(dj1)
    

