from collections import OrderedDict
from sympy import Integer
from ManNullRange.symbolic.SymMat import (
    latex_map, mat_latex, t, trace, mat_spfy, xtrace)
from ManNullRange.symbolic.flag import (
    make_stiefel_block, make_ambient_block, make_b, make_al_stiefel, make_al,
    list_spfy, list_DDR, list_xtrace)

"""
from ManNullRange.symbolic import SymMat as sm
from ManNullRange.symbolic.SymMat import (
    matrices, t, scalars, mat_spfy, xtrace, trace, stiefels, DDR,
    LatexMap, MatLatex)
"""


def pprint(expr):
    print(latex_map(mat_latex(expr), OrderedDict(
        [('fYY', r'f_{YY}'), ('fY', 'f_Y'), ('al', r'\alpha')])))


def calc_flag():
    """block Stiefel matrix with 3 blocks
    """
    p = 3
    Y = make_stiefel_block(p, 'Y')
    eta = make_ambient_block(p, 'eta')
    b = make_b(p)
    al = make_al(p)
    al = make_al_stiefel(p, [Integer(5), Integer(1)])

    # al = range(1, 1+p*(p+1)//2)

    def get_al_idx(r, j):
        return (r-1)*(p+1)+j
    
    def J(Y, eta):
        a = []
        for r in range(p):
            a.append(t(Y[r])*eta[r])
            for s in range(r+1, p):
                a.append(t(Y[r])*eta[s] + t(eta[r])*Y[s])
        return a

    def inner_ambient(v1, v2):
        expr = Integer(0)
        for j in range(p):
            expr += trace(v1[j]*t(v2[j]))
        return expr

    def inner_range_J(a1, a2):
        expr = Integer(0)
        for idx in range(len(a1)):
            expr += trace(a1[idx]*t(a2[idx]))
        return expr
    
    def J_adj(Y, a):
        dY = make_ambient_block(p, 'dY')
        jaj = []
        jsim = [mat_spfy(jj) for jj in J(Y, dY)]
        expr = mat_spfy(inner_range_J(jsim, a))
        for r in range(p):
            jaj.append(xtrace(expr, dY[r]))
        return jaj

    def g(Y, eta):
        expr = []
        for r in range(1, p+1):
            zidx = get_al_idx(r, 0)
            expr.append(al[zidx]*eta[r-1])
            for j in range(1, p+1):
                rjidx = get_al_idx(r, j)
                expr[r-1] += (al[rjidx]-al[zidx])*Y[j-1]*t(Y[j-1])*eta[r-1]
        return expr

    def g_inv(Y, eta):
        expr = []
        for r in range(1, p+1):
            zidx = get_al_idx(r, 0)
            expr.append(1/al[zidx]*eta[r-1])
            for j in range(1, p+1):
                rjidx = get_al_idx(r, j)
                expr[r-1] += (1/al[rjidx]-1/al[zidx])*Y[j-1]*t(Y[j-1])*eta[r-1]
        return expr
    
    j1 = J(Y, eta)
    j2 = J_adj(Y, b)
    print(j1)
    print(j2)
    J_giv_J_adj = list_spfy(J(Y, g_inv(Y, J_adj(Y, b))), Y, [])
    print(J_giv_J_adj)
    """
    ss = []
    for jj in J_giv_J_adj:
        j0 = mat_spfy(jj)
        ss.append(spfy_flag(j0, Y, [], rev=False))
    """
    
    def solve_JginvJadj(Y, b):
        ret = []
        cnt = 0
        for r in range(1, p+1):
            for s in range(r, p+1):
                i_r_s = get_al_idx(r, s)

                if r == s:
                    ret.append(b[cnt])
                else:
                    i_s_r = get_al_idx(s, r)
                    ret.append((al[i_r_s]*al[i_s_r]/(al[i_r_s]+al[i_s_r])) *
                               b[cnt])
                cnt += 1
        return ret

    print(solve_JginvJadj(Y, J_giv_J_adj))

    def g_inv_J_adj(Y, a):
        expr = g_inv(Y, list_spfy(J_adj(Y, a), Y, []))
        return list_spfy(expr, Y, [])

    g_inv_J_adj_expr = g_inv_J_adj(Y, b)

    def g_inv_J_adj_func(Y, a):
        rdict = dict((b[i], a[i]) for i in range(len(b)))
        return [list_spfy(g_inv_J_adj_expr[i].xreplace(rdict), Y, [])
                for i in range(len(a))]

    def proj(Y, omg):
        w = g_inv_J_adj(Y, solve_JginvJadj(Y, list_spfy(J(Y, omg), Y, [])))
        return [mat_spfy(omg[i] - w[i]) for i in range(len(omg))]
    proj_expr = list_spfy(proj(Y, eta), Y, [])
    
    def proj_func(Y, omg):
        rdict = dict((eta[i], omg[i]) for i in range(len(omg)))
        return [proj_expr[i].xreplace(rdict) for i in range(len(omg))]

    r_gradient_expr = list_spfy(
        proj_func(Y, g_inv(Y, eta)), Y, [])
                                
    def r_gradient(Y, omg):
        rdict = dict((eta[i], omg[i]) for i in range(len(omg)))
        return [r_gradient_expr[i].xreplace(rdict)
                for i in range(len(omg))]

    xi = make_ambient_block(p, 'xi')
    phi = make_ambient_block(p, 'phi')

    # jo = list_spfy(g(Y, eta), Y, [])
    # ee = list_DDR(jo, Y, phi)
    
    trilinear = mat_spfy(inner_ambient(list_DDR(g(Y, eta), Y, phi), xi))
    xcross = list_xtrace(trilinear, phi)

    ddr_xi = list_DDR(g(Y, eta), Y, xi)
    ddr_eta = list_DDR(g(Y, xi), Y, eta)
    
    K1 = [(Integer(1)/Integer(2))*(ddr_eta[jj] + ddr_xi[jj] - xcross[jj])
          for jj in range(len(xcross))]
    K = list_spfy(K1, Y, [xi, eta])
    pprint(K)

    d_proj_expr = list_DDR(proj_expr, Y, xi)
    
    def d_proj_func(Y, xi, omg):
        rdict = dict((eta[i], omg[i]) for i in range(len(omg)))
        return [aa.xreplace(rdict) for aa in d_proj_expr]

    dp_xi_eta = d_proj_func(Y, xi, eta)
    prK0 = r_gradient(Y, K)
    prK1 = list_spfy(prK0, Y, [])
    prK = list_spfy(prK1, Y, (xi, eta))
    Gamma0 = [prK[jj] - dp_xi_eta[jj] for jj in range(len(prK))]
    Gamma = mat_spfy(
        list_spfy(Gamma0, Y, (xi, eta)))
    # one more time
    Gamma = mat_spfy(
        list_spfy(Gamma, Y, (xi, eta)))
    
    print("This is the Christoffel function:")
    pprint(Gamma)
    fY = make_ambient_block(p, 'fY')

    # t(eta)*fYY*xi
    rhess02 = mat_spfy(inner_ambient([-Gamma[jj] for jj in range(p)], fY))
    rhess11_bf_gr = list_xtrace(rhess02, eta)
    print("This is the Riemannian Hessian Vector Product:")
    pprint(rhess11_bf_gr)
    

    

            
    

    
