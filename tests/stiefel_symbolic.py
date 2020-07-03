from collections import OrderedDict
from sympy import symbols, Integer
from ManNullRange.symbolic import SymMat as sm
from ManNullRange.symbolic.SymMat import (
    matrices, t, scalars, mat_spfy, xtrace, trace, stiefels, DDR,
    latex_map, mat_latex, simplify_stiefel_tangent)


def pprint(expr):
    """pretty print
    """
    print(latex_map(mat_latex(expr), OrderedDict(
        [('fYY', r'f_{YY}'), ('fY', 'f_Y'), ('al', r'\alpha')])))


def calc_stiefel():
    # Y is a matrix point
    eta = matrices('eta')
    Y = stiefels('Y')
    a = sm.sym_symb('a')
    al0, al1 = scalars('al0 al1')
    # scalars are symmetric
    sm.g_symms.update((al0, al1))
    
    def J(Y, eta):
        return mat_spfy(t(Y) * eta + t(eta) * Y).doit()

    def J_adj(Y, a):
        dY = symbols('dY', commutative=False)
        return xtrace(trace(mat_spfy(J(Y, dY) * a)), dY)

    def g(Y, eta):
        return al0*eta+(al1-al0)*Y*t(Y)*eta

    def g_inv(Y, eta):
        return mat_spfy(1/al0*eta + (1/al1-1/al0)*Y*t(Y)*eta)
    
    J_giv_J_adj = J(Y, g_inv(Y, J_adj(Y, a)))
    print(J_giv_J_adj)
        
    def proj(Y, omg):
        jo = mat_spfy(J(Y, omg))
        ifactor = al1/Integer(4)
        return omg - mat_spfy(
            g_inv(Y, mat_spfy(J_adj(Y, ifactor*jo))))
    
    def r_gradient(Y, omg):
        return mat_spfy(
            proj(Y, mat_spfy(g_inv(Y, omg))))
        
    print(r_gradient(Y, eta))
    
    xi, phi = matrices('xi phi')
    trilinear = mat_spfy(trace(DDR(g(Y, eta), Y, phi) * t(xi)))
    xcross = xtrace(trilinear, phi)
    K = (Integer(1)/Integer(2))*(DDR(g(Y, eta), Y, xi) +
                                 DDR(g(Y, xi), Y, eta) - xcross)
    
    def d_proj(Y, xi, omg):
        e = matrices('e')
        r = mat_spfy(proj(Y, e))
        expr = DDR(r, Y, xi)
        return expr.xreplace({e: omg})

    dp_xi_eta = d_proj(Y, xi, eta)
    prK = simplify_stiefel_tangent(proj(Y, g_inv(Y, K)), Y, (xi, eta))
    Gamma = mat_spfy(
        simplify_stiefel_tangent(prK - dp_xi_eta, Y, (xi, eta)))
    print("This is the Christoffel function:")
    pprint(Gamma)
    fY, fYY = matrices('fY fYY')
    rhess02 = trace(mat_spfy(t(eta)*fYY*xi-Gamma * t(fY)))
    rhess11_bf_gr = xtrace(rhess02, eta)
    print("This is the Riemannian Hessian Vector Product:")
    pprint(rhess11_bf_gr)
