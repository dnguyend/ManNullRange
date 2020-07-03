from collections import OrderedDict
from sympy import symbols, Integer
from ManNullRange.symbolic import SymMat as sm
from ManNullRange.symbolic.SymMat import (
    matrices, t, mat_spfy, xtrace, trace, DDR,
    latex_map, mat_latex, simplify_pd_tangent, inv)


def pprint(expr):
    print(latex_map(mat_latex(expr), OrderedDict(
        [('fYY', r'f_{YY}'), ('fY', 'f_Y'), ('al', r'\alpha')])))


def calc_pd():
    """ For positive definite matrices
    Y is a matrix point, a positive definite matrix
    eta is an ambient point, same size with Y not necessarily
    symmetric or invertible
    b is a point in E_J. b is antisymmetric
    """
    # eta is an ambient
    Y = sm.sym_symb('Y')
    eta = matrices('eta')
    b = sm.asym_symb('b')
    
    def J(Y, eta):
        return eta - t(eta)
    
    def J_adj(Y, a):
        dY = symbols('dY', commutative=False)
        return xtrace(trace(mat_spfy(J(Y, dY) * a)), dY)

    def g(Y, eta):
        return inv(Y)*eta*inv(Y)

    def g_inv(Y, eta):
        return Y*eta*Y
    
    J_g_inv_J_adj = J(Y, g_inv(Y, J_adj(Y, b)))
    pprint(J_g_inv_J_adj)

    def solve_JginvJadj(Y, a):
        return Integer(-1)/Integer(4)*inv(Y)*a*inv(Y)

    def proj(Y, omg):
        jo = mat_spfy(J(Y, omg))
        cJinvjo = solve_JginvJadj(Y, jo)
        return mat_spfy(omg - mat_spfy(
            g_inv(Y, mat_spfy(J_adj(Y, cJinvjo)))))

    def r_gradient(Y, omg):
        return mat_spfy(
            proj(Y, mat_spfy(g_inv(Y, omg))))

    print(proj(Y, eta))
    print(r_gradient(Y, eta))

    xi, phi = matrices('xi phi')
    xcross = xtrace(mat_spfy(trace(DDR(g(Y, eta), Y, phi) * t(xi))), phi)
    K = (Integer(1)/Integer(2))*(
        DDR(g(Y, eta), Y, xi) + DDR(g(Y, xi), Y, eta) - xcross)

    def d_proj(Y, xi, omg):
        e = matrices('e')
        r = mat_spfy(proj(Y, e))
        expr = DDR(r, Y, xi)
        return expr.xreplace({e: omg})

    dp_xi_eta = d_proj(Y, xi, eta)
    prK = simplify_pd_tangent(proj(Y, mat_spfy(g_inv(Y, K))), Y, (xi, eta))
    Gamma = mat_spfy(
        simplify_pd_tangent(-dp_xi_eta+prK, Y, (xi, eta)))
    print("This is the Christoffel function:")
    pprint(Gamma)
    fY, fYY = matrices('fY fYY')
    rhess02 = trace(mat_spfy(t(eta)*fYY*xi-Gamma * t(fY)))
    rhess11_bf_gr = xtrace(rhess02, eta)
    print("This is the Riemannian Hessian Vector Product:")
    pprint(r_gradient(Y, rhess11_bf_gr))

        

