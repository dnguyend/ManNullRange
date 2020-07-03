from . import SymMat as sm
from .SymMat import (matrices, scalars, stiefels, t, to_monomials,
                     extract_commutative, Id, mat_spfy, DDR, xtrace)
from sympy import Integer, preorder_traversal, Mul


def make_b(p):
    b = []
    for r in range(1, p+1):
        for s in range(r, p+1):
            b.append(matrices('b%d%d' % (r, s)))
    return b


def make_al(p):
    st = []
    for i in range(1, p+1):
        for j in range(p+1):
            st.append('al%d%d' % (i, j))
    return scalars(' '.join(st))


def make_al_stiefel(p, alfa=None):
    st = []
    if alfa is None:
        alfa = [Integer(2), Integer(1)]
    for i in range(1, p+1):
        for j in range(p+1):
            if j == 0:
                st.append(alfa[0])
            else:
                st.append(alfa[1])
    return st


def make_stiefel_block(p, prefix):
    st = ['%s%d' % (prefix, i) for i in range(1, p+1)]
    return stiefels(' '.join(st))


def make_ambient_block(p, prefix):
    st = ['%s%d' % (prefix, i) for i in range(1, p+1)]
    return matrices(' '.join(st))


def find_idx(a, alist):
    try:
        return alist.index(a)
    except Exception:
        return None


def in_tangent(a, many_list):
    for ll in many_list:
        fidx = find_idx(a, ll)
        if fidx is not None:
            return fidx, ll
    return None, None

    
def move_block_flag_tangent(
        xargs, i, y, y_next, point, tangentvars):
    # several tangents to a point
    # here a point is a tuple
    # a tangent is also a tuple
    rev = False
    if y.func == t:
        tidx, tg = in_tangent(y.args[0], tangentvars)
        pidx = find_idx(y_next, point)
        if (tidx is not None) and (pidx is not None) and (tidx == pidx):
            return [Integer(0)], Integer(0)
        
        tidx_rev, tg_rev = in_tangent(y_next, tangentvars)
        pidx_rev = find_idx(y.args[0],  point)
        if (tidx_rev is not None) and (pidx_rev is not None) and\
           (tidx_rev == pidx_rev):
            return [Integer(0)], Integer(0)
        # rev is t(Y)xi -> -t(xi)Y
        if rev and pidx_rev is not None and tidx_rev is not None:
            return list(xargs[:i]) + [t(tg_rev[pidx_rev]), point[tidx_rev]] +\
                list(xargs[i+2:]), Integer(-1)
        elif pidx is not None and tidx is not None:
            # case t(xi)Y -> t(Y)xi
            return list(xargs[:i]) + [t(point[tidx]), tg[pidx]] +\
                list(xargs[i+2:]), Integer(-1)
    return None, None


def spfy_flag(expr, Y, tangentvars):
    e = to_monomials(expr)
    while True:
        has_change = False
        for x in preorder_traversal(e):
            if hasattr(x, 'is_Mul') and x.is_Mul:
                comm, noncom = extract_commutative(x.args)
                if len(comm) + len(noncom) < len(x.args):
                    has_change = True
                i = 0
                not_zero = True
                while i < len(noncom) and not_zero:
                    y = noncom[i]
                    if i + 1 < len(noncom):
                        if i+1 < len(noncom):
                            new_ncom = remove_flag_pair(
                                noncom, i, y, noncom[i+1])
                            if new_ncom is not None:
                                noncom = new_ncom
                                if len(noncom) > 0 and noncom[0] == Integer(0):
                                    not_zero = False
                                has_change = True
                        if i + 1 < len(noncom):
                            new_ncom, sig = move_block_flag_tangent(
                                noncom, i, y, noncom[i+1], Y, tangentvars)
                            if new_ncom is not None:
                                noncom = new_ncom
                                comm.append(sig)
                                if sig == Integer(0):
                                    not_zero = False
                                has_change = True
                            else:
                                i += 1
                        else:
                            i += 1
                    else:
                        i += 1
                if has_change:
                    if len(comm) == 0 and len(noncom) == 0:
                        # Id * Id
                        new_term = Id(None)
                    else:
                        new_term = Mul(*(list(comm)+list(noncom)))
                    e = e.xreplace({x: new_term})
                    break
        if not has_change:
            break
    return e


def remove_flag_pair(xargs, i, y, y_next):
    if (y_next in sm.g_stiefels):
        if y.func == t and y.args[0] in sm.g_stiefels:
            if y == t(y_next):
                if i == 0:
                    return xargs[2:]
                elif i == len(xargs)-2:
                    return xargs[:-2]
                else:
                    return list(xargs[:i]) + list(xargs[i+2:])
            else:
                # return list(xargs[:i]) + [Integer(0)] + list(xargs[i+2:])
                return [Integer(0)]
        elif y_next.func == t and y in sm.g_stiefels and y_next.args[0] != y:
            return [Integer(0)]
        
    if (y_next in sm.g_cstiefels) and y == t(y_next):
        if i == 0:
            return xargs[2:]
        elif i == len(xargs)-2:
            return xargs[:-2]
        else:
            return list(xargs[:i]) + list(xargs[i+2:])
    elif (y.func == t and y_next in sm.g_cstiefels
          and y.args[0] == sm.g_cstiefels[y_next]):
        return list(xargs[:i]) + [Integer(0)] + list(xargs[i+2:])
    elif (y.func == t and y.args[0] in sm.g_cstiefels and
          y_next == sm.g_cstiefels[y.args[0]]):
        return list(xargs[:i]) + [Integer(0)] + list(xargs[i+2:])
    return None


def list_spfy(alist, Y, tangentvars):
    return [mat_spfy(spfy_flag(mat_spfy(jj), Y, tangentvars)) for
            jj in alist]


def list_DDR(expr_list, Y, v):
    ret = []
    for ex in expr_list:
        ddx = Integer(0)
        for jj in range(len(Y)):
            ddx += DDR(ex, Y[jj], v[jj])
        ret.append(ddx)
    return ret


def list_xtrace(expr, v):
    return [xtrace(expr, vv) for vv in v]

