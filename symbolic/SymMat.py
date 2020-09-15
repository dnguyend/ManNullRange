from sympy import (symbols, Function, Symbol, Add, Mul, Pow,
                   Basic, preorder_traversal, Integer)
from sympy.printing.str import StrPrinter
from sympy.printing.latex import LatexPrinter
# from sympy.utilities.lambdify import lambdify, lambdastr


def matrices(names):
    ''' Call with  A,B,C = matrix('A B C') '''
    return symbols(names, commutative=False)


g_symms = set()
g_asymms = set()
g_scalars = set()
g_stiefels = set()
g_cstiefels = dict()  # dictionary of stiefel pairs


def scalars(names):
    symbs = symbols(names, commutative=True)
    if isinstance(symbs, tuple):
        g_scalars.update(symbs)
    else:
        g_scalars.add(symbs)
    return symbs


def stiefels(names):
    symbs = symbols(names, commutative=False)
    if isinstance(symbs, tuple):
        g_stiefels.update(symbs)
    else:
        g_stiefels.add(symbs)
    return symbs


def sym_symb(names):
    symbs = symbols(names, commutative=False)
    if isinstance(symbs, tuple):
        g_symms.update(symbs)
    else:
        g_symms.add(symbs)

    return symbs


def asym_symb(names):
    symbs = symbols(names, commutative=False)
    if isinstance(symbs, tuple):
        g_asymms.update(symbs)
    else:
        g_asymms.add(symbs)
    return symbs


def sym(expr):
    return Integer(1)/Integer(2)*(expr + t(expr))


def asym(expr):
    return Integer(1)/Integer(2)*(expr - t(expr))


class t(Function):
    ''' The transposition, with special rules
        t(A+B) = t(A) + t(B) and t(AB) = t(B)t(A) '''
    is_commutative = False
    
    def __new__(cls, arg):
        # if hasattr(arg, 'is_constant') and arg.is_constant():
        if hasattr(arg, 'is_number') and arg.is_number:
            return arg
        elif arg in g_scalars:
            return arg
        elif isinstance(arg, int) or isinstance(arg, float):
            return arg
        if arg.is_Add:
            f = []
            for A in arg.args:
                f.append(t(A))
            return Add(*f)
        elif arg.is_Mul:
            L = len(arg.args)
            return Mul(*[t(arg.args[L-i-1]) for i in range(L)])
        elif arg.func == t:
            return arg.args[0]
        elif arg.func == Id:
            return Id()
        elif arg.func == inv:
            return inv(t(Function.__new__(cls, arg.args[0])))
        elif arg.func == Pow:
            return Pow(t(arg.args[0]), arg.args[1])
        elif arg in g_symms:
            return arg
        elif arg in g_asymms:
            return -arg
        else:
            return Function.__new__(cls, arg)


class inv(Function):
    ''' The transposition, with special rules
        t(A+B) = t(A) + t(B) and t(AB) = t(B)t(A) '''
    is_commutative = False

    def __new__(cls, arg):
        if arg.is_Mul:
            L = len(arg.args)
            return Mul(*[inv(arg.args[L-i-1]) for i in range(L)])
        elif arg.func == inv:
            return arg.args[0]
        elif arg.func == Pow:
            return Pow(inv(arg.args[0]), arg.args[1])
        elif arg.func == Id:
            return Id()
        else:
            return Function.__new__(cls, arg)


class Id(Function):
    # the identity to any dimension. This allows us to avoid declare dimension
    def __new__(cls, arg=None):
        return(Function.__new__(cls))

    def __mul__(cls, other):
        return other


# d = Function("d", commutative=False)
# inv = Function("inv", commutative=False)


class d(Function):
    """Unevaluated matrix differential (e.g. dX, where X is a matrix)
    """

    def __new__(cls, mat):
        return Function.__new__(cls, mat)

    @property
    def arg(self):
        return self.args[0]

    @property
    def shape(self):
        return (self.arg.rows, self.arg.cols)


class trace(Function):
    def __new__(cls, mat):
        if mat.is_Add:
            return Add(*[Function.__new__(cls, A) for A in mat.args])
        elif len(mat.args) > 0 and mat.func == trace:
            return Function.__new__(cls, mat.args[0])

        elif mat.is_Mul:
            scalars = []
            non_scalars = []
            for a in mat.args:
                if a in g_scalars:
                    scalars.append(a)
                elif a.func == trace:
                    non_scalars.append(a.args[0])
                else:
                    non_scalars.append(a)
            if not non_scalars:
                return Mul(*(scalars + Function.__new__(cls, Id(None))))
            else:
                return Mul(*(scalars + [Function.__new__(
                    cls, Mul(*non_scalars))]))
        return Function.__new__(cls, mat)

    @property
    def arg(self):
        return self.args[0]


def d_pow(e, s, ds):
    # positive integer exponent only
    try:
        if e.args[1].is_Integer:
            pw = int(e.args[1])
    except Exception:
        raise(ValueError("Can only handle positive integer power derivative"))
    if pw <= 0:
        # derivative of inv(Pow(x))
        return -e*DDR(Pow(e.args[0], -pw), s, ds)*e
    # raise(ValueError("Can only handle positive integer power derivative"))
    add_list = []
    des = DDR(e.args[0], s, ds)
    if pw == 1:
        return des
    for i in range(pw):
        if i == 0:
            add_list.append(Mul(des, Pow(e.args[0], pw-1)))
        elif i == pw - 1:
            add_list.append(Mul(Pow(e.args[0], pw-1), des))
        else:
            add_list.append(
                Mul(Pow(e.args[0], i), des, Pow(e.args[0], pw-i-1)))
    return Add(*add_list)

    
MATRIX_DIFF_RULES = {
    # e =expression, s = a list of symbols respsect to which
    # we want to differentiate

    Symbol: lambda e, s, ds: ds if (e.has(s)) else 0,
    Add: lambda e, s, ds: Add(*[DDR(arg, s, ds) for arg in e.args]),
    Mul: lambda e, s, ds: Mul(
        DDR(e.args[0], s, ds), Mul(*e.args[1:])) +\
    Mul(e.args[0], DDR(Mul(*e.args[1:]), s, ds)),
    t: lambda e, s, ds: t(DDR(e.args[0], s, ds)),
    Pow: d_pow,
    trace: lambda e, s, ds: trace(DDR(e.args[0], s, ds)),
    Id: lambda e, s, ds: 0,
    inv: lambda e, s, ds: - e * DDR(e.args[0], s, ds) * e
}


def DDR(expr, symbols, dsyms):
    """ Directional derivative in direction dsyms
    """
    if expr.__class__ in MATRIX_DIFF_RULES:
        ret = MATRIX_DIFF_RULES[expr.__class__](expr, symbols, dsyms)
        return mat_spfy(ret)
    elif expr.is_constant():
        return 0
    else:
        raise TypeError(
            "Don't know how to differentiate class %s", expr.__class__)


def noncommutative(args):
    for x in args:
        if not check_commutative(x):
            return True
    return False
        

def add_list_head(add_list, mul_list):
    f = Mul(*mul_list)
    terms = Mul(add_list[0], f)
    for w in add_list[1:]:
        terms = Add(terms, Mul(w, f))
    return terms


def add_list_tail(add_list, mul_list):
    f = Mul(*mul_list)
    terms = Mul(f, add_list[0])
    for w in add_list[1:]:
        terms = Add(terms, Mul(f, w))
    return terms


def add_list_middle(add_list, head_mul_list, tail_mul_list):
    mhead = Mul(*head_mul_list)
    mtail = Mul(*tail_mul_list)
    terms = Mul(mhead, add_list[0], mtail)
    for w in add_list[1:]:
        terms = Add(terms, Mul(mhead, w, mtail))
    return terms


def to_monomials(expr):
    try:
        e = expr.expand()
    except Exception:
        if hasattr(expr, 'copy'):
            e = expr.copy()
        else:
            e = expr
    
    while True:
        has_change = False
        for x in preorder_traversal(e):
            # print("Doing %s" % str(x))
            if hasattr(x, 'is_Mul') and x.is_Mul:
                # print("here")
                for i, y in enumerate(x.args):
                    if hasattr(y, 'is_Add') and y.is_Add and\
                       noncommutative(y.args):
                        if i == 0:
                            terms = add_list_head(y.args, x.args[1:])
                        elif i == len(y.args)-1:
                            terms = add_list_tail(y.args, x.args[:-1])
                        else:
                            terms = add_list_middle(
                                y.args, x.args[:i], x.args[i+1:])
                        enew = e.xreplace({x: terms})
                        if enew != e:
                            has_change = True
                            e = enew
                        break
            if has_change:
                break
        if not has_change:
            break
    return e
        

def replace_comp_stiefel(expr):
    e = expr.copy()
    for x in preorder_traversal(e):
        # print("Doing %s" % str(x))
        if hasattr(x, 'is_Mul') and x.is_Mul:
            # print("here")
            new_x_args = x.args.copy()
            i = 0
            while i < len(new_x_args):
                y = new_x_args[i]
                if i < len(new_x_args)-1:
                    if new_x_args[i+1].func == t and\
                       y in g_cstiefels and\
                       new_x_args[i+1].args[0] == y:
                        aterm = Id() - g_cstiefels[y] *\
                            t(g_cstiefels[y])
                        new_x_args = new_x_args[:i] + [aterm] +\
                            new_x_args[i+2:]
                i += 1
            e = e.xreplace({x: Mul(new_x_args)})
    e = to_monomials(e)
    return mat_spfy(e)


def check_commutative(arg):
    if isinstance(arg, int) or isinstance(arg, float):
        return True
    if hasattr(arg, 'is_commutative') and arg.is_commutative:
        return True
    if arg.func == Id:
        return True
    return False


def extract_commutative(args):
    comm = []
    noncomm = []
    for i in range(len(args)):
        if args[i].func == Id:
            continue
        elif check_commutative(args[i]):
            comm.append(args[i])
        else:
            noncomm.append(args[i])
    return comm, noncomm    
    

def mat_spfy(expr):
    e = to_monomials(expr)
    while True:
        has_change = False
        for x in preorder_traversal(e):
            if hasattr(x, 'is_Mul') and x.is_Mul:
                comm, noncom = extract_commutative(x.args)
                if len(comm) + len(noncom) < len(x.args):
                    has_change = True
                i = 0
                while i < len(noncom):
                    y = noncom[i]
                    if i + 1 < len(noncom):
                        new_ncom = remove_inv_pair(
                            noncom, i, y, noncom[i+1])
                        if new_ncom is not None:
                            noncom = new_ncom
                            has_change = True
                        if i+1 < len(noncom):
                            new_ncom = remove_stiefel_pair(
                                noncom, i, y, noncom[i+1])
                            if new_ncom is not None:
                                noncom = new_ncom
                                has_change = True
                        if not has_change:
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


def simplify_stiefel_tangent(expr, Y, tangentvars):
    """simplfy for tangent
    rule is move t(Y) eta to -t(eta) Y
    """
    # e = mat_spfy(expr)
    e = to_monomials(expr)
    while True:
        has_change = False
        for x in preorder_traversal(e):
            if hasattr(x, 'is_Mul') and x.is_Mul:
                comm, noncom = extract_commutative(x.args)
                if len(comm) + len(noncom) < len(x.args):
                    has_change = True
                i = 0
                while i < len(noncom):
                    y = noncom[i]
                    if i + 1 < len(noncom):
                        new_ncom = remove_inv_pair(
                            noncom, i, y, noncom[i+1])
                        if new_ncom is not None:
                            noncom = new_ncom
                            has_change = True
                        new_ncom = remove_stiefel_pair(
                            noncom, i, y, noncom[i+1])
                        if new_ncom is not None:
                            noncom = new_ncom
                            has_change = True
                        if i + 1 < len(noncom):
                            new_ncom, sig = move_stiefel_tangent(
                                noncom, i, y, noncom[i+1], Y, tangentvars)
                            if new_ncom is not None:
                                noncom = new_ncom
                                comm.append(sig)
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


def simplify_pd_tangent(expr, Y, tangentvars):
    """simplfy for tangent
    rule is move t(Y) eta to -t(eta) Y
    """
    e = to_monomials(expr)
    while True:
        has_change = False
        for x in preorder_traversal(e):
            if x.func == t:
                if x.args[0] in tangentvars:
                    e = e.xreplace({x: x.args[0]})
                    has_change = True
                    break
                elif x.args[0].func == inv:
                    e = e.xreplace({x: inv(t(x.args[0].args[0]))})
                    has_change = True
                    break
                elif x.args[0].func == t:
                    e = e.xreplace({x: x.args[0].args[0]})
        if not has_change:
            break
    e = mat_spfy(e)
    return e


def move_stiefel_tangent(
        xargs, i, y, y_next, point, tangentvars):
    if y.func == t:
        if y.args[0] == point and y_next in tangentvars:
            return list(xargs[:i]) + [t(y_next), point] + list(xargs[i+2:]),\
                Integer(-1)
    return None, None


def is_inv_pair(y, y_next):
    if y.func == inv and y.args[0] == y_next:
        return True
    elif y_next.func == inv and y_next.args[0] == y:
        return True
    return False


def remove_inv_pair(xargs, i, y, y_next):
    if is_inv_pair(y, y_next):
        if i == 0:
            return xargs[2:]
        elif i == len(xargs)-2:
            return xargs[:-2]
        else:
            return list(xargs[:i]) + list(xargs[i+2:])
    return None


def remove_inv_pair_(xargs, i, y, y_next):
    if is_inv_pair(y, y_next):
        if i == 0:
            return Mul(xargs[2:])
        elif i == len(xargs)-2:
            return Mul(xargs[:-2])
        else:
            return Mul(*(list(xargs[:i]) + list(xargs[i+2:])))
    return None
    

class MatStrPrinter(StrPrinter):
    ''' Nice printing for console mode : X¯¹, X', ∂X '''

    def _print_inv(self, expr):
        if expr.args[0].is_Symbol:
            return self._print(expr.args[0]) + '¯¹'
        else:
            return '(' + self._print(expr.args[0]) + ')¯¹'

    def _print_t(self, expr):
        return self._print(expr.args[0]) + "'"

    def _print_d(self, expr):
        if expr.args[0].is_Symbol:
            return '∂'+self._print(expr.args[0])
        else:
            return '∂('+self._print(expr.args[0])+')'


def MatPrint(m):
    mem = Basic.__str__
    Basic.__str__ = lambda self: MatStrPrinter().doprint(self)
    print(str(m).replace('*', ''))
    Basic.__str__ = mem


# Latex mode

class MatLatPrinter(LatexPrinter):
    ''' Printing instructions for latex : X^{-1},  X^T, \\partial X '''

    def _print_invs(self, expr):
        if expr.args[0].is_Symbol:
            return self._print(expr.args[0]) + '^{-1}'
        else:
            return '(' + self._print(expr.args[0]) + ')^{-1}'

    def _print_inv(self, expr, exp=None):
        if exp is None:
            exp = '1'
        if expr.args[0].is_Symbol:
            return self._print(expr.args[0]) + '^{-%s}' % exp
        else:
            return '(' + self._print(expr.args[0]) + ')^{-%s}' % exp
        
    def _print_t(self, expr):
        return self._print(expr.args[0])+'^T'

    def _print_trace(self, expr):
        return 'Tr(%s)' % self._print(expr.args[0])

    def _print_d(self, expr):
        if expr.args[0].is_Symbol:
            return r'\partial '+self._print(expr.args[0])
        else:
            return r'\partial ('+self._print(expr.args[0])+')'


def mat_latex(expr, profile=None, **kargs):
    if profile is not None:
        profile.update(kargs)
    else:
        profile = kargs
    return MatLatPrinter(profile).doprint(expr)


def trace_arrange(ii, args):
    if len(args) == 1:
        return Id(None)
    if ii == 0:
        return Mul(*args[1:])
    if ii == len(args)-1:
        return Mul(*args[:-1])
    return Mul(*(args[ii+1:] + args[:ii]))
            

def extract_inside_trace(expr, s):
    # assume mul only
    if hasattr(expr, 'is_Symbol') and expr.is_Symbol:
        if expr == s:
            return Id(None)
        return None

    for jj in range(len(expr.args)):
        if expr.args[jj].func == t and\
           expr.args[jj].args[0] == s:
            newterm = trace_arrange(jj, expr.args)
            return newterm
        elif expr.args[jj] == s:
            newterm = t(trace_arrange(jj, expr.args))
            return newterm
        elif expr.args[jj] == trace:
            newterm = extract_inside_trace(expr.args[jj], s)
            return Mul(*(expr.args[:jj] + [newterm] + expr.args[jj+1:]))
                                    
    return None


def xtrace(expr, ds):
    # if expr is a trace, sum of traces, or scalar times a trace, we can do it
    # Only works if s is not inside pow. but since ds
    # is a differential we expect that. For each monomial we expect only one
    # factor with s
    if expr == ds:
        return Id(ds)
    elif expr.func == t and expr.args[0] == ds:
        return Id(ds)
    if expr.is_Add:
        rr = [xtrace(uu, ds) for uu in expr.args]
        rr = [aa for aa in rr if aa is not None]
        return Add(*rr)
        # return Add(*[extract_trace(uu, ds) for uu in expr.args])
    if expr.func == trace:
        return extract_inside_trace(expr.args[0], ds)
    if expr.is_Mul:
        # assume whatever before and after trace are scalars
        ii = 0
        found = False
        while not found and ii < len(expr.args):
            if expr.args[ii].func == trace:
                found = True
                extracted_s = extract_inside_trace(expr.args[ii].args[0], ds)
                if extracted_s is not None:
                    return Mul(
                        *(list(expr.args[:ii]) +
                          [extract_inside_trace(expr.args[ii].args[0], ds)] +
                          list(expr.args[ii+1:])))
                    
            ii += 1
        if not found:
            raise(ValueError("expr has non trace term %s" % str(expr)))
    else:
        raise RuntimeError(
            "Don't know how to extract %s from trace %s !" % (ds, expr))


def xtrace_sym(expr, ds):
    return sym(xtrace(expr, ds))
    

def xtrace_asym(expr, ds):
    return asym(xtrace(expr, ds))


def remove_stiefel_pair(xargs, i, y, y_next):
    if (y_next in g_stiefels) and y == t(y_next):
        if i == 0:
            return xargs[2:]
        elif i == len(xargs)-2:
            return xargs[:-2]
        else:
            return list(xargs[:i]) + list(xargs[i+2:])
    if (y_next in g_cstiefels) and y == t(y_next):
        if i == 0:
            return xargs[2:]
        elif i == len(xargs)-2:
            return xargs[:-2]
        else:
            return list(xargs[:i]) + list(xargs[i+2:])
        
    elif (y.func == t and y_next in g_cstiefels
          and y.args[0] == g_cstiefels[y_next]):
        return list(xargs[:i]) + [Integer(0)] + list(xargs[i+2:])
    elif (y.func == t and y.args[0] in g_cstiefels and
          y_next == g_cstiefels[y.args[0]]):
        return list(xargs[:i]) + [Integer(0)] + list(xargs[i+2:])
    return None


def simplify_transpose(expr):
    e = expr.copy()
    for x in preorder_traversal(expr):
        if x.func == t and x.args[0] == t:
            e.xreplace({x, x.args[0].args[0]})
    return e

        
def latex_map(astr, mdict):
    for s in mdict:
        astr = astr.replace(s, mdict[s])
    return astr

    
def simplified_trace(expr):
    e = to_monomials(expr)
    newargs = []
    if e.is_Add:
        for f in e.args:
            if f.is_Mul:
                newprod = []
                for i in range(len(f.args)):
                    if f.args[i].func == trace:
                        newprod.append(f.args[i].args[0])
                    else:
                        newprod.append(f.args[i])
                newargs.append(Mul(*(newprod)))
            else:
                newargs.append(f)
        return trace(Add(*(newargs)))
    else:
        return trace(e)

    
if __name__ == '__main__':
    X, S = matrices("X S")
    u = t(X*X)
    for x in preorder_traversal(u):
        print(x.func)
        
    H = X*inv(t(X)*inv(S)*X)*t(X)*inv(S)
    print(mat_latex(DDR(H, X)))
