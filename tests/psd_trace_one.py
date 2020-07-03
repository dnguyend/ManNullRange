def calc_psd_trace_1():
    # manifold is on pair (Y, P)
    Y, P, a_P, a_YP, eta_Y, eta_P = matrices(
        'Y P a_P a_YP eta_Y eta_P')
    global symm_mat, stiefel_symbols, anti_symm_mat, comp_stfls, scalar_list

    anti_symm_mat = set((a_P,))
    comp_stfls = dict()

    al0, al1, bt, a_TR = symbols('al0 al1 bt a_TR', commutative=True)
    symm_mat = [P, al0, al1, bt, a_TR]
    scalar_list = set((al0, al1, bt, a_TR))
    stiefel_symbols = [Y]

    def g(Y, P, omg_Y, omg_P):
        return al0*omg_Y+(al1-al0)*Y*t(Y)*omg_Y, bt*inv(P)*omg_P*inv(P)

    def ginv(Y, P, omg_Y, omg_P):
        return 1/al0*omg_Y+(1/al1-1/al0)*Y*t(Y)*omg_Y, 1/bt*P*omg_P*P

    # check that ginv \circ g is id
    e1, e2 = ginv(Y, P, *(g(Y, P, eta_Y, eta_P)))
    e1 = matrix_simplify(e1)
    e2 = matrix_simplify(e2)
    print(e1, e2)

    def ambient_inner(Y, P, omg_Y, omg_P, xi_Y, xi_P):
        return matrix_simplify(
            trace(
                matrix_simplify(
                    (al0*omg_Y+(al1-al0)*Y*t(Y)*omg_Y) * t(xi_Y))) +
            trace(matrix_simplify(
                bt*inv(P)*omg_P*inv(P)*t(xi_P))))

    def EJ_inner(Y, P, a_P, a_YP, a_TR, b_P, b_YP, b_TR):
        return trace(
            matrix_simplify(-a_P * b_P) + matrix_simplify(
                a_YP*t(b_YP))) + simplified_trace(a_TR * t(b_TR))

    qat = matrices('qat')
    anti_symm_mat = anti_symm_mat.union((qat,))
    ipr = ambient_inner(Y, P, eta_Y, eta_P, Y * qat,  P*qat - qat*P)
    dqat = matrix_simplify(extract_trace(ipr, qat))
    print(dqat)

    def J(Y, P, omg_Y, omg_P):
        J_P = matrix_simplify(omg_P - t(omg_P))
        J_YP = matrix_simplify(
            al1*t(Y)*omg_Y + bt*omg_P*inv(P) - bt*inv(P)*omg_P)
        J_TR = matrix_simplify(trace(omg_P))

        return J_P, J_YP, J_TR
    
    def JT(Y, P, a_P, a_YP, a_TR):
        dY, dP = matrices('dY dP')
        ipt = matrix_simplify(
            EJ_inner(Y, P, *J(Y, P, dY, dP), a_P, a_YP, a_TR))
        jty = matrix_simplify(extract_trace(ipt, dY))
        jtp = matrix_simplify(extract_trace(ipt, dP))
        return jty, jtp

    JTa_Y, JTa_P = JT(Y, P, a_P, a_YP, a_TR)
    ginvJT = ginv(Y, P, JTa_Y, JTa_P)
    ginvJT_Y = matrix_simplify(ginvJT[0])
    ginvJT_P = matrix_simplify(ginvJT[1])
    print(ginvJT_Y)
    print(ginvJT_P)

    Jginv_P, Jginv_YP, Jginv_TR = J(Y, P, *ginv(Y, P, eta_Y, eta_P))
    print(Jginv_P)
    print(Jginv_YP)
    print(Jginv_TR)
    
    b_P, b_YP, b_TR = J(Y, P, *ginvJT)
    b_P = matrix_simplify(b_P)
    b_YP = matrix_simplify(b_YP)
    b_TR = matrix_simplify(b_TR)
    
    # print(b_P)
    # print(b_YP)

    def sym(x):
        return matrix_simplify(
            Integer(1)/Integer(2)*(x + t(x)))
    aYPev = sym(a_YP)
    
    exp1 = bt * (Integer(1)/Integer(2)*b_P - P*aYPev + aYPev*P)
    exp1 = matrix_simplify(exp1)
    print(exp1)
    """
    exp1 = bt * (Integer(1)/Integer(2)*(b_P*inv(P) - inv(P)* b_P)
                 - P*aYPev*inv(P) + 2*aYPev - inv(P)* aYPev*P)
    """
    exp1 = al1*aYPev + bt/2*(b_P*inv(P)-inv(P)*b_P) - sym(b_YP)
    print(matrix_simplify(exp1))
    aYPodd = matrix_simplify(a_YP - aYPev)
    exp2 = (al1-2*bt)*aYPodd + bt*P*aYPodd*inv(P) + bt*inv(P)*aYPodd*P -\
        Integer(1)/Integer(2)*(b_YP-t(b_YP))
    print(exp2)
        
    def even_solve(Y, P, b_P, b_YP):
        return Integer(1)/al1*sym(b_YP) +\
            bt/(Integer(2)*al1)*(inv(P)*b_P - b_P*inv(P))

    print(matrix_simplify(even_solve(Y, P, b_P, b_YP)))
    xi_Y, xi_P, phi_Y, phi_P = matrices('xi_Y xi_P phi_Y phi_P')
    gYPeta = g(Y, P, eta_Y, eta_P)
    Dgxieta_Y = dderivatives(gYPeta[0], Y, xi_Y)
    Dgxieta_P = dderivatives(gYPeta[1], P, xi_P)

    gYPxi = g(Y, P, xi_Y, xi_P)
    Dgetaxi_Y = dderivatives(gYPxi[0], Y, eta_Y)
    Dgetaxi_P = dderivatives(gYPxi[1], P, eta_P)

    Dgxiphi_Y = dderivatives(gYPeta[0], Y, phi_Y)
    Dgxiphi_P = dderivatives(gYPeta[1], P, phi_P)

    tr3 = matrix_simplify(
        ambient_inner(Y, P, Dgxiphi_Y, Dgxiphi_P, eta_Y, eta_P))
    xcross_Y = extract_trace(tr3, phi_Y)
    xcross_P = extract_trace(tr3, phi_P)
        
    K_Y = (Integer(1)/Integer(2))*(Dgxieta_Y + Dgetaxi_Y - xcross_Y)
    K_P = (Integer(1)/Integer(2))*(Dgxieta_P + Dgetaxi_P - xcross_P)
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
        expr_P, expr_YP, expr_TR = J(Y, P, eta_Y, eta_P)
        der_P = dderivatives(expr_P, Y, xi_Y)+dderivatives(expr_P, P, xi_P)
        der_YP = dderivatives(expr_YP, Y, xi_Y)+dderivatives(expr_YP, P, xi_P)
        der_TR = dderivatives(expr_TR, Y, xi_Y)+dderivatives(expr_TR, P, xi_P)
        return matrix_simplify(der_P), matrix_simplify(der_YP),\
            matrix_simplify(der_TR)

    dj1 = DJ(Y, P, xi_Y, xi_P, eta_Y, eta_P)
    print(dj1)
