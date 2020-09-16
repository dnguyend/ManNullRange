# ManNullRange
This project provides the python implementation of the paper <em> Operator-Valued Formulas for Riemannian Gradient and Hessian and families of tractable metrics in optimization and machine learning</em>.

The main idea of the paper is given a manifold where its tangent space is a nullspace of a family of full rank operators J (e.g. the Jacobian of the constraint equations) parametrized by points on the manifold, we can introduce a Riemannian metrics as a operator-valued function on the manifold. With a technical requirement on the metric, this setup allows us to compute the Riemannian gradient, the Levi-Civita connection (which gives the geodesics equation), the Riemannian Hessian by operator-valued formulas. The formulas could be evaluated symbolically in many cases, which simplifies to give known or new formulas on manifolds arising in optimization, machine learning and statistical problems, and could be evaluted numerically when there is no algebraic simplification.

We apply this to several families of metrics on manifolds arising in practice, as explained in the paper:

1. We provide a new family of metrics, connecting both the embedded metric and the canonical metric on *Stiefel* manifolds ([RealStiefel.py](https://github.com/dnguyend/ManNullRange/blob/master/manifolds/RealStiefel.py) and [ComplexStiefel.py](https://github.com/dnguyend/ManNullRange/blob/master/manifolds/ComplexStiefel.py)), with explicit formulas for gradient and Levi-Civita connection and Hessian. We show it has closed-form geodesics, generalizing both geodesics formulas in [1]. The symbolic calculation is presented in [](). Test and an optimization program are in []():

2. We recover the fairly complex formulas for the Riemannian connection for the manifold of *positive-definite* matrix via symbolic calculation. The symbolic calculation is in []().

3. We provide a new family of metrics on the manifold of *positive-semidefinite* matrices of fixed rank ([RealPositivesemidefinite.py](https://github.com/dnguyend/ManNullRange/blob/master/manifolds/RealPositiveSemidefinite.py) and [ComplexPositivesemidefinite.py](https://github.com/dnguyend/ManNullRange/blob/master/manifolds/ComplexPositiveSemidefinite.py)), with the affine behavior in the positive-definite factor. This improves [2]. The symbolic calculation is in []().

4. We provide a new family of metrics on the manifold of *fixed-rank* matrices ([RealFixedRank.py]() and [ComplexFixedRank.py]()). The symbolic calculation is in [](). This class of metrics extends the metric in [3].

5. We provide a new family of metrics for *flag manifold*s ([RealFlag.py](https://github.com/dnguyend/ManNullRange/blob/master/manifolds/RealFlag.py) and [ComplexFlag.py](https://github.com/dnguyend/ManNullRange/blob/master/manifolds/RealFlag.py)), which extends the only metric with explicit connection in the literature, the canonical metric in [4-5]. We provide the full implementation, with first and second-order methods.

Features:
1. Both first and second-order methods are implemented.
2. We implement both real and complex versions.
3. We allow metric parameteres.
4. We provide geodesics for all cases, except for the flag manifolds (Our family of metric for flag manifolds is quite large, at present only a subfamily, corresponding to the Stiefel metrics above has closed-form geodesics.)

[1] A. Edelman, T. A. Arias, and S. T. Smith, The geometry of algorithms with orthogonality
constraints, SIAM J. Matrix Anal. Appl., 20 (1999), pp. 303–353.

[2] S. Bonnabel and R. Sepulchre, Riemannian metric and geometric mean for positive semi-
definite matrices of fixed rank, SIAM Journal on Matrix Analysis and Applications, 31
(2010), pp. 1055–1070.

[3] B. Mishra, G. Meyer, S. Bonnabel, and R. Sepulchre, Fixed-rank matrix factorizations
and riemannian low-rank optimization, Computational Statistics, 29 (2014), pp. 591, 621.

[4] Y. Nishimori, S. Akaho, and M. D. Plumbley, Riemannian optimization method on the
flag manifold for independent subspace analysis, in Independent Component Analysis and
Blind Signal Separation, J. Rosca, D. Erdogmus, J. C. Príncipe, and S. Haykin, eds., Berlin,
Heidelberg, 2006, Springer Berlin Heidelberg, pp. 295–302.

[5] K. Ye, K. S.-W. Wong, and L.-H. Lim, Optimization on flag manifolds, 2019, [https://arxiv.
org/abs/arXiv:1907.00949](https://arxiv.
org/abs/arXiv:1907.00949).


