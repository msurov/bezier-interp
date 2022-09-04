import numpy as np
from scipy.special import binom, factorial
from numpy.polynomial.polynomial import polymul, polypow, polyval
from scipy.integrate import ode


def bezier2poly(coefs):
    R'''
        Parameters
        ----------
        `coefs` numpy.array of dimensions NxM
        the coefficients define a Bezier curve of order N-1
        in M-dimensional space \
        `returns` polynomial coefficients
    '''
    coefs = np.array(coefs)
    ncoefs,dim = np.shape(coefs)
    n = ncoefs - 1
    _t = [0., 1.]
    _1_minus_t = [1., -1.]
    res_poly = np.zeros((ncoefs, dim))

    for i in range(n + 1):
        p = binom(n, i) * polymul(polypow(_t, i), polypow(_1_minus_t, n - i))
        res_poly += np.outer(p, coefs[i,:])

    return res_poly


def eval_basis_scalar(n):
    R'''
        Evaluate scalar product 
            $$A_ij = \int_0^1 b_{in}(t) * b_{jn}(t) dt$$
        of all basis functions of degree `n`
        The result is the matrix of dimension (n+1)x(n+1)
    '''
    C = binom(n, np.arange(0, n + 1, dtype=int))
    A = np.zeros((n + 1, n + 1))

    for i in range(n + 1):
        _i_to_n = np.arange(i, n + 1, dtype=int)
        row = C[i] * C[i:] * factorial(i + _i_to_n) * \
            factorial(2*n - i - _i_to_n) / factorial(2*n + 1)
        A[i,i:] = row
        A[i+1:,i] = row[1:]

    return A


def get_deriv_mat(n, nder):
    R'''
        k-th derivative of Bezier is expressed as
        $$p^{(k)}(t) = \sum_i b_{i,n-k}(t) D_{n,k} c_i$$
        the function returns $D_k$ of dimension (n-k+1)x(n+1)
    '''
    J = np.zeros((n + 1 - nder, n + 1))
    d = binom(nder, np.arange(0, nder + 1))
    d[-2::-2] *= -1
    d *= n * (n - 1)
    for i in range(n + 1 - nder):
        J[i,i:i+nder+1] = d
    return J


def eval_basis(n : int, t : float):
    R'''
        Evaluate basis functions of Bezier
            b_{i,n}(t)
        at point `t`

        Parameters
        ----------
        `n` is the degree of the Bezier
        `t` is the argument in range [0,1]
        returns one-dimensional ndarray of length n+1 
        with element [b_0, b_1, ..., b_n]
    '''
    i = np.arange(0, n + 1, dtype=int)
    C = np.reshape(binom(n, i), (1, -1))
    t = np.reshape(t, (-1,1))
    return np.power(t, i) * np.power(1 - t, n - i) * C


def eval_bezier(ctrls : np.ndarray, t : np.ndarray):
    assert np.all(t <= 1.) and np.all(t >= 0.)
    nctrls,_ = ctrls.shape
    B = eval_basis(nctrls - 1, t)
    return B @ ctrls


def eval_bezier_deriv(ctrls : np.ndarray, t : np.ndarray):
    assert np.all(t <= 1.) and np.all(t >= 0.)
    nctrls,_ = ctrls.shape
    DB = eval_basis(nctrls - 2, t)
    return (nctrls - 1) * DB @ np.diff(ctrls, axis=0)


def eval_bezier_length(ctrls : np.ndarray, t : np.ndarray):
    assert np.all(t <= 1.) and np.all(t >= 0.)
    nctrls,_ = ctrls.shape
    DC = (nctrls - 1) * np.diff(ctrls, axis=0)

    def rhs(v, _):
        DB = eval_basis(nctrls - 2, v)
        return np.linalg.norm(DB @ DC)

    solver = ode(rhs)
    solver.set_integrator('dopri5')
    solver.set_initial_value(0., 0.)
    nt = len(t)
    vals = np.zeros(nt)

    for i in range(nt):
        if t[i] == 0.:
            continue
        solver.integrate(t[i])
        if not solver.successful:
            raise Exception('integration error')
        assert np.allclose(solver.t, t[i])
        vals[i] = solver.y
    
    return vals


def minimize_trace(A, B, W):
    R'''
        minimize trace(X' W X)
           X
        s.t.
           A X = B

        Parameters
        ----------
        `A` np.ndarray of size NxM, N<=M \\
        `B` np.ndarray of size NxK \\
        `W` np.ndarray of size MxM \\
        returns `X` np.ndarray of size MxK 
    '''
    eps = 1e-5
    n,m = A.shape
    _,k = B.shape
    assert m >= n
    assert W.shape == (m,m)
    assert B.shape == (n,k)

    U,s,Vt = np.linalg.svd(A)

    # 1. Find a right annihilator A_perp of A
    d = m - n + np.sum(s <= eps)
    J = np.flip(np.eye(m, d))
    A_perp = Vt.T @ J

    # 2. Find a particular solution X1 of the linear system A X = B
    s_inv = [1/si if si > eps else 0.0 for si in s]
    S_inv = np.zeros((m, n))
    np.fill_diagonal(S_inv, s_inv)
    Z = S_inv @ U.T @ B
    X1 = Vt.T @ Z

    # 3. The general solution of the linear system 
    # is parametrized by Y as X = X1 + A_perp @ Y

    # 4. Will find Y which minimizes X' W X
    Y = np.linalg.solve(A_perp.T @ W @ A_perp, -A_perp.T @ W @ X1)
    X = X1 + A_perp @ Y
    return X


def interpolate(pts, order, derivs_left=[], derivs_right=[]):
    R'''
        Interpolate list of points `pts` by a Bezier curve

        Parameters
        ----------
        `pts` ndarray of size NxD, the points to be interpolated \
        `order` degree of the Bezier curve \
        `derivs_left` left boundary conditions \
        `derivs_right` right boundary conditions in the form \
        [(derivative_order, derivative_value), ...]
        where integer value `derivative_order` is the order of derivative,
        and ndarray of size D `derivative_value` is the value of Bezier's 
        derivative at the boundary \
        `returns` ndarray of dimension (N-1)x(order+1)x(D);
        array of control points for each chank
    '''
    npts,dim = pts.shape
    ncoefs = order + 1

    nequations = (npts - 2) * ncoefs + len(derivs_left) + len(derivs_right) + 2
    nunknowns = (npts - 1) * ncoefs
    assert nequations <= nunknowns, 'Too many conditions, remove some of boundary conditions'

    # TODO: use sparse matrix

    # continuity constraints
    A = np.zeros((nequations, nunknowns))
    B = np.zeros((nequations, dim))

    A11 = np.zeros((ncoefs, ncoefs))
    A10 = np.zeros((ncoefs, ncoefs))
    for k in range(0, order):
        for j in range(0, k + 1):
            a = binom(k, j) * (-1)**(k-j)
            A11[k,j] = a
            A10[k,order - k + j] = -a
    A11[-1,-1] = 1.

    i = 0
    j = 2
    while j < npts:
        A[i : i + ncoefs, i : i + ncoefs] = A10
        A[i : i + ncoefs, i + ncoefs : i + 2 * ncoefs] = A11
        B[i + ncoefs - 1,:] = pts[j,:]
        i += ncoefs
        j += 1

    A[i, 0] = 1.
    B[i, :] = pts[0,:]
    i += 1
    A[i, ncoefs-1] = 1.
    B[i, :] = pts[1,:]
    i += 1

    # boundary conditions
    for condition in derivs_left:
        nder, value = condition
        assert nder > 0 and nder <= order
        assert np.shape(value) == (dim,)
        d = binom(nder, range(0, nder + 1))
        d[-2::-2] *= -1
        A[i, 0:len(d)] = factorial(order) / factorial(order - nder) * d
        B[i, :] = value
        i += 1

    for condition in derivs_right:
        nder, value = condition
        assert nder > 0 and nder <= order
        assert np.shape(value) == (dim,)
        d = np.zeros(nder + 1)
        d = binom(nder, range(0, nder + 1))
        d[-2::-2] *= -1
        A[i, -len(d):] = factorial(order) / factorial(order - nder) * d
        B[i, :] = value
        i += 1

    if np.linalg.matrix_rank(A) < nunknowns:
        # minimize 2nd derivative
        Q = np.zeros((ncoefs * (npts - 1), ncoefs * (npts - 1)))
        W = eval_basis_scalar(order - 2)
        D = get_deriv_mat(order, 2)
        Qi = D.T @ W @ D
        for i in range(npts - 1):
            Q[i*ncoefs:(i+1)*ncoefs,i*ncoefs:(i+1)*ncoefs] = Qi
        X = minimize_trace(A, B, Q)
    else:
        X = np.linalg.solve(A, B)

    X = np.reshape(X, (npts-1, order + 1, dim))
    return X

