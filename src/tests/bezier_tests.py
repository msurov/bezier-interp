import numpy as np
from bezier_interp.bezier import \
    interpolate, eval_basis, eval_bezier, \
    eval_basis_scalar, bezier2poly, get_deriv_mat, \
    eval_bezier_deriv, eval_bezier_length
from numpy.polynomial.polynomial import polymul, polypow, polyval, polyder


def test_bezier_deriv():
    t = np.linspace(0, 1, 100)

    # gen bezier of degree n
    n = 3
    c = np.random.normal(size=(n+1,3))

    # evaluate p''
    poly = bezier2poly(c)
    D2_poly = polyder(poly, 2)
    val1 = polyval(t, D2_poly).T

    # evaluate p''
    B = eval_basis(n-2, t)
    D2 = get_deriv_mat(n, 2)
    val2 = B @ D2 @ c

    # compare
    assert np.allclose(val1, val2)


def test_eval_basis_scalar():
    n = 3
    c = np.random.normal(size=(n+1,3))

    poly = bezier2poly(c)
    D2_poly = polyder(poly, 2)
    t = np.linspace(0, 1, 1000)
    D2_p = polyval(t, D2_poly).T
    norm_D2_p = np.linalg.norm(D2_p, axis=1)
    val1 = np.trapz(norm_D2_p**2, t)

    I_bb = eval_basis_scalar(n - 2)
    D2 = get_deriv_mat(n, 2)
    val2 = np.trace(c.T @ D2.T @ I_bb @ D2 @ c)

    assert np.allclose(val1, val2)


def test_interpolate():
    order = 5
    dim = 3
    pts = np.random.normal(size=(order+1, dim))
    D_left = np.random.normal(size=(dim))
    D_right = np.random.normal(size=(dim))

    X = interpolate(pts, order,
        derivs_left=[(1, D_left)], 
        derivs_right=[(1, D_right)]
    )

    poly = bezier2poly(X[0])
    D_poly = polyder(poly)
    D_p = polyval(0, D_poly)
    assert np.allclose(D_p, D_left), 'Didn\'t fit derivative at left'

    poly = bezier2poly(X[-1])
    D_poly = polyder(poly)
    D_p = polyval(1, D_poly)
    assert np.allclose(D_p, D_right), 'Didn\'t fit derivative at right'

    for i in range(len(X)):
        poly = bezier2poly(X[i])
        assert np.allclose(pts[i], polyval(0, poly)), 'Bezier curve is discontinuous'
        assert np.allclose(pts[i+1], polyval(1, poly)), 'Bezier curve is discontinuous'

    for i in range(len(X) - 1):
        poly1 = bezier2poly(X[i])
        poly2 = bezier2poly(X[i+1])
        for k in range(1, order):
            D_poly1 = polyder(poly1, k)
            D_poly2 = polyder(poly2, k)
            assert np.allclose(polyval(1, D_poly1), polyval(0, D_poly2)), \
                'Derivative of order %d is discontinuous' % k


def test_eval_basis():
    order = 3
    t = np.linspace(0, 1, 100)
    C = np.random.normal(size=(order+1,3))
    poly = bezier2poly(C)
    val1 = polyval(t, poly).T
    B = eval_basis(order, t)
    val2 = B @ C
    assert np.allclose(val1, val2)


def test_eval_bezier():
    order = 3
    t = np.linspace(0, 1, 100)
    C = np.random.normal(size=(order+1,3))
    poly = bezier2poly(C)
    vals1 = eval_bezier(C, t)
    vals2 = polyval(t, poly).T
    assert np.allclose(vals1, vals2)


def test_bezier_deriv():
    from scipy.interpolate import make_interp_spline
    order = 3
    t = np.linspace(0, 1, 100)
    C = np.random.normal(size=(order+1,2))
    vals1 = eval_bezier(C, t)
    sp = make_interp_spline(t, vals1)
    vals2 = eval_bezier_deriv(C, t)
    assert np.allclose(sp(t, 1), vals2)


def test_bezier_length():
    order = 3
    t = np.linspace(0, 1, 1000)
    C = np.random.normal(size=(order+1,2))
    vals = eval_bezier(C, t)
    d = np.diff(vals, axis=0)
    l1 = np.sum(np.linalg.norm(d, axis=1))
    l2 = eval_bezier_length(C, np.array([0.4, 0.6, 1.]))
    assert np.allclose(l1, l2, atol=1e-3)


def test_all():
    np.random.seed(0)
    test_bezier_length()
    test_bezier_deriv()
    test_eval_basis_scalar()
    test_interpolate()
    test_eval_basis()


if __name__ == '__main__':
    test_all()
