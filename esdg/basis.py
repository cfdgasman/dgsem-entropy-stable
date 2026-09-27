"""Legendre-Gauss-Lobatto nodes, weights and the spectral differentiation matrix."""

import numpy as np
from numpy.polynomial import legendre as L


def lgl(n):
    """N+1 LGL nodes and weights on [-1, 1] for polynomial degree n."""
    if n == 1:
        return np.array([-1.0, 1.0]), np.array([1.0, 1.0])
    interior = np.sort(L.legroots(L.legder([0] * n + [1])).real)
    x = np.concatenate(([-1.0], interior, [1.0]))
    Pn = L.legval(x, [0] * n + [1])
    w = 2.0 / (n * (n + 1) * Pn**2)
    return x, w


def diff_matrix(x):
    """Lagrange differentiation matrix D_ij = l_j'(x_i) via barycentric weights."""
    n = len(x)
    dx = x[:, None] - x[None, :]
    np.fill_diagonal(dx, 1.0)
    bw = 1.0 / dx.prod(axis=1)
    D = (bw[None, :] / bw[:, None]) / dx
    np.fill_diagonal(D, 0.0)
    np.fill_diagonal(D, -D.sum(axis=1))
    return D


def interp_matrix(x, xi):
    """Matrix that interpolates nodal values at x to the points xi."""
    dx = x[:, None] - x[None, :]
    np.fill_diagonal(dx, 1.0)
    bw = 1.0 / dx.prod(axis=1)
    diff = xi[:, None] - x[None, :]
    exact = np.isclose(diff, 0.0)
    diff[exact] = 1.0
    M = bw[None, :] / diff
    M /= M.sum(axis=1, keepdims=True)
    rows, cols = np.nonzero(exact)
    M[rows] = 0.0
    M[rows, cols] = 1.0
    return M
