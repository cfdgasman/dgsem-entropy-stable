import numpy as np
import pytest

from esdg import Scheme, Solver, conservative
from esdg.basis import diff_matrix, lgl
from esdg.euler import entropy_variables, log_mean, physical_flux, ranocha_flux

rng = np.random.default_rng(0)


def random_states(n=500):
    return conservative(rng.uniform(0.5, 2, n), rng.uniform(-1, 1, n), rng.uniform(-1, 1, n), rng.uniform(0.5, 2, n))


@pytest.mark.parametrize("N", [3, 6, 9])
def test_sbp_property(N):
    x, w = lgl(N)
    Q = np.diag(w) @ diff_matrix(x)
    B = np.zeros((N + 1, N + 1))
    B[0, 0], B[-1, -1] = -1, 1
    assert np.abs(Q + Q.T - B).max() < 1e-12


def test_log_mean_accurate_near_equal_arguments():
    a = np.array([1.0, 1.0, 2.0])
    b = a * (1 + np.array([1e-9, 1e-3, 0.5]))
    exact = (a - b) / (np.log(a) - np.log(b))
    assert np.allclose(log_mean(a, b), exact, rtol=1e-12)


@pytest.mark.parametrize("axis", [0, 1])
def test_ranocha_flux_consistent_and_entropy_conservative(axis):
    UL, UR = random_states(), random_states()
    assert np.allclose(ranocha_flux(UL, UL, axis), physical_flux(UL, axis), atol=1e-14)
    # Tadmor's condition  [[v]] . f# = [[psi]],  psi = rho u_n
    dv = entropy_variables(UR) - entropy_variables(UL)
    lhs = (dv * ranocha_flux(UL, UR, axis)).sum(axis=0)
    assert np.allclose(lhs, UR[1 + axis] - UL[1 + axis], atol=1e-12)


def _rough_state(s):
    """Smooth field plus element-wise random perturbations, so interfaces have jumps."""
    pert = 0.05 * rng.standard_normal(s.x.shape)
    rho = 1 + 0.3 * np.sin(np.pi * s.x) * np.cos(np.pi * s.y) + pert
    u = 0.3 + 0.1 * np.cos(np.pi * s.y) + pert
    v = -0.2 + 0.1 * np.sin(np.pi * s.x)
    p = 1 + 0.2 * np.cos(np.pi * (s.x + s.y)) + pert
    return conservative(rho, u, v, p)


def test_semidiscrete_entropy_budgets():
    s_ec = Solver(4, 4, 4, scheme=Scheme("ec", "ec"))
    U = _rough_state(s_ec)
    assert abs(s_ec.entropy_rate(U)) < 1e-12  # entropy conservative to round-off
    assert Solver(4, 4, 4, scheme=Scheme("ec", "es")).entropy_rate(U) < -1e-6  # strictly dissipative


def test_free_stream_preserved():
    s = Solver(5, 3, 3, scheme=Scheme("ec", "es"))
    U = conservative(*(np.full(s.x.shape, c) for c in (1.3, 0.4, -0.7, 2.0)))
    assert np.abs(s.rhs(U)).max() < 1e-13


def test_standard_dgsem_equals_central_flux_differencing():
    """2 sum_k D_ik {f}_ik = (D f)_i  because rows of D sum to zero."""
    s = Solver(4, 2, 2, scheme=Scheme("central", "es"))
    U = _rough_state(s)
    D = s.D
    f = physical_flux(U, 0)
    direct = np.einsum("ik,cxykj->cxyij", D, f)
    assert np.allclose(s._volume(U, 0), direct, atol=1e-12)
