"""Entropy-stable split-form DGSEM for the 2D compressible Euler equations.

Periodic Cartesian mesh of Kx x Ky elements, tensor-product LGL nodes of degree N.

Semi-discrete form (Gassner, Winters & Kopriva 2016; Chen & Shu 2017), per direction:

    du_i/dt = -(2/h) [ 2 sum_k D_ik f#(u_i, u_k)                 (volume: flux differencing)
                       + delta_iN (f*_R - f(u_N)) / w_N            (surface, right)
                       - delta_i0 (f*_L - f(u_0)) / w_0 ]          (surface, left)

D is the LGL differentiation matrix, which is SBP:  M D + (M D)^T = diag(-1, 0, ..., 0, 1).

* f# = entropy-conserving, kinetic-energy-preserving two-point flux (Ranocha 2018)
      -> "split form"; entropy is conserved inside every element.
* f# = central average {f}                -> reduces exactly to the standard DGSEM (D f).
* f* = f# + Rusanov dissipation            -> entropy stable (total entropy can only decrease)
* f* = f#                                  -> entropy conservative (total entropy constant
                                              in the semi-discretisation, to round-off).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .basis import diff_matrix, lgl

GAMMA = 1.4


# ---------------------------------------------------------------- physics


def primitives(U):
    rho = U[0]
    u = U[1] / rho
    v = U[2] / rho
    p = (GAMMA - 1) * (U[3] - 0.5 * rho * (u * u + v * v))
    return rho, u, v, p


def conservative(rho, u, v, p):
    return np.stack([rho, rho * u, rho * v, p / (GAMMA - 1) + 0.5 * rho * (u * u + v * v)])


def physical_flux(U, axis):
    rho, u, v, p = primitives(U)
    un = u if axis == 0 else v
    return np.stack([rho * un, U[1] * un + (p if axis == 0 else 0), U[2] * un + (p if axis == 1 else 0),
                     (U[3] + p) * un])


def entropy(U):
    """Mathematical entropy S = -rho s / (gamma - 1), s = ln(p rho^-gamma)."""
    rho, _, _, p = primitives(U)
    return -rho * (np.log(p) - GAMMA * np.log(rho)) / (GAMMA - 1)


def entropy_variables(U):
    rho, u, v, p = primitives(U)
    s = np.log(p) - GAMMA * np.log(rho)
    b = rho / p
    return np.stack([(GAMMA - s) / (GAMMA - 1) - 0.5 * b * (u * u + v * v), b * u, b * v, -b])


def log_mean(a, b):
    """Numerically stable logarithmic mean (a - b) / (ln a - ln b)  (Ismail & Roe 2009)."""
    zeta = a / b
    f = (zeta - 1) / (zeta + 1)
    uu = f * f
    F = np.where(uu < 1e-4, 1 + uu / 3 + uu * uu / 5 + uu**3 / 7, np.log(zeta) / (2 * f + (uu < 1e-4)))
    return (a + b) / (2 * F)


def ranocha_flux(UL, UR, axis):
    """Entropy-conserving and kinetic-energy-preserving flux of Ranocha (2018)."""
    rl, ul, vl, pl = primitives(UL)
    rr, ur, vr, pr = primitives(UR)
    rho_log = log_mean(rl, rr)
    beta_log = log_mean(rl / pl, rr / pr)  # log mean of rho / p
    u_avg, v_avg, p_avg = 0.5 * (ul + ur), 0.5 * (vl + vr), 0.5 * (pl + pr)
    un_l, un_r = (ul, ur) if axis == 0 else (vl, vr)
    un_avg = 0.5 * (un_l + un_r)
    f_rho = rho_log * un_avg
    f_mx = f_rho * u_avg + (p_avg if axis == 0 else 0)
    f_my = f_rho * v_avg + (p_avg if axis == 1 else 0)
    f_E = f_rho * (0.5 * (ul * ur + vl * vr) + 1 / ((GAMMA - 1) * beta_log)) + 0.5 * (pl * un_r + pr * un_l)
    return np.stack([f_rho, f_mx, f_my, f_E])


def central_flux(UL, UR, axis):
    return 0.5 * (physical_flux(UL, axis) + physical_flux(UR, axis))


def max_speed(U, axis):
    rho, u, v, p = primitives(U)
    return np.abs(u if axis == 0 else v) + np.sqrt(GAMMA * p / rho)


# ---------------------------------------------------------------- discretisation


@dataclass
class Scheme:
    """volume: 'ec' (split form, entropy conservative) or 'central' (standard DGSEM).
    surface: 'ec' (entropy conservative) or 'es' (EC + Rusanov, entropy stable)."""

    volume: str = "ec"
    surface: str = "es"


class Solver:
    def __init__(self, N, Kx, Ky, domain=(-1.0, 1.0, -1.0, 1.0), scheme=Scheme()):
        self.N, self.Kx, self.Ky = N, Kx, Ky
        self.xi, self.w = lgl(N)
        self.D = diff_matrix(self.xi)
        x0, x1, y0, y1 = domain
        self.hx, self.hy = (x1 - x0) / Kx, (y1 - y0) / Ky
        ex = x0 + self.hx * (np.arange(Kx)[:, None, None, None] + (self.xi[None, None, :, None] + 1) / 2)
        ey = y0 + self.hy * (np.arange(Ky)[None, :, None, None] + (self.xi[None, None, None, :] + 1) / 2)
        self.x, self.y = np.broadcast_arrays(ex, ey)  # (Kx, Ky, n, n)
        self.scheme = scheme
        self.vol_flux = ranocha_flux if scheme.volume == "ec" else central_flux
        self.W = self.w[:, None] * self.w[None, :] * self.hx * self.hy / 4  # quadrature weights incl. Jacobian

    # U has shape (4, Kx, Ky, n, n); element axis for x is 1, node axis for x is 3
    def _volume(self, U, axis):
        n = self.N + 1
        if axis == 0:
            Ui = U[:, :, :, :, None, :]  # (4, Kx, Ky, n_i, 1, n_j)
            Uk = U[:, :, :, None, :, :]  # (4, Kx, Ky, 1, n_k, n_j)
            Fs = self.vol_flux(np.broadcast_to(Ui, U.shape[:3] + (n, n, n)),
                               np.broadcast_to(Uk, U.shape[:3] + (n, n, n)), 0)
            return 2 * np.einsum("ik,cxyikj->cxyij", self.D, Fs)
        Ui = U[:, :, :, :, :, None]  # (4, Kx, Ky, n_i, n_j, 1)
        Uk = U[:, :, :, :, None, :]  # (4, Kx, Ky, n_i, 1, n_k)
        Fs = self.vol_flux(np.broadcast_to(Ui, U.shape[:3] + (n, n, n)),
                           np.broadcast_to(Uk, U.shape[:3] + (n, n, n)), 1)
        return 2 * np.einsum("jk,cxyijk->cxyij", self.D, Fs)

    def _surface(self, U, axis):
        if axis == 0:
            UL = U[:, :, :, -1, :]  # right trace of each element
            UR = np.roll(U[:, :, :, 0, :], -1, axis=1)  # left trace of the right neighbour
        else:
            UL = U[:, :, :, :, -1]
            UR = np.roll(U[:, :, :, :, 0], -1, axis=2)
        fstar = ranocha_flux(UL, UR, axis)
        if self.scheme.surface == "es":
            lam = np.maximum(max_speed(UL, axis), max_speed(UR, axis))
            fstar = fstar - 0.5 * lam * (UR - UL)
        out = np.zeros_like(U)
        ax_el = 1 + axis
        f_left = np.roll(fstar, 1, axis=ax_el)  # f* on each element's left face
        if axis == 0:
            out[:, :, :, -1, :] += (fstar - physical_flux(UL, 0)) / self.w[-1]
            out[:, :, :, 0, :] -= (f_left - physical_flux(U[:, :, :, 0, :], 0)) / self.w[0]
        else:
            out[:, :, :, :, -1] += (fstar - physical_flux(UL, 1)) / self.w[-1]
            out[:, :, :, :, 0] -= (f_left - physical_flux(U[:, :, :, :, 0], 1)) / self.w[0]
        return out

    def rhs(self, U):
        return -(2 / self.hx) * (self._volume(U, 0) + self._surface(U, 0)) - (2 / self.hy) * (
            self._volume(U, 1) + self._surface(U, 1)
        )

    # ------------------------------------------------------------ diagnostics
    def integrate(self, q):
        return float((self.W * q).sum())

    def total_entropy(self, U):
        return self.integrate(entropy(U))

    def entropy_rate(self, U):
        """Semi-discrete dS/dt = sum W v(U) . rhs(U)."""
        return self.integrate((entropy_variables(U) * self.rhs(U)).sum(axis=0))

    def dt(self, U, cfl=0.4):
        """CFL condition on the smallest LGL node spacing, which shrinks like 1/N^2 near element edges."""
        d = (self.xi[1] - self.xi[0]) / 2
        rate = max_speed(U, 0).max() / (d * self.hx) + max_speed(U, 1).max() / (d * self.hy)
        return cfl / rate

    def run(self, U, t_end, cfl=0.4, callback=None):
        """Explicit SSP-RK3 (Shu-Osher). Stops early and returns the failure time on NaN/negative state."""
        t = 0.0
        while t < t_end - 1e-14:
            dt = min(self.dt(U, cfl), t_end - t)
            U1 = U + dt * self.rhs(U)
            U2 = 0.75 * U + 0.25 * (U1 + dt * self.rhs(U1))
            U = U / 3 + 2 / 3 * (U2 + dt * self.rhs(U2))
            t += dt
            rho, _, _, p = primitives(U)
            if not (np.isfinite(U).all() and rho.min() > 0 and p.min() > 0):
                return U, t, False
            if callback is not None:
                callback(t, U)
        return U, t, True
