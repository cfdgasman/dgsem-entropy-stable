"""Entropy-stable split-form DGSEM: acoustic pulse (vs exact), entropy budgets and KHI robustness."""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.special import j0

from esdg import Scheme, Solver, conservative, primitives
from esdg.euler import GAMMA

P0 = 1 / GAMMA  # background pressure with rho0 = 1  ->  sound speed c = 1
EPS, B = 1e-5, 0.1  # pulse amplitude and half-width
ALPHA = np.log(2) / B**2


def pulse_initial(s):
    r2 = s.x**2 + s.y**2
    dp = EPS * np.exp(-ALPHA * r2)
    return conservative(1 + dp, 0 * r2, 0 * r2, P0 + dp)  # isentropic: rho' = p' / c^2


def pulse_exact(r, t, nxi=40_000):
    """Linear acoustics (Tam 1995): p'(r,t) = eps/(2 alpha) int exp(-xi^2/4alpha) cos(xi t) J0(xi r) xi dxi."""
    xi = np.linspace(0, 12 * np.sqrt(ALPHA), nxi)
    kern = np.exp(-(xi**2) / (4 * ALPHA)) * np.cos(xi * t) * xi
    r = np.atleast_1d(np.asarray(r, float))
    out = np.array([np.trapezoid(kern * j0(xi * rr), xi) for rr in r.ravel()])
    return (EPS / (2 * ALPHA) * out).reshape(r.shape)


def to_image(s, f):
    """Element-wise (Kx, Ky, n, n) array -> 2D image (Kx*n, Ky*n)."""
    return f.transpose(0, 2, 1, 3).reshape(s.x.shape[0] * s.x.shape[2], -1)


def acoustic(t_end=0.5):
    print("Acoustic pulse, t = 0.5, 8 x 8 elements: p-refinement")
    print("| N | points per direction | max error / amplitude | relative L2 error | entropy change |")
    print("|---|---|---|---|---|")
    res = []
    for N in range(2, 11):
        s = Solver(N, 8, 8, scheme=Scheme("ec", "es"))
        U0 = pulse_initial(s)
        S0 = s.total_entropy(U0)
        U, t, ok = s.run(U0, t_end)
        dp = primitives(U)[3] - P0
        ex = pulse_exact(np.hypot(s.x, s.y), t_end)
        emax = np.abs(dp - ex).max() / EPS
        el2 = np.sqrt(s.integrate((dp - ex) ** 2) / s.integrate(ex**2))
        dS = s.total_entropy(U) - S0
        res.append((N, emax, el2))
        print(f"| {N} | {8 * (N + 1)} | {emax:.2e} | {el2:.2e} | {dS:+.1e} |")
    return np.array(res)


def acoustic_figures():
    s = Solver(7, 16, 16, scheme=Scheme("ec", "es"))
    U0 = pulse_initial(s)
    grab_at = np.linspace(0.02, 0.6, 30)
    frames = [(0.0, primitives(U0)[3] - P0)]

    def cb(t, U):
        if len(frames) <= len(grab_at) and t >= grab_at[len(frames) - 1] - 1e-12:
            frames.append((t, primitives(U)[3] - P0))

    s.run(U0, 0.6, callback=cb)

    fig, ax = plt.subplots(figsize=(6, 3.8))
    rline = np.linspace(0, 1, 400)
    for c, target in (("C0", 0.25), ("C3", 0.5)):
        U2, _, _ = s.run(U0, target)
        dp = (primitives(U2)[3] - P0).ravel() / EPS
        rr = np.hypot(s.x, s.y).ravel()
        m = (np.abs(s.y.ravel()) < 1e-12) & (s.x.ravel() >= 0)
        ax.plot(rline, pulse_exact(rline, target) / EPS, "-", color=c, lw=1.5, label=f"exact, t = {target}")
        ax.plot(rr[m], dp[m], "o", ms=3, mfc="none", color=c, label=f"DGSEM N = 7, t = {target}")
    ax.set(xlabel="r", ylabel="p′ / ε", title="Acoustic pulse along y = 0 vs exact linear solution")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("docs/acoustic_profile.png", dpi=120)
    plt.close(fig)

    X, Y = to_image(s, s.x), to_image(s, s.y)
    fig, ax = plt.subplots(figsize=(3.8, 3.8))
    fig.subplots_adjust(0, 0, 1, 0.9)
    pc = ax.pcolormesh(X, Y, to_image(s, frames[0][1]) / EPS, cmap="RdBu_r", vmin=-0.35, vmax=0.35, shading="gouraud")
    ax.set(aspect="equal", xticks=[], yticks=[])
    title = ax.set_title("")

    def update(k):
        t, f = frames[k]
        pc.set_array(to_image(s, f).ravel() / EPS)
        title.set_text(f"Acoustic pulse p′/ε, t = {t:.2f}")
        return [pc]

    FuncAnimation(fig, update, frames=len(frames)).save("docs/acoustic.gif", writer=PillowWriter(fps=10), dpi=70)
    plt.close(fig)


def khi_initial(s):
    Bf = np.tanh(15 * s.y + 7.5) - np.tanh(15 * s.y - 7.5)
    return conservative(0.5 + 0.75 * Bf, 0.5 * (Bf - 1), 0.1 * np.sin(2 * np.pi * s.x), 1.0 + 0 * s.x)


KHI_SCHEMES = {
    "split form, entropy stable": Scheme("ec", "es"),
    "split form, entropy conservative": Scheme("ec", "ec"),
    "standard DGSEM": Scheme("central", "es"),
}


def khi(N=3, K=16, t_end=5.0):
    print(f"\nKelvin-Helmholtz instability, N = {N}, {K} x {K} elements, to t = {t_end}")
    runs = {}
    for label, sch in KHI_SCHEMES.items():
        s = Solver(N, K, K, scheme=sch)
        U0 = khi_initial(s)
        S0 = s.total_entropy(U0)
        hist = [(0.0, 0.0)]
        snaps = [(0.0, U0[0].copy())]

        def cb(t, U, s=s, S0=S0, hist=hist, snaps=snaps):
            hist.append((t, s.total_entropy(U) - S0))
            if t >= len(snaps) * 0.05:
                snaps.append((t, U[0].copy()))

        t0 = time.perf_counter()
        _, t, ok = s.run(U0, t_end, callback=cb)
        status = "completed" if ok else f"CRASHED at t = {t:.2f}"
        print(f"  {label:34s}: {status} ({time.perf_counter() - t0:.0f} s), entropy change {hist[-1][1]:+.3e}")
        runs[label] = (s, np.array(hist), snaps, ok, t)
    return runs


def khi_figures(runs):
    fig, ax = plt.subplots(figsize=(6.2, 4))
    for label, (_, hist, _, ok, t) in runs.items():
        ax.plot(hist[:, 0], hist[:, 1], label=label + ("" if ok else f" (crashed, t = {t:.2f})"))
        if not ok:
            ax.plot(hist[-1, 0], hist[-1, 1], "kx", ms=9)
    ax.set(xlabel="t", ylabel="S(t) − S(0)", title="Total entropy, Kelvin–Helmholtz (N = 3, 16×16)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("docs/entropy_history.png", dpi=120)
    plt.close(fig)

    s, _, snaps, _, _ = runs["split form, entropy stable"]
    X, Y = to_image(s, s.x), to_image(s, s.y)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
    for ax, target in zip(axes, (0.0, 1.5, 3.0, 5.0)):
        t, rho = min(snaps, key=lambda f: abs(f[0] - target))
        ax.pcolormesh(X, Y, to_image(s, rho), cmap="magma", vmin=0.3, vmax=2.2, shading="gouraud")
        ax.set(aspect="equal", xticks=[], yticks=[], title=f"ρ, t = {t:.1f}")
    fig.suptitle("Entropy-stable split-form DGSEM, Kelvin–Helmholtz (N = 3, 16×16 elements)")
    fig.tight_layout()
    fig.savefig("docs/khi.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.8, 3.8))
    fig.subplots_adjust(0, 0, 1, 0.9)
    pc = ax.pcolormesh(X, Y, to_image(s, snaps[0][1]), cmap="magma", vmin=0.3, vmax=2.2, shading="gouraud")
    ax.set(aspect="equal", xticks=[], yticks=[])
    title = ax.set_title("")

    def update(k):
        t, rho = snaps[k]
        pc.set_array(to_image(s, rho).ravel())
        title.set_text(f"KHI, entropy-stable DG, t = {t:.2f}")
        return [pc]

    FuncAnimation(fig, update, frames=len(snaps)).save("docs/khi.gif", writer=PillowWriter(fps=15), dpi=70)
    plt.close(fig)


def convergence_figure(res):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.semilogy(res[:, 0], res[:, 1], "o-", label="max error / ε")
    ax.semilogy(res[:, 0], res[:, 2], "s-", label="relative L2 error")
    ax.set(xlabel="polynomial order N (8×8 elements)", ylabel="error", title="Acoustic pulse, t = 0.5")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig("docs/acoustic_convergence.png", dpi=120)
    plt.close(fig)


def main():
    res = acoustic()
    convergence_figure(res)
    acoustic_figures()
    khi_figures(khi())


if __name__ == "__main__":
    main()
