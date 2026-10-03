"""Figures of the paper (single-column, 3.4 in wide)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fast_eval as fe
from step1_linear import Mixture, grid2, entropy, J2
from step2_quartic import quartic_ratio

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.serif": ["DejaVu Serif"],
                     "font.size": 8.5, "axes.labelsize": 9, "legend.fontsize": 7.2, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "lines.linewidth": 1.2, "axes.linewidth": 0.6})
W = 3.4
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
RES = os.path.join(HERE, "..", "results")
os.makedirs(FIG, exist_ok=True)


def fig1():
    """(a) S_C(t) for a harmonic oscillator; (b) threshold sigma_dot/sigma = |1 - w^2|/2."""
    fig, ax = plt.subplots(1, 2, figsize=(W, 1.75), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.42})
    Z, dA = grid2(L=9.0, n=401)
    mix = Mixture([0.55, 0.45], [np.array([1.2, -0.4]), np.array([-1.0, 0.9])],
                  [np.array([[0.30, 0.12], [0.12, 0.20]]), np.array([[0.15, -0.05], [-0.05, 0.40]])])
    ts = np.linspace(0, 4, 81)
    s0 = 0.3
    cases = [(1.0, 0.0, "C0", "$\\omega=1$, $\\dot\\sigma=0$"),
             (np.sqrt(2.5), 0.0, "C3", "$\\omega^2=2.5$, $\\dot\\sigma=0$"),
             (np.sqrt(2.5), 0.75 * 1.0001, "C2", "$\\omega^2=2.5$, $\\dot\\sigma/\\sigma=0.75$")]
    for w, r, c, lab in cases:
        B = J2 @ np.diag([w * w, 1.0])
        S = [entropy(Z, dA, mix.at(B, t, (s0 * np.exp(r * t)) ** 2 * np.eye(2))) for t in ts]
        ax[0].plot(ts, np.array(S) - S[0], color=c, label=lab)
    ax[0].set_xlabel("$t$"); ax[0].set_ylabel("$S_C(t)-S_C(0)$")
    ax[0].legend(loc="upper right", frameon=False, handlelength=1.2, borderaxespad=0.2, fontsize=6.4)
    ax[0].set_ylim(-0.12, 0.75)
    ax[0].text(0.97, 0.05, "(a)", transform=ax[0].transAxes, ha="right")
    w = np.linspace(0, 2, 401)
    thr = np.abs(1 - w ** 2) / 2
    ax[1].fill_between(w, thr, 1.6, color="C2", alpha=0.18, lw=0)
    ax[1].plot(w, thr, "k")
    ax[1].text(1.0, 1.15, "arrow for\nevery state", ha="center", va="center", fontsize=7)
    ax[1].text(0.42, 0.12, "some states\ndecrease", ha="center", va="center", fontsize=7)
    ax[1].set_xlim(0, 2); ax[1].set_ylim(0, 1.6)
    ax[1].set_xlabel("$\\omega$"); ax[1].set_ylabel("$\\dot\\sigma/\\sigma$")
    ax[1].text(0.97, 0.05, "(b)", transform=ax[1].transAxes, ha="right")
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.22, top=0.97)
    fig.savefig(os.path.join(FIG, "fig1_linear.pdf")); plt.close(fig)


def fig2():
    """Fig. 3 -- quartic oscillator: rate needed to keep a localised state from losing entropy."""
    fig, ax = plt.subplots(figsize=(W, 2.0))
    q0 = np.linspace(0, 4, 41)
    out = np.array([quartic_ratio(q) for q in q0])
    qq = np.linspace(0, 4, 400)
    ax.plot(qq, np.abs(1 - 3 * qq ** 2) / 2, "k", label="local strain $\\lambda_{\\max}[\\mathrm{Sym}\\,Df(q_0)]$")
    ax.plot(q0, out[:, 1], color="C0", ls="--", label="linearised flow")
    ax.plot(q0, out[:, 0], "o", ms=2.6, color="C3", label="exact nonlinear flow")
    ax.set_xlabel("$q_0$"); ax.set_ylabel("$R=-\\Sigma/(\\sigma^2\\,\\mathrm{tr}\\,F)$")
    ax.set_xlim(0, 4); ax.set_ylim(0, 24)
    ax.legend(frameon=False, loc="upper left")
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.2, top=0.97)
    fig.savefig(os.path.join(FIG, "fig3_quartic.pdf")); plt.close(fig)
    return q0, out


def fig3():
    """Fig. 2 -- pendulum: R = -Sigma/(sigma^2 tr F) versus x = sigma^2 tr F; theorem and conjecture."""
    from step3_nonlinear_bound import random_state
    rng = np.random.default_rng(7)
    X, R = [], []
    for _ in range(200):
        ws, mus, Ss, sig, n = random_state(rng)
        _, trF, Sg = fe.dSdt(ws, mus, Ss, sig, 0, 'pendulum', n=n)
        X.append(sig * sig * trF); R.append(-Sg / (sig * sig * trF))
    fig, ax = plt.subplots(figsize=(W, 2.55))
    x = np.linspace(0.05, 2, 400)
    ax.fill_between(x, (2 - x) / x, 3, color="0.85", lw=0)
    ax.plot(x, (2 - x) / x, "k", label="Theorem 6: $R\\leq\\kappa(d-x)/x$")
    ax.axhline(1, color="C3", ls="--", label="Conjecture 1: $R\\leq\\kappa=1$")
    ax.scatter(X, R, s=4, color="C0", alpha=0.6, lw=0, label="random states")
    from step5_search import optima
    mk = {0.1: "v", 0.3: "s", 1.0: "D"}
    for sig, xo, ro in optima():
        ax.scatter([xo], [ro], marker=mk[sig], s=14, facecolor="none", edgecolor="k", lw=0.7)
    ax.scatter([], [], marker="s", s=14, facecolor="none", edgecolor="k", lw=0.7, label="adversarial search")
    ax.axvline(1, color="0.4", lw=0.6, ls=":")
    ax.set_xlim(0, 2); ax.set_ylim(-1.2, 2.2)
    ax.set_xlabel("$x=\\sigma^2\\,\\mathrm{tr}\\,F$"); ax.set_ylabel("$R=-\\Sigma/x$", labelpad=1)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.6, columnspacing=1.0, handlelength=1.6)
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.16, top=0.83)
    fig.savefig(os.path.join(FIG, "fig2_pendulum.pdf")); plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3()   # fig2() writes Fig. 3 (quartic), fig3() writes Fig. 2 (pendulum)
    print("figures written to figures/")
