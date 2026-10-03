"""Step 1 -- linear Hamiltonian flows: Theorem 3 and Corollaries 4-5 (paper Sec. III, Eqs. 8-13, Fig. 1).

Setting: z in R^{2n}, H = 1/2 z^T M z, flow dz/dt = B z with B = J M (tr B = 0).
Coarse-grained density rho_C = rho * N(0, C(t)).
Claims checked:
 (A) P L P^{-1} = L - 1/2 grad^T (B C + C B^T) grad   (exact, series terminates)
     -> rho_C obeys a Fokker-Planck eq. with drift Bz and diffusion G = Cdot - BC - CB^T
 (B) dS_C/dt = 1/2 Tr(G F),  F = int grad rho_C grad rho_C^T / rho_C   (any initial rho)
 (C) dS_C/dt >= 0 for every initial density  <=>  G >= 0  (sharp)
 (D) isotropic C = sigma^2 I, n = 1, H = (p^2 + w^2 q^2)/2: condition sigma_dot/sigma >= |1-w^2|/2;
     w = 0 gives the free-particle condition sigma_dot >= sigma/2, necessary and sufficient
 (E) fixed resolution (Cdot = 0): G = -(BC+CB^T) is traceless-weighted -> either G = 0 (matched,
     dS/dt = 0 for all states) or some state has dS/dt < 0: no arrow without resolution flow.
"""
import numpy as np
from scipy.linalg import expm

J2 = np.array([[0., 1.], [-1., 0.]])
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}\n         {detail}")


def gauss(Z, mu, S):
    d = Z - mu
    Si = np.linalg.inv(S)
    k = len(mu)
    return np.exp(-0.5 * np.einsum('...i,ij,...j->...', d, Si, d)) / np.sqrt((2 * np.pi) ** k * np.linalg.det(S))


class Mixture:
    """Gaussian mixture evolved by the linear flow and smoothed by C(t): exact rho_C(t)."""
    def __init__(self, w, mus, Ss):
        self.w, self.mus, self.Ss = w, mus, Ss

    def at(self, B, t, C):
        E = expm(B * t)
        return [(wi, E @ m, E @ S @ E.T + C) for wi, m, S in zip(self.w, self.mus, self.Ss)]


def grid2(L=9.0, n=561):
    x = np.linspace(-L, L, n)
    Q, P = np.meshgrid(x, x, indexing='ij')
    return np.stack([Q, P], -1), (x[1] - x[0]) ** 2


def density(Z, comps):
    return sum(w * gauss(Z, m, S) for w, m, S in comps)


def grad_density(Z, comps):
    g = 0
    for w, m, S in comps:
        g = g + w * gauss(Z, m, S)[..., None] * (-(Z - m) @ np.linalg.inv(S).T)
    return g


def entropy(Z, dA, comps):
    r = density(Z, comps)
    r = np.where(r > 1e-300, r, 1e-300)
    return -np.sum(r * np.log(r)) * dA


def fisher(Z, dA, comps):
    r = density(Z, comps)
    g = grad_density(Z, comps)
    m = r > 1e-300
    return np.einsum('ki,kj->ij', g[m] / r[m, None], g[m]) * dA


def dSdt_numeric(Z, dA, mix, B, t, Cfun, h=1e-4):
    return (entropy(Z, dA, mix.at(B, t + h, Cfun(t + h))) - entropy(Z, dA, mix.at(B, t - h, Cfun(t - h)))) / (2 * h)


def run():
    Z, dA = grid2()
    mix = Mixture([0.55, 0.45],
                  [np.array([1.2, -0.4]), np.array([-1.0, 0.9])],
                  [np.array([[0.30, 0.12], [0.12, 0.20]]), np.array([[0.15, -0.05], [-0.05, 0.40]])])

    # (A)+(B): non-Gaussian initial state, anisotropic time-dependent C, several oscillators
    worst = 0
    for w2 in (0.0, 0.5, 2.3):
        M = np.diag([w2, 1.0])
        B = J2 @ M
        C0 = np.array([[0.20, 0.05], [0.05, 0.12]])
        Cd = np.array([[0.04, 0.01], [0.01, 0.07]])
        Cfun = lambda t: C0 + Cd * t
        t = 0.7
        comps = mix.at(B, t, Cfun(t))
        G = Cd - B @ Cfun(t) - Cfun(t) @ B.T
        pred = 0.5 * np.trace(G @ fisher(Z, dA, comps))
        num = dSdt_numeric(Z, dA, mix, B, t, Cfun)
        worst = max(worst, abs(num - pred) / max(abs(pred), 1e-3))
    check("(A,B) dS_C/dt = 1/2 Tr(G F), G = Cdot - BC - CB^T (non-Gaussian state, 3 oscillators, anisotropic C(t))",
          worst < 1e-5, f"max relative deviation numeric vs formula = {worst:.1e}")

    # (C) sufficiency: G >= 0  -> dS/dt >= 0 for random mixtures
    rng = np.random.default_rng(1)
    B = J2 @ np.diag([2.0, 1.0])
    sig, = (0.5,)
    rate = 0.5 * np.max(np.linalg.eigvalsh(B + B.T)) * 1.0001    # sigma_dot/sigma at threshold
    Cfun = lambda t: (sig * np.exp(rate * t)) ** 2 * np.eye(2)
    mins = []
    for _ in range(25):
        k = rng.integers(1, 4)
        ws = rng.dirichlet(np.ones(k))
        mus = [rng.normal(0, 1.5, 2) for _ in range(k)]
        Ss = []
        for _ in range(k):
            A = rng.normal(0, 0.6, (2, 2)); Ss.append(A @ A.T + 0.01 * np.eye(2))
        m = Mixture(ws, mus, Ss)
        mins.append(dSdt_numeric(Z, dA, m, B, 0.0, Cfun))
    check("(C) sufficiency: at sigma_dot/sigma just above threshold, dS/dt >= 0 for 25 random mixtures",
          min(mins) > -1e-7, f"min dS/dt = {min(mins):.2e}")

    # (C) necessity / sharpness: just below threshold a Gaussian state with dS/dt < 0 exists
    rate_lo = 0.5 * np.max(np.linalg.eigvalsh(B + B.T)) * 0.9
    Cl = lambda t: (sig * np.exp(rate_lo * t)) ** 2 * np.eye(2)
    G = 2 * rate_lo * Cl(0) - B @ Cl(0) - Cl(0) @ B.T
    ev, V = np.linalg.eigh(G)
    v = V[:, 0]                                       # negative direction of G
    S0 = 1e-4 * np.outer(v, v) + 50 * np.outer(V[:, 1], V[:, 1])   # narrow along v, wide otherwise
    worst_state = Mixture([1.0], [np.zeros(2)], [S0])
    Zw, dAw = grid2(L=30.0, n=1201)
    d_lo = dSdt_numeric(Zw, dAw, worst_state, B, 0.0, Cl)
    pred = 0.5 * np.trace(G @ np.linalg.inv(S0 + Cl(0)))
    check("(C) sharpness: 10% below threshold the state narrow along the negative eigenvector of G has dS/dt < 0",
          d_lo < 0 and abs(d_lo / pred - 1) < 1e-4, f"dS/dt = {d_lo:.4f} (formula {pred:.4f}); lambda_min(G) = {ev[0]:.3f}")

    # (D) isotropic oscillator threshold |1 - w^2|/2 ; w = 0 is the free particle
    for w in (0.0, 0.6, 1.7):
        B = J2 @ np.diag([w * w, 1.0])
        thr = 0.5 * np.max(np.linalg.eigvalsh(B + B.T))
        check(f"(D) w = {w}: threshold sigma_dot/sigma = lambda_max(B+B^T)/2 = |1 - w^2|/2",
              abs(thr - abs(1 - w * w) / 2) < 1e-14, f"{thr:.6f} vs {abs(1 - w * w) / 2:.6f}"
              + ("  (free particle: sigma_dot >= sigma/2, necessary and sufficient)" if w == 0 else ""))

    # (E) fixed resolution
    Cfix = lambda t: 0.3 * np.eye(2)
    Bm = J2 @ np.eye(2)                                # matched: w = 1
    ds = [dSdt_numeric(Z, dA, mix, Bm, t, Cfix) for t in (0.0, 0.8, 2.1)]
    check("(E) matched (w = 1), fixed sigma: dS/dt = 0 for a non-Gaussian state at all times (no arrow, no decrease)",
          max(map(abs, ds)) < 1e-8, "dS/dt = " + ", ".join(f"{d:.1e}" for d in ds))
    Bu = J2 @ np.diag([2.5, 1.0])
    ts = np.linspace(0, 3, 31)
    S = [entropy(Z, dA, mix.at(Bu, t, Cfix(t))) for t in ts]
    dS = np.diff(S)
    check("(E) unmatched (w^2 = 2.5), fixed sigma: coarse-grained entropy rises AND falls (oscillates)",
          dS.max() > 1e-4 and dS.min() < -1e-4, f"max step {dS.max():.3e}, min step {dS.min():.3e}")

    # (n = 2) coupled oscillators, 4D phase space, Gaussian state: closed-form check of (B)
    J4 = np.block([[np.zeros((2, 2)), np.eye(2)], [-np.eye(2), np.zeros((2, 2))]])
    Mq = np.array([[2.0, 0.6], [0.6, 1.3]]); M = np.block([[Mq, np.zeros((2, 2))], [np.zeros((2, 2)), np.eye(2)]])
    B = J4 @ M
    C0 = 0.1 * np.eye(4) + 0.02; Cd = 0.05 * np.eye(4)
    S0 = np.diag([0.3, 0.5, 0.2, 0.4]); t, h = 0.9, 1e-5
    Ssum = lambda t: expm(B * t) @ S0 @ expm(B * t).T + C0 + Cd * t
    Sent = lambda t: 0.5 * np.log(np.linalg.det(2 * np.pi * np.e * Ssum(t)))
    num = (Sent(t + h) - Sent(t - h)) / (2 * h)
    G = Cd - B @ (C0 + Cd * t) - (C0 + Cd * t) @ B.T
    pred = 0.5 * np.trace(G @ np.linalg.inv(Ssum(t)))
    check("(n = 2) coupled oscillators, anisotropic C: dS/dt = 1/2 Tr(G F)",
          abs(num / pred - 1) < 1e-7, f"numeric {num:.8f}, formula {pred:.8f}")

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return sum(results), len(results)


if __name__ == "__main__":
    p, t = run()
    raise SystemExit(0 if p == t else 1)
