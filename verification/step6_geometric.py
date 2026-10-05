"""Step 6 -- geometric picture and relation to diffusion models (paper Appendix A).

Claims checked:
 (a) Pull-back identity: Ctil = M^{-1} C M^{-T}, M = exp(Bt), satisfies dCtil/dt = M^{-1} G M^{-T}
     (random Hamiltonian B in d = 4, arbitrary SPD path C(t)).
 (b) S_C(t) = h(X_0 + N(0, Ctil(t))) for a non-Gaussian initial density (det M = 1).
 (c) Finite-time criterion: S_C(t2) >= S_C(t1) for every initial density  <=>  Ctil(t2) >= Ctil(t1)
     (sufficiency on random mixtures; necessity by an explicit Gaussian state).
 (d) Finite-time criterion is weaker than G >= 0 on [t1, t2]: an example where G has a negative
     eigenvalue during the interval but Ctil(t2) >= Ctil(t1).
 (e) General linear drift (tr B != 0): dS_C/dt = tr B + 1/2 tr(G F).
 (f) Realizability: a covariance schedule C(t) is produced by the linear SDE dZ = BZ dt + L dW with
     L L^T = G (G >= 0); the Lyapunov equation reproduces C(t).
 (g) Scalar drift B = b(t) I (variance-preserving/exploding diffusion): G = -sigma^2 d ln SNR/dt I,
     SNR = alpha^2/sigma^2, so G >= 0 <=> SNR non-increasing.
 (h) Hamiltonian drift, isotropic schedule C = sigma^2 I: a linear SDE realizes it iff
     sigma_dot/sigma >= lambda_max(Sym B) (smallest eigenvalue of G changes sign at the threshold).
"""
import numpy as np
from scipy.linalg import expm, sqrtm
from scipy.integrate import solve_ivp
from step1_linear import Mixture, grid2, entropy, fisher, J2

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}\n         {detail}")


def gauss_entropy(S):
    d = S.shape[0]
    return 0.5 * np.log((2 * np.pi * np.e) ** d * np.linalg.det(S))


def run():
    results.clear()
    rng = np.random.default_rng(7)

    # (a) pull-back identity, d = 4
    J4 = np.block([[np.zeros((2, 2)), np.eye(2)], [-np.eye(2), np.zeros((2, 2))]])
    H = rng.normal(size=(4, 4)); H = H + H.T
    B = J4 @ H
    A0 = rng.normal(size=(4, 4)); C0 = A0 @ A0.T + np.eye(4)
    C1 = rng.normal(size=(4, 4)); C1 = C1 + C1.T
    Cf = lambda t: C0 + t * C1 + t * t * np.eye(4)
    Cdf = lambda t: C1 + 2 * t * np.eye(4)
    Ctil = lambda t: expm(-t * B) @ Cf(t) @ expm(-t * B).T
    err = 0
    for t in (0.1, 0.3, 0.6):
        h = 1e-6
        lhs = (Ctil(t + h) - Ctil(t - h)) / (2 * h)
        G = Cdf(t) - B @ Cf(t) - Cf(t) @ B.T
        rhs = expm(-t * B) @ G @ expm(-t * B).T
        err = max(err, np.abs(lhs - rhs).max() / np.abs(rhs).max())
    check("(a) pull-back identity dCtil/dt = M^-1 G M^-T (Hamiltonian B, d = 4, arbitrary SPD path)",
          err < 1e-6 and abs(np.linalg.det(expm(0.6 * B)) - 1) < 1e-9,
          f"max relative deviation = {err:.1e}, det M - 1 = {np.linalg.det(expm(0.6 * B)) - 1:.1e}")

    # (b) S_C(t) = h(X0 + N(0, Ctil)) for a non-Gaussian state, d = 2
    Z, dA = grid2()
    mix = Mixture([0.55, 0.45], [np.array([1.2, -0.4]), np.array([-1.0, 0.9])],
                  [np.array([[0.30, 0.12], [0.12, 0.20]]), np.array([[0.15, -0.05], [-0.05, 0.40]])])
    B2 = J2 @ np.diag([2.3, 1.0])
    C2 = lambda t: np.array([[0.20, 0.05], [0.05, 0.12]]) + np.array([[0.04, 0.01], [0.01, 0.07]]) * t
    dev = 0
    for t in (0.4, 1.1):
        M = expm(B2 * t)
        Ct = np.linalg.inv(M) @ C2(t) @ np.linalg.inv(M).T
        dev = max(dev, abs(entropy(Z, dA, mix.at(B2, t, C2(t))) - entropy(Z, dA, mix.at(B2, 0.0, Ct))))
    check("(b) S_C(t) = h(X_0 + N(0, Ctil(t))) for a two-component mixture (oscillator w^2 = 2.3)",
          dev < 1e-8, f"max |difference| = {dev:.1e}")

    # (c) finite-time criterion, sufficiency on random mixtures
    B2 = J2 @ np.diag([2.0, 1.0])
    t1, t2 = 0.2, 0.9
    M1, M2 = expm(B2 * t1), expm(B2 * t2)
    Ct1 = np.diag([0.10, 0.15])
    Ct2 = Ct1 + np.array([[0.05, 0.02], [0.02, 0.03]])             # Ctil(t2) >= Ctil(t1)
    C_at1, C_at2 = M1 @ Ct1 @ M1.T, M2 @ Ct2 @ M2.T
    gaps = []
    for _ in range(20):
        k = rng.integers(1, 4)
        ws = rng.dirichlet(np.ones(k)); mus = [rng.normal(0, 1.2, 2) for _ in range(k)]
        Ss = []
        for _ in range(k):
            A = rng.normal(0, 0.5, (2, 2)); Ss.append(A @ A.T + 0.005 * np.eye(2))
        m = Mixture(ws, mus, Ss)
        gaps.append(entropy(Z, dA, m.at(B2, t2, C_at2)) - entropy(Z, dA, m.at(B2, t1, C_at1)))
    check("(c) sufficiency: Ctil(t2) >= Ctil(t1) gives S_C(t2) >= S_C(t1) for 20 random mixtures",
          min(gaps) > -1e-8, f"min S_C(t2) - S_C(t1) = {min(gaps):.2e}")

    # (c) necessity: Ctil(t2) not >= Ctil(t1) -> a Gaussian state loses entropy
    Ct2b = Ct1 + np.array([[0.08, 0.06], [0.06, 0.02]])            # indefinite difference
    K = np.linalg.inv(sqrtm(Ct1)) @ Ct2b @ np.linalg.inv(sqrtm(Ct1))
    lam, U = np.linalg.eigh(K); u = U[:, 0]
    eps, Lam = 1e-6, 1e6
    S0 = sqrtm(Ct1) @ (eps * np.outer(u, u) + Lam * (np.eye(2) - np.outer(u, u))) @ sqrtm(Ct1)
    dS = gauss_entropy(S0 + Ct2b) - gauss_entropy(S0 + Ct1)        # entropies via the pull-back
    check("(c) necessity: if Ctil(t2) - Ctil(t1) has a negative eigenvalue, a Gaussian state has S_C(t2) < S_C(t1)",
          lam[0] < 1 and dS < 0 and abs(dS - 0.5 * np.log(lam[0])) < 1e-4,
          f"lambda_min(Ctil1^-1/2 Ctil2 Ctil1^-1/2) = {lam[0]:.4f}, Delta S = {dS:.4f} -> 1/2 ln lambda_min = {0.5*np.log(lam[0]):.4f}")

    # (d) finite-time weaker than pointwise: sigma(t) with a dip in the rate
    B2 = J2 @ np.diag([4.0, 1.0])                                  # w = 2, kappa = 1.5
    kap = 0.5 * np.max(np.linalg.eigvalsh(B2 + B2.T))
    rate = lambda t: 2.4 if (t < 0.3 or t > 0.5) else 0.6           # dips below kappa on (0.3, 0.5)
    ts = np.linspace(0, 1, 2001)
    lnsig = np.concatenate([[0], np.cumsum([rate(t) * (ts[1] - ts[0]) for t in ts[:-1]])]) + np.log(0.3)
    sig = lambda t: np.exp(np.interp(t, ts, lnsig))
    Gmin = min(np.min(np.linalg.eigvalsh(2 * sig(t) ** 2 * (rate(t) * np.eye(2) - 0.5 * (B2 + B2.T)))) for t in ts)
    Ctf = lambda t: expm(-B2 * t) @ (sig(t) ** 2 * np.eye(2)) @ expm(-B2 * t).T
    gap = np.min(np.linalg.eigvalsh(Ctf(1.0) - Ctf(0.0)))
    check("(d) G has a negative eigenvalue on (0.3, 0.5), yet Ctil(1) >= Ctil(0): no state loses entropy over [0, 1]",
          Gmin < 0 and gap > 0, f"min eigenvalue of G = {Gmin:.3f}, lambda_min(Ctil(1) - Ctil(0)) = {gap:.3f}, kappa = {kap}")

    # (e) general linear drift, tr B != 0
    Bd = np.array([[-0.3, 1.0], [-2.0, -0.5]])
    C2 = lambda t: np.array([[0.20, 0.05], [0.05, 0.12]]) + np.array([[0.04, 0.01], [0.01, 0.07]]) * t
    t, h = 0.6, 1e-4
    comps = mix.at(Bd, t, C2(t))
    G = np.array([[0.04, 0.01], [0.01, 0.07]]) - Bd @ C2(t) - C2(t) @ Bd.T
    pred = np.trace(Bd) + 0.5 * np.trace(G @ fisher(Z, dA, comps))
    num = (entropy(Z, dA, mix.at(Bd, t + h, C2(t + h))) - entropy(Z, dA, mix.at(Bd, t - h, C2(t - h)))) / (2 * h)
    check("(e) non-volume-preserving linear drift: dS_C/dt = tr B + 1/2 tr(G F)",
          abs(num - pred) / abs(pred) < 1e-5, f"numeric {num:.6f}, formula {pred:.6f}")

    # (f) realizability: Lyapunov equation with Q = G reproduces C(t)
    B2 = J2 @ np.diag([2.0, 1.0])
    C0 = np.array([[0.3, 0.05], [0.05, 0.2]])
    Cf = lambda t: C0 + t * np.array([[1.5, 0.2], [0.2, 1.2]]) + t ** 2 * np.eye(2)
    Cdf = lambda t: np.array([[1.5, 0.2], [0.2, 1.2]]) + 2 * t * np.eye(2)
    Gf = lambda t: Cdf(t) - B2 @ Cf(t) - Cf(t) @ B2.T
    psd = min(np.min(np.linalg.eigvalsh(Gf(t))) for t in np.linspace(0, 1, 201))
    sol = solve_ivp(lambda t, y: (B2 @ y.reshape(2, 2) + y.reshape(2, 2) @ B2.T + Gf(t)).ravel(),
                    [0, 1], C0.ravel(), rtol=1e-11, atol=1e-13)
    errC = np.abs(sol.y[:, -1].reshape(2, 2) - Cf(1.0)).max()
    L = np.real(sqrtm(Gf(0.5)))
    check("(f) G >= 0 on [0,1]; the SDE dZ = BZ dt + G^(1/2) dW has covariance C(t) (Lyapunov equation)",
          psd >= 0 and errC < 1e-8 and np.abs(L @ L.T - Gf(0.5)).max() < 1e-10,
          f"min eigenvalue of G = {psd:.3f}, |C_SDE(1) - C(1)| = {errC:.1e}")

    # (g) scalar drift: G = -sigma^2 d ln SNR/dt
    al = lambda t: np.exp(-t - 0.3 * t ** 2)
    s2 = lambda t: 1 - np.exp(-2 * t) + 0.2 * np.sin(4 * t) ** 2
    dev = 0
    for t in (0.2, 0.7, 1.3):
        h = 1e-6
        b = (np.log(al(t + h)) - np.log(al(t - h))) / (2 * h)
        G = (s2(t + h) - s2(t - h)) / (2 * h) - 2 * b * s2(t)
        dlnsnr = (np.log(al(t + h) ** 2 / s2(t + h)) - np.log(al(t - h) ** 2 / s2(t - h))) / (2 * h)
        dev = max(dev, abs(G + s2(t) * dlnsnr))
    check("(g) scalar drift B = (d ln alpha/dt) I: G = -sigma^2 (d ln SNR/dt) I, so G >= 0 <=> SNR non-increasing",
          dev < 1e-6, f"max |G + sigma^2 d ln SNR/dt| = {dev:.1e}")

    # (h) Hamiltonian drift, isotropic schedule: threshold for realizability
    B2 = J2 @ np.diag([4.0, 1.0])
    kap = 0.5 * np.max(np.linalg.eigvalsh(B2 + B2.T))
    gm = lambda r: np.min(np.linalg.eigvalsh(2 * (r * np.eye(2) - 0.5 * (B2 + B2.T))))   # sigma = 1
    check("(h) isotropic schedule, oscillator w = 2: realizable by a linear SDE iff sigma_dot/sigma >= kappa = 1.5",
          abs(kap - 1.5) < 1e-12 and gm(1.4) < 0 and abs(gm(1.5)) < 1e-12 and gm(1.6) > 0,
          f"lambda_min(G)/sigma^2 at r = 1.4, 1.5, 1.6: {gm(1.4):.2f}, {gm(1.5):.2f}, {gm(1.6):.2f}")

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return sum(results), len(results)


if __name__ == "__main__":
    run()
