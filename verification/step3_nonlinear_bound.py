"""Step 3 -- nonlinear Hamiltonian flows: a rigorous lower bound and the pendulum conjecture.

Theorem (proved in the notes):  for any divergence-free f on R^d with
    inf_z lambda_min(sym Df(z)) >= -kappa,
and isotropic coarse-graining C = sigma^2 I,
    Sigma = d/dt h(X_t + N)  >=  -kappa (d - sigma^2 tr F).
Proof chain: Sigma = E[div f_eff(Y)],  div f_eff = tr Cov(f(X), X | Y)/sigma^2,
tr Cov(f(X), X | Y) >= -kappa tr Cov(X | Y),  E tr Cov(X|Y) = d sigma^2 - sigma^4 tr F (Tweedie).
Consequence: with sigma_dot/sigma = r >= kappa, dS/dt >= 0 for every state with sigma^2 tr F >= d/2.
Conjecture (open): Sigma >= -kappa sigma^2 tr F for every state (true for linear flows, Step 1).
Pendulum H = p^2/2 - cos q: kappa = 1, d = 2.
"""
import numpy as np
import fast_eval as fe

res = []


def check(name, ok, detail):
    res.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}\n         {detail}")


def random_state(rng):
    sig = 10 ** rng.uniform(-1, 0.3)
    K = rng.integers(1, 4)
    ws = rng.dirichlet(np.ones(K))
    mus = [np.array([rng.uniform(-4, 4), rng.normal(0, 1.5)]) for _ in range(K)]
    Ss = []
    for _ in range(K):
        A = rng.normal(0, 1, (2, 2)) * 10 ** rng.uniform(-1.5, 0.3)
        Ss.append(A @ A.T + 1e-6 * np.eye(2))
    span = max(np.sqrt(np.linalg.eigvalsh(S + sig ** 2 * np.eye(2)).max()) for S in Ss)
    n = int(min(1500, max(301, 16 * (8 * span + 8) / sig)))
    return ws, mus, Ss, sig, n


def run():
    rng = np.random.default_rng(7)
    thm, conj, half = [], [], []
    for _ in range(200):
        ws, mus, Ss, sig, n = random_state(rng)
        _, trF, Sg = fe.dSdt(ws, mus, Ss, sig, 0, 'pendulum', n=n)
        x = sig ** 2 * trF
        thm.append(Sg + (2 - x))
        conj.append(Sg + x)
        if x >= 1:
            half.append(x + Sg)            # dS/dt at r = kappa = 1
    check("Theorem: Sigma >= -kappa (d - sigma^2 tr F) for 200 random Gaussian-mixture states (pendulum)",
          min(thm) > -1e-6, f"min margin = {min(thm):.2e}")
    check("Corollary: at sigma_dot/sigma = kappa, every state with sigma^2 tr F >= d/2 has dS/dt >= 0",
          min(half) >= -1e-6, f"{len(half)} such states, min dS/dt = {min(half):.3e}")

    # tightness of the theorem for the linearised flow (narrow along the stretching direction)
    for q0 in (np.pi,):
        sig = 0.002
        v = np.array([1, 1]) / np.sqrt(2)             # stretching eigenvector of sym Df at q = pi
        w = np.array([1, -1]) / np.sqrt(2)
        S = 1e-12 * np.outer(v, v) + 1e-12 * np.outer(w, w)
        _, trF, Sg = fe.dSdt([1], [np.array([q0, 0])], [1e-12 * np.outer(v, v) + 2.5e-3 * np.outer(w, w)], sig, 0, 'pendulum', n=1601)
        x = sig ** 2 * trF
        check("Tightness: narrow along the stretching direction at q = pi, sigma -> 0: Sigma -> -kappa (d - sigma^2 tr F)",
              abs(Sg + (2 - x)) < 0.02 and abs(Sg + 1) < 0.02, f"Sigma = {Sg:.4f}, -kappa(d - x) = {-(2 - x):.4f}, sigma^2 trF = {x:.4f}")
        _, trF, Sg = fe.dSdt([1], [np.array([q0, 0])], [S], sig, 0, 'pendulum', n=801)
        check("... and a state sharp in both directions gives Sigma ~ 0 (bound -kappa(d - 2) = 0 also tight)",
              abs(Sg) < 1e-3, f"Sigma = {Sg:.2e}, sigma^2 trF = {sig**2*trF:.4f}")

    # asymptotic necessity: r < kappa -> some state has dS/dt < 0 (sigma small)
    sig, r = 0.002, 0.9
    v = np.array([1, 1]) / np.sqrt(2); w = np.array([1, -1]) / np.sqrt(2)
    d, _, _ = fe.dSdt([1], [np.array([np.pi, 0])], [1e-12 * np.outer(v, v) + 2.5e-3 * np.outer(w, w)], sig, r, 'pendulum', n=1601)
    check("Necessity (sigma -> 0): at sigma_dot/sigma = 0.9 < kappa a state at the hyperbolic point has dS/dt < 0",
          d < 0, f"dS/dt = {d:.4f}")

    # conjecture: evidence
    check("Conjecture evidence: Sigma >= -kappa sigma^2 tr F on all 200 random states",
          min(conj) > -1e-6, f"min margin = {min(conj):.3e}")
    print(f"\n{sum(res)}/{len(res)} checks passed")
    return sum(res), len(res)


if __name__ == "__main__":
    p, t = run()
    raise SystemExit(0 if p == t else 1)
