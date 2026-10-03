"""Step 2 -- quartic oscillator H = p^2/2 + q^4/4 (unbounded strain rate) and validation of fast_eval.

Paper: Sec. V B and Fig. 3; the validation supports the method statement at the start of Sec. V.
Checks:
 (a,b) fast_eval (exact, no time stepping) agrees with direct time differentiation of S_C along the
       exact nonlinear flow (quartic and pendulum)
 (c)   R(q0) = -Sigma/(sigma^2 tr F) for a localised state follows the local strain rate |1-3q0^2|/2:
       R = 22.7 at q0 = 4 where the strain rate is 23.5  (sigma = 0.05)
 (d)   R(q0) grows without bound: for every rate r in {1, 3, 10} some localised state has dS/dt < 0
"""
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.integrate import solve_ivp
import fast_eval as fe

FORCE = {'quartic': lambda q: -q ** 3, 'pendulum': lambda q: -np.sin(q)}


def flow(pts, t, kind):
    if t == 0:
        return pts
    n = pts.shape[0]
    F = FORCE[kind]
    rhs = lambda _, y: np.concatenate([y[n:], F(y[:n])])
    y = solve_ivp(rhs, (0, t), pts.T.ravel(), method='DOP853', rtol=1e-12, atol=1e-13).y[:, -1]
    return np.stack([y[:n], y[n:]], 1)


def dSdt_timestep(mu, S, sig, r, kind, k=60, h=2e-4, L=2.5, n=301):
    """Central difference of S_C(t) with the Gaussian state represented by Gauss-Hermite nodes
    pushed by the exact flow (nodes must be dense compared with sigma)."""
    x, w = hermegauss(k); w = w / w.sum()
    X, Y = np.meshgrid(x, x, indexing='ij')
    U = np.stack([X.ravel(), Y.ravel()], 1); W = np.outer(w, w).ravel()
    p0 = mu + U @ np.linalg.cholesky(S).T
    g = np.linspace(-L, L, n)
    Zq, Zp = np.meshgrid(mu[0] + g, mu[1] + g, indexing='ij'); dA = (g[1] - g[0]) ** 2

    def S_C(pts, s):
        rho = np.zeros_like(Zq)
        for (a, b), ww in zip(pts, W):
            rho += ww * np.exp(-((Zq - a) ** 2 + (Zp - b) ** 2) / (2 * s * s))
        rho = np.maximum(rho / (2 * np.pi * s * s), 1e-300)
        return -np.sum(rho * np.log(rho)) * dA
    return (S_C(flow(p0, h, kind), sig * np.exp(r * h)) - S_C(flow(p0, -h, kind), sig * np.exp(-r * h))) / (2 * h)


def localised_state(q0, sig, kind, narrow=0.01, wide=50.0):
    a = (1 - 3 * q0 * q0) / 2 if kind == 'quartic' else (1 - np.cos(q0)) / 2
    _, V = np.linalg.eigh(np.array([[0, a], [a, 0]]))
    v, w = V[:, 1], V[:, 0]
    return sig ** 2 * (narrow * np.outer(v, v) + wide * np.outer(w, w)), abs(a)


def quartic_ratio(q0, sig=0.05):
    S, lam = localised_state(q0, sig, 'quartic')
    _, trF, Sg = fe.dSdt([1.0], [np.array([q0, 0.0])], [S], sig, 0.0, 'quartic', n=801)
    lin = lam * (1 / 1.01 - 1 / 51) / (1 / 1.01 + 1 / 51)
    return -Sg / (sig ** 2 * trF), lin, lam


res = []


def check(name, ok, detail):
    res.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}\n         {detail}")


def run():
    for kind, mu, S, sig, r in (('quartic', np.array([2.0, 0.0]), np.array([[0.0105, 0.0095], [0.0095, 0.0105]]), 0.15, 1.0),
                                ('pendulum', np.array([2.5, 0.4]), np.array([[0.03, 0.01], [0.01, 0.02]]), 0.15, 0.4)):
        a = fe.dSdt([1.0], [mu], [S], sig, r, kind, n=601)[0]
        b = dSdt_timestep(mu, S, sig, r, kind)
        check(f"({'a' if kind == 'quartic' else 'b'}) {kind}: fast_eval = time derivative of S_C along the exact flow",
              abs(a - b) < 2e-4 * max(1, abs(a)), f"fast_eval {a:.6f}, time-stepped {b:.6f}")
    out = {q: quartic_ratio(q) for q in (1.0, 2.0, 3.0, 4.0)}
    R4, lin4, lam4 = out[4.0]
    check("(c) quartic, sigma = 0.05: R(q0 = 4) = 22.7 against local strain rate 23.5; follows the linearised flow",
          round(R4, 1) == 22.7 and lam4 == 23.5 and abs(R4 - lin4) < 0.15,
          ", ".join(f"q0={q:g}: R={v[0]:.3f} (lin {v[1]:.3f}, strain {v[2]:.3f})" for q, v in out.items()))
    worst = []
    for r in (1.0, 3.0, 10.0):
        q0 = next(q for q in np.arange(0, 6, 0.25) if quartic_ratio(q)[0] > r * 1.05)
        S, _ = localised_state(q0, 0.05, 'quartic')
        d = fe.dSdt([1.0], [np.array([q0, 0.0])], [S], 0.05, r, 'quartic', n=801)[0]
        worst.append((r, q0, d))
    check("(d) no finite rate suffices: for r = 1, 3, 10 a localised state has dS_C/dt < 0",
          all(d < 0 for _, _, d in worst), ", ".join(f"r={r:g}: q0={q:g}, dS/dt={d:.3f}" for r, q, d in worst))
    print(f"\n{sum(res)}/{len(res)} checks passed")
    return sum(res), len(res)


if __name__ == "__main__":
    p, t = run()
    raise SystemExit(0 if p == t else 1)
