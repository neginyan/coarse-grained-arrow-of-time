"""Step 4 -- small-resolution limit of the pendulum conjecture.

Exact identity (de Bruijn + Liouville): Sigma(sigma^2) = 1/2 int_0^{sigma^2} dI_u/dt du,
I_u = Fisher information of rho * N(0, u I), and for a divergence-free flow
    dI_0/dt = -2 E_rho[ s^T Sym(Df) s ],   s = grad log rho.
Hence for every fixed smooth state, as sigma -> 0,
    [Sigma + kappa sigma^2 tr F] / sigma^2  ->  E_rho[ s^T (kappa I - Sym Df) s ]  >= 0.
Checked here for the pendulum (kappa = 1) on Gaussian and two-component states.
"""
import numpy as np
import fast_eval as fe

res = []


def check(name, ok, detail):
    res.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}\n         {detail}")


def limit_value(ws, mus, Ss, n=801, pad=8):
    """E_rho[s^T (I - Sym Df) s] for the pendulum, by quadrature."""
    sd = [np.sqrt(np.diag(S)) for S in Ss]
    lo = np.min([m - pad * d for m, d in zip(mus, sd)], 0); hi = np.max([m + pad * d for m, d in zip(mus, sd)], 0)
    xq = np.linspace(lo[0], hi[0], n); xp = np.linspace(lo[1], hi[1], n)
    dA = (xq[1] - xq[0]) * (xp[1] - xp[0])
    Z = np.stack(np.meshgrid(xq, xp, indexing='ij'), -1)
    p = 0; g = 0
    for w, m, S in zip(ws, mus, Ss):
        Si = np.linalg.inv(S); d = Z - m
        pk = w * np.exp(-0.5 * np.einsum('...i,ij,...j->...', d, Si, d)) / (2 * np.pi * np.sqrt(np.linalg.det(S)))
        p = p + pk; g = g + pk[..., None] * (-(d @ Si.T))
    a = (1 - np.cos(Z[..., 0])) / 2                       # off-diagonal of Sym Df
    quad = g[..., 0] ** 2 + g[..., 1] ** 2 - 2 * a * g[..., 0] * g[..., 1]
    ok = p > 1e-300
    return np.sum(quad[ok] / p[ok]) * dA


def run():
    states = {
        "Gaussian at the hyperbolic point, tilted": ([1.0], [np.array([np.pi, 0.0])], [np.array([[0.5, 0.35], [0.35, 0.5]])]),
        "two-component mixture": ([0.6, 0.4], [np.array([2.5, 0.3]), np.array([3.6, -0.4])],
                                  [np.array([[0.3, 0.2], [0.2, 0.3]]), np.array([[0.2, -0.05], [-0.05, 0.4]])]),
    }
    for name, (ws, mus, Ss) in states.items():
        L = limit_value(ws, mus, Ss)
        vals = []
        for sig in (0.2, 0.1, 0.05, 0.025):
            _, trF, Sg = fe.dSdt(ws, mus, Ss, sig, 0, 'pendulum', n=801)
            vals.append((Sg + sig ** 2 * trF) / sig ** 2)
        err = [abs(v - L) for v in vals]
        rate = np.log(err[-2] / err[-1]) / np.log(2)
        check(f"{name}: [Sigma + kappa sigma^2 trF]/sigma^2 -> E[s^T(kappa I - Sym Df)s] > 0",
              L > 0 and err[-1] < 0.01 * L and rate > 1.5,
              "sigma = 0.2, 0.1, 0.05, 0.025: " + ", ".join(f"{v:.5f}" for v in vals)
              + f"  -> limit {L:.5f}; error ~ sigma^{rate:.2f}")
    print(f"\n{sum(res)}/{len(res)} checks passed")
    return sum(res), len(res)


if __name__ == "__main__":
    p, t = run()
    raise SystemExit(0 if p == t else 1)
