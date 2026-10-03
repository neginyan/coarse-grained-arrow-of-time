"""Adversarial search (slow; results are stored in results/ and re-checked by step5_search.py).
Usage: python search_safe.py SIGMA TRIALS [SMAX]

Resolution-safe adversarial search for the pendulum conjecture R = -Sigma/(sigma^2 tr F) <= 1.
Grid spacing is kept below sigma/4 in every evaluation; the optimum is re-checked on a grid twice as fine."""
import sys, json, os
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import fast_eval as fe

MUMAX, PAD = 4.0, 6.0
SMAX = 1.0          # maximal component variance; set from the command line


def unpack(x, K):
    ws, mus, Ss = [], [], []
    for k in range(K):
        a = x[7 * k:7 * k + 7]
        ws.append(np.exp(np.clip(a[0], -5, 5)))
        mus.append(MUMAX * np.tanh(np.array([a[1], a[2]]) / MUMAX))
        l1, l2 = SMAX * expit(a[3]), SMAX * expit(a[4])
        th = a[5]
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        Ss.append(R @ np.diag([l1 + 1e-10, l2 + 1e-10]) @ R.T)
    ws = np.array(ws) / sum(ws)
    return ws, mus, Ss


def grid_n(mus, Ss, sig, fine=1):
    sd = max(np.sqrt(np.linalg.eigvalsh(S).max() + sig * sig) for S in Ss)
    span = max(np.ptp([m[i] for m in mus]) for i in (0, 1)) + 2 * PAD * sd
    return int(min(3000, max(201, fine * 4 * span / sig)))


def ratio(x, K, sig, fine=1):
    ws, mus, Ss = unpack(x, K)
    n = grid_n(mus, Ss, sig, fine)
    _, trF, Sg = fe.dSdt(ws, mus, Ss, sig, 0.0, 'pendulum', n=n, pad=PAD)
    return -Sg / (sig * sig * trF)


if __name__ == "__main__":
    sig = float(sys.argv[1]); trials = int(sys.argv[2])
    if len(sys.argv) > 3:
        SMAX = float(sys.argv[3])
    rng = np.random.default_rng(int(sig * 1000))
    best = []
    for K in (1, 2, 3):
        b = (-np.inf, None)
        for _ in range(trials):
            x0 = []
            for k in range(K):
                x0 += [0.0, rng.uniform(-3.5, 3.5), rng.normal(0, 1), rng.normal(0, 2), rng.normal(-2, 2), rng.uniform(0, np.pi), 0.0]
            r = minimize(lambda x: -ratio(x, K, sig), np.array(x0), method='Nelder-Mead',
                         options={'maxiter': 700, 'xatol': 1e-5, 'fatol': 1e-7})
            if -r.fun > b[0]:
                b = (-r.fun, r.x)
        chk = ratio(b[1], K, sig, fine=2)
        best.append({"K": K, "SMAX": SMAX, "R": float(b[0]), "R_fine": float(chk), "x": b[1].tolist()})
        print(f"sigma={sig} K={K}: R={b[0]:.5f}  (grid x2: {chk:.5f})", flush=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", f"search_sigma_{sig}_smax_{SMAX:g}.json")
    json.dump(best, open(out, "w"))
