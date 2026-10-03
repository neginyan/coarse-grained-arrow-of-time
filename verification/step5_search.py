"""Step 5 -- re-check of the adversarial search for the pendulum conjecture (paper Sec. V A, Fig. 2).

The Nelder-Mead searches (search_safe.py, about one hour) store their optima in results/. This script
re-evaluates every stored optimum on a grid twice as fine and checks:
 (a) every optimum satisfies Conjecture 1, R = -Sigma/(sigma^2 tr F) <= kappa = 1
 (b) the largest values quoted in the paper: R = 0.908 (sigma = 0.1), 0.764 (0.3), 0.527 (1.0)
 (c) every optimum also satisfies Theorem 6 (sanity check of the stored states)
"""
import glob, json, os
import numpy as np
import fast_eval as fe
import search_safe
from search_safe import unpack, grid_n

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
res = []


def check(name, ok, detail):
    res.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}\n         {detail}")


def optima():
    """List of (sigma, x = sigma^2 tr F, R) for all stored optima."""
    out = []
    for fn in sorted(glob.glob(os.path.join(RES, "search_sigma_*.json"))):
        sig = float(os.path.basename(fn).split("_")[2])
        for b in json.load(open(fn)):
            search_safe.SMAX = b.get("SMAX", 1.0)
            ws, mus, Ss = unpack(np.array(b["x"]), b["K"])
            _, trF, Sg = fe.dSdt(ws, mus, Ss, sig, 0, 'pendulum', n=grid_n(mus, Ss, sig, 2), pad=6.0)
            out.append((sig, sig * sig * trF, -Sg / (sig * sig * trF)))
    return out


def run():
    opt = optima()
    check("(a) every stored optimum satisfies Conjecture 1 (R <= 1)", max(r for _, _, r in opt) < 1,
          f"{len(opt)} optima, largest R = {max(r for _, _, r in opt):.4f}")
    best = {s: max(r for ss, _, r in opt if ss == s) for s in sorted({s for s, _, _ in opt})}
    quoted = {0.1: 0.908, 0.3: 0.764, 1.0: 0.527}
    check("(b) largest R per sigma as quoted in Sec. V A",
          all(round(best[s], 3) == v for s, v in quoted.items()),
          ", ".join(f"sigma={s:g}: {r:.4f}" for s, r in best.items()))
    check("(c) every optimum satisfies Theorem 6, R <= (2 - x)/x", all(r <= (2 - x) / x + 1e-9 for _, x, r in opt), "")
    print(f"\n{sum(res)}/{len(res)} checks passed")
    return sum(res), len(res)


if __name__ == "__main__":
    p, t = run()
    raise SystemExit(0 if p == t else 1)
