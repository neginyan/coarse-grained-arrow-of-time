# Sharp criterion for the coarse-grained arrow of time in Hamiltonian dynamics

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23135448.svg)](https://doi.org/10.5281/zenodo.23135448)
[![Verification](https://github.com/neginyan/coarse-grained-arrow-of-time/actions/workflows/verify.yml/badge.svg)](https://github.com/neginyan/coarse-grained-arrow-of-time/actions/workflows/verify.yml)

Verification code for the manuscript

> T. Namba, *Sharp criterion for the coarse-grained arrow of time in
> Hamiltonian dynamics*, preprint (2026),
> [doi:10.5281/zenodo.23135743](https://doi.org/10.5281/zenodo.23135743).

Every analytical statement, number and figure of the paper is checked by the
scripts in this repository.

```
pip install -r requirements.txt
cd verification
python run_all.py          # 33 checks, about 90 seconds
python make_figures.py     # ../figures/
```

The full output is in [`results/verification_output.txt`](results/verification_output.txt).

---

## The paper in brief

The Gibbs entropy of a Hamiltonian ensemble is constant; the growth of entropy
is usually attributed to coarse-graining. The paper asks when Gaussian
coarse-graining with covariance $C(t)$ actually guarantees that the
coarse-grained entropy $S_C$ does not decrease, **for every initial ensemble**.

1. **Entropy production (Sec. II).** $dS_C/dt = \tfrac12\mathrm{tr}(\dot C F) + \Sigma$,
   with $F$ the Fisher information matrix of the smoothed density and $\Sigma$ a
   dynamical term (Lemmas 1–2).
2. **Linear flows (Sec. III).** For $\dot z = Bz$: $dS_C/dt = \tfrac12\mathrm{tr}(GF)$ with
   $G = \dot C - BC - CB^{\mathsf T}$, and $S_C$ is non-decreasing for every
   initial density **if and only if** $G \succeq 0$ (Theorem 3). For isotropic
   resolution: $\dot\sigma/\sigma \ge \lambda_{\max}(\mathrm{Sym}\,B)$ (Corollary 4).
   At fixed resolution there is no arrow of time (Corollary 5).
3. **Nonlinear flows (Sec. IV).** With $\kappa = \sup_z\lambda_{\max}[\mathrm{Sym}\,Df(z)]$:
   $\Sigma \ge -\kappa(d - \sigma^2\mathrm{tr}F)$ (Theorem 6), which proves the arrow
   for $\dot\sigma/\sigma \ge \kappa$ and every state with $\sigma^2\mathrm{tr}F \ge d/2$
   (Corollary 7); $\kappa$ cannot be lowered (Proposition 8); the criterion holds
   to leading order as $\sigma \to 0$ (Theorem 9). The general case is
   Conjecture 1.
4. **Geometric picture and diffusion models (Appendix A).** Pulled back along the
   flow, $\tilde C = M^{-1}CM^{-\mathsf T}$ obeys $\dot{\tilde C} = M^{-1}GM^{-\mathsf T}$, so
   $G \succeq 0$ means that the window seen from the initial state only grows; over a
   finite interval the arrow holds for every state iff $\tilde C(t_2) \succeq \tilde C(t_1)$
   (Proposition 10). For the linear forward SDE of a diffusion model, $G = LL^{\mathsf T}$:
   a covariance schedule is realizable iff $G \succeq 0$, which is the monotone-SNR
   condition for scalar drift and $\dot\sigma/\sigma \ge \lambda_{\max}(\mathrm{Sym}\,B)$
   for an isotropic schedule with Hamiltonian drift.
5. **Examples (Sec. V).** Pendulum ($\kappa = 1$): random and adversarial tests of
   Conjecture 1. Quartic oscillator ($\kappa = \infty$): no finite coarse-graining
   rate yields an arrow of time for all states.

---

## Where each result of the paper is checked

| Paper | Content | Script | Checks |
|---|---|---|---|
| Lemma 1, Eq. (4); Theorem 3(i), Eqs. (8)–(10) | $dS_C/dt = \tfrac12\mathrm{tr}(GF)$ for non-Gaussian states, anisotropic $C(t)$; coupled oscillators ($d = 4$) | `step1_linear.py` | (A,B), (n = 2) |
| Theorem 3(ii), Eq. (11) | $G \succeq 0$ sufficient (25 random states) and necessary (explicit counterexample) | `step1_linear.py` | (C) ×2 |
| Corollary 4, Eqs. (12)–(13) | threshold $\lvert 1-\omega^2\rvert/2$; free particle $\dot\sigma \ge \sigma/2$ | `step1_linear.py` | (D) ×3 |
| Corollary 5, Fig. 1(a) | fixed resolution: constant (matched) or oscillating (elliptic) | `step1_linear.py` | (E) ×2 |
| Sec. V (method) | exact evaluation of $\Sigma$, $\mathrm{tr}F$ agrees with time-stepping along the exact flow | `step2_quartic.py` | (a), (b) |
| Sec. V B, Fig. 3 | quartic: $R(q_0 = 4) = 22.7$ vs strain 23.5; no finite rate suffices | `step2_quartic.py` | (c), (d) |
| Theorem 6, Eq. (17); Corollary 7 | bound on 200 random pendulum states; arrow for $\sigma^2\mathrm{tr}F \ge 1$ | `step3_nonlinear_bound.py` | Theorem, Corollary |
| Sec. V A (proven statements) | bound attained at $x = 1$ and $x = 2$; Proposition 8 at $\dot\sigma/\sigma = 0.9$ | `step3_nonlinear_bound.py` | Tightness ×2, Necessity |
| Sec. V A, Fig. 2 | Conjecture 1 on 200 random states (margin 0.020) | `step3_nonlinear_bound.py` | Conjecture evidence |
| Theorem 9, Eqs. (18)–(19) | limits 12.726 and 9.656, error $\propto\sigma^2$ | `step4_small_sigma.py` | 2 checks |
| Sec. V A, Fig. 2 | adversarial optima: all $R < 1$; quoted maxima | `step5_search.py` | (a)–(c) |
| Appendix A, Eqs. (A1)–(A3) | pull-back identity; $S_C(t) = h(X_0 + \tilde N_t)$ | `step6_geometric.py` | (a), (b) |
| Appendix A, Proposition 10 | finite-time criterion: sufficiency, necessity, weaker than pointwise $G \succeq 0$ | `step6_geometric.py` | (c) ×2, (d) |
| Appendix A, Eqs. (A4)–(A5) | $dS_C/dt = \mathrm{tr}\,B + \tfrac12\mathrm{tr}(GF)$; realizability by a linear SDE; monotone SNR; isotropic threshold | `step6_geometric.py` | (e)–(h) |
| Figs. 1–3 | figures | `make_figures.py` | — |

The adversarial searches themselves (`search_safe.py`, about one hour each) store
their optima in `results/search_sigma_*.json`; `step5_search.py` re-evaluates
every stored optimum on a grid twice as fine.

## Method

For Gaussian-mixture initial densities the conditional law of $X_t$ given the
coarse-grained variable $Y$ is Gaussian per component, so the coarse-grained
velocity $E[f(X_t)\mid Y]$ is analytic for the pendulum ($-\sin q$) and the quartic
oscillator ($-q^3$). `fast_eval.py` uses this to evaluate $\Sigma$ and
$\mathrm{tr}F$ with a single quadrature over $y$; `step2_quartic.py` checks it
against direct time differentiation of $S_C$ along the exact nonlinear flow.

## What is proven and what is conjectured

| Statement | Status |
|---|---|
| Linear flows: $G \succeq 0$ iff arrow for all states | Theorem 3 (proved) |
| Nonlinear: $\Sigma \ge -\kappa(d - \sigma^2\mathrm{tr}F)$ | Theorem 6 (proved) |
| Arrow for $\sigma^2\mathrm{tr}F \ge d/2$ when $\dot\sigma/\sigma \ge \kappa$ | Corollary 7 (proved) |
| $\kappa$ cannot be lowered | Proposition 8 (proved; needs $\lvert D^2 f\rvert$ of polynomial growth) |
| Criterion to leading order as $\sigma \to 0$ | Theorem 9 (proved under a stated continuity assumption) |
| Finite times: arrow for all states iff $\tilde C(t_2) \succeq \tilde C(t_1)$ | Proposition 10 (proved) |
| Linear diffusion models: schedule realizable iff $G \succeq 0$ | Appendix A (proved) |
| Arrow for **all** states when $\dot\sigma/\sigma \ge \kappa$ | Conjecture 1 (numerical evidence only) |

## Repository layout

```
verification/
  fast_eval.py              exact dS_C/dt for Gaussian mixtures (pendulum, quartic, harmonic)
  step1_linear.py           Sec. III
  step2_quartic.py          Sec. V B and method validation
  step3_nonlinear_bound.py  Theorem 6, Corollary 7, Proposition 8, Conjecture 1 (random states)
  step4_small_sigma.py      Theorem 9
  step5_search.py           re-check of the adversarial search
  step6_geometric.py        Appendix A: pull-back picture, finite times, diffusion models
  search_safe.py            the adversarial search (slow)
  run_all.py                runs steps 1-6; exit code 0 if all checks pass
  make_figures.py           Figs. 1-3
figures/                    PDF figures of the paper
results/                    verification output and stored search optima
```

## How to cite

If you use this code, please cite the paper,

> T. Namba, *Sharp criterion for the coarse-grained arrow of time in Hamiltonian
> dynamics*, preprint (2026), [doi:10.5281/zenodo.23135743](https://doi.org/10.5281/zenodo.23135743),

and the archived version of the code:

> T. Namba, *coarse-grained-arrow-of-time: verification code for "Sharp criterion
> for the coarse-grained arrow of time in Hamiltonian dynamics"*, version v1.0.0,
> Zenodo (2026), [doi:10.5281/zenodo.23135448](https://doi.org/10.5281/zenodo.23135448).

```bibtex
@software{namba2026arrowcode,
  author    = {Namba, Taishi},
  title     = {coarse-grained-arrow-of-time: verification code for
               ``Sharp criterion for the coarse-grained arrow of time
               in Hamiltonian dynamics''},
  version   = {v1.0.0},
  publisher = {Zenodo},
  year      = {2026},
  doi       = {10.5281/zenodo.23135448},
  url       = {https://doi.org/10.5281/zenodo.23135448}
}
```

## Acknowledgment

The analysis and code were developed with the assistance of AI tools
(Google Gemini, OpenAI ChatGPT (GPT-5.6 Luna) and Anthropic Claude). Every result stated above is checked by the scripts in this
repository.

## License

MIT; see [LICENSE](LICENSE).
