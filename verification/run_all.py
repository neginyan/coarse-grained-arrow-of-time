"""Run every verification script; exit code 0 if all checks pass."""
import io, sys, time, contextlib
import step1_linear, step2_quartic, step3_nonlinear_bound, step4_small_sigma, step5_search

STEPS = [
    (step1_linear, "Step 1  Linear flows: Theorem 3, Corollaries 4-5 (Sec. III)"),
    (step2_quartic, "Step 2  Quartic oscillator and method validation (Sec. V B)"),
    (step3_nonlinear_bound, "Step 3  Nonlinear bound, Corollary 7, Proposition 8 (Secs. IV, V A)"),
    (step4_small_sigma, "Step 4  Small-resolution limit, Theorem 9 (Secs. IV, V A)"),
    (step5_search, "Step 5  Adversarial search for Conjecture 1 (Sec. V A)"),
]

if __name__ == "__main__":
    t0 = time.time(); tot = ok = 0; lines = []
    for mod, title in STEPS:
        print("=" * 78 + f"\n{title}\n" + "=" * 78, flush=True)
        p, t = mod.run()
        ok += p; tot += t
        lines.append(f"  {'OK  ' if p == t else 'FAIL'}  {p:2d}/{t:<2d}  {title}")
    print("\n" + "#" * 78 + "\nSUMMARY\n" + "#" * 78)
    print("\n".join(lines))
    print(f"\n  {ok}/{tot} checks passed in {time.time() - t0:.0f} s")
    sys.exit(0 if ok == tot else 1)
