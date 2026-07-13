# Canonical benchmark — findings (READ BEFORE WRITING THE RESULTS SECTION)

**Benchmark:** 25 mm guaranteed clearance, 3–8 obstacles ×6 = 36 generalization
scenes + 10 perturbed nominal starts, all 10 methods on identical seeded scenes,
both backstop settings. 920 rollouts, 0 errors. Parallel == serial verified.
Source: `results/baselines/../canonical/benchmark_25mm.json`.

## Bottom line

The clean, unbiased results **do not support two of the three paper claims.**
Reported faithfully so the Results section is defensible, not surprised by a reviewer.

### 1. Safety (backstop ON): TRUE but NOT a differentiator
Every method — ours, oracle SDF, circle CBF, even APF — records **0/36**
penetrations with the geometric backstop on. The exact-geometry backstop
guarantees safety for *all* methods equally, so "zero penetrations" is a
property of the shared backstop, not of the learned barrier. It cannot, alone,
be the paper's evidence of contribution.

### 2. Augmentation ablation (barrier ALONE, backstop OFF): NO benefit here
Unsafe generalization scenes: **ours (augmented) 12/36 vs fixed-pose 11/36.**
Augmentation did not reduce penetrations on this benchmark. (Note: at the
tighter 15 mm / ≤6-obstacle suite it was 8 vs 15 — augmentation helped there.
So the benefit is *not robust* to scene distribution — a real weakness, not a
clean contribution.) The genuine positive signal: ours 12/36 vs **global
scene-frozen barrier 36/36** and **nominal-DS 36/36** — i.e. the composite
*conditional* structure generalizes far better than a monolithic barrier. But
that is CN-CBF's known idea, not the paper's novel augmentation claim.

### 3. Reach / task quality: WORST in class, and it's a real live-lock
Generalization reach (backstop ON): **ours 0.28**, vs fixed-pose 0.58, oracle
0.44, cloud-ESDF 0.56, circle 0.56, per-shape 0.56. Ours collapses to **0.00 at
7–8 obstacles** while every baseline holds 0.5–0.67. Confirmed NOT an artifact:
- 100% of failures are timeouts, not clean traps;
- doubling the horizon to 24 s changes nothing (0.28 → 0.28) → **live-lock**, not
  slowness. The tool oscillates in place.
Mechanism: ours' learned barrier disagrees with geometry enough that the
discrete projection + backstop fire **49% of steps** (jerk 966, vs oracle's 0%
backstop / jerk 133). The go-around/swirl guidance (tuned for a single head-on
obstacle) appears to push toward neighbouring obstacles in dense clutter →
deadlock. This is a tuning/robustness defect of the reactive controller.

### What IS clean and reportable
- **Nominal regime (paper Table I):** ours reach 1.0, 0/10 unsafe, tight 10 mm
  deviation. Solid — but baselines match it here.
- **Composite conditional >> global barrier** (12/36 vs 36/36 unsafe, barrier
  alone). Defensible.
- The **oracle-SDF / cloud-ESDF baselines are safe AND reach better than ours**,
  which directly challenges "why learn the barrier." The honest rebuttal is that
  cloud-ESDF here is given *clean interior* point clouds with known sign; a
  realistic surface-only, noisy cloud would break its sign estimate. Our current
  b3b does NOT stress that — so it currently looks too strong.

## Options (needs a decision before the Results section is written)
- **A. Reframe honestly** around the safe full system + composite-vs-global
  result; soften/drop the augmentation-generalization and quality claims; report
  reach as the disclosed limitation.
- **B. Fix the method** (real work): retune backstop/barrier agreement + swirl to
  kill the live-lock (target reach), and re-examine why augmentation doesn't help
  (target the ablation). Then re-benchmark.
- **C. Make b3b realistic** (surface-only, noisy, unknown sign) so the "why learn"
  foil is fair — this is the strongest single move to protect the thesis.
