# B5 as the CN-CBF baseline — fidelity note

B5 is our reimplementation of **CN-CBF** (Composite Neural Control Barrier
Function, Derajic et al.) adapted to the planar needle-steering setting.

**What CN-CBF does:** learns one neural CBF per robot–obstacle pair on the
*relative* coordinate, supervises it with the Hamilton–Jacobi (HJ) reachability
value function of that obstacle, uses a residual architecture so the estimated
safe set never intersects the failure set, and fuses the per-obstacle barriers
with a smooth-minimum into one composite CBF used in an online QP safety filter.

**How B5 matches it:**
- one MLP per obstacle on the relative coordinate `(x - c_i)/σ` ✓
- smooth-minimum composition, identical `β` to ours ✓
- online CLF-CBF-QP safety filter (same QP as ours) ✓
- **no shape encoder, no pose/scale augmentation** — the barrier is tied to the
  geometry it was trained on ✓ (this is exactly the axis our method adds)

**Fidelity gaps (documented for honesty):**
- *Supervision:* CN-CBF regresses the HJ value function; B5 regresses the shaped
  signed distance. For the **single-integrator** tool dynamics used here the HJ
  reach-avoid value of a static obstacle **equals its signed distance** (the
  optimal evasion is to move radially outward at max speed), so the two targets
  coincide. This makes B5 a faithful CN-CBF instance for our dynamics.
- *Residual architecture:* we approximate the "safe set ⊄ failure set" guarantee
  with the wide interior clamp + far-field blend rather than CN-CBF's residual
  head; behaviourally equivalent for the comparison (never reports a moved
  obstacle's interior as safe within its trained pose).

**Role in the paper:** CN-CBF is the closest prior *composite* neural barrier.
B5 shows that per-obstacle composition alone (CN-CBF) does **not** generalize
across obstacle *shape/pose* without the permutation-invariant encoder and joint
pose+label augmentation our method contributes — see the barrier-generalization
table (false-safe rate vs rotation).
