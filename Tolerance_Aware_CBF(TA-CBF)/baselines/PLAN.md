# Baselines — Implementation Plan

Goal: controlled comparison against TA-CBF (ours). **Everything except the safety
mechanism is held fixed**: same nominal DS `f_θ` (final_model.pt), same CLF-QP
controller, same scenes (fixed seeds), same metrics. Each QP-based baseline is a
drop-in replacement for `model.B` (an `nn.Module` exposing `forward(x)->(B,1)`,
`gradient(x)->(B,d)`, `set_obstacles(...)`), so the ONLY experimental variable is
the barrier. Non-certificate baselines (B1, B6-APF) replace the filter itself.

## The reviewer questions each baseline answers

| # | Folder | Baseline | Kills the question |
|---|--------|----------|--------------------|
| B1 | `b1_no_filter/` | Nominal DS only (+ naive straight-line) | "Are demos alone unsafe?" |
| B2 | `b2_fixed_pose_cbf/` | Learned CBF, **no pose augmentation** (fixed_pose_model.pt) | "Is augmentation the contribution?" |
| B3a | `b3_analytic_sdf_cbf/` | **Oracle**: B = K·clip(sdf−Δ) from ground-truth geometry | "Why learn — upper bound" |
| B3b | `b3_analytic_sdf_cbf/` | **Practical**: ESDF rasterized from the raw point cloud (what a skeptic would deploy) | "Why learn — honest alternative" |
| B4 | `b4_convex_primitive_cbf/` | Classical closed-form CBF on min enclosing circles | "Where is the textbook CBF?" (conservatism on non-convex shapes) |
| B5 | `b5_cncbf_pershape/` | CN-CBF-style: per-shape-type nets on relative coords, smooth-min, **no shape encoder, no augmentation** | "How is this better than CN-CBF's design?" (fails under rotation/scale) |
| B6 | `b6_apf/` | Artificial potential field repulsion (no QP, no certificate) | "Why a CBF-QP at all?" |
| B7 | `b7_global_barrier/` | Single scene-frozen barrier B(x) (S²-NNDS-style stand-in) | "Prior full-stack LfD+barrier systems are scene-specific" |

Full S²-NNDS (their public repo, github.com/allemmbinn/S2NNDS) is a manual
follow-up; B7 reproduces its *structural* limitation (barrier frozen to one scene)
inside our stack so the comparison is controlled.

## Shared protocol (common/)

- `common/scenes.py` — fixed scene suite: nominal scene with N perturbed starts,
  plus feasibility-checked random pose/scale scenes (3–6 obstacles, fixed seeds)
  reusing `make_solvable_scene`. Every method sees the *identical* list.
- `common/runner.py` — instrumented rollout loop (same integration as
  `multi_rollout.run_from_start`): logs per-step true SDF, barrier value, QP
  solve time, discrete-projection activations, backstop activations, applied velocity.
- `common/metrics.py` — per-rollout metrics: reached, time-to-goal, path length,
  min true SDF (mm), solid-penetration steps, light-red steps, max/mean deviation
  from demo, jerk RMS, QP time, **backstop activation rate** (the discriminator
  when every method reads "0 penetrations").
- `run_all.py` — every method × {nominal, generalization} × backstop {off, on};
  JSON + markdown table into `results/baselines/`.

Two regimes, as in the paper draft:
- **EXP A (backstop OFF)** — the barrier alone must keep the tool safe. Isolates
  barrier quality (B2/B5/B7 should fail on unseen poses; B3/B4 stay safe).
- **EXP B (backstop ON)** — full pipeline; discriminates via reach rate,
  deviation, jerk, and backstop-activation rate.

## Method-specific notes

- **B3b ESDF**: world-frame interior cloud → occupancy grid (2.5 mm), dilate by
  sampling spacing, signed EDT (outside − inside), bilinear query, central-diff
  gradient, then the same K·clip(·−Δ) shaping as ours. Expected: safe but noisy
  gradients → jerk / chatter; sign ambiguity near thin features.
- **B4**: r = max‖p‖ over the centered cloud (bounding circle) per obstacle,
  b_i = K·clip(‖x−c_i‖−r−Δ), smooth-min. Expected: safe but conservative —
  blocks the crescent/star concavities the demo threads.
- **B5**: one MLP per shape *type*, input (x−c_i)/σ, trained by SDF regression +
  eikonal on the **canonical** pose only. Translation generalizes (relative
  coords); rotation/scale does not (that's the point). Uses ground-truth shape
  identity — generous to the baseline.
- **B7**: `models.BarrierNet` (absolute coords) trained on the nominal scene's
  shaped SDF. Perfect on nominal, meaningless on moved obstacles.
- Trained artifacts go to `baselines/<name>/checkpoints/` (gitignored).

## Order of implementation (easy → hard)

1. common infra → 2. B1 → 3. B3a → 4. B4 → 5. B6 (APF) → 6. B2 →
7. B3b → 8. B7 (train) → 9. B5 (train) → 10. run_all + smoke test.
