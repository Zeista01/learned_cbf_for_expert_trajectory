# Baselines

Controlled comparison suite for TA-CBF. Design: **only the safety mechanism
varies** — every method shares the nominal DS `f_θ`, the CLF-CBF-QP, the fixed
scene suite (seeded), the integrator, and the metrics. QP baselines are
drop-in replacements for `model.B`; non-certificate baselines (B1, B6) replace
the filter. See `PLAN.md` for the rationale and the reviewer question each
baseline answers.

## Quick start

```bash
cd ~/franka_ros2_ws/src/Tolerance_Aware_CBF\(TA-CBF\)

# one-time training for the two learned baselines
venv/bin/python baselines/b7_global_barrier/train.py
venv/bin/python baselines/b5_cncbf_pershape/train.py
# (B2 uses the existing checkpoints/fixed_pose_model.pt — retrain with
#  venv/bin/python src/train_fixed_pose.py if needed)

# smoke test (tiny suite)
venv/bin/python baselines/run_all.py --quick

# full protocol → results/baselines/baselines_all.json + printed table
venv/bin/python baselines/run_all.py

# a single baseline against ours
venv/bin/python baselines/b4_convex_primitive_cbf/run.py

# subset of methods, more scenes
venv/bin/python baselines/run_all.py --only ours_ta_cbf b3b_cloud_esdf --per 5
```

## Methods

| name | what it is |
|---|---|
| `ours_ta_cbf` | proposed: augmented composite barrier, zero-shot |
| `b1_nominal_ds` | learned DS, no safety filter |
| `b1_straight_line` | straight line to goal, no learning, no safety |
| `b2_fixed_pose_cbf` | same architecture, trained without augmentation |
| `b3a_oracle_sdf` | K·clip(sdf−Δ) from ground-truth geometry (upper bound) |
| `b3b_cloud_esdf` | same, but SDF rasterized from the raw point cloud |
| `b4_circle_cbf` | classical CBF on min enclosing circles |
| `b5_cncbf_pershape` | CN-CBF-style per-obstacle nets, relative coords, no encoder |
| `b6_apf` | artificial potential field (no QP, no certificate) |
| `b7_global_barrier` | single scene-frozen barrier (S²-NNDS-style stand-in) |

## Protocol

Each method × regime {nominal (perturbed starts), generalization (unseen
pose/scale scenes, 3–6 obstacles)} × geometric backstop {off, on}.

- **backstop OFF** = the method's own barrier must keep the tool safe
  (isolates barrier quality; expect B2/B5/B7 to fail off-nominal).
- **backstop ON** = full pipeline; when everything shows 0 penetrations, rank
  by `backstop_rate` / `proj_rate` (how often the exact filters had to rescue
  the barrier), `dev_max_mm` (conservatism), `jerk_rms` (smoothness),
  `qp_ms_mean` (cost), and reach rate.

Results land in `results/baselines/<out>.json`; scene suites are cached next
to them so all runs share identical scenes.
