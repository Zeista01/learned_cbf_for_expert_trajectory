# TA-CBF — Complete Run Guide

All commands are run from the project root:
```
cd /home/stanny/franka_ros2_ws/src/Tolerance_Aware_CBF\(TA-CBF\)
```

---

## 0. Activate the Virtual Environment (ALWAYS do this first)

```bash
cd /home/stanny/franka_ros2_ws/src/Tolerance_Aware_CBF\(TA-CBF\)
source venv/bin/activate
```

Verify it's active (prompt shows `(venv)` prefix). To deactivate later:
```bash
deactivate
```

---

## 1. Training

### Stage 1 — Full Co-Train (f_θ + V_θ + B_φ)

Trains all three networks jointly using S2-NNDS Algorithm 2.
Runtime: **~2–2.5 hours on GPU**, longer on CPU.
Outputs: `checkpoints/final_model.pt`, `checkpoints/best_model.pt`,
`checkpoints/norm_mean.npy`, `checkpoints/norm_std.npy`, `checkpoints/ref_path.npy`

```bash
python src/train.py
```

Resume / check training progress via TensorBoard:
```bash
tensorboard --logdir runs/
```

### Stage 2 — Fast Barrier-Only Retraining

Loads the existing `final_model.pt`, freezes f and V, re-trains only B_φ
with full pose/scale/rotation augmentation.
Runtime: **~3–5 minutes on GPU**.
Outputs: overwrites `checkpoints/final_model.pt` and `checkpoints/best_model.pt`

```bash
# Default: 4000 steps (fast, good enough for testing)
python src/train_barrier_fast.py --steps 4000

# Recommended for best accuracy:
python src/train_barrier_fast.py --steps 12000

# Custom number of steps:
python src/train_barrier_fast.py --steps 8000
```

---

## 2. Simulation — Basic Modes

### View the Scene (sanity check, no rollout)
```bash
python src/view_scene.py
# Output: results/scene_overview.png
```

### Single Simulation (bpcbf mode — our proposed method)
```bash
python src/simulate.py --mode bpcbf
# Output: results/simulation_comparison.png, results/learned_barrier_field.png
```

### All three modes side-by-side (nominal / hard_cbf / bpcbf)
```bash
python src/simulate.py --mode all
# Output: results/simulation_comparison.png
```

### Vector field plot only
```bash
python src/simulate.py --plot_field
# Output: results/vector_field_trajectory_attractor.png
```

---

## 3. Rollout Simulations (results/roll_outs/)

All rollout outputs go to `results/roll_outs/{normal,generalized,dynamical}/`.

### Run ALL three rollout types at once
```bash
python src/run_rollouts.py
# Runs: 10 normal + 4 generalization scenes (3/4/5/6 obs) + 4 dynamic motions
```

### Normal Rollouts (perturbed starting positions)
```bash
# Default: 10 rollouts, 1.8 cm perturbation radius
python src/run_rollouts.py --type normal

# Custom: 15 rollouts with wider perturbation
python src/run_rollouts.py --type normal --n 15 --radius 0.025

# Outputs:
#   results/roll_outs/normal/normal_rollout_01.png  ... normal_rollout_N.png
#   results/roll_outs/normal/normal_overlay.png
#   results/roll_outs/normal/normal_statistics.png
```

### Generalization Rollouts (random pose/scale obstacles)
```bash
# Default: 3, 4, 5, 6 obstacles — zero-shot pose generalization
python src/run_rollouts.py --type generalized

# Custom obstacle counts and seeds:
python src/run_rollouts.py --type generalized --n_obs 3 5 7 --gen_seeds 1 10 20

# Outputs:
#   results/roll_outs/generalized/gen_3obs_seed1.png
#   results/roll_outs/generalized/gen_4obs_seed13.png
#   results/roll_outs/generalized/gen_5obs_seed20.png
#   results/roll_outs/generalized/gen_6obs_seed30.png
#   results/roll_outs/generalized/gen_safety_summary.png
```

### Dynamical Rollouts (moving/rotating/growing obstacle)
```bash
# Default: all 4 motions (translate / rotate / transrotate / evolve)
python src/run_rollouts.py --type dynamical

# Specific motions only:
python src/run_rollouts.py --type dynamical --motions translate rotate

# Custom: more snapshot frames per motion
python src/run_rollouts.py --type dynamical --frames 6

# Outputs:
#   results/roll_outs/dynamical/dynamic_translate.png
#   results/roll_outs/dynamical/dynamic_rotate.png
#   results/roll_outs/dynamical/dynamic_transrotate.png
#   results/roll_outs/dynamical/dynamic_evolve.png
#   results/roll_outs/dynamical/dynamic_summary.png
```

---

## 4. Original Test Scripts (save to their own dirs)

These are the original per-script runners. They save to different directories
but are still useful for quick individual tests.

### Multi-Rollout (results/after_training/)
```bash
python src/multi_rollout.py --n 6 --radius 0.015
python src/multi_rollout.py --n 10 --radius 0.020 --seed 7
```

### Generalization Test (results/generalization/)
```bash
python src/generalization_test.py
python src/generalization_test.py --n_obs 3 5 7 --seed0 1 10 20
```

### Dynamic Environment Test (results/dynamic/)
```bash
python src/dynamic_env_test.py
python src/dynamic_env_test.py --motions rotate evolve --frames 5
python src/dynamic_env_test.py --quick    # faster, coarser grid
```

---

## 5. Paper-Quality Figures

Generates ALL final figures used for the paper: loss curves, final CBF field,
generalization results (4–8 obstacles), dynamic GIF animations.
Runtime: ~20–40 minutes depending on GPU speed.

```bash
python src/make_final.py

# Outputs (results/FINAL/):
#   losses_from_training/   — training loss curves
#   final_learned_cbf.png   — learned B_φ(x) field on canonical scene
#   final_vector_field.png  — closed-loop diverting field
#   generalization_result/  — 6 rollouts per config (4/5/6/7/8 obstacles)
#   dynamic_env/            — 4 motions, each with multi-frame snapshot + animated GIF
```

---

## 6. Collect New Demonstrations

If you want to record new demo trajectories (for a different task or path):

```bash
python src/collect_demos.py
# Follow the interactive prompts.
# Output: data/data_trajectory/demo_N.csv
```

---

## 7. Full Workflow: From Scratch to Results

```bash
# Step 0: activate venv
cd /home/stanny/franka_ros2_ws/src/Tolerance_Aware_CBF\(TA-CBF\)
source venv/bin/activate

# Step 1: full co-training (~2.5h)
python src/train.py

# Step 2: fast barrier retraining (~5min, after step 1)
python src/train_barrier_fast.py --steps 12000

# Step 3: quick sanity check — single rollout
python src/simulate.py --mode bpcbf

# Step 4: run all rollout simulations
python src/run_rollouts.py

# Step 5: paper figures
python src/make_final.py
```

---

## 8. Checkpoints & Output Directories

| Path | Contents |
|------|----------|
| `checkpoints/final_model.pt` | Main model (load this for inference) |
| `checkpoints/best_model.pt`  | Best validation checkpoint |
| `checkpoints/norm_mean.npy`  | Input normalisation mean (required at inference) |
| `checkpoints/norm_std.npy`   | Input normalisation std |
| `checkpoints/ref_path.npy`   | Demo reference path |
| `checkpoints/barrier_loss_history.npz` | Stage 2 loss curve data |
| `results/roll_outs/normal/`       | Normal rollout PNGs |
| `results/roll_outs/generalized/`  | Generalization rollout PNGs |
| `results/roll_outs/dynamical/`    | Dynamic motion PNGs |
| `results/FINAL/`                  | Paper-quality figures + GIFs |
| `runs/`                           | TensorBoard event files |

---

## 9. Key Configuration (src/config.py)

Most important parameters you may want to tune:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `DEMO_K` | 5.0 | Demo path spring gain (higher = tighter path following) |
| `B_SAFE_MARGIN` | 0.008 | CBF defends B ≥ this value (larger = more conservative) |
| `SAFETY_SDF_MARGIN` | 0.011 m | Analytic backstop minimum SDF distance |
| `GAMMA_CBF` | 3.0 | CBF exponential decay rate (higher = faster response) |
| `ALPHA_CLF` | 4.0 | CLF convergence rate |
| `SWIRL_GAIN` | 1.0 | Go-around guidance strength (0 = disabled) |
| `INFLATE_MARGIN` | 0.010 m | How far outside tissue B=0 is trained to sit |
| `BARRIER_SDF_K` | 4.0 | Barrier slope amplification |
| `OUTER_ITERS` | 10 | Stage 1 outer loop iterations |
| `INNER_EPOCHS` | 150 | Stage 1 inner epochs per outer iteration |

---

## 10. Troubleshooting

**Model not found:**
```bash
ls checkpoints/   # must have final_model.pt
# If missing: run python src/train.py first
```

**CUDA out of memory:**
```bash
# In config.py, reduce BATCH_SIZE from 256 to 128
# Or force CPU: DEVICE = "cpu" in config.py
```

**QP solver warnings ("Polishing not needed"):**
These are normal OSQP log messages — not errors. Suppress them by adding
`verbose=False` (already set in cbf_qp.py).

**Needle not reaching goal / pocket trap:**
This is a known limitation when obstacles block the path tightly.
`make_validated_scene()` in `generalization_test.py` filters these out automatically.
Try a different seed: `--gen_seeds 5 15 25 35`

**venv issues / import errors:**
```bash
# Recreate venv from scratch:
deactivate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision osqp scipy numpy matplotlib
```
