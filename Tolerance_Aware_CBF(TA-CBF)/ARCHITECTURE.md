# TA-CBF — Complete Architecture & Math Reference

This document describes the **current, full state** of `src/`: every network, every
loss, the two-stage training, the online controller, the **hybrid safety filter**,
and the generalization / dynamic-environment machinery. It is meant to be the
single reference for the paper.

The system makes a needle (3-D end-effector tip moving in a constant-Z plane)
**follow an expert demonstration** while **strictly avoiding** irregular critical
tissue ("red") obstacles, **generalizing zero-shot** to new obstacle
numbers / positions / orientations / sizes and to **slowly moving** obstacles.

---

## 0. Notation

| Symbol | Meaning |
|---|---|
| `x ∈ ℝ³` | needle-tip position (motion is planar at `Z = Z_CORRIDOR`) |
| `x̃ = (x−μ)/σ` | input-normalized position (`μ,σ` = demo mean/std) |
| `s ∈ [0,1]` | task progress along the demo |
| `x_ref(s)` | mean demo path at progress `s` |
| `e = x − x_ref(s)` | tracking error |
| `f_θ` | progress-conditioned dynamical system (the demo "flow") |
| `V_θ` | Lyapunov certificate (CLF) |
| `B_φ` | composite neural barrier (CBF) |
| `sdf(x)` | exact analytic signed distance to nearest critical obstacle (`>0` outside) |
| `Δ` | `INFLATE_MARGIN` — barrier inflation (B=0 sits `Δ` outside the true surface) |
| `K` | `BARRIER_SDF_K` — target barrier slope (so `B ≈ K·(sdf−Δ)` near the boundary) |

---

## 1. Big picture (pipeline)

```
expert demos (20 trajectories, 3-D needle-tip, Z = const)
        │  imitation (Neural-ODE RK4 rollout MSE)
        ▼
┌──────────────────────────────────────────────────────────────────┐
│ f_θ  ProgressConditionedDS      ẋ = v_net(x̃,s) + K_f·(x_ref(s)−x) │  ← WHOLE demo path is the attractor
│ V_θ  LyapunovNet                V = ‖e‖²·(1+δ·corr(e,s))           │  ← quadratic CLF (strong convergence)
└──────────────────────────────────────────────────────────────────┘
        │   (f_θ, V_θ are obstacle-AGNOSTIC — trained once, reused everywhere)
        ▼
┌──────────────────────────────────────────────────────────────────┐
│ B_φ  CompositeBarrier (CN-CBF)                                     │
│   per obstacle i:  b_i(x) = cbf( (x−c_i)/σ , Enc(cloud_i) )        │  ← shape via permutation-invariant
│   composite     :  B(x)   = −(1/β)·logΣ_i exp(−β·b_i(x))           │     PointNet embedding; smooth-min fuses
│   trained so that  B(x) ≈ K·clip(sdf(x)−Δ)  (a scaled, INFLATED SDF)│     ANY number/shape of obstacles
│   + POSE/SCALE/COUNT AUGMENTATION  → zero-shot generalization      │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼   online, per control step
┌──────────────────────────────────────────────────────────────────┐
│ (a) Go-around guidance  g  (tangential to ∇B, toward goal / away   │
│      from a moving obstacle's heading) → breaks head-on deadlock   │
│ (b) CLF-CBF QP :  min ‖u‖²+λε²                                     │
│        ∇B·(f_eff+u) ≥ −γ(B−m)         (CBF, hard, learned ∇B)      │
│        ∇V·(f_eff+u) ≤ −αV+ε           (CLF, soft)                  │
│ (c) Discrete-time learned-CBF projection: B(x+dt·ẋ) ≥ m            │
│ (d) HYBRID exact filter: bisection on analytic sdf ⇒ sdf ≥ m_geo   │  ← the HARD guarantee (uses known geometry)
└──────────────────────────────────────────────────────────────────┘
        │
        ▼   ẋ = f_θ + u + (guidance), integrated by Euler at DT
   safe needle trajectory  (hugs the demo, diverts around obstacles, reconverges)
```

The **contribution** is the learned, generalizable CLF/CBF that produces the
divert-and-reconverge field. The **hard safety guarantee** in simulation comes
from the analytic filter (d), which uses the known scene geometry — a standard
"learned nominal + exact safety filter" pattern.

---

## 2. Scene geometry & analytic SDFs (`config.py`)

State is 3-D but the task is planar: `Z_CORRIDOR = 0.44 m`, `STATE_DIM = 3`.
`X_START = [0.440, 0.062, 0.44]`, `X_GOAL = [0.548, 0.063, 0.44]`.
Workspace slab: `SLAB_X = (0.37, 0.60)`, `SLAB_Y = (−0.025, 0.165)`.

### 2.1 Critical obstacles — irregular non-convex XY shapes

Six base shapes (`CRITICAL_SHAPES`), each an extruded 2-D cross-section with an
exact SDF (`> 0` outside, `< 0` inside), `sdf_critical_shape_2d`:

| type | param | SDF idea |
|---|---|---|
| `star` (5-point) | `outer_r, inner_r, n_points, rotation` | radial profile `r − star_r(θ)` in the rotated frame |
| `crescent` | `outer_r, inner_r, offset` | `max(r_outer − R, r_inner⁻¹)` = disk minus an offset disk |
| `blob` | `radii[], offsets[]` | `min_k(‖x−c_k‖ − r_k)` = union of disks |
| `kidney` | `outer_r, inner_r, squeeze, rotation` | squashed disk minus an inner disk |
| `lshape` | `arm1, arm2, rotation` | `min` of two rotated rectangle SDFs |

`sdf_all_critical_np(x) = min_i sdf_i(x)` (union of obstacles).
These SDFs provide **training labels** for the barrier and the **exact safety
filter** at run time — the barrier `B_φ` itself is fully learned.

### 2.2 Pose / scale transforms (`transform_shape`)

`transform_shape(shape, d_rot, scale, d_trans)` returns a copy rotated by
`d_rot`, uniformly scaled by `scale` about its centroid, translated by `d_trans`.
It rotates/scales every length parameter and orientation-bearing offset vector.
This single function powers (a) training augmentation, (b) the dynamic-env motion,
(c) random generalization scenes.

---

## 3. Demos, normalization, progress

* **Demos**: 20 expert CSVs (`data/data_trajectory/demo_*.csv`), columns 8–10 =
  needle-tip `x,y,z`. Resampled to `N_POINTS = 200`.
* **Mean path** `x_ref` = mean over resampled demos → the reference path / attractor.
* **Normalization** `μ,σ` = mean/std over all demo points (`σ_z ≈ 0`, planar).
* **Progress** `s` of a query `x`: nearest reference index,
  `get_progress(x) = argmin_k‖x − x_ref_k‖ / (R−1)`.

---

## 4. Networks (`models.py`)

### 4.1 `f_θ` — `ProgressConditionedDS` (the demo flow)

```
ẋ = f_θ(x,s) = v_net([x̃, s])  +  K_f · (x_ref(s) − x)
                └ feed-forward ┘   └── feedback to the demo path ──┘
```

* `v_net`: MLP `[STATE_DIM+1 → 128 → 128 → 128 → STATE_DIM]`, Tanh; last layer
  near-zero-init so the DS starts ≈ pure demo-feedback.
* `K_f = DEMO_K = 5.0`: **strong** attraction to the *whole* demo path (not just
  the goal). This is what makes the needle hug the expert demo and reconverge
  after any detour.
* `x_ref_at_s(s)`: linear interpolation of the reference path at `s`.
* `predict_velocity`, RK4 rollout used for the imitation loss.

### 4.2 `V_θ` — `LyapunovNet` (quadratic CLF, Nawaz et al. 2023)

```
V(x, x_ref(s), s) = ‖e‖² · (1 + δ·corr(e_norm, s)),     e = x − x_ref(s),  δ = 0.05
corr ≥ 0  (Softplus head of a [STATE_DIM+1 → 64 → 64 → 1] MLP)
```

* `V = 0` **exactly** on the demo path (`e=0`), `V > 0` elsewhere (positive-definite
  by construction — no positivity loss needed).
* The multiplicative `corr` is a small learned reweighting of the quadratic bowl.
* QP uses the closed-form gradient `∇V = 2e` (the `corr` gradient vanishes as
  `e→0`, where the QP operates).

### 4.3 `B_φ` — `CompositeBarrier` (composite neural CBF, CN-CBF style)

Three pieces:

**(i) `ObstacleEncoder`** — permutation-invariant PointNet:
```
Enc(P) = max-pool_k MLP(p_k),     MLP: 2 → 128 → 128 → EMB_DIM,   EMB_DIM = 64
```
`P` = interior point cloud of one obstacle, **centered at its centroid**. Max-pool
⇒ invariant to point ordering and count. Centering ⇒ translation handled
separately (the embedding encodes only *shape*, including its current rotation/scale).

**(ii) `ConditionalObstacleCBF`** — one shared per-obstacle CBF head:
```
b_i(x) = mlp( [ (x − c_i)/σ ,  e_i ] ),     mlp: (3+64) → 256 → 256 → 256 → 1
```
Inputs: the **relative** coordinate `(x−c_i)` (so position is handled by
construction — translation-invariant) and the shape embedding `e_i`. One network
serves **any number** of obstacles.

**(iii) Composite (smooth-min, CN-CBF Eq. 18)**:
```
B(x) = −(1/β)·log Σ_i exp(−β·b_i(x)),     β = BARRIER_BETA = 1000
```
A smooth, differentiable under-approximation of `min_i b_i`. Separate boundaries
when obstacles are far apart; a single fused boundary when they nearly touch.
`β=1000` keeps the smooth-min offset `≤ ln(M)/β ≈ 1.8 mm ≪ δ`.

`CompositeBarrier.forward(x, obstacles)` and `.per_obstacle(x, obstacles)` accept
the obstacle list explicitly (cloud tensor + center); `set_obstacles(...)` installs
a default set so the single-argument `model.B(x)` works for the QP and plots.
For dynamic obstacles, the set is re-installed every control step.

### 4.4 `BPCBFModel` — container

Holds `f`, `V`, `B`; `set_norm(μ,σ)` broadcasts normalization (the barrier
normalizes the relative coordinate by `σ`); `set_obstacles`, `save/load`.

---

## 5. The barrier as a scaled, INFLATED SDF

The barrier is trained to be a **scaled, clamped, inflated signed-distance
function** rather than a 0/1 classifier:

```
target(x)  =  K · clip( sdf(x) − Δ ,  −c_in ,  +c_out )
```
with `K = 4`, `Δ = INFLATE_MARGIN = 0.010 m`, `c_in = 0.040`, `c_out = 0.013`.

Consequences:

* **`B(x) = 0 ⟺ sdf(x) = Δ`** → the learned zero-level set sits `Δ = 10 mm`
  **outside** the true tissue (the pink "light-red" CBF zone). A visible safety
  buffer, and the field starts diverting before the needle reaches the tissue.
* **Asymmetric clamp**: the *outside* is clamped tightly (`c_out=13 mm` → `B`
  saturates at `≈+0.05`), but the *inside* is clamped wide (`c_in=40 mm`) so the
  gradient **persists through the whole interior** — `∇B` always points outward
  inside an obstacle (no flat interior that lets a needle coast through).
* **Uniform slope**: `‖∇B‖ ≈ K` in the boundary band ⇒ `B = m ⟺ sdf = Δ + m/K`
  for *every* obstacle, so the needle's standoff is consistent.

---

## 6. Pose / scale / count augmentation (the generalization mechanism)

This is the key to zero-shot generalization. **Every barrier-training step**
(`_augment_scene` in `train.py`):

1. Pick a **random subset** of the 6 base shapes (count `1..6`) — trains the
   smooth-min to fuse an arbitrary number/mix of obstacles.
2. For each chosen shape, draw a random transform:
   `rotation φ ~ U(−π, π)`, `scale σ ~ U(0.65, 1.4)`, `translation ~ U(−0.05, 0.05)²`.
3. Apply the **same** transform to **both**:
   * the **point cloud** fed to `Enc` (→ embedding `e_i`), and
   * the **analytic SDF** used to label safe/unsafe points.

Because the cloud and the labels are transformed together, `cbf((x−c_i)/σ, e_i)`
learns to respond **consistently to the embedding regardless of the pose it came
from**. This is precisely what fixes the original failure (a rotated obstacle
produced an out-of-distribution embedding). Translation is free by construction
(relative coordinate); rotation + scale are covered by augmentation; count by the
random subset + smooth-min.

---

## 7. Training (two stages)

`f_θ`, `V_θ` are obstacle-agnostic and only need to be trained **once**; the
barrier needs many more steps than the joint loop affords, so it is trained
separately on the frozen `f/V`.

### Stage 1 — S2-NNDS Algorithm 2 (`train.py`) → good `f_θ, V_θ`

* **Phase 0**: pre-train `f_θ` alone on the imitation loss for 300 epochs.
  Imitation = Neural-ODE rollout MSE (Eq. 9):
  ```
  L_MSE = (1/MT) Σ_i Σ_k ‖ x_i(t_k) − x̂_i(t_k) ‖²,   x̂ = RK4 rollout of f_θ from x_i(0)
  ```
* **Outer loop** (`OUTER_ITERS = 10`, `INNER_EPOCHS = 150`): jointly train `f,V,B`
  on `L_MSE + L_lyap + w·L_bar`, then sample `N_CEX` counterexamples and add
  violating states to the working sets (Algorithm 2). The barrier weight `w`
  ramps in after the first few "Lyapunov-only" outer iters.
* **Lyapunov loss** (Eq. 10): `L_lyap = λ_V2 · mean relu( ∇V·f + α·V )` (enforce
  `V̇ ≤ −αV`; positivity holds by construction).
* The barrier from Stage 1 is intentionally under-trained (only ~1050 barrier
  steps); Stage 2 replaces it.

### Stage 2 — fast barrier-only (`train_barrier_fast.py`) → final `B_φ`

Loads the trained `f/V` (only `f.*`,`V.*` keys), **freezes** them, **re-initializes
`B`**, and trains it for **12 000 steps** (Adam, `lr = 2e-3`, cosine schedule) with
`augmented_barrier_loss`. Logs the loss history to `barrier_loss_history.npz`.
Saves `final_model.pt` / `best_model.pt`. Runs in a couple of minutes on GPU.

### 7.1 `augmented_barrier_loss` — the barrier objective

For the freshly augmented scene (Section 6), with `δ = DELTA_B = 0.01`,
`γ = GAMMA_CBF = 3`:

```
L = λ_reg·(L_reg + L_comp) + λ_FS·L_safe + λ_OB·L_uns + λ_dot·L_dot + λ_eik·L_eik
```
weights: `λ_reg = 20`, `λ_FS = 5`, `λ_OB = 12`, `λ_dot = 0.5`, `λ_eik = 0.02`.

| term | definition | purpose |
|---|---|---|
| **`L_reg`** | `mean_i MSE( b_i(X_i), target(sdf_i) )` | per-obstacle scaled-SDF regression. `X_i` = local box **+ boundary-band densified points** (interior cloud jittered ±12 mm so thin/concave tips like crescent horns are well covered) **+** strict interior. |
| **`L_comp`** | `MSE( B(X_g), target(min_i sdf_i) )` over a global slab set | the *composite* must itself be a scaled SDF (absorbs the smooth-min offset). |
| **`L_uns`** | `mean_i mean_{sdf<Δ} relu(δ + b_i)` | drive `b_i < −δ` on the obstacle interior **and the inflation shell** `0<sdf<Δ`. Supervising the shell pins `B=0` to the *inflated* boundary (not inside the true surface). |
| **`L_safe`** | `mean_{sdf>Δ+0.004} relu(δ − B)` | drive `B > δ` clearly outside. |
| **`L_dot`** | `mean_{B≥0} relu( −(∇B·f_θ + γB) )` | CBF forward-invariance `Ḃ + γB ≥ 0` on the safe set (continuity into a valid CBF). |
| **`L_eik`** | `mean relu( ‖∇B‖ − B_grad_max )`, `B_grad_max = 12` | Lipschitz cap. SDF-value regression fixes `B` only *at* samples; between them the net can interpolate with a near-step (`‖∇B‖≈300`), making `∇B·f` erratic and the QP useless. Capping `‖∇B‖` + the regression slope `K` ⇒ a smooth, well-conditioned, SDF-like gradient. |

Result: `safe-acc ≈ 85–100%`, `unsafe-acc → 100%`, `‖∇B‖ ≈ 5–12`, `B ∈ [−0.06, +0.05]`.

### 7.2 Counterexamples (`find_counterexamples`, Stage 1)

Samples states, flags: `V̇ > −αV`; `B < δ` where `sdf > Δ+0.004` (safe); `B > −δ`
where `sdf < 0` (unsafe); `Ḃ + γB < 0` where `B ≥ 0`. Violations are appended to
the working sets. NaN/Inf ⇒ abort (diverged).

---

## 8. Online controller (`cbf_qp.py`)

Per control step, given `x`, the model, the installed obstacles, progress `s`, and
optionally the moving obstacle velocity `obs_vel`:

### 8.1 Go-around guidance (swirl) — breaks head-on deadlock

A perfectly head-on obstacle leaves the CBF tangent ambiguous, so a strong demo
pull stalls the needle dead-center. When near the boundary (`B < B_ACTIVE = 0.04`):

```
n = ∇B/‖∇B‖,    t = (−n_y, n_x, 0)            (unit tangent in XY, ⊥ ∇B)
pref = (x_goal − x)/‖·‖   [ − 1.6·obs_vel/‖obs_vel‖  if the obstacle is moving ]
if t·pref < 0: t ← −t                          (circulate toward goal / away from heading)
closeness = max(0, (B_ACTIVE − B)/B_ACTIVE)
g = SWIRL_GAIN · ‖f‖ · closeness · t,          SWIRL_GAIN = 1.0
f_eff = f_θ + g
```

`g ⊥ ∇B` ⇒ it does **not** change the barrier rate `∇B·f_eff`, only the
go-around direction. For a **moving** obstacle, biasing `pref` opposite the
obstacle's heading sends the needle to the side it is *vacating* (so it isn't
chased into a block). Safe to run strong because the analytic filter (8.4) is the
hard backstop. (`obs_vel` is supported by the controller; the current dynamic test
relies on the goal-directed swirl, which empirically reaches the goal for all
motions.)

### 8.2 CLF-CBF QP (defends a positive level set `B ≥ m`)

```
min_{u, ε≥0}  ‖u‖² + λ·ε²
s.t.   ∇B·(f_eff + u) ≥ −γ·(B − m)        (CBF, hard;   m = B_SAFE_MARGIN = 0.008)
       ∇V·(f_eff + u) ≤ −α·V + ε          (CLF, soft)
```
`γ = 3`, `α = 4`, `λ = 0.5`, solved with OSQP. Defending `B ≥ m` (not `B ≥ 0`)
absorbs small positive barrier errors and yields a standoff `sdf ≈ Δ + m/K`.
Fallback (if OSQP fails): minimum-norm `u` meeting the CBF row.

### 8.3 Discrete-time learned-CBF projection (`project_safe`)

The continuous CBF only bounds `Ḃ`; with a finite Euler step the needle can drift
across the boundary. After the QP, project the velocity so the **actual next
state** is safe:
```
need = m − B(x + dt·v);    if need > 0:  v ← v + (need /(dt·‖∇B‖²))·∇B   (Newton, 3×)
```
capped to `VEL_CLIP`; if `‖∇B‖² < 1` (no usable gradient) the step is **stopped**
rather than pushed blindly (prevents overshooting *into* an obstacle).

### 8.4 HYBRID exact safety filter (`analytic_safety_filter`) — the guarantee

Final geometric check using the **known** scene SDF:
```
ensure   sdf(x + dt·v) ≥ m_geo          (m_geo = SAFETY_SDF_MARGIN = 0.011 m = Δ + 1 mm)
```
* If `sdf(x + dt·v) ≥ m_geo`: accept `v`.
* Else if already inside the shell: step **out** along the numeric `∇sdf`.
* Else: **bisection** (24 iters) for the largest fraction `α∈[0,1]` with
  `sdf(x + α·dt·v) ≥ m_geo` → `v ← α·v` (preserves direction, so the needle slides
  around the boundary).

This **guarantees the needle stays ≥ 11 mm from the tissue — i.e. fully outside
the light-red CBF zone — for any obstacle pose**, including dead-center / thin-tip
cases where the learned barrier alone is slightly mis-calibrated. The learned CBF
still *drives* the divert-and-reconverge motion; the analytic filter only clips
the rare unsafe residual.

### 8.5 Rollout (`run_from_start`, `multi_rollout.py`)

```
for each step (DT = 0.01, up to T_MAX = 12 s):
    s   = monotone progress estimate (max of demo-progress and time)
    f   = f_θ(x, s);   near the goal, blend in a small radial goal_pull
    u   = QP.solve(x, model, s)                          # 8.1–8.2
    ẋ   = clip(f + u, ±VEL_CLIP)
    ẋ   = project_safe(x, model, ẋ, DT)                  # 8.3 (learned)
    ẋ   = analytic_safety_filter(x, ẋ, DT, shapes)       # 8.4 (exact guarantee)
    x  += ẋ·DT
    stop if ‖x − x_goal‖ < GOAL_TOL (0.006)
```

---

## 9. Slow-changing environment (`dynamic_env_test.py`)

A sparse, **solvable** layout: four static obstacles form the walls of a wide
corridor; one obstacle (`MOVING_IDX`, the star) slowly changes over the rollout
as a function of `τ = t / (0.9·T_MAX)`:

| motion | schedule |
|---|---|
| `translate` | drift up into the corridor |
| `rotate` | spin in place near the path |
| `transrotate` | drift up **and** spin |
| `evolve` | slowly enlarge in place (`scale 0.6 → 1.2`) |

Every control step: compute the obstacle's current pose, rebuild its cloud
(`scale·canon @ R(rot)ᵀ`), `set_obstacles`, run the controller. The analytic filter
uses the obstacle's **current** pose, so safety holds throughout. The barrier is
**not retrained** — it re-evaluates against the moving pose. Rollouts run **until
the goal is reached**. Verified `obstacle = 0`, `light_red = 0`, `reached = True`
for all four motions.

---

## 10. Generalization scenes (`generalization_test.py`)

* **`make_solvable_scene(n_obs, seed)`**: obstacles **spread across the whole
  slab** with a minimum centre separation `min_sep = 0.045` (no clumping), random
  pose+scale, with **≥1 obstacle placed on the demo path**. Accepted only if a
  free start→goal path with clearance `> SCENE_CLEARANCE` exists (`_free_path_exists`,
  a grid BFS on the inflated free space) — i.e. a real passage *outside* the
  light-red zone is guaranteed.
* **`make_validated_scene`**: additionally runs a rollout and keeps the scene only
  if it **reaches** the goal, **never enters the tissue** (`obstacle = 0`), and
  **visibly deviates** (so the figure is illustrative).
* **`random_shapes`, `merge_two_shapes`**: legacy generators (random reposition /
  rotate; force two obstacles to nearly touch) used for stress tests.

---

## 11. Field plotting (`field_plot.py`)

`plot_diverting_field` renders `ẋ = f_θ + u_safe` streamlines, **masks the
keep-out set `{B < m}`** (so streamlines route *around* the light-red zone instead
of being drawn through it), overlays the learned `B = 0` contour (purple dashed),
the obstacle fill, and the demo path. Used by the final field figure, the
generalization panels, and the dynamic snapshots.

---

## 12. Final figures (`make_final.py`)

`python src/make_final.py` writes everything under `results/FINAL/`:

```
losses_from_training/  barrier_training.png (loss + accuracy curves)
                       full_training_losses.png (Stage-1 f/V/B co-training)
final_learned_cbf.png  learned B_φ(x) with the B=0 boundary on the canonical scene
final_vector_field.png closed-loop field — diverts around the light-red, reconverges
generalization_result/ gen_{4,5,6,7,8}obs.png — spread obstacles, ≥1 on path,
                       6 perturbed-start rollouts each (all obstacle=0 & outside light-red)
dynamic_env/           dynamic_{translate,rotate,transrotate,evolve}.{png,gif}
                       PNG snapshots + animated GIFs of the moving-obstacle rollout
```

---

## 13. Final configuration (all key constants)

| Group | Constant | Value | Meaning |
|---|---|---|---|
| state | `STATE_DIM` / `Z_CORRIDOR` | 3 / 0.44 | planar 3-D |
| demo flow | `DEMO_K` | **5.0** | demo-path attraction (strong hug) |
| | `F_HIDDEN` | `[128,128,128]` | `v_net` |
| CLF | `V_HIDDEN` / `δ` | `[64,64]` / 0.05 | quadratic CLF correction |
| | `ALPHA_CLF` | 4.0 | `V̇ ≤ −αV` |
| barrier net | `EMB_DIM` | 64 | shape embedding |
| | cbf hidden | `(256,256,256)` | per-obstacle CBF |
| | `BARRIER_BETA` | 1000 | smooth-min sharpness |
| barrier SDF | `INFLATE_MARGIN Δ` | **0.010** | B=0 sits 10 mm outside tissue |
| | `BARRIER_SDF_K` | 4.0 | barrier slope |
| | `..CLAMP_OUT / _IN` | 0.013 / 0.040 | asymmetric clamp (persistent interior ∇B) |
| | `B_GRAD_MAX` | 12 | Lipschitz cap |
| loss weights | `λ_reg/FS/OB/dot/eik` | 20 / 5 / 12 / 0.5 / 0.02 | barrier loss |
| | `DELTA_B` | 0.01 | classification margin δ |
| augment | `AUG_ROT/SCALE/TRANS` | `(−π,π)`/`(0.65,1.4)`/0.05 | pose/scale/translation |
| CBF QP | `GAMMA_CBF` | 3.0 | class-K rate |
| | `B_SAFE_MARGIN m` | **0.008** | defended level set `B ≥ m` |
| | `LAMBDA_SLACK` | 0.5 | CLF slack penalty |
| guidance | `SWIRL_GAIN` / `B_ACTIVE` | **1.0** / 0.04 | go-around strength / activation |
| **hybrid** | `SAFETY_SDF_MARGIN` | **0.011** | exact geometric standoff (= Δ+1 mm) |
| sim | `DT` / `VEL_CLIP` / `GOAL_TOL` / `T_MAX` | 0.01 / 0.25 / 0.006 / 12 | integrator |
| train | `OUTER/INNER` | 10 / 150 | Stage-1 loop |
| | `LR` / `LR_BARRIER` | 3e-4 / 2e-3 | optimizers |
| | barrier steps | 12 000 | Stage-2 |

---

## 14. Guarantees, generalization, and honest limitations

**Guaranteed (in simulation, known geometry):** the needle **never enters the
critical tissue and stays outside the light-red CBF zone** (`sdf ≥ 11 mm`), for any
number / position / orientation / size of obstacles and for slowly moving
obstacles — enforced exactly by the analytic safety filter (§8.4).

**Generalizes zero-shot (no retraining):** the learned `B_φ` produces sensible,
SDF-like barriers and a divert-and-reconverge field for new obstacle
counts/poses/scales — thanks to the pose/scale/count augmentation (§6), the
relative-coordinate + PointNet-embedding architecture (§4.3), and the smooth-min
composition. Verified for 1–11 obstacles and all four slow motions.

**Strong demo convergence:** `K_f = 5` + the quadratic CLF make the needle hug the
expert demo and snap back to it after any detour.

**Honest limitations:**
* The **hard guarantee uses the known scene geometry** (analytic SDF), not the
  learned net alone — this is the "learned nominal + exact safety filter" design,
  stated as such.
* The **learned barrier's edge accuracy** on thin, concave, rotated shapes (star
  spikes, crescent horns) is imperfect; that is exactly why the positive margin
  `m` and the analytic backstop exist.
* The controller is **reactive** (no global planner). A perfectly head-on obstacle
  would deadlock without the swirl guidance (§8.1); with it, head-on cases go
  around and reach the goal. Truly sealed corridors (no feasible passage) are
  out of scope — the scene generator guarantees a passage exists.

---

## 15. File map

```
src/
  config.py            scene geometry, shape SDFs, transforms, augmentation,
                       all hyperparameters, sampling helpers
  models.py            f_θ (ProgressConditionedDS), V_θ (LyapunovNet),
                       B_φ (ObstacleEncoder + ConditionalObstacleCBF + CompositeBarrier)
  train.py             Stage-1 S2-NNDS Algorithm 2 + augmented_barrier_loss + plots
  train_barrier_fast.py Stage-2 barrier-only training on frozen f/V (+ loss logging)
  cbf_qp.py            BPCBFController (swirl + CLF-CBF QP + project_safe)
                       + analytic_safety_filter (hybrid exact guarantee)
  multi_rollout.py     run_from_start (full rollout loop) + perturbed-start overlays
  dynamic_env_test.py  slow-changing-obstacle scenes + run_dynamic
  generalization_test.py solvable/validated spread scenes + BFS feasibility
  field_plot.py        plot_diverting_field (masked divert-reconverge field)
  make_final.py        all FINAL paper figures (losses, barrier, field, gen, dynamic GIFs)
  simulate.py          load_model + comparison sims
  analytical_sdf.py    torch/np SDFs (labels only)
checkpoints/           final_model.pt, best_model.pt, norm_*.npy, ref_path.npy,
                       barrier_loss_history.npz
results/FINAL/         paper figures (see §12)
```
