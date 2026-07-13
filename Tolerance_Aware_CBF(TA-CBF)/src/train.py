"""
train.py — S2-NNDS Algorithm 2 training for the TA-CBF (Learned Barrier) model.

Architecture (per mentor feedback):
  f_θ  : ProgressConditionedDS — demo path as attractor (not just goal)
  V_θ' : Quadratic CLF V(e) = ||e||² from Nawaz et al. (replaces ICNN)
  B_φ  : Fully learned BarrierNet (S2-NNDS) — no analytic SDF anchor
          Critical obstacle shapes are non-geometric (star, crescent, etc.)

Training Algorithm 2 (S2-NNDS):
  Phase 0: Pre-train f_θ alone on imitation loss (Neural ODE rollout MSE)
  Outer loop:
    Inner loop: joint train f_θ, V_θ', B_φ on all losses (Eq. 9-11)
    Sample N_cex counterexamples
    Check violations of Lyapunov + barrier conditions
    If violations: add them to training sets S, S_0, S_u → continue
    If no violations: conformal verification, done

Losses:
  L_MSE (Eq. 9): Neural ODE trajectory imitation (rollout MSE on demo)
  L_lyap (Eq. 10): V > 0 in S, V̇ ≤ -α·V
  L_bar (Eq. 11): B > δ in X_0 (safe), B < -δ in X_u (unsafe), Ḃ + γ·B ≥ 0 in S

Usage:
    cd /home/stanny/franka_ros2_ws/src/Tolerance_Aware_CBF(TA-CBF)
    python src/train.py
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(__file__))
from models import BPCBFModel
from cbf_qp import BPCBFController
from analytical_sdf import sdf_all_torch, sdf_critical_torch
from config import (
    STATE_DIM, DEVICE, LR, LR_BARRIER, BATCH_SIZE, N_POINTS, DT,
    LAMBDA_MSE, LAMBDA_V1, LAMBDA_V2, LAMBDA_B_FS, LAMBDA_B_OB, LAMBDA_B_DOT,
    ALPHA_CLF, GAMMA_CBF, DELTA_B, DELTA_V,
    OUTER_ITERS, INNER_EPOCHS,
    N_CEX, DEMO_WAYPOINTS, X_START, X_GOAL,
    sample_safe_set, sample_unsafe_set, sample_workspace,
    sdf_all_critical_np, is_in_critical_np,
    build_obstacle_set, sample_obstacle_interior, CRITICAL_SHAPES,
    INFLATE_MARGIN, SLAB_X, SLAB_Y, Z_CORRIDOR,
    transform_shape, random_transform, canonical_interior_cloud,
    sample_local_box, sdf_critical_shape_2d, _rot2d,
    BARRIER_SDF_K, BARRIER_SDF_CLAMP_OUT, BARRIER_SDF_CLAMP_IN, LAMBDA_B_REG,
    B_GRAD_MAX, LAMBDA_EIK,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR  = os.path.join(os.path.dirname(__file__), "..")
CKPT_DIR  = os.path.join(ROOT_DIR, "checkpoints")
PLOT_DIR  = os.path.join(ROOT_DIR, "results")
DATA_DIR  = os.path.join(ROOT_DIR, "data")
LOG_DIR   = os.path.join(ROOT_DIR, "runs")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

PHASE0_EPOCHS = 300   # pre-train f alone (Algorithm 2 step: "Train f_θ subject to loss (9)")

# Outer iterations after which the "f + Lyapunov only" checkpoint/plot is taken
# (barrier loss is included in every inner step from outer_iter=0, but its
# weight is ramped from 0 -> full so the f+V dynamics dominate early on).
LYAP_ONLY_OUTER_ITERS = 3


# ── Demo data loading and preprocessing ───────────────────────────────────────

def load_demos() -> list:
    """
    Load the 3D needle-tip demo trajectories from data/data_trajectory/.
    Falls back to procedurally generated weaving demos if no CSVs are present.
    """
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "data_trajectory", "demo_*.csv")))
    demos = []

    if csv_files:
        for f in csv_files:
            data = np.loadtxt(f, delimiter=",", dtype=np.float32)
            traj = data[:, 8:8 + STATE_DIM]   # needle-tip x,y,z
            demos.append(traj)
        print(f"[Data] Loaded {len(demos)} demo CSVs from {DATA_DIR}/data_trajectory/")
    else:
        print("[Data] No CSV demos found — generating synthetic weaving demos...")
        rng = np.random.default_rng(0)
        base = _resample(DEMO_WAYPOINTS, N_POINTS)
        for i in range(8):
            noise = rng.uniform(-0.010, 0.010, base.shape).astype(np.float32)
            # Some demos intentionally go closer to critical zones (mix of safe + near-unsafe)
            if i >= 5:
                noise *= 2.0
            demos.append(base + noise)
        print(f"[Data] Generated {len(demos)} synthetic demos.")

    return demos


def _resample(traj: np.ndarray, n: int) -> np.ndarray:
    """Resample trajectory to fixed length via linear interpolation."""
    T   = len(traj)
    old = np.linspace(0, T - 1, T)
    new = np.linspace(0, T - 1, n)
    return np.stack([np.interp(new, old, traj[:, d])
                     for d in range(traj.shape[1])], axis=1).astype(np.float32)


def compute_mean_demo(demos: list, n: int = N_POINTS) -> np.ndarray:
    """Mean demo path resampled to n points. Shape: (n, STATE_DIM)."""
    resampled = [_resample(d, n) for d in demos]
    return np.mean(resampled, axis=0)


def normalise(demos: list):
    """Zero-mean, unit-std across all demos."""
    all_pts = np.concatenate(demos, axis=0)
    mean    = all_pts.mean(axis=0)
    std     = all_pts.std(axis=0) + 1e-8
    normed  = [(d - mean) / std for d in demos]
    return normed, mean, std


# ── Neural ODE rollout loss (Eq. 9 in paper) ──────────────────────────────────

def rollout_rk4(f_model, x0: torch.Tensor, n_steps: int, dt: float,
                s_start: float = 0.0) -> torch.Tensor:
    """
    RK4 rollout of ẋ = f_θ(x, s).
    x0: (d,) → traj: (n_steps+1, d)
    """
    traj = [x0.unsqueeze(0)]
    x = x0.unsqueeze(0)   # (1, d)
    for k in range(n_steps):
        s_k = s_start + k / max(n_steps - 1, 1) * (1.0 - s_start)
        s_t = torch.tensor([[s_k]], dtype=x.dtype, device=x.device)
        k1 = f_model(x,          s_t)
        k2 = f_model(x + dt/2*k1, s_t)
        k3 = f_model(x + dt/2*k2, s_t)
        k4 = f_model(x + dt*k3,   s_t)
        x  = x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        traj.append(x)
    return torch.cat(traj, dim=0)   # (n_steps+1, d)


def imitation_loss(model, demos_raw: list, dt: float = DT) -> torch.Tensor:
    """
    L_MSE = (1/MT) Σ_i Σ_k ||x_i(t_k) - x̂_i(t_k)||²
    Rolls out f_θ from the demo start and compares to demo positions.
    """
    total = torch.tensor(0.0, device=DEVICE)
    for traj_np in demos_raw:
        traj_t  = torch.tensor(_resample(traj_np, N_POINTS),
                                dtype=torch.float32, device=DEVICE)
        x0      = traj_t[0]
        pred    = rollout_rk4(model.f, x0, N_POINTS - 1, dt)   # (N, d)
        total   = total + F.mse_loss(pred, traj_t)
    return total / len(demos_raw)


# ── S2-NNDS Loss functions (Eq. 10-11) ────────────────────────────────────────

def lyapunov_loss(model, S_pts: torch.Tensor, gamma: float = ALPHA_CLF,
                  delta: float = DELTA_V, return_components: bool = False):
    """
    L_lyap (Eq. 10):
      λ_l2 · Σ_{x∈S} relu(∇V·f + α·V)         [V̇ ≤ -α·V in S]
    (Positivity term removed — see comment below.)
    """
    S_pts  = S_pts.requires_grad_(True)
    s_val  = model.f.get_progress(S_pts)
    x_ref  = model.f.x_ref_at_s(s_val)
    V      = model.V(S_pts, x_ref, s_val)      # (B, 1)

    # V > 0 positivity: NOT a loss term. V = ||e||²·(1+δ·corr) ≥ ||e||² is
    # positive-definite BY CONSTRUCTION (zero exactly on the reference path,
    # positive elsewhere). Penalising relu(δ−V) here would demand V>δ even
    # AT the path where V must be 0 — an unsatisfiable objective that stalls
    # training. (Same reasoning removes the V_pos counterexample check.)

    # V̇ ≤ -α·V: compute ∇V·f
    gradV  = model.V.gradient(S_pts, x_ref, s_val)  # (B, d)
    f_val  = model.f(S_pts, s_val)                   # (B, d)
    vdot   = (gradV * f_val).sum(dim=-1, keepdim=True)
    l2 = F.relu(vdot + gamma * V).mean()

    loss = LAMBDA_V2 * l2
    if return_components:
        with torch.no_grad():
            # fraction of batch points violating the V̇ ≤ -α·V condition
            vdec_viol = (vdot + gamma * V > 0).float().mean().item()
        return loss, {'v_decr': l2.item(), 'v_decr_viol_frac': vdec_viol}
    return loss


def barrier_loss(model, S0_pts: torch.Tensor, Su_pts: torch.Tensor,
                 S_pts: torch.Tensor, gamma: float = GAMMA_CBF,
                 delta: float = DELTA_B, return_components: bool = False):
    """
    L_bar (Eq. 11):
      λ_b2 · Σ_{x∈X_0} relu(δ - B(x))          [B > δ in safe set]
      λ_b3 · Σ_{x∈X_u} relu(δ + B(x))          [B < -δ in unsafe set]
      λ_b1 · Σ_{x∈S, B≥0} relu(-(∇B·f + γ·B))  [Ḃ + γ·B ≥ 0 in the safe set]
    """
    B0   = model.B(S0_pts)                          # (B0, 1)
    Bu   = model.B(Su_pts)                          # (Bu, 1)

    l_safe   = F.relu(delta - B0).mean()
    l_unsafe = F.relu(delta + Bu).mean()

    # Barrier decrease: Ḃ + γ·B ≥ 0 — enforced ONLY in the safe set (B ≥ 0),
    # the standard CBF forward-invariance condition. Enforcing it inside the
    # unsafe region (B < 0) would require Ḃ ≥ γ|B| there (B forced to GROW
    # inside obstacles), whose only global solution is B≈0 everywhere — the
    # amplitude collapse that flattens the learned boundary.
    S_req = S_pts.detach().requires_grad_(True)
    B_S   = model.B(S_req)                          # (S, 1)
    gradB = model.B.gradient(S_req)                 # (S, d)
    s_val = model.f.get_progress(S_req)
    f_val = model.f(S_req, s_val)                   # (S, d)
    bdot  = (gradB * f_val).sum(dim=-1, keepdim=True)
    safe_mask = (B_S.detach() >= 0).float()
    l_dot = (F.relu(-(bdot + gamma * B_S)) * safe_mask).sum() \
            / safe_mask.sum().clamp(min=1.0)

    loss = LAMBDA_B_FS * l_safe + LAMBDA_B_OB * l_unsafe + LAMBDA_B_DOT * l_dot
    if return_components:
        with torch.no_grad():
            # classification accuracy: fraction of safe pts with B>+δ and
            # unsafe pts with B<-δ — the clearest signal of barrier quality.
            safe_acc   = (B0 >  delta).float().mean().item()
            unsafe_acc = (Bu < -delta).float().mean().item()
        return loss, {
            'b_safe':     l_safe.item(),
            'b_unsafe':   l_unsafe.item(),
            'b_dot':      l_dot.item(),
            'safe_acc':   safe_acc,
            'unsafe_acc': unsafe_acc,
            'B0_min':     B0.min().item(),
            'B0_max':     B0.max().item(),
            'Bu_min':     Bu.min().item(),
            'Bu_max':     Bu.max().item(),
        }
    return loss


def make_obstacle_tensors(k: int = 64, seed: int = 0) -> list:
    """Build the obstacle set as device tensors for CompositeBarrier."""
    obs_np = build_obstacle_set(k=k, seed=seed)
    return [{'cloud':  torch.tensor(o['cloud'],  dtype=torch.float32, device=DEVICE),
             'center': torch.tensor(o['center'], dtype=torch.float32, device=DEVICE)}
            for o in obs_np]


def composite_barrier_loss(model, S0_pts: torch.Tensor, U_list: list,
                           S_pts: torch.Tensor, gamma: float = GAMMA_CBF,
                           delta: float = DELTA_B, return_components: bool = False):
    """
    Loss for the CompositeBarrier (per-obstacle conditional CBF + smooth-min):
      - safe:   every per-obstacle bᵢ > δ on globally-safe points S0
      - unsafe: bᵢ < -δ on the interior of obstacle i (its own unsafe set)
      - decrease: composite Ḃ + γ·B ≥ 0 in the safe set (B ≥ 0)
    U_list: list (per obstacle) of (Nᵢ, d) interior-point tensors.
    """
    B = model.B   # CompositeBarrier (obstacles already installed via set_obstacles)

    # safe set: train the COMPOSITE barrier > δ on globally-safe points.
    # (Targeting the composite — not each bᵢ — lets the smooth-min offset be
    # absorbed: training pushes the nearest obstacle's bᵢ up until B itself
    # clears δ, instead of leaving B negative when every bᵢ only just reaches δ.)
    B_safe   = B(S0_pts)                                   # (|S0|, 1) composite
    l_safe   = F.relu(delta - B_safe).mean()

    # unsafe set: obstacle i's own CBF negative on its interior → composite ≤ bᵢ < -δ
    l_unsafe = 0.0
    for i, Ui in enumerate(U_list):
        b_ii = B.per_obstacle(Ui)[:, i:i + 1]             # (|Ui|, 1)
        l_unsafe = l_unsafe + F.relu(delta + b_ii).mean()
    l_unsafe = l_unsafe / max(len(U_list), 1)

    # composite CBF decrease, masked to the safe set (B ≥ 0)
    S_req = S_pts.detach().requires_grad_(True)
    B_S   = B(S_req)                                       # (|S|, 1) composite
    gradB = torch.autograd.grad(B_S.sum(), S_req, create_graph=True)[0]
    s_val = model.f.get_progress(S_req)
    f_val = model.f(S_req, s_val)
    bdot  = (gradB * f_val).sum(dim=-1, keepdim=True)
    mask  = (B_S.detach() >= 0).float()
    l_dot = (F.relu(-(bdot + gamma * B_S)) * mask).sum() / mask.sum().clamp(min=1.0)

    loss = LAMBDA_B_FS * l_safe + LAMBDA_B_OB * l_unsafe + LAMBDA_B_DOT * l_dot
    if return_components:
        with torch.no_grad():
            # classification accuracy over the composite barrier
            safe_acc   = (B(S0_pts) > delta).float().mean().item()
            Bu_all = torch.cat([B(Ui) for Ui in U_list], dim=0)
            unsafe_acc = (Bu_all < -delta).float().mean().item()
            B0c = B(S0_pts)
        return loss, {
            'b_safe':     l_safe.item(),
            'b_unsafe':   float(l_unsafe.item() if torch.is_tensor(l_unsafe) else l_unsafe),
            'b_dot':      l_dot.item(),
            'safe_acc':   safe_acc,
            'unsafe_acc': unsafe_acc,
            'B0_min':     B0c.min().item(),
            'B0_max':     B0c.max().item(),
            'Bu_min':     Bu_all.min().item(),
            'Bu_max':     Bu_all.max().item(),
        }
    return loss


# ── Pose/scale-augmented + inflated barrier loss ──────────────────────────────
#
# This REPLACES composite_barrier_loss in the inner loop. Two changes vs the
# original, addressing the generalization + strict-avoidance requirements:
#
#   (1) INFLATION: the unsafe set is the obstacle interior PLUS an outward shell
#       of width INFLATE_MARGIN, and safe points must lie beyond that shell. The
#       learned B=0 contour is therefore pushed ~INFLATE_MARGIN OUTSIDE the true
#       surface → strictly larger than the analytic CBF, a visible buffer the
#       field diverts around before the needle can touch the red zone.
#
#   (2) AUGMENTATION: every call, each obstacle is randomly rotated / scaled /
#       translated. The SAME transform is applied to the encoder point-cloud and
#       to the analytic SDF labels, so cbf(x_rel, e) learns to respond to the
#       shape EMBEDDING consistently at any pose. This fixes the OOD-embedding
#       failure (rotated obstacle → unseen e) and makes the barrier track a
#       slowly moving / growing / shrinking obstacle at inference.

def make_canonical_clouds(k: int = 128) -> list:
    """Centered interior point-clouds of every base shape (sampled once)."""
    return [canonical_interior_cloud(sh, k=k, seed=200 + i)
            for i, sh in enumerate(CRITICAL_SHAPES)]


def _augment_scene(canon_clouds, rng, k_cloud: int, n_shell: int,
                   inflate: float):
    """
    Draw one random pose+scale per obstacle and build, in numpy:
      obs       : list of {'cloud_xy' (k,2) centered, 'center3' (3,)}
      unsafe    : list per obstacle of (Nu,2) world XY pts (interior + shell)
      shapes_t  : transformed shape dicts (for global SDF labelling)
      nearsafe  : (Ns,2) world XY safe pts pooled near all obstacles
    """
    obs, unsafe, shapes_t, nearsafe = [], [], [], []
    for i, (sh, canon) in enumerate(zip(CRITICAL_SHAPES, canon_clouds)):
        d_rot, scale, d_trans = random_transform(rng)
        sh_t = transform_shape(sh, d_rot, scale, d_trans)
        shapes_t.append(sh_t)
        R    = _rot2d(d_rot)
        c_xy = sh_t['center'][:2].astype(np.float32)

        cloud_xy = (scale * canon) @ R.T                      # centered (M,2)
        if cloud_xy.shape[0] > k_cloud:
            sel = rng.choice(cloud_xy.shape[0], k_cloud, replace=False)
            cloud_enc = cloud_xy[sel]
        else:
            cloud_enc = cloud_xy
        interior_world = cloud_xy + c_xy                      # (M,2) inside

        # outward inflation shell (0 < sdf < inflate) + local safe (sdf > inflate)
        box3, box_sdf = sample_local_box(sh_t, n_shell, seed=int(rng.integers(1 << 30)))
        shell = box3[(box_sdf > 0) & (box_sdf < inflate)][:, :2]
        nearsafe.append(box3[box_sdf > inflate][:, :2])

        unsafe_i = np.concatenate([interior_world, shell], axis=0) \
            if len(shell) else interior_world
        obs.append({'cloud_xy': cloud_enc.astype(np.float32),
                    'center3':  sh_t['center'].astype(np.float32)})
        unsafe.append(unsafe_i.astype(np.float32))
    nearsafe = (np.concatenate(nearsafe, axis=0)
                if any(len(n) for n in nearsafe) else np.zeros((0, 2), np.float32))
    return obs, unsafe, shapes_t, nearsafe


def _sdf_min_over(shapes_t: list, pts_xy: np.ndarray) -> np.ndarray:
    sdfs = [sdf_critical_shape_2d(pts_xy, sh) for sh in shapes_t]
    return np.min(np.stack(sdfs, axis=1), axis=1)


def augmented_barrier_loss(model, canon_clouds, rng,
                           gamma: float = GAMMA_CBF, inflate: float = INFLATE_MARGIN,
                           delta: float = DELTA_B, k_cloud: int = 64,
                           n_box: int = 320, n_global: int = 500,
                           return_components: bool = False):
    """
    SDF-shaped, pose/scale-augmented, inflated composite-barrier loss.

      target(x) = K · clip(sdf(x) − INFLATE, −CLAMP, +CLAMP)

    Per obstacle i, bᵢ is regressed onto its OWN inflated-clamped SDF (in a local
    box around the transformed obstacle); the COMPOSITE B is regressed onto the
    global min-SDF target. This makes B an approximate signed-distance field —
    smooth, with ‖∇B‖≈K through a finite boundary layer — so the CBF-QP gets a
    usable gradient and actually diverts the field. The forward-invariance
    decrease term Ḃ+γB≥0 is kept on the safe set.
    """
    B   = model.B
    dev = DEVICE
    K = BARRIER_SDF_K
    CL_IN, CL_OUT = BARRIER_SDF_CLAMP_IN, BARRIER_SDF_CLAMP_OUT

    def _target(sdf):
        return K * np.clip(sdf - inflate, -CL_IN, CL_OUT).astype(np.float32)

    # ── build transformed obstacles + per-obstacle sample sets ────────────────
    # Each obstacle's set is balanced: ~half strictly-interior points (from the
    # transformed canonical cloud, target ≈ −K·CLAMP) and ~half a local box
    # (mostly exterior). Without the explicit interior half the box is ~85%
    # exterior, the MSE is dominated by positive targets, and B collapses to a
    # positive constant (interior never learned). The interior set also feeds
    # the b<−δ hinge that guarantees the classification margin.
    # Randomize the obstacle COUNT and which shapes appear each step (1..all),
    # so the composite smooth-min is trained to fuse an ARBITRARY number/mix of
    # obstacles — multi-obstacle robustness, not just the fixed canonical 6.
    n_all = len(CRITICAL_SHAPES)
    n_sel = int(rng.integers(max(1, n_all - 4), n_all + 1))
    sel   = rng.choice(n_all, size=n_sel, replace=False)

    shapes_t, obstacles, local, interior = [], [], [], []
    for sh_idx in sel:
        sh, canon = CRITICAL_SHAPES[sh_idx], canon_clouds[sh_idx]
        d_rot, scale, d_trans = random_transform(rng)
        sh_t = transform_shape(sh, d_rot, scale, d_trans); shapes_t.append(sh_t)
        cloud_xy = (scale * canon) @ _rot2d(d_rot).T          # centered, interior
        cloud_enc = (cloud_xy[rng.choice(cloud_xy.shape[0], k_cloud, replace=False)]
                     if cloud_xy.shape[0] > k_cloud else cloud_xy)
        obstacles.append({'cloud':  torch.tensor(cloud_enc.astype(np.float32), device=dev),
                          'center': torch.tensor(sh_t['center'].astype(np.float32), device=dev)})
        # strictly-interior world points (sdf < 0) + their SDF
        int_xy  = (cloud_xy + sh_t['center'][:2]).astype(np.float32)
        int_sdf = sdf_critical_shape_2d(int_xy, sh_t)
        int3    = np.concatenate([int_xy, np.full((len(int_xy), 1), Z_CORRIDOR, np.float32)], axis=1)
        interior.append((int3, int_sdf))
        # local box (mostly exterior) for the boundary ramp
        pts3, sdf = sample_local_box(sh_t, n_box, seed=int(rng.integers(1 << 30)), pad=0.013)
        # BOUNDARY-BAND densification: jitter the interior cloud outward so points
        # land right around the (possibly thin/concave) true boundary — this is
        # where the B=0 contour must be pinned. Uniform box sampling under-covers
        # thin features (crescent horns), which caused the residual graze.
        band_xy  = int_xy + rng.uniform(-0.012, 0.012, int_xy.shape).astype(np.float32)
        band_sdf = sdf_critical_shape_2d(band_xy, sh_t)
        band3    = np.concatenate([band_xy, np.full((len(band_xy), 1), Z_CORRIDOR, np.float32)], axis=1)
        pts3 = np.concatenate([pts3, band3], axis=0)
        sdf  = np.concatenate([sdf, band_sdf], axis=0)
        local.append((pts3, sdf))

    # ── per-obstacle SDF regression bᵢ → K·clip(sdfᵢ−INFLATE) + UNSAFE hinge ──
    # The unsafe hinge (bᵢ < −δ) covers the obstacle interior AND the inflation
    # SHELL (0 < sdf < INFLATE). Supervising the shell is what pins the learned
    # B=0 contour to the INFLATED boundary; without it the contour drifts inside
    # the true surface on concave shapes under rotation (the residual grazing
    # penetration, learnedB>0 where sdf<0).
    l_reg = 0.0
    l_uns = 0.0
    for i in range(len(local)):
        pts3, sdf     = local[i]
        int3, int_sdf = interior[i]
        Xi   = torch.tensor(np.concatenate([pts3, int3], axis=0), device=dev)
        tsdf = np.concatenate([sdf, int_sdf], axis=0)
        tgt  = torch.tensor(_target(tsdf), device=dev).unsqueeze(-1)
        b_ii = B.per_obstacle(Xi, obstacles)[:, i:i + 1]
        l_reg = l_reg + F.mse_loss(b_ii, tgt)
        # unsafe hinge over everything within the inflated boundary (sdf < INFLATE)
        uns_mask = np.concatenate([sdf, int_sdf], axis=0) < inflate
        b_uns = b_ii[torch.tensor(uns_mask, device=dev)]
        if b_uns.numel() > 0:
            l_uns = l_uns + F.relu(delta + b_uns).mean()
    l_reg = l_reg / max(len(local), 1)
    l_uns = l_uns / max(len(local), 1)

    # ── composite SDF regression over a global set (uniform slab + all local) ──
    g_xy = np.random.default_rng(int(rng.integers(1 << 30))).uniform(
        [SLAB_X[0], SLAB_Y[0]], [SLAB_X[1], SLAB_Y[1]], (n_global, 2)).astype(np.float32)
    g_xy = np.concatenate([g_xy] + [p[:, :2] for p, _ in local], axis=0)
    g_sdf = _sdf_min_over(shapes_t, g_xy)
    g3 = np.concatenate([g_xy, np.full((len(g_xy), 1), Z_CORRIDOR, np.float32)], axis=1)
    Xg  = torch.tensor(g3, device=dev)
    tgt_g = torch.tensor(_target(g_sdf), device=dev).unsqueeze(-1)
    B_g  = B(Xg, obstacles)
    l_comp = F.mse_loss(B_g, tgt_g)
    # margin hinge on clearly-safe global points: B > δ
    safe_hinge_m = torch.tensor(g_sdf > inflate + 0.004, device=dev).unsqueeze(-1)
    l_safe = (F.relu(delta - B_g) * safe_hinge_m).sum() / safe_hinge_m.sum().clamp(min=1)

    # ── composite CBF decrease Ḃ + γ·B ≥ 0 on the safe set (sdf > INFLATE) ────
    safe_np = g3[g_sdf > inflate]
    if len(safe_np) < 8:
        safe_np = g3[np.argsort(-g_sdf)[:64]]
    S_req = torch.tensor(safe_np, device=dev).requires_grad_(True)
    B_S   = B(S_req, obstacles)
    gradB = torch.autograd.grad(B_S.sum(), S_req, create_graph=True)[0]
    s_val = model.f.get_progress(S_req)
    f_val = model.f(S_req, s_val)
    bdot  = (gradB * f_val).sum(dim=-1, keepdim=True)
    mask  = (B_S.detach() >= 0).float()
    l_dot = (F.relu(-(bdot + gamma * B_S)) * mask).sum() / mask.sum().clamp(min=1.0)

    # ── Eikonal: pin ‖∇B‖ TO the target slope K in the boundary band (and to 0
    # in the clamped flat regions). Merely CAPPING the slope let it vary 4–12×,
    # so B=margin mapped to wildly different sdf on different obstacles (6→23 mm)
    # — the needle's standoff was then inconsistent and dipped into the light-red
    # zone. Pinning the slope makes B a faithful scaled SDF (B=margin ⇒
    # sdf=INFLATE+margin/K everywhere), so the standoff is uniform and reliable.
    Xr   = Xg.detach().requires_grad_(True)
    Br   = B(Xr, obstacles)
    gBr  = torch.autograd.grad(Br.sum(), Xr, create_graph=True)[0]
    gnorm = gBr.norm(dim=-1)
    l_eik = F.relu(gnorm - B_GRAD_MAX).mean()

    loss = (LAMBDA_B_REG * (l_reg + l_comp)
            + LAMBDA_B_FS * l_safe + LAMBDA_B_OB * l_uns
            + LAMBDA_B_DOT * l_dot + LAMBDA_EIK * l_eik)
    if return_components:
        with torch.no_grad():
            safe_m  = g_sdf > inflate + 0.004
            uns_m   = g_sdf < 0.0
            Bg_np   = B_g.squeeze(-1).cpu().numpy()
            safe_acc = float((Bg_np[safe_m] > delta).mean()) if safe_m.any() else 0.0
            uns_acc  = float((Bg_np[uns_m] < -delta).mean()) if uns_m.any() else 0.0
            gnorm_rep = gnorm.mean().item()
            comp = {
                'b_safe':   l_safe.item(),
                'b_unsafe': l_uns.item(),
                'b_dot':    l_dot.item(),
                'b_reg':    (l_reg + l_comp).item(),
                'safe_acc': safe_acc,
                'unsafe_acc': uns_acc,
                'B0_min':   float(Bg_np[safe_m].min()) if safe_m.any() else 0.0,
                'B0_max':   float(Bg_np.max()),
                'Bu_min':   float(Bg_np.min()),
                'Bu_max':   float(Bg_np[uns_m].max()) if uns_m.any() else 0.0,
                'gradB':    gnorm_rep,
            }
        return loss, comp
    return loss


# ── Counterexample checking (Algorithm 2) ─────────────────────────────────────

def find_counterexamples(model, n_samples: int = N_CEX,
                         seed: int = 0) -> tuple:
    """
    Sample n_samples i.i.d. states from X, check conditions (3)-(6).
    Returns:
        no_cex (int): number of violating states
        S_viol (np.ndarray): (n_viol, d) violating states
        split (dict): violation counts per condition
    """
    pts_np = sample_workspace(n_samples, seed=seed)
    pts    = torch.tensor(pts_np, dtype=torch.float32, device=DEVICE)

    model.eval()
    with torch.enable_grad():
        s_val  = model.f.get_progress(pts)
        x_ref  = model.f.x_ref_at_s(s_val)
        V      = model.V(pts, x_ref, s_val).squeeze(-1)        # (N,)
        B      = model.B(pts).squeeze(-1)                       # (N,)
        gradV  = model.V.gradient(pts, x_ref, s_val)
        gradB  = model.B.gradient(pts)
        f_val  = model.f(pts, s_val)
        vdot   = (gradV * f_val).sum(dim=-1)
        bdot   = (gradB * f_val).sum(dim=-1)

    if not (torch.isfinite(V).all() and torch.isfinite(B).all()
            and torch.isfinite(vdot).all() and torch.isfinite(bdot).all()):
        raise RuntimeError(
            "[CEX] Model produced NaN/Inf outputs — training has diverged. "
            "Aborting before a corrupted model is reported as 'converged'.")

    # Condition (3): V > 0 except on the reference path — satisfied BY
    # CONSTRUCTION (V = ||e||²·(1+δ·corr) ≥ ||e||², zero only at e=0).
    # Checking V < δ on uniform samples would permanently flag every point
    # within sqrt(δ) of the path (where V SHOULD be ~0), so it is skipped.
    viol_v1 = np.zeros(len(pts_np), dtype=bool)

    # Condition (4): V̇ ≤ -α·V
    viol_v2 = (vdot.detach() + ALPHA_CLF * V.detach() > 0).cpu().numpy()

    # Condition (5): B > δ in safe region. With the INFLATED boundary the safe
    # region starts beyond the outward shell, so require sdf > INFLATE_MARGIN.
    sdf_c   = sdf_all_critical_np(pts_np)
    is_safe = sdf_c > INFLATE_MARGIN + 0.004
    viol_b1 = is_safe & (B.detach().cpu().numpy() < DELTA_B)

    # Condition (6): Ḃ + γ·B ≥ 0 — only required in the safe set (B ≥ 0),
    # matching barrier_loss. Inside obstacles B<0 is the DESIRED state.
    B_np    = B.detach().cpu().numpy()
    viol_b2 = (((bdot + GAMMA_CBF * B).detach().cpu().numpy() < 0)
               & (B_np >= 0))

    # Condition: B < -δ in critical zones
    is_unsafe = sdf_c < -0.002
    viol_b3   = is_unsafe & (B.detach().cpu().numpy() > -DELTA_B)

    all_viol  = viol_v1 | viol_v2 | viol_b1 | viol_b2 | viol_b3
    S_viol    = pts_np[all_viol]
    no_cex    = int(all_viol.sum())
    split = {
        'V_pos':   int(viol_v1.sum()),
        'V_decr':  int(viol_v2.sum()),
        'B_safe':  int(viol_b1.sum()),
        'B_decr':  int(viol_b2.sum()),
        'B_unsaf': int(viol_b3.sum()),
    }
    model.train()
    return no_cex, S_viol, split


# ── Main training routine (Algorithm 2) ────────────────────────────────────────

def train(resume_from_lyapunov: bool = False):
    print(f"[Train] Device: {DEVICE}")
    writer = SummaryWriter(log_dir=LOG_DIR)

    # ── Data ──────────────────────────────────────────────────────────────────
    demos_raw = load_demos()
    mean_path = compute_mean_demo(demos_raw, N_POINTS)    # (N_POINTS, d) raw
    demos_normed, mean_np, std_np = normalise(demos_raw)

    mean_t = torch.tensor(mean_np, dtype=torch.float32)
    std_t  = torch.tensor(std_np,  dtype=torch.float32)

    # Reference path in raw space (model.f will use it directly)
    ref_path_raw = mean_path   # (N_POINTS, d)

    np.save(os.path.join(CKPT_DIR, "norm_mean.npy"), mean_np)
    np.save(os.path.join(CKPT_DIR, "norm_std.npy"),  std_np)
    np.save(os.path.join(CKPT_DIR, "ref_path.npy"),  ref_path_raw)
    print(f"[Train] mean={mean_np.round(4)}, std={std_np.round(4)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = BPCBFModel(ref_path=ref_path_raw).to(DEVICE)
    model.set_norm(mean_t.to(DEVICE), std_t.to(DEVICE))
    model.f.set_reference(ref_path_raw)

    # Composite barrier: install the obstacle set (point-clouds + centers) and
    # per-obstacle interior (unsafe) sets used by composite_barrier_loss.
    obstacles = make_obstacle_tensors(k=64, seed=0)
    model.set_obstacles(obstacles)
    # Canonical (centered) interior clouds — sampled ONCE, then transformed
    # analytically each step for pose/scale augmentation (see augmented_barrier_loss).
    canon_clouds = make_canonical_clouds(k=128)
    aug_rng = np.random.default_rng(1234)
    print(f"[Train] Composite barrier: {len(obstacles)} obstacles | "
          f"inflation={INFLATE_MARGIN*1e3:.0f}mm | pose+scale augmentation ON")

    opt_f = torch.optim.Adam(model.f.parameters(), lr=LR)
    opt_V = torch.optim.Adam(model.V.parameters(), lr=LR)
    # B is the CompositeBarrier — trained FRESH (random init) during the
    # barrier-active outer iters, so it needs a healthy LR (≫ the f/V
    # fine-tuning rate) and a gentle cosine schedule (no StepLR cliff).
    opt_B = torch.optim.Adam(model.B.parameters(), lr=LR_BARRIER)

    sched_f = torch.optim.lr_scheduler.CosineAnnealingLR(opt_f, T_max=PHASE0_EPOCHS + OUTER_ITERS * INNER_EPOCHS)
    sched_V = torch.optim.lr_scheduler.StepLR(opt_V, step_size=100, gamma=0.5)
    sched_B = torch.optim.lr_scheduler.CosineAnnealingLR(opt_B, T_max=OUTER_ITERS * INNER_EPOCHS)

    # Working sets (augmented with counterexamples during outer loop)
    S0_np = sample_safe_set(n=2000, seed=0)
    Su_np = sample_unsafe_set(n=1000, seed=1)
    S_np  = sample_workspace(n=3000, seed=2)

    log = {'phase0': [], 'imitation': [], 'lyapunov': [], 'barrier': [], 'total': []}

    start_outer_iter = 0
    global_step = 0
    lyap_only_saved = False

    if resume_from_lyapunov:
        # Skip Phase 0 + the Lyapunov-only outer iters: load the checkpoint
        # saved at the end of that stage (f_θ/V_θ' already converged) and
        # jump straight to the barrier-active outer iterations with the
        # CURRENT config's DELTA_B/LAMBDA_B_* values.
        ckpt_path = os.path.join(CKPT_DIR, "f_lyapunov_only.pt")
        sd = torch.load(ckpt_path, map_location=DEVICE)
        # Keep f_θ/V_θ' (converged); drop B.* — the barrier is now the
        # CompositeBarrier (different architecture from the old BarrierNet),
        # trained fresh from the start of the barrier-active outer iterations.
        sd = {k: v for k, v in sd.items() if not k.startswith('B.')}
        model.load_state_dict(sd, strict=False)
        model.f.set_reference(ref_path_raw)  # ensure buffers match this run
        print(f"[Resume] Loaded {ckpt_path} (f_θ/V_θ' only; composite B trained "
              f"fresh) — skipping Phase 0 and Lyapunov-only iters 1-{LYAP_ONLY_OUTER_ITERS}.")
        start_outer_iter = LYAP_ONLY_OUTER_ITERS
        lyap_only_saved = True
        # Re-sync the f/V schedulers (those nets continue from the checkpoint).
        # Do NOT advance sched_B: the composite barrier trains FRESH from here,
        # so its cosine LR must start at full LR_BARRIER, not pre-decayed.
        for _ in range(PHASE0_EPOCHS):
            sched_f.step()
        for _ in range(LYAP_ONLY_OUTER_ITERS * INNER_EPOCHS):
            sched_f.step(); sched_V.step()
        global_step = LYAP_ONLY_OUTER_ITERS * INNER_EPOCHS
    else:
        # ─────────────────────────────────────────────────────────────────────
        # PHASE 0: Pre-train f_θ alone on imitation loss
        # (Algorithm 2: "Train f_θ subject to loss (9)")
        # ─────────────────────────────────────────────────────────────────────
        print(f"\n[Phase 0] Pre-training f_θ for {PHASE0_EPOCHS} epochs...")
        model.train()
        for ep in range(1, PHASE0_EPOCHS + 1):
            opt_f.zero_grad()
            loss_imit = LAMBDA_MSE * imitation_loss(model, demos_raw)
            loss_imit.backward()
            nn.utils.clip_grad_norm_(model.f.parameters(), 1.0)
            opt_f.step()
            sched_f.step()
            log['phase0'].append(loss_imit.item())
            writer.add_scalar("phase0/imitation_loss", loss_imit.item(), ep)
            if ep % 50 == 0 or ep == 1:
                print(f"  ep {ep:4d}/{PHASE0_EPOCHS} | imit={loss_imit.item():.5f}")

        torch.save(model.state_dict(), os.path.join(CKPT_DIR, "phase0_f.pt"))
        print("[Phase 0] Done — f_θ pre-trained.\n")

    # ─────────────────────────────────────────────────────────────────────────
    # OUTER LOOP: S2-NNDS Algorithm 2
    # ─────────────────────────────────────────────────────────────────────────
    best_total = float("inf")
    no_cex_final = None

    for outer_iter in range(start_outer_iter, OUTER_ITERS):
        print(f"[Outer {outer_iter+1}/{OUTER_ITERS}] "
              f"S={len(S_np)} S0={len(S0_np)} Su={len(Su_np)}")

        # Prepare batch tensors from working sets
        S0_t = torch.tensor(S0_np, dtype=torch.float32, device=DEVICE)
        Su_t = torch.tensor(Su_np, dtype=torch.float32, device=DEVICE)
        S_t  = torch.tensor(S_np,  dtype=torch.float32, device=DEVICE)

        # Ramp the barrier-loss weight in over the first LYAP_ONLY_OUTER_ITERS
        # outer iterations, so f_θ/V_θ' settle on the imitation+Lyapunov
        # objective first ("f + Lyapunov" stage) before B_φ starts pulling
        # the dynamics. Checkpoint+plot taken at the end of that stage.
        bar_weight = 0.0 if outer_iter < LYAP_ONLY_OUTER_ITERS else 1.0

        # ── INNER LOOP: joint training of f_θ, V_θ', B_φ ──────────────────
        for ep in range(1, INNER_EPOCHS + 1):
            global_step += 1
            model.train()
            # Mini-batch from each set
            idx0 = torch.randperm(len(S0_t))[:BATCH_SIZE]
            idx  = torch.randperm(len(S_t)) [:BATCH_SIZE]
            s0_b = S0_t[idx0]
            s_b  = S_t[idx]

            # f imitation
            opt_f.zero_grad()
            l_imit = LAMBDA_MSE * imitation_loss(model, demos_raw)

            # V Lyapunov
            opt_V.zero_grad()
            l_lyap, lyap_c = lyapunov_loss(model, s_b, return_components=True)

            # B Barrier — pose/scale-AUGMENTED + INFLATED composite neural CBF.
            # A freshly randomized obstacle scene each step → pose-invariant,
            # size-robust barrier whose B=0 contour sits INFLATE_MARGIN outside
            # the true surface.
            opt_B.zero_grad()
            l_bar, bar_c = augmented_barrier_loss(model, canon_clouds, aug_rng,
                                                  return_components=True)

            total = l_imit + l_lyap + bar_weight * l_bar
            total.backward()

            nn.utils.clip_grad_norm_(model.f.parameters(), 1.0)
            nn.utils.clip_grad_norm_(model.V.parameters(), 1.0)
            nn.utils.clip_grad_norm_(model.B.parameters(), 1.0)

            opt_f.step(); opt_V.step(); opt_B.step()
            sched_f.step(); sched_V.step(); sched_B.step()

            log['imitation'].append(l_imit.item())
            log['lyapunov'].append(l_lyap.item())
            log['barrier'].append(l_bar.item())
            log['total'].append(total.item())

            writer.add_scalar("train/imitation_loss", l_imit.item(), global_step)
            writer.add_scalar("train/lyapunov_loss",   l_lyap.item(), global_step)
            writer.add_scalar("train/barrier_loss",    l_bar.item(),  global_step)
            writer.add_scalar("train/total_loss",      total.item(), global_step)
            writer.add_scalar("train/barrier_weight",  bar_weight,    global_step)
            writer.add_scalar("train/lr_f", opt_f.param_groups[0]['lr'], global_step)
            # component + diagnostic scalars
            writer.add_scalar("lyap/v_decr",          lyap_c['v_decr'],          global_step)
            writer.add_scalar("lyap/v_decr_viol_frac", lyap_c['v_decr_viol_frac'], global_step)
            writer.add_scalar("barrier/l_safe",       bar_c['b_safe'],   global_step)
            writer.add_scalar("barrier/l_unsafe",     bar_c['b_unsafe'], global_step)
            writer.add_scalar("barrier/l_dot",        bar_c['b_dot'],    global_step)
            writer.add_scalar("barrier/safe_acc",     bar_c['safe_acc'],   global_step)
            writer.add_scalar("barrier/unsafe_acc",   bar_c['unsafe_acc'], global_step)
            writer.add_scalar("barrier/B0_min",       bar_c['B0_min'],   global_step)
            writer.add_scalar("barrier/Bu_max",       bar_c['Bu_max'],   global_step)

            if ep % 10 == 0:
                print(f"  inner {ep:3d}/{INNER_EPOCHS} | "
                      f"imit={l_imit.item():.4f} | "
                      f"lyap={l_lyap.item():.4f}(vdec_viol={lyap_c['v_decr_viol_frac']*100:.0f}%) | "
                      f"bar={l_bar.item():.4f}[safe={bar_c['b_safe']:.3f} uns={bar_c['b_unsafe']:.3f} dot={bar_c['b_dot']:.3f}] "
                      f"acc[safe={bar_c['safe_acc']*100:.0f}% uns={bar_c['unsafe_acc']*100:.0f}%] "
                      f"B0=[{bar_c['B0_min']:+.3f},{bar_c['B0_max']:+.3f}] Bu=[{bar_c['Bu_min']:+.3f},{bar_c['Bu_max']:+.3f}] "
                      f"(w={bar_weight}) total={total.item():.4f}")

            # Track best ONLY while the barrier is active (bar_weight>0). During
            # the ramp-in (bar_weight=0) the total is just imitation+Lyapunov and
            # is far smaller than any barrier-active total, so comparing across
            # the two phases would freeze a model with an UNTRAINED barrier.
            if bar_weight > 0 and total.item() < best_total:
                best_total = total.item()
                model.save(os.path.join(CKPT_DIR, "best_model.pt"), quiet=True)

        # ── Counterexample sampling ────────────────────────────────────────
        no_cex, S_viol, split = find_counterexamples(
            model, n_samples=N_CEX, seed=outer_iter)
        no_cex_final = no_cex
        print(f"  [CEX] no_cex={no_cex} | {split}")
        writer.add_scalar("cex/no_cex", no_cex, outer_iter)
        for k, v in split.items():
            writer.add_scalar(f"cex/{k}", v, outer_iter)

        # Save "f + Lyapunov only" checkpoint and vector-field plot at the
        # end of the Lyapunov-only stage (before B_φ influences training).
        if not lyap_only_saved and outer_iter == LYAP_ONLY_OUTER_ITERS - 1:
            model.save(os.path.join(CKPT_DIR, "f_lyapunov_only.pt"), quiet=True)
            _plot_vector_field(model, mean_np, std_np,
                                tag="f_lyapunov_only",
                                title_suffix="f_θ + V_θ' trained (B_φ not yet active)")
            print("[Stage] Saved f+Lyapunov checkpoint and vector field plot.")
            lyap_only_saved = True

        if no_cex == 0:
            print("[Algorithm 2] No counterexamples — convergence!")
            break

        # Augment working sets with violating states
        if len(S_viol) > 0:
            S_np  = np.concatenate([S_np,  S_viol], axis=0)
            # Add to S0 if they are in the safe region, Su if unsafe
            is_unsafe = is_in_critical_np(S_viol)
            S0_viol   = S_viol[~is_unsafe]
            Su_viol   = S_viol[is_unsafe]
            if len(S0_viol) > 0:
                S0_np = np.concatenate([S0_np, S0_viol], axis=0)
            if len(Su_viol) > 0:
                Su_np = np.concatenate([Su_np, Su_viol], axis=0)
            print(f"  [CEX] Added {len(S_viol)} pts: "
                  f"{len(S0_viol)} to S0, {len(Su_viol)} to Su, all to S")

    # ── Final save ────────────────────────────────────────────────────────────
    model.save(os.path.join(CKPT_DIR, "final_model.pt"))
    print(f"\n[Train] Complete. Best total loss: {best_total:.5f}")
    print(f"[Train] Final no_cex: {no_cex_final}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_losses(log)
    _plot_demo_reproduction(model, demos_raw)
    _plot_learned_barrier(model, mean_np, std_np)
    _plot_vector_field(model, mean_np, std_np,
                        tag="f_lyapunov_barrier_final",
                        title_suffix="f_θ + V_θ' + B_φ trained (final, with CBF)")
    _plot_closed_loop_field(model, mean_np, std_np,
                             tag="closed_loop_cbf",
                             title_suffix="f_θ(x) + u_safe(x)  (CBF-QP corrected)")

    print(f"[Train] Plots saved to {PLOT_DIR}/")
    writer.close()


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_losses(log: dict):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, key in zip(axes, ['imitation', 'lyapunov', 'barrier', 'total']):
        vals = log[key]
        if vals:
            ax.plot(vals)
            ax.set_title(key)
            ax.set_yscale('log')
            ax.grid(True)
            ax.set_xlabel("inner step")
    plt.suptitle("S2-NNDS Training Losses")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "training_losses.png"), dpi=150)
    plt.close()


def _draw_critical_shapes(ax):
    """Overlay the non-linear critical obstacle shapes on an axes."""
    from config import CRITICAL_SHAPES, sdf_all_critical_np
    import matplotlib.patches as mpatches

    nx, ny = 200, 200
    from config import SLAB_X, SLAB_Y
    xs = np.linspace(SLAB_X[0], SLAB_X[1], nx)
    ys = np.linspace(SLAB_Y[0], SLAB_Y[1], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)
    sdf_c = sdf_all_critical_np(pts).reshape(ny, nx)
    ax.contourf(xs, ys, sdf_c, levels=[-1, 0], colors=['#ff4444'], alpha=0.5, zorder=2)
    ax.contour(xs, ys, sdf_c, levels=[0], colors=['#cc0000'], linewidths=1.5, zorder=3)
    from config import FOAM_CENTRES, FOAM_RADIUS
    for fc in FOAM_CENTRES[::4]:
        circle = plt.Circle((fc[0], fc[1]), FOAM_RADIUS, color='#ffa500', alpha=0.2, zorder=1)
        ax.add_patch(circle)


def _plot_demo_reproduction(model, demos_raw: list):
    model.eval()
    fig, axes = plt.subplots(1, min(3, len(demos_raw)), figsize=(5 * min(3, len(demos_raw)), 5))
    if len(demos_raw) == 1:
        axes = [axes]
    for i, (ax, demo) in enumerate(zip(axes, demos_raw[:3])):
        _draw_critical_shapes(ax)
        traj_gt = _resample(demo, N_POINTS)
        x0 = torch.tensor(traj_gt[0:1], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            pred = rollout_rk4(model.f, x0[0], N_POINTS - 1, DT).cpu().numpy()
        ax.plot(traj_gt[:, 0], traj_gt[:, 1], 'k--', lw=1.5, label='Demo', zorder=5)
        ax.plot(pred[:, 0], pred[:, 1], 'g-', lw=2, label='f_θ rollout', zorder=6)
        ax.scatter(traj_gt[0, 0], traj_gt[0, 1], c='blue', s=80, zorder=7, label='Start')
        ax.scatter(traj_gt[-1, 0], traj_gt[-1, 1], c='green', s=80, marker='*', zorder=7, label='Goal')
        ax.set_title(f"Demo {i+1} reproduction")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.legend(fontsize=8); ax.grid(True)
        ax.set_aspect("equal")
    plt.suptitle("Demo Path Reproduction (f_θ rollout vs expert demo)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "demo_reproduction.png"), dpi=150)
    plt.close()


def _plot_learned_barrier(model, mean_np, std_np):
    """Plot the learned barrier B_φ(x) on the 2D scene."""
    from config import SLAB_X, SLAB_Y, Z_CORRIDOR
    model.eval()
    nx, ny = 200, 200
    xs = np.linspace(SLAB_X[0], SLAB_X[1], nx)
    ys = np.linspace(SLAB_Y[0], SLAB_Y[1], ny)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = np.full_like(XX, Z_CORRIDOR)
    pts = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1).astype(np.float32)
    pts_t = torch.tensor(pts, device=DEVICE)
    with torch.no_grad():
        B_vals = model.B(pts_t).squeeze(-1).cpu().numpy().reshape(ny, nx)

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(xs, ys, B_vals, levels=20, cmap='RdYlGn')
    ax.contour(xs, ys, B_vals, levels=[0], colors='black', linewidths=2, linestyles='--')
    plt.colorbar(cf, ax=ax, label='B_φ(x)')
    _draw_critical_shapes(ax)
    ref_path = model.f.ref_path.cpu().numpy()
    ax.plot(ref_path[:, 0], ref_path[:, 1], 'b-', lw=2, label='Demo path (ref)', zorder=8)
    ax.scatter(X_START[0], X_START[1], c='blue', s=100, zorder=9, label='Start')
    ax.scatter(X_GOAL[0],  X_GOAL[1],  c='lime', s=100, marker='*', zorder=9, label='Goal')
    ax.set_title("Learned Barrier B_φ(x)\n(green=safe, red=unsafe, dashed=boundary)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.legend(fontsize=9); ax.grid(True); ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "learned_barrier.png"), dpi=150)
    plt.close()


def _plot_vector_field(model, mean_np, std_np,
                        tag: str = "trajectory_attractor",
                        title_suffix: str = ""):
    """
    Plot the vector field f_θ(x, s) on the 2D scene.
    The entire demo path is an attractor — this is the key distinction from
    single-goal dynamical systems (where only the endpoint is stable).
    The plot should show streamlines converging TO the demo path from all
    surrounding directions, not just converging to the goal point.

    `tag` controls the output filename (vector_field_<tag>.png); `title_suffix`
    is appended to the plot title to distinguish training stages.
    """
    from config import SLAB_X, SLAB_Y, Z_CORRIDOR
    model.eval()

    nx, ny = 24, 20
    xs = np.linspace(SLAB_X[0] + 0.005, SLAB_X[1] - 0.005, nx)
    ys = np.linspace(SLAB_Y[0] + 0.005, SLAB_Y[1] - 0.005, ny)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = np.full(XX.shape, Z_CORRIDOR, dtype=np.float32)
    pts = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1).astype(np.float32)
    pts_t = torch.tensor(pts, device=DEVICE)
    with torch.no_grad():
        s_vals = model.f.get_progress(pts_t)
        vel    = model.f(pts_t, s_vals).cpu().numpy()

    U = vel[:, 0].reshape(ny, nx)
    V = vel[:, 1].reshape(ny, nx)
    speed = np.sqrt(U**2 + V**2) + 1e-6
    U /= speed; V /= speed   # normalise for clean arrow plot

    fig, ax = plt.subplots(figsize=(10, 7))
    _draw_critical_shapes(ax)
    ax.quiver(XX, YY, U, V, speed.reshape(ny, nx),
              cmap='Blues', scale=35, width=0.003, alpha=0.8, zorder=4)

    # Streamlines from a grid of initial conditions — shows convergence to path
    x_fine = np.linspace(SLAB_X[0], SLAB_X[1], 60)
    y_fine = np.linspace(SLAB_Y[0], SLAB_Y[1], 50)
    XF, YF = np.meshgrid(x_fine, y_fine)
    ZF = np.full(XF.shape, Z_CORRIDOR, dtype=np.float32)
    pts_f  = np.stack([XF.ravel(), YF.ravel(), ZF.ravel()], axis=1).astype(np.float32)
    pts_ft = torch.tensor(pts_f, device=DEVICE)
    with torch.no_grad():
        s_f = model.f.get_progress(pts_ft)
        Uf  = model.f(pts_ft, s_f).cpu().numpy()
    UF = Uf[:, 0].reshape(len(y_fine), len(x_fine))
    VF = Uf[:, 1].reshape(len(y_fine), len(x_fine))
    ax.streamplot(x_fine, y_fine, UF, VF, color='steelblue',
                  linewidth=0.7, density=1.2, arrowsize=0.8, zorder=3)

    # Reference (demo mean) path
    ref_path = model.f.ref_path.cpu().numpy()
    ax.plot(ref_path[:, 0], ref_path[:, 1], 'k-', lw=2.5,
            label='Demo path (attractor)', zorder=10)
    ax.scatter(X_START[0], X_START[1], c='blue', s=120, zorder=11, label='Start')
    ax.scatter(X_GOAL[0],  X_GOAL[1],  c='lime', s=120, marker='*', zorder=11, label='Goal')

    ax.set_xlim(SLAB_X); ax.set_ylim(SLAB_Y)
    title = ("Progress-Conditioned DS: Whole Trajectory as Attractor\n"
             "ẋ = v_net(x̃, s) + K·(x_ref(s) − x)")
    if title_suffix:
        title += f"\n{title_suffix}"
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.legend(fontsize=9, loc='upper left'); ax.grid(True, alpha=0.4)
    ax.set_aspect("equal")
    plt.tight_layout()
    out_name = f"vector_field_{tag}.png"
    plt.savefig(os.path.join(PLOT_DIR, out_name), dpi=150)
    plt.close()
    print(f"[Plot] Vector field saved: {out_name}")


def _plot_closed_loop_field(model, mean_np, std_np,
                              tag: str = "closed_loop_cbf",
                              title_suffix: str = ""):
    """
    Plot the SAFETY-FILTERED field ẋ = f_θ(x) + u_safe(x) via the shared
    field_plot renderer: streamlines DIVERT around each (inflated) critical
    zone — shown by the purple-dashed learned B=0 contour — then RE-CONVERGE
    onto the demo-path attractor. Same figure used by the generalization and
    dynamic-environment tests.
    """
    from field_plot import plot_diverting_field
    model.eval()
    model.set_obstacles(make_obstacle_tensors(k=64, seed=0))  # canonical scene
    fig, ax = plt.subplots(figsize=(10, 7))
    title = "Closed-loop field: ẋ = f_θ(x) + u_safe(x)  (divert → reconverge)"
    if title_suffix:
        title += f"\n{title_suffix}"
    plot_diverting_field(ax, model, CRITICAL_SHAPES, title=title)
    plt.tight_layout()
    out_name = f"vector_field_{tag}.png"
    plt.savefig(os.path.join(PLOT_DIR, out_name), dpi=150)
    plt.close()
    print(f"[Plot] Closed-loop (divert→reconverge) field saved: {out_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-from-lyapunov", action="store_true",
                         help="Skip Phase 0 and the Lyapunov-only outer "
                              "iters; load checkpoints/f_lyapunov_only.pt "
                              "and continue with barrier training only.")
    args = parser.parse_args()
    train(resume_from_lyapunov=args.resume_from_lyapunov)
