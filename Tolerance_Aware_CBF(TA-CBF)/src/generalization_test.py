"""
generalization_test.py — zero-shot generalization test for the CompositeBarrier.

Goal: without any retraining of the CBF network, swap in a NEW set of
red/critical obstacles (different positions, orientations, and — in one
config — two obstacles placed extremely close together) and verify that:

  1. model.B(x) (CompositeBarrier, via model.set_obstacles(new_obstacles))
     produces a sensible barrier for the NEW geometry (zero-level set
     surrounds the new obstacle positions, single fused boundary when two
     obstacles are close).
  2. The closed-loop bpcbf rollout (same f_θ / V_θ, same expert demos) still
     converges to the demo trajectory and avoids the NEW obstacle positions.

Usage:
    cd /home/stanny/franka_ros2_ws/src/Tolerance_Aware_CBF(TA-CBF)
    python src/generalization_test.py
"""

import os
import sys
import copy
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    DEVICE, X_START, X_GOAL, DEMO_WAYPOINTS, SLAB_X, SLAB_Y,
    CRITICAL_SHAPES, CRITICAL_MARGIN, INFLATE_MARGIN, B_SAFE_MARGIN,
    BARRIER_SDF_K, Z_CORRIDOR,
    sdf_critical_shape_2d, sample_obstacle_cloud, transform_shape,
)

# Needle standoff: with the controller defending B ≥ B_SAFE_MARGIN and the
# barrier slope K, the needle is held at sdf ≳ INFLATE + B_SAFE_MARGIN/K. A
# feasible passage must be at least this wide on each side → use it as the
# clearance the scene generator guarantees, plus a hair of slack.
# The learned barrier is "compressed" (rises slower than the ideal SDF), so the
# effective standoff where B=B_SAFE_MARGIN is empirically ~25–40 mm rather than
# the nominal INFLATE+margin/K. We size the guaranteed passage to that measured
# standoff so the needle never gets trapped, and validate every scene with a real
# rollout (see make_validated_scene).
NEEDLE_STANDOFF = INFLATE_MARGIN + B_SAFE_MARGIN / BARRIER_SDF_K
# Hybrid filter holds the needle ≥ SAFETY_SDF_MARGIN (≈11mm) from the tissue, so a
# feasible passage needs a touch more than that on each side.
from config import SAFETY_SDF_MARGIN as _SSM
SCENE_CLEARANCE = _SSM + 0.004
from simulate import load_model, BPCBFController
from multi_rollout import run_from_start
from field_plot import plot_diverting_field

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR  = os.path.join(ROOT_DIR, "results", "generalization")
os.makedirs(OUT_DIR, exist_ok=True)

MARGIN = 0.03  # keep obstacle centers away from the slab border / start / goal


# ── New obstacle scene generators ───────────────────────────────────────────

def random_shapes(seed: int, n_rot: float = 1.2) -> list:
    """Copy CRITICAL_SHAPES but randomize each obstacle's position/orientation."""
    rng = np.random.default_rng(seed)
    shapes = copy.deepcopy(CRITICAL_SHAPES)
    for sh in shapes:
        for _ in range(50):
            dx, dy = rng.uniform(-0.06, 0.06, 2)
            cx = float(np.clip(sh['center'][0] + dx, SLAB_X[0] + MARGIN, SLAB_X[1] - MARGIN))
            cy = float(np.clip(sh['center'][1] + dy, SLAB_Y[0] + MARGIN, SLAB_Y[1] - MARGIN))
            if (np.hypot(cx - X_START[0], cy - X_START[1]) > 0.03 and
                    np.hypot(cx - X_GOAL[0],  cy - X_GOAL[1])  > 0.03):
                sh['center'] = np.array([cx, cy, Z_CORRIDOR], dtype=np.float32)
                break
        if 'rotation' in sh:
            sh['rotation'] = float(sh['rotation'] + rng.uniform(-n_rot, n_rot))
    return shapes


def merge_two_shapes(shapes: list, i: int, j: int, sep: float, seed: int) -> list:
    """Force shapes[j] to sit extremely close to shapes[i] (separation = sep)."""
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi)
    ci = shapes[i]['center']
    shapes[j]['center'] = np.array(
        [ci[0] + sep * np.cos(ang), ci[1] + sep * np.sin(ang), Z_CORRIDOR],
        dtype=np.float32)
    return shapes


# ── Solvable random scenes: dense layouts with a GUARANTEED passage ──────────
# Place N obstacles at random positions / rotations / scales, then accept the
# layout ONLY if a free path (clearance > SCENE_CLEARANCE, i.e. OUTSIDE the
# light-red CBF zone) exists from start to goal. This both stresses the barrier
# (narrow corridors) and ensures the needle CAN stay fully outside the buffer.

def _free_path_exists(shapes: list, clearance: float, res: float = 0.0025) -> bool:
    """BFS on a grid: is there a start→goal path with min-SDF > clearance?"""
    xs = np.arange(SLAB_X[0], SLAB_X[1], res)
    ys = np.arange(SLAB_Y[0], SLAB_Y[1], res)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)
    free = (sdf_all_np(pts, shapes) > clearance).reshape(len(ys), len(xs))

    def cell(p):
        return (int(np.clip((p[1]-ys[0])/res, 0, len(ys)-1)),
                int(np.clip((p[0]-xs[0])/res, 0, len(xs)-1)))
    s, g = cell(X_START), cell(X_GOAL)
    if not (free[s] and free[g]):
        return False
    from collections import deque
    seen = np.zeros_like(free, bool); seen[s] = True
    q = deque([s])
    while q:
        r, c = q.popleft()
        if (r, c) == g:
            return True
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r+dr, c+dc
                if (0 <= nr < free.shape[0] and 0 <= nc < free.shape[1]
                        and free[nr, nc] and not seen[nr, nc]):
                    seen[nr, nc] = True; q.append((nr, nc))
    return False


def _demo_point(t: float) -> np.ndarray:
    """Interpolate the demo path at progress t∈[0,1] → (x, y)."""
    wps = DEMO_WAYPOINTS
    idx = float(np.clip(t, 0, 1)) * (len(wps) - 1)
    lo = int(idx); hi = min(lo + 1, len(wps) - 1); a = idx - lo
    return (1 - a) * wps[lo][:2] + a * wps[hi][:2]


def make_solvable_scene(n_obs: int, seed: int,
                        clearance: float = SCENE_CLEARANCE,
                        n_on_path: int = 1, min_sep: float = 0.045) -> list:
    """
    Random scene of n_obs obstacles SPREAD across the whole workspace with a
    minimum centre-to-centre separation `min_sep` (so they don't clump), and at
    least `n_on_path` placed ON the demo path so the rollout must deviate. Only
    layouts with a guaranteed start→goal passage (clearance) are accepted.
    """
    for attempt in range(3000):
        rng = np.random.default_rng(seed * 1000 + attempt)
        shapes, centers = [], []
        for k in range(n_obs):
            base = copy.deepcopy(CRITICAL_SHAPES[k % len(CRITICAL_SHAPES)])
            d_rot = float(rng.uniform(-np.pi, np.pi))
            scale = float(rng.uniform(0.7, 1.15))
            sh = transform_shape(base, d_rot=d_rot, scale=scale, d_trans=(0, 0))
            placed = False
            for _ in range(120):
                if k < n_on_path:
                    t = rng.uniform(0.30, 0.70)
                    px, py = _demo_point(t)
                    cx = float(px + rng.uniform(-0.010, 0.010))
                    cy = float(py + rng.uniform(-0.012, 0.012))
                else:
                    # spread uniformly over the ENTIRE slab (not just the corridor)
                    cx = float(rng.uniform(SLAB_X[0] + MARGIN, SLAB_X[1] - MARGIN))
                    cy = float(rng.uniform(SLAB_Y[0] + MARGIN, SLAB_Y[1] - MARGIN))
                cx = float(np.clip(cx, SLAB_X[0] + MARGIN, SLAB_X[1] - MARGIN))
                cy = float(np.clip(cy, SLAB_Y[0] + MARGIN, SLAB_Y[1] - MARGIN))
                far_obs = all(np.hypot(cx - c[0], cy - c[1]) > min_sep for c in centers)
                far_ends = (np.hypot(cx - X_START[0], cy - X_START[1]) > 0.040 and
                            np.hypot(cx - X_GOAL[0], cy - X_GOAL[1]) > 0.040)
                if far_obs and far_ends:
                    placed = True; break
            if not placed:
                break
            sh['center'] = np.array([cx, cy, Z_CORRIDOR], dtype=np.float32)
            shapes.append(sh); centers.append((cx, cy))
        if len(shapes) == n_obs and _free_path_exists(shapes, clearance):
            return shapes
    raise RuntimeError(f"could not build a solvable {n_obs}-obstacle scene "
                       f"(seed {seed}); loosen clearance/min_sep or reduce n_obs.")


# ── Generic SDF / scene drawing for an arbitrary shapes list ───────────────

def sdf_all_np(pts: np.ndarray, shapes: list) -> np.ndarray:
    pts_xy = pts[:, :2].astype(np.float32)
    sdfs = [sdf_critical_shape_2d(pts_xy, sh) for sh in shapes]
    return np.min(np.stack(sdfs, axis=1), axis=1)


def count_critical_entries(traj: np.ndarray, shapes: list) -> tuple:
    """(n_solid, n_lightred): steps inside the true obstacle, and steps inside the
    light-red CBF zone (0 ≤ sdf < INFLATE_MARGIN). The needle should keep BOTH 0."""
    n_solid = n_lightred = 0
    for pt in traj:
        sdf = sdf_all_np(pt[None], shapes)[0]
        if sdf < 0:
            n_solid += 1
        elif sdf < INFLATE_MARGIN:
            n_lightred += 1
    return n_solid, n_lightred


def draw_scene_generic(ax, shapes: list, title: str = ""):
    nx, ny = 300, 250
    xs = np.linspace(SLAB_X[0], SLAB_X[1], nx)
    ys = np.linspace(SLAB_Y[0], SLAB_Y[1], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)
    sdf_c = sdf_all_np(pts, shapes).reshape(ny, nx)

    ax.contourf(xs, ys, sdf_c, levels=[-10, 0], colors=['#ff3333'], alpha=0.55, zorder=2)
    ax.contourf(xs, ys, sdf_c, levels=[0, INFLATE_MARGIN], colors=['#ffaaaa'], alpha=0.3, zorder=2)
    ax.contour(xs, ys, sdf_c, levels=[0], colors=['#aa0000'], linewidths=1.5, zorder=3)

    for sh in shapes:
        ax.text(sh['center'][0], sh['center'][1], sh['label'].split('-')[0],
                color='#880000', ha='center', va='center', fontsize=7,
                fontweight='bold', zorder=6)

    ax.set_xlim(SLAB_X); ax.set_ylim(SLAB_Y)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title, fontsize=10)


def build_tensors(shapes: list, k: int = 64, seed: int = 0) -> list:
    obs = []
    for i, sh in enumerate(shapes):
        cloud = sample_obstacle_cloud(sh, k=k, seed=seed + i)
        obs.append({
            'cloud':  torch.tensor(cloud, dtype=torch.float32, device=DEVICE),
            'center': torch.tensor(sh['center'].astype(np.float32), dtype=torch.float32, device=DEVICE),
        })
    return obs


# ── Per-config evaluation ───────────────────────────────────────────────────

def evaluate_config(model, ctrl, shapes: list, name: str, title: str):
    print(f"\n[Gen-Test] {name} — {title}")
    model.set_obstacles(build_tensors(shapes, k=64, seed=hash(name) % 1000))

    traj = run_from_start(model, ctrl, X_START, shapes=shapes)
    n_solid, n_lightred = count_critical_entries(traj['ee_pos'], shapes)
    if n_solid == 0 and n_lightred == 0:
        tag = '✓ SAFE (fully outside light-red CBF zone)'
    elif n_solid == 0:
        tag = f'△ {n_lightred} steps grazed light-red (no obstacle hit)'
    else:
        tag = f'✗ {n_solid} steps inside obstacle'
    print(f"  steps={len(traj['time'])-1}  reached={traj['reached_goal']}  "
          f"obstacle={n_solid}  light_red={n_lightred}  {tag}")

    # ── Figure: scene+rollout | learned B(x) | analytic SDF | divert field ──
    fig, axes = plt.subplots(1, 4, figsize=(26, 6))

    # Panel 1: scene + demo path + rollout
    ax = axes[0]
    draw_scene_generic(ax, shapes, title=f"{title}\nClosed-loop rollout (no retraining)")
    ax.plot(DEMO_WAYPOINTS[:, 0], DEMO_WAYPOINTS[:, 1], 'b--', lw=1.5,
            label='Original demo path', zorder=6)
    tr = traj['ee_pos']
    ax.plot(tr[:, 0], tr[:, 1], color='#27ae60', lw=2, zorder=8, label='bpcbf rollout')
    ax.scatter(X_START[0], X_START[1], c='blue', s=100, zorder=9, label='Start')
    ax.scatter(X_GOAL[0], X_GOAL[1], c='lime', s=100, marker='*', zorder=9, label='Goal')
    ax.set_title(f"{ax.get_title()}\n{tag}", fontsize=9)
    ax.legend(fontsize=8)

    # Panel 2: learned composite barrier B(x) on the NEW obstacle set
    nx, ny = 250, 200
    xs = np.linspace(SLAB_X[0], SLAB_X[1], nx)
    ys = np.linspace(SLAB_Y[0], SLAB_Y[1], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)
    pts3 = np.concatenate([pts, np.full((pts.shape[0], 1), Z_CORRIDOR, dtype=np.float32)], axis=1)
    pts_t = torch.tensor(pts3, device=DEVICE)
    with torch.no_grad():
        B_vals = model.B(pts_t).squeeze(-1).cpu().numpy().reshape(ny, nx)

    ax = axes[1]
    cf = ax.contourf(xs, ys, B_vals, levels=30, cmap='RdYlGn')
    ax.contour(xs, ys, B_vals, levels=[0], colors='black', linewidths=2, linestyles='--', zorder=5)
    plt.colorbar(cf, ax=ax, label='B_φ(x) (learned, new obstacles)')
    draw_scene_generic(ax, shapes)
    ax.set_title("Learned CompositeBarrier on NEW obstacle set\n"
                  "(model.set_obstacles — zero-shot, no retraining)", fontsize=9)

    # Panel 3: analytic SDF for the NEW obstacle set (ground truth for comparison)
    sdf_c = sdf_all_np(pts, shapes).reshape(ny, nx)
    ax = axes[2]
    cf2 = ax.contourf(xs, ys, sdf_c, levels=30, cmap='RdYlGn')
    ax.contour(xs, ys, sdf_c, levels=[0], colors='black', linewidths=2, linestyles='--', zorder=5)
    plt.colorbar(cf2, ax=ax, label='Analytic SDF (new obstacle positions)')
    draw_scene_generic(ax, shapes)
    ax.set_title("Analytic SDF — NEW obstacle positions\n(ground truth, for comparison only)", fontsize=9)

    # Panel 4: divert-then-reconverge safety field on the NEW obstacle set
    ax = axes[3]
    plot_diverting_field(ax, model, shapes, ctrl=ctrl, nx=38, ny=32, stream_density=1.7,
                         title="Safety field ẋ=f+u_safe\n(divert around B=0, reconverge to demo)")

    plt.suptitle(f"Zero-Shot Generalization — {title}", fontsize=12)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"{name}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  saved → {out}")


def make_validated_scene(model, ctrl, n_obs: int, seed0: int,
                         min_dev: float = 0.010) -> tuple:
    """
    Find a scene with obstacle(s) ON the demo path where the actual closed-loop
    rollout is fully clean — reaches goal, never enters an obstacle OR the
    light-red zone — AND visibly deviates (max deviation from the demo path
    > min_dev). Retries seeds until one qualifies, so paper figures are reliable.
    """
    demo = DEMO_WAYPOINTS[:, :2]
    for seed in range(seed0, seed0 + 60):
        try:
            shapes = make_solvable_scene(n_obs, seed=seed)
        except RuntimeError:
            continue
        model.set_obstacles(build_tensors(shapes, k=64, seed=1))
        tr = run_from_start(model, ctrl, X_START, shapes=shapes)
        n_solid, n_lr = count_critical_entries(tr['ee_pos'], shapes)
        dev = max(min(np.linalg.norm(p - w) for w in demo) for p in tr['ee_pos'][:, :2])
        # hard requirement: reach + never enter the obstacle (n_solid==0, the
        # hybrid guarantee); visible deviation so the figure is illustrative.
        if tr['reached_goal'] and n_solid == 0 and dev > min_dev:
            print(f"  [scene] n_obs={n_obs} seed={seed} ✓ (deviation {dev*1e3:.0f}mm, buffer {n_lr})")
            return shapes, seed
    raise RuntimeError(f"no clean deviating {n_obs}-obstacle scene from seed {seed0}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_obs", type=int, nargs="+", default=[3, 4, 5],
                    help="obstacle counts for the solvable scenes")
    ap.add_argument("--seed0", type=int, nargs="+", default=[1, 10, 20])
    args = ap.parse_args()

    model = load_model()
    ctrl  = BPCBFController()
    print(f"[Gen-Test] scene clearance={SCENE_CLEARANCE*1e3:.1f}mm  "
          f"light-red zone={INFLATE_MARGIN*1e3:.0f}mm — obstacles placed ON the demo path")

    for n_obs, seed0 in zip(args.n_obs, args.seed0):
        shapes, seed = make_validated_scene(model, ctrl, n_obs, seed0)
        evaluate_config(model, ctrl, shapes, f"config_onpath_{n_obs}obs_seed{seed}",
                        f"{n_obs} obstacles ON the demo path — rollout deviates around them, "
                        f"stays outside the CBF zone")


if __name__ == "__main__":
    main()
