"""
dynamic_env_test.py — slow-changing-environment test for the learned CBF.

A red/critical obstacle slowly changes during the rollout while the SAME learned
CompositeBarrier (no retraining) is re-evaluated every control step against the
obstacle's CURRENT pose/size. Four motions are tested:

    translate     : obstacle drifts up across the corridor
    rotate        : obstacle spins in place near the path
    transrotate   : obstacle drifts up AND spins
    evolve        : obstacle slowly grows (enlarging) into the path

For each motion the needle must keep avoiding the obstacle WHEREVER it currently
is. We render frame snapshots: current obstacle pose (red), the learned B=0
contour (purple dashed), the divert-then-reconverge safety field, and the needle
path so far.

Usage:
    cd /home/stanny/franka_ros2_ws/src/Tolerance_Aware_CBF(TA-CBF)
    python src/dynamic_env_test.py                 # all 4 motions
    python src/dynamic_env_test.py --motions rotate --frames 3 --quick
"""

import os
import sys
import copy
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    DEVICE, X_START, X_GOAL, SLAB_X, SLAB_Y, Z_CORRIDOR,
    CRITICAL_SHAPES, CRITICAL_MARGIN, INFLATE_MARGIN,
    VEL_CLIP, GOAL_TOL, T_MAX, DT,
    transform_shape, _rot2d, canonical_interior_cloud, sdf_critical_shape_2d,
)
from simulate import load_model, BPCBFController
from cbf_qp import analytic_safety_filter
from field_plot import plot_diverting_field

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR  = os.path.join(ROOT_DIR, "results", "dynamic")
os.makedirs(OUT_DIR, exist_ok=True)

MOVING_IDX = 0   # the 'blocker' (star) is the slowly-moving obstacle

# Curated SOLVABLE dynamic layout: four STATIC obstacles form the walls of a wide
# mid corridor (top row ~y=0.15, bottom row ~y=-0.01), well clear of the demo
# path at y≈0.09. One obstacle (the star) then slowly intrudes into the corridor
# from below but NEVER seals it — a feasible passage above it always remains, so
# strict avoidance is achievable (unlike the old clustered scene, where the big
# standoff sealed the corridor and trapped the needle).
# Static obstacles in the four CORNERS (top row + far bottom corners), leaving the
# centre-below region open so the needle can detour BELOW the on-path mover. The
# learned standoff is large (~30mm), so the detour corridor must stay clear.
DYN_STATIC = [
    (5, (0.450, 0.150)),   # 'structure' (L)      top-left
    (3, (0.550, 0.150)),   # 'artery'    (kidney) top-right
    (2, (0.420, 0.005)),   # 'nerve'     (blob)   far bottom-left
    (4, (0.580, 0.005)),   # 'vein'      (blob)   far bottom-right
]
MOVER_BASE = np.array([0.500, 0.068], dtype=np.float32)   # star parked ON the path


def motion_params(motion: str, tau: float):
    """(d_rot, scale, d_trans-from-MOVER_BASE) for the slowly changing star, which
    sits ON the demo path so the needle must deviate BELOW it. Gentle amplitudes
    keep the below-corridor open and the needle outside the light-red zone."""
    if motion == "translate":      # slowly drift up across the path
        return 0.0, 0.7, np.array([0.0, -0.010 + 0.022 * tau], np.float32)
    if motion == "rotate":         # spin in place on the path
        return 1.3 * np.pi * tau, 0.7, np.array([0.0, 0.0], np.float32)
    if motion == "transrotate":    # drift up AND spin
        return 1.1 * np.pi * tau, 0.7, np.array([0.0, 0.010 * tau], np.float32)
    if motion == "evolve":         # slowly enlarge in place on the path
        return 0.0, 0.5 + 0.38 * tau, np.array([0.0, 0.0], np.float32)
    raise ValueError(f"unknown motion {motion}")


def build_scene(motion: str, tau: float, canon_clouds: list):
    """Transformed shapes (for SDF/plot) + obstacle tensors (cloud+center)."""
    shapes_t, obstacles = [], []

    # moving star
    d_rot, scale, d_trans = motion_params(motion, tau)
    mover = transform_shape(CRITICAL_SHAPES[MOVING_IDX], d_rot=d_rot, scale=scale)
    mover['center'] = np.array([MOVER_BASE[0] + d_trans[0],
                                MOVER_BASE[1] + d_trans[1], Z_CORRIDOR], np.float32)
    entries = [(mover, MOVING_IDX, d_rot, scale)]
    # static walls
    for sh_idx, (cx, cy) in DYN_STATIC:
        st = transform_shape(CRITICAL_SHAPES[sh_idx], d_rot=0.0, scale=1.0)
        st['center'] = np.array([cx, cy, Z_CORRIDOR], np.float32)
        entries.append((st, sh_idx, 0.0, 1.0))

    for sh_t, sh_idx, rot, scl in entries:
        shapes_t.append(sh_t)
        cloud_xy = (scl * canon_clouds[sh_idx]) @ _rot2d(rot).T
        obstacles.append({
            'cloud':  torch.tensor(cloud_xy.astype(np.float32), device=DEVICE),
            'center': torch.tensor(sh_t['center'].astype(np.float32), device=DEVICE),
        })
    return shapes_t, obstacles


def sdf_min(shapes_t, pt_xy):
    return min(sdf_critical_shape_2d(np.asarray(pt_xy, np.float32)[None], sh)[0]
               for sh in shapes_t)


# Predictive look-ahead: the controller plans against where the obstacle WILL be
# a fraction LOOKAHEAD of the motion ahead, so the needle diverts to the side the
# obstacle is LEAVING (opposite its heading) instead of being chased into a block.
LOOKAHEAD = 0.10


def run_dynamic(model, ctrl, motion: str, canon_clouds: list, max_seconds: float = 30.0):
    """
    Closed-loop rollout with the obstacle scene changing every step. Runs until
    the needle reaches the goal (up to max_seconds). The obstacle finishes its
    slow motion at τ=1 and then holds, so the needle can always complete.
    """
    x_np = X_START.copy().astype(np.float64)
    t, step = 0.0, 0
    max_step = int(max_seconds / DT)            # run UNTIL goal (generous cap)
    tau_dur = T_MAX * 0.9                        # obstacle finishes moving by here
    s_prev = 0.0
    x0_x, xg_x = float(X_START[0]), float(X_GOAL[0])
    T_DEMO = T_MAX * 0.7

    traj = [x_np.copy()]
    taus = [0.0]
    n_solid = n_buffer = 0
    reached = False

    while step < max_step:
        tau      = min(1.0, t / tau_dur)
        tau_pred = min(1.0, tau + LOOKAHEAD)    # where the obstacle is heading

        # actual scene (for collision accounting) and PREDICTED scene (for control)
        shapes_t,    _          = build_scene(motion, tau,      canon_clouds)
        shapes_pred, obs_pred   = build_scene(motion, tau_pred, canon_clouds)
        model.set_obstacles(obs_pred)           # controller plans against the future

        x_t = torch.tensor(x_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        s_x = float(np.clip((x_np[0] - x0_x) / (xg_x - x0_x + 1e-9), 0.0, 1.0))
        s_t = min(1.0, t / T_DEMO)
        s = max(s_prev, 0.5 * (s_x + s_t)); s_prev = s

        with torch.no_grad():
            f_val = model.f(x_t, s).cpu().numpy().flatten()
        dist_goal = float(np.linalg.norm(x_np - X_GOAL))
        if dist_goal < 0.025:
            a = 1.0 - dist_goal / 0.025
            f_val = (1 - a) * f_val + a * 0.04 * (X_GOAL - x_np) / (dist_goal + 1e-9)

        u_safe, _ = ctrl.solve(x_np.astype(np.float32), model, device=DEVICE, s=s)
        xdot = np.clip(f_val + u_safe, -VEL_CLIP, VEL_CLIP)
        xdot = ctrl.project_safe(x_np.astype(np.float32), model, xdot, DT, device=DEVICE)
        # HYBRID exact geometric backstop vs the obstacle's ACTUAL current pose
        xdot = analytic_safety_filter(x_np.astype(np.float32), xdot, DT, shapes_t)
        x_np = x_np + xdot * DT
        t += DT; step += 1
        traj.append(x_np.copy()); taus.append(tau)

        sdf_now = sdf_min(shapes_t, x_np[:2])   # safety against the ACTUAL pose
        if sdf_now < 0:
            n_solid += 1
        elif sdf_now < CRITICAL_MARGIN:
            n_buffer += 1

        if np.linalg.norm(x_np - X_GOAL) < GOAL_TOL:
            reached = True
            break

    return np.array(traj), np.array(taus), n_solid, n_buffer, reached


def render_motion(model, ctrl, motion: str, canon_clouds: list,
                  n_frames: int, quick: bool):
    print(f"\n[Dynamic] motion='{motion}'")
    traj, taus, n_solid, n_buffer, reached = run_dynamic(model, ctrl, motion, canon_clouds)
    safe = '✓ SAFE' if n_solid == 0 else f'✗ {n_solid} steps inside obstacle'
    goal = 'reached goal' if reached else 'DID NOT reach goal'
    print(f"  steps={len(traj)-1}  crit_solid={n_solid}  crit_buffer={n_buffer}  {safe}  ({goal})")

    frame_idx = np.linspace(0, len(traj) - 1, n_frames).astype(int)
    nx, ny = (18, 15) if quick else (26, 22)

    fig, axes = plt.subplots(1, n_frames, figsize=(6 * n_frames, 5.6))
    if n_frames == 1:
        axes = [axes]
    for ax, fi in zip(axes, frame_idx):
        tau = float(taus[fi])
        shapes_t, obstacles = build_scene(motion, tau, canon_clouds)
        model.set_obstacles(obstacles)
        plot_diverting_field(ax, model, shapes_t, ctrl=ctrl, nx=nx, ny=ny,
                             title=f"{motion}  τ={tau:.2f}  (step {fi})")
        ax.plot(traj[:fi + 1, 0], traj[:fi + 1, 1], color='#27ae60', lw=2.4,
                zorder=10, label='needle so far')
        ax.scatter(traj[fi, 0], traj[fi, 1], c='black', s=55, zorder=11)
        ax.legend(fontsize=7, loc='upper left')

    plt.suptitle(f"Slow-changing environment — '{motion}'   {safe}  |  {goal}", fontsize=13)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"dynamic_{motion}.png")
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"  saved → {out}")
    return n_solid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motions", nargs="+",
                    default=["translate", "rotate", "transrotate", "evolve"])
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--quick", action="store_true",
                    help="coarser field grid (faster, for smoke tests)")
    args = ap.parse_args()

    model = load_model()
    ctrl  = BPCBFController()
    canon_clouds = [canonical_interior_cloud(sh, k=128, seed=300 + i)
                    for i, sh in enumerate(CRITICAL_SHAPES)]

    results = {}
    for m in args.motions:
        results[m] = render_motion(model, ctrl, m, canon_clouds,
                                    args.frames, args.quick)

    print("\n[Dynamic] Summary:")
    for m, ns in results.items():
        print(f"  {m:12s}: {'SAFE' if ns == 0 else f'{ns} intrusions'}")


if __name__ == "__main__":
    main()
