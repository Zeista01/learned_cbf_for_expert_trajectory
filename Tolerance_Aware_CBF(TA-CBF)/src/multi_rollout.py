"""
multi_rollout.py — run several closed-loop bpcbf rollouts from perturbed
starting positions (around X_START) and plot each one, plus an overlay,
into results/after_training/.

Usage:
    cd /home/stanny/franka_ros2_ws/src/Tolerance_Aware_CBF(TA-CBF)
    python src/multi_rollout.py --n 6 --radius 0.015
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from config import X_START, X_GOAL, VEL_CLIP, GOAL_TOL, T_MAX, DT
from simulate import (load_model, _draw_scene, _count_critical_entries,
                       BPCBFController)
import torch
from config import STATE_DIM, DEVICE

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR  = os.path.join(ROOT_DIR, "results", "after_training")
os.makedirs(OUT_DIR, exist_ok=True)


def run_from_start(model, ctrl, x_start: np.ndarray, shapes=None) -> dict:
    """Run a single bpcbf rollout from a custom starting point.

    If `shapes` (the analytic obstacle list) is given, the HYBRID exact safety
    filter is applied as the final step → the needle is guaranteed to never enter
    the critical tissue, regardless of learned-barrier error."""
    from cbf_qp import analytic_safety_filter
    x_np = x_start.copy().astype(np.float64)
    t = 0.0
    step = 0
    max_step = int(T_MAX / DT)
    s_prev = 0.0
    x0_x, xg_x = float(X_START[0]), float(X_GOAL[0])
    T_DEMO = T_MAX * 0.7

    traj = {'ee_pos': [x_np.copy()], 'time': [0.0], 'reached_goal': False}

    while step < max_step:
        x_t = torch.tensor(x_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        s_x = float(np.clip((x_np[0] - x0_x) / (xg_x - x0_x + 1e-9), 0.0, 1.0))
        s_t = min(1.0, t / T_DEMO)
        s = max(s_prev, 0.5 * (s_x + s_t))
        s_prev = s

        with torch.no_grad():
            f_val = model.f(x_t, s).cpu().numpy().flatten()

        dist_goal = float(np.linalg.norm(x_np - X_GOAL))
        if dist_goal < 0.025:
            alpha_g = 1.0 - dist_goal / 0.025
            goal_pull = 0.04 * (X_GOAL - x_np) / (dist_goal + 1e-9)
            f_val = (1 - alpha_g) * f_val + alpha_g * goal_pull

        u_safe, _ = ctrl.solve(x_np.astype(np.float32), model, device=DEVICE, s=s)

        xdot = np.clip(f_val + u_safe, -VEL_CLIP, VEL_CLIP)
        # discrete-time learned-CBF safety projection (drives the divert field)
        xdot = ctrl.project_safe(x_np.astype(np.float32), model, xdot, DT, device=DEVICE)
        # HYBRID exact geometric backstop (guarantees obstacle=0)
        if shapes is not None:
            xdot = analytic_safety_filter(x_np.astype(np.float32), xdot, DT, shapes)
        x_np = x_np + xdot * DT
        t += DT
        step += 1

        traj['ee_pos'].append(x_np.copy())
        traj['time'].append(t)

        if np.linalg.norm(x_np - X_GOAL) < GOAL_TOL:
            traj['reached_goal'] = True
            break

    traj['ee_pos'] = np.array(traj['ee_pos'])
    traj['time']   = np.array(traj['time'])
    return traj


def main(n_rollouts: int, radius: float, seed: int):
    model = load_model()
    ctrl  = BPCBFController()
    rng   = np.random.default_rng(seed)

    # Perturbed starting points around X_START (XY only, Z fixed)
    starts = [X_START.copy()]
    for _ in range(n_rollouts - 1):
        offset = rng.uniform(-radius, radius, size=2)
        sp = X_START.copy()
        sp[0] += offset[0]
        sp[1] += offset[1]
        starts.append(sp)

    rollouts = []
    for i, sp in enumerate(starts):
        traj = run_from_start(model, ctrl, sp)
        n_solid, n_buffer = _count_critical_entries(traj['ee_pos'])
        tag = '✓ SAFE' if n_solid == 0 else f'✗ {n_solid} critical entries'
        print(f"[Rollout {i}] start=({sp[0]:.4f},{sp[1]:.4f})  "
              f"steps={len(traj['time'])-1}  reached={traj['reached_goal']}  {tag}")
        rollouts.append((sp, traj, n_solid))

    # ── Individual plots ────────────────────────────────────────────────────
    for i, (sp, traj, n_solid) in enumerate(rollouts):
        fig, ax = plt.subplots(figsize=(7, 6))
        _draw_scene(ax)
        tr = traj['ee_pos']
        ax.plot(tr[:, 0], tr[:, 1], color='#27ae60', lw=2, zorder=7)
        ax.scatter(sp[0], sp[1], c='blue', s=100, zorder=9, label='Start')
        ax.scatter(X_GOAL[0], X_GOAL[1], c='lime', s=100, marker='*', zorder=9, label='Goal')
        tag = '✓ SAFE' if n_solid == 0 else f'✗ {n_solid} critical entries'
        ax.set_title(f"Rollout {i} — bpcbf (Learned Barrier + Demo-Tracking DS)\n"
                      f"start=({sp[0]:.4f}, {sp[1]:.4f})  reached={traj['reached_goal']}  {tag}",
                      fontsize=10)
        ax.legend()
        plt.tight_layout()
        out = os.path.join(OUT_DIR, f"rollout_{i}.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  saved → {out}")

    # ── Overlay plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    _draw_scene(ax, title="Multiple bpcbf Rollouts — Perturbed Starting Points")
    cmap = plt.cm.viridis(np.linspace(0, 1, len(rollouts)))
    for i, (sp, traj, n_solid) in enumerate(rollouts):
        tr = traj['ee_pos']
        ax.plot(tr[:, 0], tr[:, 1], color=cmap[i], lw=2, zorder=7,
                label=f"rollout {i}" + ("" if n_solid == 0 else f" ({n_solid} crit)"))
        ax.scatter(sp[0], sp[1], color=cmap[i], s=60, zorder=9, edgecolors='k')
    ax.scatter(X_GOAL[0], X_GOAL[1], c='lime', s=150, marker='*', zorder=10, label='Goal')
    ax.legend(fontsize=8, loc='upper left')
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "rollouts_overlay.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=6, help='number of rollouts')
    parser.add_argument('--radius', type=float, default=0.015,
                         help='perturbation radius (m) around X_START for rollouts 1..n-1')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    main(args.n, args.radius, args.seed)
