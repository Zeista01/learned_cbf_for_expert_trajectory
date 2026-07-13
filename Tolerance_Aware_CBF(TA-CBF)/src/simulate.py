"""
simulate.py — 2D simulation and visualisation for the TA-CBF (Learned Barrier) model.

Three comparison modes:
  nominal  : ẋ = f_θ(x, s)           (no safety filter)
  hard_cbf : ẋ = f_θ + u,  standard CBF B_φ ≥ 0 (strict, no penetration)
  bpcbf    : ẋ = f_θ + u,  learned barrier B_φ QP (proposed)

Key plots generated:
  1. vector_field_trajectory_attractor.png
     The vector field ẋ = f_θ(x, s) showing the WHOLE demo path as an attractor.
     Streamlines converge to the demo path from all surrounding directions —
     not just to the goal point.  This is the single-attractor vs trajectory-
     attractor distinction the mentor requested.

  2. simulation_comparison.png
     Side-by-side: nominal / hard_cbf / proposed bpcbf trajectories on the
     2D scene with non-linear critical obstacles.

  3. learned_barrier_field.png
     Contour plot of B_φ(x) on the scene.  The zero-level set should surround
     the non-linear critical zones.

Usage:
    cd /home/stanny/franka_ros2_ws/src/Tolerance_Aware_CBF(TA-CBF)
    python src/simulate.py --mode all
    python src/simulate.py --mode bpcbf
    python src/simulate.py --plot_field   # just plot the vector field
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

sys.path.insert(0, os.path.dirname(__file__))
from models import BPCBFModel
from cbf_qp import BPCBFController
from analytical_sdf import sdf_all_np, sdf_critical_np
from train import make_obstacle_tensors
from config import (
    STATE_DIM, DEVICE, X_START, X_GOAL, DEMO_WAYPOINTS,
    SLAB_X, SLAB_Y, CRITICAL_SHAPES, FOAM_CENTRES, FOAM_RADIUS,
    sdf_all_critical_np, is_in_critical_np,
    CRITICAL_MARGIN,
    VEL_CLIP, GOAL_TOL, T_MAX, DT,
)

ROOT_DIR   = os.path.join(os.path.dirname(__file__), "..")
CKPT_DIR   = os.path.join(ROOT_DIR, "checkpoints")
RESULT_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(path: str = None) -> BPCBFModel:
    if path is None:
        # Prefer final_model.pt: it is the END-of-training state with the fully
        # trained + counterexample-augmented barrier. best_model.pt is selected
        # by lowest TOTAL loss, which (because the barrier weight ramps 0→1) can
        # land in the barrier-INACTIVE phase and freeze an untrained barrier.
        path = os.path.join(CKPT_DIR, "final_model.pt")
        if not os.path.exists(path):
            path = os.path.join(CKPT_DIR, "best_model.pt")
    m = BPCBFModel()
    m.load(path, device=DEVICE)
    # Restore normalization
    mean_path = os.path.join(CKPT_DIR, "norm_mean.npy")
    std_path  = os.path.join(CKPT_DIR, "norm_std.npy")
    ref_path  = os.path.join(CKPT_DIR, "ref_path.npy")
    if os.path.exists(mean_path) and os.path.exists(std_path):
        mean = torch.tensor(np.load(mean_path), dtype=torch.float32)
        std  = torch.tensor(np.load(std_path),  dtype=torch.float32)
        m.set_norm(mean.to(DEVICE), std.to(DEVICE))
    if os.path.exists(ref_path):
        ref = np.load(ref_path)
        m.f.set_reference(ref)
    m.set_obstacles(make_obstacle_tensors(k=64, seed=0))  # composite barrier needs the obstacle set
    from config import DEMO_K
    m.f.K = DEMO_K   # strong attraction back to the expert demo path
    m.eval()
    return m


# ── Simulation loop ───────────────────────────────────────────────────────────

def _count_critical_entries(traj: np.ndarray) -> tuple:
    n_solid = 0; n_buffer = 0
    for pt in traj:
        sdf = sdf_all_critical_np(pt[None])[0]
        if sdf < 0:
            n_solid += 1
        elif sdf < CRITICAL_MARGIN:
            n_buffer += 1
    return n_solid, n_buffer


def run_simulation(mode: str = 'bpcbf',
                   model: BPCBFModel = None,
                   save: bool = True) -> dict:
    print(f"\n[Sim] Mode: {mode}")
    if model is None:
        model = load_model()

    ctrl   = BPCBFController()
    x_np   = X_START.copy().astype(np.float64)
    t      = 0.0
    step   = 0
    max_step = int(T_MAX / DT)
    s_prev   = 0.0

    traj = {
        'ee_pos':      [x_np.copy()],
        'cbf_val':     [],
        'clf_val':     [],
        'time':        [0.0],
        'reached_goal': False,
        'mode':        mode,
    }

    x0_x, xg_x = float(X_START[0]), float(X_GOAL[0])
    T_DEMO = T_MAX * 0.7

    while step < max_step:
        x_t = torch.tensor(x_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        # Progress: blend X-position progress + time progress
        s_x = float(np.clip((x_np[0] - x0_x) / (xg_x - x0_x + 1e-9), 0.0, 1.0))
        s_t = min(1.0, t / T_DEMO)
        s   = max(s_prev, 0.5 * (s_x + s_t))
        s_prev = s

        if mode == 'nominal':
            with torch.no_grad():
                f_val  = model.f(x_t, s).cpu().numpy().flatten()
            u_safe = np.zeros(STATE_DIM)
            info   = {'cbf_val': float(sdf_all_critical_np(x_np[None])[0]),
                      'clf_val': 0.0}

        elif mode == 'shortest_path':
            # Naive baseline: straight-line shortest path to the goal,
            # no learned dynamics, no safety filter.
            goal_dir = X_GOAL - x_np
            dist_g   = np.linalg.norm(goal_dir) + 1e-9
            f_val    = 0.04 * goal_dir / dist_g
            u_safe   = np.zeros(STATE_DIM)
            info     = {'cbf_val': float(sdf_all_critical_np(x_np[None])[0]),
                        'clf_val': 0.0}

        else:  # bpcbf
            with torch.no_grad():
                f_val = model.f(x_t, s).cpu().numpy().flatten()
            # Near-goal: blend in a gentle goal pull
            dist_goal = float(np.linalg.norm(x_np - X_GOAL))
            if dist_goal < 0.025:
                alpha_g = 1.0 - dist_goal / 0.025
                goal_pull = 0.04 * (X_GOAL - x_np) / (dist_goal + 1e-9)
                f_val = (1 - alpha_g) * f_val + alpha_g * goal_pull
            u_safe, info = ctrl.solve(x_np.astype(np.float32), model,
                                      device=DEVICE, s=s)

        xdot = np.clip(f_val + u_safe, -VEL_CLIP, VEL_CLIP)
        x_np = x_np + xdot * DT
        t   += DT
        step += 1

        traj['ee_pos'].append(x_np.copy())
        traj['cbf_val'].append(info.get('cbf_val', 0.0))
        traj['clf_val'].append(info.get('clf_val', 0.0))
        traj['time'].append(t)

        if np.linalg.norm(x_np - X_GOAL) < GOAL_TOL:
            print(f"  [Sim] Goal reached at t={t:.2f}s, step={step}")
            traj['reached_goal'] = True
            break

    for k in ['ee_pos', 'cbf_val', 'clf_val', 'time']:
        traj[k] = np.array(traj[k])

    n_solid, n_buffer = _count_critical_entries(traj['ee_pos'])
    safety_tag = ('✓ SAFE' if n_solid == 0
                  else f'✗ CRITICAL ENTRIES: {n_solid}')
    print(f"  steps={step}  reached={traj['reached_goal']}  "
          f"crit_solid={n_solid}  crit_buffer={n_buffer}  {safety_tag}")

    if save:
        np.save(os.path.join(RESULT_DIR, f"traj_{mode}.npy"), traj, allow_pickle=True)
        print(f"  Saved → {RESULT_DIR}/traj_{mode}.npy")

    return traj


def run_all(save: bool = True) -> dict:
    model   = load_model()
    results = {}
    for mode in ('nominal', 'shortest_path', 'bpcbf'):
        results[mode] = run_simulation(mode=mode, model=model, save=save)
    _plot_comparison(results)
    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def _draw_scene(ax, title: str = "", show_foam: bool = True):
    """Draw the 2D surgical scene on an axes."""
    from analytical_sdf import sdf_critical_np as sdf_crit
    nx, ny = 300, 250
    xs = np.linspace(SLAB_X[0], SLAB_X[1], nx)
    ys = np.linspace(SLAB_Y[0], SLAB_Y[1], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)

    # Critical zones (non-linear shapes — red fill)
    sdf_c = sdf_all_critical_np(pts).reshape(ny, nx)
    ax.contourf(xs, ys, sdf_c, levels=[-10, 0], colors=['#ff3333'], alpha=0.55, zorder=2)
    ax.contourf(xs, ys, sdf_c, levels=[0, CRITICAL_MARGIN],
                colors=['#ffaaaa'], alpha=0.3, zorder=2)
    ax.contour(xs, ys, sdf_c, levels=[0], colors=['#aa0000'], linewidths=1.5, zorder=3)

    # Add shape labels
    for shape in CRITICAL_SHAPES:
        ax.text(shape['center'][0], shape['center'][1],
                shape['label'].split('-')[0], color='#880000',
                ha='center', va='center', fontsize=7, fontweight='bold', zorder=6)

    # Foam (orange circles — deformable, can penetrate)
    if show_foam:
        for fc in FOAM_CENTRES[::5]:  # subsample for speed
            circle = plt.Circle(fc, FOAM_RADIUS, color='#ff8800', alpha=0.15, zorder=1)
            ax.add_patch(circle)

    ax.set_xlim(SLAB_X); ax.set_ylim(SLAB_Y)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title, fontsize=10)


def _plot_comparison(results: dict):
    """Side-by-side trajectory comparison plot."""
    modes  = list(results.keys())
    colors = {'nominal': '#e74c3c', 'shortest_path': '#e67e22', 'bpcbf': '#27ae60'}
    labels = {'nominal': 'Nominal DS (no safety)',
              'shortest_path': 'Naive Shortest-Path to Goal (no learning, no safety)',
              'bpcbf': 'Proposed: Learned Barrier + Demo-Tracking DS'}

    fig, axes = plt.subplots(1, len(modes), figsize=(6 * len(modes), 6))
    if len(modes) == 1:
        axes = [axes]

    for ax, mode in zip(axes, modes):
        _draw_scene(ax, title=labels.get(mode, mode))
        traj = results[mode]['ee_pos']
        ax.plot(traj[:, 0], traj[:, 1], color=colors.get(mode, 'blue'),
                lw=2, zorder=7, label=mode)
        ax.scatter(X_START[0], X_START[1], c='blue', s=100, zorder=9, label='Start')
        ax.scatter(X_GOAL[0], X_GOAL[1], c='lime', s=100, marker='*', zorder=9, label='Goal')
        n_solid, _ = _count_critical_entries(traj)
        tag = f"✓ SAFE" if n_solid == 0 else f"✗ {n_solid} critical entries"
        ax.set_title(f"{labels.get(mode, mode)}\n{tag}", fontsize=9)

    plt.suptitle("Trajectory Comparison — 2D Surgical Needle Navigation\n"
                 "Non-linear critical obstacles (star, crescent, blob, kidney, L-shape)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "simulation_comparison.png"), dpi=150)
    plt.close()
    print(f"[Plot] Comparison saved.")


def plot_vector_field(model: BPCBFModel = None):
    """
    Plot the vector field f_θ(x, s) showing the FULL TRAJECTORY as an attractor.

    This is what mentor requested: instead of a single-goal attractor (where
    all streamlines go to one point), the progress-conditioned DS makes the
    ENTIRE demo path an attractor manifold.  Streamlines converge to the path
    from both above and below, showing it acts as a stable limit set.

    Compare to:
    - The uploaded image (single attractor at goal): all streamlines
      converge to one point.
    - This plot: streamlines converge to the WHOLE demo path, with points
      ALONG the path moving forward (along the path direction) rather than
      being fixed attractors.
    """
    if model is None:
        model = load_model()
    model.eval()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax_idx, (ax, show_streamlines) in enumerate(zip(axes, [False, True])):
        _draw_scene(ax, show_foam=False)

        # Reference path
        ref_path = model.f.ref_path.cpu().numpy()
        ax.plot(ref_path[:, 0], ref_path[:, 1], 'k-', lw=3,
                label='Demo path (trajectory attractor)', zorder=10)
        ax.scatter(X_START[0], X_START[1], c='blue', s=150, zorder=11, label='Start')
        ax.scatter(X_GOAL[0], X_GOAL[1], c='lime', s=150, marker='*', zorder=11, label='Goal')

        # Vector field on a grid
        nx, ny = 25, 22
        xs = np.linspace(SLAB_X[0] + 0.003, SLAB_X[1] - 0.003, nx)
        ys = np.linspace(SLAB_Y[0] + 0.003, SLAB_Y[1] - 0.003, ny)
        XX, YY = np.meshgrid(xs, ys)
        pts = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)
        pts_t = torch.tensor(pts, device=DEVICE)

        with torch.no_grad():
            s_vals = model.f.get_progress(pts_t)
            vel    = model.f(pts_t, s_vals).cpu().numpy()

        U = vel[:, 0].reshape(ny, nx)
        V = vel[:, 1].reshape(ny, nx)
        speed = np.sqrt(U**2 + V**2) + 1e-6
        speed_grid = speed.copy()

        if show_streamlines:
            # Dense streamplot to show flow converging to path
            x_fine = np.linspace(SLAB_X[0], SLAB_X[1], 80)
            y_fine = np.linspace(SLAB_Y[0], SLAB_Y[1], 65)
            XF, YF = np.meshgrid(x_fine, y_fine)
            ptsf   = np.stack([XF.ravel(), YF.ravel()], axis=1).astype(np.float32)
            ptsf_t = torch.tensor(ptsf, device=DEVICE)
            with torch.no_grad():
                sf = model.f.get_progress(ptsf_t)
                vf = model.f(ptsf_t, sf).cpu().numpy()
            UF = vf[:, 0].reshape(len(y_fine), len(x_fine))
            VF = vf[:, 1].reshape(len(y_fine), len(x_fine))
            ax.streamplot(x_fine, y_fine, UF, VF,
                          color='steelblue', linewidth=0.8,
                          density=1.5, arrowsize=0.9, zorder=5)
            ax.set_title(
                "Streamlines — Whole Trajectory as Attractor\n"
                "Flow converges to the demo path, not just the goal",
                fontsize=10)
        else:
            # Quiver (arrow field)
            Un = U / speed; Vn = V / speed
            q = ax.quiver(XX, YY, Un, Vn, speed_grid,
                          cmap='Blues', scale=40, width=0.003,
                          alpha=0.85, zorder=5)
            plt.colorbar(q, ax=ax, label='Speed [m/s]')
            ax.set_title(
                "Vector Field f_θ(x, s) — Progress-Conditioned DS\n"
                "ẋ = v_net(x̃, s) + K·(x_ref(s) − x)",
                fontsize=10)

        ax.legend(fontsize=9, loc='upper left')

    plt.suptitle(
        "Trajectory-Attractor Dynamical System\n"
        "The ENTIRE demo path is an attractor (not just the goal point)\n"
        "Key difference from single-goal DS: path is stable limit set",
        fontsize=12)
    plt.tight_layout()
    out = os.path.join(RESULT_DIR, "vector_field_trajectory_attractor.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[Plot] Vector field saved → {out}")


def plot_barrier(model: BPCBFModel = None):
    """Plot the learned barrier B_φ(x) — should be positive in safe region, negative in critical."""
    if model is None:
        model = load_model()
    model.eval()

    nx, ny = 250, 200
    xs = np.linspace(SLAB_X[0], SLAB_X[1], nx)
    ys = np.linspace(SLAB_Y[0], SLAB_Y[1], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)
    pts_t = torch.tensor(pts, device=DEVICE)

    with torch.no_grad():
        B_vals = model.B(pts_t).squeeze(-1).cpu().numpy().reshape(ny, nx)

    from analytical_sdf import sdf_critical_np as sdf_crit
    sdf_c = sdf_all_critical_np(pts).reshape(ny, nx)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: learned barrier
    ax = axes[0]
    cf = ax.contourf(xs, ys, B_vals, levels=30, cmap='RdYlGn')
    ax.contour(xs, ys, B_vals, levels=[0], colors='black', linewidths=2, linestyles='--', zorder=5)
    plt.colorbar(cf, ax=ax, label='B_φ(x) (learned)')
    _draw_scene(ax, show_foam=False)
    ref_path = model.f.ref_path.cpu().numpy()
    ax.plot(ref_path[:, 0], ref_path[:, 1], 'b-', lw=2, label='Demo path', zorder=8)
    ax.set_title("Learned Barrier B_φ(x)\n(dashed=zero level, green=safe, red=unsafe)")
    ax.legend()

    # Right: analytic SDF for comparison
    ax2 = axes[1]
    cf2 = ax2.contourf(xs, ys, sdf_c, levels=30, cmap='RdYlGn')
    ax2.contour(xs, ys, sdf_c, levels=[0], colors='black', linewidths=2, linestyles='--', zorder=5)
    plt.colorbar(cf2, ax=ax2, label='Analytic SDF (ground truth label)')
    _draw_scene(ax2, show_foam=False)
    ax2.plot(ref_path[:, 0], ref_path[:, 1], 'b-', lw=2, label='Demo path', zorder=8)
    ax2.set_title("Analytic SDF (used as labels only)\n"
                  "B_φ learned from scratch — no analytic formula in B_φ itself")
    ax2.legend()

    plt.suptitle("Learned Barrier Certificate vs Analytic SDF\n"
                 "Critical obstacles have non-linear shapes (star, crescent, kidney, etc.)\n"
                 "B_φ is trained to match this boundary via S2-NNDS loss", fontsize=11)
    plt.tight_layout()
    out = os.path.join(RESULT_DIR, "learned_barrier_field.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[Plot] Barrier field saved → {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='all',
                        choices=['nominal', 'shortest_path', 'bpcbf', 'all'])
    parser.add_argument('--plot_field', action='store_true',
                        help='Just plot the vector field (no sim)')
    parser.add_argument('--plot_barrier', action='store_true',
                        help='Just plot the learned barrier field')
    args = parser.parse_args()

    if args.plot_field:
        plot_vector_field()
    elif args.plot_barrier:
        plot_barrier()
    elif args.mode == 'all':
        run_all()
    else:
        traj = run_simulation(mode=args.mode)
        model = load_model()
        plot_vector_field(model)
        plot_barrier(model)
