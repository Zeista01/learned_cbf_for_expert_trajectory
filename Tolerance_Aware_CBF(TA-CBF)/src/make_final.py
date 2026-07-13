"""
make_final.py — generate ALL final, paper-quality figures for the hybrid
learned-CLF/CBF + exact-safety-filter needle controller.

Outputs (under results/FINAL/):
  losses_from_training/   training-loss curves (f/V co-training + barrier)
  final_learned_cbf.png   learned barrier B_φ(x) on the canonical scene
  final_vector_field.png  closed-loop field — diverts around the light-red zone
  generalization_result/  multiple cases (#obstacles, pose/rotation), 6 rollouts each
  dynamic_env/            slow-moving-obstacle results (4 motions), 6 rollouts each

Run:
    cd /home/stanny/franka_ros2_ws/src/Tolerance_Aware_CBF(TA-CBF)
    python src/make_final.py
"""

import os
import sys
import copy
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, os.path.dirname(__file__))
from config import (DEVICE, X_START, X_GOAL, DEMO_WAYPOINTS, SLAB_X, SLAB_Y,
                    CRITICAL_SHAPES, INFLATE_MARGIN, B_SAFE_MARGIN, SAFETY_SDF_MARGIN,
                    Z_CORRIDOR, DEMO_K)
from simulate import load_model, BPCBFController
from multi_rollout import run_from_start
from field_plot import plot_diverting_field
from generalization_test import (make_validated_scene, build_tensors, sdf_all_np,
                                  count_critical_entries, draw_scene_generic)
import dynamic_env_test as D
from train import make_obstacle_tensors

ROOT = os.path.join(os.path.dirname(__file__), "..")
FINAL = os.path.join(ROOT, "results", "FINAL")
DIR_LOSS = os.path.join(FINAL, "losses_from_training")
DIR_GEN  = os.path.join(FINAL, "generalization_result")
DIR_DYN  = os.path.join(FINAL, "dynamic_env")
for d in (FINAL, DIR_LOSS, DIR_GEN, DIR_DYN):
    os.makedirs(d, exist_ok=True)

CKPT = os.path.join(ROOT, "checkpoints")
GREEN = '#1a9850'


# ── helpers ──────────────────────────────────────────────────────────────────

def perturbed_starts(n, radius=0.012, seed=0):
    rng = np.random.default_rng(seed)
    starts = [X_START.copy()]
    for _ in range(n - 1):
        s = X_START.copy(); s[:2] += rng.uniform(-radius, radius, 2)
        starts.append(s)
    return starts


def draw_demo_and_markers(ax):
    ax.plot(DEMO_WAYPOINTS[:, 0], DEMO_WAYPOINTS[:, 1], 'b--', lw=1.6,
            label='Expert demo path', zorder=6)
    ax.scatter(X_START[0], X_START[1], c='blue', s=90, zorder=9, label='Start')
    ax.scatter(X_GOAL[0], X_GOAL[1], c='lime', s=130, marker='*', zorder=9,
               edgecolors='k', label='Goal')


# ── A. training losses ───────────────────────────────────────────────────────

def plot_training_losses():
    # barrier fast-training history
    p = os.path.join(CKPT, "barrier_loss_history.npz")
    if os.path.exists(p):
        h = np.load(p)
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        s = h['step']
        ax[0].plot(s, h['loss'],  label='total', color='k', lw=1)
        ax[0].plot(s, h['reg'],   label='SDF regression', lw=1)
        ax[0].plot(s, h['safe'],  label='safe hinge', lw=1)
        ax[0].plot(s, h['unsafe'], label='unsafe hinge', lw=1)
        ax[0].set_yscale('log'); ax[0].set_xlabel('training step'); ax[0].set_ylabel('loss')
        ax[0].set_title('Barrier $B_\\varphi$ training losses'); ax[0].grid(True, alpha=0.3); ax[0].legend(fontsize=9)
        w = max(1, len(s) // 100)
        def sm(a): return np.convolve(a, np.ones(w)/w, mode='valid')
        ax[1].plot(sm(h['safe_acc'])*100,  label='safe accuracy (B>$\\delta$ outside)', lw=1.5)
        ax[1].plot(sm(h['unsafe_acc'])*100, label='unsafe accuracy (B<$-\\delta$ inside)', lw=1.5)
        ax[1].set_xlabel('training step'); ax[1].set_ylabel('classification accuracy [%]')
        ax[1].set_ylim(0, 101); ax[1].set_title('Barrier classification accuracy')
        ax[1].grid(True, alpha=0.3); ax[1].legend(fontsize=9, loc='lower right')
        plt.suptitle('Barrier (CBF) training convergence', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(DIR_LOSS, 'barrier_training.png'), dpi=160); plt.close()
        print("  saved barrier_training.png")

    # full f/V/B co-training losses (from train.py) — copy if present
    src = os.path.join(ROOT, "results", "training_losses.png")
    if os.path.exists(src):
        import shutil
        shutil.copy(src, os.path.join(DIR_LOSS, "full_training_losses.png"))
        print("  copied full_training_losses.png")


# ── B. final learned barrier + diverting field (canonical scene) ─────────────

def plot_final_barrier(model):
    model.set_obstacles(make_obstacle_tensors(k=64, seed=0))
    nx, ny = 320, 260
    xs = np.linspace(SLAB_X[0], SLAB_X[1], nx); ys = np.linspace(SLAB_Y[0], SLAB_Y[1], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.ravel(), YY.ravel(), np.full(XX.size, Z_CORRIDOR, np.float32)], 1).astype(np.float32)
    with torch.no_grad():
        B = model.B(torch.tensor(pts, device=DEVICE)).squeeze(-1).cpu().numpy().reshape(ny, nx)
    fig, ax = plt.subplots(figsize=(9, 6.6))
    cf = ax.contourf(xs, ys, B, levels=30, cmap='RdYlGn')
    plt.colorbar(cf, ax=ax, label='learned barrier  $B_\\varphi(x)$')
    ax.contour(xs, ys, B, levels=[0], colors='purple', linewidths=2.2, linestyles='--')
    draw_scene_generic(ax, CRITICAL_SHAPES)
    draw_demo_and_markers(ax)
    ax.set_title('Final learned CBF  $B_\\varphi(x)$  (purple dashed = $B{=}0$ boundary)\n'
                 'green = safe, red core = critical tissue, pink = inflated CBF zone', fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    plt.tight_layout(); plt.savefig(os.path.join(FINAL, "final_learned_cbf.png"), dpi=170); plt.close()
    print("  saved final_learned_cbf.png")


def plot_final_field(model, ctrl):
    model.set_obstacles(make_obstacle_tensors(k=64, seed=0))
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_diverting_field(ax, model, CRITICAL_SHAPES, ctrl=ctrl, nx=40, ny=34, stream_density=1.8,
                         title='Closed-loop safety field  $\\dot{x}=f_\\theta(x)+u_{safe}(x)$\n'
                               'streamlines DIVERT around the light-red CBF zone and re-converge to the demo')
    plt.tight_layout(); plt.savefig(os.path.join(FINAL, "final_vector_field.png"), dpi=170); plt.close()
    print("  saved final_vector_field.png")


# ── C. generalization: multiple cases, 6 rollouts each ───────────────────────

def gen_case(model, ctrl, shapes, name, title, n_roll=6):
    model.set_obstacles(build_tensors(shapes, k=64, seed=1))
    starts = perturbed_starts(n_roll, radius=0.012, seed=hash(name) % 997)
    rolls = []
    worst = 0
    for sp in starts:
        tr = run_from_start(model, ctrl, sp, shapes=shapes)
        ns, _ = count_critical_entries(tr['ee_pos'], shapes)
        worst = max(worst, ns); rolls.append((sp, tr, ns))
    tag = '✓ ALL SAFE (obstacle=0)' if worst == 0 else f'✗ {worst} steps inside'

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.4))
    # panel 1: overlaid rollouts
    ax = axes[0]
    draw_scene_generic(ax, shapes, title=f"{title}\n{n_roll} rollouts (perturbed starts) — {tag}")
    draw_demo_and_markers(ax)
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(rolls)))
    for i, (sp, tr, ns) in enumerate(rolls):
        t = tr['ee_pos']
        ax.plot(t[:, 0], t[:, 1], color=cmap[i], lw=1.8, zorder=8)
        ax.scatter(sp[0], sp[1], color=cmap[i], s=30, zorder=9, edgecolors='k', linewidths=0.4)
    ax.legend(fontsize=8, loc='upper left')
    # panel 2: learned barrier on this scene
    ax = axes[1]
    nx, ny = 240, 200
    xs = np.linspace(SLAB_X[0], SLAB_X[1], nx); ys = np.linspace(SLAB_Y[0], SLAB_Y[1], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.ravel(), YY.ravel(), np.full(XX.size, Z_CORRIDOR, np.float32)], 1).astype(np.float32)
    with torch.no_grad():
        B = model.B(torch.tensor(pts, device=DEVICE)).squeeze(-1).cpu().numpy().reshape(ny, nx)
    cf = ax.contourf(xs, ys, B, levels=26, cmap='RdYlGn')
    ax.contour(xs, ys, B, levels=[0], colors='purple', linewidths=2, linestyles='--', zorder=5)
    plt.colorbar(cf, ax=ax, label='$B_\\varphi(x)$ (learned, zero-shot)')
    draw_scene_generic(ax, shapes)
    ax.set_title("Learned CBF on the NEW obstacle set\n(zero-shot — no retraining)", fontsize=10)
    # panel 3: diverting field
    ax = axes[2]
    plot_diverting_field(ax, model, shapes, ctrl=ctrl, nx=26, ny=22, stream_density=1.5,
                         title="Safety field — diverts around B=0,\nreconverges to the demo")
    plt.suptitle(f"Generalization — {title}", fontsize=13)
    plt.tight_layout()
    out = os.path.join(DIR_GEN, f"{name}.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  saved {name}.png  ({tag})")


def run_generalization(model, ctrl):
    cases = [
        (4, 1,  "4 obstacles (spread, random pose + scale)"),
        (5, 8,  "5 obstacles (spread, random pose + scale)"),
        (6, 20, "6 obstacles (spread, random pose + scale)"),
        (7, 40, "7 obstacles (spread, random pose + scale)"),
        (8, 70, "8 obstacles (spread, random pose + scale)"),
    ]
    for n_obs, seed0, title in cases:
        shapes, seed = make_validated_scene(model, ctrl, n_obs, seed0)
        gen_case(model, ctrl, shapes, f"gen_{n_obs}obs", title)


# ── D. dynamic env: 4 motions ────────────────────────────────────────────────

def _sdf_grid(shapes, xs, ys):
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.ravel(), YY.ravel()], 1).astype(np.float32)
    sdfs = [sdf_all_np(pts, [sh]) for sh in shapes]
    return np.min(np.stack(sdfs, 1), 1).reshape(len(ys), len(xs))


def make_dynamic_gif(model, ctrl, motion, canon, traj, taus, reached, ns,
                     n_frames=48, fps=12):
    """Animated GIF: the obstacle slowly changes while the needle navigates."""
    idx = np.linspace(0, len(traj) - 1, n_frames).astype(int)
    nx, ny = 150, 125
    xs = np.linspace(SLAB_X[0], SLAB_X[1], nx)
    ys = np.linspace(SLAB_Y[0], SLAB_Y[1], ny)
    XX, YY = np.meshgrid(xs, ys)
    gpts = np.stack([XX.ravel(), YY.ravel(), np.full(XX.size, Z_CORRIDOR, np.float32)], 1).astype(np.float32)
    tag = '✓ SAFE (never enters tissue)' if ns == 0 else f'✗ {ns} inside'

    fig, ax = plt.subplots(figsize=(7.2, 6.2))

    def draw(fi):
        ax.clear()
        tau = float(taus[fi])
        shapes_t, obstacles = D.build_scene(motion, tau, canon)
        model.set_obstacles(obstacles)
        sdf = _sdf_grid(shapes_t, xs, ys)
        ax.contourf(xs, ys, sdf, levels=[-10, 0], colors=['#e8302a'], alpha=0.85, zorder=2)
        ax.contourf(xs, ys, sdf, levels=[0, INFLATE_MARGIN], colors=['#ffb3b3'], alpha=0.45, zorder=2)
        ax.contour(xs, ys, sdf, levels=[0], colors=['#900'], linewidths=1.3, zorder=3)
        with torch.no_grad():
            B = model.B(torch.tensor(gpts, device=DEVICE)).squeeze(-1).cpu().numpy().reshape(ny, nx)
        ax.contour(xs, ys, B, levels=[0], colors=['purple'], linewidths=1.8, linestyles='--', zorder=4)
        ax.plot(DEMO_WAYPOINTS[:, 0], DEMO_WAYPOINTS[:, 1], 'b--', lw=1.4, zorder=5, label='expert demo')
        ax.plot(traj[:fi + 1, 0], traj[:fi + 1, 1], color=GREEN, lw=2.8, zorder=8, label='needle')
        ax.scatter(traj[fi, 0], traj[fi, 1], c='k', s=55, zorder=9)
        ax.scatter(X_START[0], X_START[1], c='blue', s=70, zorder=7)
        ax.scatter(X_GOAL[0], X_GOAL[1], c='lime', s=120, marker='*', edgecolors='k', zorder=7)
        ax.set_xlim(SLAB_X); ax.set_ylim(SLAB_Y); ax.set_aspect('equal')
        ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc='upper left')
        ax.set_title(f"Slow-changing obstacle — '{motion}'   τ={tau:.2f}\n"
                     f"{tag}   (purple dashed = learned B=0)", fontsize=10)

    anim = FuncAnimation(fig, draw, frames=idx, interval=1000 / fps)
    out = os.path.join(DIR_DYN, f"dynamic_{motion}.gif")
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close()
    print(f"  saved dynamic_{motion}.gif")


def run_dynamic_env(model, ctrl):
    canon = [D.canonical_interior_cloud(sh, k=128, seed=300 + i)
             for i, sh in enumerate(CRITICAL_SHAPES)]
    demo = DEMO_WAYPOINTS[:, :2]
    for motion in ['translate', 'rotate', 'transrotate', 'evolve']:
        traj, taus, ns, nb, reached = D.run_dynamic(model, ctrl, motion, canon)
        tag = '✓ SAFE (obstacle=0)' if ns == 0 else f'✗ {ns} inside'
        # multi-frame snapshot figure
        n_frames = 4
        frame_idx = np.linspace(0, len(traj) - 1, n_frames).astype(int)
        fig, axes = plt.subplots(1, n_frames, figsize=(6 * n_frames, 5.6))
        for ax, fi in zip(axes, frame_idx):
            tau = float(taus[fi])
            shapes_t, obstacles = D.build_scene(motion, tau, canon)
            model.set_obstacles(obstacles)
            plot_diverting_field(ax, model, shapes_t, ctrl=ctrl, nx=24, ny=20, stream_density=1.3,
                                 title=f"{motion}   τ={tau:.2f}  (step {fi})")
            ax.plot(traj[:fi + 1, 0], traj[:fi + 1, 1], color=GREEN, lw=2.6, zorder=10, label='needle')
            ax.scatter(traj[fi, 0], traj[fi, 1], c='black', s=55, zorder=11)
            ax.legend(fontsize=7, loc='upper left')
        plt.suptitle(f"Slow-changing environment — '{motion}'   "
                     f"{tag},  reached={reached}", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(DIR_DYN, f"dynamic_{motion}.png"), dpi=140); plt.close()
        print(f"  saved dynamic_{motion}.png  ({tag}, reached={reached})")
        # animated GIF of the same rollout
        make_dynamic_gif(model, ctrl, motion, canon, traj, taus, reached, ns)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"[FINAL] config: DEMO_K={DEMO_K}  B_margin={B_SAFE_MARGIN}  "
          f"sdf_safety={SAFETY_SDF_MARGIN*1e3:.0f}mm  light-red={INFLATE_MARGIN*1e3:.0f}mm")
    model = load_model()
    ctrl = BPCBFController()
    print("\n[A] training losses ..."); plot_training_losses()
    print("\n[B] final barrier + field ..."); plot_final_barrier(model); plot_final_field(model, ctrl)
    print("\n[C] generalization ..."); run_generalization(model, ctrl)
    print("\n[D] dynamic env ..."); run_dynamic_env(model, ctrl)
    print(f"\n[FINAL] all figures written under {FINAL}")


if __name__ == "__main__":
    main()
