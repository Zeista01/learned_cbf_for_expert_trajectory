"""
run_rollouts.py — Orchestrator for all simulation rollouts.

Saves everything under results/roll_outs/{normal, generalized, dynamical}/

Usage:
    cd /home/stanny/franka_ros2_ws/src/Tolerance_Aware_CBF(TA-CBF)
    python src/run_rollouts.py                        # all three types
    python src/run_rollouts.py --type normal          # only normal rollouts
    python src/run_rollouts.py --type generalized     # only generalization
    python src/run_rollouts.py --type dynamical       # only dynamic motions
    python src/run_rollouts.py --n 12 --radius 0.020  # 12 normal rollouts
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Project root resolution ───────────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR  = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, SRC_DIR)

ROLL_DIR = os.path.join(ROOT_DIR, "results", "roll_outs")
DIR_NORM = os.path.join(ROLL_DIR, "normal")
DIR_GEN  = os.path.join(ROLL_DIR, "generalized")
DIR_DYN  = os.path.join(ROLL_DIR, "dynamical")

for d in (DIR_NORM, DIR_GEN, DIR_DYN):
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Shared imports (after sys.path is set)
# ─────────────────────────────────────────────────────────────────────────────
from config import (
    X_START, X_GOAL, CRITICAL_SHAPES, DEMO_WAYPOINTS,
    SLAB_X, SLAB_Y, INFLATE_MARGIN, B_SAFE_MARGIN,
    BARRIER_SDF_K, Z_CORRIDOR,
    sdf_critical_shape_2d, canonical_interior_cloud,
    SAFETY_SDF_MARGIN, VEL_CLIP, GOAL_TOL, T_MAX, DT, DEVICE,
)
from simulate import load_model, _draw_scene, _count_critical_entries
from cbf_qp import BPCBFController, analytic_safety_filter
import torch


# ─────────────────────────────────────────────────────────────────────────────
#  Normal Rollouts
# ─────────────────────────────────────────────────────────────────────────────

def run_from_start_local(model, ctrl, x_start, shapes=None):
    """Single bpcbf rollout; mirrors multi_rollout.run_from_start."""
    from multi_rollout import run_from_start
    return run_from_start(model, ctrl, x_start, shapes=shapes)


def run_normal_rollouts(model, ctrl, n_rollouts=10, radius=0.018, seed=0):
    """n_rollouts from perturbed starts; saves individual + overlay PNG."""
    rng = np.random.default_rng(seed)
    shapes = CRITICAL_SHAPES

    starts = [X_START.copy()]
    for _ in range(n_rollouts - 1):
        sp = X_START.copy()
        sp[:2] += rng.uniform(-radius, radius, 2)
        starts.append(sp)

    rollouts = []
    for i, sp in enumerate(starts):
        traj = run_from_start_local(model, ctrl, sp, shapes=shapes)
        n_solid, n_buf = _count_critical_entries(traj['ee_pos'])
        tag = "SAFE" if n_solid == 0 else f"{n_solid} CRITICAL ENTRIES"
        print(f"  [Normal {i+1:2d}/{n_rollouts}] start=({sp[0]:.4f},{sp[1]:.4f})  "
              f"reached={traj['reached_goal']}  {tag}")
        rollouts.append((sp, traj, n_solid))

    # Individual plots
    for i, (sp, traj, n_solid) in enumerate(rollouts):
        fig, ax = plt.subplots(figsize=(7, 6))
        _draw_scene(ax)
        tr = np.array(traj['ee_pos'])
        ax.plot(tr[:, 0], tr[:, 1], color='#2196F3', lw=2, zorder=7, label='Trajectory')
        ax.scatter(sp[0], sp[1], c='blue', s=120, zorder=9, label='Start')
        ax.scatter(X_GOAL[0], X_GOAL[1], c='lime', s=160, marker='*', zorder=9, label='Goal')
        ax.plot(DEMO_WAYPOINTS[:, 0], DEMO_WAYPOINTS[:, 1],
                'k--', lw=1, alpha=0.5, label='Demo path')
        status = "✓ SAFE" if n_solid == 0 else f"✗ {n_solid} penetrations"
        ax.set_title(f"Normal Rollout {i+1}  |  {status}\n"
                     f"start=({sp[0]:.4f},{sp[1]:.4f})  reached={traj['reached_goal']}",
                     fontsize=10)
        ax.legend(fontsize=8)
        plt.tight_layout()
        out = os.path.join(DIR_NORM, f"normal_rollout_{i+1:02d}.png")
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    → saved: {out}")

    # Overlay
    fig, ax = plt.subplots(figsize=(9, 7))
    _draw_scene(ax, title=f"Normal Rollouts — {n_rollouts} perturbed starts (radius={radius*100:.1f}cm)")
    cmap = plt.cm.plasma(np.linspace(0.1, 0.9, len(rollouts)))
    safe_count = 0
    for i, (sp, traj, n_solid) in enumerate(rollouts):
        tr = np.array(traj['ee_pos'])
        lbl = f"R{i+1}" + ("" if n_solid == 0 else f" ({n_solid}✗)")
        ax.plot(tr[:, 0], tr[:, 1], color=cmap[i], lw=1.8, zorder=7, label=lbl)
        ax.scatter(sp[0], sp[1], color=cmap[i], s=50, zorder=9, edgecolors='k', lw=0.5)
        if n_solid == 0:
            safe_count += 1
    ax.plot(DEMO_WAYPOINTS[:, 0], DEMO_WAYPOINTS[:, 1],
            'k--', lw=1.5, alpha=0.6, label='Demo', zorder=3)
    ax.scatter(X_GOAL[0], X_GOAL[1], c='lime', s=200, marker='*', zorder=10, label='Goal')
    ax.legend(fontsize=7, loc='upper left', ncol=2)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    fig.text(0.5, 0.01, f"Safe rollouts: {safe_count}/{n_rollouts}  |  Obstacles: {len(shapes)}",
             ha='center', fontsize=10, color='#333')
    plt.tight_layout()
    out = os.path.join(DIR_NORM, "normal_overlay.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    → overlay: {out}")

    # Statistics plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    path_lengths = [np.sum(np.linalg.norm(np.diff(np.array(t['ee_pos']), axis=0), axis=1))
                    for _, t, _ in rollouts]
    times        = [t['time'][-1] for _, t, _ in rollouts]
    reached      = [int(t['reached_goal']) for _, t, _ in rollouts]
    n_pen        = [n for _, _, n in rollouts]

    axes[0].bar(range(1, n_rollouts + 1), path_lengths, color=cmap, edgecolor='k', lw=0.5)
    axes[0].axhline(np.mean(path_lengths), color='red', ls='--', label=f'Mean={np.mean(path_lengths)*100:.1f}cm')
    axes[0].set_xlabel("Rollout #"); axes[0].set_ylabel("Path length [m]")
    axes[0].set_title("Path Lengths"); axes[0].legend()

    bar_colors = ['#27ae60' if r else '#e74c3c' for r in reached]
    axes[1].bar(range(1, n_rollouts + 1), times, color=bar_colors, edgecolor='k', lw=0.5)
    axes[1].set_xlabel("Rollout #"); axes[1].set_ylabel("Time to goal [s]")
    axes[1].set_title(f"Time to Goal  (green=reached, red=failed)\n"
                      f"Safety: {safe_count}/{n_rollouts} safe  |  "
                      f"Reached: {sum(reached)}/{n_rollouts}")
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(color='#27ae60', label='Reached'),
                             Patch(color='#e74c3c', label='Failed')], fontsize=8)
    plt.tight_layout()
    out = os.path.join(DIR_NORM, "normal_statistics.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    → stats: {out}")

    print(f"\n  [Normal] Summary: {safe_count}/{n_rollouts} safe, "
          f"{sum(reached)}/{n_rollouts} reached goal")


# ─────────────────────────────────────────────────────────────────────────────
#  Generalized Rollouts
# ─────────────────────────────────────────────────────────────────────────────

def run_generalized_rollouts(model, ctrl, n_obs_list=None, seeds=None, rollouts_per=4):
    """Zero-shot generalization: random pose/scale obstacle sets."""
    import multi_rollout as MR
    from generalization_test import (make_validated_scene, build_tensors,
                                      count_critical_entries, draw_scene_generic,
                                      sdf_all_np)

    if n_obs_list is None:
        n_obs_list = [3, 4, 5, 6]
    if seeds is None:
        seeds = [1, 10, 20, 30]

    all_results = []
    for n_obs, seed0 in zip(n_obs_list, seeds):
        print(f"\n  [Gen] n_obs={n_obs}, seed={seed0} — building validated scene...")
        try:
            shapes, used_seed = make_validated_scene(model, ctrl, n_obs, seed0)
        except Exception as e:
            print(f"    ⚠ Could not make validated scene: {e}")
            continue

        # Multiple rollouts on this scene
        scene_rollouts = []
        starts = [X_START.copy()]
        rng2 = np.random.default_rng(used_seed + 500)
        for _ in range(rollouts_per - 1):
            sp = X_START.copy(); sp[:2] += rng2.uniform(-0.010, 0.010, 2)
            starts.append(sp)

        for i, sp in enumerate(starts):
            traj = MR.run_from_start(model, ctrl, sp, shapes=shapes)
            n_s, n_lr = count_critical_entries(np.array(traj['ee_pos']), shapes)
            tag = "SAFE" if n_s == 0 else f"{n_s} penetrations"
            print(f"    Rollout {i+1}: reached={traj['reached_goal']}  {tag}")
            scene_rollouts.append((sp, traj, n_s, n_lr))

        # Plot this scene
        fig, ax = plt.subplots(figsize=(8, 7))
        draw_scene_generic(ax, shapes)
        ax.plot(DEMO_WAYPOINTS[:, 0], DEMO_WAYPOINTS[:, 1],
                'k--', lw=1.5, alpha=0.5, label='Demo')
        cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(scene_rollouts)))
        safe_ct = 0
        for j, (sp, traj, n_s, n_lr) in enumerate(scene_rollouts):
            tr = np.array(traj['ee_pos'])
            ax.plot(tr[:, 0], tr[:, 1], color=cmap[j], lw=2, zorder=7,
                    label=f"R{j+1} {'✓' if n_s==0 else '✗'}")
            ax.scatter(sp[0], sp[1], color=cmap[j], s=60, zorder=9,
                       edgecolors='k', lw=0.5)
            if n_s == 0:
                safe_ct += 1
        ax.scatter(X_GOAL[0], X_GOAL[1], c='lime', s=200, marker='*', zorder=10, label='Goal')
        ax.legend(fontsize=8); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.set_title(f"Generalization — {n_obs} obstacles (random pose+scale+rotation)\n"
                     f"Zero-shot: no barrier retraining  |  "
                     f"Safe: {safe_ct}/{len(scene_rollouts)}", fontsize=10)
        plt.tight_layout()
        name = f"gen_{n_obs}obs_seed{used_seed}.png"
        out = os.path.join(DIR_GEN, name)
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    → saved: {out}")

        all_results.append((n_obs, safe_ct, len(scene_rollouts)))

    # Summary bar chart
    if all_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = [f"{n} obs" for n, _, _ in all_results]
        safe_rates = [s / t * 100 for _, s, t in all_results]
        bars = ax.bar(labels, safe_rates,
                      color=['#27ae60' if r == 100 else '#e67e22' if r >= 50 else '#e74c3c'
                             for r in safe_rates],
                      edgecolor='k', lw=0.8)
        ax.bar_label(bars, labels=[f"{r:.0f}%" for r in safe_rates], fontsize=12)
        ax.set_ylim(0, 110)
        ax.set_ylabel("Safety rate [%]")
        ax.set_title("Generalization Safety Rate vs Obstacle Count\n"
                     "(zero-shot: same model, different poses/scales)")
        ax.axhline(100, color='green', ls='--', alpha=0.5, label='100%')
        ax.legend(fontsize=9)
        plt.tight_layout()
        out = os.path.join(DIR_GEN, "gen_safety_summary.png")
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n    → summary: {out}")

    print(f"\n  [Gen] Done: {len(all_results)} scene configs tested")


# ─────────────────────────────────────────────────────────────────────────────
#  Dynamical Rollouts
# ─────────────────────────────────────────────────────────────────────────────

def run_dynamical_rollouts(model, ctrl, motions=None, frames=5):
    """All 4 dynamic motions with multi-frame snapshots."""
    import dynamic_env_test as D

    if motions is None:
        motions = ["translate", "rotate", "transrotate", "evolve"]

    canon_clouds = [canonical_interior_cloud(sh, k=128, seed=300 + i)
                    for i, sh in enumerate(CRITICAL_SHAPES)]

    summary = {}
    for motion in motions:
        print(f"\n  [Dynamic] Running motion: {motion} ...")
        n_intrusions = D.render_motion(model, ctrl, motion, canon_clouds,
                                        frames, quick=False)
        summary[motion] = n_intrusions

        # Move output from default results/dynamic/ to our dir
        src_file = os.path.join(ROOT_DIR, "results", "dynamic", f"dynamic_{motion}.png")
        dst_file = os.path.join(DIR_DYN, f"dynamic_{motion}.png")
        if os.path.exists(src_file):
            import shutil
            shutil.copy2(src_file, dst_file)
            print(f"    → copied: {dst_file}")

    # Summary plot
    fig, ax = plt.subplots(figsize=(8, 5))
    motion_labels = [m.capitalize() for m in motions]
    colors = ['#27ae60' if summary[m] == 0 else '#e74c3c' for m in motions]
    bars = ax.bar(motion_labels,
                  [1 if summary[m] == 0 else 0 for m in motions],
                  color=colors, edgecolor='k', lw=0.8)
    for i, m in enumerate(motions):
        label = "SAFE ✓" if summary[m] == 0 else f"✗ {summary[m]}"
        ax.text(i, 0.5, label, ha='center', va='center',
                fontsize=13, color='white', fontweight='bold')
    ax.set_ylim(0, 1.2)
    ax.set_yticks([])
    ax.set_title("Dynamic Environment — Safety Summary\n"
                 "(obstacle moves/rotates/grows during needle insertion)", fontsize=11)
    ax.set_xlabel("Motion Type")
    plt.tight_layout()
    out = os.path.join(DIR_DYN, "dynamic_summary.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n    → summary: {out}")

    print(f"\n  [Dynamic] Summary: " +
          "  ".join(f"{m}={'SAFE' if n==0 else f'{n}✗'}" for m, n in summary.items()))


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Run all TA-CBF rollout simulations")
    ap.add_argument("--type", choices=["normal", "generalized", "dynamical", "all"],
                    default="all", help="which rollout type to run")
    ap.add_argument("--n", type=int, default=10,
                    help="number of normal rollouts (default: 10)")
    ap.add_argument("--radius", type=float, default=0.018,
                    help="start perturbation radius in meters (default: 0.018)")
    ap.add_argument("--seed", type=int, default=42,
                    help="random seed for normal rollouts")
    ap.add_argument("--n_obs", type=int, nargs="+", default=[3, 4, 5, 6],
                    help="obstacle counts for generalization (default: 3 4 5 6)")
    ap.add_argument("--gen_seeds", type=int, nargs="+", default=[1, 10, 20, 30],
                    help="seeds for generalization scenes")
    ap.add_argument("--motions", nargs="+",
                    default=["translate", "rotate", "transrotate", "evolve"],
                    help="dynamic motions to test")
    ap.add_argument("--frames", type=int, default=5,
                    help="snapshot frames per dynamic motion")
    args = ap.parse_args()

    print(f"\n{'='*60}")
    print(f"  TA-CBF Rollout Runner")
    print(f"  Output: {ROLL_DIR}")
    print(f"{'='*60}\n")

    print("Loading model...")
    model = load_model()
    ctrl  = BPCBFController()
    print("Model loaded.\n")

    run_type = args.type

    if run_type in ("normal", "all"):
        print(f"{'─'*40}")
        print(f"  NORMAL ROLLOUTS  (n={args.n}, radius={args.radius*100:.1f}cm)")
        print(f"{'─'*40}")
        run_normal_rollouts(model, ctrl,
                            n_rollouts=args.n,
                            radius=args.radius,
                            seed=args.seed)

    if run_type in ("generalized", "all"):
        print(f"\n{'─'*40}")
        print(f"  GENERALIZED ROLLOUTS  (n_obs={args.n_obs})")
        print(f"{'─'*40}")
        run_generalized_rollouts(model, ctrl,
                                  n_obs_list=args.n_obs,
                                  seeds=args.gen_seeds)

    if run_type in ("dynamical", "all"):
        print(f"\n{'─'*40}")
        print(f"  DYNAMICAL ROLLOUTS  (motions={args.motions})")
        print(f"{'─'*40}")
        run_dynamical_rollouts(model, ctrl,
                               motions=args.motions,
                               frames=args.frames)

    print(f"\n{'='*60}")
    print(f"  All outputs saved to: {ROLL_DIR}")
    print(f"  Subdirs: normal/  generalized/  dynamical/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
