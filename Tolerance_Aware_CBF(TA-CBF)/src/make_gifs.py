"""
make_gifs.py — 2D closed-loop rollout GIFs of the learned-CBF needle controller,
showing safety (never entering a critical region / its inflated buffer) and goal
reachability, with the SAME barrier re-evaluated each control step.

Three sections, three GIFs each, written under results/:
  positional_augmentation/  obstacle at three different static poses
  dynamic_env/              obstacle expanding / shrinking / expanding+rotating
  moving_env/               obstacle translating / rotating / translating+rotating

Usage:
    cd /home/stanny/franka_ros2_ws/src/Tolerance_Aware_CBF(TA-CBF)
    CUDA_VISIBLE_DEVICES="" python src/make_gifs.py            # all nine
    CUDA_VISIBLE_DEVICES="" python src/make_gifs.py --only moving_env
"""
import argparse
import os
import sys

import imageio.v2 as imageio
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from config import (DEVICE, X_START, X_GOAL, DEMO_WAYPOINTS, SLAB_X, SLAB_Y,
                    Z_CORRIDOR, CRITICAL_SHAPES, INFLATE_MARGIN, VEL_CLIP,
                    GOAL_TOL, T_MAX, DT, transform_shape, _rot2d,
                    canonical_interior_cloud, sdf_critical_shape_2d)
from simulate import load_model, BPCBFController
from cbf_qp import analytic_safety_filter

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")

# solvable curated layout: four static "wall" obstacles in the corners + one
# subject obstacle near the demo path (the one that moves / grows / is re-posed).
STATIC_WALLS = [
    (5, (0.450, 0.150)), (3, (0.550, 0.150)),
    (2, (0.420, 0.005)), (4, (0.580, 0.005)),
]
SUBJECT_IDX = 0                                   # the star
SUBJECT_BASE = np.array([0.500, 0.068], np.float32)

CANON = None   # lazy per-shape canonical clouds


def _canon():
    global CANON
    if CANON is None:
        CANON = [canonical_interior_cloud(sh, k=128, seed=300 + i)
                 for i, sh in enumerate(CRITICAL_SHAPES)]
    return CANON


def build_scene(subj_rot, subj_scale, subj_trans):
    """Transformed shapes (for SDF/plot) + obstacle tensors (cloud+center)."""
    canon = _canon()
    shapes_t, obstacles = [], []
    entries = []
    subj = transform_shape(CRITICAL_SHAPES[SUBJECT_IDX], d_rot=subj_rot, scale=subj_scale)
    subj['center'] = np.array([SUBJECT_BASE[0] + subj_trans[0],
                               SUBJECT_BASE[1] + subj_trans[1], Z_CORRIDOR], np.float32)
    entries.append((subj, SUBJECT_IDX, subj_rot, subj_scale))
    for idx, (cx, cy) in STATIC_WALLS:
        st = transform_shape(CRITICAL_SHAPES[idx], d_rot=0.0, scale=1.0)
        st['center'] = np.array([cx, cy, Z_CORRIDOR], np.float32)
        entries.append((st, idx, 0.0, 1.0))
    for sh_t, idx, rot, scl in entries:
        shapes_t.append(sh_t)
        cloud = (scl * canon[idx]) @ _rot2d(rot).T
        obstacles.append({'cloud': torch.tensor(cloud.astype(np.float32), device=DEVICE),
                          'center': torch.tensor(sh_t['center'].astype(np.float32), device=DEVICE)})
    return shapes_t, obstacles


def sdf_min(shapes_t, xy):
    return min(sdf_critical_shape_2d(np.asarray(xy, np.float32)[None], sh)[0]
               for sh in shapes_t)


def run_rollout(model, ctrl, scene_at, tau_dur, lookahead, max_seconds=30.0):
    """Closed-loop rollout; scene_at(tau)->(shapes_t, obstacles).
    Returns traj (N,3), taus (N,), reached, min_sdf."""
    x = X_START.copy().astype(np.float64)
    t, step, s_prev = 0.0, 0, 0.0
    max_step = int(max_seconds / DT)
    x0x, xgx = float(X_START[0]), float(X_GOAL[0])
    T_DEMO = T_MAX * 0.7
    traj, taus = [x.copy()], [0.0]
    reached = False
    min_sdf = 1e9
    while step < max_step:
        tau = min(1.0, t / tau_dur)
        tau_p = min(1.0, tau + lookahead)
        shapes_now, _ = scene_at(tau)
        _, obs_pred = scene_at(tau_p)
        model.set_obstacles(obs_pred)

        xt = torch.tensor(x, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        s_x = float(np.clip((x[0] - x0x) / (xgx - x0x + 1e-9), 0.0, 1.0))
        s = max(s_prev, 0.5 * (s_x + min(1.0, t / T_DEMO))); s_prev = s
        with torch.no_grad():
            f_val = model.f(xt, s).cpu().numpy().flatten()
        dg = float(np.linalg.norm(x - X_GOAL))
        if dg < 0.025:
            a = 1.0 - dg / 0.025
            f_val = (1 - a) * f_val + a * 0.04 * (X_GOAL - x) / (dg + 1e-9)
        u, _ = ctrl.solve(x.astype(np.float32), model, device=DEVICE, s=s)
        xdot = np.clip(f_val + u, -VEL_CLIP, VEL_CLIP)
        xdot = ctrl.project_safe(x.astype(np.float32), model, xdot, DT, device=DEVICE)
        xdot = analytic_safety_filter(x.astype(np.float32), xdot, DT, shapes_now)
        x = x + xdot * DT
        t += DT; step += 1
        traj.append(x.copy()); taus.append(tau)
        min_sdf = min(min_sdf, sdf_min(shapes_now, x[:2]))
        if np.linalg.norm(x - X_GOAL) < GOAL_TOL:
            reached = True; break
    return np.array(traj), np.array(taus), reached, min_sdf


# ── frame drawing ──────────────────────────────────────────────────────────────
_GX = np.linspace(SLAB_X[0], SLAB_X[1], 220)
_GY = np.linspace(SLAB_Y[0], SLAB_Y[1], 180)
_XX, _YY = np.meshgrid(_GX, _GY)
_PTS = np.stack([_XX.ravel(), _YY.ravel()], 1).astype(np.float32)
_BX = np.linspace(SLAB_X[0], SLAB_X[1], 110)
_BY = np.linspace(SLAB_Y[0], SLAB_Y[1], 90)
_BXX, _BYY = np.meshgrid(_BX, _BY)
_BPTS3 = np.concatenate([np.stack([_BXX.ravel(), _BYY.ravel()], 1),
                         np.full((_BXX.size, 1), Z_CORRIDOR)], 1).astype(np.float32)


def _sdf_grid(shapes_t):
    sd = np.min(np.stack([sdf_critical_shape_2d(_PTS, sh) for sh in shapes_t], 1), 1)
    return sd.reshape(_XX.shape)


def draw_frame(fig, ax, shapes_t, model, traj, fi, title, safe):
    ax.clear()
    sd = _sdf_grid(shapes_t)
    ax.contourf(_GX, _GY, sd, levels=[-10, 0], colors=['#ff3333'], alpha=0.60, zorder=2)
    ax.contourf(_GX, _GY, sd, levels=[0, INFLATE_MARGIN], colors=['#ffb0b0'], alpha=0.5, zorder=2)
    ax.contour(_GX, _GY, sd, levels=[0], colors=['#aa0000'], linewidths=1.2, zorder=3)
    # learned barrier zero level set (purple dashed) — re-evaluated for this pose
    with torch.no_grad():
        B = model.B(torch.tensor(_BPTS3, device=DEVICE)).squeeze(-1).cpu().numpy().reshape(_BXX.shape)
    ax.contour(_BX, _BY, B, levels=[0], colors=['#7a1fa2'], linewidths=1.4,
               linestyles='--', zorder=4)
    # demo path + rollout so far
    ax.plot(DEMO_WAYPOINTS[:, 0], DEMO_WAYPOINTS[:, 1], 'b--', lw=1.2, alpha=0.7, zorder=5)
    tr = traj[:fi + 1]
    ax.plot(tr[:, 0], tr[:, 1], color='#0b8f3a', lw=2.6, zorder=8)
    ax.scatter(traj[fi, 0], traj[fi, 1], c='black', s=60, zorder=9)
    ax.scatter(X_START[0], X_START[1], c='blue', s=55, zorder=7)
    ax.scatter(X_GOAL[0], X_GOAL[1], c='lime', marker='*', s=180,
               edgecolors='k', linewidths=0.6, zorder=7)
    ax.set_xlim(SLAB_X); ax.set_ylim(SLAB_Y); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    tag = 'SAFE' if safe else 'UNSAFE'
    ax.set_title(f"{title}\n{tag} — barrier re-evaluated each step", fontsize=10)
    # legend proxies
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(fc='#ff3333', alpha=0.6, label='critical region'),
        Patch(fc='#ffb0b0', alpha=0.6, label='safety buffer'),
        Line2D([0], [0], color='#7a1fa2', ls='--', label='learned B=0'),
        Line2D([0], [0], color='#0b8f3a', lw=2.4, label='needle'),
        Line2D([0], [0], marker='*', color='lime', ls='', mec='k', label='goal'),
    ], fontsize=6.5, loc='lower left', framealpha=0.9)


def render_gif(model, ctrl, scene_at, out_path, title, tau_dur, lookahead,
               n_frames=64, fps=14):
    traj, taus, reached, min_sdf = run_rollout(model, ctrl, scene_at, tau_dur, lookahead)
    safe = min_sdf >= 0
    idx = np.linspace(0, len(traj) - 1, min(n_frames, len(traj))).astype(int)
    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=95)
    frames = []
    for fi in idx:
        shapes_t, obs = scene_at(float(taus[fi]))
        model.set_obstacles(obs)
        draw_frame(fig, ax, shapes_t, model, traj, fi, title, safe)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    # hold last frame
    frames += [frames[-1]] * int(fps * 1.2)
    plt.close(fig)
    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    st = 'reached goal' if reached else 'did not reach'
    print(f"  saved {os.path.relpath(out_path, ROOT)}  "
          f"[{'SAFE' if safe else 'UNSAFE'}, {st}, min_sdf={min_sdf*1e3:.1f}mm, "
          f"{len(idx)} frames]", flush=True)
    return safe, reached


# ── scenario definitions ───────────────────────────────────────────────────────
def _wrap(scene_fn):
    """(subj_rot,subj_scale,subj_trans) motion -> scene_at(tau)->(shapes,obs)."""
    return lambda tau: build_scene(*scene_fn(tau))


def moving_dynamic_scenarios():
    z = np.zeros(2, np.float32)
    return {
        'moving_env': [
            ('translation', _wrap(lambda tau: (0.0, 0.7, np.array([0.0, -0.010 + 0.024 * tau], np.float32))),
             T_MAX * 0.9, 0.10, "Obstacle translation"),
            ('rotation', _wrap(lambda tau: (1.3 * np.pi * tau, 0.7, z)),
             T_MAX * 0.9, 0.06, "Obstacle rotation"),
            ('translation_rotation', _wrap(lambda tau: (1.1 * np.pi * tau, 0.7, np.array([0.0, 0.012 * tau], np.float32))),
             T_MAX * 0.9, 0.08, "Obstacle translation + rotation"),
        ],
        'dynamic_env': [
            ('expanding', _wrap(lambda tau: (0.0, 0.5 + 0.42 * tau, z)),
             T_MAX * 0.9, 0.10, "Obstacle expanding"),
            ('shrinking', _wrap(lambda tau: (0.0, 0.92 - 0.42 * tau, z)),
             T_MAX * 0.9, 0.0, "Obstacle shrinking"),
            ('expanding_rotating', _wrap(lambda tau: (1.0 * np.pi * tau, 0.5 + 0.40 * tau, z)),
             T_MAX * 0.9, 0.08, "Obstacle expanding + rotating"),
        ],
    }


def positional_scenarios(model, ctrl):
    """The SAME validated static scenes as the gen_Nobs.png generalization figures
    (make_final.run_generalization: same (n_obs, seed0)), animated as rollouts.
    Each is validated so the needle reaches the goal (reachability where possible)."""
    from generalization_test import build_tensors, make_validated_scene
    items = []
    for n_obs, seed0 in [(4, 1), (5, 8), (6, 20)]:
        shapes, seed = make_validated_scene(model, ctrl, n_obs, seed0)
        obs = build_tensors(shapes, k=64, seed=1)
        scene_at = (lambda tau, sh=shapes, ob=obs: (sh, ob))
        items.append((f"gen_{n_obs}obs", scene_at, 1.0, 0.0,
                      f"{n_obs} obstacles (random pose + scale, zero-shot)"))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None,
                    help="subset of sections to render")
    args = ap.parse_args()
    model = load_model(); ctrl = BPCBFController()

    secs = moving_dynamic_scenarios()
    if not args.only or 'positional_augmentation' in args.only:
        secs['positional_augmentation'] = positional_scenarios(model, ctrl)

    for section, items in secs.items():
        if args.only and section not in args.only:
            continue
        print(f"\n=== {section} ===", flush=True)
        outdir = os.path.join(RES, section); os.makedirs(outdir, exist_ok=True)
        for name, scene_at, tau_dur, la, title in items:
            render_gif(model, ctrl, scene_at, os.path.join(outdir, f"{name}.gif"),
                       title, tau_dur, la)


if __name__ == "__main__":
    main()
