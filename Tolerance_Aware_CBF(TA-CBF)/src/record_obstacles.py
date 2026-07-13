#!/usr/bin/env python3
"""
record_obstacles.py — Sim-to-real step VI.6.3 (obstacle calibration).

Trace each REAL tissue/critical zone with the needle tip (hand-guided in
gravity compensation) and turn the traced outlines into the exact obstacle
representation the trained barrier B_phi expects:

    list of  {'cloud': (K,2) interior points centered at centroid,
              'center': (3,) world centroid in the FR3 base frame}

This is the SAME format produced by config.sample_obstacle_cloud /
build_obstacle_set and consumed by model.set_obstacles(...). We reconstruct a
FILLED interior from the traced boundary (uniform interior sampling) so the
point distribution matches what the PointNet encoder saw at training time —
tracing only the outline would shift the embedding.

READ-ONLY w.r.t. the robot: it never commands motion. You hand-guide the arm
while a gravity-compensation controller is active.

Usage (from project root, with the venv + ROS sourced):
    python src/record_obstacles.py                 # default K=64, fr3
    python src/record_obstacles.py --k 64 --grav   # also switch to gravity_compensation
    python src/record_obstacles.py --out checkpoints/real_obstacles.npz

Workflow:
    For each obstacle:
      [ENTER]  start tracing  -> move the needle tip slowly around the OUTLINE
      [ENTER]  stop tracing   -> obstacle saved, ready for the next one
    [q] or [ESC]              -> finish, build clouds, save + plot

Output:
    checkpoints/real_obstacles.npz   (centers, clouds, labels, raw boundaries)
    results/real_obstacles.png       (verification plot)

The deploy node loads it with load_real_obstacles(path, device) below.
"""

import argparse
import os
import select
import sys
import termios
import time
import tty

import numpy as np

# project src/ is on sys.path when run as `python src/record_obstacles.py`
from config import (
    CRITICAL_RADIUS,
    DEMO_WAYPOINTS,
    X_GOAL,
    X_START,
    Z_CORRIDOR,
)

# ─────────────────────────── interior reconstruction ───────────────────────────
def boundary_to_interior_cloud(boundary_xy: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """
    From an ordered (N,2) boundary trace, sample K points uniformly inside the
    polygon, CENTERED at the polygon centroid. Returns (K,2).

    Matches config.sample_obstacle_cloud: interior points, centered, 2-D.
    """
    from matplotlib.path import Path

    poly = Path(boundary_xy)
    c2 = boundary_xy.mean(axis=0)

    lo = boundary_xy.min(axis=0)
    hi = boundary_xy.max(axis=0)
    span = np.maximum(hi - lo, 1e-4)

    rng = np.random.default_rng(seed)
    pts = []
    attempts = 0
    # reject-sample inside the traced polygon
    while len(pts) < k and attempts < 200000:
        batch = lo + rng.uniform(0.0, 1.0, size=(512, 2)) * span
        inside = poly.contains_points(batch)
        for p in batch[inside]:
            pts.append(p - c2)            # centered, like training clouds
            if len(pts) >= k:
                break
        attempts += 512

    if len(pts) < 3:
        # degenerate trace: fall back to a small disc of radius CRITICAL_RADIUS
        ang = rng.uniform(0, 2 * np.pi, k)
        rad = CRITICAL_RADIUS * np.sqrt(rng.uniform(0, 1, k))
        pts = np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)
        return pts.astype(np.float32)

    return np.array(pts[:k], dtype=np.float32)


# ─────────────────────────── loader (used by deploy) ───────────────────────────
def load_real_obstacles(path: str, device: str = "cpu") -> list:
    """
    Load a recorded .npz into the set_obstacles-ready list of torch tensors:
        [{'cloud': (K,2) tensor, 'center': (3,) tensor}, ...]
    """
    import torch

    data = np.load(path, allow_pickle=True)
    centers = data["centers"]            # (M,3)
    clouds = data["clouds"]              # object array of (K,2)
    obs = []
    for i in range(len(centers)):
        obs.append({
            "cloud":  torch.tensor(np.asarray(clouds[i], dtype=np.float32),
                                   dtype=torch.float32, device=device),
            "center": torch.tensor(np.asarray(centers[i], dtype=np.float32),
                                   dtype=torch.float32, device=device),
        })
    return obs


# ─────────────────────────── keyboard (non-blocking) ───────────────────────────
# Only valid when stdin is a real TTY. Guarded so the module can be imported
# (e.g. load_real_obstacles) from a non-interactive process / the deploy node.
try:
    _settings = termios.tcgetattr(sys.stdin)
except (termios.error, ValueError):
    _settings = None


def _get_key(timeout: float = 0.0) -> str:
    tty.setraw(sys.stdin.fileno())
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    key = sys.stdin.read(1) if r else ""
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, _settings)
    return key


# ─────────────────────────────────── main ──────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="fr3", help="crisp_py robot name")
    ap.add_argument("--k", type=int, default=64, help="points per obstacle cloud (training used 64)")
    ap.add_argument("--rate", type=float, default=20.0, help="boundary sampling rate (Hz)")
    ap.add_argument("--min-step", type=float, default=0.002,
                    help="min tip movement (m) between recorded boundary points")
    ap.add_argument("--grav", action="store_true",
                    help="switch to the 'gravity_compensation' controller for hand-guiding")
    ap.add_argument("--out", default="checkpoints/real_obstacles.npz")
    ap.add_argument("--plot", default="results/real_obstacles.png")
    args = ap.parse_args()

    from crisp_py.robot import make_robot

    robot = make_robot(args.robot)
    print("Waiting for robot (current_pose + joint_states)…")
    robot.wait_until_ready()

    if args.grav:
        try:
            robot.controller_switcher_client.switch_controller("gravity_compensation")
            print("Switched to gravity_compensation — arm is free to hand-guide.")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] could not switch controller ({e}); "
                  "make sure a gravity-comp controller is active manually.")

    print("\n" + "=" * 64)
    print("OBSTACLE BOUNDARY RECORDER")
    print("  [ENTER] start tracing an obstacle  → guide tip around its outline")
    print("  [ENTER] stop  tracing              → obstacle stored")
    print("  [q]/[ESC] finish, build clouds, save")
    print("=" * 64 + "\n")

    boundaries = []          # list of (N,2) raw traces
    tracing = False
    cur = []
    last_p = None
    dt = 1.0 / args.rate

    try:
        while True:
            key = _get_key(timeout=dt)

            if key in ("q", "\x1b"):                       # finish
                if tracing and len(cur) >= 3:
                    boundaries.append(np.array(cur, dtype=np.float32))
                    print(f"  obstacle {len(boundaries)} stored ({len(cur)} pts)")
                break

            if key == "\r":                                # toggle trace
                if not tracing:
                    tracing = True
                    cur = []
                    last_p = None
                    print(f"\n● TRACING obstacle {len(boundaries) + 1} … move the tip "
                          "around the outline, [ENTER] to stop.")
                else:
                    tracing = False
                    if len(cur) >= 3:
                        boundaries.append(np.array(cur, dtype=np.float32))
                        print(f"■ stored obstacle {len(boundaries)} "
                              f"({len(cur)} boundary pts)\n")
                    else:
                        print("  too few points — discarded. Try again.\n")
                continue

            if tracing:
                pos = robot.end_effector_pose.position      # [x, y, z] base frame
                p = np.array([pos[0], pos[1]], dtype=np.float32)
                if last_p is None or np.linalg.norm(p - last_p) >= args.min_step:
                    cur.append(p)
                    last_p = p
                    print(f"\r  pts={len(cur):4d}  tip=({p[0]:+.3f}, {p[1]:+.3f}) m",
                          end="", flush=True)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, _settings)

    if not boundaries:
        print("\nNo obstacles recorded. Nothing saved.")
        return

    # ── build interior clouds + centers ──
    centers, clouds = [], []
    for i, b in enumerate(boundaries):
        cloud = boundary_to_interior_cloud(b, k=args.k, seed=i)
        c2 = b.mean(axis=0)
        centers.append([float(c2[0]), float(c2[1]), float(Z_CORRIDOR)])
        clouds.append(cloud)
        print(f"  obstacle {i + 1}: center=({c2[0]:+.3f}, {c2[1]:+.3f}, "
              f"{Z_CORRIDOR:.3f})  cloud={cloud.shape}")

    centers = np.array(centers, dtype=np.float32)
    labels = np.array([f"real_{i}" for i in range(len(centers))])

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(
        out,
        centers=centers,
        clouds=np.array(clouds, dtype=object),
        labels=labels,
        boundaries=np.array(boundaries, dtype=object),
    )
    print(f"\nSaved {len(centers)} obstacles → {out}")

    # ── verification plot ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 7))
        for i, (b, cl, c) in enumerate(zip(boundaries, clouds, centers)):
            world_cloud = cl + c[:2]
            ax.scatter(world_cloud[:, 0], world_cloud[:, 1], s=8, alpha=0.5,
                       label=f"obstacle {i + 1} interior")
            bb = np.vstack([b, b[0]])
            ax.plot(bb[:, 0], bb[:, 1], lw=1.0, color="k", alpha=0.6)
            ax.plot(c[0], c[1], "x", color="red")
        ax.plot(DEMO_WAYPOINTS[:, 0], DEMO_WAYPOINTS[:, 1], "b--", lw=1.5, label="demo path")
        ax.plot(*X_START[:2], "go", label="start")
        ax.plot(*X_GOAL[:2], "m*", ms=14, label="goal")
        ax.set_aspect("equal")
        ax.set_xlabel("x (m, base frame)")
        ax.set_ylabel("y (m, base frame)")
        ax.set_title("Recorded real obstacles (interior clouds) vs demo path")
        ax.legend(fontsize=8, loc="best")
        plot = os.path.abspath(args.plot)
        os.makedirs(os.path.dirname(plot), exist_ok=True)
        fig.savefig(plot, dpi=130, bbox_inches="tight")
        print(f"Verification plot → {plot}")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] plot failed: {e}")


if __name__ == "__main__":
    main()
