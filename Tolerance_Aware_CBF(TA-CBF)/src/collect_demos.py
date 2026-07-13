"""
collect_demos.py — generate kinesthetic-style demonstration CSVs.

CSV format (no header row, 11 columns) — identical to node_clf_cbf:
  col  0  : time [s]
  col 1-7 : joint positions q1..q7 [rad]  (from IK at each step)
  col 8-10: end-effector position x, y, z [m]  ← NEEDLE TIP position

The needle tip position is the physically meaningful state — it's what
navigates through the foam tissue.  The attachment_site (standard FR3 TCP)
is 0.16 m above the needle tip in world Z.

For N_DEMOS demonstrations:
  - Start position is X_START ± small noise (±3 mm, Z fixed)
  - Middle waypoints perturbed by ±5 mm
  - Goal fixed at X_GOAL
  - 100 Hz, T_TOTAL seconds → ~800 rows each

Usage:
    cd <project_root>
    python src/collect_demos.py
"""

import os, sys
import numpy as np
import mujoco
from scipy.interpolate import CubicSpline

sys.path.insert(0, os.path.dirname(__file__))
from config import (X_START, X_GOAL, DEMO_WAYPOINTS,
                    FOAM_CENTRES, FOAM_RADIUS,
                    CRITICAL_CENTRES, CRITICAL_RADIUS)
from analytical_sdf import (sdf_all_np, penetration_depth_np,
                            is_in_critical_np, is_in_critical_buffer_np)

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE     = os.path.dirname(os.path.realpath(__file__))
FR3_SCENE = os.path.realpath(
    os.path.join(_HERE, "../../node_clf_cbf/assets/mujoco_menagerie"
                        "/franka_fr3/scene.xml"))

DATA_DIR   = "data/data_trajectory"    # CSVs saved here (matches node_clf_cbf)
NPY_DIR    = "data/demos"              # combined .npy for training
N_DEMOS    = 20
DT         = 0.01      # 100 Hz
T_TOTAL    = 8.0       # seconds per demo → ~800 rows
# Noise levels: enough diversity so demos are visually distinct, but all demos
# must still arc OVER C0 through the C0–C3 channel (centre y≈0.091).
# Channel half-width = 13mm → ±5mm Y-shift keeps all demos safely inside.
NOISE_POS  = 0.004     # ±4 mm intermediate waypoint noise (visibly different paths)
NOISE_START = 0.003    # ±3 mm start offset XY (Z fixed)

# Needle geometry (must match view_scene.py / config)
NEEDLE_OFFSET_Z = 0.1034 + 0.080   # 0.1834 m from link7 local +Z to needle centre
NEEDLE_HALF_LEN = 0.080             # 8 cm
EE_ABOVE_TIP    = 2 * NEEDLE_HALF_LEN   # 0.16 m: attachment_site is this ABOVE tip


# ── MuJoCo helpers ─────────────────────────────────────────────────────────────

def build_model():
    """FR3 + needle capsule attached to fr3_link7."""
    spec  = mujoco.MjSpec.from_file(FR3_SCENE)
    link7 = spec.worldbody.find_child('fr3_link7')
    nb    = link7.add_body()
    nb.name = 'needle'; nb.pos = [0.0, 0.0, NEEDLE_OFFSET_Z]
    ng    = nb.add_geom()
    ng.type = mujoco.mjtGeom.mjGEOM_CAPSULE
    ng.size[0] = 0.0035; ng.size[1] = NEEDLE_HALF_LEN
    ng.contype = 0; ng.conaffinity = 0
    return spec.compile()


def ik_6dof(model, data, target_ee, n: int = 60, alpha: float = 0.08):
    """
    6-DOF iterative IK: move attachment_site → target_ee
    while keeping tool-Z = [0, 0, -1] (needle pointing straight down).
    Warm-started from data.qpos — call after setting initial config.
    """
    sid    = model.site('attachment_site').id
    tgt_z  = np.array([0.0, 0.0, -1.0])
    for _ in range(n):
        mujoco.mj_forward(model, data)
        p_err = target_ee - data.site_xpos[sid]
        mat   = data.site_xmat[sid].reshape(3, 3)
        o_err = np.cross(mat[:, 2], tgt_z) * 2.0
        if np.linalg.norm(p_err) < 8e-5 and np.linalg.norm(o_err) < 0.008:
            break
        Jp = np.zeros((3, model.nv)); Jr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, Jp, Jr, sid)
        J6 = np.vstack([Jp[:, :7], Jr[:, :7]])
        dq = J6.T @ np.linalg.solve(J6 @ J6.T + 0.05 * np.eye(6),
                                     np.concatenate([p_err, o_err]))
        dq = np.clip(dq * alpha, -0.06, 0.06)
        data.qpos[:7] += dq
        for j in range(7):
            if model.jnt_limited[j]:
                data.qpos[j] = np.clip(data.qpos[j],
                                       model.jnt_range[j, 0],
                                       model.jnt_range[j, 1])
    mujoco.mj_forward(model, data)


def get_needle_tip(data, model) -> np.ndarray:
    """
    Needle tip world position.
    tip = attachment_site_pos + mat[:,2] * EE_ABOVE_TIP
    (mat[:,2] = tool-Z ≈ [0,0,-1] → tip is 0.16 m BELOW attachment_site)
    """
    sid = model.site('attachment_site').id
    mat = data.site_xmat[sid].reshape(3, 3)
    ee  = data.site_xpos[sid]
    return ee + mat[:, 2] * EE_ABOVE_TIP


def find_start_config(model, data, x_start: np.ndarray) -> np.ndarray:
    """IK from home pose to place needle tip at x_start."""
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    target_ee = x_start + np.array([0.0, 0.0, EE_ABOVE_TIP])
    ik_6dof(model, data, target_ee, n=800, alpha=0.04)
    return data.qpos[:7].copy()


# ── Demo generation ────────────────────────────────────────────────────────────

def generate_demo(model, data,
                  qpos_start: np.ndarray,
                  seed: int = 0) -> np.ndarray:
    """
    Generate one demonstration trajectory.

    Returns array of shape (T, 11):
      [time, q1..q7, x_tip, y_tip, z_tip]
    """
    rng = np.random.default_rng(seed)

    # Per-demo Y-channel bias: shift the whole arc up/down inside the channel.
    # C0 is at y=0.050 (+CRITICAL_RADIUS+MARGIN=0.028 → buffer top at 0.078)
    # C3 is at y=0.132 (-CRITICAL_RADIUS-MARGIN=0.028 → buffer bottom at 0.104)
    # Channel safe band: [0.079, 0.103] → centre 0.091, half-width ~12mm.
    # Bias ±4mm keeps all demos in [0.083, 0.099] — fully within the safe band.
    channel_bias_y = rng.uniform(-0.004, 0.004)

    # Perturb waypoints (start ±NOISE_START XY, middle ±NOISE_POS XY, goal fixed)
    wpts = DEMO_WAYPOINTS.astype(np.float64).copy()
    # Apply channel bias to Y of ALL middle waypoints (not start/goal)
    wpts[1:-1, 1] += channel_bias_y

    # Then add per-waypoint noise for local path variation
    start_offset = rng.uniform(-NOISE_START, NOISE_START, 3)
    start_offset[2] = 0.0                           # keep Z fixed
    wpts[0] += start_offset
    mid_noise = rng.standard_normal(wpts[1:-1].shape) * NOISE_POS
    mid_noise[:, 2] = 0.0                           # keep Z fixed for middle too
    wpts[1:-1] += mid_noise
    wpts[-1]  = X_GOAL.astype(np.float64)     # goal always fixed

    # Cubic spline of needle-tip positions
    t_wpts  = np.linspace(0.0, T_TOTAL, len(wpts))
    t_dense = np.arange(0.0, T_TOTAL, DT)
    cs      = CubicSpline(t_wpts, wpts, bc_type='clamped')
    tip_pos = cs(t_dense)          # (T, 3)  needle-tip targets

    # Reset to start config
    data.qpos[:7] = qpos_start
    mujoco.mj_forward(model, data)

    rows = []
    for i, t in enumerate(t_dense):
        ee_target = tip_pos[i] + np.array([0.0, 0.0, EE_ABOVE_TIP])

        # More iterations for first step; warm-started for the rest
        ik_6dof(model, data, ee_target,
                n=(120 if i == 0 else 30),
                alpha=0.10)

        tip_actual = get_needle_tip(data, model)
        row = ([t]
               + list(data.qpos[:7])
               + list(tip_actual))
        rows.append(row)

    return np.array(rows, dtype=np.float64)   # (T, 11)


# ── Main ───────────────────────────────────────────────────────────────────────

def collect_all_demos():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(NPY_DIR,  exist_ok=True)

    print("[Demos] Building MuJoCo model with needle...")
    model = build_model()
    data  = mujoco.MjData(model)

    print("[Demos] Finding start configuration via IK...")
    qpos_start = find_start_config(model, data, X_START)
    tip_check  = get_needle_tip(data, model)
    print(f"        Needle tip at start: {np.round(tip_check, 4)}"
          f"  (target: {X_START})")
    ik_err = np.linalg.norm(tip_check - X_START) * 1000
    print(f"        IK error: {ik_err:.1f} mm")

    all_pos, all_vel, all_sdf, all_pen = [], [], [], []
    all_crit, all_crit_buf = [], []

    print(f"\n[Demos] Generating {N_DEMOS} demonstrations @ {int(1/DT)} Hz, "
          f"{T_TOTAL}s each...")

    for i in range(N_DEMOS):
        traj = generate_demo(model, data, qpos_start=qpos_start, seed=i)

        # Save CSV (no header — identical format to node_clf_cbf)
        csv_path = f"{DATA_DIR}/demo_{i+1}.csv"
        np.savetxt(csv_path, traj, delimiter=',', fmt='%.10f')

        # Extract EE positions and compute velocities via finite differences
        pos = traj[:, 8:11].astype(np.float32)     # (T, 3)  needle-tip x,y,z
        vel = np.gradient(pos, DT, axis=0).astype(np.float32)  # (T, 3)

        sdf_v      = sdf_all_np(pos).astype(np.float32)
        pen_v      = penetration_depth_np(pos).astype(np.float32)
        crit_v     = is_in_critical_np(pos).astype(np.float32)        # solid sphere
        crit_buf_v = is_in_critical_buffer_np(pos).astype(np.float32) # + safety margin

        all_pos.append(pos)
        all_vel.append(vel)
        all_sdf.append(sdf_v)
        all_pen.append(pen_v)
        all_crit.append(crit_v)
        all_crit_buf.append(crit_buf_v)

        n_pen  = (pen_v > 1e-4).sum()
        n_crit = crit_v.sum()
        n_buf  = crit_buf_v.sum()
        print(f"  demo_{i+1:02d}.csv : {len(traj)} rows | "
              f"pen={n_pen} ({100*n_pen/len(traj):.0f}%) | "
              f"in_crit={int(n_crit)} | in_buffer={int(n_buf)} | "
              f"max_pen={pen_v.max()*1000:.1f}mm | "
              f"tip_z∈[{pos[:,2].min():.3f},{pos[:,2].max():.3f}]")

    # Combine and save .npy for fast training load
    positions  = np.vstack(all_pos)
    velocities = np.vstack(all_vel)
    sdf_vals   = np.hstack(all_sdf)
    pen_depth  = np.hstack(all_pen)
    in_crit    = np.hstack(all_crit)
    in_crit_buf = np.hstack(all_crit_buf)

    np.save(f"{NPY_DIR}/all_positions.npy",   positions)
    np.save(f"{NPY_DIR}/all_velocities.npy",  velocities)
    np.save(f"{NPY_DIR}/all_sdf.npy",         sdf_vals)
    np.save(f"{NPY_DIR}/all_pen_depth.npy",   pen_depth)
    np.save(f"{NPY_DIR}/all_in_critical.npy", in_crit)
    np.save(f"{NPY_DIR}/all_in_critical_buffer.npy", in_crit_buf)

    print(f"\n[Demos] {len(positions)} total states")
    print(f"        CSVs:  {DATA_DIR}/demo_{{1..{N_DEMOS}}}.csv")
    print(f"        NPYs:  {NPY_DIR}/all_*.npy  (for fast training load)")
    print(f"        Penetrating      : {(pen_depth>1e-4).sum()} "
          f"({100*(pen_depth>1e-4).mean():.1f}%)")
    print(f"        In critical (solid)  : {int(in_crit.sum())} "
          f"({100*in_crit.mean():.2f}%)")
    print(f"        In critical (+buffer): {int(in_crit_buf.sum())} "
          f"({100*in_crit_buf.mean():.2f}%)")
    print(f"        Max pen          : {pen_depth.max()*1000:.2f} mm")
    return positions, velocities, sdf_vals, pen_depth, in_crit, in_crit_buf


# ── Also expose a CSV-based loader for train.py ────────────────────────────────

def load_demos_from_csv(data_dir: str = DATA_DIR):
    """
    Load all CSVs from data_dir.
    Returns positions, velocities, sdf, pen_depth, in_critical, in_critical_buffer
    (all length-N arrays; positions/velocities are (N,3)).
    Velocities computed by finite differences on EE positions (cols 8-10).
    """
    import glob
    files = sorted(glob.glob(f"{data_dir}/demo_*.csv"))
    if not files:
        raise FileNotFoundError(f"No demo CSVs found in {data_dir}")

    all_pos, all_vel, all_sdf, all_pen = [], [], [], []
    all_crit, all_crit_buf = [], []
    for f in files:
        d   = np.loadtxt(f, delimiter=',', dtype=np.float32)
        pos = d[:, 8:11]                                    # (T, 3)
        vel = np.gradient(pos, DT, axis=0)                  # (T, 3)
        all_pos.append(pos)
        all_vel.append(vel)
        all_sdf.append(sdf_all_np(pos).astype(np.float32))
        all_pen.append(penetration_depth_np(pos).astype(np.float32))
        all_crit.append(is_in_critical_np(pos).astype(np.float32))
        all_crit_buf.append(is_in_critical_buffer_np(pos).astype(np.float32))

    return (np.vstack(all_pos), np.vstack(all_vel),
            np.hstack(all_sdf), np.hstack(all_pen),
            np.hstack(all_crit), np.hstack(all_crit_buf))


if __name__ == "__main__":
    collect_all_demos()
