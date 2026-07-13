"""
view_scene.py — MuJoCo viewer for the TA-CBF surgical scene.

Scene (top-down camera to see 2D trajectory clearly):
  Foam            : ORIGINAL orange 3D spheres (two Z layers, unchanged)
  Critical zones  : NON-LINEAR red shapes in XY (star, crescent, blob, kidney, L-shape)
                    approximated using overlapping MuJoCo primitives
  Needle          : silver capsule on fr3_link7, tip at X_START
  White dots      : expert demo path waypoints
  Blue sphere     : start position
  Green sphere    : goal (tumour target)
  Gray box        : operating tray

Camera set near-top-down so the 2D trajectory in XY is clearly visible,
while still showing the 3D foam spheres for depth.

Usage:
    cd /home/stanny/franka_ros2_ws/src/Tolerance_Aware_CBF(TA-CBF)
    source venv/bin/activate
    python src/view_scene.py
"""

import os
import sys
import numpy as np
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from config import (
    DEMO_WAYPOINTS, CRITICAL_SHAPES, FOAM_CENTRES, FOAM_RADIUS,
    X_START, X_GOAL, Z_CORRIDOR, Z_BOTTOM, SLAB_X, SLAB_Y,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE     = os.path.dirname(os.path.realpath(__file__))
FR3_SCENE = os.path.realpath(
    os.path.join(_HERE, "../../node_clf_cbf/assets/mujoco_menagerie/franka_fr3/scene.xml"))

# ── Needle geometry ───────────────────────────────────────────────────────────
NEEDLE_RADIUS   = 0.0035
NEEDLE_HALF_LEN = 0.080
NEEDLE_OFFSET_Z = 0.1034 + NEEDLE_HALF_LEN
EE_ABOVE_TIP    = 2 * NEEDLE_HALF_LEN   # 0.160 m

Z_COR = float(Z_CORRIDOR)

# Tray
TRAY_CENTER = np.array([
    (SLAB_X[0] + SLAB_X[1]) / 2,
    (SLAB_Y[0] + SLAB_Y[1]) / 2,
    float(Z_BOTTOM) - FOAM_RADIUS - 0.004,
])
TRAY_HALF = np.array([
    (SLAB_X[1] - SLAB_X[0]) / 2 + 0.010,
    (SLAB_Y[1] - SLAB_Y[0]) / 2 + 0.010,
    0.004,
])

RED  = [0.88, 0.06, 0.06, 0.92]


# ── MuJoCo primitive helpers ──────────────────────────────────────────────────

def _sphere(wb, name, x, y, z, r, rgba):
    b = wb.add_body(); b.name = name; b.pos = [x, y, z]
    g = b.add_geom(); g.type = mujoco.mjtGeom.mjGEOM_SPHERE
    g.size[0] = r; g.rgba[:] = rgba
    g.contype = 0; g.conaffinity = 0


def _cylinder(wb, name, x, y, z, r, half_h, rgba):
    """Upright cylinder — looks like a disk from top-down."""
    b = wb.add_body(); b.name = name; b.pos = [x, y, z]
    g = b.add_geom(); g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    g.size[0] = r; g.size[1] = half_h
    g.rgba[:] = rgba; g.contype = 0; g.conaffinity = 0


def _capsule_fromto(wb, name, x1, y1, z1, x2, y2, z2, r, rgba):
    """Capsule defined by two endpoints — works in any direction."""
    b = wb.add_body(); b.name = name; b.pos = [0, 0, 0]
    g = b.add_geom(); g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
    g.fromto[0] = x1; g.fromto[1] = y1; g.fromto[2] = z1
    g.fromto[3] = x2; g.fromto[4] = y2; g.fromto[5] = z2
    g.size[0] = r; g.rgba[:] = rgba
    g.contype = 0; g.conaffinity = 0


# ── Non-linear critical shape builders ───────────────────────────────────────

def _build_star(wb, shape, z):
    c = shape['center'][:2]; outer = shape['outer_r']
    inner = shape['inner_r']; n = shape['n_points']
    rot = shape['rotation']; label = shape['label']
    th = 0.004  # half-thickness

    # Central disk
    _cylinder(wb, f"{label}_core", c[0], c[1], z, inner * 0.92, th, RED)

    # N pointed spokes
    for k in range(n):
        angle = rot + k * (2 * np.pi / n)
        mx = c[0] + (inner + outer) / 2 * np.cos(angle)
        my = c[1] + (inner + outer) / 2 * np.sin(angle)
        half_len = (outer - inner) / 2
        x1 = mx - half_len * np.cos(angle)
        y1 = my - half_len * np.sin(angle)
        x2 = mx + half_len * np.cos(angle)
        y2 = my + half_len * np.sin(angle)
        _capsule_fromto(wb, f"{label}_pt{k}",
                        x1, y1, z, x2, y2, z, inner * 0.52, RED)


def _build_crescent(wb, shape, z):
    c = shape['center'][:2]; outer = shape['outer_r']
    inner = shape['inner_r']; offset = shape['offset']; label = shape['label']
    inner_c = c + offset
    th = 0.004

    # Arc of small spheres along outer circle, excluding inner region
    n_arc = 22
    r_dot = outer * 0.17
    for k in range(n_arc):
        angle = k * 2 * np.pi / n_arc
        px = c[0] + outer * 0.83 * np.cos(angle)
        py = c[1] + outer * 0.83 * np.sin(angle)
        if np.sqrt((px - inner_c[0])**2 + (py - inner_c[1])**2) > inner * 1.05:
            _cylinder(wb, f"{label}_a{k}", px, py, z, r_dot, th, RED)


def _build_blob(wb, shape, z):
    c = shape['center'][:2]; radii = shape['radii']
    offsets = shape['offsets']; label = shape['label']
    th = 0.004

    for i, (r, off) in enumerate(zip(radii, offsets)):
        _cylinder(wb, f"{label}_s{i}",
                  float(c[0] + off[0]), float(c[1] + off[1]), z,
                  float(r) * 0.90, th, RED)


def _build_kidney(wb, shape, z):
    c = shape['center'][:2]; outer = shape['outer_r']
    inner = shape['inner_r']; sq = shape['squeeze']
    rot = shape['rotation']; label = shape['label']
    th = 0.004

    n_arc = 18
    r_dot = outer * 0.19
    for k in range(n_arc):
        angle = rot + k * 2 * np.pi / n_arc
        rx = outer * np.cos(angle)
        ry = outer * sq * np.sin(angle)
        notch_d = np.sqrt((rx - outer * 0.4)**2 + ry**2)
        if notch_d > inner * 1.1:
            _cylinder(wb, f"{label}_k{k}", c[0] + rx, c[1] + ry, z, r_dot, th, RED)


def _build_lshape(wb, shape, z):
    c = shape['center'][:2]; arm1 = shape['arm1']
    arm2 = shape['arm2']; rot = float(shape['rotation']); label = shape['label']
    hw1, hh1 = arm1[0] / 2, arm1[1] / 2
    hw2, hh2 = arm2[0] / 2, arm2[1] / 2
    th = 0.004

    # Arm 1 — horizontal (angle = rot + 90°)
    a1 = rot + np.pi / 2
    x1 = c[0] - hw1 * np.cos(a1); y1 = c[1] - hw1 * np.sin(a1) - hh2
    x2 = c[0] + hw1 * np.cos(a1); y2 = c[1] + hw1 * np.sin(a1) - hh2
    _capsule_fromto(wb, f"{label}_a1", x1, y1, z, x2, y2, z, hh1, RED)

    # Arm 2 — vertical (angle = rot)
    a2 = rot
    bx = c[0] - hw1 + hw2
    by = c[1]
    x1 = bx - hh2 * np.cos(a2); y1 = by - hh2 * np.sin(a2) + hh1
    x2 = bx + hh2 * np.cos(a2); y2 = by + hh2 * np.sin(a2) + hh1
    _capsule_fromto(wb, f"{label}_a2", x1, y1, z, x2, y2, z, hw2, RED)


def _add_critical_shapes(wb):
    dispatch = {
        'star':     _build_star,
        'crescent': _build_crescent,
        'blob':     _build_blob,
        'kidney':   _build_kidney,
        'lshape':   _build_lshape,
    }
    for shape in CRITICAL_SHAPES:
        dispatch[shape['type']](wb, shape, Z_COR + 0.002)
    print(f"[Scene] Critical shapes: {len(CRITICAL_SHAPES)} non-linear zones")


# ── IK ────────────────────────────────────────────────────────────────────────

def run_ik(model, data, target_ee, n=800, alpha=0.04):
    sid   = model.site('attachment_site').id
    tgt_z = np.array([0.0, 0.0, -1.0])
    p_err = np.ones(3)
    for i in range(n):
        mujoco.mj_forward(model, data)
        p_err = target_ee - data.site_xpos[sid]
        mat   = data.site_xmat[sid].reshape(3, 3)
        o_err = np.cross(mat[:, 2], tgt_z) * 2.0
        if np.linalg.norm(p_err) < 4e-4 and np.linalg.norm(o_err) < 0.015:
            break
        Jp = np.zeros((3, model.nv)); Jr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, Jp, Jr, sid)
        J6   = np.vstack([Jp[:, :7], Jr[:, :7]])
        err6 = np.concatenate([p_err, o_err])
        dq   = J6.T @ np.linalg.solve(J6 @ J6.T + 0.05 * np.eye(6), err6)
        data.qpos[:7] += np.clip(dq * alpha, -0.06, 0.06)
        for j in range(7):
            if model.jnt_limited[j]:
                data.qpos[j] = np.clip(data.qpos[j],
                                       model.jnt_range[j, 0],
                                       model.jnt_range[j, 1])
    mujoco.mj_forward(model, data)
    return i + 1, float(np.linalg.norm(p_err))


# ── Scene builder ─────────────────────────────────────────────────────────────

def build_model() -> mujoco.MjModel:
    if not os.path.exists(FR3_SCENE):
        raise FileNotFoundError(
            f"FR3 scene not found:\n  {FR3_SCENE}\n"
            "Clone mujoco_menagerie into src/node_clf_cbf/assets/ first.")

    spec = mujoco.MjSpec.from_file(FR3_SCENE)
    wb   = spec.worldbody

    # 1. Needle on fr3_link7
    link7 = spec.worldbody.find_child('fr3_link7')
    nb = link7.add_body(); nb.name = 'needle'; nb.pos = [0, 0, NEEDLE_OFFSET_Z]
    ng = nb.add_geom()
    ng.name = 'needle_geom'; ng.type = mujoco.mjtGeom.mjGEOM_CAPSULE
    ng.size[0] = NEEDLE_RADIUS; ng.size[1] = NEEDLE_HALF_LEN
    ng.rgba[:] = [0.82, 0.84, 0.92, 0.97]
    ng.contype = 0; ng.conaffinity = 0

    # 2. Operating tray
    tb = wb.add_body(); tb.name = 'tray'; tb.pos = TRAY_CENTER.tolist()
    tg = tb.add_geom(); tg.name = 'tray_geom'
    tg.type = mujoco.mjtGeom.mjGEOM_BOX; tg.size[:] = TRAY_HALF.tolist()
    tg.rgba[:] = [0.72, 0.74, 0.78, 1.0]
    tg.contype = 0; tg.conaffinity = 0

    # 3. Foam — ORIGINAL 3D spheres (two Z layers, orange semi-transparent)
    print(f"[Scene] Foam spheres: {len(FOAM_CENTRES)}")
    for i, c in enumerate(FOAM_CENTRES):
        _sphere(wb, f"foam_{i:04d}",
                float(c[0]), float(c[1]), float(c[2]),
                FOAM_RADIUS,
                [0.95, 0.72, 0.22, 0.43])

    # 4. Critical zones — non-linear shapes (red, above foam)
    _add_critical_shapes(wb)

    # 5. Demo waypoints — small white spheres at corridor Z
    for i, wp in enumerate(DEMO_WAYPOINTS):
        _sphere(wb, f"wp_{i}",
                float(wp[0]), float(wp[1]), float(wp[2]) + 0.004,
                0.005, [1.0, 1.0, 1.0, 0.95])

    # 6. Start marker — blue sphere
    _sphere(wb, 'start_marker',
            float(X_START[0]), float(X_START[1]), float(X_START[2]) + 0.006,
            0.012, [0.10, 0.25, 0.95, 1.0])

    # 7. Goal marker — green sphere
    _sphere(wb, 'goal_marker',
            float(X_GOAL[0]), float(X_GOAL[1]), float(X_GOAL[2]) + 0.006,
            0.014, [0.05, 0.92, 0.15, 1.0])

    model = spec.compile()
    print(f"[Scene] Total bodies: {model.nbody}  |  geoms: {model.ngeom}")
    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  TA-CBF Surgical Scene — Foam Spheres + Non-Linear Critical Zones")
    print("=" * 65)
    print()
    print("  ORANGE spheres   : foam tissue (3D, original) — needle can penetrate")
    print("  RED shapes       : critical zones (non-linear XY cross-section):")
    print("                     star=blocker · crescent=vessel · blob=nerve")
    print("                     kidney=artery · blob=vein · L-shape=structure")
    print("  White dots       : expert demo waypoints (trajectory path)")
    print("  Blue sphere      : start position")
    print("  Green sphere     : goal (tumour target)")
    print()
    print("  Camera: near top-down to show 2D XY trajectory clearly.")
    print("  Scroll=zoom  Ctrl+drag=pan  Alt+drag=rotate to see 3D depth")
    print("=" * 65)

    model = build_model()
    data  = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)

    target_ee = X_START.copy() + np.array([0.0, 0.0, EE_ABOVE_TIP])
    print("\n[IK] Placing needle tip at start…")
    iters, perr = run_ik(model, data, target_ee)
    print(f"     Converged in {iters} iters, pos_err = {perr*1000:.2f} mm")

    sid = model.site('attachment_site').id
    mat = data.site_xmat[sid].reshape(3, 3)
    ee  = data.site_xpos[sid]
    tip = ee + mat[:, 2] * EE_ABOVE_TIP
    print(f"     Needle tip: {np.round(tip, 4)}  (target: {X_START.round(4)})")
    print()

    # Hold the IK-solved joint configuration via position actuators
    # so the arm doesn't sag/drift away from the start pose under gravity.
    if model.nu >= 7:
        data.ctrl[:7] = data.qpos[:7]

    with mujoco.viewer.launch_passive(model, data) as v:
        # Near-top-down camera — trajectory in XY plane is clearly visible,
        # 3D foam spheres visible with slight angle for depth
        v.cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
        v.cam.distance  = 0.55
        v.cam.azimuth   = 90.0
        v.cam.elevation = -80.0    # ~80° top-down (slight angle for 3D depth)
        v.cam.lookat[:] = [
            (SLAB_X[0] + SLAB_X[1]) / 2,
            (SLAB_Y[0] + SLAB_Y[1]) / 2,
            Z_COR,
        ]
        print("[Viewer] Running — close the window to exit.")
        while v.is_running():
            mujoco.mj_step(model, data)
            v.sync()


if __name__ == "__main__":
    main()
