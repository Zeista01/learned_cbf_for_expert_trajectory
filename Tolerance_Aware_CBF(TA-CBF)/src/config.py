"""
config.py — all hyperparameters and scene geometry for the TA-CBF project.

Scene summary
-------------
  Horizontal tissue slab (X-Y plane at Z = 0.44 m) on an operating tray.
  ~188 orange foam SPHERES fill the entire slab (deformable tissue) — unchanged.
  5 red critical zones have NON-LINEAR cross-sections in XY (star, crescent,
  blob, kidney, L-shape) representing irregular blood vessels / nerves.
  Needle navigates from X_START to X_GOAL through the solid foam, avoiding
  critical zones strictly while foam allows bounded penetration.

Architecture changes (mentor feedback):
  1. ProgressConditionedDS: whole demo path as attractor (not just goal)
  2. Quadratic CLF V(e) = ||e||^2 from Nawaz et al. (replaces ICNN)
  3. Fully learned BarrierNet (S2-NNDS style) — no analytic SDF, because
     critical tissue shapes are non-geometric and require learning
  4. S2-NNDS Algorithm 2: simultaneous joint training with counterexample
     refinement of f_θ, V_θ', B_φ
"""

import copy
import numpy as np

# ─── Scene geometry ────────────────────────────────────────────────────────────
Z_CORRIDOR  = 0.44          # height of the needle corridor (m)
Z_BOTTOM    = 0.40          # lower foam layer
FOAM_Z_LEVELS = [Z_BOTTOM, Z_CORRIDOR]

STATE_DIM = 3               # 3D state (x, y, z) — trajectory is 2D in XY at fixed Z

X_START = np.array([0.440, 0.062, Z_CORRIDOR], dtype=np.float32)
X_GOAL  = np.array([0.548, 0.063, Z_CORRIDOR], dtype=np.float32)

# Expert demo waypoints — unchanged from original
# Path threads between obstacles, arcing ABOVE C0 through C0–C3 channel
DEMO_WAYPOINTS = np.array([
    [0.440, 0.062, Z_CORRIDOR],
    [0.452, 0.080, Z_CORRIDOR],
    [0.462, 0.088, Z_CORRIDOR],
    [0.475, 0.091, Z_CORRIDOR],
    [0.488, 0.091, Z_CORRIDOR],
    [0.500, 0.091, Z_CORRIDOR],
    [0.512, 0.091, Z_CORRIDOR],
    [0.525, 0.088, Z_CORRIDOR],
    [0.536, 0.075, Z_CORRIDOR],
    [0.548, 0.063, Z_CORRIDOR],
], dtype=np.float32)

# ─── Critical tissue zones — NON-LINEAR shapes in XY ─────────────────────────
# The XY cross-section of each critical zone is non-circular.
# Positions match the original critical sphere centres.
# The barrier B_φ is LEARNED from scratch to fit these irregular shapes.
CRITICAL_SHAPES = [
    {
        'type':    'star',
        'center':  np.array([0.500, 0.050, Z_CORRIDOR], dtype=np.float32),  # C0 BLOCKER
        'outer_r': 0.022,
        'inner_r': 0.011,
        'n_points': 5,
        'rotation': 0.3,
        'label':   'blocker',
    },
    {
        'type':    'crescent',
        'center':  np.array([0.448, 0.120, Z_CORRIDOR], dtype=np.float32),  # C1 vessel
        'outer_r': 0.020,
        'inner_r': 0.013,
        'offset':  np.array([0.009, 0.000], dtype=np.float32),
        'label':   'vessel',
    },
    {
        'type':    'blob',
        'center':  np.array([0.470, -0.005, Z_CORRIDOR], dtype=np.float32),  # C2 nerve
        'radii':   np.array([0.016, 0.011, 0.010], dtype=np.float32),
        'offsets': np.array([
            [0.000,  0.000],
            [0.012,  0.004],
            [-0.007, 0.009],
        ], dtype=np.float32),
        'label':   'nerve',
    },
    {
        'type':    'kidney',
        'center':  np.array([0.500, 0.132, Z_CORRIDOR], dtype=np.float32),  # C3 artery
        'outer_r': 0.019,
        'inner_r': 0.011,
        'squeeze': 0.60,
        'rotation': 0.8,
        'label':   'artery',
    },
    {
        'type':    'blob',
        'center':  np.array([0.520, -0.005, Z_CORRIDOR], dtype=np.float32),  # C4 vein
        'radii':   np.array([0.014, 0.010], dtype=np.float32),
        'offsets': np.array([
            [0.000, 0.000],
            [0.010, 0.006],
        ], dtype=np.float32),
        'label':   'vein',
    },
    {
        'type':    'lshape',
        'center':  np.array([0.545, 0.120, Z_CORRIDOR], dtype=np.float32),  # C5 structure
        'arm1':    np.array([0.028, 0.010], dtype=np.float32),
        'arm2':    np.array([0.010, 0.028], dtype=np.float32),
        'rotation': -0.3,
        'label':   'structure',
    },
]
CRIT_LABELS = [s['label'] for s in CRITICAL_SHAPES]

# Effective radius for quick bounding-box exclusion (used by foam generator)
CRITICAL_RADIUS = 0.018        # bounding sphere radius (same as original)
CRITICAL_MARGIN = 0.010        # safety buffer beyond obstacle boundary

# Legacy: critical centres as (N,3) array for backward compat with old view_scene
CRITICAL_CENTRES = np.array([s['center'] for s in CRITICAL_SHAPES], dtype=np.float32)


def sdf_critical_shape_2d(pts_xy: np.ndarray, shape: dict) -> np.ndarray:
    """
    SDF in XY plane for one critical shape.
    pts_xy: (N, 2). Returns (N,). Positive = outside, negative = inside.
    """
    c2 = shape['center'][:2]
    t  = shape['type']
    if t == 'star':
        return _sdf_star(pts_xy, c2, shape['outer_r'], shape['inner_r'],
                         shape['n_points'], shape['rotation'])
    elif t == 'crescent':
        return _sdf_crescent(pts_xy, c2, shape['outer_r'], shape['inner_r'],
                             shape['offset'])
    elif t == 'blob':
        return _sdf_blob(pts_xy, c2, shape['radii'], shape['offsets'])
    elif t == 'kidney':
        return _sdf_kidney(pts_xy, c2, shape['outer_r'], shape['inner_r'],
                           shape['squeeze'], shape['rotation'])
    elif t == 'lshape':
        return _sdf_lshape(pts_xy, c2, shape['arm1'], shape['arm2'],
                           shape['rotation'])
    raise ValueError(f"Unknown shape: {t}")


def sdf_all_critical_np(pts: np.ndarray) -> np.ndarray:
    """
    SDF to union of all critical shapes (min over shapes).
    pts: (N, 2) or (N, 3) — only XY used. Returns (N,).
    Positive = outside all critical zones, negative = inside.
    """
    if pts.ndim == 1:
        pts = pts[None, :]
    pts_xy = pts[:, :2].astype(np.float32)
    sdfs = [sdf_critical_shape_2d(pts_xy, sh) for sh in CRITICAL_SHAPES]
    return np.min(np.stack(sdfs, axis=1), axis=1)


def is_in_critical_np(pts: np.ndarray) -> np.ndarray:
    return sdf_all_critical_np(pts) < 0


def is_in_critical_buffer_np(pts: np.ndarray) -> np.ndarray:
    return sdf_all_critical_np(pts) < CRITICAL_MARGIN


# ── Shape SDF implementations ─────────────────────────────────────────────────

def _sdf_star(pts, center, outer_r, inner_r, n_points, rotation):
    p = pts - center
    c, s = np.cos(-rotation), np.sin(-rotation)
    px = c * p[:, 0] - s * p[:, 1]
    py = s * p[:, 0] + c * p[:, 1]
    angle = np.arctan2(py, px)
    r = np.sqrt(px**2 + py**2 + 1e-10)
    sector = np.pi / n_points
    sector_angle = np.mod(angle, 2 * sector) - sector
    star_r = inner_r + (outer_r - inner_r) * (1 - np.abs(sector_angle) / sector)
    return r - star_r


def _sdf_crescent(pts, center, outer_r, inner_r, offset):
    p = pts - center
    r_outer = np.sqrt((p**2).sum(axis=1))
    p_inner = pts - (center + offset)
    r_inner = np.sqrt((p_inner**2).sum(axis=1))
    return np.maximum(r_outer - outer_r, inner_r - r_inner)


def _sdf_blob(pts, center, radii, offsets):
    sdfs = []
    for r, off in zip(radii, offsets):
        c = center + off
        sdfs.append(np.sqrt(((pts - c)**2).sum(axis=1)) - r)
    return np.min(np.stack(sdfs, axis=1), axis=1)


def _sdf_kidney(pts, center, outer_r, inner_r, squeeze, rotation):
    p = pts - center
    c_r, s_r = np.cos(-rotation), np.sin(-rotation)
    px = c_r * p[:, 0] - s_r * p[:, 1]
    py = s_r * p[:, 0] + c_r * p[:, 1]
    r_outer = np.sqrt(px**2 + (py / squeeze)**2 + 1e-10) - outer_r
    r_inner = inner_r - np.sqrt((px - outer_r * 0.4)**2 + py**2 + 1e-10)
    return np.maximum(r_outer, r_inner)


def _sdf_lshape(pts, center, arm1, arm2, rotation):
    p = pts - center
    c_r, s_r = np.cos(-rotation), np.sin(-rotation)
    px = c_r * p[:, 0] - s_r * p[:, 1]
    py = s_r * p[:, 0] + c_r * p[:, 1]

    def rect_sdf(qx, qy, hw, hh):
        dx = np.abs(qx) - hw
        dy = np.abs(qy) - hh
        return (np.sqrt(np.maximum(dx, 0)**2 + np.maximum(dy, 0)**2)
                + np.minimum(np.maximum(dx, dy), 0))

    hw1, hh1 = arm1[0] / 2, arm1[1] / 2
    hw2, hh2 = arm2[0] / 2, arm2[1] / 2
    s1 = rect_sdf(px, py + hh2, hw1, hh1)
    s2 = rect_sdf(px + hw1 - hw2, py - hh1, hw2, hh2)
    return np.minimum(s1, s2)


# ─── Foam tissue  (deformable — ORIGINAL 3D spheres, unchanged) ───────────────
FOAM_RADIUS     = 0.018
FOAM_SPACING_XY = 0.018
FOAM_JITTER_XY  = 0.003
FOAM_Z_JITTER   = 0.003
FOAM_SEED       = 13
SLAB_X          = (0.37, 0.60)
SLAB_Y          = (-0.025, 0.165)


def generate_foam_centres() -> np.ndarray:
    """Return (N, 3) array of foam sphere centres — identical to original."""
    rng = np.random.default_rng(FOAM_SEED)
    xs  = np.arange(SLAB_X[0], SLAB_X[1], FOAM_SPACING_XY)
    ys  = np.arange(SLAB_Y[0], SLAB_Y[1], FOAM_SPACING_XY)
    out = []
    for x in xs:
        for y in ys:
            jx, jy = rng.uniform(-FOAM_JITTER_XY, FOAM_JITTER_XY, 2)
            px, py = x + jx, y + jy
            if not (SLAB_X[0] < px < SLAB_X[1] and SLAB_Y[0] < py < SLAB_Y[1]):
                continue
            # Keep foam centres outside the bounding sphere of each critical shape
            if any(np.linalg.norm([px - c[0], py - c[1]]) < CRITICAL_RADIUS
                   for c in CRITICAL_CENTRES):
                continue
            if (np.linalg.norm([px - X_START[0], py - X_START[1]]) < 0.020 or
                    np.linalg.norm([px - X_GOAL[0],  py - X_GOAL[1]])  < 0.020):
                continue
            for z_base in FOAM_Z_LEVELS:
                jz = rng.uniform(-FOAM_Z_JITTER, FOAM_Z_JITTER)
                out.append([px, py, float(z_base) + jz])
    return np.array(out, dtype=np.float32)


FOAM_CENTRES = generate_foam_centres()   # shape (N_foam, 3)

# ─── S2-NNDS training set helpers ────────────────────────────────────────────

def sample_safe_set(n: int = 2000, seed: int = 0) -> np.ndarray:
    """Sample safe points (near demo path + general free space). Returns (N, 3)."""
    rng = np.random.default_rng(seed)
    n_path = n // 2
    n_free = n - n_path

    pts_path = []
    for _ in range(n_path):
        t = rng.uniform(0, 1)
        idx_f = t * (len(DEMO_WAYPOINTS) - 1)
        lo = int(idx_f); hi = min(lo + 1, len(DEMO_WAYPOINTS) - 1)
        alpha = idx_f - lo
        base = DEMO_WAYPOINTS[lo] + alpha * (DEMO_WAYPOINTS[hi] - DEMO_WAYPOINTS[lo])
        noise = np.array([rng.uniform(-0.015, 0.015),
                          rng.uniform(-0.015, 0.015),
                          0.0], dtype=np.float32)
        pts_path.append(base + noise)
    pts_path = np.array(pts_path, dtype=np.float32)

    pts_free = []
    while len(pts_free) < n_free:
        x = rng.uniform(SLAB_X[0], SLAB_X[1])
        y = rng.uniform(SLAB_Y[0], SLAB_Y[1])
        pt = np.array([[x, y]])
        if sdf_all_critical_np(pt)[0] > CRITICAL_MARGIN:
            pts_free.append([x, y, Z_CORRIDOR])
    return np.concatenate([pts_path,
                           np.array(pts_free, dtype=np.float32)], axis=0)


def sample_unsafe_set(n: int = 1000, seed: int = 1) -> np.ndarray:
    """Sample points strictly inside critical shapes. Returns (N, 3)."""
    rng = np.random.default_rng(seed)
    pts = []
    n_per = n // len(CRITICAL_SHAPES)
    for shape_idx, shape in enumerate(CRITICAL_SHAPES):
        c = shape['center'][:2]
        r = CRITICAL_RADIUS * 0.7
        attempts = 0
        while len(pts) < n_per * (shape_idx + 1) and attempts < 10000:
            pt2 = c + rng.uniform(-r, r, 2).astype(np.float32)
            if sdf_all_critical_np(pt2[None])[0] < -0.002:
                pts.append([float(pt2[0]), float(pt2[1]), Z_CORRIDOR])
            attempts += 1
    return np.array(pts, dtype=np.float32) if pts else np.zeros((1, 3), dtype=np.float32)


def sample_workspace(n: int, seed: int = 2) -> np.ndarray:
    """Sample uniformly from 3D workspace. Returns (N, 3)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(SLAB_X[0], SLAB_X[1], n).astype(np.float32)
    y = rng.uniform(SLAB_Y[0], SLAB_Y[1], n).astype(np.float32)
    z = np.full(n, Z_CORRIDOR, dtype=np.float32)
    return np.stack([x, y, z], axis=1)


# ─── Per-obstacle sampling for the Composite Neural Barrier ────────────────────
# These feed the CompositeBarrier (one learned conditional CBF per obstacle,
# fused by smooth-min). The analytic per-shape SDF is used ONLY to generate
# training labels / point-clouds — never as the barrier at inference.

def sample_obstacle_cloud(shape: dict, k: int = 64, seed: int = 0) -> np.ndarray:
    """
    Point-cloud of one obstacle's interior, CENTERED at its centroid.
    Returns (k, 2) — the shape descriptor fed to ObstacleEncoder. Centering
    makes the embedding translation-invariant (position enters separately via
    the relative coordinate), so the same shape anywhere yields the same code.
    """
    rng = np.random.default_rng(seed)
    c2  = shape['center'][:2]
    pts = []
    attempts = 0
    while len(pts) < k and attempts < 20000:
        p = c2 + rng.uniform(-CRITICAL_RADIUS * 1.6, CRITICAL_RADIUS * 1.6, 2)
        if sdf_critical_shape_2d(p[None].astype(np.float32), shape)[0] < 0:
            pts.append(p - c2)          # centered
        attempts += 1
    if not pts:                          # degenerate fallback
        pts = [np.zeros(2, np.float32)]
    return np.array(pts, dtype=np.float32)


def build_obstacle_set(k: int = 64, seed: int = 0) -> list:
    """
    Returns a list (one entry per CRITICAL_SHAPES) of dicts:
        {'cloud': (k,2) centered point-cloud, 'center': (3,) world centroid}
    train.py wraps these as tensors for CompositeBarrier.
    """
    obs = []
    for i, sh in enumerate(CRITICAL_SHAPES):
        obs.append({
            'cloud':  sample_obstacle_cloud(sh, k=k, seed=seed + i),
            'center': sh['center'].astype(np.float32).copy(),
        })
    return obs


def sample_obstacle_interior(shape: dict, n: int = 256, seed: int = 1) -> np.ndarray:
    """Points strictly INSIDE one obstacle (its per-obstacle unsafe set). (N,3)."""
    rng = np.random.default_rng(seed)
    c2  = shape['center'][:2]
    pts = []
    attempts = 0
    while len(pts) < n and attempts < 50000:
        p = c2 + rng.uniform(-CRITICAL_RADIUS * 1.6, CRITICAL_RADIUS * 1.6, 2)
        if sdf_critical_shape_2d(p[None].astype(np.float32), shape)[0] < -0.001:
            pts.append([p[0], p[1], Z_CORRIDOR])
        attempts += 1
    if not pts:
        pts = [[c2[0], c2[1], Z_CORRIDOR]]
    return np.array(pts, dtype=np.float32)


# ─── Pose / scale transforms for obstacles (generalization + slow env) ─────────
# Every obstacle is parameterised by (rotation, scale, translation). The SAME
# transform is applied to (a) the centered point-cloud fed to ObstacleEncoder and
# (b) the analytic SDF used to label safe/unsafe points, so the conditional CBF
# learns to respond to the shape EMBEDDING consistently regardless of pose.
# This is what fixes the zero-shot generalization failure (embedding went OOD
# under rotation) and what lets the barrier track a slowly translating / rotating
# / growing / shrinking obstacle at inference (re-evaluate each control step).

def _rot2d(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def transform_shape(shape: dict, d_rot: float = 0.0, scale: float = 1.0,
                    d_trans=(0.0, 0.0)) -> dict:
    """
    Return a copy of `shape` rotated by d_rot (rad), uniformly scaled by `scale`
    about its centroid, and translated by d_trans (dx, dy) in metres.
    Handles every shape type in CRITICAL_SHAPES.
    """
    s = copy.deepcopy(shape)
    R = _rot2d(d_rot)

    c = np.array(shape['center'], dtype=np.float32).copy()
    c[0] += float(d_trans[0]); c[1] += float(d_trans[1])
    s['center'] = c

    # uniform scale of every length parameter
    for key in ('outer_r', 'inner_r'):
        if key in s:
            s[key] = float(s[key]) * scale
    if 'radii' in s:
        s['radii'] = np.asarray(s['radii'], dtype=np.float32) * scale
    if 'arm1' in s:
        s['arm1'] = np.asarray(s['arm1'], dtype=np.float32) * scale
    if 'arm2' in s:
        s['arm2'] = np.asarray(s['arm2'], dtype=np.float32) * scale

    # rotate + scale the orientation-bearing offset vectors
    if 'offset' in s:
        s['offset'] = (R @ (np.asarray(shape['offset'], np.float32) * scale))
    if 'offsets' in s:
        s['offsets'] = (np.asarray(shape['offsets'], np.float32) * scale) @ R.T

    # shapes that carry an explicit rotation field just accumulate the delta
    if 'rotation' in s:
        s['rotation'] = float(shape['rotation']) + float(d_rot)

    return s


def random_transform(rng, rot_range=None, scale_range=None, trans_range=None):
    """Draw a random (d_rot, scale, d_trans) within the configured ranges."""
    rr = AUG_ROT_RANGE   if rot_range   is None else rot_range
    sr = AUG_SCALE_RANGE if scale_range is None else scale_range
    tr = AUG_TRANS_RANGE if trans_range is None else trans_range
    d_rot = float(rng.uniform(*rr))
    scale = float(rng.uniform(*sr))
    d_trans = rng.uniform(-tr, tr, 2).astype(np.float32)
    return d_rot, scale, d_trans


def canonical_interior_cloud(shape: dict, k: int = 96, seed: int = 0) -> np.ndarray:
    """
    Centered interior point-cloud of `shape` in its OWN canonical frame
    (centroid at origin), sampled once. Apply a 2x2 transform R·scale to this
    to obtain the cloud/interior of any pose WITHOUT re-running rejection
    sampling in the training loop. Returns (M, 2), M <= k.
    """
    rng = np.random.default_rng(seed)
    c2  = np.array(shape['center'][:2], dtype=np.float32)
    pts = []
    attempts = 0
    while len(pts) < k and attempts < 40000:
        p = c2 + rng.uniform(-CRITICAL_RADIUS * 1.8, CRITICAL_RADIUS * 1.8, 2)
        if sdf_critical_shape_2d(p[None].astype(np.float32), shape)[0] < 0:
            pts.append(p - c2)
        attempts += 1
    if not pts:
        pts = [np.zeros(2, np.float32)]
    return np.array(pts, dtype=np.float32)


def sample_local_box(shape: dict, n: int, seed: int = 0, pad: float = 0.022):
    """
    Uniform XY points in a box around `shape`'s centroid, with the box sized to
    the shape's current scale. Returns (pts3 (n,3), sdf (n,)) where sdf is the
    raw signed distance to `shape` (negative inside). Used to draw safe / unsafe
    samples around a transformed obstacle for the conditional CBF.
    """
    rng = np.random.default_rng(seed)
    c2  = np.array(shape['center'][:2], dtype=np.float32)
    half = CRITICAL_RADIUS + pad
    xy = c2 + rng.uniform(-half, half, (n, 2)).astype(np.float32)
    sdf = sdf_critical_shape_2d(xy, shape)
    z  = np.full((n, 1), Z_CORRIDOR, dtype=np.float32)
    pts3 = np.concatenate([xy, z], axis=1)
    return pts3, sdf


# ─── Training hyperparameters ─────────────────────────────────────────────────
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

F_HIDDEN   = [128, 128, 128]
V_HIDDEN   = [64, 64]
B_HIDDEN   = [128, 128, 128, 128]
ETA_HIDDEN = [64, 64]

# S2-NNDS loss weights
LAMBDA_MSE     = 50.0
LAMBDA_V1      = 5.0
LAMBDA_V2      = 10.0
LAMBDA_B_FS    = 5.0
LAMBDA_B_OB    = 12.0   # interior (b<−δ) hinge — raised: strict avoidance needs
                       # interior accuracy ≈100%, not the ~88% a weak hinge gave
LAMBDA_B_DOT   = 0.5

# QP parameters (Nawaz et al. values)
GAMMA_CBF    = 3.0
ALPHA_CLF    = 4.0
LAMBDA_SLACK = 0.5

# S2-NNDS Algorithm 2
OUTER_ITERS  = 10
INNER_EPOCHS = 150
N_CEX        = 2000
LR           = 3e-4
LR_BARRIER   = 2e-3    # composite barrier trains fresh → needs a higher LR than
                       # the f/V fine-tuning rate (verified: ~98% safe-acc in a
                       # short run at this rate)
BATCH_SIZE   = 256
N_POINTS     = 200
DT           = 0.01
DELTA_B      = 0.01   # B must exceed +δ in safe set, fall below -δ in unsafe set
DELTA_V      = 1e-3

# ── Boundary inflation: the learned B=0 contour is trained to sit INFLATE_MARGIN
# OUTSIDE the true obstacle surface (unsafe set = interior PLUS an outward shell
# of this width). This (a) makes the learned CBF strictly larger than the
# analytic SDF — a visible safety buffer — and (b) guarantees the field starts
# diverting before the needle ever touches the red zone.
INFLATE_MARGIN = 0.010   # 10 mm outward buffer = the drawn "light-red" CBF zone.
                         # The learned B=0 contour sits at this distance OUTSIDE
                         # the true surface, so {B≥0} is the full light-red halo;
                         # the controller then holds the needle OUTSIDE it.

# ── Pose / scale augmentation ranges (per obstacle, drawn every training step).
# Full rotation + ±30% scale + small translation teaches the conditional CBF to
# be pose-invariant and size-robust → zero-shot generalization to new layouts
# and robustness to slowly evolving (growing/shrinking/moving) obstacles.
AUG_ROT_RANGE   = (-np.pi, np.pi)
AUG_SCALE_RANGE = (0.65, 1.4)
AUG_TRANS_RANGE = 0.05   # ± metres in x and y

# ── SDF-shaped barrier target. The barrier is regressed onto a scaled, inflated,
# clamped signed distance  b_target = K · clip(sdf − INFLATE, −CLAMP, +CLAMP)
# instead of only hinge-classifying its sign. This makes the learned B an
# approximate signed-distance field: SMOOTH, with a near-constant gradient
# (‖∇B‖ ≈ K) through a finite boundary layer, instead of a saturated near-step
# (the |∇B|≈290, <1 mm transition that made the CBF-QP read u_safe=0 on approach
# and let the needle pass through). A usable gradient is what gives real
# divert-around-the-obstacle avoidance.
BARRIER_SDF_K       = 4.0     # barrier slope (per metre); boundary layer ≈ 2δ/K ≈ 5 mm
# ASYMMETRIC clamp on the SDF target K·clip(sdf−INFLATE, −CLAMP_IN, +CLAMP_OUT):
#   • outside (safe): clamp tightly so B is bounded above (~+0.05) far away.
#   • inside (unsafe): clamp WIDE so the gradient PERSISTS through the whole
#     interior. A symmetric clamp flattened B deep inside (∇B→0), so once the
#     needle overshot into the interior there was no force to push it back —
#     that was the residual solid-penetration. With a wide interior clamp ∇B≈K
#     points outward everywhere inside → the QP always drives the needle out.
BARRIER_SDF_CLAMP_OUT = 0.013   # +K·0.013 ≈ +0.05 ceiling
BARRIER_SDF_CLAMP_IN  = 0.040   # interior band ≥ obstacle radius (~0.02) → ∇B persists
BARRIER_SDF_CLAMP   = BARRIER_SDF_CLAMP_OUT   # legacy alias
LAMBDA_B_REG      = 20.0    # weight on the SDF-regression term

# ── Lipschitz / eikonal regularization. SDF-value regression alone constrains B
# only AT sample points; between them the net still interpolates with a near-step
# (|∇B|≈300), which makes ∇B·f erratic and the CBF-QP return u_safe=0. An L1
# hinge that caps ‖∇B‖ forces a bounded, SDF-like slope → smooth, well-conditioned
# gradient the QP can actually use to divert the field. (Natural slope is K≈4, so
# a cap a few× above that leaves headroom without re-creating the cliff.)
B_GRAD_MAX = 12.0     # legacy (cap); the eikonal now pins ‖∇B‖→K, see train.py
LAMBDA_EIK = 0.02     # weight on the (‖∇B‖ − K·in_band)² eikonal term

# Controller defends the level set B ≥ B_SAFE_MARGIN. Kept SMALL so the needle
# hugs the demo path and only bends sharply RIGHT AT the obstacle (just outside
# the light-red zone) instead of detouring early — the surgical use case wants
# strict adherence to the expert demo, deviating only when a critical zone is
# actually in the way, then snapping straight back.
B_SAFE_MARGIN = 0.008

# Demo-path attraction gain used at inference (overrides the training K=3 on the
# ProgressConditionedDS). Raised so the closed loop converges STRONGLY back onto
# the expert demo wherever it is clear; the CBF still wins locally at obstacles.
DEMO_K = 5.0   # strong attraction back to the expert demo (tight hugging)

# Go-around guidance (cbf_qp): when B < B_ACTIVE the controller adds a tangential
# velocity that circulates around the obstacle toward the goal (and opposite a
# moving obstacle's heading), breaking the head-on deadlock a strong demo pull
# would otherwise cause. SWIRL_GAIN scales it relative to the demo speed.
# Hybrid safety filter: exact geometric backstop margin (m). The learned CBF
# drives the field; this guarantees sdf ≥ SAFETY_SDF_MARGIN to the true tissue.
# Set to the inflation distance so the needle stays FULLY OUTSIDE the light-red
# CBF zone (sdf ≥ INFLATE), not just outside the solid tissue.
SAFETY_SDF_MARGIN = 0.011

B_ACTIVE   = 0.040
SWIRL_GAIN = 1.0   # tangential go-around guidance: circulates around the obstacle
                   # toward the goal (and opposite a moving obstacle's heading),
                   # breaking the head-on deadlock. Safe to run strong because the
                   # exact analytic filter (analytic_safety_filter) is the hard
                   # backstop — any unsafe swirl is geometrically clipped.

# Legacy compat
LAMBDA_F        = LAMBDA_MSE
LAMBDA_B_COND   = LAMBDA_B_DOT
LAMBDA_B_SDF    = 10.0
LAMBDA_D_PEN    = 35.0
LAMBDA_D_FREE   = 5.0
LAMBDA_D_SMOOTH = 0.02
N_EPOCHS        = 500
SAMPLE_SIZE     = 4096
DELTA_V2        = -1e-3
DELTA_B_FS      = 1e-3
DELTA_B_OB      = 1e-3
PHASE2_START    = 200
CONFORMAL_ALPHA = 0.05
N_CALIB         = 2000

# Workspace bounds
WS_LO = np.array([0.36, -0.09, 0.37], dtype=np.float32)
WS_HI = np.array([0.62,  0.26, 0.50], dtype=np.float32)

# Simulation
VEL_CLIP = 0.25
GOAL_TOL = 0.006
T_MAX    = 12.0
