"""
analytical_sdf.py — signed distance functions for the TA-CBF scene.

Foam obstacles  : ORIGINAL 3D spheres (unchanged from original codebase)
Critical zones  : NON-LINEAR XY cross-sections (star, crescent, blob, kidney, L-shape)
                  extruded in Z — the barrier B_φ is LEARNED, not this SDF.
                  This file only provides ground-truth labels for training.

Convention: SDF > 0 outside obstacle, SDF < 0 inside.
"""

import numpy as np
import torch
from config import (FOAM_CENTRES, FOAM_RADIUS, CRITICAL_MARGIN,
                    CRITICAL_SHAPES, sdf_all_critical_np, Z_CORRIDOR)


# ── NumPy versions ────────────────────────────────────────────────────────────

def sdf_foam_np(x: np.ndarray) -> np.ndarray:
    """SDF to union of foam SPHERES (original 3D geometry). x: (..., 3) → (...,)."""
    x   = np.asarray(x, dtype=np.float64)
    orig = x.shape[:-1]
    xf  = x.reshape(-1, 3)
    c   = np.asarray(FOAM_CENTRES, dtype=np.float64)
    d   = np.linalg.norm(xf[:, None, :] - c[None, :, :], axis=-1) - FOAM_RADIUS
    return d.min(axis=1).reshape(orig)


def sdf_critical_np(x: np.ndarray) -> np.ndarray:
    """
    SDF to non-linear critical shapes (XY cross-section extruded in Z).
    x: (..., 3) or (..., 2) → (...,). Positive = outside all critical zones.
    """
    x   = np.asarray(x, dtype=np.float64)
    orig = x.shape[:-1]
    xf  = x.reshape(-1, x.shape[-1])
    return sdf_all_critical_np(xf.astype(np.float32)).reshape(orig)


def sdf_all_np(x: np.ndarray) -> np.ndarray:
    """Composite SDF = min(foam, critical). x: (..., 3) → (...,)."""
    return np.minimum(sdf_foam_np(x), sdf_critical_np(x))


def penetration_depth_np(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, -sdf_all_np(x))


def is_in_obstacle_np(x: np.ndarray) -> np.ndarray:
    return sdf_all_np(x) < 0


def is_in_critical_np(x: np.ndarray) -> np.ndarray:
    return sdf_critical_np(x) < 0


def is_in_critical_buffer_np(x: np.ndarray) -> np.ndarray:
    return sdf_critical_np(x) < CRITICAL_MARGIN


def is_in_foam_only_np(x: np.ndarray) -> np.ndarray:
    return (sdf_foam_np(x) < 0) & (sdf_critical_np(x) >= 0)


# ── PyTorch differentiable versions ───────────────────────────────────────────

def sdf_foam_torch(x: torch.Tensor) -> torch.Tensor:
    """SDF to foam spheres. x: (B, 3) → (B, 1)."""
    c = torch.tensor(FOAM_CENTRES, dtype=x.dtype, device=x.device)
    d = torch.norm(x.unsqueeze(-2) - c, dim=-1) - FOAM_RADIUS
    return d.min(dim=-1, keepdim=True).values


def _sdf_star_torch(x2, center, outer_r, inner_r, n_points, rotation):
    c = torch.tensor(center[:2], dtype=x2.dtype, device=x2.device)
    p = x2 - c
    rot = float(rotation)
    cos_r = torch.tensor(np.cos(-rot), dtype=x2.dtype, device=x2.device)
    sin_r = torch.tensor(np.sin(-rot), dtype=x2.dtype, device=x2.device)
    px = cos_r * p[:, 0] - sin_r * p[:, 1]
    py = sin_r * p[:, 0] + cos_r * p[:, 1]
    angle = torch.atan2(py, px)
    r = torch.sqrt(px**2 + py**2 + 1e-8)
    sector = np.pi / n_points
    sa = torch.remainder(angle, 2 * sector) - sector
    star_r = inner_r + (outer_r - inner_r) * (1 - torch.abs(sa) / sector)
    return r - star_r


def _sdf_crescent_torch(x2, center, outer_r, inner_r, offset):
    c  = torch.tensor(center[:2], dtype=x2.dtype, device=x2.device)
    off = torch.tensor(offset,    dtype=x2.dtype, device=x2.device)
    r_outer = torch.sqrt(((x2 - c)**2).sum(dim=1) + 1e-8)
    r_inner = torch.sqrt(((x2 - c - off)**2).sum(dim=1) + 1e-8)
    return torch.maximum(r_outer - outer_r, inner_r - r_inner)


def _sdf_blob_torch(x2, center, radii, offsets):
    c = torch.tensor(center[:2], dtype=x2.dtype, device=x2.device)
    sdfs = []
    for r, off in zip(radii, offsets):
        ci = c + torch.tensor(off, dtype=x2.dtype, device=x2.device)
        sdfs.append(torch.sqrt(((x2 - ci)**2).sum(dim=1) + 1e-8) - float(r))
    return torch.stack(sdfs, dim=1).min(dim=1).values


def _sdf_kidney_torch(x2, center, outer_r, inner_r, squeeze, rotation):
    c = torch.tensor(center[:2], dtype=x2.dtype, device=x2.device)
    p = x2 - c
    rot = float(rotation)
    cos_r = torch.tensor(np.cos(-rot), dtype=x2.dtype, device=x2.device)
    sin_r = torch.tensor(np.sin(-rot), dtype=x2.dtype, device=x2.device)
    px = cos_r * p[:, 0] - sin_r * p[:, 1]
    py = sin_r * p[:, 0] + cos_r * p[:, 1]
    r_outer = torch.sqrt(px**2 + (py / squeeze)**2 + 1e-8) - outer_r
    r_inner = inner_r - torch.sqrt((px - outer_r * 0.4)**2 + py**2 + 1e-8)
    return torch.maximum(r_outer, r_inner)


def _sdf_lshape_torch(x2, center, arm1, arm2, rotation):
    c = torch.tensor(center[:2], dtype=x2.dtype, device=x2.device)
    p = x2 - c
    rot = float(rotation)
    cos_r = torch.tensor(np.cos(-rot), dtype=x2.dtype, device=x2.device)
    sin_r = torch.tensor(np.sin(-rot), dtype=x2.dtype, device=x2.device)
    px = cos_r * p[:, 0] - sin_r * p[:, 1]
    py = sin_r * p[:, 0] + cos_r * p[:, 1]

    def rect_sdf_t(qx, qy, hw, hh):
        dx = torch.abs(qx) - hw
        dy = torch.abs(qy) - hh
        return (torch.sqrt(torch.clamp(dx, min=0)**2 + torch.clamp(dy, min=0)**2 + 1e-8)
                + torch.clamp(torch.maximum(dx, dy), max=0))

    hw1, hh1 = float(arm1[0]) / 2, float(arm1[1]) / 2
    hw2, hh2 = float(arm2[0]) / 2, float(arm2[1]) / 2
    s1 = rect_sdf_t(px, py + hh2, hw1, hh1)
    s2 = rect_sdf_t(px + hw1 - hw2, py - hh1, hw2, hh2)
    return torch.minimum(s1, s2)


def sdf_critical_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Differentiable SDF to non-linear critical shapes.
    x: (B, 3) → (B, 1). Uses only XY, ignores Z.
    """
    x2 = x[:, :2]   # work in XY only
    sdfs = []
    for shape in CRITICAL_SHAPES:
        t = shape['type']
        if   t == 'star':     s = _sdf_star_torch(x2, shape['center'], shape['outer_r'], shape['inner_r'], shape['n_points'], shape['rotation'])
        elif t == 'crescent': s = _sdf_crescent_torch(x2, shape['center'], shape['outer_r'], shape['inner_r'], shape['offset'])
        elif t == 'blob':     s = _sdf_blob_torch(x2, shape['center'], shape['radii'], shape['offsets'])
        elif t == 'kidney':   s = _sdf_kidney_torch(x2, shape['center'], shape['outer_r'], shape['inner_r'], shape['squeeze'], shape['rotation'])
        elif t == 'lshape':   s = _sdf_lshape_torch(x2, shape['center'], shape['arm1'], shape['arm2'], shape['rotation'])
        sdfs.append(s)
    return torch.stack(sdfs, dim=1).min(dim=1, keepdim=True).values


def sdf_all_torch(x: torch.Tensor) -> torch.Tensor:
    """Composite SDF = min(foam, critical). x: (B, 3) → (B, 1)."""
    return torch.minimum(sdf_foam_torch(x), sdf_critical_torch(x))
