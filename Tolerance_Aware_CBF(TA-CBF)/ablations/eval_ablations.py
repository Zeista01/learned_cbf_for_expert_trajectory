"""
Evaluate the ablations and print the three tables the paper needs:

  A1  Augmentation: barrier false-safe rate vs obstacle rotation, per aug mode.
      (which augmentation carries the pose-generalization?)
  A2  Eikonal: mean gradient magnitude ||grad B|| and false-safe, eikonal on vs off.
      (does removing the eikonal shaping blow up the gradient / destabilize?)
  A3  Smooth-min vs hard-min: gradient smoothness when two obstacles are close.
      (does the smooth minimum keep the fused barrier differentiable?)

Run:  venv/bin/python ablations/eval_ablations.py
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import numpy as np
import torch

from simulate import load_model
from generalization_test import build_tensors
from config import (CRITICAL_SHAPES, CRITICAL_RADIUS, Z_CORRIDOR, INFLATE_MARGIN,
                    B_SAFE_MARGIN, transform_shape, sdf_critical_shape_2d)

CK = os.path.join(_HERE, "checkpoints")
ROTS = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
SCALES = (0.8, 1.0, 1.2)


def _pts(sh_t, scale, seed):
    rng = np.random.default_rng(seed)
    c = sh_t['center'][:2]; half = CRITICAL_RADIUS*scale*1.8
    xy = (c + rng.uniform(-half, half, (4000, 2))).astype(np.float32)
    sdf = sdf_critical_shape_2d(xy, sh_t)
    pts = np.concatenate([xy, np.full((len(xy), 1), Z_CORRIDOR, np.float32)], 1)
    return torch.tensor(pts), sdf


def false_safe_at(model, rot):
    vals = []
    for sc in SCALES:
        for i, base in enumerate(CRITICAL_SHAPES):
            sh_t = transform_shape(base, d_rot=rot, scale=sc)
            model.set_obstacles(build_tensors([sh_t], k=64, seed=100+i))
            pts, sdf = _pts(sh_t, sc, 100+i)
            with torch.no_grad():
                B = model.B(pts).squeeze(-1).numpy()
            infl = sdf < INFLATE_MARGIN
            vals.append((B[infl] > B_SAFE_MARGIN).mean() if infl.any() else 0.0)
    return float(np.mean(vals))*100


def grad_norm(model, n=3000):
    """Mean ||grad B|| over the workspace on the nominal scene."""
    from config import SLAB_X, SLAB_Y
    model.set_obstacles(build_tensors(CRITICAL_SHAPES, k=64, seed=0))
    rng = np.random.default_rng(0)
    xy = np.stack([rng.uniform(SLAB_X[0], SLAB_X[1], n),
                   rng.uniform(SLAB_Y[0], SLAB_Y[1], n)], 1).astype(np.float32)
    p = np.concatenate([xy, np.full((n,1), Z_CORRIDOR, np.float32)], 1)
    g = model.B.gradient(torch.tensor(p)).detach().numpy()
    gn = np.linalg.norm(g[:, :2], axis=1)
    return float(gn.mean()), float(np.percentile(gn, 99))


def load(tag):
    p = os.path.join(CK, tag)
    return load_model(p) if os.path.exists(p) else None


def main():
    print("="*70)
    print("A1  AUGMENTATION — barrier false-safe (%) vs rotation (lower=better)")
    print("="*70)
    modes = [('none','abl_none_eik-on.pt'), ('trans','abl_trans_eik-on.pt'),
             ('scale','abl_scale_eik-on.pt'), ('rot','abl_rot_eik-on.pt'),
             ('full','abl_full_eik-on.pt')]
    print(f"{'aug mode':<10}" + "".join(f"{int(np.degrees(r)):>7}°" for r in ROTS) + f"{'mean':>8}")
    for name, tag in modes:
        m = load(tag)
        if m is None: print(f"{name:<10} (missing {tag})"); continue
        vs = [false_safe_at(m, r) for r in ROTS]
        print(f"{name:<10}" + "".join(f"{v:>8.1f}" for v in vs) + f"{np.mean(vs):>8.1f}")

    print("\n" + "="*70)
    print("A2  EIKONAL — gradient magnitude and false-safe, eikonal on vs off")
    print("="*70)
    print(f"{'variant':<16}{'mean|gradB|':>13}{'p99|gradB|':>12}{'false-safe':>12}")
    for name, tag in [('eikonal ON','abl_full_eik-on.pt'), ('eikonal OFF','abl_full_eik-off.pt')]:
        m = load(tag)
        if m is None: print(f"{name:<16} (missing {tag})"); continue
        gm, gp = grad_norm(m)
        fs = np.mean([false_safe_at(m, r) for r in ROTS])
        print(f"{name:<16}{gm:>13.1f}{gp:>12.1f}{fs:>11.1f}%")

    print("\n" + "="*70)
    print("A3  SMOOTH-MIN vs HARD-MIN — fused-barrier gradient smoothness")
    print("(two obstacles placed close; report max ||grad B|| along the seam,")
    print(" where hard-min has a non-differentiable ridge)")
    print("="*70)
    m = load('abl_full_eik-on.pt')
    if m is not None:
        import copy
        a = copy.deepcopy(CRITICAL_SHAPES[2]); a['center'] = np.array([0.49,0.06,Z_CORRIDOR],np.float32)
        b = copy.deepcopy(CRITICAL_SHAPES[3]); b['center'] = np.array([0.49,0.10,Z_CORRIDOR],np.float32)
        m.set_obstacles(build_tensors([a, b], k=64, seed=1))
        ys = np.linspace(0.06, 0.10, 400)
        p = np.stack([np.full_like(ys,0.49), ys, np.full_like(ys,Z_CORRIDOR)],1).astype(np.float32)
        pt = torch.tensor(p)
        for beta, lab in [(m.B.beta,'smooth-min (beta=%d)'%int(m.B.beta)), (1e6,'hard-min (beta=1e6)')]:
            old = m.B.beta; m.B.beta = beta
            g = m.B.gradient(pt).detach().numpy()[:,1]
            m.B.beta = old
            jerk = np.abs(np.diff(g, 2))
            print(f"  {lab:<26} max|d2B/dy2| along seam = {jerk.max():.1f}  "
                  f"(smoother = smaller)")


if __name__ == "__main__":
    main()
