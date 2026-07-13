"""
Table 2 generator — barrier generalization under obstacle rotation.

Measures the SAFETY-CRITICAL false-safe rate = fraction of truly-unsafe points a
learned barrier calls safe, as the obstacle is rotated. Isolates the barrier as a
FUNCTION (no controller, no backstop), which is the honest way to test the
pose-generalization claim. Compares:
  * ours (augmented composite barrier + encoder)   -> should stay FLAT
  * fixed-pose (same net, no augmentation)          -> degrades under rotation
  * CN-CBF (b5 per-obstacle nets, no encoder/aug)   -> degrades under rotation

Run:  venv/bin/python baselines/canonical/barrier_generalization.py
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))

import numpy as np
import torch

from simulate import load_model
from generalization_test import build_tensors
from config import (CRITICAL_SHAPES, CRITICAL_RADIUS, Z_CORRIDOR, INFLATE_MARGIN,
                    B_SAFE_MARGIN, transform_shape, sdf_critical_shape_2d)

ROOT = os.path.join(_HERE, "..", "..")
aug = load_model(os.path.join(ROOT, "checkpoints", "final_model.pt"))
fix = load_model(os.path.join(ROOT, "checkpoints", "fixed_pose_model.pt"))
from b5_cncbf_pershape.method import PerShapeBarrier   # noqa: E402


def _sample(sh_t, scale, seed):
    rng = np.random.default_rng(seed)
    c = sh_t['center'][:2]; half = CRITICAL_RADIUS * scale * 1.8
    xy = (c + rng.uniform(-half, half, (4000, 2))).astype(np.float32)
    sdf = sdf_critical_shape_2d(xy, sh_t)
    pts = np.concatenate([xy, np.full((len(xy), 1), Z_CORRIDOR, np.float32)], 1)
    return torch.tensor(pts), sdf


def false_safe_composite(model, sh_t, scale, seed):
    model.set_obstacles(build_tensors([sh_t], k=64, seed=seed))
    pts, sdf = _sample(sh_t, scale, seed)
    with torch.no_grad():
        B = model.B(pts).squeeze(-1).numpy()
    infl = sdf < INFLATE_MARGIN
    return (B[infl] > B_SAFE_MARGIN).mean() if infl.any() else 0.0


def false_safe_cncbf(sh_t, scale, seed):
    bar = PerShapeBarrier([sh_t])
    pts, sdf = _sample(sh_t, scale, seed)
    with torch.no_grad():
        B = bar(pts).squeeze(-1).numpy()
    infl = sdf < INFLATE_MARGIN
    return (B[infl] > B_SAFE_MARGIN).mean() if infl.any() else 0.0


def main():
    rots = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
    print("Table 2 - barrier false-safe rate (%) vs obstacle rotation")
    print("(inside 10mm keep-out but barrier says safe; lower=better; "
          "avg over 5 shapes x 3 scales)\n")
    print(f"{'rotation':>10}{'ours(aug)':>12}{'fixed-pose':>12}{'CN-CBF':>10}")
    for rot in rots:
        A, Fx, C = [], [], []
        for scale in (0.8, 1.0, 1.2):
            for i, base in enumerate(CRITICAL_SHAPES):
                sh_t = transform_shape(base, d_rot=rot, scale=scale)
                A.append(false_safe_composite(aug, sh_t, scale, 100 + i))
                Fx.append(false_safe_composite(fix, sh_t, scale, 100 + i))
                C.append(false_safe_cncbf(sh_t, scale, 100 + i))
        print(f"{np.degrees(rot):>9.0f} {np.mean(A) * 100:>10.1f}%"
              f"{np.mean(Fx) * 100:>10.1f}%{np.mean(C) * 100:>8.1f}%")


if __name__ == "__main__":
    main()
