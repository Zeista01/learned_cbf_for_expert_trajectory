"""
Novel-shape evaluation: can the barrier certify a shape it never saw in training?

For each held-out shape we measure the SAFETY-CRITICAL false-safe rate (fraction
of truly-unsafe points the barrier labels safe), averaged over rotations & scales:
  * Ours (full)      : trained on ALL shapes  -> reference (shape was in training)
  * Ours (hold-out)  : trained WITHOUT this shape -> zero-shot to a novel shape
  * CN-CBF           : has no network for an unseen shape. Its best option is to
                       apply the network of another (wrong) trained shape; we
                       report the BEST such wrong-net (lowest false-safe) as a
                       generous upper bound on what CN-CBF could do.

Run:  venv/bin/python baselines/novel_shape/eval_novel.py
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))          # baselines/  (for b5_cncbf_pershape)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))

import numpy as np
import torch

import config
from simulate import load_model
from generalization_test import build_tensors
from config import (CRITICAL_SHAPES, CRITICAL_RADIUS, Z_CORRIDOR, INFLATE_MARGIN,
                    B_SAFE_MARGIN, transform_shape, sdf_critical_shape_2d)

ROOT = os.path.join(_HERE, "..", "..")
HOLDOUTS = {0: 'star', 1: 'crescent', 3: 'kidney', 5: 'L-shape'}
ROTS = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
SCALES = (0.8, 1.0, 1.2)


def _sample(sh_t, scale, seed):
    rng = np.random.default_rng(seed)
    c = sh_t['center'][:2]; half = CRITICAL_RADIUS * scale * 1.8
    xy = (c + rng.uniform(-half, half, (4000, 2))).astype(np.float32)
    sdf = sdf_critical_shape_2d(xy, sh_t)
    pts = np.concatenate([xy, np.full((len(xy), 1), Z_CORRIDOR, np.float32)], 1)
    return torch.tensor(pts), sdf


def fs_composite(model, sh_t, scale, seed):
    model.set_obstacles(build_tensors([sh_t], k=64, seed=seed))
    pts, sdf = _sample(sh_t, scale, seed)
    with torch.no_grad():
        B = model.B(pts).squeeze(-1).numpy()
    infl = sdf < INFLATE_MARGIN
    return (B[infl] > B_SAFE_MARGIN).mean() if infl.any() else 0.0


def avg_fs(model, shape, seed0=100):
    vals = []
    for rot in ROTS:
        for sc in SCALES:
            sh_t = transform_shape(shape, d_rot=rot, scale=sc)
            vals.append(fs_composite(model, sh_t, sc, seed0))
    return float(np.mean(vals)) * 100


def main():
    full = load_model(os.path.join(ROOT, "checkpoints", "final_model.pt"))
    from b5_cncbf_pershape.method import PerShapeBarrier

    print("Novel-shape barrier false-safe rate (%) — lower is better")
    print("(held-out shape never seen in training; avg over 5 rotations x 3 scales)\n")
    print(f"{'held-out shape':<16}{'Ours (full)':>13}{'Ours (holdout)':>16}{'CN-CBF (wrong-net)':>20}")
    rows = []
    for j, name in HOLDOUTS.items():
        shape = CRITICAL_SHAPES[j]
        ho_path = os.path.join(_HERE, "checkpoints", f"holdout_{j}.pt")
        if not os.path.exists(ho_path):
            print(f"{name:<16}  (missing {ho_path})"); continue
        ho = load_model(ho_path)
        fs_full = avg_fs(full, shape)
        fs_ho = avg_fs(ho, shape)

        # CN-CBF: no net for shape j. It CANNOT identify the novel shape, so it
        # must apply some other shape's net blindly. We report the AVERAGE over
        # all wrong-net choices = the expected false-safe of a blind assignment
        # (best-case would cherry-pick a lucky net it cannot actually select).
        per_wrong = []
        for k in [i for i in range(len(CRITICAL_SHAPES)) if i != j]:
            vals = []
            for rot in ROTS:
                for sc in SCALES:
                    sh_true = transform_shape(shape, d_rot=rot, scale=sc)
                    wrong_named = dict(shape); wrong_named['label'] = CRITICAL_SHAPES[k]['label']
                    sh_wrong = transform_shape(wrong_named, d_rot=rot, scale=sc)
                    bar = PerShapeBarrier([sh_wrong])
                    pts, sdf = _sample(sh_true, sc, 100)
                    with torch.no_grad():
                        B = bar(pts).squeeze(-1).numpy()
                    infl = sdf < INFLATE_MARGIN
                    vals.append((B[infl] > B_SAFE_MARGIN).mean() if infl.any() else 0.0)
            per_wrong.append(float(np.mean(vals)) * 100)
        cn = float(np.mean(per_wrong))
        print(f"{name:<16}{fs_full:>12.1f}%{fs_ho:>15.1f}%{cn:>19.1f}%")
        rows.append((name, fs_full, fs_ho, cn))

    if rows:
        arr = np.array([[r[1], r[2], r[3]] for r in rows])
        print(f"\n{'MEAN':<16}{arr[:,0].mean():>12.1f}%{arr[:,1].mean():>15.1f}%{arr[:,2].mean():>19.1f}%")
    np.save(os.path.join(_HERE, "novel_shape_results.npy"),
            np.array(rows, dtype=object), allow_pickle=True)


if __name__ == "__main__":
    main()
