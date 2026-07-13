"""
Leave-one-shape-out training of OURS' composite barrier.

Trains the exact same tolerance-aware, pose/scale-augmented composite barrier as
final_model, but on a subset of shapes with one shape TYPE held out. The held-out
shape is never seen during training; at eval time we test whether the
point-cloud encoder still produces a correct barrier for it (zero-shot to a novel
shape). This is the capability CN-CBF's per-shape networks structurally lack.

Run:  venv/bin/python baselines/novel_shape/train_holdout.py --holdout 0
      (holdout index: 0=star 1=crescent 3=kidney 5=lshape)
Writes: baselines/novel_shape/checkpoints/holdout_<idx>.pt
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

import config
import train  # noqa: E402  (patched below)
from config import (DEVICE, LR_BARRIER, canonical_interior_cloud)
from models import BPCBFModel, CompositeBarrier

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
SRC_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    all_shapes = config.CRITICAL_SHAPES
    keep = [s for i, s in enumerate(all_shapes) if i != args.holdout]
    held = all_shapes[args.holdout]
    print(f"[holdout {args.holdout}] held out = {held['type']}/{held['label']}; "
          f"training on {[s['type'] for s in keep]}", flush=True)

    # patch the shape set that augmented_barrier_loss / make_obstacle_tensors see
    train.CRITICAL_SHAPES = keep
    canon = [canonical_interior_cloud(sh, k=128, seed=200 + i)
             for i, sh in enumerate(keep)]

    # load f/V/norm/ref from final_model; fresh barrier (only B trains)
    mean = np.load(os.path.join(SRC_CKPT, "norm_mean.npy"))
    std = np.load(os.path.join(SRC_CKPT, "norm_std.npy"))
    ref = np.load(os.path.join(SRC_CKPT, "ref_path.npy"))
    m = BPCBFModel(ref_path=ref).to(DEVICE)
    sd = torch.load(os.path.join(SRC_CKPT, "final_model.pt"), map_location=DEVICE)
    sd = {k: v for k, v in sd.items() if k.startswith('f.') or k.startswith('V.')}
    m.load_state_dict(sd, strict=False)
    m.set_norm(torch.tensor(mean, dtype=torch.float32).to(DEVICE),
               torch.tensor(std, dtype=torch.float32).to(DEVICE))
    m.f.set_reference(ref)
    m.B = CompositeBarrier().to(DEVICE)
    m.B.set_scale(torch.tensor(std, dtype=torch.float32).to(DEVICE))
    for p in list(m.f.parameters()) + list(m.V.parameters()):
        p.requires_grad_(False)

    rng = np.random.default_rng(args.seed)
    opt = torch.optim.Adam(m.B.parameters(), lr=LR_BARRIER)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    for it in range(1, args.steps + 1):
        opt.zero_grad()
        loss, c = train.augmented_barrier_loss(m, canon, rng, return_components=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.B.parameters(), 5.0)
        opt.step(); sched.step()
        if it % 500 == 0 or it == 1:
            print(f"  step {it:4d}/{args.steps} loss={loss.item():.3f} "
                  f"acc[safe={c['safe_acc']*100:.0f}% uns={c['unsafe_acc']*100:.0f}%]",
                  flush=True)

    out = os.path.join(CKPT_DIR, f"holdout_{args.holdout}.pt")
    m.save(out)
    print(f"[holdout {args.holdout}] saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
