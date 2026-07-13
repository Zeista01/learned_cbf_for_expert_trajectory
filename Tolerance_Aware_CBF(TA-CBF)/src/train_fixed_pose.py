"""
train_fixed_pose.py — BASELINE: train the composite barrier with pose/scale
augmentation DISABLED (obstacles seen only at their canonical pose).

Everything else (frozen f/V, SDF-shaped target, multi-obstacle smooth-min,
eikonal cap, loss weights, step count) is IDENTICAL to the augmented trainer
train_barrier_fast.py. The ONLY difference is that random_transform is replaced
by the identity, so the encoder never sees a rotated or rescaled obstacle. This
is the honest ablation for the pose-generalization claim.

Saves to checkpoints/fixed_pose_model.pt (NEVER touches final_model.pt).

Usage:
    python src/train_fixed_pose.py --steps 4000
"""
import os, sys, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import train  # so we can monkeypatch its module-global random_transform
from models import CompositeBarrier
from config import DEVICE, LR_BARRIER, INFLATE_MARGIN, BARRIER_SDF_K
from train import make_obstacle_tensors, make_canonical_clouds, CKPT_DIR
from train_barrier_fast import load_fV


def identity_transform(rng, rot_range=None, scale_range=None, trans_range=None):
    """Replacement for config.random_transform: NO augmentation (canonical pose)."""
    return 0.0, 1.0, np.zeros(2, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--src", default="final_model.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # ── DISABLE augmentation: patch the name used inside augmented_barrier_loss ─
    train.random_transform = identity_transform
    print(f"[FixedPose] augmentation DISABLED (canonical pose only). "
          f"device={DEVICE} steps={args.steps}")

    m = load_fV(args.src)                       # frozen good f/V + fresh barrier
    for p in list(m.f.parameters()) + list(m.V.parameters()):
        p.requires_grad_(False)

    canon = make_canonical_clouds(k=128)
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
            print(f"  step {it:4d}/{args.steps} | loss={loss.item():.3f} "
                  f"acc[safe={c['safe_acc']*100:.0f}% uns={c['unsafe_acc']*100:.0f}%] "
                  f"|gradB|={c['gradB']:.1f}")

    m.set_obstacles(make_obstacle_tensors(k=64, seed=0))
    out = os.path.join(CKPT_DIR, "fixed_pose_model.pt")
    m.save(out)
    print(f"[FixedPose] done -> {out}")


if __name__ == "__main__":
    main()
