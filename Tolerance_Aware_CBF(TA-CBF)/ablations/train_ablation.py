"""
Ablation barrier trainer. Trains the composite barrier on ALL shapes (same recipe
as final_model) but with one component toggled, so each ablation differs from the
full model in exactly one place:

  --aug {full,none,rot,scale,trans}   which augmentation is active
  --eikonal {on,off}                  eikonal slope regularization

Writes: ablations/checkpoints/abl_<aug>_eik-<on|off>.pt

Run:  venv/bin/python ablations/train_ablation.py --aug rot --eikonal on
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import config
import train
from config import DEVICE, LR_BARRIER, canonical_interior_cloud, CRITICAL_SHAPES
from models import BPCBFModel, CompositeBarrier

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
SRC_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

AUG = {  # (rot_range, scale_range, trans_range) per mode
    'full':  ((-np.pi, np.pi), (0.65, 1.4), 0.05),
    'none':  ((0.0, 0.0),      (1.0, 1.0),  0.0),
    'rot':   ((-np.pi, np.pi), (1.0, 1.0),  0.0),
    'scale': ((0.0, 0.0),      (0.65, 1.4), 0.0),
    'trans': ((0.0, 0.0),      (1.0, 1.0),  0.05),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", choices=list(AUG), default="full")
    ap.add_argument("--eikonal", choices=["on", "off"], default="on")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # toggle augmentation ranges (random_transform reads these config globals)
    rr, sr, tr = AUG[args.aug]
    config.AUG_ROT_RANGE = rr; config.AUG_SCALE_RANGE = sr; config.AUG_TRANS_RANGE = tr
    # toggle eikonal (the loss reads train.LAMBDA_EIK, imported by value)
    if args.eikonal == "off":
        train.LAMBDA_EIK = 0.0
    print(f"[abl aug={args.aug} eik={args.eikonal}] rot={rr} scale={sr} trans={tr} "
          f"LAMBDA_EIK={train.LAMBDA_EIK}", flush=True)

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

    canon = [canonical_interior_cloud(sh, k=128, seed=200 + i)
             for i, sh in enumerate(CRITICAL_SHAPES)]
    rng = np.random.default_rng(args.seed)
    opt = torch.optim.Adam(m.B.parameters(), lr=LR_BARRIER)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    last_grad = 0.0
    for it in range(1, args.steps + 1):
        opt.zero_grad()
        loss, c = train.augmented_barrier_loss(m, canon, rng, return_components=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.B.parameters(), 5.0)
        opt.step(); sched.step()
        last_grad = c['gradB']
        if it % 500 == 0 or it == 1:
            print(f"  step {it:4d}/{args.steps} loss={loss.item():.3f} "
                  f"acc[safe={c['safe_acc']*100:.0f}% uns={c['unsafe_acc']*100:.0f}%] "
                  f"|gradB|={c['gradB']:.1f}", flush=True)

    out = os.path.join(CKPT_DIR, f"abl_{args.aug}_eik-{args.eikonal}.pt")
    m.save(out)
    print(f"[abl] saved -> {out}  (final |gradB|={last_grad:.1f})", flush=True)


if __name__ == "__main__":
    main()
