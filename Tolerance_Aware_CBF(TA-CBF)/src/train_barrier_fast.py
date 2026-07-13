"""
train_barrier_fast.py — fast, barrier-ONLY training of the composite CBF.

Rationale: f_θ (demo dynamical system) and V_θ (CLF) are obstacle-AGNOSTIC and
are already converged in the existing checkpoint. Only the barrier B needs the
SDF-shaped, pose/scale/count-augmented recipe. Freezing f/V and training B alone
skips the expensive RK4 imitation rollout, so the whole barrier converges in a
couple of minutes on GPU instead of hours — and we validate the END-TO-END
diverting field before committing, so it only has to be trained once.

What this barrier is trained to handle, in ONE shot:
  • multiple obstacles + arbitrary count   (random subset each step + smooth-min)
  • position changes                        (conditional CBF sees only x − center)
  • rotation                                (random rotation augmentation)
  • enlarge / shrink                        (random scale augmentation)
  • slow dynamics                           (re-evaluated each control step)

Usage:
    cd /home/stanny/franka_ros2_ws/src/Tolerance_Aware_CBF(TA-CBF)
    python src/train_barrier_fast.py --steps 4000
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from models import BPCBFModel, CompositeBarrier
from config import DEVICE, LR_BARRIER, INFLATE_MARGIN, BARRIER_SDF_K
from train import (make_obstacle_tensors, make_canonical_clouds,
                   augmented_barrier_loss, CKPT_DIR)

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")


def load_fV(src: str = "final_model.pt") -> BPCBFModel:
    """Load f/V/norm/ref from an existing checkpoint; reset B to a fresh net."""
    mean = np.load(os.path.join(CKPT_DIR, "norm_mean.npy"))
    std  = np.load(os.path.join(CKPT_DIR, "norm_std.npy"))
    ref  = np.load(os.path.join(CKPT_DIR, "ref_path.npy"))
    m = BPCBFModel(ref_path=ref).to(DEVICE)
    sd = torch.load(os.path.join(CKPT_DIR, src), map_location=DEVICE)
    # take ONLY f/V (the barrier is reinitialized below, and its architecture
    # may differ from the checkpoint — never copy B.* here)
    sd = {k: v for k, v in sd.items() if k.startswith('f.') or k.startswith('V.')}
    m.load_state_dict(sd, strict=False)
    m.set_norm(torch.tensor(mean, dtype=torch.float32).to(DEVICE),
               torch.tensor(std,  dtype=torch.float32).to(DEVICE))
    m.f.set_reference(ref)
    # fresh barrier (the only thing we retrain)
    m.B = CompositeBarrier().to(DEVICE)
    m.B.set_scale(torch.tensor(std, dtype=torch.float32).to(DEVICE))
    m.set_obstacles(make_obstacle_tensors(k=64, seed=0))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--src", default="final_model.pt",
                    help="checkpoint to take the trained f/V from")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"[FastB] device={DEVICE}  inflation={INFLATE_MARGIN*1e3:.0f}mm  K={BARRIER_SDF_K}")
    m = load_fV(args.src)
    # freeze f and V — only the barrier learns
    for p in list(m.f.parameters()) + list(m.V.parameters()):
        p.requires_grad_(False)

    canon = make_canonical_clouds(k=128)
    rng = np.random.default_rng(args.seed)
    opt = torch.optim.Adam(m.B.parameters(), lr=LR_BARRIER)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    best = -1.0
    hist = {'step': [], 'loss': [], 'reg': [], 'safe': [], 'unsafe': [],
            'safe_acc': [], 'unsafe_acc': [], 'gradB': []}
    for it in range(1, args.steps + 1):
        opt.zero_grad()
        loss, c = augmented_barrier_loss(m, canon, rng, return_components=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.B.parameters(), 5.0)
        opt.step(); sched.step()

        hist['step'].append(it); hist['loss'].append(loss.item())
        hist['reg'].append(c['b_reg']); hist['safe'].append(c['b_safe'])
        hist['unsafe'].append(c['b_unsafe']); hist['safe_acc'].append(c['safe_acc'])
        hist['unsafe_acc'].append(c['unsafe_acc']); hist['gradB'].append(c['gradB'])
        score = 0.5 * (c['safe_acc'] + c['unsafe_acc'])
        if it > args.steps // 2 and score > best:
            best = score
            m.set_obstacles(make_obstacle_tensors(k=64, seed=0))
            m.save(os.path.join(CKPT_DIR, "best_model.pt"), quiet=True)
        if it % 200 == 0 or it == 1:
            print(f"  step {it:4d}/{args.steps} | loss={loss.item():.3f} "
                  f"reg={c['b_reg']:.4f} safe={c['b_safe']:.4f} uns={c['b_unsafe']:.4f} "
                  f"acc[safe={c['safe_acc']*100:.0f}% uns={c['unsafe_acc']*100:.0f}%] "
                  f"|gradB|={c['gradB']:.1f} B=[{c['Bu_min']:+.3f},{c['B0_max']:+.3f}]")

    # final save (canonical obstacles installed)
    m.set_obstacles(make_obstacle_tensors(k=64, seed=0))
    m.save(os.path.join(CKPT_DIR, "final_model.pt"))
    np.savez(os.path.join(CKPT_DIR, "barrier_loss_history.npz"),
             **{k: np.asarray(v) for k, v in hist.items()})
    print(f"[FastB] done. best mean-acc={best:.3f}. saved final_model.pt + best_model.pt "
          f"+ barrier_loss_history.npz")


if __name__ == "__main__":
    main()
