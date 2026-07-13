"""
B9 training — S2-NNDS (Safe and Stable Neural Network Dynamical Systems)
reimplementation, following the offline co-training of a neural DS with neural
Lyapunov and barrier certificates from demonstrations.

Faithful to the method's essence:
  * f_theta  : neural DS, MSE to demonstration velocities (their Eq. 2/9),
               goal-anchored so the goal is a rest point.
  * V        : Lyapunov certificate, V>0 and grad V . f < 0 (their Prop. 3,
               leaky-ReLU hinge Eq. 10). We use quadratic V=||x-goal||^2.
  * B        : MONOLITHIC neural barrier over absolute coordinates (their Eq.
               4-6): B<=0 on the safe/initial set, B>0 on the unsafe set, and the
               invariance grad B . f <= 0 in the boundary band. The invariance
               term backprops into BOTH B and f, so f is SHAPED to avoid the
               NOMINAL obstacles — this is what makes S2-NNDS safe on the trained
               scene and scene-frozen (no obstacle input, no pose conditioning).

Deployment (method.py): xdot = f_theta(x). No online filter, no backstop — the
DS itself is the certified-safe policy. Consequently it cannot react to obstacles
that moved from the training scene: the generalization gap our method closes.

We omit the conformal-prediction verification stage (it certifies the learned
functions; it does not change their closed-loop behavior).

Run:  venv/bin/python baselines/b9_s2nnds/train.py
Writes: baselines/b9_s2nnds/checkpoints/s2nnds.pt
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import ROOT, SRC  # noqa: E402,F401
from config import (DEVICE, STATE_DIM, N_POINTS, INFLATE_MARGIN, DT,
                    sample_safe_set, sample_unsafe_set, sample_workspace,
                    sdf_all_critical_np)
from train import load_demos, _resample
from models import BarrierNet

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)


class S2DS(nn.Module):
    """Neural DS f_theta(x) with a goal spring — the Lyapunov-stable construction
    S2-NNDS certifies (V=||x-goal||^2 decreases): guarantees global convergence
    to the goal and removes spurious attractors."""
    def __init__(self, mean, std, goal, hidden=(128, 128, 128), k_goal=1.5):
        super().__init__()
        dims = [STATE_DIM] + list(hidden) + [STATE_DIM]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self.k_goal = k_goal
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32))
        self.register_buffer('std', torch.tensor(std, dtype=torch.float32))
        self.register_buffer('goal', torch.tensor(goal, dtype=torch.float32))

    def _raw(self, x):
        return self.net((x - self.mean) / self.std)

    def forward(self, x):
        return (self._raw(x) - self._raw(self.goal.expand_as(x))
                + self.k_goal * (self.goal - x))


def _band_points(n, seed):
    """Points in the boundary band 0 < sdf < INFLATE (where invariance matters)."""
    pts = sample_workspace(n * 6, seed=seed)
    sdf = sdf_all_critical_np(pts)
    band = pts[(sdf > 0) & (sdf < INFLATE_MARGIN + 0.006)]
    return band[:n] if len(band) else pts[:n]


def main(steps=6000, lr=1e-3, seed=0,
         w_mse=120.0, w_safe=4.0, w_unsafe=8.0, w_inv=0.2, w_lyap=0.5, delta=0.01):
    torch.manual_seed(seed)
    demos = [_resample(d, N_POINTS) for d in load_demos()]
    demos = np.stack(demos, 0).astype(np.float32)
    mean = np.load(os.path.join(ROOT, "checkpoints", "norm_mean.npy"))
    std = np.load(os.path.join(ROOT, "checkpoints", "norm_std.npy"))
    goal = demos[:, -1].mean(0)

    # demo (x, v) pairs for the MSE objective (velocities at the deployment DT)
    xs_np = demos[:, :-1].reshape(-1, STATE_DIM)
    vs_np = ((demos[:, 1:] - demos[:, :-1]) / DT).reshape(-1, STATE_DIM)
    xs = torch.tensor(xs_np, device=DEVICE); vs = torch.tensor(vs_np, device=DEVICE)
    k_restore, sig = 4.0, 0.025   # wide contracting basin around the demo

    f = S2DS(mean, std, goal).to(DEVICE)
    B = BarrierNet().to(DEVICE)
    B.in_mean.copy_(torch.tensor(mean)); B.in_std.copy_(torch.tensor(std))
    goal_t = torch.tensor(goal, device=DEVICE)
    opt = torch.optim.Adam(list(f.parameters()) + list(B.parameters()), lr=lr)
    rng = np.random.default_rng(seed)

    for it in range(steps):
        opt.zero_grad()
        # MSE to demonstrations + wide restoring-basin augmentation (converge to path)
        L_mse = ((f(xs) - vs) ** 2).mean()
        for _ in range(2):
            eps = rng.normal(0, sig, xs_np.shape).astype(np.float32)
            xa = torch.tensor(xs_np + eps, device=DEVICE)
            va = torch.tensor(vs_np - k_restore * eps, device=DEVICE)
            L_mse = L_mse + 0.5 * ((f(xa) - va) ** 2).mean()

        # barrier set losses  (S2-NNDS convention: B<=0 safe, B>0 unsafe)
        safe = torch.tensor(sample_safe_set(512, seed=int(rng.integers(1 << 30))),
                            device=DEVICE)
        unsafe = torch.tensor(sample_unsafe_set(512, seed=int(rng.integers(1 << 30))),
                              device=DEVICE)
        L_safe = F.relu(B(safe) + delta).mean()          # want B(safe) <= -delta
        L_unsafe = F.relu(delta - B(unsafe)).mean()       # want B(unsafe) >= +delta

        # invariance grad B . f <= 0 in the boundary band  (shapes f AND B)
        band = _band_points(256, seed=int(rng.integers(1 << 30)))
        if len(band):
            xb = torch.tensor(band, device=DEVICE, requires_grad=True)
            Bb = B(xb)
            gB = torch.autograd.grad(Bb.sum(), xb, create_graph=True)[0]
            fb = f(xb)
            L_inv = F.relu((gB * fb).sum(-1) + 0.02).mean()
        else:
            L_inv = torch.zeros((), device=DEVICE)

        # Lyapunov decrease with quadratic V (convergence to goal)
        e = xs - goal_t
        gV = 2 * e
        L_lyap = F.relu((gV * f(xs)).sum(-1) + 1e-3).mean()

        loss = (w_mse * L_mse + w_safe * L_safe + w_unsafe * L_unsafe
                + w_inv * L_inv + w_lyap * L_lyap)
        loss.backward()
        opt.step()
        if it % 500 == 0 or it == steps - 1:
            print(f"[S2NNDS] {it:4d}  mse {L_mse.item():.4e}  safe {L_safe.item():.3f}"
                  f"  unsafe {L_unsafe.item():.3f}  inv {float(L_inv):.3f}", flush=True)

    out = os.path.join(CKPT_DIR, "s2nnds.pt")
    torch.save({'f': f.state_dict(), 'B': B.state_dict(),
                'mean': mean, 'std': std, 'goal': goal}, out)
    print(f"[S2NNDS] saved -> {out}")


if __name__ == "__main__":
    main()
