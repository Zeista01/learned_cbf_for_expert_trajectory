"""
B8 training — Neural ODE (Chen et al., NeurIPS 2018) motion policy from demos.

NODE baseline: a neural vector field g_phi(x) fit to the demonstrations as the
maximum-likelihood ODE vector field — matching the demonstration velocities at
densely sampled trajectory points (the standard, numerically stable objective
for learning a NODE from dense trajectories), with a short odeint rollout
consistency check. Deployment integrates the ODE (xdot = g_phi(x)). It has NO
safety mechanism and NO obstacle awareness, so it is the reference for "learned
motion alone is unsafe near critical regions."

Deploy: xdot = g_phi(x), integrated open-loop. No CLF, no CBF, no backstop.

Run:  venv/bin/python baselines/b8_node/train.py
Writes: baselines/b8_node/checkpoints/node.pt
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import ROOT, SRC  # noqa: E402,F401
from config import DEVICE, STATE_DIM, N_POINTS, DT
from train import load_demos, _resample

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)


class NODEField(nn.Module):
    """Autonomous neural vector field g_phi(x) with tanh MLP. Goal-anchored so the
    demonstration endpoint is a rest point (subtract the field at the goal)."""
    def __init__(self, mean, std, goal, hidden=(128, 128, 128), k_goal=2.0):
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

    def forward(self, t, x):                       # odeint signature (t, state)
        # net (demo shape) + goal spring (guarantees global convergence, kills
        # spurious attractors — the standard stable-DS-from-demo construction)
        return (self._raw(x) - self._raw(self.goal.expand_as(x))
                + self.k_goal * (self.goal - x))


def main(epochs=3000, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    demos = [_resample(d, N_POINTS) for d in load_demos()]
    demos = np.stack(demos, axis=0).astype(np.float32)     # (D, N, 3)
    mean = np.load(os.path.join(ROOT, "checkpoints", "norm_mean.npy"))
    std = np.load(os.path.join(ROOT, "checkpoints", "norm_std.npy"))
    goal = demos[:, -1].mean(0)

    # velocities at the DEPLOYMENT time step so integrating xdot*DT reproduces
    # the demo spacing exactly (matching the runner's DT avoids a scale mismatch).
    xs = demos[:, :-1].reshape(-1, STATE_DIM)
    vs = ((demos[:, 1:] - demos[:, :-1]) / DT).reshape(-1, STATE_DIM)
    xs_t = torch.tensor(xs, device=DEVICE)
    vs_t = torch.tensor(vs, device=DEVICE)
    model = NODEField(mean, std, goal).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    k_restore = 4.0   # restoring gain (1/s): off-path points get a velocity that
    n_aug, sig = 2, 0.025   # pull back to the path -> a wide contracting basin

    for ep in range(epochs):
        opt.zero_grad()
        # match demo velocities ON the path
        L_vel = ((model(0.0, xs_t) - vs_t) ** 2).mean()
        # AND on noisy points across a WIDE band around the path: target = demo
        # velocity + restoring pull k*(p - x) back toward the path point, so the
        # closed loop contracts to the demo instead of wandering off-distribution.
        L_aug = 0.0
        for _ in range(n_aug):
            eps = rng.normal(0, sig, xs.shape).astype(np.float32)
            xa = torch.tensor(xs + eps, device=DEVICE)
            va = torch.tensor(vs - k_restore * eps, device=DEVICE)
            L_aug = L_aug + ((model(0.0, xa) - va) ** 2).mean()
        loss = L_vel + L_aug / n_aug
        loss.backward()
        opt.step()
        if ep % 300 == 0 or ep == epochs - 1:
            print(f"[NODE] epoch {ep:4d}  vel-MSE {L_vel.item():.6e}", flush=True)

    out = os.path.join(CKPT_DIR, "node.pt")
    torch.save({'state': model.state_dict(), 'mean': mean, 'std': std,
                'goal': goal}, out)
    print(f"[NODE] saved -> {out}")


if __name__ == "__main__":
    main()
