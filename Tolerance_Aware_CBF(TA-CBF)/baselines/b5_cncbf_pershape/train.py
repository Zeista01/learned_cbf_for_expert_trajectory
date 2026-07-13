"""
B5 training — CN-CBF-style per-obstacle barrier networks.

Faithful planar adaptation of the CN-CBF design (Derajic et al.): one MLP per
robot-obstacle pair, evaluated on RELATIVE coordinates, fused by smooth-min.
For our single-integrator tool the HJ value function of a static obstacle
reduces to its (shaped) signed distance, so each net regresses the same
K*clip(sdf - Delta) target ours uses — trained on the obstacle's CANONICAL
pose only (their design has no shape encoder and no pose augmentation; that is
precisely the delta our method adds).

Relative coordinates make it translation-invariant; rotation and scale of the
shape are baked in at training pose, so transformed scenes go out of
distribution — the comparison the paper needs.

One net per CRITICAL_SHAPES entry, keyed by its label (the two blobs differ).

Run:  venv/bin/python baselines/b5_cncbf_pershape/train.py
Writes: baselines/b5_cncbf_pershape/checkpoints/pershape_nets.pt
"""
import copy
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import SRC  # noqa: E402,F401
from config import (BARRIER_SDF_CLAMP_IN, BARRIER_SDF_CLAMP_OUT,  # noqa: E402
                    BARRIER_SDF_K, CRITICAL_SHAPES, DEVICE, INFLATE_MARGIN,
                    LAMBDA_EIK, Z_CORRIDOR, sample_local_box,
                    sdf_critical_shape_2d)

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

REL_SCALE = 0.05  # metres -> O(1) net inputs


def make_net():
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(2, 128), nn.Tanh(),
        nn.Linear(128, 128), nn.Tanh(),
        nn.Linear(128, 128), nn.Tanh(),
        nn.Linear(128, 1),
    )


def train_one(shape, steps=6000, batch=1024, lr=1e-3, seed=0):
    canon = copy.deepcopy(shape)
    canon['center'] = np.array([0.0, 0.0, Z_CORRIDOR], dtype=np.float32)
    net = make_net().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps,
                                                       eta_min=lr * 1e-2)
    rng = np.random.default_rng(seed)
    for it in range(steps):
        # pad past the runtime far-field blend radius (method.R_OUT = 0.048)
        # so every point the blend can hand the net is in-distribution
        pts3, sdf = sample_local_box(canon, n=batch, seed=seed * 100000 + it,
                                     pad=0.036)
        # the regression error that matters is at the zero level set: densify
        # the band around it (jittered copies of near-boundary samples)
        band = pts3[np.abs(sdf - INFLATE_MARGIN) < 0.02]
        if len(band):
            extra = band.copy()
            extra[:, :2] += rng.normal(0, 0.004, (len(band), 2)).astype(np.float32)
            sdf_e = sdf_critical_shape_2d(extra[:, :2], canon)
            pts3 = np.concatenate([pts3, extra], axis=0)
            sdf = np.concatenate([sdf, sdf_e], axis=0)
        target = BARRIER_SDF_K * np.clip(
            sdf - INFLATE_MARGIN, -BARRIER_SDF_CLAMP_IN, BARRIER_SDF_CLAMP_OUT)
        x = torch.tensor(pts3[:, :2] / REL_SCALE, dtype=torch.float32,
                         device=DEVICE, requires_grad=True)
        y = torch.tensor(target, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
        B = net(x)
        mse = ((B - y) ** 2).mean()
        g = torch.autograd.grad(B.sum(), x, create_graph=True)[0] / REL_SCALE
        in_band = torch.tensor(
            ((sdf - INFLATE_MARGIN > -BARRIER_SDF_CLAMP_IN)
             & (sdf - INFLATE_MARGIN < BARRIER_SDF_CLAMP_OUT)).astype(np.float32),
            device=DEVICE)
        eik = ((g.norm(dim=1) - BARRIER_SDF_K * in_band) ** 2).mean()
        loss = mse + LAMBDA_EIK * eik
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 1000 == 0 or it == steps - 1:
            print(f"    step {it:5d}  mse {mse.item():.6f}", flush=True)
    return net


def main():
    state = {}
    for i, sh in enumerate(CRITICAL_SHAPES):
        print(f"[B5 train] net {i}: {sh['label']} ({sh['type']})", flush=True)
        net = train_one(sh, seed=i)
        state[sh['label']] = net.state_dict()
    out = os.path.join(CKPT_DIR, "pershape_nets.pt")
    torch.save(state, out)
    print(f"[B5 train] saved -> {out}")


if __name__ == "__main__":
    main()
