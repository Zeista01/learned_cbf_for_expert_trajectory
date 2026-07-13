"""
B7 training — single global barrier B(x) on ABSOLUTE coordinates, fit to the
NOMINAL scene's shaped SDF (same K*clip target + eikonal smoothing as ours).

This is the structural stand-in for scene-frozen certificate systems
(S2-NNDS-style): no obstacle conditioning, no compositional structure, so the
barrier is welded to the scene it was trained on. Perfect on nominal; has no
mechanism to follow a moved/rotated/rescaled obstacle.

Run:  venv/bin/python baselines/b7_global_barrier/train.py
Writes: baselines/b7_global_barrier/checkpoints/global_barrier.pt
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import ROOT, SRC  # noqa: E402,F401
from config import (BARRIER_SDF_CLAMP_IN, BARRIER_SDF_CLAMP_OUT,  # noqa: E402
                    BARRIER_SDF_K, DEVICE, INFLATE_MARGIN, LAMBDA_EIK,
                    SLAB_X, SLAB_Y, Z_CORRIDOR, sdf_all_critical_np)
from models import BarrierNet  # noqa: E402

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)


def sample_batch(rng, n):
    """Uniform slab samples + a band concentrated near obstacle boundaries."""
    xy_u = np.stack([rng.uniform(SLAB_X[0], SLAB_X[1], n),
                     rng.uniform(SLAB_Y[0], SLAB_Y[1], n)], axis=1)
    # boundary band: perturb points whose |sdf| is small
    sdf_u = sdf_all_critical_np(xy_u.astype(np.float32))
    band = xy_u[np.abs(sdf_u) < 0.03]
    if len(band):
        band = band + rng.normal(0, 0.006, band.shape)
    xy = np.concatenate([xy_u, band], axis=0).astype(np.float32)
    sdf = sdf_all_critical_np(xy)
    target = BARRIER_SDF_K * np.clip(sdf - INFLATE_MARGIN,
                                     -BARRIER_SDF_CLAMP_IN, BARRIER_SDF_CLAMP_OUT)
    pts3 = np.concatenate([xy, np.full((len(xy), 1), Z_CORRIDOR, np.float32)],
                          axis=1)
    return (torch.tensor(pts3, device=DEVICE),
            torch.tensor(target, dtype=torch.float32, device=DEVICE).unsqueeze(-1),
            torch.tensor(sdf, dtype=torch.float32, device=DEVICE))


def main(steps=4000, batch=2048, lr=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    net = BarrierNet().to(DEVICE)
    mean = torch.tensor(np.load(os.path.join(ROOT, "checkpoints", "norm_mean.npy")),
                        dtype=torch.float32, device=DEVICE)
    std = torch.tensor(np.load(os.path.join(ROOT, "checkpoints", "norm_std.npy")),
                       dtype=torch.float32, device=DEVICE)
    net.in_mean.copy_(mean); net.in_std.copy_(std)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for it in range(steps):
        x, y, sdf = sample_batch(rng, batch)
        x.requires_grad_(True)
        B = net(x)
        mse = ((B - y) ** 2).mean()
        g = torch.autograd.grad(B.sum(), x, create_graph=True)[0][:, :2]
        # eikonal: pin the planar slope to K inside the clip band, 0 outside
        in_band = ((sdf - INFLATE_MARGIN > -BARRIER_SDF_CLAMP_IN)
                   & (sdf - INFLATE_MARGIN < BARRIER_SDF_CLAMP_OUT)).float()
        eik = ((g.norm(dim=1) - BARRIER_SDF_K * in_band) ** 2).mean()
        loss = mse + LAMBDA_EIK * eik
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 500 == 0 or it == steps - 1:
            print(f"[B7 train] step {it:5d}  mse {mse.item():.6f}  "
                  f"eik {eik.item():.4f}", flush=True)

    out = os.path.join(CKPT_DIR, "global_barrier.pt")
    torch.save(net.state_dict(), out)
    print(f"[B7 train] saved -> {out}")


if __name__ == "__main__":
    main()
