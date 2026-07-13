"""
B5 — CN-CBF-style composite barrier at runtime.

Per-obstacle nets on relative coordinates + the same smooth-min as ours.
Uses ground-truth obstacle identity (generous to the baseline — a real system
would have to classify the shape first). Translation generalizes by
construction; rotation/scale of the shapes was baked in at training pose.
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.methods import BarrierSwapMethod  # noqa: E402
from b5_cncbf_pershape.train import CKPT_DIR, REL_SCALE, make_net  # noqa: E402
from config import (BARRIER_SDF_CLAMP_OUT, BARRIER_SDF_K, DEVICE)  # noqa: E402
from models import BARRIER_BETA  # noqa: E402

CKPT = os.path.join(CKPT_DIR, "pershape_nets.pt")

# Far-field blend (the role CN-CBF's residual architecture plays): the nets are
# only trained inside a local box around each obstacle, so beyond it we blend
# to the saturated safe value instead of trusting extrapolation.
R_IN, R_OUT = 0.032, 0.048
B_FAR = BARRIER_SDF_K * BARRIER_SDF_CLAMP_OUT


class PerShapeBarrier(nn.Module):
    def __init__(self, shapes):
        super().__init__()
        state = torch.load(CKPT, map_location=DEVICE)
        self.nets = nn.ModuleDict()
        for label, sd in state.items():
            net = make_net()
            net.load_state_dict(sd)
            self.nets[label] = net.to(DEVICE).eval()
        self.set_obstacles(shapes)

    def set_obstacles(self, shapes):
        self._labels = [sh['label'] for sh in shapes]
        import numpy as np
        self._centers = torch.tensor(
            np.asarray([sh['center'][:2] for sh in shapes], dtype=np.float32),
            device=DEVICE)

    def _per_obstacle(self, x):
        cols = []
        for label, c in zip(self._labels, self._centers):
            rel = x[:, :2] - c
            b_net = self.nets[label](rel / REL_SCALE)
            d = rel.norm(dim=-1, keepdim=True)
            w = ((d - R_IN) / (R_OUT - R_IN)).clamp(0.0, 1.0)
            cols.append((1 - w) * b_net + w * B_FAR)
        return torch.cat(cols, dim=-1)  # (B, M)

    def forward(self, x):
        b = self._per_obstacle(x)
        return -(1.0 / BARRIER_BETA) * torch.logsumexp(
            -BARRIER_BETA * b, dim=-1, keepdim=True)

    def gradient(self, x):
        xr = x.detach().requires_grad_(True)
        B = self.forward(xr)
        return torch.autograd.grad(B.sum(), xr)[0]


class CNCBFStyleMethod(BarrierSwapMethod):
    name = "b5_cncbf_pershape"
    barrier_cls = PerShapeBarrier


def get_methods():
    if not os.path.exists(CKPT):
        raise FileNotFoundError(
            f"{CKPT} missing - run: venv/bin/python "
            f"baselines/b5_cncbf_pershape/train.py")
    return [CNCBFStyleMethod()]
