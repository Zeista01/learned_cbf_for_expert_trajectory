"""
B8 — Neural ODE motion policy at runtime.

Deploys xdot = g_phi(x) open-loop: no CLF, no CBF, no backstop, no obstacle
input. Reproduces the demonstration but cannot react to obstacles that moved,
so on generalization scenes it drives straight through them. This is the
"learned motion alone is unsafe" reference and the NODE (Chen et al.) baseline.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.methods import UnfilteredMethod  # noqa: E402
from b8_node.train import NODEField  # noqa: E402
from config import DEVICE, VEL_CLIP, X_GOAL  # noqa: E402


def _near_goal_blend(v, x_np):
    """Same near-goal assist the runner applies to f_val for every method."""
    d = float(np.linalg.norm(x_np - X_GOAL))
    if d < 0.025:
        a = 1.0 - d / 0.025
        v = (1 - a) * v + a * 0.04 * (X_GOAL - x_np) / (d + 1e-9)
    return v

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "node.pt")

# module-level field cache (robust to the runner caching the model, not the method)
_FIELD = None


def _field():
    global _FIELD
    if _FIELD is None:
        ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
        f = NODEField(ck['mean'], ck['std'], ck['goal']).to(DEVICE)
        f.load_state_dict(ck['state']); f.eval()
        _FIELD = f
    return _FIELD


class NODEMethod(UnfilteredMethod):
    name = "b8_node"

    def make_model(self):
        _field()  # warm the cache
        from simulate import load_model
        return load_model()   # for the runner's call path; velocity overridden below

    def prepare(self, model, shapes):
        pass

    def filter(self, model, ctrl, x_np, f_val, s):
        x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            v = _field()(0.0, x).cpu().numpy().flatten()
        return np.clip(_near_goal_blend(v, x_np), -VEL_CLIP, VEL_CLIP), {}


def get_methods():
    if not os.path.exists(CKPT):
        raise FileNotFoundError(
            f"{CKPT} missing - run: venv/bin/python baselines/b8_node/train.py")
    return [NODEMethod()]
