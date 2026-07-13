"""
B9 — S2-NNDS at runtime: deploy the certified DS directly, xdot = f_theta(x).

No online filter, no backstop, no obstacle input — exactly as the method
prescribes. Safe on the training scene (f was shaped by the co-trained barrier to
avoid the NOMINAL obstacles); on generalization scenes the obstacles moved but f
still avoids the OLD locations, so it drives into the new ones. This is the
scene-frozen limitation our conditioning + augmentation removes.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.methods import UnfilteredMethod  # noqa: E402
from b9_s2nnds.train import S2DS  # noqa: E402
from config import DEVICE, VEL_CLIP, X_GOAL  # noqa: E402


def _near_goal_blend(v, x_np):
    """Same near-goal assist the runner applies to f_val for every method."""
    d = float(np.linalg.norm(x_np - X_GOAL))
    if d < 0.025:
        a = 1.0 - d / 0.025
        v = (1 - a) * v + a * 0.04 * (X_GOAL - x_np) / (d + 1e-9)
    return v

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "s2nnds.pt")

_F = None


def _ds():
    global _F
    if _F is None:
        ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
        f = S2DS(ck['mean'], ck['std'], ck['goal']).to(DEVICE)
        f.load_state_dict(ck['f']); f.eval()
        _F = f
    return _F


class S2NNDSMethod(UnfilteredMethod):
    name = "b9_s2nnds"

    def make_model(self):
        _ds()
        from simulate import load_model
        return load_model()   # for the runner's call path; velocity overridden below

    def prepare(self, model, shapes):
        pass

    def filter(self, model, ctrl, x_np, f_val, s):
        x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            v = _ds()(x).cpu().numpy().flatten()
        return np.clip(_near_goal_blend(v, x_np), -VEL_CLIP, VEL_CLIP), {}


def get_methods():
    if not os.path.exists(CKPT):
        raise FileNotFoundError(
            f"{CKPT} missing - run: venv/bin/python baselines/b9_s2nnds/train.py")
    return [S2NNDSMethod()]
