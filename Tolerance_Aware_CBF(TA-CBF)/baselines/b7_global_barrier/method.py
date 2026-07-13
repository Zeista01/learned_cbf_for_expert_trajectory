"""
B7 — global scene-frozen barrier (S2-NNDS-style stand-in) at runtime.

The trained BarrierNet ignores the current scene entirely (set_obstacles is a
no-op): it answers with the NOMINAL scene's barrier wherever the obstacles
actually are. Expected: matches ours on the nominal regime, fails on any scene
whose obstacles moved — the structural limitation of scene-specific
certificates that our conditioning + augmentation removes.
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.methods import BarrierSwapMethod  # noqa: E402
from config import DEVICE  # noqa: E402
from models import BarrierNet  # noqa: E402

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "checkpoints", "global_barrier.pt")


class GlobalBarrier(nn.Module):
    """model.B interface around the scene-frozen BarrierNet."""

    def __init__(self, shapes=None):  # shapes intentionally unused
        super().__init__()
        self.net = BarrierNet()
        self.net.load_state_dict(torch.load(CKPT, map_location=DEVICE))
        self.net.to(DEVICE).eval()

    def set_obstacles(self, *a, **k):
        pass

    def forward(self, x):
        return self.net(x)

    def gradient(self, x):
        return self.net.gradient(x)


class GlobalBarrierMethod(BarrierSwapMethod):
    name = "b7_global_barrier"
    barrier_cls = GlobalBarrier


def get_methods():
    if not os.path.exists(CKPT):
        raise FileNotFoundError(
            f"{CKPT} missing - run: venv/bin/python "
            f"baselines/b7_global_barrier/train.py")
    return [GlobalBarrierMethod()]
