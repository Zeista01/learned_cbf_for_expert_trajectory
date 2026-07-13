"""
B1 — no safety filter.

  nominal_ds     : the learned progress-conditioned DS alone (tracking without
                   any barrier). Shows demonstrations alone are unsafe once the
                   scene differs from the demo (the "Do Nothing" analog).
  straight_line  : constant-speed straight line to the goal — no learning, no
                   safety. The floor every method must beat.
"""
import numpy as np

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.methods import UnfilteredMethod  # noqa: E402
from config import X_GOAL  # noqa: E402


class NominalDS(UnfilteredMethod):
    name = "b1_nominal_ds"


class StraightLine(UnfilteredMethod):
    name = "b1_straight_line"

    def filter(self, model, ctrl, x_np, f_val, s):
        d = X_GOAL - x_np
        n = np.linalg.norm(d) + 1e-9
        return 0.04 * d / n, {}


def get_methods():
    return [NominalDS(), StraightLine()]
