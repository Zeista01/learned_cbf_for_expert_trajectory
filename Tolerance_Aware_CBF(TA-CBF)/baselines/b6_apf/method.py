"""
B6 — artificial potential field (Khatib 1986 style), the classic reactive
alternative to a CBF-QP.

Same nominal DS f_theta as every method (attractive field = demo tracking);
the safety mechanism is the standard repulsive potential

    U_rep = 0.5 * eta * (1/rho - 1/rho0)^2      for rho < rho0
    F_rep = eta * (1/rho - 1/rho0) / rho^2 * grad(rho)

with rho the scene SDF (we grant it the ORACLE SDF — generous: a point-cloud
version would only be worse). No QP, no certificate, no forward-invariance
guarantee. Expected failure modes: local minima in concavities (star/crescent),
oscillation between demo pull and repulsion, tuning sensitivity.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.methods import UnfilteredMethod  # noqa: E402
from generalization_test import sdf_all_np  # noqa: E402


class APFMethod(UnfilteredMethod):
    name = "b6_apf"

    def __init__(self, eta=4e-7, rho0=0.030, rho_min=0.002, f_max=0.15):
        self.eta, self.rho0, self.rho_min, self.f_max = eta, rho0, rho_min, f_max
        self.shapes = None

    def prepare(self, model, shapes):
        self.shapes = shapes

    def _rho(self, xy):
        return float(sdf_all_np(np.asarray(xy, np.float32)[None],
                                self.shapes)[0])

    def filter(self, model, ctrl, x_np, f_val, s):
        rho = max(self._rho(x_np[:2]), self.rho_min)
        info = {'cbf_val': rho}
        if rho >= self.rho0:
            return f_val, info
        e = 1e-4
        grad = np.array([
            self._rho(x_np[:2] + [e, 0]) - self._rho(x_np[:2] - [e, 0]),
            self._rho(x_np[:2] + [0, e]) - self._rho(x_np[:2] - [0, e]),
        ]) / (2 * e)
        mag = self.eta * (1.0 / rho - 1.0 / self.rho0) / rho ** 2
        F = np.zeros(3)
        F[:2] = np.clip(mag, 0.0, None) * grad
        n = np.linalg.norm(F)
        if n > self.f_max:
            F *= self.f_max / n
        return f_val + F, info


def get_methods():
    return [APFMethod()]
