"""
B4 — classical closed-form CBF on convex primitives.

Each obstacle is replaced by its minimum bounding circle (fit from the same
point cloud every method receives), giving the textbook barrier
b_i(x) = ||x - c_i|| - r_i, composed with the same smooth-min and shaped with
the same K*clip as ours. Guaranteed safe (circle superset of the shape) but
CONSERVATIVE: the enclosing circle of a crescent/star swallows exactly the
concavity the demo threads, so expect blocked corridors / large deviations on
non-convex clutter. This quantifies the value of non-convex learned shapes.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.methods import BarrierSwapMethod  # noqa: E402
from b3_analytic_sdf_cbf.method import _NumpyBarrier, _shape_barrier  # noqa: E402
from config import sample_obstacle_cloud  # noqa: E402
from models import BARRIER_BETA  # noqa: E402


def _min_circle(pts):
    """Ritter's approximate minimum enclosing circle. pts: (K,2) -> (c, r)."""
    p0 = pts[0]
    p1 = pts[np.linalg.norm(pts - p0, axis=1).argmax()]
    p2 = pts[np.linalg.norm(pts - p1, axis=1).argmax()]
    c = 0.5 * (p1 + p2)
    r = 0.5 * np.linalg.norm(p2 - p1)
    for p in pts:
        d = np.linalg.norm(p - c)
        if d > r:
            r = 0.5 * (r + d)
            c = c + (d - r) / (d + 1e-12) * (p - c)
    return c, r


class CircleBarrier(_NumpyBarrier):
    def __init__(self, shapes, k_pts=96, sample_pad=0.002):
        super().__init__()
        self.centers, self.radii = [], []
        for i, sh in enumerate(shapes):
            cloud = sample_obstacle_cloud(sh, k=k_pts, seed=200 + i)
            c_local, r = _min_circle(cloud)
            self.centers.append(np.asarray(sh['center'][:2], np.float64) + c_local)
            # pad by the cloud's sampling spacing so the circle truly encloses
            self.radii.append(r + sample_pad)
        self.centers = np.asarray(self.centers)
        self.radii = np.asarray(self.radii)

    def _B_np(self, xy):
        d = (np.linalg.norm(xy[:, None, :] - self.centers[None], axis=-1)
             - self.radii[None])
        from scipy.special import logsumexp
        b = _shape_barrier(d)  # (N, M) per-circle shaped barrier
        # same smooth-min composition as ours
        return -(1.0 / BARRIER_BETA) * logsumexp(-BARRIER_BETA * b, axis=1)


class CirclePrimitiveMethod(BarrierSwapMethod):
    name = "b4_circle_cbf"
    barrier_cls = CircleBarrier


def get_methods():
    return [CirclePrimitiveMethod()]
