"""
B3 — analytic-SDF CBFs. Two variants of "don't learn the barrier":

  b3a_oracle_sdf   : B(x) = K*clip(sdf(x) - Delta) from the closed-form
                     ground-truth geometry. Safety upper bound — but requires
                     geometry that is NOT available from perception.
  b3b_cloud_esdf   : the deployable version of the same idea — rasterize the
                     raw interior point cloud to an occupancy grid, take a
                     signed Euclidean distance transform, query bilinearly.
                     This is what a skeptic would ship instead of a learned
                     barrier; its staircase gradients are the failure mode our
                     smooth learned field avoids (compare jerk / proj_rate).

Both plug into the SAME CLF-CBF-QP: only model.B changes.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.methods import BarrierSwapMethod  # noqa: E402
from config import (BARRIER_SDF_CLAMP_IN, BARRIER_SDF_CLAMP_OUT,  # noqa: E402
                    BARRIER_SDF_K, INFLATE_MARGIN, SLAB_X, SLAB_Y,
                    sample_obstacle_cloud)
from generalization_test import sdf_all_np  # noqa: E402


def _shape_barrier(sdf):
    """The same K*clip shaping ours regresses onto — identical level sets."""
    return BARRIER_SDF_K * np.clip(sdf - INFLATE_MARGIN,
                                   -BARRIER_SDF_CLAMP_IN, BARRIER_SDF_CLAMP_OUT)


class _NumpyBarrier(nn.Module):
    """Adapter: numpy B(x) field -> the model.B interface the QP expects."""

    def _B_np(self, xy):  # (N,2) -> (N,)
        raise NotImplementedError

    def set_obstacles(self, *a, **k):
        pass

    def forward(self, x_t):
        x = x_t.detach().cpu().numpy().astype(np.float64)
        b = self._B_np(x[:, :2])
        return torch.tensor(b, dtype=torch.float32,
                            device=x_t.device).unsqueeze(-1)

    def gradient(self, x_t):
        x = x_t.detach().cpu().numpy().astype(np.float64)
        e = 1e-4
        g = np.zeros_like(x)
        for d in range(2):  # planar barrier: z component stays 0
            xp = x.copy(); xp[:, d] += e
            xm = x.copy(); xm[:, d] -= e
            g[:, d] = (self._B_np(xp[:, :2]) - self._B_np(xm[:, :2])) / (2 * e)
        return torch.tensor(g, dtype=torch.float32, device=x_t.device)


class OracleSDFBarrier(_NumpyBarrier):
    """Exact geometry (closed-form scene SDF)."""

    def __init__(self, shapes):
        super().__init__()
        self.shapes = shapes

    def _B_np(self, xy):
        return _shape_barrier(sdf_all_np(xy.astype(np.float32), self.shapes))


class CloudESDFBarrier(_NumpyBarrier):
    """SDF estimated from the raw interior point cloud (perception-realistic).

    Interior samples -> occupancy grid (cells within `dilate` of any point are
    occupied, matching the cloud's sampling spacing) -> signed EDT
    (outside distance minus inside distance) -> bilinear query + central-diff
    gradient. No ground-truth geometry anywhere.
    """

    def __init__(self, shapes, res=0.0025, dilate=0.004, k_pts=96, pad=0.03):
        super().__init__()
        from scipy.ndimage import distance_transform_edt

        pts = []
        for i, sh in enumerate(shapes):
            cloud = sample_obstacle_cloud(sh, k=k_pts, seed=100 + i)
            pts.append(cloud + np.asarray(sh['center'][:2], np.float32))
        pts = np.concatenate(pts, axis=0)

        self.res = res
        self.x0, self.y0 = SLAB_X[0] - pad, SLAB_Y[0] - pad
        nx = int((SLAB_X[1] - SLAB_X[0] + 2 * pad) / res) + 1
        ny = int((SLAB_Y[1] - SLAB_Y[0] + 2 * pad) / res) + 1

        point_mask = np.zeros((ny, nx), dtype=bool)
        ix = np.clip(((pts[:, 0] - self.x0) / res).round().astype(int), 0, nx - 1)
        iy = np.clip(((pts[:, 1] - self.y0) / res).round().astype(int), 0, ny - 1)
        point_mask[iy, ix] = True

        dist_to_pts = distance_transform_edt(~point_mask, sampling=res)
        occ = dist_to_pts <= dilate
        d_out = distance_transform_edt(~occ, sampling=res)
        d_in = distance_transform_edt(occ, sampling=res)
        self.B_grid = _shape_barrier(d_out - d_in)
        self.ny, self.nx = ny, nx

    def _B_np(self, xy):
        gx = np.clip((xy[:, 0] - self.x0) / self.res, 0, self.nx - 1.001)
        gy = np.clip((xy[:, 1] - self.y0) / self.res, 0, self.ny - 1.001)
        x0 = gx.astype(int); y0 = gy.astype(int)
        fx = gx - x0; fy = gy - y0
        g = self.B_grid
        return ((1 - fx) * (1 - fy) * g[y0, x0] + fx * (1 - fy) * g[y0, x0 + 1]
                + (1 - fx) * fy * g[y0 + 1, x0] + fx * fy * g[y0 + 1, x0 + 1])


class OracleSDFMethod(BarrierSwapMethod):
    name = "b3a_oracle_sdf"
    barrier_cls = OracleSDFBarrier


class CloudESDFMethod(BarrierSwapMethod):
    name = "b3b_cloud_esdf"
    barrier_cls = CloudESDFBarrier


def get_methods():
    return [OracleSDFMethod(), CloudESDFMethod()]
