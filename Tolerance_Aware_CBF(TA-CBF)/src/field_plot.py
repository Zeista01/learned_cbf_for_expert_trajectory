"""
field_plot.py — shared rendering of the closed-loop safety field.

Draws the field  ẋ = f_θ(x) + u_safe(x)  where u_safe is the CBF-QP correction,
so the streamlines:
  • diverge AROUND each critical zone (because the learned, inflated CBF pushes
    the velocity tangent to the B=0 contour), then
  • re-converge ONTO the demo path (because the progress-conditioned DS f_θ has
    the whole demo trajectory as an attractor, with the CLF keeping tracking).

Used by train.py (final field plot), generalization_test.py and
dynamic_env_test.py so every figure shows the same divert-then-reconverge field.
"""

import numpy as np
import torch

from config import (DEVICE, SLAB_X, SLAB_Y, Z_CORRIDOR, X_START, X_GOAL,
                    INFLATE_MARGIN, B_SAFE_MARGIN, sdf_critical_shape_2d)
from cbf_qp import BPCBFController


def _grid(nx, ny, pad=0.004):
    xs = np.linspace(SLAB_X[0] + pad, SLAB_X[1] - pad, nx)
    ys = np.linspace(SLAB_Y[0] + pad, SLAB_Y[1] - pad, ny)
    return xs, ys


def _safety_field(model, ctrl, xs, ys):
    """Evaluate ẋ = f_θ + u_safe on the (xs × ys) grid. Returns U, V (ny, nx)."""
    ny, nx = len(ys), len(xs)
    U = np.zeros((ny, nx), np.float32)
    V = np.zeros((ny, nx), np.float32)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            x_np = np.array([x, y, Z_CORRIDOR], dtype=np.float32)
            x_t  = torch.tensor(x_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                s_val = model.f.get_progress(x_t)
                f_val = model.f(x_t, s_val).cpu().numpy().flatten()
            u_safe, _ = ctrl.solve(x_np, model, device=DEVICE)
            v = f_val + u_safe
            U[j, i], V[j, i] = v[0], v[1]
    return U, V


def _sdf_grid(shapes, xs, ys):
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)
    sdfs = [sdf_critical_shape_2d(pts, sh) for sh in shapes]
    return np.min(np.stack(sdfs, axis=1), axis=1).reshape(len(ys), len(xs))


def plot_diverting_field(ax, model, shapes, ctrl=None,
                         nx=26, ny=22, stream_density=1.3,
                         show_demo=True, title=None):
    """
    Render the divert-then-reconverge safety field on `ax` for the obstacle
    `shapes` currently installed on `model.B`. Caller is responsible for having
    called model.set_obstacles(...) for the SAME `shapes`.
    """
    if ctrl is None:
        ctrl = BPCBFController()

    xs, ys = _grid(nx, ny)
    XX, YY = np.meshgrid(xs, ys)
    U, V = _safety_field(model, ctrl, xs, ys)
    speed = np.sqrt(U**2 + V**2) + 1e-9
    Un, Vn = U / speed, V / speed

    # critical-zone fill + inflated learned-barrier intent (red core, pink shell)
    sdf_c = _sdf_grid(shapes, xs, ys)
    ax.contourf(xs, ys, sdf_c, levels=[-10, 0], colors=['#ff3333'], alpha=0.55, zorder=2)
    ax.contourf(xs, ys, sdf_c, levels=[0, INFLATE_MARGIN], colors=['#ffb0b0'], alpha=0.35, zorder=2)
    ax.contour(xs, ys, sdf_c, levels=[0], colors=['#aa0000'], linewidths=1.2, zorder=3)

    # learned B=0 contour (the actual safety boundary the field respects)
    pts3 = np.stack([XX.ravel(), YY.ravel(),
                     np.full(XX.size, Z_CORRIDOR, np.float32)], axis=1).astype(np.float32)
    with torch.no_grad():
        B = model.B(torch.tensor(pts3, device=DEVICE)).squeeze(-1).cpu().numpy().reshape(ny, nx)
    ax.contour(xs, ys, B, levels=[0], colors=['#7000a0'], linewidths=2.0,
               linestyles='--', zorder=6)

    # MASK the field inside the keep-out set {B < margin}: the controller is never
    # evaluated there (the needle never enters), and masking forces streamlines to
    # route AROUND the light-red zone instead of being drawn straight through it.
    keepout = B < B_SAFE_MARGIN
    Un = np.ma.masked_where(keepout, Un)
    Vn = np.ma.masked_where(keepout, Vn)
    ax.streamplot(xs, ys, Un, Vn, color='steelblue',
                  linewidth=0.8, density=stream_density, arrowsize=0.9, zorder=4)

    if show_demo:
        ref = model.f.ref_path.cpu().numpy()
        ax.plot(ref[:, 0], ref[:, 1], 'k-', lw=2.2, label='Demo path (attractor)', zorder=8)
    ax.scatter(X_START[0], X_START[1], c='blue', s=90, zorder=9, label='Start')
    ax.scatter(X_GOAL[0],  X_GOAL[1],  c='lime', s=110, marker='*', zorder=9, label='Goal')

    ax.set_xlim(SLAB_X); ax.set_ylim(SLAB_Y)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc='upper left')
    return ax
