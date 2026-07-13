"""
runner.py — instrumented closed-loop rollout, identical integration scheme to
src/multi_rollout.run_from_start, plus per-step logging of everything the
metrics need: true SDF, barrier value, QP solve time, discrete-projection and
geometric-backstop activations, applied velocity.
"""
import time

import numpy as np
import torch

from . import SRC  # noqa: F401
from config import (DEVICE, DT, GOAL_TOL, T_MAX, VEL_CLIP, X_GOAL, X_START)
from cbf_qp import analytic_safety_filter
from generalization_test import sdf_all_np


def run_rollout(model, ctrl, method, x_start, shapes, backstop=False):
    """One closed-loop rollout of `method` on the scene `shapes`.

    Returns a log dict with the trajectory and per-step diagnostics. The
    integration, progress schedule, and near-goal blend match
    multi_rollout.run_from_start exactly, so 'ours' reproduces paper rollouts.
    """
    x_np = np.asarray(x_start, dtype=np.float64).copy()
    t, step = 0.0, 0
    max_step = int(T_MAX / DT)
    s_prev = 0.0
    x0_x, xg_x = float(X_START[0]), float(X_GOAL[0])
    T_DEMO = T_MAX * 0.7

    log = {
        'ee_pos': [x_np.copy()], 'time': [0.0], 'vel': [],
        'sdf_true': [], 'B_val': [], 'qp_ms': [],
        'proj_active': [], 'backstop_active': [],
        'reached_goal': False,
    }

    while step < max_step:
        x_t = torch.tensor(x_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        s_x = float(np.clip((x_np[0] - x0_x) / (xg_x - x0_x + 1e-9), 0.0, 1.0))
        s_t = min(1.0, t / T_DEMO)
        s = max(s_prev, 0.5 * (s_x + s_t))
        s_prev = s

        with torch.no_grad():
            f_val = model.f(x_t, s).cpu().numpy().flatten()

        dist_goal = float(np.linalg.norm(x_np - X_GOAL))
        if dist_goal < 0.025:
            alpha_g = 1.0 - dist_goal / 0.025
            goal_pull = 0.04 * (X_GOAL - x_np) / (dist_goal + 1e-9)
            f_val = (1 - alpha_g) * f_val + alpha_g * goal_pull

        t0 = time.perf_counter()
        xdot, info = method.filter(model, ctrl, x_np, f_val, s)
        qp_ms = (time.perf_counter() - t0) * 1e3

        xdot = np.clip(xdot, -VEL_CLIP, VEL_CLIP)

        proj_active = False
        if method.use_projection:
            xdot_p = ctrl.project_safe(x_np.astype(np.float32), model, xdot,
                                       DT, device=DEVICE)
            proj_active = bool(np.linalg.norm(xdot_p - xdot) > 1e-9)
            xdot = xdot_p

        bs_active = False
        if backstop:
            xdot_b = analytic_safety_filter(x_np.astype(np.float32), xdot,
                                            DT, shapes)
            bs_active = bool(np.linalg.norm(np.asarray(xdot_b, dtype=np.float64)
                                            - np.asarray(xdot, dtype=np.float64)) > 1e-9)
            xdot = xdot_b

        x_np = x_np + np.asarray(xdot, dtype=np.float64) * DT
        t += DT
        step += 1

        log['ee_pos'].append(x_np.copy())
        log['time'].append(t)
        log['vel'].append(np.asarray(xdot, dtype=np.float64).copy())
        log['sdf_true'].append(float(sdf_all_np(x_np[None], shapes)[0]))
        log['B_val'].append(float(info.get('cbf_val', np.nan)))
        log['qp_ms'].append(qp_ms)
        log['proj_active'].append(proj_active)
        log['backstop_active'].append(bs_active)

        if np.linalg.norm(x_np - X_GOAL) < GOAL_TOL:
            log['reached_goal'] = True
            break

    for k in ('ee_pos', 'time', 'vel', 'sdf_true', 'B_val', 'qp_ms'):
        log[k] = np.asarray(log[k])
    return log
