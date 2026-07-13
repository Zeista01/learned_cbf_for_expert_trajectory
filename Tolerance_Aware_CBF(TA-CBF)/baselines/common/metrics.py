"""
metrics.py — per-rollout metrics and aggregation.

Safety alone rarely separates methods (many read "0 penetrations"); the
discriminators are conservatism (deviation, reach), smoothness (jerk), QP cost,
and how often the exact backstop / discrete projection had to intervene —
a high intervention rate means the barrier itself was NOT doing the work.
"""
import numpy as np

from . import SRC  # noqa: F401
from config import DEMO_WAYPOINTS, DT, INFLATE_MARGIN


def _demo_polyline(n=400):
    wps = DEMO_WAYPOINTS[:, :2]
    ts = np.linspace(0, 1, n) * (len(wps) - 1)
    lo = np.clip(ts.astype(int), 0, len(wps) - 2)
    a = (ts - lo)[:, None]
    return (1 - a) * wps[lo] + a * wps[lo + 1]


_DEMO = _demo_polyline()


def rollout_metrics(log):
    tr = log['ee_pos']
    sdf = log['sdf_true']
    vel = log['vel']

    dev = np.linalg.norm(tr[:, None, :2] - _DEMO[None], axis=-1).min(axis=1)
    m = {
        'reached': bool(log['reached_goal']),
        'time_s': float(log['time'][-1]),
        'path_len_mm': float(np.linalg.norm(np.diff(tr, axis=0), axis=1).sum() * 1e3),
        'min_sdf_mm': float(sdf.min() * 1e3),
        'pen_steps': int((sdf < 0).sum()),
        'lightred_steps': int(((sdf >= 0) & (sdf < INFLATE_MARGIN)).sum()),
        'dev_max_mm': float(dev.max() * 1e3),
        'dev_mean_mm': float(dev.mean() * 1e3),
        'qp_ms_mean': float(np.mean(log['qp_ms'])),
        'proj_rate': float(np.mean(log['proj_active'])),
        'backstop_rate': float(np.mean(log['backstop_active'])),
    }
    if len(vel) >= 3:
        acc = np.diff(vel, axis=0) / DT
        jerk = np.diff(acc, axis=0) / DT
        m['jerk_rms'] = float(np.sqrt((jerk ** 2).sum(axis=1).mean()))
    else:
        m['jerk_rms'] = 0.0
    return m


def aggregate(rows):
    """Aggregate a list of rollout_metrics dicts into one summary row."""
    if not rows:
        return {}
    agg = {
        'n_rollouts': len(rows),
        'reach_rate': float(np.mean([r['reached'] for r in rows])),
        'unsafe_rollouts': int(sum(r['pen_steps'] > 0 for r in rows)),
        'total_pen_steps': int(sum(r['pen_steps'] for r in rows)),
        'min_sdf_mm': float(min(r['min_sdf_mm'] for r in rows)),
    }
    for k in ('dev_max_mm', 'dev_mean_mm', 'path_len_mm', 'time_s',
              'jerk_rms', 'qp_ms_mean', 'proj_rate', 'backstop_rate',
              'lightred_steps'):
        agg[k] = float(np.mean([r[k] for r in rows]))
    return agg
