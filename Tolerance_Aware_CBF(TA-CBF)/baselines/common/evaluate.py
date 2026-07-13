"""
evaluate.py — shared evaluation driver: run a list of Methods over the fixed
scene suite in both regimes, write JSON, print a comparison table.

Protocol (mirrors the paper's two experiments):
  EXP A  backstop OFF — the method's own safety mechanism alone.
  EXP B  backstop ON  — full pipeline; backstop_rate discriminates barrier quality.
"""
import json
import os
import traceback

import numpy as np

from . import RESULTS_DIR, SRC  # noqa: F401
from .metrics import aggregate, rollout_metrics
from .runner import run_rollout
from .scenes import generalization_suite, nominal_scene, nominal_starts
from config import X_START


def evaluate_method(method, scenes, starts, backstop):
    """Run `method` on every (scene, start) pair. Returns (summary, rows)."""
    from cbf_qp import BPCBFController
    model = method.make_model()
    ctrl = BPCBFController()
    rows = []
    for scene_name, shapes in scenes:
        method.prepare(model, shapes)
        for x0 in starts:
            log = run_rollout(model, ctrl, method, x0, shapes, backstop=backstop)
            r = rollout_metrics(log)
            r['scene'] = scene_name
            rows.append(r)
    return aggregate(rows), rows


def run_protocol(methods, quick=False, regimes=('nominal', 'generalization'),
                 backstops=(False, True), out_name='baselines_all',
                 n_starts=6, per=3, n_obs_list=(3, 4, 5, 6)):
    if quick:
        n_starts, per, n_obs_list = 2, 1, (3, 5)

    suites = {}
    if 'nominal' in regimes:
        suites['nominal'] = ([('nominal', nominal_scene())],
                             nominal_starts(n_starts))
    if 'generalization' in regimes:
        suites['generalization'] = (generalization_suite(n_obs_list, per=per),
                                    [X_START.copy()])

    results = {}
    for method in methods:
        results[method.name] = {}
        for regime, (scenes, starts) in suites.items():
            for bs in backstops:
                key = f"{regime}_backstop_{'on' if bs else 'off'}"
                print(f"[{method.name}] {key} "
                      f"({len(scenes)} scenes x {len(starts)} starts) ...",
                      flush=True)
                try:
                    summary, rows = evaluate_method(method, scenes, starts, bs)
                    results[method.name][key] = {'summary': summary, 'rows': rows}
                    print(f"    reach {summary['reach_rate']:.2f}  "
                          f"unsafe {summary['unsafe_rollouts']}/{summary['n_rollouts']}  "
                          f"min_sdf {summary['min_sdf_mm']:.1f}mm  "
                          f"dev_max {summary['dev_max_mm']:.1f}mm  "
                          f"backstop {summary['backstop_rate']:.3f}", flush=True)
                except Exception:
                    print(f"    FAILED:\n{traceback.format_exc()}", flush=True)
                    results[method.name][key] = {'error': traceback.format_exc()}

    out = os.path.join(RESULTS_DIR, f"{out_name}.json")
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out}")
    print_table(results)
    return results


def print_table(results):
    cols = ('reach_rate', 'unsafe_rollouts', 'total_pen_steps', 'min_sdf_mm',
            'dev_max_mm', 'jerk_rms', 'qp_ms_mean', 'backstop_rate')
    keys = sorted({k for m in results.values() for k in m})
    for key in keys:
        print(f"\n=== {key} ===")
        hdr = f"{'method':<26}" + "".join(f"{c:>16}" for c in cols)
        print(hdr)
        for name, per_key in results.items():
            entry = per_key.get(key)
            if not entry or 'summary' not in entry:
                print(f"{name:<26}{'ERROR':>16}")
                continue
            s = entry['summary']
            vals = "".join(
                f"{s[c]:>16.3f}" if isinstance(s[c], float) else f"{s[c]:>16}"
                for c in cols)
            print(f"{name:<26}{vals}")
