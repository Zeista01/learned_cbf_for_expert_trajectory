"""
run_full.py — the FULL, correct baseline protocol, parallelized across CPU cores.

Why CPU, not GPU: each rollout is a batch-size-1 sequential loop whose per-step
cost is dominated by the OSQP solve and numpy SDF / backstop finite-differences
(both CPU-only). Measured: CPU is ~1.45x faster per rollout than CUDA here, and
CPU-only lets us fork worker processes cleanly (CUDA does not survive fork). We
therefore pin CPU and parallelize across rollouts — the real speedup.

Correctness: rollouts are deterministic (no RNG inside the loop; scenes are
generated ONCE in the parent and passed explicitly), so parallel results are
identical to sequential. `--verify` checks this on a subset before the full run.

Usage:
    venv/bin/python baselines/run_full.py                 # full protocol
    venv/bin/python baselines/run_full.py --verify        # parallel==serial check
    venv/bin/python baselines/run_full.py --starts 8 --per 6 --workers 11
"""
# Pin CPU BEFORE importing torch (must precede any torch import, incl. via common)
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")      # 1 thread/worker: we parallelize
os.environ.setdefault("MKL_NUM_THREADS", "1")      # across rollouts, not within BLAS
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import importlib
import json
import multiprocessing as mp
import time
import traceback

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import RESULTS_DIR
from common.methods import OursMethod
from common.metrics import aggregate, rollout_metrics
from common.runner import run_rollout
from common.scenes import generalization_suite, nominal_scene, nominal_starts
from common.evaluate import print_table
from config import X_START, DEVICE

BASELINE_PKGS = [
    "b1_no_filter", "b2_fixed_pose_cbf", "b3_analytic_sdf_cbf",
    "b4_convex_primitive_cbf", "b5_cncbf_pershape", "b6_apf",
    "b7_global_barrier", "b8_node", "b9_s2nnds",
]

# per-process cache: method.name -> (model, ctrl). Avoids reloading a method's
# checkpoint for every rollout that process handles.
_CACHE = {}


def _get_model_ctrl(method):
    if method.name not in _CACHE:
        from cbf_qp import BPCBFController
        _CACHE[method.name] = (method.make_model(), BPCBFController())
    return _CACHE[method.name]


def _worker(task):
    """Run one rollout. task = (method, key, scene_name, shapes, start, backstop)."""
    method, key, scene_name, shapes, start, backstop = task
    try:
        model, ctrl = _get_model_ctrl(method)
        method.prepare(model, shapes)
        log = run_rollout(model, ctrl, method, start, shapes, backstop=backstop)
        r = rollout_metrics(log)
        r['scene'] = scene_name
        return (method.name, key, r, None)
    except Exception:
        return (method.name, key, None, traceback.format_exc())


def collect_methods(only=None):
    methods = [OursMethod()]
    for pkg in BASELINE_PKGS:
        try:
            mod = importlib.import_module(f"{pkg}.method")
            methods.extend(mod.get_methods())
        except FileNotFoundError as e:
            print(f"[skip] {pkg}: {e}", flush=True)
    if only:
        methods = [m for m in methods if m.name in only]
    return methods


def build_tasks(methods, starts, gen_scenes):
    """One task per (method, regime, scene/start, backstop)."""
    nominal = ('nominal', nominal_scene())
    tasks = []
    for m in methods:
        for backstop in (False, True):
            bs = 'on' if backstop else 'off'
            for x0 in starts:  # nominal: 1 scene, many perturbed starts
                tasks.append((m, f"nominal_backstop_{bs}", 'nominal',
                              nominal[1], x0, backstop))
            for name, shapes in gen_scenes:  # generalization: many scenes, 1 start
                tasks.append((m, f"generalization_backstop_{bs}", name,
                              shapes, X_START.copy(), backstop))
    return tasks


def run(tasks, workers):
    rows = {}   # (method, key) -> [metrics]
    errors = []
    t0 = time.perf_counter()
    done = 0
    total = len(tasks)
    with mp.Pool(workers) as pool:
        for name, key, r, err in pool.imap_unordered(_worker, tasks, chunksize=1):
            done += 1
            if err is not None:
                errors.append((name, key, err))
            else:
                rows.setdefault((name, key), []).append(r)
            if done % 25 == 0 or done == total:
                el = time.perf_counter() - t0
                print(f"  {done}/{total} rollouts  ({el:.0f}s, "
                      f"{el / done:.2f}s/rollout avg, {len(errors)} errors)",
                      flush=True)
    return rows, errors


def assemble(methods, rows):
    results = {}
    for m in methods:
        results[m.name] = {}
        for key in sorted({k for (nm, k) in rows if nm == m.name}):
            rlist = rows[(m.name, key)]
            results[m.name][key] = {'summary': aggregate(rlist), 'rows': rlist}
    return results


def write_markdown(results, path):
    cols = [('reach_rate', 'reach', '{:.2f}'),
            ('unsafe_rollouts', 'unsafe', '{:.0f}'),
            ('total_pen_steps', 'pen_steps', '{:.0f}'),
            ('min_sdf_mm', 'min_sdf', '{:.1f}'),
            ('dev_max_mm', 'dev_max', '{:.1f}'),
            ('jerk_rms', 'jerk', '{:.0f}'),
            ('qp_ms_mean', 'qp_ms', '{:.2f}'),
            ('backstop_rate', 'bkstop', '{:.3f}')]
    keys = sorted({k for m in results.values() for k in m})
    lines = ["# Baseline results\n"]
    for key in keys:
        lines.append(f"\n## {key}\n")
        lines.append("| method | " + " | ".join(c[1] for c in cols) + " |")
        lines.append("|" + "---|" * (len(cols) + 1))
        for name, per_key in results.items():
            e = per_key.get(key)
            if not e or 'summary' not in e:
                continue
            s = e['summary']
            cells = [name] + [c[2].format(s[c[0]]) for c in cols]
            lines.append("| " + " | ".join(cells) + " |")
    with open(path, 'w') as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts", type=int, default=8,
                    help="perturbed starts for the nominal scene")
    ap.add_argument("--per", type=int, default=6,
                    help="generalization scenes per obstacle count")
    ap.add_argument("--n_obs", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--out", default="baselines_full")
    ap.add_argument("--outdir", default=RESULTS_DIR,
                    help="directory for the JSON/MD outputs")
    ap.add_argument("--clearance", type=float, default=None,
                    help="guaranteed start->goal passage (m); None=project default")
    ap.add_argument("--verify", action="store_true",
                    help="check parallel==serial on a small subset, then exit")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    assert DEVICE == "cpu", f"expected CPU, got {DEVICE} (CUDA_VISIBLE_DEVICES?)"
    print(f"[run_full] DEVICE={DEVICE}  workers={args.workers}", flush=True)

    methods = collect_methods(args.only)
    print("[run_full] methods:", [m.name for m in methods], flush=True)

    # Generate scenes ONCE (shared by every method for a controlled comparison).
    gen_scenes = generalization_suite(tuple(args.n_obs), per=args.per,
                                      clearance=args.clearance)
    starts = nominal_starts(args.starts)
    clr_mm = "default" if args.clearance is None else f"{args.clearance*1e3:.0f}mm"
    print(f"[run_full] {len(gen_scenes)} generalization scenes "
          f"(clearance={clr_mm}), {len(starts)} nominal starts", flush=True)

    if args.verify:
        return verify(methods[:3], starts[:2], gen_scenes[:2], args.workers)

    tasks = build_tasks(methods, starts, gen_scenes)
    print(f"[run_full] {len(tasks)} total rollouts\n", flush=True)
    rows, errors = run(tasks, args.workers)
    results = assemble(methods, rows)

    out_json = os.path.join(args.outdir, f"{args.out}.json")
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    write_markdown(results, os.path.join(args.outdir, f"{args.out}.md"))

    print(f"\nSaved -> {out_json}")
    print(f"Saved -> {os.path.join(args.outdir, args.out + '.md')}")
    if errors:
        print(f"\n!! {len(errors)} rollout errors. First:\n{errors[0][2]}")
    print_table(results)


def verify(methods, starts, gen_scenes, workers):
    """Prove parallel == serial: run the same subset both ways, compare metrics."""
    tasks = build_tasks(methods, starts, gen_scenes)
    print(f"[verify] {len(tasks)} rollouts, serial vs {workers} workers")

    ser = {}
    for t in tasks:
        name, key, r, err = _worker(t)
        assert err is None, err
        ser[(name, key, r['scene'], round(r['dev_max_mm'], 6))] = r
    par_rows, _ = run(tasks, workers)

    mism = 0
    for (nm, key), rlist in par_rows.items():
        for r in rlist:
            k = (nm, key, r['scene'], round(r['dev_max_mm'], 6))
            if k not in ser:
                mism += 1
                continue
            s = ser[k]
            for field in ('reached', 'min_sdf_mm', 'pen_steps', 'backstop_rate',
                          'path_len_mm'):
                if abs(float(s[field]) - float(r[field])) > 1e-6:
                    print(f"  MISMATCH {nm} {key} {r['scene']} {field}: "
                          f"{s[field]} vs {r[field]}")
                    mism += 1
    print(f"[verify] {'OK — parallel == serial' if mism == 0 else f'{mism} MISMATCHES'}")


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
