"""
run_all.py — evaluate every baseline (plus ours) on the shared scene suite.

Usage:
    venv/bin/python baselines/run_all.py --quick          # smoke test
    venv/bin/python baselines/run_all.py                  # full protocol
    venv/bin/python baselines/run_all.py --only b4_circle_cbf b6_apf

Trained baselines (B5, B7) are skipped with a message until their train.py
has been run; B2 uses the existing checkpoints/fixed_pose_model.pt.
"""
import argparse
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.evaluate import run_protocol  # noqa: E402
from common.methods import OursMethod  # noqa: E402

BASELINE_PKGS = [
    "b1_no_filter",
    "b2_fixed_pose_cbf",
    "b3_analytic_sdf_cbf",
    "b4_convex_primitive_cbf",
    "b5_cncbf_pershape",
    "b6_apf",
    "b7_global_barrier",
]


def collect_methods(only=None):
    methods = [OursMethod()]
    for pkg in BASELINE_PKGS:
        try:
            mod = importlib.import_module(f"{pkg}.method")
            methods.extend(mod.get_methods())
        except FileNotFoundError as e:
            print(f"[skip] {pkg}: {e}")
    if only:
        methods = [m for m in methods if m.name in only]
    return methods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="tiny suite for a smoke test")
    ap.add_argument("--only", nargs="+", default=None,
                    help="method names to run (default: all available)")
    ap.add_argument("--regimes", nargs="+",
                    default=("nominal", "generalization"))
    ap.add_argument("--starts", type=int, default=6)
    ap.add_argument("--per", type=int, default=3,
                    help="generalization scenes per obstacle count")
    ap.add_argument("--out", default="baselines_all")
    args = ap.parse_args()

    methods = collect_methods(args.only)
    print("Methods:", [m.name for m in methods])
    run_protocol(methods, quick=args.quick, regimes=tuple(args.regimes),
                 n_starts=args.starts, per=args.per, out_name=args.out)


if __name__ == "__main__":
    main()
