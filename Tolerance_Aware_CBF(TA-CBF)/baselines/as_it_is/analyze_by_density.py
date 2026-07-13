"""Breakdown of the as-is baseline results by obstacle count.
Reads baselines_full.json sitting next to this script."""
import json, os, re
import numpy as np
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "baselines_full.json")))


def nobs(scene):
    m = re.search(r"gen_(\d+)obs", scene)
    return int(m.group(1)) if m else None


methods = ["ours_ta_cbf", "b2_fixed_pose_cbf", "b3a_oracle_sdf",
           "b3b_cloud_esdf", "b4_circle_cbf", "b5_cncbf_pershape"]

print("=== GENERALIZATION, BACKSTOP ON: reach rate by #obstacles ===")
print(f"{'method':<22}" + "".join(f"{n:>8}obs" for n in (3, 4, 5, 6)) + f"{'all':>9}")
for m in methods:
    rows = d[m]["generalization_backstop_on"]["rows"]
    by = defaultdict(list)
    for r in rows:
        by[nobs(r["scene"])].append(r["reached"])
    cells = "".join(f"{np.mean(by[n]):>11.2f}" for n in (3, 4, 5, 6))
    print(f"{m:<22}{cells}{np.mean([r['reached'] for r in rows]):>9.2f}")

print("\n=== GENERALIZATION, BACKSTOP OFF: unsafe scenes by #obstacles "
      "(learned barrier alone) ===")
print(f"{'method':<22}" + "".join(f"{n:>8}obs" for n in (3, 4, 5, 6)) + f"{'all':>9}")
for m in ["ours_ta_cbf", "b2_fixed_pose_cbf", "b5_cncbf_pershape",
          "b7_global_barrier"]:
    rows = d[m]["generalization_backstop_off"]["rows"]
    by = defaultdict(list)
    for r in rows:
        by[nobs(r["scene"])].append(r["pen_steps"] > 0)
    cells = "".join(f"{sum(by[n]):>6}/{len(by[n])} " for n in (3, 4, 5, 6))
    print(f"{m:<22}{cells}{sum(r['pen_steps']>0 for r in rows):>6}/{len(rows)}")

print("\n=== SAFETY (backstop ON): penetrating scenes / total ===")
for m in d:
    rows = d[m]["generalization_backstop_on"]["rows"]
    u = sum(r["pen_steps"] > 0 for r in rows)
    print(f"  {m:<22} {u}/{len(rows)} penetrate   "
          f"min_sdf {min(r['min_sdf_mm'] for r in rows):+.1f}mm")
