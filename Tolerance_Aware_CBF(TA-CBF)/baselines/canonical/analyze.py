"""
analyze.py — the canonical benchmark breakdown (25 mm clearance, 3-8 obstacles).

Produces the tables the paper reports:
  - SAFETY headline (backstop ON): penetrations per method (must be 0 for ours)
  - AUGMENTATION ablation (backstop OFF): ours vs fixed-pose, per density
  - REACH per obstacle count (backstop ON): honest degradation with density
Writes TABLES.md next to this script and prints them.
"""
import json
import os
import re
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "benchmark_25mm.json")))
COUNTS = (3, 4, 5, 6, 7, 8)


def nobs(scene):
    m = re.search(r"gen_(\d+)obs", scene)
    return int(m.group(1)) if m else None


def by_count(method, key, field, reduce):
    rows = d[method][key]["rows"]
    b = defaultdict(list)
    for r in rows:
        b[nobs(r["scene"])].append(r[field])
    return {n: reduce(b[n]) for n in COUNTS if b[n]}, rows


out = ["# Canonical benchmark — 25 mm clearance, 3-8 obstacles\n"]

# 1. Safety headline
out.append("## Safety (generalization, backstop ON) — penetrating scenes\n")
out.append("| method | penetrating | min_sdf_mm |")
out.append("|---|---|---|")
for m in d:
    rows = d[m]["generalization_backstop_on"]["rows"]
    u = sum(r["pen_steps"] > 0 for r in rows)
    out.append(f"| {m} | {u}/{len(rows)} | "
               f"{min(r['min_sdf_mm'] for r in rows):+.1f} |")

# 2. Augmentation ablation (backstop OFF)
out.append("\n## Augmentation ablation (backstop OFF) — unsafe scenes by #obstacles\n")
out.append("| method | " + " | ".join(f"{n}obs" for n in COUNTS) + " | total |")
out.append("|" + "---|" * (len(COUNTS) + 2))
for m in ("ours_ta_cbf", "b2_fixed_pose_cbf", "b7_global_barrier"):
    per, rows = by_count(m, "generalization_backstop_off", "pen_steps",
                         lambda v: sum(x > 0 for x in v))
    cnt, _ = by_count(m, "generalization_backstop_off", "pen_steps", len)
    cells = " | ".join(f"{per.get(n,0)}/{cnt.get(n,0)}" for n in COUNTS)
    tot = sum(r["pen_steps"] > 0 for r in rows)
    out.append(f"| {m} | {cells} | {tot}/{len(rows)} |")

# 3. Reach per density (backstop ON)
out.append("\n## Reach per #obstacles (generalization, backstop ON)\n")
out.append("| method | " + " | ".join(f"{n}obs" for n in COUNTS) + " | all |")
out.append("|" + "---|" * (len(COUNTS) + 2))
for m in ("ours_ta_cbf", "b2_fixed_pose_cbf", "b3a_oracle_sdf",
          "b3b_cloud_esdf", "b4_circle_cbf", "b5_cncbf_pershape"):
    per, rows = by_count(m, "generalization_backstop_on", "reached", np.mean)
    cells = " | ".join(f"{per.get(n,float('nan')):.2f}" for n in COUNTS)
    allr = np.mean([r["reached"] for r in rows])
    out.append(f"| {m} | {cells} | {allr:.2f} |")

# 4. Nominal (the paper's tracking regime)
out.append("\n## Nominal (perturbed starts, backstop ON)\n")
out.append("| method | reach | unsafe | min_sdf_mm | dev_max_mm |")
out.append("|---|---|---|---|---|")
for m in ("ours_ta_cbf", "b2_fixed_pose_cbf", "b3a_oracle_sdf", "b6_apf"):
    s = d[m]["nominal_backstop_on"]["summary"]
    out.append(f"| {m} | {s['reach_rate']:.2f} | "
               f"{s['unsafe_rollouts']}/{s['n_rollouts']} | "
               f"{s['min_sdf_mm']:.1f} | {s['dev_max_mm']:.1f} |")

text = "\n".join(out) + "\n"
with open(os.path.join(HERE, "TABLES.md"), "w") as f:
    f.write(text)
print(text)
print(f"Saved -> {os.path.join(HERE, 'TABLES.md')}")
