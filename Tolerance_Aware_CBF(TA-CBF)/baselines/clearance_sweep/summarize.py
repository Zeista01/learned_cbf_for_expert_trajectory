"""
summarize.py — assemble the clearance sweep into a reach-vs-clearance story.

For each clearance run (clrNNmm.json) it reports, on the generalization regime
with the backstop ON (the deployable, strict-safety pipeline):
  - reach rate (did the tool arrive at the target — the surgical requirement)
  - unsafe scenes (must stay 0)
  - backstop activation rate (how hard the exact filter had to fight the barrier)
Plus the 15 mm baseline from ../as_it_is for the full curve, and a matplotlib
plot reach-vs-clearance for ours and the key baselines.
"""
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ASIS = os.path.join(HERE, "..", "as_it_is", "baselines_full.json")

FOCUS = ["ours_ta_cbf", "b2_fixed_pose_cbf", "b3a_oracle_sdf",
         "b3b_cloud_esdf", "b4_circle_cbf", "b5_cncbf_pershape"]
REGIME = "generalization_backstop_on"


def load_points():
    """clearance_mm -> {method -> summary dict}."""
    pts = {}
    if os.path.exists(ASIS):
        pts[15] = json.load(open(ASIS))
    for f in sorted(glob.glob(os.path.join(HERE, "clr*mm.json"))):
        mm = int(re.search(r"clr(\d+)mm", f).group(1))
        pts[mm] = json.load(open(f))
    return dict(sorted(pts.items()))


def summarize(pts):
    lines = ["# Clearance sweep — generalization, backstop ON\n",
             "Reach = fraction of scenes where the tool arrived at the target.",
             "Safety must remain 0 unsafe at every clearance.\n"]
    for mm, data in pts.items():
        n = len(data["ours_ta_cbf"][REGIME]["rows"])
        lines.append(f"\n## clearance {mm} mm  ({n} generalization scenes)\n")
        lines.append(f"| method | reach | unsafe | min_sdf_mm | dev_max_mm | "
                     f"backstop_rate |")
        lines.append("|---|---|---|---|---|---|")
        for m in FOCUS:
            if m not in data:
                continue
            s = data[m][REGIME]["summary"]
            lines.append(f"| {m} | {s['reach_rate']:.2f} | "
                         f"{s['unsafe_rollouts']}/{s['n_rollouts']} | "
                         f"{s['min_sdf_mm']:.1f} | {s['dev_max_mm']:.1f} | "
                         f"{s['backstop_rate']:.3f} |")
    out = os.path.join(HERE, "SUMMARY.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved -> {out}")


def plot(pts):
    xs = sorted(pts)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for m in FOCUS:
        reach = [pts[mm][m][REGIME]["summary"]["reach_rate"]
                 if m in pts[mm] else np.nan for mm in xs]
        bs = [pts[mm][m][REGIME]["summary"]["backstop_rate"]
              if m in pts[mm] else np.nan for mm in xs]
        lw = 3 if m == "ours_ta_cbf" else 1.5
        ax1.plot(xs, reach, "-o", lw=lw, label=m)
        ax2.plot(xs, bs, "-o", lw=lw, label=m)
    ax1.set_xlabel("guaranteed passage clearance [mm]")
    ax1.set_ylabel("reach rate (arrived at target)")
    ax1.set_title("Reachability vs clearance (backstop ON)")
    ax1.set_ylim(-0.02, 1.02); ax1.grid(alpha=0.3); ax1.legend(fontsize=8)
    ax2.set_xlabel("guaranteed passage clearance [mm]")
    ax2.set_ylabel("backstop activation rate")
    ax2.set_title("How hard the exact backstop had to fight the barrier")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(HERE, "reach_vs_clearance.png")
    plt.savefig(out, dpi=150)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    pts = load_points()
    if not pts:
        raise SystemExit("no sweep results found yet")
    summarize(pts)
    plot(pts)
