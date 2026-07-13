"""Final canonical-benchmark report: 25mm clearance, 3-8 obstacles, per-density."""
import json, re, numpy as np
from collections import defaultdict
d = json.load(open("baselines/canonical/benchmark_25mm.json"))
NOBS=(3,4,5,6,7,8)
def nobs(s):
    m=re.search(r"gen_(\d+)obs",s); return int(m.group(1)) if m else None
def rows(m,k): return d[m][k]["rows"]
def agg(m,k): return d[m][k]["summary"]
ALL=list(d.keys())

print("="*92)
print("CANONICAL BENCHMARK  |  25mm clearance  |  3-8 obstacles x6 = 36 gen scenes + 10 nominal starts")
print("="*92)

print("\n[1] SAFETY HEADLINE — penetrating scenes (any step inside a critical region)")
print(f"{'method':<22}{'nominal off':>13}{'gen off':>12}{'nominal on':>13}{'gen on':>10}")
for m in ALL:
    def unsafe(k): return sum(r['pen_steps']>0 for r in rows(m,k))
    def tot(k): return len(rows(m,k))
    print(f"{m:<22}{unsafe('nominal_backstop_off'):>6}/{tot('nominal_backstop_off'):<6}"
          f"{unsafe('generalization_backstop_off'):>5}/{tot('generalization_backstop_off'):<6}"
          f"{unsafe('nominal_backstop_on'):>6}/{tot('nominal_backstop_on'):<6}"
          f"{unsafe('generalization_backstop_on'):>4}/{tot('generalization_backstop_on'):<5}")

print("\n[2] AUGMENTATION ABLATION — learned barrier ALONE (backstop OFF), unsafe scenes by #obstacles")
print(f"{'method':<22}"+"".join(f"{n:>7}obs" for n in NOBS)+f"{'total':>10}")
for m in ("ours_ta_cbf","b2_fixed_pose_cbf","b5_cncbf_pershape","b7_global_barrier"):
    by=defaultdict(list)
    for r in rows(m,"generalization_backstop_off"): by[nobs(r['scene'])].append(r['pen_steps']>0)
    cells="".join(f"{sum(by[n]):>4}/{len(by[n])} " for n in NOBS)
    tot=sum(r['pen_steps']>0 for r in rows(m,"generalization_backstop_off"))
    print(f"{m:<22}{cells}{tot:>6}/36")

print("\n[3] REACH RATE (backstop ON, full pipeline) by #obstacles")
print(f"{'method':<22}"+"".join(f"{n:>7}obs" for n in NOBS)+f"{'all':>8}")
for m in ("ours_ta_cbf","b2_fixed_pose_cbf","b3a_oracle_sdf","b3b_cloud_esdf","b4_circle_cbf","b5_cncbf_pershape"):
    by=defaultdict(list)
    for r in rows(m,"generalization_backstop_on"): by[nobs(r['scene'])].append(r['reached'])
    cells="".join(f"{np.mean(by[n]):>10.2f}" for n in NOBS)
    allr=np.mean([r['reached'] for r in rows(m,"generalization_backstop_on")])
    print(f"{m:<22}{cells}{allr:>8.2f}")

print("\n[4] CONSERVATISM & COST (generalization, backstop ON)")
print(f"{'method':<22}{'reach':>7}{'devmax_mm':>11}{'jerk':>9}{'qp_ms':>8}{'bkstop%':>9}")
for m in ALL:
    s=agg(m,"generalization_backstop_on")
    print(f"{m:<22}{s['reach_rate']:>7.2f}{s['dev_max_mm']:>11.1f}{s['jerk_rms']:>9.0f}{s['qp_ms_mean']:>8.2f}{100*s['backstop_rate']:>8.1f}%")

print("\n[5] NOMINAL scene (10 perturbed starts) — the paper's Table I regime")
print(f"{'method':<22}{'reach':>7}{'unsafe':>8}{'min_sdf_mm':>12}{'devmax_mm':>11}")
for m in ("ours_ta_cbf","b2_fixed_pose_cbf","b3a_oracle_sdf","b4_circle_cbf","b6_apf"):
    s=agg(m,"nominal_backstop_on")
    print(f"{m:<22}{s['reach_rate']:>7.2f}{s['unsafe_rollouts']:>5}/10{s['min_sdf_mm']:>12.1f}{s['dev_max_mm']:>11.1f}")
