import json, re, numpy as np, sys
sys.path.insert(0,"src")
from collections import defaultdict
d = json.load(open("baselines/canonical/benchmark_25mm_v2.json"))
NOBS=(3,4,5,6,7,8)
def nobs(s):
    m=re.search(r"gen_(\d+)obs",s); return int(m.group(1)) if m else None
def rows(m,k): return d[m][k]["rows"] if k in d[m] and "rows" in d[m][k] else []
LABEL={'ours_ta_cbf':'Ours (TA-CBF)','b8_node':'NODE [Chen+18]',
 'b9_s2nnds':'S2-NNDS','b5_cncbf_pershape':'CN-CBF','b2_fixed_pose_cbf':'Fixed-pose (no aug)',
 'b1_nominal_ds':'DS only (no filter)','b3a_oracle_sdf':'Analytic-SDF (oracle)',
 'b3b_cloud_esdf':'Point-cloud ESDF','b4_circle_cbf':'Convex-primitive','b6_apf':'APF',
 'b7_global_barrier':'Global barrier','b1_straight_line':'Straight line'}

print("="*94)
print("PRIOR-WORK COMPARISON  (25mm, 3-8 obstacles x6 = 36 gen scenes + 10 nominal starts)")
print("Prior DS methods (NODE, S2-NNDS) shown in NATIVE deployment = no backstop.")
print("="*94)

print("\n[MAIN] Generalization SAFETY — each method in its native config")
print(f"{'method':<26}{'deploy':>12}{'reach':>7}{'unsafe scenes':>15}{'pen steps':>11}{'min_sdf_mm':>12}")
def line(m, key, deploy):
    r=rows(m,key)
    if not r: return
    u=sum(x['pen_steps']>0 for x in r); reach=np.mean([x['reached'] for x in r])
    pen=sum(x['pen_steps'] for x in r); msdf=min(x['min_sdf_mm'] for x in r)
    print(f"{LABEL[m]:<26}{deploy:>12}{reach:>7.2f}{u:>8}/{len(r):<6}{pen:>11}{msdf:>12.1f}")
line('b8_node','generalization_backstop_off','no filter')
line('b9_s2nnds','generalization_backstop_off','no filter')
line('b5_cncbf_pershape','generalization_backstop_off','QP only')
line('b1_nominal_ds','generalization_backstop_off','no filter')
line('ours_ta_cbf','generalization_backstop_on','full (ours)')
print("  -> prior methods reproduce the demo but PENETRATE moved obstacles; ours stays safe.")

print("\n[SAFETY across regimes] penetrating scenes / total")
print(f"{'method':<26}{'nom(native)':>13}{'gen(native)':>13}")
for m,key in [('b8_node','backstop_off'),('b9_s2nnds','backstop_off'),
              ('b5_cncbf_pershape','backstop_off'),('ours_ta_cbf','backstop_on')]:
    nr=rows(m,f'nominal_{key}'); gr=rows(m,f'generalization_{key}')
    nu=sum(x['pen_steps']>0 for x in nr); gu=sum(x['pen_steps']>0 for x in gr)
    print(f"{LABEL[m]:<26}{nu:>7}/{len(nr):<5}{gu:>8}/{len(gr):<5}")

print("\n[ABLATION] augmentation — learned barrier ALONE (backstop off), unsafe scenes by #obs")
print(f"{'method':<26}"+"".join(f"{n:>6}o" for n in NOBS)+f"{'tot':>8}")
for m in ('ours_ta_cbf','b2_fixed_pose_cbf'):
    by=defaultdict(list)
    for r in rows(m,'generalization_backstop_off'): by[nobs(r['scene'])].append(r['pen_steps']>0)
    cells="".join(f"{sum(by[n]):>3}/{len(by[n])}" for n in NOBS)
    tot=sum(r['pen_steps']>0 for r in rows(m,'generalization_backstop_off'))
    print(f"{LABEL[m]:<26}{cells}{tot:>6}/36")

print("\n[NOMINAL] all methods reproduce the demo safely (paper Table I regime, native deploy)")
print(f"{'method':<26}{'reach':>7}{'unsafe':>8}{'min_sdf_mm':>12}{'dev_mm':>9}")
for m,key in [('ours_ta_cbf','nominal_backstop_on'),('b8_node','nominal_backstop_off'),
              ('b9_s2nnds','nominal_backstop_off'),('b5_cncbf_pershape','nominal_backstop_off')]:
    r=rows(m,key); s=d[m][key]['summary']
    u=sum(x['pen_steps']>0 for x in r)
    print(f"{LABEL[m]:<26}{s['reach_rate']:>7.2f}{u:>5}/{len(r):<3}{s['min_sdf_mm']:>12.1f}{s['dev_max_mm']:>9.1f}")
