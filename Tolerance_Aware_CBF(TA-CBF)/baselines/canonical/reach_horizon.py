"""Re-measure generalization reach with a longer horizon (T_MAX=24s) so reach
reflects genuine trapping, not the 12s clock. All 36 canonical scenes, backstop ON."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES",""); os.environ.setdefault("OMP_NUM_THREADS","1")
import sys, multiprocessing as mp, numpy as np, re
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))          # baselines/
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))  # src/
import config; config.T_MAX = 24.0
import common.runner as runner; runner.T_MAX = 24.0   # runner imported T_MAX by value
from common.scenes import generalization_suite
from common.methods import OursMethod
from common.runner import run_rollout
from common.metrics import rollout_metrics
from cbf_qp import BPCBFController
from config import X_START
import importlib
def gm(pkg): return importlib.import_module(f"{pkg}.method").get_methods()
_by_name = {m.name: m for pkg in ("b2_fixed_pose_cbf","b3_analytic_sdf_cbf",
            "b4_convex_primitive_cbf","b5_cncbf_pershape") for m in gm(pkg)}
METHODS = [OursMethod()] + [_by_name[n] for n in
           ("b2_fixed_pose_cbf","b3a_oracle_sdf","b3b_cloud_esdf",
            "b4_circle_cbf","b5_cncbf_pershape")]
SCENES = generalization_suite((3,4,5,6,7,8), per=6, clearance=0.025)
def nobs(s): return int(re.search(r"gen_(\d+)obs",s).group(1))
def work(t):
    mname, sh, name = t
    m = {x.name:x for x in METHODS}[mname]
    model = m.make_model(); ctrl = BPCBFController(); m.prepare(model, sh)
    log = run_rollout(model, ctrl, m, X_START.copy(), sh, backstop=True)
    r = rollout_metrics(log); r['scene']=name; r['method']=mname; return r
tasks = [(m.name, sh, name) for m in METHODS for name,sh in SCENES]
with mp.Pool(11) as p: rows = p.map(work, tasks)
print(f"T_MAX=24s | reach by #obstacles (backstop ON, 36 scenes)")
print(f"{'method':<20}"+"".join(f"{n:>7}obs" for n in (3,4,5,6,7,8))+f"{'all':>8}")
from collections import defaultdict
for m in METHODS:
    mr=[r for r in rows if r['method']==m.name]
    by=defaultdict(list)
    for r in mr: by[nobs(r['scene'])].append(r['reached'])
    cells="".join(f"{np.mean(by[n]):>10.2f}" for n in (3,4,5,6,7,8))
    print(f"{m.name:<20}{cells}{np.mean([r['reached'] for r in mr]):>8.2f}")
