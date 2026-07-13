"""
baselines.py — genuine baseline comparisons for the paper.

Two experiments, both on the SAME set of random pose/scale scenes (fixed seeds),
so the comparison is controlled.

EXP A  (Pose-generalization ablation; analytic backstop DISABLED so the LEARNED
        barrier alone must keep the tool safe):
   - Ours (augmented barrier, final_model.pt)
   - Fixed-pose barrier (no augmentation, fixed_pose_model.pt)
   Metric: penetrations (steps with sdf<0) on novel poses. This isolates whether
   augmentation is what prevents the learned barrier from failing out of
   distribution.

EXP B  (Value of the learned field vs pure geometry; full pipeline WITH backstop):
   - Ours (learned barrier + backstop)
   - Analytic-SDF CBF (oracle B = K*clip(sdf-INFLATE); needs the closed-form
     scene distance, which is NOT available from a point cloud at deployment)
   Metric: reach rate + penetrations. Shows whether ours matches the oracle on
   navigation without access to the SDF.

Usage:
    python src/baselines.py --scenes 24 --n_obs 3 4 5 6
Writes results/baselines/baselines.json and prints a comparison table.
"""
import os, sys, json, argparse, copy
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from config import (X_START, X_GOAL, INFLATE_MARGIN, BARRIER_SDF_K,
                    BARRIER_SDF_CLAMP_IN, BARRIER_SDF_CLAMP_OUT, STATE_DIM,
                    DEVICE, Z_CORRIDOR)
from simulate import load_model, BPCBFController
from multi_rollout import run_from_start
from generalization_test import (make_solvable_scene, sdf_all_np,
                                  count_critical_entries, build_tensors)

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT  = os.path.join(ROOT, "results", "baselines")
os.makedirs(OUT, exist_ok=True)


# ── Analytic-SDF barrier (oracle) with the CompositeBarrier interface ─────────
class AnalyticBarrier(torch.nn.Module):
    """B(x) = K*clip(min_i sdf_i(x) - INFLATE, -CL_IN, CL_OUT). Exact geometry.
    Drop-in for model.B (nn.Module): forward(x)->(B,1) and gradient(x)->(B,d)."""
    def __init__(self, shapes):
        super().__init__()
        self.shapes = shapes
        self.K = BARRIER_SDF_K
        self.CL_IN, self.CL_OUT = BARRIER_SDF_CLAMP_IN, BARRIER_SDF_CLAMP_OUT

    def set_obstacles(self, *a, **k):
        pass

    def _B_np(self, xy):
        sdf = sdf_all_np(xy, self.shapes)
        return self.K * np.clip(sdf - INFLATE_MARGIN, -self.CL_IN, self.CL_OUT)

    def forward(self, x_t):
        x = x_t.detach().cpu().numpy()
        b = self._B_np(x[:, :2].astype(np.float32))
        return torch.tensor(b, dtype=torch.float32, device=x_t.device).unsqueeze(-1)

    def gradient(self, x_t):
        x = x_t.detach().cpu().numpy().astype(np.float64)
        e = 1e-4
        g = np.zeros_like(x)
        for d in range(2):  # XY only; z fixed
            xp = x.copy(); xp[:, d] += e
            xm = x.copy(); xm[:, d] -= e
            g[:, d] = (self._B_np(xp[:, :2]) - self._B_np(xm[:, :2])) / (2 * e)
        return torch.tensor(g, dtype=torch.float32, device=x_t.device)


def _metrics(traj, shapes):
    tr = np.array(traj['ee_pos'])
    n_solid, n_lr = count_critical_entries(tr, shapes)
    return dict(pen=int(n_solid), reached=bool(traj['reached_goal']),
                min_clear=float(sdf_all_np(tr, shapes).min()))


def make_scenes(n_obs_list, per, seed, clearance=0.028):
    """Fixed set of feasible random pose/scale scenes shared by all methods."""
    scenes = []
    for n_obs in n_obs_list:
        built, s = 0, 0
        while built < per and s < per * 40:
            try:
                sh = make_solvable_scene(n_obs, seed=seed * 100 + s, clearance=clearance)
                scenes.append((n_obs, sh)); built += 1
            except RuntimeError:
                pass
            s += 1
    return scenes


def run_method(model, ctrl, scenes, use_backstop, analytic=False):
    rows = []
    for n_obs, shapes in scenes:
        if analytic:
            model.B = AnalyticBarrier(shapes)
        else:
            model.set_obstacles(build_tensors(shapes, k=64, seed=1))
        traj = run_from_start(model, ctrl, X_START.copy(),
                              shapes=(shapes if use_backstop else None))
        rows.append(_metrics(traj, shapes))
    pen_scenes = sum(1 for r in rows if r['pen'] > 0)
    return dict(
        scenes=len(rows),
        penetrating_scenes=pen_scenes,
        penetration_rate=pen_scenes / max(len(rows), 1),
        total_pen_steps=int(sum(r['pen'] for r in rows)),
        reached=int(sum(r['reached'] for r in rows)),
        reach_rate=float(np.mean([r['reached'] for r in rows])),
        min_clear_mm=float(np.min([r['min_clear'] for r in rows]) * 1000),
    ), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=6, help="scenes per obstacle count")
    ap.add_argument("--n_obs", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ctrl = BPCBFController()
    scenes = make_scenes(args.n_obs, args.scenes, args.seed)
    print(f"[Baselines] {len(scenes)} shared scenes "
          f"({args.scenes} each of n_obs={args.n_obs})\n")

    results = {}

    # ── EXP A: augmentation ablation, BACKSTOP OFF (learned barrier alone) ────
    print("EXP A  (backstop OFF; learned barrier must keep safety)")
    m_aug = load_model(os.path.join(ROOT, "checkpoints", "final_model.pt"))
    a_ours, _ = run_method(m_aug, ctrl, scenes, use_backstop=False, analytic=False)
    print(f"  ours (augmented)  : pen scenes {a_ours['penetrating_scenes']}/{a_ours['scenes']}"
          f"  reach {a_ours['reached']}/{a_ours['scenes']}")

    m_fix = load_model(os.path.join(ROOT, "checkpoints", "fixed_pose_model.pt"))
    a_fix, _ = run_method(m_fix, ctrl, scenes, use_backstop=False, analytic=False)
    print(f"  fixed-pose (base) : pen scenes {a_fix['penetrating_scenes']}/{a_fix['scenes']}"
          f"  reach {a_fix['reached']}/{a_fix['scenes']}")
    results['expA_backstop_off'] = dict(ours_augmented=a_ours, fixed_pose=a_fix)

    # ── EXP B: learned vs analytic oracle, BACKSTOP ON (full pipeline) ───────
    print("\nEXP B  (backstop ON; reach rate + safety, full pipeline)")
    m_aug2 = load_model(os.path.join(ROOT, "checkpoints", "final_model.pt"))
    b_ours, _ = run_method(m_aug2, ctrl, scenes, use_backstop=True, analytic=False)
    print(f"  ours (learned)    : reach {b_ours['reached']}/{b_ours['scenes']}"
          f"  pen {b_ours['total_pen_steps']}  clear {b_ours['min_clear_mm']:.1f}mm")

    m_an = load_model(os.path.join(ROOT, "checkpoints", "final_model.pt"))
    b_an, _ = run_method(m_an, ctrl, scenes, use_backstop=True, analytic=True)
    print(f"  analytic (oracle) : reach {b_an['reached']}/{b_an['scenes']}"
          f"  pen {b_an['total_pen_steps']}  clear {b_an['min_clear_mm']:.1f}mm")
    results['expB_backstop_on'] = dict(ours_learned=b_ours, analytic_oracle=b_an)

    with open(os.path.join(OUT, "baselines.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ── summary table ────────────────────────────────────────────────────────
    def frac(a, b):
        return "{}/{}".format(a, b)
    def pct(r):
        return "{:.0f}%".format(100 * r)

    print("\n" + "=" * 66)
    print("  BASELINE COMPARISON")
    print("=" * 66)
    print("\nEXP A  learned barrier ALONE (no geometric backstop), {} novel scenes:".format(a_ours['scenes']))
    print("  {:<22}{:<24}{}".format("method", "scenes w/ penetration", "reach"))
    print("  {:<22}{:<24}{}".format("ours (augmented)",
          frac(a_ours['penetrating_scenes'], a_ours['scenes']),
          frac(a_ours['reached'], a_ours['scenes'])))
    print("  {:<22}{:<24}{}".format("fixed-pose (no aug)",
          frac(a_fix['penetrating_scenes'], a_fix['scenes']),
          frac(a_fix['reached'], a_fix['scenes'])))
    print("\nEXP B  full pipeline WITH backstop, {} novel scenes:".format(b_ours['scenes']))
    print("  {:<22}{:<16}{:<14}{}".format("method", "penetrations", "reach rate", "min clear"))
    print("  {:<22}{:<16}{:<14}{:.1f}mm".format("ours (learned)",
          b_ours['total_pen_steps'], pct(b_ours['reach_rate']), b_ours['min_clear_mm']))
    print("  {:<22}{:<16}{:<14}{:.1f}mm".format("analytic (oracle)",
          b_an['total_pen_steps'], pct(b_an['reach_rate']), b_an['min_clear_mm']))
    print("\nSaved -> {}".format(os.path.join(OUT, 'baselines.json')))


if __name__ == "__main__":
    main()
