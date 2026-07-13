"""
paper_stats.py — genuine quantitative evaluation for the paper.

Runs three experiment suites and reports REAL statistics (mean +/- std over
multiple seeds). No cherry-picking, no filtering of failures.

  Nominal        : N rollouts from perturbed starts on the canonical scene.
  Generalization : M random pose/scale scenes per obstacle count. Scenes are
                   guaranteed to have a FEASIBLE start->goal passage (a global
                   planner could solve them), so the reach rate honestly
                   measures how often the REACTIVE controller finds it.
  Dynamic        : the 4 slow-motion scenes.

Metrics per rollout: penetrations (steps with sdf<0), light-red entries
(0<=sdf<INFLATE), reached goal, path length, time to goal, MIN clearance to the
true surface (= realized standoff), MAX deviation from the demo path.

Usage:
    python src/paper_stats.py --nominal 30 --gen_per 12 --seed 0
Writes results/paper_stats/stats.json and prints a summary table.
"""
import os, sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import (X_START, X_GOAL, CRITICAL_SHAPES, INFLATE_MARGIN,
                    DEMO_WAYPOINTS, SLAB_X, SLAB_Y)
from simulate import load_model, BPCBFController
from multi_rollout import run_from_start
from generalization_test import (make_solvable_scene, sdf_all_np,
                                  count_critical_entries, build_tensors)

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT  = os.path.join(ROOT, "results", "paper_stats")
os.makedirs(OUT, exist_ok=True)


def _traj_metrics(traj_dict, shapes):
    """Compute all per-rollout metrics from a rollout dict."""
    tr = np.array(traj_dict['ee_pos'])                    # (T, 3)
    n_solid, n_lightred = count_critical_entries(tr, shapes)
    reached = bool(traj_dict['reached_goal'])
    # path length
    seglens = np.linalg.norm(np.diff(tr[:, :2], axis=0), axis=1)
    path_len = float(seglens.sum())
    # time to goal
    t_goal = float(traj_dict['time'][-1])
    # min clearance to the TRUE surface (realized standoff); >0 means outside
    sdf_series = sdf_all_np(tr, shapes)
    min_clear = float(sdf_series.min())
    # max deviation from the demo polyline (nearest waypoint distance)
    dw = DEMO_WAYPOINTS[:, :2]
    devs = [float(np.min(np.linalg.norm(dw - p[:2], axis=1))) for p in tr]
    max_dev = float(np.max(devs))
    return dict(n_solid=n_solid, n_lightred=n_lightred, reached=reached,
                path_len=path_len, t_goal=t_goal, min_clear=min_clear,
                max_dev=max_dev)


def _agg(vals):
    a = np.array(vals, dtype=float)
    return dict(mean=float(a.mean()), std=float(a.std()),
                min=float(a.min()), max=float(a.max()), n=int(a.size))


def run_nominal(model, ctrl, n, seed):
    print(f"\n[NOMINAL] {n} rollouts from perturbed starts (radius 1.8 cm)")
    # Install the canonical obstacle set so the LEARNED barrier sees it.
    model.set_obstacles(build_tensors(CRITICAL_SHAPES, k=64, seed=0))
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        sp = X_START.copy()
        if i > 0:
            sp[:2] += rng.uniform(-0.018, 0.018, 2)
        traj = run_from_start(model, ctrl, sp, shapes=CRITICAL_SHAPES)
        m = _traj_metrics(traj, CRITICAL_SHAPES)
        rows.append(m)
        print(f"  {i+1:2d}/{n}  reached={m['reached']}  pen={m['n_solid']}  "
              f"len={m['path_len']*100:.1f}cm  t={m['t_goal']:.2f}s  "
              f"clear={m['min_clear']*1000:.1f}mm  dev={m['max_dev']*1000:.1f}mm")
    summary = dict(
        n=n,
        penetrations_total=int(sum(r['n_solid'] for r in rows)),
        lightred_total=int(sum(r['n_lightred'] for r in rows)),
        reached=int(sum(r['reached'] for r in rows)),
        path_len_cm=_agg([r['path_len']*100 for r in rows]),
        t_goal_s=_agg([r['t_goal'] for r in rows]),
        min_clear_mm=_agg([r['min_clear']*1000 for r in rows]),
        max_dev_mm=_agg([r['max_dev']*1000 for r in rows]),
    )
    _plot_nominal(rows, summary)
    return summary, rows


def _plot_nominal(rows, summary):
    """Distribution figure over all nominal rollouts (for the paper)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pl = [r['path_len']*100 for r in rows]
    cl = [r['min_clear']*1000 for r in rows]
    dv = [r['max_dev']*1000 for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for a, data, lab, mu, sd, ref, reflab in [
        (ax[0], pl, "Path length (cm)", summary['path_len_cm']['mean'], summary['path_len_cm']['std'], None, None),
        (ax[1], cl, "Min clearance to tissue (mm)", summary['min_clear_mm']['mean'], summary['min_clear_mm']['std'], 0.0, "penetration (0)"),
        (ax[2], dv, "Max deviation from demo (mm)", summary['max_dev_mm']['mean'], summary['max_dev_mm']['std'], None, None),
    ]:
        a.hist(data, bins=10, color="#3b78b0", edgecolor="white")
        a.axvline(mu, color="#c0392b", ls="--", lw=1.5, label=f"mean {mu:.1f} $\\pm$ {sd:.1f}")
        if ref is not None:
            a.axvline(ref, color="k", ls=":", lw=1.2, label=reflab)
        a.set_xlabel(lab); a.set_ylabel("rollouts"); a.legend(fontsize=8)
    fig.suptitle(f"Nominal scene: {summary['n']} rollouts, "
                 f"{summary['reached']}/{summary['n']} reached goal, "
                 f"{summary['penetrations_total']} penetrations", fontsize=11)
    fig.tight_layout()
    out = os.path.join(OUT, "nominal_stats.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  figure -> {out}")


def run_generalization(model, ctrl, n_obs_list, per, seed, clearance=0.028):
    # clearance = guaranteed start->goal passage width. Set >= the controller's
    # realized standoff (~25-40 mm) so the test asks a FAIR question: "given a
    # passage the controller can physically use, how often does the reactive
    # policy find it?" (rather than threading passages narrower than the standoff).
    print(f"\n[GENERALIZATION] {per} random feasible scenes per obstacle count "
          f"{n_obs_list}  (guaranteed passage >= {clearance*1000:.0f} mm)")
    per_count = {}
    all_rows = []
    for n_obs in n_obs_list:
        rows = []
        built = 0
        s = 0
        while built < per and s < per * 40:
            try:
                shapes = make_solvable_scene(n_obs, seed=seed * 100 + s,
                                             clearance=clearance)
            except RuntimeError:
                s += 1
                continue
            s += 1
            # Install THIS scene's obstacles so the learned barrier is zero-shot
            # conditioned on them (no retraining).
            model.set_obstacles(build_tensors(shapes, k=64, seed=s))
            traj = run_from_start(model, ctrl, X_START.copy(), shapes=shapes)
            m = _traj_metrics(traj, shapes)
            rows.append(m); all_rows.append(m)
            built += 1
            print(f"  n_obs={n_obs}  scene {built:2d}/{per}  "
                  f"reached={m['reached']}  pen={m['n_solid']}  "
                  f"clear={m['min_clear']*1000:.1f}mm  dev={m['max_dev']*1000:.1f}mm")
        per_count[n_obs] = dict(
            scenes=len(rows),
            penetrations_total=int(sum(r['n_solid'] for r in rows)),
            reached=int(sum(r['reached'] for r in rows)),
            reach_rate=float(np.mean([r['reached'] for r in rows])) if rows else 0.0,
            min_clear_mm=_agg([r['min_clear']*1000 for r in rows]) if rows else None,
        )
    overall = dict(
        scenes=len(all_rows),
        penetrations_total=int(sum(r['n_solid'] for r in all_rows)),
        reached=int(sum(r['reached'] for r in all_rows)),
        reach_rate=float(np.mean([r['reached'] for r in all_rows])),
        min_clear_mm=_agg([r['min_clear']*1000 for r in all_rows]),
        max_dev_mm=_agg([r['max_dev']*1000 for r in all_rows]),
    )
    return dict(per_obstacle=per_count, overall=overall), all_rows


def run_dynamic(model, ctrl):
    print(f"\n[DYNAMIC] 4 slow-motion scenes")
    import dynamic_env_test as D
    from config import canonical_interior_cloud
    canon = [canonical_interior_cloud(sh, k=128, seed=300+i)
             for i, sh in enumerate(CRITICAL_SHAPES)]
    out = {}
    for motion in ["translate", "rotate", "transrotate", "evolve"]:
        traj, taus, n_solid, n_buffer, reached = D.run_dynamic(model, ctrl, motion, canon)
        out[motion] = dict(penetrations=int(n_solid), buffer=int(n_buffer),
                           reached=bool(reached), steps=int(len(traj)-1))
        print(f"  {motion:12s}  reached={reached}  pen={n_solid}  buffer={n_buffer}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nominal", type=int, default=30)
    ap.add_argument("--gen_per", type=int, default=12)
    ap.add_argument("--n_obs", type=int, nargs="+", default=[3,4,5,6,7,8])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model = load_model()
    ctrl  = BPCBFController()

    results = {}
    results['nominal'], _   = run_nominal(model, ctrl, args.nominal, args.seed)
    results['generalization'], _ = run_generalization(model, ctrl, args.n_obs,
                                                       args.gen_per, args.seed)
    results['dynamic']      = run_dynamic(model, ctrl)

    with open(os.path.join(OUT, "stats.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ── Print clean summary ──────────────────────────────────────────────────
    print("\n" + "="*64)
    print("  GENUINE RESULTS SUMMARY")
    print("="*64)
    nm = results['nominal']
    print(f"\nNOMINAL ({nm['n']} rollouts):")
    print(f"  penetrations (steps inside tissue): {nm['penetrations_total']}")
    print(f"  light-red entries:                  {nm['lightred_total']}")
    print(f"  reached goal:                       {nm['reached']}/{nm['n']}")
    print(f"  path length:   {nm['path_len_cm']['mean']:.1f} +/- {nm['path_len_cm']['std']:.1f} cm")
    print(f"  time to goal:  {nm['t_goal_s']['mean']:.2f} +/- {nm['t_goal_s']['std']:.2f} s")
    print(f"  min clearance: {nm['min_clear_mm']['mean']:.1f} +/- {nm['min_clear_mm']['std']:.1f} mm "
          f"(range {nm['min_clear_mm']['min']:.1f}-{nm['min_clear_mm']['max']:.1f})")
    print(f"  max deviation: {nm['max_dev_mm']['mean']:.1f} +/- {nm['max_dev_mm']['std']:.1f} mm")

    gg = results['generalization']['overall']
    print(f"\nGENERALIZATION ({gg['scenes']} random feasible scenes):")
    print(f"  penetrations (total):  {gg['penetrations_total']}")
    print(f"  reached goal:          {gg['reached']}/{gg['scenes']} "
          f"({100*gg['reach_rate']:.0f}%)")
    print(f"  min clearance: {gg['min_clear_mm']['mean']:.1f} +/- {gg['min_clear_mm']['std']:.1f} mm "
          f"(range {gg['min_clear_mm']['min']:.1f}-{gg['min_clear_mm']['max']:.1f})")
    print(f"  max deviation: {gg['max_dev_mm']['mean']:.1f} +/- {gg['max_dev_mm']['std']:.1f} mm")
    print("  by obstacle count:")
    for n_obs, d in results['generalization']['per_obstacle'].items():
        print(f"    {n_obs} obs: {d['scenes']} scenes, pen={d['penetrations_total']}, "
              f"reach={d['reached']}/{d['scenes']} ({100*d['reach_rate']:.0f}%)")

    dy = results['dynamic']
    print(f"\nDYNAMIC (4 motions):")
    pen = sum(d['penetrations'] for d in dy.values())
    rch = sum(d['reached'] for d in dy.values())
    print(f"  penetrations (total): {pen}")
    print(f"  reached goal:         {rch}/4")
    print(f"\nSaved -> {os.path.join(OUT, 'stats.json')}")


if __name__ == "__main__":
    main()
