# Clearance sweep — does widening the passage make ours reach the target?

Re-runs the full generalization protocol at guaranteed passages of 20/25/30 mm
(the 15 mm project default is the `../as_it_is` baseline), all 10 methods,
both backstop settings, 24 fresh scenes per clearance. Goal: find a clearance
at which ours reliably ARRIVES at the target (surgery needs the tool to reach
the tumour), while safety stays at 0 penetrations.

Run:  ./run_sweep.sh   then   python summarize.py
Artifacts: clrNNmm.json/.md, SUMMARY.md, reach_vs_clearance.png

## Verdict: clearance is NOT the lever for reachability.

Generalization reach (backstop ON), ours vs the best analytic baselines:

| clearance | ours | b3b cloud-ESDF | b3a oracle | b4 circle |
|---|---|---|---|---|
| 15 mm | 0.34 | 0.72 | 0.50 | 0.59 |
| 20 mm | 0.54 | 0.71 | 0.46 | 0.71 |
| 25 mm | 0.62 | 0.79 | 0.46 | 0.79 |
| 30 mm | 0.50 | 0.88 | 0.71 | 0.83 |

Safety stays perfect (0/24 penetrations, min clearance +11 mm) at every setting.

Two things this shows:
1. Widening the passage helps a bit (ours 0.34 -> ~0.5-0.62) but ours stays
   BELOW every analytic-SDF barrier at every clearance. The 25->30 dip is
   scene-sampling noise (scenes are regenerated per clearance; 24 scenes ~=+-10%).
2. The low reach is largely INTRINSIC, not a tight-passage artifact:
   ours' backstop-OFF reach is ~0.62-0.71 and flat across clearance, i.e. the
   nominal DS + learned barrier itself traps ~30-40% of dense scenes. The exact
   backstop then fights the compressed learned barrier (~40-60% activation),
   costing further reach in tight scenes (0.66->0.34 at 15 mm).

## Why (and what actually fixes reachability)
The reactive certificate controller has no global planning / local-minimum
recovery, and the "compressed" learned barrier has a bumpier gradient field than
a clean SDF, so the tool settles into safe local pockets. The paper already
notes this ("not a global planner ... trap ... discard by a feasibility check");
the sweep quantifies the trapped fraction as ~35-50% on random dense scenes.

Levers that WOULD raise reachability (future work, not clearance):
- Cut the backstop<->barrier conflict: lower the defended margin / retune the
  geometric backstop so it stops scaling velocity to zero (recovers the tight-
  scene 0.34->0.66 gap).
- Add local-minimum recovery or a lightweight global seed (e.g. RRT waypoints
  feeding the DS) — the real fix for the intrinsic ~35% trapping.
- Smooth the learned barrier (stronger eikonal, less compression) so its
  gradient field has fewer spurious minima.
- Match benchmark density to the clinical setting (a surgical field rarely has
  6 non-convex no-go zones straddling the direct path).
