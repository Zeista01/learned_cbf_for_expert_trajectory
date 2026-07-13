# as_it_is — honest full-protocol results (15 mm scene clearance)

Full protocol, 840 rollouts, all 10 methods x {nominal, generalization} x
backstop {off, on}. 10 perturbed nominal starts + 32 generalization scenes
(8 each of 3/4/5/6 obstacles). Scenes guarantee only a ~15 mm geometric passage
(project default SCENE_CLEARANCE) — this sits right at ours' effective standoff,
so reach here conflates conservatism with infeasibility. See `clearance_sweep/`
for the disentangled reachability study.

- `baselines_full.json` — full per-rollout records + aggregates
- `baselines_full.md`   — summary table (4 regimes)
- `density_breakdown.txt`, `analyze_by_density.py` — reach/safety by #obstacles

## Headline
- SAFETY (backstop ON): every method 0 penetrations across all 42 scenes.
- AUGMENTATION ABLATION (backstop OFF, barrier alone): ours 8/32 unsafe vs
  fixed-pose 15/32 — augmentation ~halves penetrations.
- REACH (generalization, backstop ON): ours 0.34, below every analytic-SDF
  baseline. Cause: ours' backstop fires ~60% of steps and traps in dense,
  tight-passage scenes. This is the 15 mm-clearance confound the sweep resolves.
