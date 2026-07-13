# Ablations

Each ablation trains the composite barrier on all shapes with the *identical*
recipe as `final_model`, toggling **one** component, then evaluates the effect.
All results verified on real runs (no inherited claims).

```bash
# train the six variants (aug modes x eikonal)
for a in full none rot scale trans; do venv/bin/python ablations/train_ablation.py --aug $a --eikonal on; done
venv/bin/python ablations/train_ablation.py --aug full --eikonal off
# evaluate all three ablations
venv/bin/python ablations/eval_ablations.py
venv/bin/python ablations/fig_augmentation.py     # A1 figure
```

## Results (verified 2026-07)

**A1 — Augmentation** (barrier false-safe %, avg over rotations; lower better):

| None | Trans only | Scale only | Rot only | Full |
|---|---|---|---|---|
| 6.8 | 8.1 | 1.4 | 2.1 | 2.7 |

Rotation and scale augmentation carry the pose generalization; translation does
not (no better than none). Figure: `fig_augmentation.png`.

**A2 — Eikonal** (verifies/corrects the paper): removing the eikonal penalty does
**not** blow up the gradient — the SDF-shaped target already bounds the slope.

| variant | mean ‖∇B‖ | p99 ‖∇B‖ | false-safe |
|---|---|---|---|
| eikonal ON | 1.2 | 5.6 | 2.7% |
| eikonal OFF | 1.4 | 5.9 | 3.2% |

The old paper claim ("gradient exceeds two hundred, near-step barrier") was from
an earlier hinge-only barrier and is **not** true for the SDF-regression method;
the paper text was corrected to report the eikonal as a minor refinement.

**A3 — Smooth-min vs hard-min** (two obstacles close; max |d²B/dy²| along the seam;
smaller = smoother): smooth-min (β=1000) **0.5** vs hard-min **8.2** — the smooth
minimum keeps the fused barrier ~16× smoother, confirming the fusion claim.

## Files
- `train_ablation.py` — one trainer, `--aug {full,none,rot,scale,trans} --eikonal {on,off}`
- `eval_ablations.py` — prints A1/A2/A3 tables
- `fig_augmentation.py` — A1 bar chart
- `checkpoints/abl_<aug>_eik-<on|off>.pt` — trained variants (gitignored)
