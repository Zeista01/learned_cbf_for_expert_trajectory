"""Baseline comparison bar chart: penetrations + reach (true benchmark_25mm_v2 nos)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# order: safe methods first, ours highlighted. TRUE numbers from benchmark_25mm_v2.json
methods = ['Ours\n(TA-CBF)', 'Oracle\nCBF', 'CN-CBF', 'S²-NNDS', 'NODE', 'APF']
pen_pct   = [0.0, 0.0, 0.0, 100.0, 100.0, 91.7]     # % of 36 gen scenes with a penetration
reach_pct = [28.0, 42.0, 56.0, 100.0, 100.0, 100.0] # % reaching the goal

OURS = '#2c7fb8'; BASE = '#bfc4cb'; EDGE = '#5b6169'
colors = [OURS] + [BASE]*5

plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans', 'axes.linewidth': 0.8})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 2.9))
x = np.arange(len(methods))

def draw(ax, vals, title, ylab, ymax):
    bars = ax.bar(x, vals, width=0.66, color=colors, edgecolor=EDGE, linewidth=0.8, zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + ymax*0.02, f'{v:.0f}', ha='center', va='bottom',
                fontsize=8.5, color='#222', fontweight='bold' if xi==0 else 'normal')
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=8)
    ax.set_ylim(0, ymax); ax.set_ylabel(ylab, fontsize=9)
    ax.set_title(title, fontsize=9.5, fontweight='bold', pad=6)
    ax.grid(axis='y', color='#e6e8eb', linewidth=0.7, zorder=0)
    for s in ('top','right'): ax.spines[s].set_visible(False)
    ax.tick_params(length=0)

draw(axA, pen_pct,   '(a) Critical-region penetrations', 'Scenes with penetration [%]', 112)
draw(axB, reach_pct, '(b) Goal reach rate',              'Reached goal [%]', 112)

# annotate the two takeaways
axA.text(1.0, 60, 'safe\n(0%)', ha='center', fontsize=7.5, color=OURS, style='italic')
axA.text(4.0, 55, 'unsafe\n(no runtime\nbarrier)', ha='center', fontsize=7.5, color='#b2182b', style='italic')

# shared legend for "ours vs baselines"
from matplotlib.patches import Patch
fig.legend([Patch(fc=OURS, ec=EDGE), Patch(fc=BASE, ec=EDGE)],
           ['Ours (TA-CBF)', 'Baselines'], loc='upper center', ncol=2,
           frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.03))
plt.tight_layout(rect=[0,0,1,0.94])
out = 'baselines/canonical/fig_baselines_comparison.png'
plt.savefig(out, dpi=200, bbox_inches='tight'); plt.savefig(out.replace('.png','.pdf'), bbox_inches='tight')
print('saved', out)
