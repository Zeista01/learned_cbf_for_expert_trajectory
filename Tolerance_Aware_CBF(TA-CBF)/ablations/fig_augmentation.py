"""A1 augmentation ablation: mean barrier false-safe vs augmentation mode."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
modes = ['None', 'Trans.\nonly', 'Scale\nonly', 'Rot.\nonly', 'Full\n(ours)']
fs    = [6.8, 8.1, 1.4, 2.1, 2.7]
OURS='#2c7fb8'; POOR='#d1495b'; OK='#7fb0d3'
colors=[POOR, POOR, OK, OK, OURS]
plt.rcParams.update({'font.size':9.5,'font.family':'DejaVu Sans','axes.linewidth':0.8})
fig, ax = plt.subplots(figsize=(5.2, 3.2))
x=np.arange(len(modes))
b=ax.bar(x, fs, width=0.62, color=colors, edgecolor='#3a3f45', linewidth=0.7, zorder=3)
for xi,v in zip(x,fs): ax.text(xi, v+0.2, f'{v:.1f}', ha='center', va='bottom', fontsize=8.5,
                               fontweight='bold' if xi==4 else 'normal')
ax.set_xticks(x); ax.set_xticklabels(modes, fontsize=8.5)
ax.set_ylabel('Barrier false-safe rate [%]\n(avg over rotations; lower = better)', fontsize=9)
ax.set_ylim(0, 9.6)
ax.set_title('Augmentation ablation: which transform carries\npose generalization', fontsize=10, fontweight='bold', pad=8)
ax.grid(axis='y', color='#e6e8eb', lw=0.7, zorder=0)
for s in ('top','right'): ax.spines[s].set_visible(False)
ax.tick_params(length=0)
plt.tight_layout()
out='ablations/fig_augmentation.png'
plt.savefig(out, dpi=200, bbox_inches='tight'); plt.savefig(out.replace('.png','.pdf'), bbox_inches='tight')
print('saved', out)
