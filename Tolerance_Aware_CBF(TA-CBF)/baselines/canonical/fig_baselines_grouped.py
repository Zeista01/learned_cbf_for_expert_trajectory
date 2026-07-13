"""Single-column grouped bar chart: SAFE + reach per method (higher=better)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

methods = ['Ours', 'Oracle', 'CN-CBF', 'S²-NNDS', 'NODE', 'APF']
safe_pct  = [100.0, 100.0, 100.0, 0.0, 0.0, 8.3]
reach_pct = [28.0, 42.0, 56.0, 100.0, 100.0, 100.0]
SAFE='#2a9d5c'; REACH='#2c7fb8'
plt.rcParams.update({'font.size':8,'font.family':'DejaVu Sans','axes.linewidth':0.7})
fig, ax = plt.subplots(figsize=(3.5, 3.1))
x=np.arange(len(methods)); w=0.40
ax.axvspan(-0.5, 0.5, color='#f2f4f7', zorder=0)
b1=ax.bar(x-w/2, safe_pct,  w, label='Safe (no penetr.)', color=SAFE, edgecolor='#1c6b3e', lw=0.6, zorder=3)
b2=ax.bar(x+w/2, reach_pct, w, label='Goal reach', color=REACH, edgecolor='#1c5480', lw=0.6, zorder=3)
for bars,vals in ((b1,safe_pct),(b2,reach_pct)):
    for bar,v in zip(bars,vals): ax.text(bar.get_x()+bar.get_width()/2, v+1.5, f'{v:.0f}', ha='center', va='bottom', fontsize=6.3)
ax.set_xticks(x); lab=ax.set_xticklabels(methods, fontsize=7, rotation=20, ha='right'); lab[0].set_fontweight('bold')
ax.set_ylabel('% of 36 gen. scenes (higher better)', fontsize=7.5)
ax.set_ylim(0,113)
ax.grid(axis='y', color='#e6e8eb', lw=0.6, zorder=0)
for s in ('top','right'): ax.spines[s].set_visible(False)
ax.tick_params(length=0)
ax.legend(frameon=False, fontsize=7, loc='upper center', ncol=2, bbox_to_anchor=(0.5,1.13), columnspacing=1.0, handlelength=1.2)
plt.tight_layout()
out='baselines/canonical/fig_baselines_grouped.png'
plt.savefig(out, dpi=220, bbox_inches='tight'); plt.savefig(out.replace('.png','.pdf'), bbox_inches='tight')
print('saved', out)
