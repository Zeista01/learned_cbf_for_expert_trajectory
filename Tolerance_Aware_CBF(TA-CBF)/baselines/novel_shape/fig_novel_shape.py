"""Single-column novel-shape chart: unsafe correctly detected (higher=better)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
shapes=['Star','Cresc.','Kidney','L-sh.','Mean']
ours_full=[98.4,98.4,92.1,96.3,96.3]; ours_ho=[99.9,99.6,97.0,96.2,98.2]; cncbf=[50.5,44.7,58.5,64.7,54.6]
FULL='#a9cce3'; HO='#2c7fb8'; CN='#d1495b'
plt.rcParams.update({'font.size':8,'font.family':'DejaVu Sans','axes.linewidth':0.7})
fig, ax = plt.subplots(figsize=(3.5, 3.1))
x=np.arange(len(shapes)); w=0.27
ax.axvspan(3.5,4.5,color='#f2f4f7',zorder=0)
b1=ax.bar(x-w, ours_full, w, label='Ours (seen)', color=FULL, edgecolor='#5b6169', lw=0.5, zorder=3)
b2=ax.bar(x,   ours_ho,   w, label='Ours (novel)', color=HO, edgecolor='#1c5480', lw=0.5, zorder=3)
b3=ax.bar(x+w, cncbf,     w, label='CN-CBF', color=CN, edgecolor='#7a2532', lw=0.5, zorder=3)
for bars,vals in ((b1,ours_full),(b2,ours_ho),(b3,cncbf)):
    for bar,v in zip(bars,vals): ax.text(bar.get_x()+bar.get_width()/2, v+1.2, f'{v:.0f}', ha='center', va='bottom', fontsize=5.6)
ax.set_xticks(x); lab=ax.set_xticklabels(shapes, fontsize=7)
for l in lab:
    if l.get_text()=='Mean': l.set_fontweight('bold')
ax.set_ylabel('Unsafe correctly detected [%]', fontsize=7.5)
ax.set_ylim(0,118)
ax.grid(axis='y', color='#e6e8eb', lw=0.6, zorder=0)
for s in ('top','right'): ax.spines[s].set_visible(False)
ax.tick_params(length=0)
ax.legend(frameon=False, fontsize=6.8, loc='upper center', ncol=3, bbox_to_anchor=(0.5,-0.10), handlelength=1.1, columnspacing=1.0)
plt.tight_layout()
out='baselines/novel_shape/fig_novel_shape.png'
plt.savefig(out, dpi=220, bbox_inches='tight'); plt.savefig(out.replace('.png','.pdf'), bbox_inches='tight')
print('saved', out)
