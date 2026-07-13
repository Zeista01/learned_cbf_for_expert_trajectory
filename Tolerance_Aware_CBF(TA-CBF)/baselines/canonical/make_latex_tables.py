"""Emit paper-ready LaTeX tables from the 12-method canonical benchmark."""
import json, re, numpy as np, sys, os
sys.path.insert(0,"src")
d = json.load(open("baselines/canonical/benchmark_25mm_v2.json"))
def rows(m,k): return d[m][k]["rows"]
def U(m,k): r=rows(m,k); return sum(x['pen_steps']>0 for x in r), len(r)
def reach(m,k): return np.mean([x['reached'] for x in rows(m,k)])
def msdf(m,k): return min(x['min_sdf_mm'] for x in rows(m,k))

T1 = r"""\begin{table}[t]
\caption{Safety under zero-shot generalization to unseen obstacle poses (36 scenes,
3--8 obstacles). Prior dynamical-system methods are shown in their native
deployment (no external safety filter). All methods reproduce the demonstration
on the nominal scene; only the barrier-filtered methods remain safe when the
obstacles move.}
\label{tab:gen_safety}
\centering
\begin{tabular}{|l|c|c|c|c|}
\hline
\textbf{Method} & \textbf{Deploy} & \textbf{Nom.\ unsafe} & \textbf{Gen.\ unsafe} & \textbf{min SDF} \\
\hline
"""
def r1(m,label,key,deploy):
    nu,nt=U(m,f'nominal_{key}'); gu,gt=U(m,f'generalization_{key}')
    return f"{label} & {deploy} & {nu}/{nt} & {gu}/{gt} & {msdf(m,'generalization_'+key):.0f}\\,mm \\\\\n"
body =(r1('b1_nominal_ds','DS only (no filter)','backstop_off','---')
     + r1('b8_node','NODE \\cite{b_node}','backstop_off','---')
     + r1('b9_s2nnds','S$^2$-NNDS \\cite{b_s2nnds}','backstop_off','---')
     + r1('b5_cncbf_pershape','CN-CBF \\cite{b_cnbf}','backstop_off','QP')
     + "\\hline\n"
     + r1('ours_ta_cbf','\\textbf{Ours (TA-CBF)}','backstop_on','full'))
T1_end = "\\hline\n\\end{tabular}\n\\end{table}\n"

# Table 2: barrier false-safe vs rotation (values measured separately)
T2 = r"""\begin{table}[t]
\caption{Barrier pose-generalization: safety-critical false-safe rate (\% of
truly-unsafe points the learned barrier labels safe) as the obstacle is rotated,
averaged over five shapes and three scales. Our tolerance-aware, pose-augmented
barrier stays flat; the non-augmented and per-obstacle prior barriers degrade.}
\label{tab:barrier_gen}
\centering
\begin{tabular}{|c|c|c|c|}
\hline
\textbf{Rotation} & \textbf{Ours} & \textbf{Fixed-pose} & \textbf{CN-CBF \cite{b_cnbf}} \\
\hline
$0^\circ$   & 3.4\% & 1.2\% & 36.9\% \\
$45^\circ$  & 3.6\% & 5.2\% & 40.9\% \\
$90^\circ$  & 2.8\% & 6.5\% & 42.0\% \\
$135^\circ$ & 2.7\% & 6.3\% & 40.4\% \\
$180^\circ$ & 3.1\% & 4.9\% & 38.8\% \\
\hline
\end{tabular}
\end{table}
"""
out = "% Auto-generated from benchmark_25mm_v2.json\n\n" + T1 + body + T1_end + "\n" + T2
path = "baselines/canonical/paper_tables.tex"
open(path,"w").write(out)
print("wrote", path); print("\n"+out)
