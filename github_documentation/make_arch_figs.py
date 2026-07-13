"""
make_arch_figs.py — render the TRUE architecture diagrams for the README,
faithful to src/models.py (BPCBFModel) and src/cbf_qp.py (BPCBFController).

Outputs (into ./architecture/):
    system_pipeline.png     end-to-end: demos -> augmentation -> 3 nets -> QP -> rollout
    composite_barrier.png   B_phi : PointNet encoder -> conditional CBF -> smooth-min
    online_controller.png   the 4-stage online safety filter
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

OUT = os.path.join(os.path.dirname(__file__), "architecture")
os.makedirs(OUT, exist_ok=True)

# ── palette ──────────────────────────────────────────────────────────────────
INK   = "#1f2933"
NAVY  = "#12324f"
BLUE  = "#2c6fb5"
LBLUE = "#dbe9f6"
TEAL  = "#0f7d6b"
LTEAL = "#d3 efe9".replace(" ", "")
LTEAL = "#d3efe9"
PURP  = "#6a4bb0"
LPURP = "#e7e0f5"
AMBER = "#c9821a"
LAMBER= "#fbeccd"
RED   = "#c0392b"
LRED  = "#f7dcd8"
GREY  = "#5b6169"
LGREY = "#eef1f4"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "mathtext.fontset": "cm",
})


def box(ax, x, y, w, h, fc, ec, title=None, lines=None, tfs=11, lfs=8.4,
        title_color="white", body_color=INK, round_=0.02, lw=1.4, align="center"):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.004,rounding_size={round_}",
                       fc=fc, ec=ec, lw=lw, zorder=3)
    ax.add_patch(p)
    cx = x + w / 2
    ty = y + h - 0.052
    if title:
        # header band
        ax.add_patch(FancyBboxPatch((x, y + h - 0.075), w, 0.075,
                     boxstyle=f"round,pad=0.004,rounding_size={round_}",
                     fc=ec, ec=ec, lw=0, zorder=4))
        ax.text(cx, y + h - 0.038, title, ha="center", va="center",
                fontsize=tfs, color=title_color, fontweight="bold", zorder=6)
        ty = y + h - 0.11
    if lines:
        if align == "center":
            for ln in lines:
                ax.text(cx, ty, ln, ha="center", va="top", fontsize=lfs,
                        color=body_color, zorder=6)
                ty -= 0.049
        else:
            lx = x + 0.028
            for ln in lines:
                ax.text(lx, ty, ln, ha="left", va="top", fontsize=lfs,
                        color=body_color, zorder=6)
                ty -= 0.049
    return (cx, y + h / 2)


def arrow(ax, p0, p1, color=NAVY, lw=2.2, style="-|>", ls="-", rad=0.0):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=18,
                        color=color, lw=lw, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}", zorder=2,
                        shrinkA=2, shrinkB=2)
    ax.add_patch(a)


def new_ax(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.12)
    plt.close(fig)
    print("saved", path)


# ══════════════════════════════════════════════════════════════════════════════
# 1) COMPOSITE NEURAL BARRIER  B_phi   (models.py: CompositeBarrier)
# ══════════════════════════════════════════════════════════════════════════════
def composite_barrier():
    fig, ax = new_ax(13.5, 7.2)
    ax.text(0.5, 0.975, r"Learned Composite Neural Barrier  $B_\phi(x)$",
            ha="center", va="center", fontsize=16, fontweight="bold", color=NAVY)
    ax.text(0.5, 0.935,
            "permutation-invariant set encoder  •  one shared conditional CBF  •  smooth-min fusion  →  any number of obstacles, any shape",
            ha="center", va="center", fontsize=9.3, color=GREY, style="italic")

    # ---- PART A : ObstacleEncoder (PointNet) ----
    box(ax, 0.02, 0.10, 0.30, 0.78, LBLUE, BLUE,
        title="A  ·  ObstacleEncoder  (PointNet)")
    box(ax, 0.045, 0.735, 0.25, 0.078, "white", BLUE, lines=None)
    ax.text(0.17, 0.786, r"cloud  $P \in \mathbb{R}^{K\times 2}$", ha="center", va="center", fontsize=10, color=INK)
    ax.text(0.17, 0.758, "K interior pts, centered at centroid $c_i$", ha="center", va="center", fontsize=7.6, color=GREY)
    ax.text(0.17, 0.706, "per-point MLP  (shared over K points)", ha="center", va="center", fontsize=8, color=NAVY, style="italic")
    for j, txt in enumerate(["Linear(2 → 128) + ReLU",
                             "Linear(128 → 128) + ReLU",
                             "Linear(128 → 64)"]):
        yy = 0.625 - j * 0.073
        box(ax, 0.055, yy, 0.23, 0.058, "white", BLUE)
        ax.text(0.17, yy + 0.029, txt, ha="center", va="center", fontsize=8.4, color=INK)
    ax.text(0.17, 0.383, "feature map  $(K, 64)$", ha="center", va="center", fontsize=8.2, color=GREY)
    box(ax, 0.055, 0.29, 0.23, 0.066, NAVY, NAVY)
    ax.text(0.17, 0.323, "Max-Pool over K", ha="center", va="center", fontsize=9.6, color="white", fontweight="bold")
    box(ax, 0.055, 0.135, 0.23, 0.11, "#eaf2fb", BLUE)
    ax.text(0.17, 0.207, r"$e_i \in \mathbb{R}^{64}$", ha="center", va="center", fontsize=11, color=NAVY, fontweight="bold")
    ax.text(0.17, 0.168, "shape embedding\n(encodes rotation + scale)", ha="center", va="center", fontsize=7.4, color=GREY)
    # vertical flow arrows in A
    for y0, y1 in [(0.625, 0.594), (0.552, 0.521), (0.29, 0.245)]:
        arrow(ax, (0.17, y0), (0.17, y1), color=BLUE, lw=1.6)

    # ---- PART B : ConditionalObstacleCBF ----
    box(ax, 0.355, 0.10, 0.30, 0.78, LPURP, PURP,
        title="B  ·  ConditionalObstacleCBF  (shared)")
    box(ax, 0.378, 0.70, 0.115, 0.088, "white", PURP)
    ax.text(0.4355, 0.744, r"$x_{\mathrm{rel}}=\frac{x-c_i}{\sigma}$", ha="center", va="center", fontsize=9.5, color=INK)
    ax.text(0.4355, 0.715, r"$\in \mathbb{R}^{3}$", ha="center", va="center", fontsize=8, color=GREY)
    box(ax, 0.515, 0.70, 0.115, 0.088, "white", PURP)
    ax.text(0.5725, 0.744, r"$e_i$  (Part A)", ha="center", va="center", fontsize=9.5, color=INK)
    ax.text(0.5725, 0.715, r"$\in \mathbb{R}^{64}$", ha="center", va="center", fontsize=8, color=GREY)
    box(ax, 0.40, 0.612, 0.21, 0.058, "#efe9fa", PURP)
    ax.text(0.505, 0.641, r"concat  $\rightarrow\ \mathbb{R}^{67}$", ha="center", va="center", fontsize=9, color=NAVY, fontweight="bold")
    for j, txt in enumerate(["Linear(67 → 256) + Tanh",
                             "Linear(256 → 256) + Tanh",
                             "Linear(256 → 256) + Tanh",
                             "Linear(256 → 1)"]):
        yy = 0.535 - j * 0.072
        fc = "#efe9fa" if j < 3 else "#c9b8ec"
        box(ax, 0.40, yy, 0.21, 0.055, fc, PURP)
        ax.text(0.505, yy + 0.0275, txt, ha="center", va="center", fontsize=8.3, color=INK)
    box(ax, 0.40, 0.135, 0.21, 0.085, "#e5dcf6", PURP)
    ax.text(0.505, 0.191, r"$b_i(x) \in \mathbb{R}$", ha="center", va="center", fontsize=11, color=PURP, fontweight="bold")
    ax.text(0.505, 0.153, r"$>0$ outside obstacle  •  $<0$ inside", ha="center", va="center", fontsize=7.6, color=GREY)
    arrow(ax, (0.4355, 0.70), (0.47, 0.67), color=PURP, lw=1.5)
    arrow(ax, (0.5725, 0.70), (0.54, 0.67), color=PURP, lw=1.5)
    for y0, y1 in [(0.612, 0.59), (0.535, 0.463), (0.463, 0.391), (0.391, 0.319), (0.319, 0.22)]:
        arrow(ax, (0.505, y0), (0.505, y1), color=PURP, lw=1.6)

    # ---- PART C : Smooth-Min fusion ----
    box(ax, 0.69, 0.10, 0.29, 0.78, LTEAL, TEAL,
        title="C  ·  Smooth-Min Fusion")
    ax.text(0.835, 0.79, "for each obstacle $i$: run A + B  →  $b_i(x)$", ha="center", va="center",
            fontsize=8.2, color=NAVY, style="italic")
    for j in range(3):
        yy = 0.70 - j * 0.052
        box(ax, 0.72, yy, 0.23, 0.044, "#e6f4f0", TEAL)
        lbl = [r"$b_1(x)$", r"$b_2(x)$", r"$b_M(x)$"][j]
        if j == 2:
            ax.text(0.835, 0.622 + 0.017, r"$\vdots$", ha="center", va="center", fontsize=11, color=TEAL)
        ax.text(0.835, yy + 0.022, lbl, ha="center", va="center", fontsize=9.2, color=INK)
    box(ax, 0.72, 0.415, 0.23, 0.12, "#0f7d6b", TEAL)
    ax.text(0.835, 0.505, "Smooth-Min   (CN-CBF Eq. 18)", ha="center", va="center", fontsize=8.6, color="white", fontweight="bold")
    ax.text(0.835, 0.462, r"$B(x)=-\frac{1}{\beta}\,\log\!\sum_i e^{-\beta\,b_i(x)}$",
            ha="center", va="center", fontsize=11.5, color="white")
    ax.text(0.835, 0.428, r"$\beta=1000$  •  under-approx of $\min_i b_i$", ha="center", va="center", fontsize=7.4, color="#d7f0ea")
    box(ax, 0.72, 0.23, 0.23, 0.10, "#e6f4f0", TEAL)
    ax.text(0.835, 0.297, r"$B(x) \in \mathbb{R}$", ha="center", va="center", fontsize=12.5, color=TEAL, fontweight="bold")
    ax.text(0.835, 0.256, "composite barrier over the obstacle SET", ha="center", va="center", fontsize=7.3, color=GREY)
    box(ax, 0.72, 0.135, 0.23, 0.072, "#fbeccd", AMBER)
    ax.text(0.835, 0.185, "trained via inflated-SDF regression", ha="center", va="center", fontsize=7.8, color="#7a520f")
    ax.text(0.835, 0.156, r"+ pose / scale / translation augmentation", ha="center", va="center", fontsize=7.8, color="#7a520f")
    arrow(ax, (0.835, 0.598), (0.835, 0.535), color=TEAL, lw=1.7)
    arrow(ax, (0.835, 0.415), (0.835, 0.33), color=TEAL, lw=1.7)
    arrow(ax, (0.835, 0.23), (0.835, 0.207), color=AMBER, lw=1.5)

    # inter-part arrows
    arrow(ax, (0.29, 0.19), (0.515, 0.744), color=BLUE, lw=1.8, rad=-0.18, style="-|>")
    arrow(ax, (0.61, 0.19), (0.72, 0.19), color=PURP, lw=1.9)
    ax.text(0.665, 0.205, "×M", ha="center", va="bottom", fontsize=8.5, color=PURP, fontweight="bold")

    # footer
    ax.add_patch(FancyBboxPatch((0.02, 0.015), 0.96, 0.058, boxstyle="round,pad=0.004,rounding_size=0.02",
                fc=NAVY, ec=NAVY, zorder=3))
    ax.text(0.5, 0.044,
            "data flow:   cloud $P_i$  →[A]→  $e_i$  →[B]→  $b_i(x)$   for all $i$,    then    "
            "$\\{b_i\\}$  →[C]→  $B(x)$          input dim to B  =  3 ($x_{\\mathrm{rel}}$) + 64 ($e_i$)  =  67",
            ha="center", va="center", fontsize=8.6, color="white")
    save(fig, "composite_barrier.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2) ONLINE CONTROLLER   (cbf_qp.py: BPCBFController.solve + project_safe + filter)
# ══════════════════════════════════════════════════════════════════════════════
def online_controller():
    fig, ax = new_ax(14.5, 5.2)
    ax.text(0.5, 0.955, "Online Safety Filter  ·  runs every control step  (cbf_qp.py)",
            ha="center", va="center", fontsize=15, fontweight="bold", color=NAVY)

    y, h = 0.24, 0.52
    # state
    box(ax, 0.008, 0.38, 0.11, 0.26, LGREY, GREY, round_=0.03)
    ax.text(0.063, 0.545, "current\nstate", ha="center", va="center", fontsize=9.5, color=INK, fontweight="bold")
    ax.text(0.063, 0.45, r"$x \in \mathbb{R}^{3}$", ha="center", va="center", fontsize=10.5, color=NAVY)

    # (a) guidance
    box(ax, 0.145, y, 0.185, h, LAMBER, AMBER, title="(a) Go-Around Guidance",
        tfs=9.6, lfs=7.8, align="left", lines=[
        r"$n=\nabla B/\|\nabla B\|$  (normal)",
        r"$t=(-n_y,\,n_x)$  (tangent)",
        r"flip $t$ if $t\!\cdot\!$pref $<0$",
        r"avoid moving-obstacle heading",
        r"$f_{\mathrm{eff}}=f_\theta(x,s)+g$"])

    # (b) QP
    box(ax, 0.345, y, 0.235, h, LBLUE, BLUE, title="(b) CLF–CBF QP  (OSQP)",
        tfs=9.6, lfs=7.7, align="left", lines=[
        r"$\min_{u,\varepsilon\geq 0}\ \| u\|^2+\lambda\varepsilon^2$",
        r"s.t. $\nabla B^{\!\top}(f_{\mathrm{eff}}\!+\!u)\geq-\gamma(B\!-\!m)$",
        r"$\qquad$ [CBF, hard]",
        r"$\nabla V^{\!\top}(f_{\mathrm{eff}}\!+\!u)\leq-\alpha V+\varepsilon$",
        r"$\qquad$ [CLF, soft]",
        r"$\gamma{=}3,\ \alpha{=}4,\ \lambda{=}0.5,\ m{=}8$mm"])

    # (c) project_safe
    box(ax, 0.595, y, 0.185, h, LPURP, PURP, title="(c) project_safe",
        tfs=9.6, lfs=7.7, align="left", lines=[
        "discrete-step guard on the",
        "LEARNED  $B$:",
        r"ensure $B(x+dt\,\dot x)\geq m$",
        "bisection over step frac.",
        "preserves direction",
        "(tangential go-around lives)"])

    # (d) analytic filter
    box(ax, 0.795, y, 0.20, h, LRED, RED, title="(d) Analytic SDF Filter",
        tfs=9.6, lfs=7.7, align="left", lines=[
        "HARD GUARANTEE",
        r"ensure $\mathrm{sdf}(x+dt\,\dot x)\geq 11$mm",
        "uses EXACT known geometry",
        "scale step to largest safe",
        "frac (keeps direction), or",
        "push out along $\\nabla\\mathrm{sdf}$"])

    # output
    box(ax, 0.008, 0.02, 0.987, 0.085, NAVY, NAVY, round_=0.015)
    ax.text(0.5, 0.062,
            "(a)+(b)+(c) use the LEARNED $B_\\phi$   |   (d) is the EXACT analytic backstop — the hard safety guarantee   |   output  $\\dot x = f_\\theta + u$,  applied via damped-LS Jacobian IK",
            ha="center", va="center", fontsize=8.8, color="white")

    for x0, x1 in [(0.118, 0.145), (0.33, 0.345), (0.58, 0.595), (0.78, 0.795)]:
        arrow(ax, (x0, 0.5), (x1, 0.5), color=NAVY, lw=2.4)
    arrow(ax, (0.995, 0.24), (0.9, 0.105), color=NAVY, lw=2.2, rad=-0.2)
    save(fig, "online_controller.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3) SYSTEM PIPELINE  (end to end)
# ══════════════════════════════════════════════════════════════════════════════
def system_pipeline():
    fig, ax = new_ax(14.5, 6.4)
    ax.text(0.5, 0.965, "System Overview  ·  from expert demonstrations to a pose-generalizing safe controller",
            ha="center", va="center", fontsize=14.5, fontweight="bold", color=NAVY)

    # OFFLINE band
    ax.add_patch(FancyBboxPatch((0.01, 0.5), 0.98, 0.4, boxstyle="round,pad=0.004,rounding_size=0.015",
                fc="#f6f8fb", ec="#c7d2df", lw=1.2, zorder=1))
    ax.text(0.028, 0.865, "OFFLINE  ·  training", ha="left", va="center", fontsize=9, color=BLUE, fontweight="bold")

    box(ax, 0.03, 0.56, 0.16, 0.26, "white", GREY, title="Expert demos", tfs=10, lfs=8, lines=[
        "MuJoCo FR3 rollouts", "needle-tip path $x^*(t)$", "critical-tissue scenes"])
    box(ax, 0.215, 0.56, 0.185, 0.26, LAMBER, AMBER, title="Augmentation", tfs=10, lfs=7.8, lines=[
        r"rot $\in[-\pi,\pi]$, scale $[0.65,1.4]$", r"translate $\pm 5$cm / obstacle",
        r"matched inflated-SDF labels", r"$\Delta = 10$mm tolerance buffer"])

    box(ax, 0.42, 0.56, 0.165, 0.26, LBLUE, BLUE, title=r"$f_\theta$  flow", tfs=10, lfs=7.8, lines=[
        "progress-conditioned DS", r"$\dot x=v(\tilde x,s)+K(x_{\mathrm{ref}}(s)-x)$",
        r"[3+1]→128→128→128→3"])
    box(ax, 0.605, 0.56, 0.165, 0.26, LGREY, GREY, title=r"$V_\theta$  CLF", tfs=10, lfs=7.8, lines=[
        r"$V(e)=\| e\|^2(1+\delta\,\mathrm{corr})$", r"$=0$ on demo path",
        "Softplus correction"])
    box(ax, 0.79, 0.56, 0.20, 0.26, LPURP, PURP, title=r"$B_\phi$  composite CBF", tfs=10, lfs=7.8, lines=[
        "PointNet enc → shared CBF", r"smooth-min over obstacle set",
        "learned, shape-independent"])

    # ONLINE band
    ax.add_patch(FancyBboxPatch((0.01, 0.06, ), 0.98, 0.36, boxstyle="round,pad=0.004,rounding_size=0.015",
                fc="#f6faf7", ec="#bfe0cd", lw=1.2, zorder=1))
    ax.text(0.028, 0.39, "ONLINE  ·  deployment  (1 kHz)", ha="left", va="center", fontsize=9, color=TEAL, fontweight="bold")

    box(ax, 0.03, 0.10, 0.15, 0.22, "white", GREY, title="state $x$", tfs=9.5, lfs=8, lines=["needle-tip", "from sim / FR3"])
    box(ax, 0.205, 0.10, 0.17, 0.22, LAMBER, AMBER, title="guidance $g$", tfs=9.5, lfs=7.8, lines=["tangential go-around", r"$\perp\nabla B$"])
    box(ax, 0.40, 0.10, 0.17, 0.22, LBLUE, BLUE, title="CLF–CBF QP", tfs=9.5, lfs=7.8, lines=["OSQP, per step", r"$\min\| u\|^2+\lambda\varepsilon^2$"])
    box(ax, 0.595, 0.10, 0.17, 0.22, LPURP, PURP, title="project_safe", tfs=9.5, lfs=7.8, lines=["discrete-step guard", "on learned $B$"])
    box(ax, 0.79, 0.10, 0.20, 0.22, LRED, RED, title="analytic backstop", tfs=9.5, lfs=7.8, lines=["exact sdf  ≥ 11 mm", "hard guarantee"])

    # arrows within bands
    for x0, x1 in [(0.19, 0.215), (0.585, 0.605), (0.77, 0.79)]:
        arrow(ax, (x0, 0.69), (x1, 0.69), color=BLUE, lw=1.8)
    arrow(ax, (0.40, 0.69), (0.42, 0.69), color=BLUE, lw=1.8)
    for x0, x1 in [(0.18, 0.205), (0.375, 0.40), (0.57, 0.595), (0.765, 0.79)]:
        arrow(ax, (x0, 0.21), (x1, 0.21), color=TEAL, lw=1.8)
    # offline -> online
    arrow(ax, (0.5, 0.5), (0.5, 0.42), color=NAVY, lw=2.6, style="-|>")
    ax.text(0.52, 0.46, "trained $f_\\theta,V_\\theta,B_\\phi$", ha="left", va="center", fontsize=8.5, color=NAVY)
    arrow(ax, (0.865, 0.10), (0.865, 0.02), color=TEAL, lw=2.0)
    ax.text(0.5, 0.028, r"safe velocity  $\dot x = f_\theta(x,s) + u$   →   damped-LS Jacobian IK   →   joint commands",
            ha="center", va="center", fontsize=9, color=NAVY, fontweight="bold")
    save(fig, "system_pipeline.png")


if __name__ == "__main__":
    system_pipeline()
    composite_barrier()
    online_controller()
    print("all diagrams rendered ->", OUT)
