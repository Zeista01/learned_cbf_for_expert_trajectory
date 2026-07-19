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

OUT = os.path.join(os.path.dirname(__file__), "architecture")
os.makedirs(OUT, exist_ok=True)

# ── palette ──────────────────────────────────────────────────────────────────
INK    = "#1f2933"
NAVY   = "#12324f"
BLUE   = "#2c6fb5"
LBLUE  = "#dbe9f6"
TEAL   = "#0f7d6b"
LTEAL  = "#d3efe9"
PURP   = "#6a4bb0"
LPURP  = "#e7e0f5"
AMBER  = "#c9821a"
LAMBER = "#fbeccd"
RED    = "#c0392b"
LRED   = "#f7dcd8"
GREY   = "#5b6169"
LGREY  = "#eef1f4"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "mathtext.fontset": "cm",
})


# ── layout helpers ───────────────────────────────────────────────────────────
def rbox(ax, x, y, w, h, fc, ec, lw=1.4, r=0.015, z=3):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, zorder=z)
    ax.add_patch(p)
    return p


def titled_box(ax, x, y, w, h, fc, ec, title, lines=(), tfs=10, lfs=8.2,
               th=0.07, align="center", body_color=INK, r=0.015):
    """Rounded box with a colored header band and evenly spaced body lines."""
    rbox(ax, x, y, w, h, fc, ec, r=r)
    # header band (square bottom corners hidden by overlap)
    ax.add_patch(FancyBboxPatch((x, y + h - th), w, th,
                 boxstyle=f"round,pad=0,rounding_size={r}",
                 fc=ec, ec=ec, lw=0, zorder=4))
    ax.add_patch(plt.Rectangle((x, y + h - th), w, th / 2, fc=ec, ec=ec,
                               lw=0, zorder=4))
    ax.text(x + w / 2, y + h - th / 2, title, ha="center", va="center",
            fontsize=tfs, color="white", fontweight="bold", zorder=6)
    body_lines(ax, x, y, w, h - th, lines, lfs=lfs, align=align,
               color=body_color)
    return x + w / 2, y + h / 2


def body_lines(ax, x, y, w, h, lines, lfs=8.2, align="center", color=INK):
    """Evenly space `lines` inside the rectangle (x, y, w, h)."""
    n = len(lines)
    if n == 0:
        return
    step = h / n
    for k, ln in enumerate(lines):
        yy = y + h - (k + 0.5) * step
        if align == "center":
            ax.text(x + w / 2, yy, ln, ha="center", va="center",
                    fontsize=lfs, color=color, zorder=6)
        else:
            ax.text(x + 0.014, yy, ln, ha="left", va="center",
                    fontsize=lfs, color=color, zorder=6)


def arrow(ax, p0, p1, color=NAVY, lw=2.2, style="-|>", rad=0.0):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=16,
                        color=color, lw=lw,
                        connectionstyle=f"arc3,rad={rad}", zorder=5,
                        shrinkA=1, shrinkB=1)
    ax.add_patch(a)


def row_positions(x0, x1, n, gap):
    """n equal-width boxes spanning [x0, x1] with `gap` between them."""
    w = (x1 - x0 - (n - 1) * gap) / n
    return [(x0 + i * (w + gap), w) for i in range(n)]


def new_ax(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white",
                pad_inches=0.15)
    plt.close(fig)
    print("saved", path)


# ══════════════════════════════════════════════════════════════════════════════
# 1) SYSTEM PIPELINE  (end to end)
# ══════════════════════════════════════════════════════════════════════════════
def system_pipeline():
    fig, ax = new_ax(14.5, 6.4)
    ax.text(0.5, 0.965,
            "System Overview  ·  from expert demonstrations to a pose-generalizing safe controller",
            ha="center", va="center", fontsize=14.5, fontweight="bold", color=NAVY)

    cols = row_positions(0.035, 0.965, 5, 0.026)

    # ── OFFLINE band ─────────────────────────────────────────────────────────
    rbox(ax, 0.012, 0.515, 0.976, 0.385, "#f6f8fb", "#c7d2df", lw=1.2, z=1)
    ax.text(0.035, 0.868, "OFFLINE  ·  training", ha="left", va="center",
            fontsize=9.5, color=BLUE, fontweight="bold")

    oy, oh = 0.555, 0.27
    specs_off = [
        ("white", GREY, "Expert demos",
         ["MuJoCo FR3 rollouts", r"needle-tip path $x^*(t)$",
          "critical-tissue scenes"]),
        (LAMBER, AMBER, "Augmentation",
         [r"rot $[-\pi,\pi]$ · scale $[0.65,1.4]$",
          r"translate $\pm 5\,$cm / obstacle",
          "matched inflated-SDF labels",
          r"$\Delta=10\,$mm tolerance buffer"]),
        (LBLUE, BLUE, r"$f_\theta$ · flow",
         ["progress-conditioned DS",
          r"$\dot x = v(\tilde x,s)+K(x_{\mathrm{ref}}(s)-x)$",
          r"$[3{+}1]{\to}128{\to}128{\to}128{\to}3$"]),
        (LGREY, GREY, r"$V_\theta$ · CLF",
         [r"$V(e)=\|e\|^2(1+\delta\,\mathrm{corr})$",
          r"$=0$ on demo path",
          "Softplus correction"]),
        (LPURP, PURP, r"$B_\phi$ · composite CBF",
         [r"PointNet enc $\to$ shared CBF",
          "smooth-min over obstacle set",
          "learned, shape-independent"]),
    ]
    for (xx, ww), (fc, ec, ti, ls) in zip(cols, specs_off):
        titled_box(ax, xx, oy, ww, oh, fc, ec, ti, ls, tfs=10.5, lfs=7.9,
                   th=0.065)
    ymid_off = oy + (oh - 0.065) / 2
    for i in range(4):
        x_r = cols[i][0] + cols[i][1]
        x_l = cols[i + 1][0]
        arrow(ax, (x_r, ymid_off), (x_l, ymid_off), color=BLUE, lw=2.0)

    # ── offline → online ─────────────────────────────────────────────────────
    arrow(ax, (0.5, 0.515), (0.5, 0.443), color=NAVY, lw=2.6)
    ax.text(0.515, 0.479, r"trained  $f_\theta,\ V_\theta,\ B_\phi$",
            ha="left", va="center", fontsize=9, color=NAVY)

    # ── ONLINE band ──────────────────────────────────────────────────────────
    rbox(ax, 0.012, 0.10, 0.976, 0.335, "#f6faf7", "#bfe0cd", lw=1.2, z=1)
    ax.text(0.035, 0.405, "ONLINE  ·  deployment  (1 kHz)", ha="left",
            va="center", fontsize=9.5, color=TEAL, fontweight="bold")

    ny, nh = 0.135, 0.235
    specs_on = [
        ("white", GREY, "state $x$",
         ["needle-tip position", "from sim / FR3"]),
        (LAMBER, AMBER, "guidance $g$",
         ["tangential go-around", r"$g \perp \nabla B$"]),
        (LBLUE, BLUE, "CLF–CBF QP",
         ["OSQP, per step", r"$\min\|u\|^2+\lambda\varepsilon^2$"]),
        (LPURP, PURP, "project_safe",
         ["discrete-step guard", "on learned $B$"]),
        (LRED, RED, "analytic backstop",
         [r"exact sdf $\geq 11\,$mm", "hard guarantee"]),
    ]
    for (xx, ww), (fc, ec, ti, ls) in zip(cols, specs_on):
        titled_box(ax, xx, ny, ww, nh, fc, ec, ti, ls, tfs=10.5, lfs=8.2,
                   th=0.062)
    ymid_on = ny + (nh - 0.062) / 2
    for i in range(4):
        x_r = cols[i][0] + cols[i][1]
        x_l = cols[i + 1][0]
        arrow(ax, (x_r, ymid_on), (x_l, ymid_on), color=TEAL, lw=2.0)

    # ── output bar ───────────────────────────────────────────────────────────
    x_last = cols[4][0] + cols[4][1] / 2
    arrow(ax, (x_last, ny), (x_last, 0.072), color=TEAL, lw=2.2)
    rbox(ax, 0.012, 0.008, 0.976, 0.062, NAVY, NAVY, r=0.012)
    ax.text(0.5, 0.039,
            r"safe velocity  $\dot x = f_\theta(x,s) + u$   →   damped-LS Jacobian IK   →   joint commands",
            ha="center", va="center", fontsize=9.5, color="white",
            fontweight="bold")
    save(fig, "system_pipeline.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2) COMPOSITE NEURAL BARRIER  B_phi   (models.py: CompositeBarrier)
# ══════════════════════════════════════════════════════════════════════════════
def composite_barrier():
    fig, ax = new_ax(14.0, 7.6)
    ax.text(0.5, 0.972, r"Learned Composite Neural Barrier  $B_\phi(x)$",
            ha="center", va="center", fontsize=16, fontweight="bold",
            color=NAVY)
    ax.text(0.5, 0.928,
            "permutation-invariant set encoder  •  one shared conditional CBF  •  smooth-min fusion",
            ha="center", va="center", fontsize=9.5, color=GREY, style="italic")

    P_Y, P_H, P_TH = 0.115, 0.775, 0.062
    panels = row_positions(0.025, 0.975, 3, 0.05)

    # ---- PART A : ObstacleEncoder (PointNet) ----
    (ax_x, ax_w) = panels[0]
    titled_box(ax, ax_x, P_Y, ax_w, P_H, LBLUE, BLUE,
               "A · ObstacleEncoder  (PointNet)", tfs=10.5, th=P_TH)
    ia_x, ia_w = ax_x + 0.025, ax_w - 0.05
    cxA = ax_x + ax_w / 2

    rbox(ax, ia_x, 0.715, ia_w, 0.095, "white", BLUE)
    body_lines(ax, ia_x, 0.715, ia_w, 0.095,
               [r"point cloud  $P_i \in \mathbb{R}^{K\times 2}$",
                r"$K$ interior pts, centered at $c_i$"], lfs=8.4)
    arrow(ax, (cxA, 0.715), (cxA, 0.688), color=BLUE, lw=1.6)
    ax.text(cxA, 0.669, "shared per-point MLP", ha="center", va="center",
            fontsize=8.2, color=NAVY, style="italic")
    for j, txt in enumerate(["Linear(2 → 128) + ReLU",
                             "Linear(128 → 128) + ReLU",
                             "Linear(128 → 64)"]):
        yy = 0.583 - j * 0.073
        rbox(ax, ia_x, yy, ia_w, 0.052, "white", BLUE)
        ax.text(cxA, yy + 0.026, txt, ha="center", va="center",
                fontsize=8.4, color=INK, zorder=6)
        if j:
            arrow(ax, (cxA, yy + 0.073), (cxA, yy + 0.052 + 0.001),
                  color=BLUE, lw=1.5)
    arrow(ax, (cxA, 0.437), (cxA, 0.405), color=BLUE, lw=1.6)
    ax.text(cxA, 0.386, r"feature map  $(K,\,64)$", ha="center", va="center",
            fontsize=8.2, color=GREY)
    rbox(ax, ia_x, 0.30, ia_w, 0.062, NAVY, NAVY)
    ax.text(cxA, 0.331, "Max-Pool over K", ha="center", va="center",
            fontsize=9.6, color="white", fontweight="bold", zorder=6)
    arrow(ax, (cxA, 0.30), (cxA, 0.272), color=BLUE, lw=1.6)
    rbox(ax, ia_x, 0.155, ia_w, 0.115, "#eaf2fb", BLUE)
    body_lines(ax, ia_x, 0.155, ia_w, 0.115,
               [r"$e_i \in \mathbb{R}^{64}$", "shape embedding",
                "(encodes rotation + scale)"], lfs=8.0)
    ax.texts[-3].set_fontsize(11); ax.texts[-3].set_color(NAVY)
    ax.texts[-3].set_fontweight("bold")
    ax.texts[-2].set_color(GREY); ax.texts[-1].set_color(GREY)

    # ---- PART B : ConditionalObstacleCBF ----
    (bx_x, bx_w) = panels[1]
    titled_box(ax, bx_x, P_Y, bx_w, P_H, LPURP, PURP,
               "B · ConditionalObstacleCBF  (shared)", tfs=10.5, th=P_TH)
    ib_x, ib_w = bx_x + 0.025, bx_w - 0.05
    cxB = bx_x + bx_w / 2
    half = (ib_w - 0.014) / 2

    rbox(ax, ib_x, 0.70, half, 0.11, "white", PURP)
    body_lines(ax, ib_x, 0.70, half, 0.11,
               [r"$x_{\mathrm{rel}}=\frac{x-c_i}{\sigma}$",
                r"$\in \mathbb{R}^{3}$"], lfs=8.6)
    rbox(ax, ib_x + half + 0.014, 0.70, half, 0.11, "white", PURP)
    body_lines(ax, ib_x + half + 0.014, 0.70, half, 0.11,
               [r"$e_i$  (from A)", r"$\in \mathbb{R}^{64}$"], lfs=8.6)
    arrow(ax, (ib_x + half / 2, 0.70), (cxB - 0.02, 0.664), color=PURP, lw=1.5)
    arrow(ax, (ib_x + half + 0.014 + half / 2, 0.70), (cxB + 0.02, 0.664),
          color=PURP, lw=1.5)

    rbox(ax, ib_x, 0.61, ib_w, 0.052, "#efe9fa", PURP)
    ax.text(cxB, 0.636, r"concat  →  $\mathbb{R}^{67}$", ha="center",
            va="center", fontsize=9, color=NAVY, fontweight="bold", zorder=6)
    for j, txt in enumerate(["Linear(67 → 256) + Tanh",
                             "Linear(256 → 256) + Tanh",
                             "Linear(256 → 256) + Tanh",
                             "Linear(256 → 1)"]):
        yy = 0.533 - j * 0.07
        fc = "#efe9fa" if j < 3 else "#c9b8ec"
        rbox(ax, ib_x, yy, ib_w, 0.05, fc, PURP)
        ax.text(cxB, yy + 0.025, txt, ha="center", va="center",
                fontsize=8.4, color=INK, zorder=6)
        y_above = 0.61 if j == 0 else 0.533 - (j - 1) * 0.07
        arrow(ax, (cxB, y_above), (cxB, yy + 0.05 + 0.001), color=PURP,
              lw=1.5)
    arrow(ax, (cxB, 0.323), (cxB, 0.272), color=PURP, lw=1.6)
    rbox(ax, ib_x, 0.155, ib_w, 0.115, "#e5dcf6", PURP)
    body_lines(ax, ib_x, 0.155, ib_w, 0.115,
               [r"$b_i(x) \in \mathbb{R}$",
                r"$>0$ outside  ·  $<0$ inside"], lfs=8.2)
    ax.texts[-2].set_fontsize(11); ax.texts[-2].set_color(PURP)
    ax.texts[-2].set_fontweight("bold")
    ax.texts[-1].set_color(GREY)

    # ---- PART C : Smooth-Min fusion ----
    (cx_x, cx_w) = panels[2]
    titled_box(ax, cx_x, P_Y, cx_w, P_H, LTEAL, TEAL,
               "C · Smooth-Min Fusion", tfs=10.5, th=P_TH)
    ic_x, ic_w = cx_x + 0.025, cx_w - 0.05
    cxC = cx_x + cx_w / 2

    ax.text(cxC, 0.79, r"run A + B for every obstacle $i$", ha="center",
            va="center", fontsize=8.4, color=NAVY, style="italic")
    for j, (yy, lbl) in enumerate([(0.712, r"$b_1(x)$"),
                                   (0.655, r"$b_2(x)$"),
                                   (0.556, r"$b_M(x)$")]):
        rbox(ax, ic_x, yy, ic_w, 0.045, "#e6f4f0", TEAL)
        ax.text(cxC, yy + 0.0225, lbl, ha="center", va="center",
                fontsize=9.2, color=INK, zorder=6)
    ax.text(cxC, 0.627, r"$\vdots$", ha="center", va="center", fontsize=12,
            color=TEAL)
    arrow(ax, (cxC, 0.556), (cxC, 0.522), color=TEAL, lw=1.7)

    rbox(ax, ic_x, 0.385, ic_w, 0.135, TEAL, TEAL)
    body_lines(ax, ic_x, 0.385, ic_w, 0.135,
               ["Smooth-Min  (CN-CBF Eq. 18)",
                r"$B(x)=-\frac{1}{\beta}\log\sum_i e^{-\beta\,b_i(x)}$",
                r"$\beta=1000$ · under-approx. of $\min_i b_i$"],
               lfs=7.8, color="white")
    ax.texts[-3].set_fontweight("bold")
    ax.texts[-2].set_fontsize(11)
    ax.texts[-1].set_color("#d7f0ea")
    arrow(ax, (cxC, 0.385), (cxC, 0.352), color=TEAL, lw=1.7)

    rbox(ax, ic_x, 0.25, ic_w, 0.10, "#e6f4f0", TEAL)
    body_lines(ax, ic_x, 0.25, ic_w, 0.10,
               [r"$B(x) \in \mathbb{R}$",
                "composite barrier over the obstacle set"], lfs=7.8)
    ax.texts[-2].set_fontsize(12); ax.texts[-2].set_color(TEAL)
    ax.texts[-2].set_fontweight("bold")
    ax.texts[-1].set_color(GREY)
    arrow(ax, (cxC, 0.25), (cxC, 0.225), color=AMBER, lw=1.5)
    rbox(ax, ic_x, 0.145, ic_w, 0.078, LAMBER, AMBER)
    body_lines(ax, ic_x, 0.145, ic_w, 0.078,
               ["trained via inflated-SDF regression",
                "+ pose / scale / translation augmentation"],
               lfs=7.8, color="#7a520f")

    # ---- panel-to-panel flow arrows ----
    y_flow = 0.2125          # aligned with the e_i / b_i output boxes
    arrow(ax, (ia_x + ia_w, y_flow), (bx_x, y_flow), color=NAVY, lw=2.4)
    ax.text((ia_x + ia_w + bx_x) / 2, y_flow + 0.032, r"$e_i$",
            ha="center", va="center", fontsize=10, color=NAVY,
            fontweight="bold")
    arrow(ax, (ib_x + ib_w, y_flow), (cx_x, y_flow), color=NAVY, lw=2.4)
    ax.text((ib_x + ib_w + cx_x) / 2, y_flow + 0.032, r"$\times M$",
            ha="center", va="center", fontsize=10, color=NAVY,
            fontweight="bold")

    # ---- footer ----
    rbox(ax, 0.025, 0.012, 0.95, 0.062, NAVY, NAVY, r=0.012)
    ax.text(0.5, 0.043,
            r"data flow:   cloud $P_i$ →[A]→ $e_i$ →[B]→ $b_i(x)$  for all $i$,  then  $\{b_i\}$ →[C]→ $B(x)$"
            r"        ·        input dim to B  =  3 ($x_{\mathrm{rel}}$) + 64 ($e_i$)  =  67",
            ha="center", va="center", fontsize=8.8, color="white")
    save(fig, "composite_barrier.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3) ONLINE CONTROLLER   (cbf_qp.py: BPCBFController.solve + project_safe + filter)
# ══════════════════════════════════════════════════════════════════════════════
def online_controller():
    fig, ax = new_ax(14.5, 5.4)
    ax.text(0.5, 0.945,
            "Online Safety Filter  ·  runs every control step  (cbf_qp.py)",
            ha="center", va="center", fontsize=15, fontweight="bold",
            color=NAVY)

    sy, sh, th = 0.24, 0.58, 0.075
    ymid = sy + (sh - th) / 2

    # state box, vertically centered on the stage row
    st_h = 0.26
    rbox(ax, 0.018, ymid - st_h / 2, 0.115, st_h, LGREY, GREY, r=0.02)
    body_lines(ax, 0.018, ymid - st_h / 2, 0.115, st_h,
               ["current state", r"$x \in \mathbb{R}^{3}$"], lfs=9.5)
    ax.texts[-2].set_fontweight("bold")
    ax.texts[-1].set_fontsize(10.5); ax.texts[-1].set_color(NAVY)

    stages = row_positions(0.165, 0.978, 4, 0.022)
    specs = [
        (LAMBER, AMBER, "(a) Go-Around Guidance",
         [r"$n=\nabla B/\|\nabla B\|$  (normal)",
          r"$t=(-n_y,\ n_x)$  (tangent)",
          r"flip $t$ if $t\cdot$pref $<0$",
          "avoid moving-obstacle heading",
          r"$f_{\mathrm{eff}} = f_\theta(x,s) + g$"]),
        (LBLUE, BLUE, "(b) CLF–CBF QP  (OSQP)",
         [r"$\min_{u,\,\varepsilon\geq 0}\ \|u\|^2+\lambda\varepsilon^2$",
          r"s.t.  $\nabla B^{\top}(f_{\mathrm{eff}}{+}u)\geq-\gamma(B{-}m)$",
          "        [CBF, hard]",
          r"       $\nabla V^{\top}(f_{\mathrm{eff}}{+}u)\leq-\alpha V+\varepsilon$",
          "        [CLF, soft]",
          r"$\gamma{=}3$,  $\alpha{=}4$,  $\lambda{=}0.5$,  $m{=}8\,$mm"]),
        (LPURP, PURP, "(c) project_safe",
         ["discrete-step guard",
          "on the LEARNED $B$:",
          r"ensure $B(x+dt\,\dot x)\geq m$",
          "bisection over step fraction",
          "preserves direction",
          "(go-around survives)"]),
        (LRED, RED, "(d) Analytic SDF Filter",
         ["HARD GUARANTEE",
          r"ensure $\mathrm{sdf}(x+dt\,\dot x)\geq 11\,$mm",
          "uses EXACT known geometry",
          "scale step to largest safe",
          "fraction (keeps direction),",
          r"or push out along $\nabla\mathrm{sdf}$"]),
    ]
    for (xx, ww), (fc, ec, ti, ls) in zip(stages, specs):
        titled_box(ax, xx, sy, ww, sh, fc, ec, ti, ls, tfs=9.8, lfs=7.8,
                   th=th, align="left")

    # arrows: state -> (a) -> (b) -> (c) -> (d)
    arrow(ax, (0.133, ymid), (stages[0][0], ymid), lw=2.4)
    for i in range(3):
        x_r = stages[i][0] + stages[i][1]
        x_l = stages[i + 1][0]
        arrow(ax, (x_r, ymid), (x_l, ymid), lw=2.4)

    # (d) -> output bar
    x_d = stages[3][0] + stages[3][1] / 2
    arrow(ax, (x_d, sy), (x_d, 0.135), lw=2.4)
    rbox(ax, 0.018, 0.025, 0.96, 0.105, NAVY, NAVY, r=0.015)
    body_lines(ax, 0.018, 0.025, 0.96, 0.105,
               [r"(a) + (b) + (c) use the LEARNED $B_\phi$    |    (d) is the EXACT analytic backstop — the hard safety guarantee",
                r"output   $\dot x = f_\theta + u$,   applied via damped-LS Jacobian IK"],
               lfs=8.8, color="white")
    save(fig, "online_controller.png")


if __name__ == "__main__":
    system_pipeline()
    composite_barrier()
    online_controller()
    print("all diagrams rendered ->", OUT)
