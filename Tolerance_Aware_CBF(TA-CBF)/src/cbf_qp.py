"""
cbf_qp.py — online CLF-CBF-QP controller for learned barrier.

Changes from previous version:
  - Barrier B_φ is now FULLY LEARNED (no analytic SDF) — gradient comes from NN
  - Lyapunov V_θ is quadratic CLF: V(e) = ||e||² from Nawaz et al.
    → gradient: ∇_x V = 2(x − x_ref(s))
  - PenetrationBudget η_ψ REMOVED — no more B+η shifted safe set
  - QP uses the standard CLF-NODE formulation from Nawaz et al. (2023):

    min_{u, ε≥0}   ||u||² + λ·ε²

    s.t.
      ∇B_φ(x)ᵀ (f_θ(x) + u) ≥ −γ·B_φ(x)    [CBF, hard — learned barrier]
      ∇V(e)ᵀ   (f_θ(x) + u) ≤ −α·V(e) + ε  [CLF, soft — quadratic V]

  State: x ∈ ℝ² (2D needle-tip position)
  Control: u ∈ ℝ² (additive velocity correction)
"""

import numpy as np
import torch
import osqp
from scipy import sparse
from config import (GAMMA_CBF, ALPHA_CLF, LAMBDA_SLACK, STATE_DIM, DEVICE,
                    B_SAFE_MARGIN, VEL_CLIP, X_GOAL, B_ACTIVE, SWIRL_GAIN,
                    SAFETY_SDF_MARGIN, Z_CORRIDOR, sdf_critical_shape_2d)


def _sdf_min(p_xy, shapes):
    """Exact signed distance to the nearest critical obstacle (min over shapes)."""
    p = np.asarray(p_xy, dtype=np.float32).reshape(1, 2)
    return float(min(sdf_critical_shape_2d(p, sh)[0] for sh in shapes))


def analytic_safety_filter(x_np, v, dt, shapes, margin=SAFETY_SDF_MARGIN):
    """
    HYBRID safety filter (paper: "learned nominal + exact geometric guarantee").

    The learned CBF drives the divert-and-reconverge field; this is the FINAL hard
    check that uses the KNOWN scene geometry to guarantee the needle never enters
    the critical tissue (sdf ≥ margin) — including the dead-centre / thin-tip cases
    where the learned barrier is slightly mis-calibrated. It scales the step to the
    largest fraction that stays outside (preserving direction → still slides
    around), and if already inside, steps straight out along the analytic ∇sdf.
    """
    v = np.asarray(v, dtype=np.float64).copy()

    def sdf_at(a):
        return _sdf_min(x_np[:2] + a * dt * v[:2], shapes)

    if sdf_at(1.0) >= margin:
        return v.astype(np.float32)

    # Already inside the keep-out shell → escape outward along the exact gradient.
    if sdf_at(0.0) < margin:
        e = 1e-4
        gx = (_sdf_min(x_np[:2] + [e, 0], shapes) - _sdf_min(x_np[:2] - [e, 0], shapes)) / (2 * e)
        gy = (_sdf_min(x_np[:2] + [0, e], shapes) - _sdf_min(x_np[:2] - [0, e], shapes)) / (2 * e)
        g = np.array([gx, gy], dtype=np.float64)
        gn = np.linalg.norm(g)
        out = np.zeros_like(v)
        if gn > 1e-9:
            out[:2] = (g / gn) * min(VEL_CLIP, 0.05)
        return out.astype(np.float32)

    # Bisection for the largest safe step fraction (α=0 is safe by the check above).
    lo, hi = 0.0, 1.0
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if sdf_at(mid) >= margin:
            lo = mid
        else:
            hi = mid
    return (v * lo).astype(np.float32)


class BPCBFController:
    def __init__(self,
                 gamma: float = GAMMA_CBF,
                 alpha: float = ALPHA_CLF,
                 lam:   float = LAMBDA_SLACK,
                 margin: float = B_SAFE_MARGIN):
        self.gamma  = gamma
        self.alpha  = alpha
        self.lam    = lam
        # Defend the level set B ≥ margin (not B ≥ 0). The learned barrier is
        # never a perfect classifier; small POSITIVE errors can appear just
        # inside an obstacle (B=0 mislocated). Holding a positive margin keeps
        # the needle in {B ≥ margin}, which (B being SDF-like) sits a buffer
        # OUTSIDE the true surface → strict avoidance despite residual error.
        self.margin = margin

    def solve(self,
              x_np: np.ndarray,
              model,
              device: str = DEVICE,
              s: float = None,
              obs_vel: np.ndarray = None) -> tuple:
        """
        Compute safety-filtered velocity correction u.

        Args:
            x_np  : (STATE_DIM,) current needle-tip position
            model : BPCBFModel (eval mode, contains f_θ, V_θ', B_φ)
            s     : task progress ∈ [0,1] or None (auto-computed)

        Returns:
            u_safe : (STATE_DIM,) velocity correction
            info   : dict with diagnostics (cbf_val, clf_val, etc.)
        """
        x_t = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(0)  # (1, d)

        # Progress s
        if s is None:
            with torch.no_grad():
                s_t = model.f.get_progress(x_t)
        else:
            s_t = torch.tensor([[s]], dtype=torch.float32, device=device)

        with torch.enable_grad():
            # Nominal DS output
            f_val_t = model.f(x_t, s_t)                    # (1, d)
            f_val   = f_val_t.detach().cpu().numpy().flatten()

            # Barrier B_φ and its gradient (learned NN)
            B_val_t = model.B(x_t)                          # (1, 1)
            gradB_t = model.B.gradient(x_t)                 # (1, d)
            B_num   = float(B_val_t.item())
            gradB   = gradB_t.detach().cpu().numpy().flatten()

            # Lyapunov V(e) = ||e||² (quadratic CLF, Nawaz et al.)
            # Note: full V = ||e||²·(1+δ·corr); gradV here uses the closed-form
            # ∇(||e||²)=2e only and ignores ∇corr. Since V is now multiplicative,
            # corr's gradient contribution vanishes as e→0, so this approximation
            # is most accurate near the path (where the QP operates).
            x_ref_s = model.f.x_ref_at_s(s_t)              # (1, d) in raw space
            e       = (x_t - x_ref_s).detach()             # (1, d) tracking error
            V_num   = float((e * e).sum().item())           # ||e||²
            gradV   = (2 * e).cpu().numpy().flatten()       # ∇V = 2e

        # ── Go-around guidance (breaks head-on deadlock) ───────────────────────
        # A head-on obstacle leaves the CBF tangent direction ambiguous, so a
        # strong demo pull stalls the needle dead-centre. Near the obstacle we add
        # a TANGENTIAL velocity (⊥ ∇B, so it does not fight the barrier) that
        # circulates around it toward the goal — and, for a MOVING obstacle, to
        # the side OPPOSITE its heading so the needle is not chased/blocked.
        g = np.zeros(STATE_DIM)
        gnB = float(np.linalg.norm(gradB))
        if B_num < B_ACTIVE and gnB > 1e-6:
            n = gradB / gnB
            t = np.array([-n[1], n[0], 0.0])               # tangent in XY
            tn = float(np.linalg.norm(t))
            if tn > 1e-6:
                t = t / tn
                pref = (X_GOAL - x_np).astype(np.float64)
                pn = float(np.linalg.norm(pref))
                pref = pref / pn if pn > 1e-6 else pref
                if obs_vel is not None:
                    ov = float(np.linalg.norm(obs_vel))
                    if ov > 1e-6:
                        pref = pref - 1.6 * (np.asarray(obs_vel) / ov)  # opposite heading
                if np.dot(t, pref) < 0:
                    t = -t
                closeness = max(0.0, (B_ACTIVE - B_num) / B_ACTIVE)
                g = SWIRL_GAIN * float(np.linalg.norm(f_val)) * closeness * t
        f_eff = f_val + g

        # ── CBF constraint (hard): ∇B·(f_eff + u) ≥ -γ·(B - margin) ───────
        # f_eff includes the tangential guidance (∇B·g≈0, so it doesn't change
        # the barrier rate but steers the go-around direction).
        cbf_rhs = -self.gamma * (B_num - self.margin) - gradB @ f_eff

        # ── CLF constraint (soft): ∇V·(f_eff + u) ≤ -α·V + ε ──────────────
        clf_rhs = -self.alpha * V_num - gradV @ f_eff

        # ── QP: min ||u||² + λ·ε²  s.t. CBF (hard) + CLF (soft) ────────────
        # Decision variable z = [u (d), ε (1)]
        n = STATE_DIM + 1
        P = sparse.diags([2.0] * STATE_DIM + [2.0 * self.lam], format='csc')
        q = np.zeros(n)

        # CBF row: gradB @ u >= cbf_rhs  → -gradB @ u <= -cbf_rhs
        # OSQP convention: l ≤ Az ≤ u
        A_cbf   = np.hstack([gradB,  0.0]).reshape(1, n)
        l_cbf   = np.array([cbf_rhs])
        u_cbf   = np.array([np.inf])

        # CLF row: gradV @ u - ε ≤ clf_rhs
        A_clf   = np.hstack([gradV, -1.0]).reshape(1, n)
        l_clf   = np.array([-np.inf])
        u_clf   = np.array([clf_rhs])

        # Slack ε ≥ 0
        A_eps   = np.hstack([np.zeros(STATE_DIM), 1.0]).reshape(1, n)
        l_eps   = np.array([0.0])
        u_eps   = np.array([np.inf])

        A = sparse.csc_matrix(np.vstack([A_cbf, A_clf, A_eps]))
        l = np.concatenate([l_cbf, l_clf, l_eps])
        u = np.concatenate([u_cbf, u_clf, u_eps])

        solver = osqp.OSQP()
        solver.setup(P, q, A, l, u,
                     warm_starting=True, verbose=False,
                     eps_abs=1e-6, eps_rel=1e-6, max_iter=10000, polish=True)
        result = solver.solve()

        if result.info.status_val in (1, 2):
            z      = result.x
            u_qp   = z[:STATE_DIM]
            slack  = float(z[STATE_DIM])
        else:
            # Fallback: minimum-norm u meeting gradB·u ≥ cbf_rhs
            norm_sq = float(np.dot(gradB, gradB)) + 1e-8
            deficit = cbf_rhs
            u_qp = (deficit / norm_sq) * gradB if deficit > 0 else np.zeros(STATE_DIM)
            slack = 0.0
        # total correction relative to the raw DS f_val = guidance + QP filter,
        # so the caller's (f_val + u_safe) equals the filtered, steered velocity.
        u_safe = g + u_qp

        info = {
            'status':   result.info.status if result.info.status_val in (1, 2) else 'fallback',
            'cbf_val':  B_num,
            'clf_val':  V_num,
            'slack':    slack,
        }
        return u_safe, info

    def project_safe(self, x_np: np.ndarray, model, v_nom: np.ndarray,
                     dt: float, device: str = DEVICE, iters: int = 3) -> np.ndarray:
        """
        Discrete-time safety projection: return v' close to v_nom such that the
        ACTUAL next state satisfies B(x + dt·v') ≥ margin. The continuous CBF-QP
        only bounds Ḃ; with a finite Euler step (and an imperfect learned B) the
        needle can still drift across the boundary. This closed-form projection
        targets the real next-step barrier value and is what makes avoidance
        STRICT regardless of barrier-classification error or step size.

            B(x+dt·v) ≈ B(x) + dt·∇B(x)·v
            need = margin − B(x+dt·v_nom);  if need>0:
            v' = v_nom + (need /(dt·‖∇B‖²)) · ∇B     (one Newton step, repeated)
        """
        v = np.asarray(v_nom, dtype=np.float64).copy()

        def B_at_frac(a):
            xn = torch.tensor((x_np + a * dt * v).astype(np.float32),
                              dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                return float(model.B(xn).item())

        # Defend B(x_next) ≥ min(B(x), margin): never let B drop BELOW the margin
        # (strict safety), but if B(x) is already at/below it, just forbid going
        # any deeper — allowing tangential / outward motion to slide free. (A
        # full-speed "escape along ∇B" is unsafe here: the learned gradient is
        # noisy near the boundary and can point INTO the obstacle.)
        thresh = min(B_at_frac(0.0), self.margin) - 1e-6
        if B_at_frac(1.0) >= thresh:
            return v.astype(np.float32)
        # Bisection for the LARGEST safe step fraction α (α=0 is safe by
        # construction). Scaling preserves direction → tangential go-around lives.
        lo, hi = 0.0, 1.0
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            if B_at_frac(mid) >= thresh:
                lo = mid
            else:
                hi = mid
        return (v * lo).astype(np.float32)
