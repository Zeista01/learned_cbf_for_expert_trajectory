"""
models.py — three neural networks for the revised TA-CBF framework.

Architecture changes per mentor feedback:
  1. f_θ  (ProgressConditionedDS) — unchanged: demo path as attractor
  2. V_θ  (LyapunovNet)           — REPLACED: quadratic CLF V(e)=||e||² from
                                    Nawaz et al. instead of ICNN.
                                    Small learned correction adapts level sets.
  3. B_φ  (BarrierNet)            — REPLACED: fully learned MLP barrier (S2-NNDS
                                    style). NO analytic SDF, NO residual.
                                    Critical tissue shapes are non-geometric
                                    so the barrier is learned entirely from labels.
  4. η_ψ  (PenetrationBudget)     — REMOVED (not needed with learned barrier)

References:
  - Nawaz et al. 2023: CLF quadratic V(e) = ||e||²
  - S2-NNDS (Binny et al.): learned V_θ' and B_φ, Algorithm 2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import STATE_DIM, F_HIDDEN, V_HIDDEN, B_HIDDEN, X_GOAL, DEMO_WAYPOINTS, N_POINTS


# ── Normalisation mixin ───────────────────────────────────────────────────────

class _Normed(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('in_mean', torch.zeros(STATE_DIM))
        self.register_buffer('in_std',  torch.ones(STATE_DIM))

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.in_mean) / self.in_std


# ── 1. Progress-Conditioned Dynamical System f_θ ─────────────────────────────

class ProgressConditionedDS(_Normed):
    """
    ẋ = f_θ(x, s) = v_net(x̃, s)  +  K · (x_ref(s) − x)
                     └ feed-fwd ┘     └──── feedback toward demo ────┘

    Making the ENTIRE demo path an attractor (not just the goal) as requested.
    s ∈ [0,1] = task progress; x_ref(s) = mean demo path at progress s.

    Architecture: [STATE_DIM+1 → 128 → 128 → 128 → STATE_DIM], Tanh.
    """
    def __init__(self, ref_path: np.ndarray = None, K: float = 3.0):
        super().__init__()
        dims   = [STATE_DIM + 1] + F_HIDDEN + [STATE_DIM]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].bias)
        nn.init.uniform_(self.net[-1].weight, -1e-3, 1e-3)

        self.K = K
        REF_N = N_POINTS
        if ref_path is None:
            ref_path = np.tile(X_GOAL, (REF_N, 1)).astype(np.float32)
        self.register_buffer('ref_path',
                             torch.tensor(ref_path, dtype=torch.float32))

    def set_reference(self, ref_path: np.ndarray):
        """Install the demo reference path. ref_path: (R, STATE_DIM)."""
        self.ref_path = torch.tensor(ref_path, dtype=torch.float32,
                                     device=self.ref_path.device)

    def x_ref_at_s(self, s: torch.Tensor) -> torch.Tensor:
        """Linearly interpolate the reference path at s ∈ [0,1]. s: (B, 1) → (B, d)."""
        R   = self.ref_path.shape[0]
        idx = s.clamp(0, 1).squeeze(-1) * (R - 1)
        lo  = idx.floor().long().clamp(0, R - 1)
        hi  = (lo + 1).clamp(0, R - 1)
        w   = (idx - lo.float()).unsqueeze(-1)
        return (1 - w) * self.ref_path[lo] + w * self.ref_path[hi]

    def get_progress(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute s from nearest point on reference path.
        x: (B, d) → s: (B, 1)
        """
        diff  = x.unsqueeze(1) - self.ref_path.unsqueeze(0)  # (B, R, d)
        dists = diff.norm(dim=-1)                              # (B, R)
        idx   = dists.argmin(dim=-1).float()                   # (B,)
        s     = idx / max(self.ref_path.shape[0] - 1, 1)
        return s.unsqueeze(-1)                                 # (B, 1)

    def _prep_s(self, x: torch.Tensor, s) -> torch.Tensor:
        if s is None:
            return self.get_progress(x)
        if not torch.is_tensor(s):
            return torch.full((x.shape[0], 1), float(s),
                              device=x.device, dtype=x.dtype)
        if s.dim() == 0:
            return s.expand(x.shape[0]).unsqueeze(-1)
        if s.dim() == 1:
            return s.unsqueeze(-1)
        return s

    def forward(self, x: torch.Tensor, s=None) -> torch.Tensor:
        """
        x: (B, d) raw (un-normalised) position
        s: (B, 1) or scalar or None
        Returns: dx/dt (B, d) in raw units
        """
        s         = self._prep_s(x, s)
        x_norm    = self.norm(x)
        inp       = torch.cat([x_norm, s], dim=-1)
        v_ff      = self.net(inp)
        x_ref_s   = self.x_ref_at_s(s)
        v_fb      = self.K * (x_ref_s - x)
        return v_ff + v_fb

    def predict_velocity(self, x_np: np.ndarray, s: float = None) -> np.ndarray:
        """x_np: (d,) numpy → (d,) numpy."""
        with torch.no_grad():
            x_t = torch.tensor(x_np[None], dtype=torch.float32)
            v = self.forward(x_t, s=s)
            return v[0].numpy()


# ── 2. Lyapunov certificate: Quadratic CLF (Nawaz et al.) ───────────────────

class LyapunovNet(_Normed):
    """
    V_θ(x, s) = ||e||² · (1 + δ · correction(e, s))
    where e = x − x_ref(s) is the tracking error along the demo path.

    The quadratic ||e||² is the CLF from Nawaz et al. (2023):
      - V(x_ref(s)) = 0  EXACTLY by construction (||e||²=0 forces V=0
        regardless of correction(0,s), since the correction is multiplicative)
      - V(x) > 0 elsewhere (correction >= 0 via Softplus, so 1+delta*corr > 0)
      - Penalises deviation from the demo path (not just goal distance)

    A small Softplus correction (δ ≪ 1) locally reweights the quadratic bowl
    to adapt to the corridor geometry, while keeping V dominated by the
    quadratic term and guaranteeing V(x_ref(s))=0 with no extra forward pass.

    This replaces the ICNN from the old code, per mentor's instruction.
    """
    def __init__(self, delta: float = 0.05):
        super().__init__()
        self.delta = delta
        dims   = [STATE_DIM + 1] + V_HIDDEN + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.Softplus())
        layers.append(nn.Softplus())   # last layer: non-negative correction
        self.corr = nn.Sequential(*layers)
        self.register_buffer('x_goal',
                             torch.tensor(X_GOAL, dtype=torch.float32))

    def forward(self, x: torch.Tensor, x_ref_s: torch.Tensor,
                s: torch.Tensor) -> torch.Tensor:
        """
        x:       (B, d) — current position
        x_ref_s: (B, d) — reference position at current progress s
        s:       (B, 1)
        Returns V: (B, 1)
        """
        e      = x - x_ref_s                                    # tracking error
        v_quad = (e * e).sum(dim=-1, keepdim=True)              # ||e||²
        e_norm = self.norm(x) - self.norm(x_ref_s)
        v_corr = self.corr(torch.cat([e_norm, s], dim=-1))      # >= 0 (Softplus)
        return v_quad * (1.0 + self.delta * v_corr)             # (B, 1), =0 when e=0

    def gradient(self, x: torch.Tensor, x_ref_s: torch.Tensor,
                 s: torch.Tensor) -> torch.Tensor:
        """∇_x V via autograd."""
        xr = x.detach().requires_grad_(True)
        V  = self.forward(xr, x_ref_s, s)
        return torch.autograd.grad(V.sum(), xr, create_graph=True)[0]


# ── 3. Barrier certificate: Fully Learned NN (S2-NNDS approach) ───────────────

class BarrierNet(_Normed):
    """
    B_φ(x) — fully learned barrier certificate (S2-NNDS style).
    Independent of obstacle shape/geometry — generalizes to any
    safe/unsafe point-cloud labeling, including changing environments.

    Sign convention (S2-NNDS paper):
        B_φ(x) > δ  →  safe   (X_0: demo path neighbourhood, free space)
        B_φ(x) < -δ →  unsafe (X_u: inside critical zones)

    Architecture: deep MLP with Tanh activations, NO analytic anchor.
    """
    def __init__(self):
        super().__init__()
        dims   = [STATE_DIM] + B_HIDDEN + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].bias)
        nn.init.uniform_(self.net[-1].weight, -1e-2, 1e-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, d) → B: (B, 1)"""
        return self.net(self.norm(x))

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        """∇_x B(x) via autograd. x: (B, d) → (B, d)."""
        xr = x.detach().requires_grad_(True)
        B  = self.forward(xr)
        return torch.autograd.grad(B.sum(), xr, create_graph=True)[0]


# ── 3b. Composite Neural Barrier (CN-CBF style) ──────────────────────────────
#
# Generalizes the barrier to an ARBITRARY number of obstacles of ARBITRARY
# shape, learned (no analytic SDF at inference):
#
#   per obstacle i:  b_i(x) = cbf_mlp([ (x - c_i)/scale ,  enc(cloud_i) ])
#   composite     :  B(x)   = -1/β · ln Σ_i exp(-β · b_i(x))          (smooth-min)
#
# - enc() is a permutation-invariant PointNet over a point-cloud of the
#   obstacle (centered at its centroid) → a shape embedding. This is what
#   makes the barrier shape-independent and reusable on unseen shapes.
# - smooth-min (CN-CBF Eq. 18) composes the per-obstacle CBFs: separate
#   boundaries when obstacles are far apart, a merged/attached boundary when
#   they are very close (β controls the blend). It is a conservative
#   under-approximation of the true min, so the composite safe set never
#   exceeds the union of per-obstacle safe sets.
# - moving/slow-changing obstacles are handled by conditioning on the
#   RELATIVE coordinate (x - c_i) and re-evaluating each control step.

EMB_DIM      = 64       # obstacle shape-embedding dimension (raised: must encode
                        # star/crescent/blob/kidney/L accurately under any rotation)
BARRIER_BETA = 1000.0   # smooth-min sharpness. The composite sits below the true
                        # min by up to ln(M)/β; for the LEARNED CBF (range ~±0.02 m)
                        # this offset must be ≪ δ, so β must be large:
                        # ln(6)/1000 ≈ 1.8 mm ≪ δ=10 mm. (β=200 gave a 9 mm offset
                        # that dragged every safe point negative.) Higher β =
                        # sharper/closer to true min; lower = smoother/more merged.


class ObstacleEncoder(nn.Module):
    """
    PointNet-style permutation-invariant shape encoder.
    cloud P: (..., K, 2) points sampled from one obstacle, centered at its
    centroid → embedding e: (..., EMB_DIM). Max-pool makes it invariant to
    point ordering and robust to variable K (any obstacle = any point set).
    """
    def __init__(self, in_dim: int = 2, emb_dim: int = EMB_DIM, hidden: int = 128):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, P: torch.Tensor) -> torch.Tensor:
        feat = self.point_mlp(P)               # (..., K, emb_dim)
        return feat.max(dim=-2).values         # (..., emb_dim)


class ConditionalObstacleCBF(nn.Module):
    """
    Single-obstacle conditional CBF: b_i(x) = mlp([ (x-c_i)/scale , e_i ]).
    b_i > 0 outside obstacle i, < 0 inside. Shared across all obstacles —
    the obstacle identity enters only through the shape embedding e_i and the
    relative coordinate, so ONE network serves any number of obstacles.
    """
    def __init__(self, state_dim: int = STATE_DIM, emb_dim: int = EMB_DIM,
                 hidden=(256, 256, 256)):
        super().__init__()
        dims   = [state_dim + emb_dim] + list(hidden) + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].bias)
        nn.init.uniform_(self.net[-1].weight, -1e-2, 1e-2)
        self.register_buffer('scale', torch.ones(state_dim))

    def forward(self, x_rel: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_rel / self.scale, e], dim=-1))


class CompositeBarrier(nn.Module):
    """
    Composite neural CBF over an arbitrary obstacle SET (CN-CBF smooth-min).

    obstacles: list of dicts, each with
        'cloud'  : (K, 2) tensor — points of the obstacle, centered at centroid
        'center' : (state_dim,) tensor — obstacle centroid in world frame
    forward(x, obstacles) → B: (B, 1)
    """
    def __init__(self, state_dim: int = STATE_DIM, emb_dim: int = EMB_DIM,
                 beta: float = BARRIER_BETA):
        super().__init__()
        self.encoder = ObstacleEncoder(in_dim=2, emb_dim=emb_dim)
        self.cbf     = ConditionalObstacleCBF(state_dim, emb_dim)
        self.beta    = beta
        self._obstacles = None   # current obstacle set (env config, not learned)

    def set_scale(self, std: torch.Tensor):
        """Set the relative-coordinate normalizer (typically input std)."""
        self.cbf.scale.copy_(std)

    def set_obstacles(self, obstacles: list):
        """
        Install the current obstacle set so model.B(x) / .gradient(x) work with
        the same single-argument interface as the old BarrierNet (used by the
        QP and plotting). For dynamic obstacles, call this each control step.
        obstacles: list of {'cloud': (K,2) tensor, 'center': (d,) tensor}.
        """
        self._obstacles = obstacles

    def _obs(self, obstacles):
        obs = obstacles if obstacles is not None else self._obstacles
        if obs is None:
            raise RuntimeError("CompositeBarrier: no obstacles set "
                               "(call set_obstacles or pass obstacles=...).")
        return obs

    def per_obstacle(self, x: torch.Tensor, obstacles: list = None) -> torch.Tensor:
        """Returns (B, M) — per-obstacle CBF values b_i(x)."""
        cols = []
        for ob in self._obs(obstacles):
            e     = self.encoder(ob['cloud'])                    # (emb_dim,)
            e_b   = e.unsqueeze(0).expand(x.shape[0], -1)        # (B, emb_dim)
            x_rel = x - ob['center']                             # (B, state_dim)
            cols.append(self.cbf(x_rel, e_b))                    # (B, 1)
        return torch.cat(cols, dim=-1)                           # (B, M)

    def forward(self, x: torch.Tensor, obstacles: list = None) -> torch.Tensor:
        b_stack = self.per_obstacle(x, obstacles)               # (B, M)
        # smooth-min = -1/β ln Σ exp(-β b)   (CN-CBF Eq. 18)
        return -(1.0 / self.beta) * torch.logsumexp(
            -self.beta * b_stack, dim=-1, keepdim=True)          # (B, 1)

    def gradient(self, x: torch.Tensor, obstacles: list = None) -> torch.Tensor:
        """∇_x B(x) via autograd. x: (B, d) → (B, d)."""
        xr = x.detach().requires_grad_(True)
        B  = self.forward(xr, obstacles)
        return torch.autograd.grad(B.sum(), xr, create_graph=True)[0]


# ── 4. Combined model ──────────────────────────────────────────────────────────

class BPCBFModel(nn.Module):
    """
    Container for f_θ, V_θ, B_φ trained jointly via S2-NNDS Algorithm 2.
    PenetrationBudget η_ψ removed per mentor feedback.
    """
    def __init__(self, ref_path: np.ndarray = None):
        super().__init__()
        self.f = ProgressConditionedDS(ref_path=ref_path)
        self.V = LyapunovNet()
        self.B = CompositeBarrier()   # composite neural CBF over an obstacle set

    def set_norm(self, mean: torch.Tensor, std: torch.Tensor):
        """Broadcast input normalisation stats to all sub-networks."""
        for m in (self.f, self.V):
            m.in_mean.copy_(mean)
            m.in_std.copy_(std)
        # CompositeBarrier normalizes the relative coordinate by std.
        self.B.set_scale(std)

    def set_obstacles(self, obstacles: list):
        """Install the obstacle set on the composite barrier."""
        self.B.set_obstacles(obstacles)

    def save(self, path: str, quiet: bool = False):
        torch.save(self.state_dict(), path)
        if not quiet:
            print(f"[Model] Saved → {path}")

    def load(self, path: str, device: str = 'cpu'):
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)
        print(f"[Model] Loaded ← {path}")
