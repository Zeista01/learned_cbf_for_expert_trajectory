"""
methods.py — the baseline Method interface.

Every baseline is a Method. The runner holds everything else fixed (nominal DS,
CLF-QP, scenes, integration), so a Method only decides two things:

  prepare(model, shapes) — install its safety mechanism for the given scene
                           (usually: swap model.B for a different barrier)
  filter(model, ctrl, x_np, f_val, s) — turn the nominal velocity f_val into
                           the commanded velocity (usually: the shared CLF-CBF-QP)

QP-based baselines differ ONLY in the barrier module they install, which is the
controlled comparison the paper needs.
"""
import numpy as np

from . import SRC  # noqa: F401  (side effect: src/ on sys.path)
from config import DEVICE


class Method:
    """Base: the shared CLF-CBF-QP acting on whatever barrier is installed."""
    name = "base"
    #: apply ctrl.project_safe (discrete-time projection on the installed barrier)
    use_projection = True
    #: checkpoint to load the model from (None → default final_model.pt)
    checkpoint = None

    def make_model(self):
        from simulate import load_model
        return load_model(self.checkpoint)

    def prepare(self, model, shapes):
        raise NotImplementedError

    def filter(self, model, ctrl, x_np, f_val, s):
        u, info = ctrl.solve(x_np.astype(np.float32), model, device=DEVICE, s=s)
        return f_val + u, info


class OursMethod(Method):
    """TA-CBF (proposed): augmented composite barrier, zero-shot on new scenes."""
    name = "ours_ta_cbf"

    def prepare(self, model, shapes):
        from generalization_test import build_tensors
        model.set_obstacles(build_tensors(shapes, k=64, seed=1))


class BarrierSwapMethod(Method):
    """QP baseline that replaces model.B with barrier_cls(shapes)."""
    barrier_cls = None

    def prepare(self, model, shapes):
        model.B = self.barrier_cls(shapes)


class UnfilteredMethod(Method):
    """No safety mechanism at all — f_val goes straight to the plant."""
    use_projection = False

    def prepare(self, model, shapes):
        pass

    def filter(self, model, ctrl, x_np, f_val, s):
        return f_val, {}
