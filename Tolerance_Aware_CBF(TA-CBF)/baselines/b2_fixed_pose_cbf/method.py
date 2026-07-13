"""
B2 — fixed-pose learned CBF (no pose/scale/translation augmentation).

Identical architecture, encoder, losses, and training recipe as ours; the ONLY
difference is that the barrier was trained on the nominal obstacle poses alone
(src/train_fixed_pose.py -> checkpoints/fixed_pose_model.pt). At test time it
receives the new scene's point clouds exactly like ours does — but its encoder
has only ever seen one geometry-to-label association, so the embedding goes
out-of-distribution under rotation/scale. This is the Robey-et-al./S2-NNDS
regime and simultaneously the paper's headline ablation (isolates the
augmentation as THE contribution).

Retrain with:  venv/bin/python src/train_fixed_pose.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import ROOT  # noqa: E402
from common.methods import OursMethod  # noqa: E402

CKPT = os.path.join(ROOT, "checkpoints", "fixed_pose_model.pt")


class FixedPoseMethod(OursMethod):
    """Same protocol as ours (set_obstacles with the scene clouds); different
    weights — the no-augmentation checkpoint."""
    name = "b2_fixed_pose_cbf"
    checkpoint = CKPT


def get_methods():
    if not os.path.exists(CKPT):
        raise FileNotFoundError(
            f"{CKPT} missing - run: venv/bin/python src/train_fixed_pose.py")
    return [FixedPoseMethod()]
