"""
scenes.py — the FIXED evaluation suite shared by every baseline.

Two regimes matching the paper:
  nominal        — the training scene (CRITICAL_SHAPES) with perturbed starts
  generalization — feasibility-checked random pose/scale scenes (unseen layouts)

Scene seeds are fixed so every method is evaluated on identical scenes; the
suite is cached to results/baselines/scene_suite.npz-adjacent pickle for reuse.
"""
import copy
import os
import pickle

import numpy as np

from . import SRC, RESULTS_DIR  # noqa: F401
from config import CRITICAL_SHAPES, X_START


def nominal_starts(n=6, radius=0.015, seed=0):
    rng = np.random.default_rng(seed)
    starts = [X_START.copy()]
    for _ in range(n - 1):
        sp = X_START.copy()
        sp[:2] += rng.uniform(-radius, radius, 2)
        starts.append(sp)
    return starts


def generalization_suite(n_obs_list=(3, 4, 5, 6), per=3, seed=0,
                         cache=True, clearance=None):
    """List of (name, shapes). Feasible layouts only (guaranteed passage).

    clearance: minimum start->goal SDF passage the scene generator guarantees
    (metres). None -> the project default SCENE_CLEARANCE (~15mm). Widening it
    disentangles ours' conservatism from genuine infeasibility.
    """
    from generalization_test import make_solvable_scene, SCENE_CLEARANCE
    clr = SCENE_CLEARANCE if clearance is None else clearance
    tag = (f"suite_nobs{'-'.join(map(str, n_obs_list))}_per{per}_seed{seed}"
           f"_clr{int(round(clr * 1000))}.pkl")
    path = os.path.join(RESULTS_DIR, tag)
    if cache and os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    # obstacles must not clump tighter than the passage they have to leave,
    # but keep the floor low enough that dense (7-8 obstacle) layouts are
    # findable in the small slab (clumped obstacles are a realistic hard case
    # the composite barrier is meant to handle).
    min_sep = max(0.038, clr + 0.012)
    scenes = []
    for n_obs in n_obs_list:
        built, s = 0, 0
        while built < per and s < per * 80:
            try:
                sh = make_solvable_scene(n_obs, seed=seed * 100 + s,
                                         clearance=clr, min_sep=min_sep)
                scenes.append((f"gen_{n_obs}obs_s{seed * 100 + s}", sh))
                built += 1
            except RuntimeError:
                pass
            s += 1
    if cache:
        with open(path, 'wb') as f:
            pickle.dump(scenes, f)
    return scenes


def nominal_scene():
    return copy.deepcopy(CRITICAL_SHAPES)
