"""Shared baseline infrastructure. Importing this package puts the project's
src/ on sys.path so baseline code can reuse config / models / cbf_qp / etc."""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

RESULTS_DIR = os.path.join(ROOT, "results", "baselines")
os.makedirs(RESULTS_DIR, exist_ok=True)
