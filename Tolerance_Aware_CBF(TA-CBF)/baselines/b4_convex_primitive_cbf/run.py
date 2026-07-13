"""Run only this baseline on the shared suite (see baselines/run_all.py)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.evaluate import run_protocol
from common.methods import OursMethod
from b4_convex_primitive_cbf.method import get_methods

if __name__ == "__main__":
    quick = "--quick" in sys.argv
    run_protocol([OursMethod()] + get_methods(), quick=quick,
                 out_name="b4_convex_primitive_cbf")
