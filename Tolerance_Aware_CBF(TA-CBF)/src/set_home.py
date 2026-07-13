#!/usr/bin/env python3
"""
set_home.py — capture a hand-posed joint configuration as the robot "home".

Use case: mount the needle, then manually rotate the last joint (and any others)
until the needle orientation/offset is what you want, and SAVE that whole
7-joint configuration as home. Later you can send the robot back to it.

The arm is put in GRAVITY COMPENSATION so it is compliant and you can move it by
hand. HOLD THE ARM — it will go limp when gravity comp activates.

Run (ROS + venv sourced, demo launch with the gripper already up):
    # 1) capture: pose by hand, press ENTER to save
    python src/set_home.py

    # 2) later: drive the robot back to the saved home
    python src/set_home.py --go

Saves to:  checkpoints/home_config.npy   (7 joint values, radians)
"""

import argparse
import os

import numpy as np

DEFAULT_OUT = "checkpoints/home_config.npy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="fr3")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--go", action="store_true",
                    help="move to the saved home instead of capturing")
    ap.add_argument("--speed", type=float, default=4.0,
                    help="[--go] seconds to reach home (larger = slower/safer)")
    args = ap.parse_args()

    from crisp_py.robot import make_robot

    robot = make_robot(args.robot)
    print("Waiting for robot…")
    robot.wait_until_ready()

    # ───────────────────────── GO TO SAVED HOME ─────────────────────────
    if args.go:
        out = os.path.abspath(args.out)
        if not os.path.exists(out):
            print(f"[err] no saved home at {out} — run without --go first.")
            return
        q = np.load(out).astype(float).tolist()
        print(f"Loaded home: {[round(v, 4) for v in q]}")
        try:
            robot.config.time_to_home = float(args.speed)
        except Exception:  # noqa: BLE001
            pass
        input("ENTER to MOVE the robot to this home (keep e-stop ready)… ")
        robot.home(home_config=q)
        print("At home.")
        return

    # ───────────────────────── CAPTURE NEW HOME ─────────────────────────
    try:
        robot.controller_switcher_client.switch_controller("gravity_compensation")
        print("\n>>> GRAVITY COMPENSATION active — HOLD THE ARM, it is now compliant.")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] could not switch to gravity_compensation ({e}).")
        print("       Activate a gravity-comp / hand-guide controller manually first.")

    print("\nManually rotate the last joint (and any others) to the desired pose.")
    print("Current joint values update live below; press ENTER to capture & save.\n")

    try:
        import sys, select  # noqa: E401
        while True:
            q = robot.joint_values
            print("\r  q = [" + ", ".join(f"{v:+.4f}" for v in q) + "]  ",
                  end="", flush=True)
            # non-blocking: capture on ENTER
            if select.select([sys.stdin], [], [], 0.1)[0]:
                sys.stdin.readline()
                break
    except KeyboardInterrupt:
        print("\nCancelled — nothing saved.")
        return

    q = np.asarray(robot.joint_values, dtype=np.float32)
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out, q)

    print("\n\nSaved home configuration:")
    print(f"  {out}")
    print(f"  q (rad) = {[round(float(v), 5) for v in q]}")
    print("\nTo make it crisp_py's permanent home, paste this into FrankaConfig.home_config")
    print("in crisp_py/robot/robot_config.py, or load it with:")
    print(f"    import numpy as np; q = np.load('{args.out}'); robot.home(home_config=q.tolist())")
    print("Or just run:  python src/set_home.py --go")


if __name__ == "__main__":
    main()
