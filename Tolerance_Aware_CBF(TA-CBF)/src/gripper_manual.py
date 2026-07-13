#!/usr/bin/env python3
"""
gripper_manual.py — manually set the Franka Hand finger gap (interactive).

The Franka Hand opens 0..0.08 m (0..80 mm). This uses the `Move` action to go
to any width you type, with +/- nudging and a grasp shortcut.

Run (with ROS + gripper launched, load_gripper:=true):
    python src/gripper_manual.py                 # default ns: franka_gripper
    python src/gripper_manual.py --ns fr3_gripper

Commands (type then ENTER):
    25        -> move to 25 mm
    +         -> open  by step (default 2 mm)
    -         -> close by step
    +5 / -5   -> nudge by 5 mm
    g [mm] [N]-> grasp at <mm> width with <N> force (default 1 mm, 30 N)
    h         -> home (recalibrate; clears any held object)
    s         -> print current width
    q         -> quit
"""

import argparse
import sys

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from franka_msgs.action import Grasp, Homing, Move

MAX_W = 0.08   # m, Franka Hand fully open
STEP = 0.002   # m, default nudge (2 mm)


class GripperManual(Node):
    def __init__(self, ns: str):
        super().__init__("gripper_manual")
        self.ns = ns.rstrip("/")
        self.width = None
        self.create_subscription(JointState, f"/{self.ns}/joint_states",
                                 self._cb_js, 10)
        self._move = ActionClient(self, Move, f"/{self.ns}/move")
        self._grasp = ActionClient(self, Grasp, f"/{self.ns}/grasp")
        self._home = ActionClient(self, Homing, f"/{self.ns}/homing")

    def _cb_js(self, msg: JointState):
        if len(msg.position) >= 2:
            self.width = msg.position[0] + msg.position[1]

    def _spin_until(self, future, timeout=15.0):
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        return future.result()

    def refresh_width(self, timeout=2.0):
        end = self.get_clock().now().nanoseconds + int(timeout * 1e9)
        while self.width is None and self.get_clock().now().nanoseconds < end:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.width

    def move_to(self, width_m: float, speed: float = 0.05):
        width_m = max(0.0, min(MAX_W, width_m))
        if not self._move.wait_for_server(timeout_sec=2.0):
            print("  [err] move server not available — check --ns / gripper launch")
            return
        goal = Move.Goal(); goal.width = width_m; goal.speed = speed
        print(f"  moving to {width_m*1000:.1f} mm …")
        gh = self._spin_until(self._move.send_goal_async(goal))
        if gh is None or not gh.accepted:
            print("  [err] move goal rejected"); return
        res = self._spin_until(gh.get_result_async())
        ok = res.result.success if res else False
        print(f"  done (success={ok}); width≈{self.refresh_width()*1000:.1f} mm"
              if self.width is not None else f"  done (success={ok})")

    def grasp(self, width_m: float = 0.001, force: float = 30.0, speed: float = 0.05):
        if not self._grasp.wait_for_server(timeout_sec=2.0):
            print("  [err] grasp server not available"); return
        goal = Grasp.Goal()
        goal.width = max(0.0, min(MAX_W, width_m))
        goal.speed = speed
        goal.force = force
        goal.epsilon.inner = 0.008
        goal.epsilon.outer = 0.008
        print(f"  grasping at {goal.width*1000:.1f} mm, {force:.0f} N …")
        gh = self._spin_until(self._grasp.send_goal_async(goal))
        if gh is None or not gh.accepted:
            print("  [err] grasp goal rejected"); return
        res = self._spin_until(gh.get_result_async())
        print(f"  grasp success={res.result.success if res else False}")

    def home(self):
        if not self._home.wait_for_server(timeout_sec=2.0):
            print("  [err] homing server not available"); return
        print("  homing (keep fingers clear) …")
        gh = self._spin_until(self._home.send_goal_async(Homing.Goal()))
        if gh and gh.accepted:
            self._spin_until(gh.get_result_async())
            print("  homed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="franka_gripper",
                    help="gripper namespace (try fr3_gripper if actions are there)")
    ap.add_argument("--step", type=float, default=STEP * 1000, help="nudge step in mm")
    args = ap.parse_args()

    rclpy.init()
    g = GripperManual(args.ns)
    step_m = args.step / 1000.0

    w = g.refresh_width()
    print(f"Connected to /{args.ns}. Current width: "
          f"{w*1000:.1f} mm" if w is not None else
          f"Connected to /{args.ns} (width unknown — is the gripper running?)")
    print(__doc__.split("Commands")[1] if "Commands" in __doc__ else "")

    try:
        while True:
            try:
                line = input("gap> ").strip()
            except EOFError:
                break
            if not line:
                continue
            cmd = line.lower()

            if cmd in ("q", "quit", "exit"):
                break
            elif cmd == "h":
                g.home()
            elif cmd == "s":
                w = g.refresh_width()
                print(f"  width ≈ {w*1000:.1f} mm" if w is not None else "  width unknown")
            elif cmd.startswith("g"):
                parts = line.split()
                width_mm = float(parts[1]) if len(parts) > 1 else 1.0
                force = float(parts[2]) if len(parts) > 2 else 30.0
                g.grasp(width_m=width_mm / 1000.0, force=force)
            elif cmd.startswith("+") or cmd.startswith("-"):
                base = g.refresh_width() or 0.0
                num = cmd[1:]
                delta = (float(num) / 1000.0) if num else step_m
                sign = 1.0 if cmd[0] == "+" else -1.0
                g.move_to(base + sign * delta)
            else:
                try:
                    g.move_to(float(cmd) / 1000.0)   # plain number = mm
                except ValueError:
                    print("  ? type mm value, +/-, 'g', 'h', 's', or 'q'")
    finally:
        g.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
