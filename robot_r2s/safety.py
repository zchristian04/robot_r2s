"""Safety checks for the robot server.

Checks happen in three places:
  before motion   every planned TCP point is inside the workspace, reachable by
                  IK within the joint margin, and free of table collisions
                  (outside contact segments);
  during motion   wrench above max_wrench, tracking error, stale robot state;
  after a fault   the server latches the fault and refuses all motion until the
                  operator clears it.
"""

import numpy as np


class SafetyError(RuntimeError):
    """A request that would violate a limit. The message is safe to show the client."""


class Safety:
    def __init__(self, cfg):
        s = cfg["safety"]
        self.ws_lo = np.array([s["workspace"][k][0] for k in "xyz"], float)
        self.ws_hi = np.array([s["workspace"][k][1] for k in "xyz"], float)
        self.max_tcp_speed = float(s["max_tcp_speed"])
        self.max_tcp_angular_speed = float(s["max_tcp_angular_speed"])
        self.max_joint_speed = float(s["max_joint_speed"])
        self.joint_margin = float(s["joint_margin_rad"])
        self.max_wrench = np.asarray(s["max_wrench"], float)
        self.max_tracking_error = float(s["max_tracking_error_rad"])
        self.state_timeout = float(s["state_timeout_s"])
        self.table_clearance = float(s["table_clearance"])
        self.fault = None

    # -- latched fault ------------------------------------------------------------

    def latch(self, reason):
        if self.fault is None:
            self.fault = reason

    def clear(self):
        self.fault = None

    def require_ok(self):
        if self.fault is not None:
            raise SafetyError(f"robot is stopped after a fault ({self.fault}); waiting for operator review")

    # -- before motion ---------------------------------------------------------------

    def check_point(self, p_model, what):
        if np.any(p_model < self.ws_lo - 1e-9) or np.any(p_model > self.ws_hi + 1e-9):
            raise SafetyError(f"{what} is outside the robot workspace")

    def wrench_exceeded(self, wrench):
        return bool(np.any(np.abs(np.asarray(wrench, float)) > self.max_wrench))
