"""Action primitives: parameter validation and motion planning.

Same primitives, parameters and segment structure as the simulated source world
(r2s_exam/rig/primitives.py), so an agent's requests mean the same thing on the
real robot. Positions are in API coordinates (table top at z = 0). Limits come
from the config instead of module constants.

Every primitive starts from the current gripper state, rises to the travel
height, moves above its approach point, does its work and finishes back at the
travel height. Segments tagged in CONTACT_TAGS may touch things: they run with
the contact gains, are exempt from the table-collision check and are watched for
force.
"""

from dataclasses import dataclass, field

import numpy as np

PRIMITIVES = ("move_to", "push", "touch", "grasp_lift", "home")
CONTACT_PRIMITIVES = ("push", "touch", "grasp_lift")
CONTACT_TAGS = ("push", "touch", "close", "lift", "lower", "release", "descend_grasp")


class ActionError(ValueError):
    """Invalid primitive or parameter. The message is safe to show the client."""


@dataclass
class Segment:
    tcp: np.ndarray            # target TCP position (API frame) at the end of the segment
    yaw: float
    width: float               # target gripper opening width
    speed: float               # TCP linear speed limit, m/s
    hold: float = 0.0          # extra time to hold the target after reaching it
    stop_force: float = None   # end the segment early when |F| exceeds this (N)
    tag: str = ""

    @property
    def contact(self):
        return self.tag in CONTACT_TAGS


@dataclass
class Plan:
    primitive: str
    params: dict
    segments: list = field(default_factory=list)

    def summary(self):
        lines = [f"{self.primitive} {self.params}"]
        for s in self.segments:
            extra = f" stop_force={s.stop_force}" if s.stop_force else ""
            lines.append(f"  {s.tag:13s} tcp={np.round(s.tcp, 3).tolist()} yaw={s.yaw:+.2f} "
                         f"width={s.width:.3f} speed={s.speed:.3f}{extra}")
        return "\n".join(lines)


def _vec(params, key, n):
    if key not in params:
        raise ActionError(f"missing parameter: {key}")
    try:
        v = np.asarray(params[key], dtype=float).reshape(-1)
    except (TypeError, ValueError):
        raise ActionError(f"parameter {key} must be a list of {n} numbers")
    if v.shape != (n,) or not np.all(np.isfinite(v)):
        raise ActionError(f"parameter {key} must be a list of {n} finite numbers")
    return v


def _num(params, key, lo, hi, default=None):
    if key not in params:
        if default is None:
            raise ActionError(f"missing parameter: {key}")
        return float(default)
    try:
        v = float(params[key])
    except (TypeError, ValueError):
        raise ActionError(f"parameter {key} must be a number")
    if not (np.isfinite(v) and lo <= v <= hi):
        raise ActionError(f"parameter {key} must be in [{lo}, {hi}]")
    return v


def _unit(v, key):
    n = np.linalg.norm(v)
    if n < 1e-6:
        raise ActionError(f"parameter {key} must be non-zero")
    return v / n


class Planner:
    def __init__(self, cfg, home_tcp, home_yaw):
        p = cfg["primitives"]
        self.safe_z = float(p["safe_z"])
        self.standoff = float(p["push_standoff"])
        self.retreat = float(p["retreat"])
        self.max_push = float(p["max_push_distance"])
        self.max_travel = float(p["max_touch_travel"])
        self.max_lift = float(p["max_lift"])
        self.fstop_lo, self.fstop_hi = p["touch_stop_force"]
        self.max_speed = float(cfg["safety"]["max_tcp_speed"])
        self.max_width = float(cfg["robot"]["gripper_max_width"])
        self.home_width = float(cfg["robot"]["gripper_home_width"])
        self.home_tcp = np.asarray(home_tcp, float)
        self.home_yaw = float(home_yaw)

    def _transit(self, tcp, yaw, width, target_xy, target_yaw):
        up = np.array([tcp[0], tcp[1], max(tcp[2], self.safe_z)])
        above = np.array([target_xy[0], target_xy[1], self.safe_z])
        return [Segment(up, yaw, width, self.max_speed, tag="rise"),
                Segment(above, target_yaw, width, self.max_speed, tag="transit")]

    def plan(self, primitive, params, tcp, yaw, width):
        """Validate a primitive call and plan it from the current commanded state."""
        if primitive not in PRIMITIVES:
            raise ActionError(f"unknown primitive {primitive!r}; available: {list(PRIMITIVES)}")
        if not isinstance(params, dict):
            raise ActionError("params must be an object")
        allowed = {
            "move_to": {"tcp", "yaw", "speed"},
            "push": {"contact_point", "direction", "distance", "speed", "yaw"},
            "touch": {"point", "direction", "max_travel", "speed", "stop_force", "yaw"},
            "grasp_lift": {"center", "yaw", "open_width", "lift", "hold"},
            "home": set(),
        }[primitive]
        extra = set(params) - allowed
        if extra:
            raise ActionError(f"unknown parameters for {primitive}: {sorted(extra)}")
        tcp = np.asarray(tcp, float)
        vmax = self.max_speed
        p = Plan(primitive, dict(params))
        S = p.segments

        if primitive == "home":
            S += self._transit(tcp, yaw, width, self.home_tcp[:2], self.home_yaw)
            S.append(Segment(self.home_tcp.copy(), self.home_yaw, self.home_width, vmax, tag="home"))

        elif primitive == "move_to":
            target = _vec(params, "tcp", 3)
            tyaw = _num(params, "yaw", -np.pi, np.pi, default=yaw)
            speed = _num(params, "speed", 0.005, vmax, default=vmax)
            S.append(Segment(target, tyaw, width, speed, tag="move"))

        elif primitive == "push":
            c = _vec(params, "contact_point", 3)
            d = np.append(_unit(_vec(params, "direction", 2), "direction"), 0.0)
            dist = _num(params, "distance", 0.0, self.max_push)
            speed = _num(params, "speed", 0.005, min(0.10, vmax))
            pyaw = _num(params, "yaw", -np.pi, np.pi, default=np.arctan2(d[1], d[0]))
            start = c - self.standoff * d
            end = c + dist * d
            back = end - self.retreat * d
            S += self._transit(tcp, yaw, width, start[:2], pyaw)
            S += [Segment(start, pyaw, 0.0, vmax, tag="descend"),
                  Segment(end, pyaw, 0.0, speed, tag="push"),
                  Segment(back, pyaw, 0.0, speed, tag="retreat"),
                  Segment(np.array([back[0], back[1], self.safe_z]), pyaw, 0.0, vmax, tag="rise")]

        elif primitive == "touch":
            q = _vec(params, "point", 3)
            d = _unit(_vec(params, "direction", 3), "direction")
            travel = _num(params, "max_travel", 0.0, self.max_travel)
            speed = _num(params, "speed", 0.002, 0.05)
            fstop = _num(params, "stop_force", self.fstop_lo, self.fstop_hi)
            tyaw = _num(params, "yaw", -np.pi, np.pi, default=yaw)
            end = q + travel * d
            S += self._transit(tcp, yaw, width, q[:2], tyaw)
            S += [Segment(q, tyaw, 0.0, vmax, tag="descend"),
                  Segment(end, tyaw, 0.0, speed, stop_force=fstop, tag="touch"),
                  # The executor rewrites the retreat target relative to where the touch stopped.
                  Segment(end - self.retreat * d, tyaw, 0.0, speed, tag="retreat"),
                  Segment(np.array([end[0], end[1], self.safe_z]), tyaw, 0.0, vmax, tag="rise")]

        elif primitive == "grasp_lift":
            c = _vec(params, "center", 3)
            gyaw = _num(params, "yaw", -np.pi, np.pi)
            ow = _num(params, "open_width", 0.01, self.max_width)
            lift = _num(params, "lift", 0.0, self.max_lift)
            hold = _num(params, "hold", 0.0, 2.0, default=0.5)
            up = c + np.array([0.0, 0.0, lift])
            above = np.array([c[0], c[1], self.safe_z])
            S += self._transit(tcp, yaw, width, c[:2], gyaw)
            S += [Segment(above, gyaw, ow, vmax, tag="open"),
                  Segment(c, gyaw, ow, vmax, tag="descend_grasp"),
                  Segment(c, gyaw, 0.0, vmax, hold=0.3, tag="close"),
                  Segment(up, gyaw, 0.0, 0.03, hold=hold, tag="lift"),
                  Segment(c, gyaw, 0.0, 0.03, tag="lower"),
                  Segment(c, gyaw, ow, vmax, hold=0.2, tag="release"),
                  Segment(above, gyaw, ow, vmax, tag="rise"),
                  Segment(above, gyaw, 0.0, vmax, tag="close_empty")]

        return p
