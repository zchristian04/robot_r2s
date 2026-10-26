"""Runs a primitive Plan on a backend, with the safety checks.

Every segment is a straight TCP line (API frame) with a min-jerk time profile.
Before anything moves, the whole plan is converted to joint targets at the
control rate by warm-started IK and checked: workspace, joint margin, joint and
TCP speed, and table collision for segments that are not allowed to touch.
During motion the executor streams those targets and checks every tick:
  - robot fault                         -> latch, hold
  - |wrench| above safety.max_wrench    -> protective stop: latch, hold
  - joint tracking error too large      -> latch, hold
  - stale robot state                   -> latch, hold
  - |F| above a touch's stop_force      -> end that segment, retreat from there
A latched fault blocks all further motion until the operator clears it.

The wrench is tared over 0.1 s at the start of every action, as in the
simulated source world.
"""

import numpy as np

from .primitives import Plan, Segment
from .safety import SafetyError

MIN_JERK_PEAK = 1.875   # peak/mean speed ratio of the min-jerk profile
TARE_S = 0.1


def _min_jerk(s):
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


class Executor:
    def __init__(self, cfg, backend, model, safety, viz=None):
        self.cfg = cfg
        self.b = backend
        self.model = model
        self.safety = safety
        self.viz = viz
        c = cfg["control"]
        self.dt = 1.0 / float(c["rate_hz"])
        self.gains = {"travel": (c["travel_kp"], c["travel_kd"]),
                      "contact": (c["contact_kp"], c["contact_kd"])}
        self.settle_tol = float(c["settle_tol_rad"])
        self.settle_timeout = float(c["settle_timeout_s"])
        self.log_dt = 1.0 / float(cfg["logging"]["log_hz"])
        self.frame_dt = 1.0 / float(cfg["camera"]["action_frames_hz"])
        r = cfg["robot"]
        self.g_force = float(r["gripper_force_cap"])
        self.g_speed = float(r["gripper_speed"])
        self.cmd_q = None
        self.cmd_tcp = None
        self.cmd_yaw = 0.0
        self.cmd_width = float(r["gripper_home_width"])

    def sync(self):
        """Take the current measured state as the commanded state."""
        st = self.b.state()
        self.cmd_q = st["q"].copy()
        self.cmd_tcp, self.cmd_yaw = self.model.fk_api(self.cmd_q)
        self.cmd_width = st["gripper_width"]
        return st

    # -- planning to joint space ----------------------------------------------------

    def _segment_joints(self, q0, p0, yaw0, seg, check_table):
        """Joint targets at the control rate for one straight TCP segment."""
        s = self.safety
        p1, yaw1 = np.asarray(seg.tcp, float), float(seg.yaw)
        dyaw = (yaw1 - yaw0 + np.pi) % (2 * np.pi) - np.pi
        length = float(np.linalg.norm(p1 - p0))
        # Coarse pass at 5 mm to find the joint path length, then time the segment.
        n0 = max(2, int(np.ceil(max(length, abs(dyaw) * 0.05) / 0.005)) + 1)
        qs, q = [], q0
        for u in np.linspace(0.0, 1.0, n0):
            q = self._ik_point(p0 + u * (p1 - p0), yaw0 + u * dyaw, q, seg, check_table)
            qs.append(q)
        qs = np.array(qs)
        joint_len = np.abs(np.diff(qs, axis=0)).sum(axis=0).max() if len(qs) > 1 else 0.0
        T = max(length / seg.speed if seg.speed > 0 else 0.0,
                abs(dyaw) / s.max_tcp_angular_speed,
                joint_len / s.max_joint_speed) * MIN_JERK_PEAK
        T = max(T, self.dt)
        n = int(np.ceil(T / self.dt))
        out, q = [], q0
        for k in range(1, n + 1):
            u = _min_jerk(k / n)
            q = self._ik_point(p0 + u * (p1 - p0), yaw0 + u * dyaw, q, seg, check_table)
            out.append(q)
        return np.array(out)

    def _ik_point(self, p_api, yaw, q_seed, seg, check_table):
        m, s = self.model, self.safety
        p_model = m.to_model(p_api)
        s.check_point(p_model, f"{seg.tag} target")
        q = m.ik(p_model, yaw, q_seed, margin=s.joint_margin)
        if q is None:
            raise SafetyError(f"{seg.tag} target is not reachable with the tool pointing down")
        if np.abs(q - q_seed).max() > 0.2:
            raise SafetyError(f"{seg.tag} needs a large joint jump (arm reconfiguration); rejected")
        if check_table and m.table_collision(q, clearance=s.table_clearance):
            raise SafetyError(f"{seg.tag} would bring the arm or gripper into the table")
        return q

    def prepare(self, plan: Plan):
        """Check the whole plan and convert it to joint targets. Raises SafetyError."""
        self.safety.require_ok()
        q, p, yaw = self.cmd_q, np.asarray(self.cmd_tcp, float), self.cmd_yaw
        out = []
        for seg in plan.segments:
            qs = self._segment_joints(q, p, yaw, seg, check_table=not seg.contact)
            out.append(qs)
            q, p, yaw = qs[-1], np.asarray(seg.tcp, float), float(seg.yaw)
        return out

    # -- execution --------------------------------------------------------------------

    async def run(self, plan: Plan, joint_paths, frame_cb=None):
        """Execute a prepared plan. Returns {"status", "events", "log"}."""
        b, s = self.b, self.safety
        log = {k: [] for k in ("t", "tcp_cmd", "tcp_meas", "yaw_cmd", "yaw_meas", "width_cmd",
                               "width_meas", "wrench", "q_cmd", "q_meas", "tau_ext", "gripper_force")}
        events = []
        st0 = b.state()
        t0 = st0["t"]
        clock = {"next_log": 0.0, "next_frame": 0.0}
        status = "ok"

        # Tare the wrench at rest.
        acc = []
        while b.state()["t"] - st0["t"] < TARE_S:
            await b.wait(self.dt)
            acc.append(b.state()["wrench"])
        bias = np.mean(acc, axis=0) if acc else np.zeros(6)

        def record(st, q_cmd, width_cmd):
            t = st["t"] - t0
            wrench = st["wrench"] - bias
            if t >= clock["next_log"]:
                clock["next_log"] = t + self.log_dt
                p_cmd, yaw_c = self.model.fk_api(q_cmd)
                p_meas, yaw_m = self.model.fk_api(st["q"])
                for k, v in (("t", t), ("tcp_cmd", p_cmd), ("tcp_meas", p_meas), ("yaw_cmd", yaw_c),
                             ("yaw_meas", yaw_m), ("width_cmd", width_cmd),
                             ("width_meas", st["gripper_width"]), ("wrench", wrench),
                             ("q_cmd", q_cmd), ("q_meas", st["q"]), ("tau_ext", st["tau_ext"]),
                             ("gripper_force", st["gripper_force"])):
                    log[k].append(np.asarray(v, float))
                if self.viz is not None:
                    self.viz.log_state(st["t"], st["q"], q_cmd, st["gripper_width"], width_cmd, wrench)
            if frame_cb is not None and t >= clock["next_frame"]:
                clock["next_frame"] = t + self.frame_dt
                frame_cb(t)
            return wrench

        def check(st, wrench, q_cmd):
            if st["fault"]:
                return f"robot fault: {st['fault']}"
            if b.now() - st["t"] > s.state_timeout:
                return "stale robot state"
            if s.wrench_exceeded(wrench):
                return "protective stop: wrench limit"
            if np.abs(st["q"] - q_cmd).max() > s.max_tracking_error:
                return "protective stop: tracking error"
            return None

        segments = list(plan.segments)
        replan = False
        for i, (seg, qs) in enumerate(zip(segments, joint_paths)):
            kp, kd = self.gains["contact" if seg.contact else "travel"]
            b.set_gains(kp, kd)
            if replan:
                # After a force stop the precomputed path no longer starts where the
                # arm is; plan the rest of the action from the current command.
                try:
                    p_now, yaw_now = self.model.fk_api(self.cmd_q)
                    qs = self._segment_joints(self.cmd_q, p_now, yaw_now, seg, check_table=not seg.contact)
                except SafetyError as e:
                    return await self._fault(f"replan after force stop failed: {e}", b.state(), events, t0, seg, log)
            if self.viz is not None:
                self.viz.log_plan(seg.tag, qs)
            # Gripper first, so open/close happens before the arm moves on.
            if abs(seg.width - self.cmd_width) > 1e-4:
                if seg.width < self.cmd_width and seg.tag == "close":
                    await b.gripper_grasp(self.g_force)
                else:
                    await b.gripper_move(seg.width, self.g_speed, self.g_force)
                self.cmd_width = seg.width
                waited = 0.0
                while waited < 3.0:
                    await b.wait(self.dt)
                    waited += self.dt
                    st = b.state()
                    wrench = record(st, self.cmd_q, self.cmd_width)
                    reason = check(st, wrench, self.cmd_q)
                    if reason:
                        return await self._fault(reason, st, events, t0, seg, log)
                    if waited > 0.2 and not st["gripper_moving"]:
                        break
            for q_cmd in qs:
                b.command(q_cmd)
                await b.wait(self.dt)
                st = b.state()
                wrench = record(st, q_cmd, self.cmd_width)
                reason = check(st, wrench, q_cmd)
                if reason:
                    self.cmd_q = q_cmd
                    return await self._fault(reason, st, events, t0, seg, log)
                self.cmd_q = q_cmd
                if seg.stop_force is not None and np.linalg.norm(wrench[:3]) > seg.stop_force:
                    events.append({"t": st["t"] - t0, "type": "stop_force", "segment": seg.tag,
                                   "force": float(np.linalg.norm(wrench[:3]))})
                    self.cmd_q = st["q"].copy()
                    b.command(self.cmd_q)
                    self._redirect_after_stop(segments, i)
                    replan = True
                    break
            self.cmd_tcp, self.cmd_yaw = self.model.fk_api(self.cmd_q)
            if seg.hold > 0:
                for _ in range(int(round(seg.hold / self.dt))):
                    b.command(self.cmd_q)
                    await b.wait(self.dt)
                    st = b.state()
                    wrench = record(st, self.cmd_q, self.cmd_width)
                    reason = check(st, wrench, self.cmd_q)
                    if reason:
                        return await self._fault(reason, st, events, t0, seg, log)
        await self._settle()
        return {"status": status, "events": events, "log": _arrays(log)}

    def _redirect_after_stop(self, segments, i):
        """After a touch stops on force, back off along the touch direction from the
        stop point and rise from there, instead of from the planned end point."""
        seg = segments[i]
        if i + 1 >= len(segments) or segments[i + 1].tag != "retreat":
            return
        p_stop, _ = self.model.fk_api(self.cmd_q)
        d = np.asarray(seg.tcp, float) - np.asarray(segments[i + 1].tcp, float)
        d = d / max(np.linalg.norm(d), 1e-9)
        back = p_stop - self.cfg["primitives"]["retreat"] * d
        r = segments[i + 1]
        segments[i + 1] = Segment(back, r.yaw, r.width, r.speed, tag="retreat")
        if i + 2 < len(segments) and segments[i + 2].tag == "rise":
            u = segments[i + 2]
            segments[i + 2] = Segment(np.array([back[0], back[1], u.tcp[2]]), u.yaw, u.width, u.speed,
                                      tag="rise")

    async def _fault(self, reason, st, events, t0, seg, log):
        await self.b.hold()
        self.safety.latch(reason)
        events.append({"t": st["t"] - t0, "type": "protective_stop", "segment": seg.tag, "reason": reason})
        self.sync()
        return {"status": "protective_stop", "events": events, "log": _arrays(log)}

    async def _settle(self):
        waited = 0.0
        while waited < self.settle_timeout:
            await self.b.wait(self.dt)
            waited += self.dt
            if np.abs(self.b.state()["q"] - self.cmd_q).max() < self.settle_tol:
                return


def _arrays(log):
    return {k: np.array(v) for k, v in log.items()}
