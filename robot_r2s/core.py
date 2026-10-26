"""The robot core and the worker process that owns it.

RobotCore holds the backend, kinematics, safety, planner, executor, camera and
Rerun view. On the real robot it runs in its own process (worker_main), so the
HTTP server, operator prompts and request logging can never delay aioflexiv's
torque loop. The server talks to it through WorkerClient over a pipe.

Every action is two steps, so the operator can approve with the ghost and path
already showing in Rerun:
  prepare(primitive, params)  plan + full safety check; returns a summary
  run()                       execute the prepared plan (or cancel())
"""

import asyncio
import multiprocessing as mp
import threading
import traceback

import numpy as np

from .executor import Executor
from .model import RobotModel
from .primitives import ActionError, CONTACT_PRIMITIVES, Planner
from .safety import Safety, SafetyError


def make_backend(cfg, kind, robot_sn=None):
    if kind == "sim":
        from .backends.sim import SimBackend
        return SimBackend(cfg)
    if kind == "real":
        from .backends.real import RealBackend
        return RealBackend(cfg, robot_sn)
    raise ValueError(f"unknown backend {kind!r}")


class RobotCore:
    def __init__(self, cfg, backend_kind, robot_sn=None, viz_mode="off"):
        self.cfg = cfg
        self.kind = backend_kind
        self.model = RobotModel(cfg)
        self.safety = Safety(cfg)
        self.backend = make_backend(cfg, backend_kind, robot_sn)
        self.viz = None
        if viz_mode and viz_mode != "off":
            from .viz import RerunViz
            self.viz = RerunViz(RobotModel(cfg), viz_mode)
        self.executor = Executor(cfg, self.backend, self.model, self.safety, self.viz)
        home_tcp, home_yaw = self.model.fk_api(self.model.q_home)
        self.planner = Planner(cfg, home_tcp, home_yaw)
        self.pending = None
        c = cfg["camera"]
        self.frames_shape = (c["frames_height"], c["frames_width"])

    async def start(self):
        await self.backend.start()
        self.executor.sync()

    async def stop(self):
        await self.backend.stop()

    # -- actions ----------------------------------------------------------------

    async def prepare(self, primitive, params):
        ex = self.executor
        ex.sync()
        plan = self.planner.plan(primitive, params, ex.cmd_tcp, ex.cmd_yaw, ex.cmd_width)
        paths = ex.prepare(plan)
        self.pending = (plan, paths)
        if self.viz is not None:
            self.viz.log_plan(plan.segments[0].tag, np.concatenate(paths))
            self.viz.log_status("waiting for approval: " + primitive)
        kind = "contact" if primitive in CONTACT_PRIMITIVES else "free"
        duration = len(np.concatenate(paths)) * ex.dt
        return {"kind": kind, "summary": plan.summary() + f"\n  planned duration ~{duration:.1f} s"}

    async def cancel(self):
        self.pending = None
        if self.viz is not None:
            self.viz.log_status("cancelled")

    async def run(self):
        if self.pending is None:
            raise ActionError("no prepared action")
        plan, paths = self.pending
        self.pending = None
        frames = {"t": [], "rgb": [], "depth": []}
        cam = self.backend.camera()
        h, w = self.frames_shape

        def grab(t):
            rgb, depth, _ = cam.grab()
            sy, sx = max(1, rgb.shape[0] // h), max(1, rgb.shape[1] // w)
            frames["t"].append(t)
            frames["rgb"].append(rgb[::sy, ::sx][:h, :w])
            frames["depth"].append(depth[::sy, ::sx][:h, :w])

        res = await self.executor.run(plan, paths, frame_cb=grab)
        res["frames"] = {k: np.array(v) for k, v in frames.items()}
        if self.viz is not None:
            self.viz.log_status(f"{plan.primitive}: {res['status']}")
        return res

    # -- sensing ----------------------------------------------------------------

    async def observe(self):
        rgb, depth, _ = self.backend.camera().grab()
        p, yaw = self.model.fk_api(self.backend.state()["q"])
        return {"rgb": rgb, "depth": depth, "tcp_meas": p, "yaw_meas": np.array(yaw)}

    def calibration(self):
        """Camera intrinsics and extrinsics from the lab model's calibrated ZED camera.

        VERIFY on the real cell against the lab calibration files and the ZED's own
        intrinsics. Extrinsics are camera-to-world in the API frame, OpenCV axes.
        """
        m = self.model.m
        c = self.cfg["camera"]
        cid = m.camera(c["sim_camera"]).id
        W, H = c["width"], c["height"]
        sx, sy = m.cam_sensorsize[cid]
        if sx > 0:
            fx = m.cam_intrinsic[cid][0] / sx * W
            fy = m.cam_intrinsic[cid][1] / sy * H
            cx = W / 2 + m.cam_intrinsic[cid][2] / sx * W
            cy = H / 2 + m.cam_intrinsic[cid][3] / sy * H
        else:
            fy = fx = 0.5 * H / np.tan(np.deg2rad(m.cam_fovy[cid]) / 2)
            cx, cy = W / 2, H / 2
        self.model.set_q(self.model.q_home)
        R = self.model.d.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = self.model.to_api(self.model.d.cam_xpos[cid])
        return {c["name"]: {"width": W, "height": H,
                            "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                            "T_parent_cam": T.tolist(), "parent_frame": "world",
                            "depth_units": "metres, z-depth along optical axis, 0 = invalid"}}

    def status(self):
        st = self.backend.state()
        p, yaw = self.model.fk_api(st["q"])
        return {"fault": self.safety.fault, "tcp": p.tolist(), "yaw": yaw,
                "gripper_width": st["gripper_width"], "backend": self.kind}

    def clear_fault(self):
        self.safety.clear()
        self.executor.sync()


# -- worker process -----------------------------------------------------------------

def worker_main(cfg, backend_kind, robot_sn, viz_mode, conn):
    async def main():
        try:
            core = RobotCore(cfg, backend_kind, robot_sn, viz_mode)
            await core.start()
        except Exception:
            conn.send(("error", traceback.format_exc()))
            return
        conn.send(("ready", None))
        try:
            while True:
                if not conn.poll():
                    await asyncio.sleep(0.005)
                    continue
                method, args = conn.recv()
                if method == "shutdown":
                    break
                try:
                    fn = getattr(core, method)
                    out = fn(*args)
                    if asyncio.iscoroutine(out):
                        out = await out
                    conn.send(("ok", out))
                except (ActionError, SafetyError) as e:
                    conn.send((type(e).__name__, str(e)))
                except Exception:
                    conn.send(("error", traceback.format_exc()))
        finally:
            await core.stop()
    asyncio.run(main())


class WorkerClient:
    """Runs RobotCore in a separate process; one call at a time."""

    def __init__(self, cfg, backend_kind, robot_sn=None, viz_mode="off"):
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=worker_main, args=(cfg, backend_kind, robot_sn, viz_mode, child),
                                daemon=True)
        self.proc.start()
        self.lock = threading.Lock()
        tag, info = self.conn.recv()
        if tag != "ready":
            raise RuntimeError(f"robot worker failed to start: {info}")

    def call(self, method, *args):
        with self.lock:
            self.conn.send((method, args))
            tag, out = self.conn.recv()
        if tag == "ok":
            return out
        if tag == "ActionError":
            raise ActionError(out)
        if tag == "SafetyError":
            raise SafetyError(out)
        raise RuntimeError(out)

    def close(self):
        try:
            with self.lock:
                self.conn.send(("shutdown", ()))
        finally:
            self.proc.join(timeout=10)
