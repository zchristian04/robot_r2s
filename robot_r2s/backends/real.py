"""Real Flexiv Rizon 4s + GN01 backend.

Arm: aioflexiv's Python-owned real-time torque loop in joint-impedance mode.
Gripper: Flexiv RDK Gripper API, with the bring-up sequence from flexiv_control
(Apache-2.0, Copyright 2026 Zihao Lu; see NOTICE).
Camera: ZED 2i through pyzed, grabbed in its own thread.

NOT YET RUN ON HARDWARE. Items marked VERIFY must be checked on the lab machine
before any motion:
  - a second flexivrdk.Robot connection for the gripper while aioflexiv's RT
    torque loop is running;
  - the wrench keys of aioflexiv's full state and their frame;
  - the ZED depth units and the camera serial.

aioflexiv's torque loop runs as an asyncio task that only yields with
asyncio.sleep(0). This backend must live in its own process (see worker.py) so
HTTP handling, camera capture and logging never delay a torque tick.
"""

import asyncio
import threading
import time

import numpy as np

from .base import Backend


def _make_controller_class():
    from aioflexiv import FlexivController

    class FullStateController(FlexivController):
        """Joint impedance that reads the full robot state every tick.

        The stock impedance mode reads state_minimal(), which has no wrench.
        Reading robot.state from outside the loop would consume a real-time
        state sample, so the wrench must be read here, inside the loop.
        """

        def step(self):
            if self.type != "impedance":
                return super().step()
            now = time.perf_counter()
            dt = 1e-3 if self._last_loop_time is None else max(now - self._last_loop_time, 1e-4)
            self._last_loop_time = now
            state = self.robot.state
            tau = self._impedance_torque(state)
            controller_tau = tau.copy()
            tau = self._clip_torque(self._apply_ext_offset(tau), dt)
            self.robot.step(tau)
            state["controller_torque"] = controller_tau
            state["last_torque"] = tau.copy()
            state["t_read"] = time.monotonic()
            self.state = state

    return FullStateController


class ZedCamera:
    """ZED 2i RGB-D capture in a background thread. VERIFY on the lab PC."""

    def __init__(self, width, height):
        import pyzed.sl as sl
        self.sl = sl
        self.cam = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD720
        init.depth_mode = sl.DEPTH_MODE.NEURAL
        init.coordinate_units = sl.UNIT.METER
        if self.cam.open(init) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("could not open the ZED camera")
        self.lock = threading.Lock()
        self.latest = None
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        sl = self.sl
        image, depth = sl.Mat(), sl.Mat()
        while self.running:
            if self.cam.grab() != sl.ERROR_CODE.SUCCESS:
                continue
            self.cam.retrieve_image(image, sl.VIEW.LEFT)
            self.cam.retrieve_measure(depth, sl.MEASURE.DEPTH)
            rgb = image.get_data()[:, :, 2::-1].copy()          # BGRA -> RGB
            d = np.nan_to_num(depth.get_data().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            with self.lock:
                self.latest = (rgb, d, time.monotonic())

    def grab(self):
        for _ in range(200):
            with self.lock:
                if self.latest is not None:
                    return self.latest
            time.sleep(0.01)
        raise RuntimeError("no frame from the ZED camera")

    def close(self):
        self.running = False
        self.cam.close()


class RealBackend(Backend):
    kind = "real"

    def __init__(self, cfg, robot_sn):
        if not robot_sn:
            raise ValueError("robot serial number required (--robot-sn or ROBOT_SN)")
        self.cfg = cfg
        self.robot_sn = robot_sn
        self.ctrl = None
        self.gripper = None
        self._robot_for_gripper = None
        self._camera = None
        self.gripper_name = cfg["robot"]["gripper_name"]

    async def start(self):
        Controller = _make_controller_class()
        # Never auto-clear faults: after a fault the robot must stay stopped
        # until the operator has looked at it.
        self.ctrl = Controller(self.robot_sn, auto_clear_fault=False)
        await self.ctrl.start()
        self.ctrl.switch("impedance")
        c = self.cfg["control"]
        self.set_gains(c["travel_kp"], c["travel_kd"])
        self.command(self.ctrl.state["qpos"])
        await asyncio.sleep(0.2)
        self._start_gripper()

    def _start_gripper(self):
        # Bring-up from flexiv_control backends/flexiv_rdk.py: Enable(name) must
        # come before Init(), otherwise every gripper command is a silent no-op.
        import flexivrdk
        self._robot_for_gripper = flexivrdk.Robot(self.robot_sn)   # VERIFY: concurrent with RT loop
        self.gripper = flexivrdk.Gripper(self._robot_for_gripper)
        self.gripper.Enable(self.gripper_name)
        self.gripper.Init()

    async def stop(self):
        if self.ctrl is not None:
            await self.ctrl.stop()
        if self._camera is not None:
            self._camera.close()

    def set_gains(self, kp, kd):
        self.ctrl.kp = np.asarray(kp, float)
        self.ctrl.kd = np.asarray(kd, float)

    def command(self, q_target):
        self.ctrl.set_now("q_desired", np.asarray(q_target, float))

    async def wait(self, dt):
        await asyncio.sleep(dt)

    async def gripper_move(self, width, speed, force):
        if self.gripper is None:
            raise RuntimeError(f"gripper {self.gripper_name!r} is not initialised")
        self.gripper.Move(float(width), float(speed), float(force))

    async def gripper_grasp(self, force):
        if self.gripper is None:
            raise RuntimeError(f"gripper {self.gripper_name!r} is not initialised")
        self.gripper.Grasp(float(force))

    async def hold(self):
        self.command(self.ctrl.state["qpos"])

    def state(self):
        s = self.ctrl.state
        if not self.ctrl.running:
            fault = "control loop stopped"
        else:
            fault = None
        gw = gf = 0.0
        gm = False
        if self.gripper is not None:
            gs = self.gripper.states()
            gw, gf = float(gs.width), float(gs.force)
            gm = bool(self.gripper.moving())
        wrench = s.get("ext_wrench_in_world")                      # VERIFY: key and frame
        return {
            "t": float(s.get("t_read", time.monotonic())),
            "q": np.asarray(s["qpos"], float),
            "dq": np.asarray(s["qvel"], float),
            "tau_ext": np.asarray(s["tau_ext"], float),
            "wrench": np.zeros(6) if wrench is None else np.asarray(wrench, float),
            "gripper_width": gw,
            "gripper_force": gf,
            "gripper_moving": gm,
            "fault": fault,
        }

    def now(self):
        return time.monotonic()

    def camera(self):
        if self._camera is None:
            c = self.cfg["camera"]
            self._camera = ZedCamera(c["width"], c["height"])
        return self._camera
