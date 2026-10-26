"""MuJoCo stand-in for the real robot, for testing the server without hardware.

Uses the same lab model (or a scene that includes it, set by cfg["sim"]["scene"]).
The arm follows joint targets through the model's position actuators, so it is
stiffer than the real joint-impedance controller; the wrench is the sum of
contact forces on robot geometry, with torques about the TCP. Time advances
only when the executor calls wait(), so tests run faster than real time.
"""

from pathlib import Path

import mujoco
import numpy as np

from ..render import Renderer
from .base import Backend


class SimCamera:
    def __init__(self, backend, cam_name, width, height):
        self.b = backend
        self.cam_name = cam_name
        self.renderer = Renderer(backend.m, width, height)

    def grab(self):
        rgb, depth = self.renderer.render(self.b.d, self.cam_name)
        return rgb, depth, self.b.d.time


class SimBackend(Backend):
    kind = "sim"

    def __init__(self, cfg):
        mc = cfg["model"]
        scene = (cfg.get("sim") or {}).get("scene") or mc["path"]
        self.m = mujoco.MjModel.from_xml_path(str(Path(scene).expanduser()))
        self.d = mujoco.MjData(self.m)
        self.arm_jids = [self.m.joint(n).id for n in mc["arm_joints"]]
        self.qadr = np.array([self.m.jnt_qposadr[j] for j in self.arm_jids])
        self.dadr = np.array([self.m.jnt_dofadr[j] for j in self.arm_jids])
        self.arm_act = np.array([self.m.actuator(n).id for n in mc["arm_joints"]])
        gj = self.m.joint(mc["gripper_joint"]).id
        self.w_adr, self.w_dadr = self.m.jnt_qposadr[gj], self.m.jnt_dofadr[gj]
        self.g_act = self.m.actuator(mc["gripper_actuator"]).id
        self.tcp_site = self.m.site(mc["tcp_site"]).id
        base = self.m.body("base").id
        bodies = {b for b in range(self.m.nbody) if self._under(b, base)}
        self.robot_geoms = {g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] in bodies}
        r = cfg["robot"]
        self.q_home = np.asarray(r["q_home"], float)
        self.g_force_cap = float(r["gripper_force_cap"])
        self.g_target = float(r["gripper_home_width"])
        self.g_force = self.g_force_cap
        self.q_target = self.q_home.copy()
        self.cam_cfg = cfg["camera"]
        self._camera = None

    def _under(self, b, root):
        while b > 0:
            if b == root:
                return True
            b = self.m.body_parentid[b]
        return False

    async def start(self):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.qadr] = self.q_home
        self.d.qpos[self.w_adr] = self.g_target
        self.q_target = self.q_home.copy()
        mujoco.mj_forward(self.m, self.d)
        await self.wait(0.5)

    async def stop(self):
        pass

    def set_gains(self, kp, kd):
        pass

    def command(self, q_target):
        self.q_target = np.asarray(q_target, float).copy()

    def _step(self):
        self.d.ctrl[self.arm_act] = self.q_target
        w, v = self.d.qpos[self.w_adr], self.d.qvel[self.w_dadr]
        f = np.clip(2000.0 * (self.g_target - w) - 20.0 * v, -self.g_force, self.g_force)
        self.d.ctrl[self.g_act] = -f          # actuator gear is -1
        mujoco.mj_step(self.m, self.d)

    async def wait(self, dt):
        for _ in range(max(1, int(round(dt / self.m.opt.timestep)))):
            self._step()

    async def gripper_move(self, width, speed, force):
        self.g_target = float(width)
        self.g_force = min(float(force), self.g_force_cap)

    async def gripper_grasp(self, force):
        await self.gripper_move(0.0, 0.0, force)

    async def hold(self):
        self.q_target = self.d.qpos[self.qadr].copy()

    def _wrench(self):
        f_sum = np.zeros(3)
        m_sum = np.zeros(3)
        tcp = self.d.site_xpos[self.tcp_site]
        f6 = np.zeros(6)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            r1, r2 = c.geom1 in self.robot_geoms, c.geom2 in self.robot_geoms
            if r1 == r2:
                continue
            mujoco.mj_contactForce(self.m, self.d, i, f6)
            f_world = c.frame.reshape(3, 3).T @ f6[:3]
            # mj_contactForce gives the force of geom1 on geom2
            f_on_robot = f_world if r2 else -f_world
            f_sum += f_on_robot
            m_sum += np.cross(c.pos - tcp, f_on_robot)
        return np.concatenate([f_sum, m_sum])

    def state(self):
        return {
            "t": float(self.d.time),
            "q": self.d.qpos[self.qadr].copy(),
            "dq": self.d.qvel[self.dadr].copy(),
            "tau_ext": np.zeros(len(self.qadr)),
            "wrench": self._wrench(),
            "gripper_width": float(self.d.qpos[self.w_adr]),
            "gripper_force": float(abs(self.d.actuator_force[self.g_act])),
            "gripper_moving": bool(abs(self.d.qvel[self.w_dadr]) > 1e-3),
            "fault": None,
        }

    def now(self):
        return float(self.d.time)

    def camera(self):
        if self._camera is None:
            c = self.cam_cfg
            self._camera = SimCamera(self, c["sim_camera"], c["width"], c["height"])
        return self._camera
