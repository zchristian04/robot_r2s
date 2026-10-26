"""Robot kinematics on the lab MuJoCo model (Rizon 4s + GN01).

The model from newton_handoff_20260921 is used for forward kinematics, inverse
kinematics, the table-collision check and the Rerun view, for both the real and
the simulated robot. TCP positions are reported in API coordinates: the model
frame shifted so the table top is z = 0.

Tool orientation is "pointing down" with a yaw about world z. At yaw 0 the tool
x axis points along world -x and the fingers open along world y, matching the
lab home posture.
"""

from pathlib import Path

import mujoco
import numpy as np

R_DOWN = np.diag([-1.0, 1.0, -1.0])


def rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def tool_rotation(yaw):
    return rot_z(yaw) @ R_DOWN


def yaw_of(R):
    return float(np.arctan2(-R[1, 0], -R[0, 0]))


def rotvec_error(R_goal, R):
    """Axis-angle vector rotating R onto R_goal, world frame."""
    E = R_goal @ R.T
    angle = np.arccos(np.clip((np.trace(E) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
    n = np.linalg.norm(axis)
    if n < 1e-9:
        # angle ~ pi: take the axis from the symmetric part
        w, v = np.linalg.eigh((E + np.eye(3)) / 2.0)
        axis = v[:, np.argmax(w)]
        return axis * angle
    return axis / n * angle


class RobotModel:
    def __init__(self, cfg):
        mc = cfg["model"]
        self.path = Path(mc["path"]).expanduser()
        self.m = mujoco.MjModel.from_xml_path(str(self.path))
        self.d = mujoco.MjData(self.m)
        self.table_top_z = float(mc["table_top_z"])
        self.offset = np.array([0.0, 0.0, self.table_top_z])
        self.arm_jids = [self.m.joint(n).id for n in mc["arm_joints"]]
        self.qadr = np.array([self.m.jnt_qposadr[j] for j in self.arm_jids])
        self.dadr = np.array([self.m.jnt_dofadr[j] for j in self.arm_jids])
        self.q_lo = np.array([self.m.jnt_range[j, 0] for j in self.arm_jids])
        self.q_hi = np.array([self.m.jnt_range[j, 1] for j in self.arm_jids])
        self.width_adr = self.m.jnt_qposadr[self.m.joint(mc["gripper_joint"]).id]
        self.tcp_site = self.m.site(mc["tcp_site"]).id
        self.table_geom = self.m.geom(mc["table_geom"]).id
        self.q_home = np.asarray(cfg["robot"]["q_home"], float)
        self.robot_bodies = self._subtree(self.m.body("base").id)
        self.robot_geoms = {g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] in self.robot_bodies}
        mujoco.mj_resetData(self.m, self.d)
        if self.m.nkey:
            mujoco.mj_resetDataKeyframe(self.m, self.d, 0)

    def _subtree(self, root):
        out = {root}
        for b in range(self.m.nbody):
            p = b
            while p > 0:
                if p == root:
                    out.add(b)
                    break
                p = self.m.body_parentid[p]
        return out

    # -- frames ---------------------------------------------------------------

    def to_api(self, p_model):
        return np.asarray(p_model, float) - self.offset

    def to_model(self, p_api):
        return np.asarray(p_api, float) + self.offset

    # -- kinematics -------------------------------------------------------------

    def set_q(self, q, width=None):
        self.d.qpos[self.qadr] = q
        if width is not None:
            self.d.qpos[self.width_adr] = width
        mujoco.mj_kinematics(self.m, self.d)

    def fk(self, q, width=None):
        """TCP position (model frame) and rotation for joint vector q."""
        self.set_q(q, width)
        return self.d.site_xpos[self.tcp_site].copy(), self.d.site_xmat[self.tcp_site].reshape(3, 3).copy()

    def fk_api(self, q):
        p, R = self.fk(q)
        return self.to_api(p), yaw_of(R)

    def within_limits(self, q, margin):
        return bool(np.all(q >= self.q_lo + margin) and np.all(q <= self.q_hi - margin))

    def ik(self, p_model, yaw, q_seed=None, margin=0.0, iters=300, tol_p=1e-4, tol_r=1e-3):
        """Joint vector reaching TCP position p_model with the tool down at yaw, or None."""
        R_goal = tool_rotation(yaw)
        seeds = [np.asarray(q_seed, float)] if q_seed is not None else []
        seeds.append(self.q_home.copy())
        lo, hi = self.q_lo + margin, self.q_hi - margin
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        for seed in seeds:
            q = np.clip(seed.copy(), lo, hi)
            for _ in range(iters):
                p, R = self.fk(q)
                ep = p_model - p
                er = rotvec_error(R_goal, R)
                if np.linalg.norm(ep) < tol_p and np.linalg.norm(er) < tol_r:
                    return q
                mujoco.mj_jacSite(self.m, self.d, jacp, jacr, self.tcp_site)
                J = np.vstack([jacp[:, self.dadr], jacr[:, self.dadr]])
                e = np.concatenate([ep, er])
                lam = 1e-2
                dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e)
                step = np.linalg.norm(dq)
                if step > 0.2:
                    dq *= 0.2 / step
                q = np.clip(q + dq, lo, hi)
        return None

    # -- collision --------------------------------------------------------------

    def table_collision(self, q, width=None, clearance=0.0):
        """True if robot geometry is within `clearance` of (or inside) the table."""
        self.set_q(q, width)
        mujoco.mj_collision(self.m, self.d)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            pair = {c.geom1, c.geom2}
            if self.table_geom in pair and pair & self.robot_geoms and c.dist < clearance:
                return True
        return False
