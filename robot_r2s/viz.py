"""Rerun debugging view for the operator (never exposed to the agent).

Shows the robot at its measured configuration and, as a translucent green
"ghost", at its commanded/target configuration, plus the planned TCP path of
the next segment, the wrist wrench and the action status, all on one timeline.

Meshes come straight from the MuJoCo model (visual mesh geoms of the robot), so
no separate URDF or mesh files are needed. Mode:
  "spawn"         open the native Rerun viewer on this machine
  "serve"         serve on localhost for a browser over an SSH tunnel
  "save:<path>"   write an .rrd file to open later with `rerun <path>`
  "off"           no visualization
"""

import numpy as np

GHOST_RGBA = [0.2, 0.9, 0.3, 0.35]


class RerunViz:
    def __init__(self, model, mode="spawn", app_id="robot_r2s"):
        import rerun as rr
        self.rr = rr
        self.model = model
        rr.init(app_id)
        if mode == "spawn":
            rr.spawn()
        elif mode == "serve":
            # Serve on localhost only; reach it through an SSH tunnel.
            rr.serve_web(open_browser=False)
        elif mode.startswith("save:"):
            rr.save(mode[5:])
        self.m = model.m
        self.bodies = sorted({int(self.m.geom_bodyid[g]) for g in self._visual_geoms()})
        self._log_meshes("robot/current", None)
        self._log_meshes("robot/target", GHOST_RGBA)
        rr.log("table", rr.Boxes3D(centers=[[0, 0, -0.01]], half_sizes=[[0.74, 0.295, 0.01]]), static=True)

    def _visual_geoms(self):
        m = self.m
        return [g for g in sorted(self.model.robot_geoms)
                if m.geom_type[g] == 7 and m.geom_group[g] in (0, 1)]   # mjGEOM_MESH

    def _set_time(self, t):
        rr = self.rr
        if hasattr(rr, "set_time"):
            rr.set_time("t", duration=float(t))
        else:
            rr.set_time_seconds("t", float(t))

    def _log_meshes(self, root, rgba):
        rr, m = self.rr, self.m
        import mujoco
        for g in self._visual_geoms():
            mid = m.geom_dataid[g]
            v0, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            f0, nf = m.mesh_faceadr[mid], m.mesh_facenum[mid]
            verts = m.mesh_vert[v0:v0 + nv]
            faces = m.mesh_face[f0:f0 + nf]
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.geom_quat[g])
            verts = verts @ R.reshape(3, 3).T + m.geom_pos[g]           # geom -> body frame
            if rgba is None:
                mat = m.geom_matid[g]
                rgba_g = m.mat_rgba[mat] if mat >= 0 else m.geom_rgba[g]
            else:
                rgba_g = rgba
            body = m.body(m.geom_bodyid[g]).name
            rr.log(f"{root}/{body}/g{g}", rr.Mesh3D(vertex_positions=verts, triangle_indices=faces,
                                                     albedo_factor=list(map(float, rgba_g))), static=True)

    def _log_pose(self, root, q, width):
        rr, model = self.rr, self.model
        model.set_q(q, width)
        d = model.d
        for b in self.bodies:
            rr.log(f"{root}/{self.m.body(b).name}",
                   rr.Transform3D(translation=model.to_api(d.xpos[b]), mat3x3=d.xmat[b].reshape(3, 3)))

    def log_state(self, t, q_meas, q_target, width_meas, width_target, wrench):
        rr = self.rr
        self._set_time(t)
        self._log_pose("robot/current", q_meas, width_meas)
        self._log_pose("robot/target", q_target, width_target)
        p, _ = self.model.fk_api(q_meas)
        rr.log("wrench/force", rr.Arrows3D(origins=[p], vectors=[np.asarray(wrench[:3]) * 0.01]))

    def log_plan(self, tag, joint_path):
        rr = self.rr
        if joint_path is None or not len(joint_path):
            return
        pts = [self.model.fk_api(q)[0] for q in joint_path[:: max(1, len(joint_path) // 50)]]
        pts.append(self.model.fk_api(joint_path[-1])[0])
        rr.log("plan/path", rr.LineStrips3D([pts]))
        rr.log("status", rr.TextLog(f"segment {tag}"))

    def log_status(self, text):
        self.rr.log("status", self.rr.TextLog(text))
