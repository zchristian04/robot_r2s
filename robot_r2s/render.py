"""RGB-D rendering of named MJCF cameras, for the simulated robot backend.

Copied from r2s_exam/world/render.py. Backends:
  gl       mujoco.Renderer (EGL/OSMesa/GLFW). Use on a machine with a GPU.
  raycast  CPU ray casting with mj_multiRay and Lambert shading, no shadows.
           For headless development machines without OpenGL.

Depth is z-depth along the optical axis in metres; 0 marks no return.
"""

import functools
import subprocess
import sys

import mujoco
import numpy as np

VISIBLE_GROUPS = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)
BACKGROUND = np.array([0.62, 0.65, 0.69])


_PROBE = """
import mujoco
m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom size='1'/></worldbody></mujoco>")
mujoco.Renderer(m, 8, 8).close()
"""


@functools.lru_cache(maxsize=1)
def gl_available():
    # A failed GL init can abort the interpreter, so probe in a subprocess.
    try:
        r = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, timeout=60)
        return r.returncode == 0
    except Exception:
        return False


class Renderer:
    def __init__(self, model, width, height, backend="auto"):
        if backend == "auto":
            backend = "gl" if gl_available() else "raycast"
        self.backend = backend
        self.m, self.w, self.h = model, width, height
        if backend == "gl":
            self._rgb = mujoco.Renderer(model, height, width)
            self._depth = mujoco.Renderer(model, height, width)
            self._depth.enable_depth_rendering()
        elif backend != "raycast":
            raise ValueError(f"unknown render backend {backend!r}")
        self._light = self._key_light_dir()

    def _key_light_dir(self):
        for i in range(self.m.nlight):
            if self.m.light_type[i] == mujoco.mjtLightType.mjLIGHT_DIRECTIONAL:
                v = self.m.light_dir[i].copy()
                return v / np.linalg.norm(v)
        return np.array([0.0, 0.0, -1.0])

    def render(self, data, camera):
        cam_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if cam_id < 0:
            raise KeyError(camera)
        if self.backend == "gl":
            self._rgb.update_scene(data, camera=camera)
            rgb = self._rgb.render().copy()
            self._depth.update_scene(data, camera=camera)
            depth = self._depth.render().astype(np.float32)
            far = self.m.vis.map.zfar * self.m.stat.extent
            depth[depth >= 0.99 * far] = 0.0
            return rgb, depth
        return self._raycast(data, cam_id)

    def _raycast(self, data, cam_id):
        m = self.m
        f = 0.5 * self.h / np.tan(np.deg2rad(m.cam_fovy[cam_id]) / 2)
        u = (np.arange(self.w) + 0.5 - self.w / 2) / f
        v = (np.arange(self.h) + 0.5 - self.h / 2) / f
        uu, vv = np.meshgrid(u, v)
        # MuJoCo camera frame: x right, y up, looking along -z. Unit z component,
        # so the returned distance is the z-depth.
        dirs_cam = np.stack([uu, -vv, -np.ones_like(uu)], -1).reshape(-1, 3)
        R = data.cam_xmat[cam_id].reshape(3, 3)
        dirs = dirs_cam @ R.T
        n = dirs.shape[0]
        geomid = np.zeros(n, dtype=np.int32)
        dist = np.zeros(n, dtype=np.float64)
        normal = np.zeros(n * 3, dtype=np.float64)
        mujoco.mj_multiRay(m, data, data.cam_xpos[cam_id].copy(), dirs.reshape(-1).copy(),
                           VISIBLE_GROUPS, 1, -1, geomid, dist, normal, n, 10.0)
        hit = geomid >= 0
        normal = normal.reshape(-1, 3)
        col = np.tile(BACKGROUND, (n, 1))
        g = geomid[hit]
        base = m.geom_rgba[g, :3].copy()
        mat = m.geom_matid[g]
        has_mat = mat >= 0
        base[has_mat] = m.mat_rgba[mat[has_mat], :3]
        ndotl = np.clip(-(normal[hit] @ self._light), 0, 1)
        col[hit] = base * (0.35 + 0.65 * ndotl[:, None])
        rgb = (np.clip(col, 0, 1) * 255).astype(np.uint8).reshape(self.h, self.w, 3)
        depth = np.where(hit, dist, 0.0).astype(np.float32).reshape(self.h, self.w)
        return rgb, depth

    def close(self):
        if self.backend == "gl":
            self._rgb.close()
            self._depth.close()
