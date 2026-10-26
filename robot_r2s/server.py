"""Robot HTTP server: the same API as the simulated source world, on the real cell.

Auth, tokens, request log and the .npz response format follow
r2s_exam/server/app.py. Differences from the sim server:
  - every motion is checked by the safety layer, then approved by the operator
    in this terminal (with the ghost and path showing in Rerun) before it runs;
  - /v1/reset sends the robot home and waits for the operator to restore the
    scene by hand;
  - a protective stop or fault blocks all motion until the operator clears it
    (robot_r2s clear-fault, which uses an operator token);
  - status values: ok, protective_stop, rejected.
"""

import hashlib
import io
import json
import os
import threading
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from .primitives import ActionError, CONTACT_PRIMITIVES, PRIMITIVES
from .safety import SafetyError

PUBLIC_LOG_FIELDS = ("t", "tcp_cmd", "tcp_meas", "yaw_cmd", "yaw_meas", "width_cmd",
                     "width_meas", "wrench")
PUBLIC_OBS_FIELDS = ("rgb", "depth", "tcp_meas", "yaw_meas")
PUBLIC_EVENT_FIELDS = ("t", "type", "segment")


def token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


def add_token(tokens_file, name, scenes, operator=False):
    """Create a new access token; only its hash is stored. Returns the token."""
    path = Path(tokens_file).expanduser()
    tokens = json.loads(path.read_text()) if path.exists() else {}
    token = uuid.uuid4().hex + uuid.uuid4().hex
    tokens[token_hash(token)] = {"name": name, "scenes": list(scenes), "operator": bool(operator)}
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(tokens, f, indent=1)
    return token


def _npz(arrays, meta):
    buf = io.BytesIO()
    np.savez_compressed(buf, meta=np.array(json.dumps(meta)), **arrays)
    return buf.getvalue()


class ResetReq(BaseModel):
    seed: str


class ObserveReq(BaseModel):
    camera: str


class ExecuteReq(BaseModel):
    primitive: str
    params: dict = {}


class Lab:
    def __init__(self, cfg, robot, approver):
        self.cfg = cfg
        self.robot = robot                      # WorkerClient (or anything with .call)
        self.approver = approver
        s = cfg["server"]
        self.scenes = list(s["scenes"])
        self.budget = s.get("budget_per_scene")
        self.extra_fields = tuple(s.get("public_extra_fields", ()))
        self.tokens = json.loads(Path(s["tokens_file"]).expanduser().read_text())
        self.log_dir = Path(s["log_dir"]).expanduser()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.log_dir / "budget_state.json"
        self.used = json.loads(self.state_file.read_text()) if self.state_file.exists() else {}
        self.active = {}        # client -> scene
        self.episodes = {}      # (client, scene) -> episode index
        self.lock = threading.Lock()

    def client(self, request, operator=False):
        auth = request.headers.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            raise HTTPException(401, "missing bearer token")
        entry = self.tokens.get(token_hash(auth[7:].strip()))
        if entry is None:
            raise HTTPException(401, "invalid token")
        if operator and not entry.get("operator"):
            raise HTTPException(403, "operator token required")
        return entry

    def remaining(self, client, scene):
        if not self.budget:
            return None
        used = self.used.get(client, {}).get(scene, {})
        return {k: v - used.get(k, 0) for k, v in self.budget.items()}

    def charge(self, client, scene, kind):
        if not self.budget:
            return
        if self.remaining(client, scene)[kind] <= 0:
            raise HTTPException(429, f"budget exhausted: {kind}")
        u = self.used.setdefault(client, {}).setdefault(scene, {})
        u[kind] = u.get(kind, 0) + 1
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.used, indent=1))
        tmp.replace(self.state_file)

    def scene(self, client):
        scene = self.active.get(client)
        if scene is None:
            raise HTTPException(409, "call reset with a scene first")
        return scene

    def log(self, record):
        with open(self.log_dir / "requests.jsonl", "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def require_no_fault(self):
        fault = self.robot.call("status")["fault"]
        if fault:
            raise HTTPException(409, "robot is stopped after a fault and is waiting for operator review")


def create_app(cfg, robot, approver):
    lab = Lab(cfg, robot, approver)
    app = FastAPI(title="robot_r2s", docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def audit(request: Request, call_next):
        t0 = time.time()
        rid = uuid.uuid4().hex[:12]
        body = await request.body()
        name = lab.tokens.get(token_hash(request.headers.get("authorization", "")[7:].strip()), {}).get("name")
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:
            lab.log({"id": rid, "error": traceback.format_exc()})
            response = JSONResponse({"error": f"internal error {rid}"}, status_code=500)
            status = 500
        lab.log({"id": rid, "time": t0, "client": name, "host": request.client.host if request.client else None,
                 "path": request.url.path, "body": body.decode(errors="replace")[:2000],
                 "status": status, "seconds": round(time.time() - t0, 3)})
        return response

    @app.exception_handler(Exception)
    async def internal(request, exc):
        rid = uuid.uuid4().hex[:12]
        lab.log({"id": rid, "error": "".join(traceback.format_exception(exc))})
        return JSONResponse({"error": f"internal error {rid}"}, status_code=500)

    def run_motion(c, scene, primitive, params):
        """Prepare, approve and run one primitive. Returns (status, result or reason)."""
        try:
            prep = lab.robot.call("prepare", primitive, params)
        except ActionError as e:
            raise HTTPException(400, str(e))
        except SafetyError as e:
            lab.log({"approval": "safety_reject", "client": c["name"], "scene": scene,
                     "primitive": primitive, "reason": str(e)})
            return "rejected", f"safety: {e}"
        ok, note = lab.approver.ask(prep["kind"], f"client {c['name']} scene {scene}\n" + prep["summary"])
        lab.log({"approval": "approved" if ok else "rejected", "client": c["name"], "scene": scene,
                 "primitive": primitive, "params": params, "note": note})
        if not ok:
            lab.robot.call("cancel")
            return "rejected", note
        return "ran", lab.robot.call("run")

    @app.get("/v1/info")
    def info(request: Request):
        c = lab.client(request)
        cam = cfg["camera"]
        return {"level": cfg["server"].get("level", "L0"), "description": cfg["server"].get("description", ""),
                "seeds": c["scenes"], "budget_per_seed": lab.budget, "cameras": [cam["name"]],
                "action_frames": {"camera": cam["name"], "width": cam["frames_width"],
                                  "height": cam["frames_height"], "hz": cam["action_frames_hz"]},
                "log_hz": cfg["logging"]["log_hz"], "support": {"type": "plane", "z": 0.0},
                "primitives": list(PRIMITIVES), "contact_primitives": list(CONTACT_PRIMITIVES),
                "robot": "Flexiv Rizon 4s + Grav (GN01)", "approval": "operator approves motion"}

    @app.get("/v1/budget")
    def budget(request: Request):
        c = lab.client(request)
        return {s: lab.remaining(c["name"], s) for s in c["scenes"]}

    @app.post("/v1/reset")
    def reset(req: ResetReq, request: Request):
        c = lab.client(request)
        if req.seed not in c["scenes"] or req.seed not in lab.scenes:
            raise HTTPException(403, "scene not available to this token")
        with lab.lock:
            lab.require_no_fault()
            k = (c["name"], req.seed)
            if k in lab.episodes:
                lab.charge(c["name"], req.seed, "resets")
            status, out = run_motion(c, req.seed, "home", {})
            if status != "ran" or out["status"] != "ok":
                raise HTTPException(409, f"could not send the robot home before the reset: "
                                         f"{out if status != 'ran' else out['status']}")
            done = lab.approver.confirm(f"RESET scene {req.seed}: restore the scene by hand, keep clear of the robot.")
            lab.log({"reset": req.seed, "client": c["name"], "confirmed": done})
            if not done:
                raise HTTPException(409, "operator did not confirm the reset")
            lab.episodes[k] = lab.episodes.get(k, -1) + 1
            lab.active[c["name"]] = req.seed
            return {"seed": req.seed, "episode": lab.episodes[k], "budget": lab.remaining(c["name"], req.seed)}

    @app.get("/v1/calibration")
    def calibration(request: Request):
        lab.client(request)
        return lab.robot.call("calibration")

    @app.post("/v1/observe")
    def observe(req: ObserveReq, request: Request):
        c = lab.client(request)
        with lab.lock:
            scene = lab.scene(c["name"])
            if req.camera != cfg["camera"]["name"]:
                raise HTTPException(400, f"unknown camera; available: {[cfg['camera']['name']]}")
            lab.charge(c["name"], scene, "observations")
            o = lab.robot.call("observe")
            arrays = {k: o[k] for k in PUBLIC_OBS_FIELDS}
            arrays["t"] = np.array(time.time())
            meta = {"camera": req.camera, "episode": lab.episodes[(c["name"], scene)],
                    "budget": lab.remaining(c["name"], scene)}
            return Response(_npz(arrays, meta), media_type="application/octet-stream")

    @app.post("/v1/execute")
    def execute(req: ExecuteReq, request: Request):
        c = lab.client(request)
        with lab.lock:
            scene = lab.scene(c["name"])
            lab.require_no_fault()
            if req.primitive not in PRIMITIVES:
                raise HTTPException(400, f"unknown primitive {req.primitive!r}; available: {list(PRIMITIVES)}")
            kind = "contact_actions" if req.primitive in CONTACT_PRIMITIVES else "free_moves"
            if lab.budget and lab.remaining(c["name"], scene)[kind] <= 0:
                raise HTTPException(429, f"budget exhausted: {kind}")
            status, out = run_motion(c, scene, req.primitive, req.params)
            meta = {"primitive": req.primitive, "params": req.params, "episode": lab.episodes[(c["name"], scene)]}
            if status == "rejected":
                meta.update(status="rejected", reason=out, events=[], budget=lab.remaining(c["name"], scene))
                return Response(_npz({}, meta), media_type="application/octet-stream")
            lab.charge(c["name"], scene, kind)
            log = out["log"]
            arrays = {k: log[k] for k in PUBLIC_LOG_FIELDS + lab.extra_fields if k in log}
            fr = out["frames"]
            arrays.update(frame_t=fr["t"], frames_rgb=fr["rgb"], frames_depth=fr["depth"])
            events = [{k: e[k] for k in PUBLIC_EVENT_FIELDS if k in e} for e in out["events"]]
            meta.update(status=out["status"], events=events, frame_camera=cfg["camera"]["name"],
                        budget=lab.remaining(c["name"], scene))
            return Response(_npz(arrays, meta), media_type="application/octet-stream")

    @app.post("/v1/operator/clear_fault")
    def clear_fault(request: Request):
        c = lab.client(request, operator=True)
        lab.robot.call("clear_fault")
        lab.log({"clear_fault": c["name"]})
        return lab.robot.call("status")

    @app.get("/v1/operator/status")
    def status(request: Request):
        lab.client(request, operator=True)
        return lab.robot.call("status")

    return app
