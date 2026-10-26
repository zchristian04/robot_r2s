"""Tests on the MuJoCo stand-in. Nothing here touches hardware."""

import asyncio
import io
import json
import threading
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from robot_r2s.core import RobotCore
from robot_r2s.model import RobotModel
from robot_r2s.primitives import ActionError
from robot_r2s.safety import SafetyError
from robot_r2s.server import add_token, create_app

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "lab.json"


def load_cfg(tmp_path=None):
    cfg = json.loads(CONFIG.read_text())
    cfg["camera"].update(width=160, height=90, frames_width=80, frames_height=45, action_frames_hz=1.0)
    if tmp_path is not None:
        cfg["server"].update(tokens_file=str(tmp_path / "tokens.json"), log_dir=str(tmp_path / "logs"),
                             budget_per_scene={"contact_actions": 2, "free_moves": 5, "resets": 2,
                                               "observations": 10})
    return cfg


class InlineRobot:
    """RobotCore on its own event-loop thread; same .call interface as WorkerClient."""

    def __init__(self, cfg):
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.core = RobotCore(cfg, "sim")
        self.call("start")

    def call(self, method, *args):
        out = getattr(self.core, method)(*args)
        if asyncio.iscoroutine(out):
            out = asyncio.run_coroutine_threadsafe(out, self.loop).result(timeout=600)
        return out


class ScriptedApprover:
    def __init__(self, answers=None):
        self.answers = list(answers or [])
        self.asked = []

    def ask(self, kind, summary):
        self.asked.append(kind)
        ok = self.answers.pop(0) if self.answers else True
        return ok, "approved" if ok else "rejected in test"

    def confirm(self, message):
        return True


@pytest.fixture(scope="module")
def robot():
    return InlineRobot(load_cfg())


def test_ik_roundtrip():
    m = RobotModel(load_cfg())
    p, R = m.fk(m.q_home)
    q = m.ik(p + np.array([0.05, -0.03, -0.04]), 0.3, m.q_home)
    assert q is not None
    p2, yaw2 = m.fk_api(q)
    assert np.allclose(p2, m.to_api(p + np.array([0.05, -0.03, -0.04])), atol=2e-4)
    assert abs(yaw2 - 0.3) < 2e-3


def test_bad_requests_rejected(robot):
    with pytest.raises(ActionError):
        robot.call("prepare", "teleport", {})
    with pytest.raises(ActionError):
        robot.call("prepare", "push", {"distance": 0.01})
    with pytest.raises(SafetyError):
        robot.call("prepare", "move_to", {"tcp": [0.0, 0.0, 0.2]})       # outside the workspace


def test_move_to(robot):
    tcp = robot.call("status")["tcp"]
    target = [tcp[0] + 0.05, tcp[1] + 0.05, 0.15]
    robot.call("prepare", "move_to", {"tcp": target})
    res = robot.call("run")
    assert res["status"] == "ok"
    assert np.linalg.norm(res["log"]["tcp_meas"][-1] - target) < 0.005


def test_touch_stops_on_table(robot):
    tcp = robot.call("status")["tcp"]
    robot.call("prepare", "touch", {"point": [tcp[0], tcp[1], 0.04], "direction": [0, 0, -1],
                                    "max_travel": 0.045, "speed": 0.01, "stop_force": 3.0})
    res = robot.call("run")
    assert res["status"] == "ok", res["events"]
    stops = [e for e in res["events"] if e["type"] == "stop_force"]
    assert len(stops) == 1
    i = np.searchsorted(res["log"]["t"], stops[0]["t"])
    assert abs(res["log"]["tcp_meas"][i, 2]) < 0.03       # grasp centre stops near the table top
    assert res["frames"]["rgb"].ndim == 4


def test_server_auth_approval_budget(tmp_path):
    cfg = load_cfg(tmp_path)
    token = add_token(cfg["server"]["tokens_file"], "tester", ["scene_01"])
    approver = ScriptedApprover([True, False])      # reset's home move approved, next action rejected
    c = TestClient(create_app(cfg, InlineRobot(cfg), approver))
    assert c.get("/v1/info").status_code == 401
    c.headers["Authorization"] = f"Bearer {token}"
    assert c.post("/v1/execute", json={"primitive": "home", "params": {}}).status_code == 409
    assert c.post("/v1/reset", json={"seed": "scene_02"}).status_code == 403
    assert c.post("/v1/reset", json={"seed": "scene_01"}).status_code == 200
    before = c.get("/v1/budget").json()["scene_01"]
    r = c.post("/v1/execute", json={"primitive": "move_to", "params": {"tcp": [0.6, -0.1, 0.18]}})
    assert r.status_code == 200
    with np.load(io.BytesIO(r.content), allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
    assert meta["status"] == "rejected"
    assert c.get("/v1/budget").json()["scene_01"] == before      # a rejected action costs nothing
    assert c.post("/v1/execute", json={"primitive": "push", "params": {"contact_point": "x"}}).status_code == 400
    text = c.get("/v1/calibration").text + c.get("/v1/info").text
    for bad in ("/home", "newton_handoff", str(tmp_path), "Traceback"):
        assert bad not in text
