"""Command line.

    python -m robot_r2s.cli add-token --name jiaxin_run1 --scenes scene_01
    python -m robot_r2s.cli add-token --name operator --scenes scene_01 --operator
    python -m robot_r2s.cli serve --config configs/lab.json --backend sim
    python -m robot_r2s.cli serve --config configs/lab.json --backend real --robot-sn <SN> --host 127.0.0.1
    R2S_OPERATOR_TOKEN=... python -m robot_r2s.cli clear-fault --url http://127.0.0.1:8100

The real backend asks the operator to approve every motion in this terminal.
--auto-approve is accepted only with the sim backend.
"""

import argparse
import json
import os
from pathlib import Path

from .server import add_token, create_app


def load_config(path):
    return json.loads(Path(path).expanduser().read_text())


def main():
    ap = argparse.ArgumentParser(prog="robot_r2s")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("add-token")
    t.add_argument("--name", required=True)
    t.add_argument("--scenes", nargs="+", required=True)
    t.add_argument("--operator", action="store_true")
    t.add_argument("--config", default="configs/lab.json")
    s = sub.add_parser("serve")
    s.add_argument("--config", required=True)
    s.add_argument("--backend", choices=("sim", "real"), required=True)
    s.add_argument("--robot-sn", default=os.environ.get("ROBOT_SN"))
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8100)
    s.add_argument("--rerun", default=None, help="spawn | serve | save:<path> | off (default: config)")
    s.add_argument("--auto-approve", action="store_true")
    f = sub.add_parser("clear-fault")
    f.add_argument("--url", default="http://127.0.0.1:8100")
    a = ap.parse_args()

    if a.cmd == "add-token":
        cfg = load_config(a.config)
        token = add_token(cfg["server"]["tokens_file"], a.name, a.scenes, a.operator)
        print(f"token for {a.name} (shown once, store it safely):\n{token}")

    elif a.cmd == "serve":
        import uvicorn
        from .approval import AutoApprover, TerminalApprover
        from .core import WorkerClient
        cfg = load_config(a.config)
        if a.auto_approve and a.backend != "sim":
            raise SystemExit("--auto-approve is only allowed with --backend sim")
        approver = AutoApprover() if a.auto_approve else TerminalApprover(cfg)
        viz = a.rerun or cfg["server"].get("rerun", "off")
        robot = WorkerClient(cfg, a.backend, a.robot_sn, viz)
        try:
            uvicorn.run(create_app(cfg, robot, approver), host=a.host, port=a.port, log_level="warning")
        finally:
            robot.close()

    elif a.cmd == "clear-fault":
        import requests
        token = os.environ["R2S_OPERATOR_TOKEN"]
        r = requests.post(a.url.rstrip("/") + "/v1/operator/clear_fault",
                          headers={"Authorization": f"Bearer {token}"}, timeout=30)
        print(r.status_code, r.text)


if __name__ == "__main__":
    main()
