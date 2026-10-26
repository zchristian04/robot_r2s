# robot_r2s

Robot server for the Agentic Real2Sim real-robot runs: a Flexiv Rizon 4s with a
Grav (GN01) gripper and a ZED 2i camera. It serves the same HTTP API as the
simulated source world (`r2s_exam`), so an agent's code works on both. Every
motion is checked by a safety layer and approved by an operator, who always
holds the emergency stop.

**Status: written but not yet run. The sim tests and the Rerun view still need
to be run (see [Testing](#testing-to-do)), and the real backend has not run on
hardware (see [Before the first hardware session](#before-the-first-hardware-session)).**

## Layout

| File | Purpose |
| --- | --- |
| `robot_r2s/model.py` | Kinematics on the lab MuJoCo model: FK, IK (tool pointing down + yaw), table-collision check |
| `robot_r2s/safety.py` | Workspace, speed, joint-margin and wrench limits; latched faults |
| `robot_r2s/primitives.py` | `move_to`, `home`, `push`, `touch`, `grasp_lift`: validation and planning (same as the sim server) |
| `robot_r2s/executor.py` | Plan -> joint targets at the control rate, pre-checked; streams them and watches force, tracking and faults |
| `robot_r2s/backends/real.py` | aioflexiv joint-impedance arm, Flexiv RDK gripper, ZED capture |
| `robot_r2s/backends/sim.py` | MuJoCo stand-in using the same lab model |
| `robot_r2s/approval.py` | Operator approval in the server terminal |
| `robot_r2s/viz.py` | Rerun view: measured robot, target ghost, planned path, wrench, status |
| `robot_r2s/core.py` | Robot core and the worker process that owns the robot |
| `robot_r2s/server.py` | HTTP API, tokens, request log, budgets, operator endpoints |
| `robot_r2s/cli.py` | `add-token`, `serve`, `clear-fault` |
| `configs/lab.json` | Cell configuration; values marked VERIFY must be checked |

The robot model is the lab package `newton_handoff_20260921/scene_robot_only.xml`
(Rizon 4s + GN01 + table + calibrated ZED), set in `configs/lab.json`.

## How a request runs

1. The server validates the primitive and its parameters.
2. The worker plans it and converts the whole motion to joint targets by IK,
   checking workspace, reachability, joint margins, speeds and table collisions.
   The planned path and the target ghost appear in Rerun.
3. The operator approves or rejects it in the server terminal.
4. The worker streams the joint targets to the arm and, every tick, checks the
   robot fault state, the wrist wrench, the tracking error and state freshness.
   A touch ends at its stop force. Anything abnormal holds the arm and latches a
   fault; no further motion is accepted until the operator runs `clear-fault`.

## Running

```bash
pip install -e ".[viz,test]"              # add [real] on the lab PC
python -m robot_r2s.cli add-token --name operator --scenes scene_01 --operator
python -m robot_r2s.cli add-token --name agent_run1 --scenes scene_01
python -m robot_r2s.cli serve --config configs/lab.json --backend sim
python -m robot_r2s.cli serve --config configs/lab.json --backend real --robot-sn <SN>
```

The serial number is never stored in the config. Bind to `127.0.0.1` and reach
the server through an SSH tunnel, unless the network is otherwise controlled.

## Testing (to do)

Nothing has been run yet. Next steps, all without hardware:

1. **Create the environment.**
   ```bash
   cd robot_r2s
   uv venv --python 3.12 .venv
   uv pip install --python .venv/bin/python -e ".[viz,test]"
   ```
2. **Run the sim tests:** `.venv/bin/python -m pytest -q tests/test_sim.py`.
   They cover the IK round trip, rejection of bad and out-of-workspace requests,
   a `move_to`, a `touch` that stops on the table, and server auth, approval and
   budget, all on the MuJoCo stand-in.
3. **Check the Rerun view.** Run the sim server with Rerun recording to a file,
   send a few actions, then open the recording:
   ```bash
   .venv/bin/python -m robot_r2s.cli serve --config configs/lab.json --backend sim --auto-approve --rerun save:/tmp/robot_r2s_test.rrd
   # in another shell: add a token, reset scene_01, run move_to / touch / grasp_lift
   rerun /tmp/robot_r2s_test.rrd
   ```
   Check that the robot meshes appear at the measured pose, the green ghost
   leads them to the target, the planned path and wrench arrow show, and the
   status log reads "waiting for approval" before each action. On a machine
   with a display, `--rerun spawn` shows the same thing live.

## Before the first hardware session

Check on the lab machine, in this order:

1. `aioflexiv status` connects; the robot has a wrist F/T sensor
   (`robot.info["has_FT_sensor"]`).
2. The GN01 can be driven through `flexivrdk.Gripper` **while aioflexiv's RT
   torque loop is running** (a second robot connection). If not, the gripper
   needs another route.
3. The key and frame of the wrench in aioflexiv's full state
   (`ext_wrench_in_world` is assumed).
4. ZED SDK installed; depth units in metres.
5. Every value marked VERIFY in `configs/lab.json`: workspace, speeds, joint
   gains, wrench limits, home posture, table height, travel height.
6. Bring-up: state only (motion disabled) -> gripper alone -> slow free moves,
   one at a time -> contact primitives at low force on a fixed object -> an
   end-to-end dry run with a script before any agent session.

## Licence notes

Parts are adapted from `flexiv_control` (Apache-2.0); `aioflexiv` (Apache-2.0)
is a dependency. See `NOTICE`.
