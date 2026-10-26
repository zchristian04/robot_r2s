"""Backend interface shared by the real robot and the MuJoCo stand-in.

The executor talks to the robot only through these calls. Every call is async
because the real backend runs inside aioflexiv's asyncio torque loop; the sim
backend steps physics instead of waiting.

state() returns a dict with:
  t               monotonic seconds of the reading
  q, dq           arm joint positions (rad) and velocities (rad/s), 7 each
  tau_ext         estimated external joint torques (Nm), 7
  wrench          external wrench on the tool from the wrist F/T sensor,
                  [fx, fy, fz, mx, my, mz] in the robot (world) frame
  gripper_width   opening width (m)
  gripper_force   finger force along the actuation direction (N)
  gripper_moving  bool
  fault           None, or a short string when the robot reports a fault
"""


class Backend:
    kind = "base"

    async def start(self):
        raise NotImplementedError

    async def stop(self):
        raise NotImplementedError

    def state(self):
        raise NotImplementedError

    def now(self):
        """Current time on the same clock as state()["t"]."""
        raise NotImplementedError

    def set_gains(self, kp, kd):
        raise NotImplementedError

    def command(self, q_target):
        """Set the joint-impedance target. Takes effect on the next control tick."""
        raise NotImplementedError

    async def wait(self, dt):
        """Let dt seconds of robot time pass (real: sleep; sim: step physics)."""
        raise NotImplementedError

    async def gripper_move(self, width, speed, force):
        raise NotImplementedError

    async def gripper_grasp(self, force):
        raise NotImplementedError

    async def hold(self):
        """Stop moving and hold the current joint positions."""
        raise NotImplementedError

    def camera(self):
        """RGB-D camera object with grab() -> (rgb uint8 HxWx3, depth float32 HxW metres, t)."""
        raise NotImplementedError
