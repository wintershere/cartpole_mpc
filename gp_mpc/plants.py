"""Plants the MPC controls. Each exposes reset(x0), step(u) and state().

MujocoPlant builds the cart-pole MJCF with the mismatch parameters, so the
"real" dynamics come from an independent physics engine, not from the
model equations. IsaacPlant lives in isaac_plant.py (needs Isaac Sim).
"""

import mujoco
import numpy as np

from dynamics import rk4_np

MJCF = """
<mujoco model="cartpole_true">
  <option gravity="{gx} 0 {gz}" timestep="{ts}">
    <flag contact="disable"/>
  </option>
  <default>
    <geom contype="0" conaffinity="0"/>
  </default>
  <worldbody>
    <body name="cart" pos="0 0 1.0">
      <joint name="slider" type="slide" axis="1 0 0"
             damping="{b_c}" frictionloss="{f_c}" armature="0"/>
      <geom type="box" size="0.2 0.15 0.08" mass="{M}"/>
      <body name="pole" pos="0 0 0.08">
        <joint name="hinge" type="hinge" axis="0 -1 0"
               damping="{b_p}" armature="0"/>
        <geom type="capsule" fromto="0 0 0 0 0 {l}" size="0.025" density="0"/>
        <body name="tip" pos="0 0 {l}">
          <geom type="sphere" size="0.05" mass="{m}"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="force" joint="slider" gear="{gear}" ctrllimited="false"/>
  </actuator>
</mujoco>
"""


class MujocoPlant:
    """Extra hidden effects (all default off):
    delay_steps: dead time, the force acts delay_steps control steps late
    motor_tau:   first-order motor lag (s) on the force, per physics step
    tilt:        rail tilt (rad); gravity is tilted, so the cart is pulled
                 towards -px and the pole's upright direction moves too
    """
    name = "mujoco"

    def __init__(self, params, dt, timestep=0.002):
        p = dict(f_c=0.0, b_c=0.0, b_p=0.0, gear=1.0, delay_steps=0, motor_tau=0.0, tilt=0.0)
        p.update(params)
        g, a = p["g"], p["tilt"]
        xml = MJCF.format(ts=timestep, gx=-g * np.sin(a), gz=-g * np.cos(a), **p)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.n_sub = max(1, round(dt / timestep))
        self.dt, self.ts = dt, timestep
        self.delay, self.tau = int(p["delay_steps"]), float(p["motor_tau"])

    def reset(self, x0):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = x0[:2]
        self.data.qvel[:] = x0[2:]
        mujoco.mj_forward(self.model, self.data)
        self.queue = [0.0] * self.delay
        self.force = 0.0

    def step(self, u):
        self.queue.append(float(u))
        u_del = self.queue.pop(0)                # dead time
        for _ in range(self.n_sub):
            if self.tau > 0:                     # motor lag
                self.force += (u_del - self.force) * self.ts / (self.tau + self.ts)
            else:
                self.force = u_del
            self.data.ctrl[0] = self.force
            mujoco.mj_step(self.model, self.data)
        return self.state()

    def state(self):
        return np.array([self.data.qpos[0], self.data.qpos[1],
                         self.data.qvel[0], self.data.qvel[1]])


class IsaacPlant:
    """Client for isaac_server.py (Isaac Sim runs in its own process)."""
    name = "isaac"

    def __init__(self, host="127.0.0.1", port=50555):
        import json
        import socket
        self._json = json
        self.sock = socket.create_connection((host, port), timeout=120)
        self.f = self.sock.makefile("rw")
        self.info = self._ask({"cmd": "info"})
        self.x = np.zeros(4)

    def _ask(self, req):
        self.f.write(self._json.dumps(req) + "\n")
        self.f.flush()
        return self._json.loads(self.f.readline())

    def reset(self, x0):
        self.x = np.array(self._ask({"cmd": "reset", "x": [float(v) for v in x0]})["x"])

    def step(self, u):
        self.x = np.array(self._ask({"cmd": "step", "u": float(u)})["x"])
        return self.x.copy()

    def state(self):
        return self.x.copy()

    def record(self, path):
        """Start (path) or stop (None) saving one rendered frame per step
        on the server (needs isaac_server.py --render)."""
        return self._ask({"cmd": "record", "dir": path})

    def target(self, px):
        """Move the rendered setpoint marker (visual only)."""
        return self._ask({"cmd": "target", "px": float(px)})

    def close(self, stop_server=False):
        if stop_server:
            self._ask({"cmd": "quit"})
        self.sock.close()


class OdePlant:
    """Model equations integrated finely. For checks: with NOMINAL params
    the nominal MPC should be near perfect on it."""
    name = "ode"

    def __init__(self, params, dt):
        self.p, self.dt, self.x = params, dt, np.zeros(4)

    def reset(self, x0):
        self.x = np.array(x0, float)

    def step(self, u):
        self.x = rk4_np(self.x, u, self.p, self.dt, substeps=20)
        return self.x.copy()

    def state(self):
        return self.x.copy()
