import ctypes
import sys

import numpy as np
from isaacsim.core.api.objects import GroundPlane, VisualCuboid
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path

# ── MPC parameters ────────────────────────────────────────────────────────────
N_HORIZON = 50
T_HORIZON = 2.0
FMAX      = 80.0
DT_MPC    = T_HORIZON / N_HORIZON   # 0.04 s
M, MP, L, G = 1.0, 0.1, 0.8, 9.81

# Steps of zero-force settling after reset before MPC takes over
SETTLE_STEPS = 5


class CartpoleMPCScript:
    def __init__(self):
        self._articulation     = None
        self._ground_plane     = None
        self._target           = None
        self._solver           = None
        self._script_generator = None
        self._slider_idx       = None
        self._hinge_idx        = None

        self.use_rti     = True
        self.add_noise   = True
        self.add_impulse = True

    # ── Asset loading ─────────────────────────────────────────────────────────
    def load_example_assets(self):
        usd_path = get_assets_root_path() + "/Isaac/Robots/IsaacSim/Cartpole/cartpole.usd"
        add_reference_to_stage(usd_path, "/World/CartPole")
        self._articulation = SingleArticulation(
            "/World/CartPole", name="cartpole", position=np.array([0.0, 0.0, 1.0])
        )

        # Red marker cube sitting above the pendulum.
        # Move it along world-Y in the viewport to steer the cart target position.
        self._target = VisualCuboid(
            prim_path="/World/Target",
            name="target",
            position=np.array([0.0, 0.0, 2.5]),
            scale=np.array([0.1, 0.1, 0.1]),
            color=np.array([1.0, 0.2, 0.2]),
        )

        self._ground_plane = GroundPlane("/World/Ground")
        return self._articulation, self._target, self._ground_plane

    # ── Post-load setup ───────────────────────────────────────────────────────
    def setup(self):
        _ACADOS_LIB = "/home/ubuntu/acados/lib"
        for _lib in (
            "libqdldl.so", "libosqp.so", "libblasfeo.so",
            "libhpipm.so", "libqpOASES_e.so", "libacados.so",
        ):
            try:
                ctypes.CDLL(f"{_ACADOS_LIB}/{_lib}", mode=ctypes.RTLD_GLOBAL)
            except OSError as e:
                print(f"[cartpole_mpc] WARNING: {_lib}: {e}")

        sys.path.insert(0, "/home/ubuntu/pend")
        from cartpole_ocp import export_cartpole_ocp_solver

        Q = np.diag([1e2, 1e2, 1e-2, 1e-2]) if self.use_rti else np.diag([2e2, 2e2, 1e-1, 1e-1])
        R = 1e-1 * np.eye(1)

        print("[cartpole_mpc] compiling OCP solver (~15 s first time)…")
        self._solver = export_cartpole_ocp_solver(
            N=N_HORIZON, T=T_HORIZON, Fmax=FMAX, use_RTI=self.use_rti, Q=Q, R=R
        )
        print("[cartpole_mpc] solver ready")

        dof_names = list(self._articulation.dof_names)
        print(f"[cartpole_mpc] DOF names: {dof_names}")
        self._slider_idx = dof_names.index("slider_to_cart")
        self._hinge_idx  = dof_names.index("cart_to_pole")

        set_camera_view(
            eye=[3.0, 3.0, 2.0], target=[0.0, 0.0, 1.0],
            camera_prim_path="/OmniverseKit_Persp",
        )

        self._init_joints()
        self._script_generator = self.my_script()

    # ── Reset ─────────────────────────────────────────────────────────────────
    def reset(self):
        self._init_joints()
        self._script_generator = self.my_script()

    def _init_joints(self):
        if self._articulation is None or self._hinge_idx is None:
            return
        n = self._articulation.num_dof
        pos = np.zeros(n)
        pos[self._hinge_idx] = np.pi
        self._articulation.set_joint_positions(pos)
        self._articulation.set_joint_velocities(np.zeros(n))
        # Zero out any residual efforts that could survive the world reset
        self._articulation.apply_action(ArticulationAction(joint_efforts=np.zeros(n)))

    # ── Called every physics step ─────────────────────────────────────────────
    def update(self, step: float):
        try:
            next(self._script_generator)
        except StopIteration:
            return True

    # ── MPC generator ─────────────────────────────────────────────────────────
    def my_script(self):
        n = self._articulation.num_dof

        # ── Settle: apply zero force for a few steps so physics stabilises
        #    after reset (handles bad state from hitting slider limits)
        for _ in range(SETTLE_STEPS):
            self._articulation.apply_action(ArticulationAction(joint_efforts=np.zeros(n)))
            yield

        # ── Read actual state (don't assume — physics may not be at exactly pi)
        pos = self._articulation.get_joint_positions()
        vel = self._articulation.get_joint_velocities()
        xcurrent = np.array([
            pos[self._slider_idx], pos[self._hinge_idx],
            vel[self._slider_idx], vel[self._hinge_idx],
        ])

        # ── Warm-start solver from actual state
        x = xcurrent.copy()
        for k in range(N_HORIZON):
            self._solver.set(k, "x", x)
            self._solver.set(k, "u", np.zeros(1))
            x = _rk4(x, 0.0, DT_MPC)
        self._solver.set(N_HORIZON, "x", x)

        if self.use_rti:
            self._solver.set(0, "lbx", xcurrent)
            self._solver.set(0, "ubx", xcurrent)
            self._solver.options_set("rti_phase", 1)
            self._solver.solve()

        i = 0
        while True:
            # ── Read target position from prim Y axis ──────────────────────
            y_target = 0.0
            if self._target is not None:
                try:
                    tgt_pos, _ = self._target.get_world_pose()
                    y_target = float(tgt_pos[1])
                except Exception:
                    pass

            yref   = np.array([y_target, 0.0, 0.0, 0.0, 0.0])
            yref_e = np.array([y_target, 0.0, 0.0, 0.0])
            for k in range(N_HORIZON):
                self._solver.set(k, "yref", yref)
            self._solver.set(N_HORIZON, "yref", yref_e)

            # ── Solve ──────────────────────────────────────────────────────
            self._solver.set(0, "lbx", xcurrent)
            self._solver.set(0, "ubx", xcurrent)
            if self.use_rti:
                self._solver.options_set("rti_phase", 2)
            status = self._solver.solve()
            if status not in [0, 2]:
                print(f"[cartpole_mpc] step {i}: solver status {status}")

            u0 = self._solver.get(0, "u")

            # ── Apply control ───────────────────────────────────────────────
            efforts = np.zeros(n)
            efforts[self._slider_idx] = float(np.clip(u0[0], -FMAX, FMAX))
            self._articulation.apply_action(ArticulationAction(joint_efforts=efforts))

            yield  # physics steps here

            # ── Read state ─────────────────────────────────────────────────
            pos = self._articulation.get_joint_positions()
            vel = self._articulation.get_joint_velocities()
            xcurrent = np.array([
                pos[self._slider_idx], pos[self._hinge_idx],
                vel[self._slider_idx], vel[self._hinge_idx],
            ])

            if self.add_noise:
                xcurrent += np.random.normal(0, [1e-3, 1e-3, 1e-2, 1e-2])

            if self.add_impulse and i == 100:
                xcurrent[1] += np.pi / 10
                _p, _v = np.zeros(n), np.zeros(n)
                _p[self._slider_idx] = xcurrent[0]
                _p[self._hinge_idx]  = xcurrent[1]
                _v[self._slider_idx] = xcurrent[2]
                _v[self._hinge_idx]  = xcurrent[3]
                self._articulation.set_joint_positions(_p)
                self._articulation.set_joint_velocities(_v)
                print("[cartpole_mpc] impulse applied at step 100")

            # ── RTI shift ──────────────────────────────────────────────────
            if self.use_rti:
                for k in range(N_HORIZON - 1):
                    self._solver.set(k, "x", self._solver.get(k + 1, "x"))
                    self._solver.set(k, "u", self._solver.get(k + 1, "u"))
                self._solver.set(N_HORIZON - 1, "x", self._solver.get(N_HORIZON, "x"))
                self._solver.options_set("rti_phase", 1)
                self._solver.solve()

            i += 1


# ── RK4 integrator (warm-start only) ─────────────────────────────────────────
def _rk4(x, u, dt):
    def f(x, u):
        _, theta, _, omega = x
        D  = M + MP * (1 - np.cos(theta) ** 2)
        ax = (-MP*L*np.sin(theta)*omega**2 + MP*G*np.cos(theta)*np.sin(theta) + u) / D
        al = (-MP*L*np.cos(theta)*np.sin(theta)*omega**2 + u*np.cos(theta)
              + (M + MP)*G*np.sin(theta)) / (L * D)
        return np.array([x[2], x[3], ax, al])

    k1 = f(x, u)
    k2 = f(x + dt / 2 * k1, u)
    k3 = f(x + dt / 2 * k2, u)
    k4 = f(x + dt * k3,     u)
    return x + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
