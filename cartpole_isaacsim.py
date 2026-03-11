"""
cartpole_isaacsim.py
====================
MPC closed-loop cart-pole swing-up in Isaac Sim using the official USD asset.

RUN FROM TERMINAL:
    cd ~/isaacsim
    ./python.sh /home/ubuntu/pend/cartpole_isaacsim.py

Make sure acados is on your PYTHONPATH before running:
    export PYTHONPATH=$PYTHONPATH:/home/ubuntu/pend
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ACADOS_SOURCE_DIR/lib
"""

import sys
import os
import numpy as np
import time

# ── acados path ────────────────────────────────────────────────────────────────
sys.path.insert(0, "/home/ubuntu/pend")

# Isaac Sim's python.sh does not inherit LD_LIBRARY_PATH, so acados shared
# libraries must be pre-loaded explicitly before acados_template is imported.
import ctypes
_ACADOS_LIB = "/home/ubuntu/acados/lib"
for _lib in ("libqdldl.so", "libosqp.so", "libblasfeo.so", "libhpipm.so", "libqpOASES_e.so", "libacados.so"):
    ctypes.CDLL(f"{_ACADOS_LIB}/{_lib}", mode=ctypes.RTLD_GLOBAL)

# ── Isaac Sim bootstrap ────────────────────────────────────────────────────────
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})

import omni
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cartpole_ocp import export_cartpole_ocp_solver

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
def _get_cartpole_usd() -> str:
    try:
        from isaacsim.storage.native import get_assets_root_path
        root = get_assets_root_path()
        return root + "/Isaac/Robots/IsaacSim/Cartpole/cartpole.usd"
    except Exception:
        return "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/IsaacSim/Cartpole/cartpole.usd"

CARTPOLE_USD = _get_cartpole_usd()

USE_RTI     = True
ADD_NOISE   = True
ADD_IMPULSE = True
N_horizon   = 50
T_horizon   = 2.0
Fmax        = 80.0
Nsim        = 200
dt_mpc      = T_horizon / N_horizon   # 0.04 s

sigma_p, sigma_theta = 1e-3, 1e-3
sigma_v, sigma_omega = 1e-2, 1e-2

M, m, l, g = 1.0, 0.1, 0.8, 9.81

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def ode_rhs(x, u):
    _, theta, _, omega = x
    D       = M + m * (1 - np.cos(theta)**2)
    f_vx    = (-m*l*np.sin(theta)*omega**2 + m*g*np.cos(theta)*np.sin(theta) + u) / D
    f_omega = (-m*l*np.cos(theta)*np.sin(theta)*omega**2 + u*np.cos(theta)
               + (M+m)*g*np.sin(theta)) / (l*D)
    return np.array([x[2], x[3], f_vx, f_omega])

def rk4(x, u, dt):
    k1 = ode_rhs(x,            u)
    k2 = ode_rhs(x + dt/2*k1,  u)
    k3 = ode_rhs(x + dt/2*k2,  u)
    k4 = ode_rhs(x + dt*k3,    u)
    return x + dt/6*(k1 + 2*k2 + 2*k3 + k4)

def warmstart(solver, x0, N, dt):
    x = x0.copy()
    for k in range(N):
        solver.set(k, "x", x)
        solver.set(k, "u", np.zeros(1))
        x = rk4(x, 0.0, dt)
    solver.set(N, "x", x)

def shift_solution(solver, N):
    for k in range(N - 1):
        solver.set(k, "x", solver.get(k+1, "x"))
        solver.set(k, "u", solver.get(k+1, "u"))
    solver.set(N-1, "x", solver.get(N, "x"))

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print(f"Cart-Pole MPC | RTI={USE_RTI} Noise={ADD_NOISE} Impulse={ADD_IMPULSE}")
    print("=" * 60)

    world = World(
        stage_units_in_meters=1.0,
        physics_dt=dt_mpc,
        rendering_dt=dt_mpc,
    )

    # ── Load USD ──────────────────────────────────────────────────────────────
    print(f"\nLoading: {CARTPOLE_USD}")
    add_reference_to_stage(usd_path=CARTPOLE_USD, prim_path="/World/CartPole")

    cartpole = world.scene.add(
        Articulation(prim_path="/World/CartPole", name="cartpole", position=np.array([0.0, 0.0, 1.0]))
    )
    world.scene.add_default_ground_plane()
    world.reset()

    # ── Inspect DOFs ──────────────────────────────────────────────────────────
    print(f"Num DOFs : {cartpole.num_dof}")
    print(f"DOF names: {cartpole.dof_names}")

    dof_names = list(cartpole.dof_names)

    slider_idx = dof_names.index("slider_to_cart")
    hinge_idx  = dof_names.index("cart_to_pole")

    print(f"Slider DOF: index={slider_idx} name={dof_names[slider_idx]}")
    print(f"Hinge  DOF: index={hinge_idx}  name={dof_names[hinge_idx]}")

    # ── Initial state: pendulum hanging down (theta = pi) ─────────────────────
    init_pos = np.zeros(cartpole.num_dof)
    init_vel = np.zeros(cartpole.num_dof)
    init_pos[hinge_idx] = np.pi

    cartpole.set_joint_positions(init_pos)
    cartpole.set_joint_velocities(init_vel)

    for _ in range(5):
        world.step(render=True)

    x0 = np.array([0.0, np.pi, 0.0, 0.0])

    # ── MPC solver ────────────────────────────────────────────────────────────
    Q = np.diag([1e2, 1e2, 1e-2, 1e-2]) if USE_RTI else np.diag([2e2, 2e2, 1e-1, 1e-1])
    R = 1e-1 * np.eye(1)

    print("\nBuilding OCP solver (~15s first time)...")
    solver = export_cartpole_ocp_solver(
        N=N_horizon, T=T_horizon, Fmax=Fmax, use_RTI=USE_RTI, Q=Q, R=R
    )
    print("Solver ready.\n")

    warmstart(solver, x0, N_horizon, dt_mpc)

    if USE_RTI:
        solver.set(0, "lbx", x0)
        solver.set(0, "ubx", x0)
        solver.options_set("rti_phase", 1)
        solver.solve()

    # ── Logging ───────────────────────────────────────────────────────────────
    simX     = []
    simU     = []
    times    = []
    simX.append(x0.copy())
    xcurrent = x0.copy()

    print(f"Running indefinitely (close window to stop and plot)\n")

    # ── MPC loop ──────────────────────────────────────────────────────────────
    i = 0
    while simulation_app.is_running():

        solver.set(0, "lbx", xcurrent)
        solver.set(0, "ubx", xcurrent)

        t0 = time.perf_counter()
        if USE_RTI:
            solver.options_set("rti_phase", 2)
        status = solver.solve()
        dt = time.perf_counter() - t0
        times.append(dt)

        if status not in [0, 2]:
            print(f"  Step {i:3d}: WARNING status {status}")

        u0 = solver.get(0, "u")
        simU.append(u0.copy())

        # Apply force to slider joint only
        efforts = np.zeros(cartpole.num_dof)
        efforts[slider_idx] = float(np.clip(u0[0], -Fmax, Fmax))
        cartpole.apply_action(ArticulationAction(joint_efforts=efforts))

        world.step(render=True)

        # Read back state
        pos = cartpole.get_joint_positions()
        vel = cartpole.get_joint_velocities()
        xcurrent = np.array([
            pos[slider_idx],
            pos[hinge_idx],
            vel[slider_idx],
            vel[hinge_idx],
        ])

        if ADD_NOISE:
            xcurrent += np.array([
                np.random.normal(0, sigma_p),
                np.random.normal(0, sigma_theta),
                np.random.normal(0, sigma_v),
                np.random.normal(0, sigma_omega),
            ])

        if ADD_IMPULSE and i == 100:
            print(f"  Step {i:3d}: Impulse Δθ = π/10")
            xcurrent[1] += np.pi / 10
            _pos = np.zeros(cartpole.num_dof)
            _vel = np.zeros(cartpole.num_dof)
            _pos[slider_idx] = xcurrent[0]
            _pos[hinge_idx]  = xcurrent[1]
            _vel[slider_idx] = xcurrent[2]
            _vel[hinge_idx]  = xcurrent[3]
            cartpole.set_joint_positions(_pos)
            cartpole.set_joint_velocities(_vel)

        simX.append(xcurrent.copy())

        if USE_RTI:
            shift_solution(solver, N_horizon)
            solver.options_set("rti_phase", 1)
            solver.solve()

        if (i+1) % 50 == 0:
            print(f"  Step {i+1:4d} | {dt*1e3:.2f}ms | "
                  f"θ={xcurrent[1]:.3f} | u={u0[0]:.1f}N")

        i += 1

    # ── Results ───────────────────────────────────────────────────────────────
    simX  = np.array(simX)
    simU  = np.array(simU)
    times = np.array(times)

    print("\n" + "=" * 60)
    print(f"  Mean: {np.mean(times)*1e3:.2f}ms  Max: {np.max(times)*1e3:.2f}ms  dt: {dt_mpc*1e3:.1f}ms")
    print("  ✓ Real-time capable" if np.mean(times) < dt_mpc else "  ✗ Too slow")

    plot_results(simX, simU, times)
    simulation_app.close()


def plot_results(simX, simU, times):
    Nsim = simU.shape[0]
    t  = np.linspace(0, Nsim * dt_mpc, Nsim + 1)
    tu = t[:-1]

    fig, ax = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle(
        f"Cart-Pole MPC in Isaac Sim (RTI={USE_RTI}, Noise={ADD_NOISE}, Impulse={ADD_IMPULSE})",
        fontsize=13,
    )
    ax[0,0].plot(t, simX[:,0], "steelblue");   ax[0,0].set_ylabel("px (m)")
    ax[0,0].set_title("Cart Position");         ax[0,0].grid(True)
    ax[0,1].plot(t, simX[:,1], "darkorange");   ax[0,1].set_ylabel("θ (rad)")
    ax[0,1].axhline(0, color="gray", ls="--", alpha=.5, label="target")
    ax[0,1].axhline(np.pi, color="red", ls=":", alpha=.4, label="start")
    ax[0,1].set_title("Pendulum Angle");        ax[0,1].legend(fontsize=8); ax[0,1].grid(True)
    ax[1,0].plot(t, simX[:,2], "steelblue");   ax[1,0].set_ylabel("vx (m/s)")
    ax[1,0].set_title("Cart Velocity");         ax[1,0].grid(True)
    ax[1,1].plot(t, simX[:,3], "darkorange");   ax[1,1].set_ylabel("ω (rad/s)")
    ax[1,1].set_title("Angular Velocity");      ax[1,1].grid(True)
    ax[2,0].step(tu, simU[:,0], where="post", color="green")
    ax[2,0].axhline( Fmax, color="red", ls=":", alpha=.5, label="±Fmax")
    ax[2,0].axhline(-Fmax, color="red", ls=":", alpha=.5)
    ax[2,0].set_ylabel("F (N)");  ax[2,0].set_xlabel("Time (s)")
    ax[2,0].set_title("Control Force"); ax[2,0].legend(fontsize=8); ax[2,0].grid(True)
    ax[2,1].plot(tu, times*1000, color="purple", alpha=.7)
    ax[2,1].axhline(dt_mpc*1000, color="red", ls="--", label=f"dt={dt_mpc*1e3:.1f}ms")
    ax[2,1].set_ylabel("Solve time (ms)"); ax[2,1].set_xlabel("Time (s)")
    ax[2,1].set_title("OCP Solve Time"); ax[2,1].legend(fontsize=8); ax[2,1].grid(True)

    plt.tight_layout()
    out = "/home/ubuntu/pend/cartpole_isaacsim_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out}")


if __name__ == "__main__":
    main()
