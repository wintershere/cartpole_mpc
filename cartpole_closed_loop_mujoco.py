"""
cartpole_closed_loop_mujoco.py
==============================
MPC closed-loop with MuJoCo plant and full trajectory replay.

The simulation runs first (no viewer), recording every state.
Then the viewer opens and replays the entire trajectory from the start
at real-time speed so you can see the full swing-up.

Press Ctrl+C in the terminal to skip to plots.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
from cartpole_ocp import export_cartpole_ocp_solver

# ====================================================================
# CONFIGURATION
# ====================================================================
USE_RTI      = True
ADD_NOISE    = True
ADD_IMPULSE  = True
N_horizon    = 50
T_horizon    = 2.0
Fmax         = 80.0
Nsim         = 200
dt_mpc       = T_horizon / N_horizon
XML_PATH     = "cartpole.xml"
sigma_p, sigma_theta, sigma_v, sigma_omega = 1e-3, 1e-3, 1e-2, 1e-2
M, m, l, g  = 1.0, 0.1, 0.8, 9.81

# Replay speed: 1.0 = real-time, 0.5 = half speed (easier to watch)
REPLAY_SPEED = 1.0

# ====================================================================
# HELPERS
# ====================================================================

def get_state(data):
    return np.array([data.qpos[0], data.qpos[1], data.qvel[0], data.qvel[1]])

def set_state(data, x):
    """Set MuJoCo state from [px, theta, vx, omega]."""
    data.qpos[0], data.qpos[1] = x[0], x[1]
    data.qvel[0], data.qvel[1] = x[2], x[3]

def mujoco_step(mj_model, mj_data, u, dt):
    n = max(1, round(dt / mj_model.opt.timestep))
    mj_data.ctrl[0] = u
    for _ in range(n):
        mujoco.mj_step(mj_model, mj_data)

def shift_solution(solver, N):
    for k in range(N - 1):
        solver.set(k, "x", solver.get(k + 1, "x"))
        solver.set(k, "u", solver.get(k + 1, "u"))
    solver.set(N - 1, "x", solver.get(N, "x"))

def ode_rhs(x, u):
    _, theta, _, omega = x
    D       = M + m * (1 - np.cos(theta)**2)
    f_vx    = (-m*l*np.sin(theta)*omega**2 + m*g*np.cos(theta)*np.sin(theta) + u) / D
    f_omega = (-m*l*np.cos(theta)*np.sin(theta)*omega**2 + u*np.cos(theta)
               + (M+m)*g*np.sin(theta)) / (l*D)
    return np.array([x[2], x[3], f_vx, f_omega])

def rk4(x, u, dt):
    k1 = ode_rhs(x,           u)
    k2 = ode_rhs(x + dt/2*k1, u)
    k3 = ode_rhs(x + dt/2*k2, u)
    k4 = ode_rhs(x + dt*k3,   u)
    return x + dt/6*(k1 + 2*k2 + 2*k3 + k4)

def warmstart(solver, x0, N, dt):
    x = x0.copy()
    for k in range(N):
        solver.set(k, "x", x)
        solver.set(k, "u", np.zeros(1))
        x = rk4(x, 0.0, dt)
    solver.set(N, "x", x)
    print(f"  Rollout: x[0]={x0.round(3)}  x[N]={x.round(3)}")


# ====================================================================
# PHASE 1: Run MPC simulation (no viewer — just compute)
# ====================================================================

def run_simulation(mj_model, mj_data):
    """Run the full MPC loop, return recorded trajectories."""

    Q = np.diag([1e2, 1e2, 1e-2, 1e-2]) if USE_RTI else np.diag([2e2, 2e2, 1e-1, 1e-1])
    R = 1e-1 * np.eye(1)

    print("Building OCP solver...")
    solver = export_cartpole_ocp_solver(
        N=N_horizon, T=T_horizon, Fmax=Fmax, use_RTI=USE_RTI, Q=Q, R=R
    )
    print("OCP solver ready.\n")

    nx, nu = 4, 1
    simX   = np.zeros((Nsim + 1, nx))
    simU   = np.zeros((Nsim, nu))
    times  = np.zeros(Nsim)

    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    mujoco.mj_forward(mj_model, mj_data)
    xcurrent   = get_state(mj_data)
    simX[0, :] = xcurrent

    print("Warm-starting with zero-control rollout...")
    warmstart(solver, xcurrent, N_horizon, dt_mpc)

    if USE_RTI:
        solver.set(0, "lbx", xcurrent)
        solver.set(0, "ubx", xcurrent)
        solver.options_set("rti_phase", 1)
        solver.solve()

    print(f"Simulating {Nsim} steps × {dt_mpc:.3f}s = {Nsim*dt_mpc:.1f}s ...")

    for i in range(Nsim):
        solver.set(0, "lbx", xcurrent)
        solver.set(0, "ubx", xcurrent)

        t0 = time.perf_counter()
        if USE_RTI:
            solver.options_set("rti_phase", 2)
        status = solver.solve()
        times[i] = time.perf_counter() - t0

        if status not in [0, 2]:
            print(f"  Step {i:3d}: WARNING status {status}")

        u0 = solver.get(0, "u")
        simU[i, :] = u0

        mujoco_step(mj_model, mj_data, u0[0], dt_mpc)
        xcurrent = get_state(mj_data)

        if ADD_NOISE:
            xcurrent += np.array([np.random.normal(0, s) for s in
                                   [sigma_p, sigma_theta, sigma_v, sigma_omega]])
            set_state(mj_data, xcurrent)

        if ADD_IMPULSE and i == Nsim // 2:
            print(f"  Step {i:3d}: Impulse Δθ=π/10")
            xcurrent[1] += np.pi / 10
            set_state(mj_data, xcurrent)

        simX[i + 1, :] = xcurrent

        if USE_RTI:
            shift_solution(solver, N_horizon)
            solver.options_set("rti_phase", 1)
            solver.solve()

        if (i + 1) % 50 == 0:
            print(f"  Step {i+1:3d}/{Nsim} | {times[i]*1000:.2f}ms | "
                  f"θ={xcurrent[1]:.3f} | u={u0[0]:.1f}N")

    print(f"\nSimulation complete.")
    print(f"  Mean solve: {np.mean(times)*1e3:.2f}ms  "
          f"Max: {np.max(times)*1e3:.2f}ms  dt: {dt_mpc*1e3:.1f}ms")
    print("  ✓ Real-time capable" if np.mean(times) < dt_mpc else "  ✗ Too slow")

    return simX, simU, times


# ====================================================================
# PHASE 2: Replay recorded trajectory in MuJoCo viewer
# ====================================================================

def replay_trajectory(mj_model, mj_data, simX, simU):
    """Open the viewer and play back the recorded trajectory at real-time."""

    print(f"\n{'=' * 60}")
    print(f"REPLAY  ({REPLAY_SPEED:.1f}x speed)")
    print(f"  Showing full trajectory from t=0 to t={Nsim*dt_mpc:.1f}s")
    print(f"  Close the viewer window when done.")
    print(f"{'=' * 60}\n")

    # Reset to initial state so viewer starts with pendulum hanging down
    set_state(mj_data, simX[0])
    mujoco.mj_forward(mj_model, mj_data)

    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

    # Camera: side view so you see the full x-z swing plane
    viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 135, -20, 4.0
    viewer.cam.lookat[:] = [0.0, 0.0, 1.5]  # center on the cart rail

    # Give the viewer a moment to initialize
    time.sleep(0.5)
    viewer.sync()
    time.sleep(0.5)

    print("Playing back trajectory... (close viewer to continue to plots)")

    try:
        for i in range(Nsim):
            if not viewer.is_running():
                break

            t_frame_start = time.perf_counter()

            # Set the recorded state into MuJoCo
            set_state(mj_data, simX[i])
            if i < len(simU):
                mj_data.ctrl[0] = simU[i, 0]
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()

            # Wait to maintain real-time (or slow-motion) playback
            elapsed = time.perf_counter() - t_frame_start
            wait = (dt_mpc / REPLAY_SPEED) - elapsed
            if wait > 0:
                time.sleep(wait)

            # Print progress
            if (i + 1) % 50 == 0 or i == 0:
                t_sim = i * dt_mpc
                print(f"  t={t_sim:5.2f}s  θ={simX[i,1]:+.3f}  "
                      f"px={simX[i,0]:+.3f}  F={simU[i,0]:+.1f}N")

        # Hold the final state
        set_state(mj_data, simX[-1])
        mujoco.mj_forward(mj_model, mj_data)
        viewer.sync()

        print(f"\n  Replay finished. Final state: θ={simX[-1,1]:.4f} rad")
        print("  Close the viewer window to continue to plots.")

        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n  Replay interrupted.")

    viewer.close()


# ====================================================================
# PLOTTING
# ====================================================================

def plot_results(simX, simU, times):
    Nsim = simU.shape[0]
    t    = np.linspace(0, Nsim * dt_mpc, Nsim + 1)
    tu   = t[:-1]
    fig, ax = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle(
        f"Cart-Pole MPC+MuJoCo (RTI={USE_RTI}, Noise={ADD_NOISE}, Impulse={ADD_IMPULSE})",
        fontsize=13,
    )

    ax[0,0].plot(t, simX[:,0], "steelblue")
    ax[0,0].axhline(0, color="gray", ls="--", alpha=.5)
    ax[0,0].set_ylabel("px (m)"); ax[0,0].set_title("Cart Position"); ax[0,0].grid(True)

    ax[0,1].plot(t, simX[:,1], "darkorange")
    ax[0,1].axhline(0, color="gray", ls="--", alpha=.5, label="target")
    ax[0,1].axhline(np.pi, color="red", ls=":", alpha=.4, label="start")
    ax[0,1].set_ylabel("θ (rad)"); ax[0,1].set_title("Pendulum Angle (0=upright)")
    ax[0,1].legend(fontsize=8); ax[0,1].grid(True)

    ax[1,0].plot(t, simX[:,2], "steelblue")
    ax[1,0].axhline(0, color="gray", ls="--", alpha=.5)
    ax[1,0].set_ylabel("vx (m/s)"); ax[1,0].set_title("Cart Velocity"); ax[1,0].grid(True)

    ax[1,1].plot(t, simX[:,3], "darkorange")
    ax[1,1].axhline(0, color="gray", ls="--", alpha=.5)
    ax[1,1].set_ylabel("ω (rad/s)"); ax[1,1].set_title("Angular Velocity"); ax[1,1].grid(True)

    ax[2,0].step(tu, simU[:,0], where="post", color="green")
    ax[2,0].axhline( Fmax, color="red", ls=":", alpha=.5, label="±Fmax")
    ax[2,0].axhline(-Fmax, color="red", ls=":", alpha=.5)
    ax[2,0].set_ylabel("F (N)"); ax[2,0].set_xlabel("Time (s)")
    ax[2,0].set_title("Control Force"); ax[2,0].legend(fontsize=8); ax[2,0].grid(True)

    ax[2,1].plot(tu, times*1000, color="purple", alpha=.7)
    ax[2,1].axhline(dt_mpc*1000, color="red", ls="--", label=f"dt={dt_mpc*1e3:.1f}ms")
    ax[2,1].set_ylabel("Solve time (ms)"); ax[2,1].set_xlabel("Time (s)")
    ax[2,1].set_title("OCP Solve Time"); ax[2,1].legend(fontsize=8); ax[2,1].grid(True)

    plt.tight_layout()
    plt.savefig("cartpole_mujoco_results.png", dpi=150, bbox_inches="tight")
    print("Saved → cartpole_mujoco_results.png")
    plt.show()


# ====================================================================
# ENTRY POINT
# ====================================================================

def run_closed_loop():
    print("=" * 60)
    print(f"Cart-pole MPC+MuJoCo | RTI={USE_RTI} Noise={ADD_NOISE} Impulse={ADD_IMPULSE}")
    print("=" * 60)

    mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
    mj_data  = mujoco.MjData(mj_model)
    print("MuJoCo loaded.\n")

    # Phase 1: Run the MPC simulation (fast, no viewer)
    print("─" * 60)
    print("PHASE 1: Running MPC simulation...")
    print("─" * 60)
    simX, simU, times = run_simulation(mj_model, mj_data)

    # Phase 2: Replay in viewer (real-time, from the beginning)
    replay_trajectory(mj_model, mj_data, simX, simU)

    # Phase 3: Show plots
    plot_results(simX, simU, times)

    return simX, simU, times


if __name__ == "__main__":
    run_closed_loop()
