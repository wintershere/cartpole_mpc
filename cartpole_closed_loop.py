"""
cartpole_closed_loop.py
=======================
Runs the MPC closed-loop simulation for the cart-pole swing-up.

This implements Exercises 3 and 4 from the assignment:
  - Exercise 3a: Basic MPC loop, measure solve time
  - Exercise 3b: Add model-plant mismatch (noise + impulse disturbance)
  - Exercise 4:  Real-Time Iteration (RTI) scheme, compare timing

How the MPC loop works:
  1. Read current state x0 from the "plant" (simulated system)
  2. Give x0 to acados as the initial state constraint
  3. Call solver.solve() → get optimal control sequence
  4. Extract first control u0 from the solution
  5. Apply u0 to the plant for one step → get new state
  6. Repeat from step 1

The "plant" here is a separate acados integrator (AcadosSimSolver),
which simulates the real physical system. In the real world, this
would be the actual robot.

FIXES vs original:
  1. RTI warm-starting: after each step, the solution trajectory is
     shifted forward by one node so the preparation phase linearises
     around a trajectory that is close to the next optimal solution.
     Without this, RTI re-linearises around a stale guess every step
     and the controller diverges (cart drifts, pendulum spins).
  2. Softer Q weights for RTI: the original Q=diag(2e3,...) is too
     aggressive for a single SQP step. Reduced to diag(1e2,...) so
     the linearised subproblem stays well-conditioned.
  3. Initial preparation seeded with x0: the first prep phase now
     has the correct initial state constraint set before solving,
     giving a meaningful linearisation point from the start.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from acados_template import AcadosSimSolver, AcadosSim

from cartpole_model import export_cartpole_model
from cartpole_ocp import export_cartpole_ocp_solver


# ====================================================================
# CONFIGURATION — Edit these to run different experiments
# ====================================================================

USE_RTI        = True    # Exercise 4: set True to use RTI scheme
ADD_NOISE      = True     # Exercise 3b: add Gaussian noise to plant
ADD_IMPULSE    = True     # Exercise 3b: kick pendulum at mid-simulation

N_horizon      = 50       # MPC prediction horizon steps
T_horizon      = 2.0      # MPC prediction horizon (seconds)
Fmax           = 80.0     # Max control force (N)

Nsim           = 200      # Total simulation steps
dt             = T_horizon / N_horizon   # sampling time = 0.04 s

# Noise parameters (Exercise 3b)
sigma_p     = 1e-3   # std of position noise
sigma_theta = 1e-3   # std of angle noise
sigma_v     = 1e-2   # std of velocity noise
sigma_omega = 1e-2   # std of angular velocity noise

# ====================================================================
# HELPER: Build a sim solver (integrates the "real" plant)
# ====================================================================

def build_sim_solver(dt: float) -> AcadosSimSolver:
    """
    Creates an AcadosSimSolver to simulate the plant.
    
    This is separate from the OCP solver — it plays the role of the
    physical system. In the real world you wouldn't need this; the
    robot hardware takes the place of this integrator.
    """
    model = export_cartpole_model()

    sim = AcadosSim()
    sim.model = model
    sim.solver_options.T               = dt       # one integration step = one MPC step
    sim.solver_options.integrator_type = "IRK"    # same integrator as OCP for consistency
    sim.solver_options.num_stages      = 4        # 4-stage IRK (accurate)
    sim.solver_options.num_steps       = 1        # 1 step per dt

    sim_solver = AcadosSimSolver(sim, json_file="cartpole_sim.json")
    return sim_solver


# ====================================================================
# HELPER: Shift solution trajectory forward by one step (warm-start)
# ====================================================================

def shift_solution(ocp_solver, N: int):
    """
    Shift the stored solution trajectory forward by one shooting node.

    WHY THIS IS CRITICAL FOR RTI:
    RTI only performs one SQP iteration per MPC step. The quality of
    that single iteration depends entirely on the linearisation point
    (the current solution guess). If we don't shift, the guess stays
    at the old time grid — off by one dt from the true next solution.
    After a few steps the linearisation point is completely wrong and
    the controller diverges.

    By shifting we ensure node k of the guess corresponds to the same
    physical time as node k of the next OCP, so the linearisation is
    as accurate as possible.
    """
    for k in range(N - 1):
        x_next = ocp_solver.get(k + 1, "x")
        u_next = ocp_solver.get(k + 1, "u")
        ocp_solver.set(k, "x", x_next)
        ocp_solver.set(k, "u", u_next)

    # Hold the last node steady (best guess for the new terminal node)
    x_last = ocp_solver.get(N, "x")
    ocp_solver.set(N - 1, "x", x_last)
    # Terminal node x[N] stays unchanged (it's a state node, no control)


# ====================================================================
# MAIN SIMULATION LOOP
# ====================================================================

def run_closed_loop():
    print("=" * 60)
    print(f"Cart-pole MPC  |  RTI={USE_RTI}  Noise={ADD_NOISE}  Impulse={ADD_IMPULSE}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Cost matrices
    # For full SQP: aggressive weights are fine (iterates to convergence)
    # For RTI: softer weights keep the single SQP step well-conditioned
    # ------------------------------------------------------------------
    if USE_RTI:
        Q = np.diag([1e2, 1e2, 1e-2, 1e-2])   # FIX: reduced from 2e3 → 1e2
        R = 1e-1 * np.eye(1)
    else:
        Q = np.diag([2e3, 2e3, 1e-2, 1e-2])   # original aggressive weights
        R = 1e-1 * np.eye(1)

    # ------------------------------------------------------------------
    # Build solvers
    # ------------------------------------------------------------------
    print("\nBuilding OCP solver (this compiles C code, ~15s first time)...")
    ocp_solver = export_cartpole_ocp_solver(
        N=N_horizon, T=T_horizon, Fmax=Fmax, use_RTI=USE_RTI, Q=Q, R=R
    )

    print("Building plant simulator...")
    sim_solver = build_sim_solver(dt)

    # ------------------------------------------------------------------
    # Initial state: pendulum hanging down
    # x = [px=0, theta=pi, vx=0, omega=0]
    # Goal: swing up to theta=0 (upright)
    # ------------------------------------------------------------------
    x0 = np.array([0.0, np.pi, 0.0, 0.0])

    # Storage for logging
    nx = 4
    nu = 1
    simX = np.zeros((Nsim + 1, nx))   # state trajectory
    simU = np.zeros((Nsim, nu))        # control trajectory
    solve_times = np.zeros(Nsim)       # feedback-phase solve time per step

    simX[0, :] = x0
    xcurrent = x0.copy()

    print(f"\nStarting simulation: {Nsim} steps × {dt:.3f}s = {Nsim*dt:.1f}s total\n")

    # ------------------------------------------------------------------
    # If using RTI: seed the initial preparation phase with x0 set.
    # FIX: original code ran prep without setting x0 first, giving a
    # poor initial linearisation point.
    # ------------------------------------------------------------------
    if USE_RTI:
        ocp_solver.set(0, "lbx", x0)
        ocp_solver.set(0, "ubx", x0)
        ocp_solver.options_set("rti_phase", 1)  # preparation phase
        ocp_solver.solve()

    # ------------------------------------------------------------------
    # Main MPC loop
    # ------------------------------------------------------------------
    for i in range(Nsim):

        # ---- Step 1: Set current state as initial constraint ----
        ocp_solver.set(0, "lbx", xcurrent)
        ocp_solver.set(0, "ubx", xcurrent)

        # ---- Step 2: Solve OCP ----
        t_start = time.perf_counter()

        if USE_RTI:
            # Feedback phase only — fast, uses prep work from previous step
            ocp_solver.options_set("rti_phase", 2)
            status = ocp_solver.solve()
        else:
            # Full SQP: iterate until convergence
            status = ocp_solver.solve()

        t_solve = time.perf_counter() - t_start
        solve_times[i] = t_solve

        # Warn if solver didn't converge (status != 0)
        if status not in [0, 2]:
            print(f"  Step {i:3d}: WARNING — solver status {status}")

        # ---- Step 3: Extract first control input ----
        u0 = ocp_solver.get(0, "u")
        simU[i, :] = u0

        # ---- Step 4: Apply u0 to the plant (simulate one step) ----
        sim_solver.set("x", xcurrent)
        sim_solver.set("u", u0)
        sim_solver.solve()
        xcurrent = sim_solver.get("x")

        # ---- Step 5 (Exercise 3b): Add noise to simulate model-plant mismatch ----
        if ADD_NOISE:
            noise = np.array([
                np.random.normal(0, sigma_p),
                np.random.normal(0, sigma_theta),
                np.random.normal(0, sigma_v),
                np.random.normal(0, sigma_omega),
            ])
            xcurrent = xcurrent + noise

        # ---- Step 6 (Exercise 3b): Impulse disturbance at halfway point ----
        if ADD_IMPULSE and i == Nsim // 2:
            print(f"  Step {i:3d}: Applying impulse disturbance (Δθ = π/10)")
            xcurrent[1] += np.pi / 10

        # ---- Store state ----
        simX[i + 1, :] = xcurrent

        # ---- RTI: shift solution then run preparation for next step ----
        if USE_RTI:
            # FIX: shift the trajectory guess forward before preparing.
            # This ensures the next prep-phase linearises around a point
            # that is one dt ahead — close to the true next solution.
            shift_solution(ocp_solver, N_horizon)

            ocp_solver.options_set("rti_phase", 1)  # preparation phase
            ocp_solver.solve()

        # Progress update every 50 steps
        if (i + 1) % 50 == 0:
            print(f"  Step {i+1:3d}/{Nsim}  |  solve time: {t_solve*1000:.2f} ms  |  "
                  f"θ={xcurrent[1]:.3f} rad  |  u={u0[0]:.1f} N")

    # ------------------------------------------------------------------
    # Report timing statistics
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TIMING RESULTS")
    print("=" * 60)
    print(f"  Mean solve time:   {np.mean(solve_times)*1000:.3f} ms")
    print(f"  Max  solve time:   {np.max(solve_times)*1000:.3f} ms")
    print(f"  Min  solve time:   {np.min(solve_times)*1000:.3f} ms")
    print(f"  Sampling time dt:  {dt*1000:.1f} ms")
    if np.mean(solve_times) < dt:
        print(f"  ✓ Solver is REAL-TIME CAPABLE (solve < dt)")
    else:
        print(f"  ✗ Solver is TOO SLOW for real-time (solve > dt)")

    # ------------------------------------------------------------------
    # Plot results
    # ------------------------------------------------------------------
    plot_results(simX, simU, solve_times, dt)

    return simX, simU, solve_times


# ====================================================================
# PLOTTING
# ====================================================================

def plot_results(simX, simU, solve_times, dt):
    Nsim = simU.shape[0]
    t = np.linspace(0, Nsim * dt, Nsim + 1)
    t_u = t[:-1]

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle(f"Cart-Pole MPC  (RTI={USE_RTI}, Noise={ADD_NOISE}, Impulse={ADD_IMPULSE})",
                 fontsize=13)

    # -- Cart position --
    axes[0, 0].plot(t, simX[:, 0], color="steelblue")
    axes[0, 0].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[0, 0].set_ylabel("px (m)")
    axes[0, 0].set_title("Cart Position")
    axes[0, 0].grid(True)

    # -- Pendulum angle --
    axes[0, 1].plot(t, simX[:, 1], color="darkorange")
    axes[0, 1].axhline(0, color="gray", linestyle="--", alpha=0.5, label="target (upright)")
    axes[0, 1].axhline(np.pi, color="red", linestyle=":", alpha=0.4, label="start (hanging)")
    axes[0, 1].set_ylabel("θ (rad)")
    axes[0, 1].set_title("Pendulum Angle  (0 = upright)")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True)

    # -- Cart velocity --
    axes[1, 0].plot(t, simX[:, 2], color="steelblue")
    axes[1, 0].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[1, 0].set_ylabel("vx (m/s)")
    axes[1, 0].set_title("Cart Velocity")
    axes[1, 0].grid(True)

    # -- Angular velocity --
    axes[1, 1].plot(t, simX[:, 3], color="darkorange")
    axes[1, 1].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[1, 1].set_ylabel("ω (rad/s)")
    axes[1, 1].set_title("Angular Velocity")
    axes[1, 1].grid(True)

    # -- Control force --
    axes[2, 0].step(t_u, simU[:, 0], where="post", color="green")
    axes[2, 0].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[2, 0].axhline(80, color="red", linestyle=":", alpha=0.5, label="±Fmax")
    axes[2, 0].axhline(-80, color="red", linestyle=":", alpha=0.5)
    axes[2, 0].set_ylabel("F (N)")
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 0].set_title("Control Force")
    axes[2, 0].legend(fontsize=8)
    axes[2, 0].grid(True)

    # -- Solve times --
    axes[2, 1].plot(t_u, solve_times * 1000, color="purple", alpha=0.7)
    axes[2, 1].axhline(dt * 1000, color="red", linestyle="--",
                       label=f"dt = {dt*1000:.1f} ms (real-time limit)")
    axes[2, 1].set_ylabel("Solve time (ms)")
    axes[2, 1].set_xlabel("Time (s)")
    axes[2, 1].set_title("OCP Solve Time per Step  (feedback phase only)")
    axes[2, 1].legend(fontsize=8)
    axes[2, 1].grid(True)

    plt.tight_layout()
    plt.savefig("cartpole_results.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to cartpole_results.png")
    plt.show()


# ====================================================================
# ENTRY POINT
# ====================================================================

if __name__ == "__main__":
    simX, simU, solve_times = run_closed_loop()
