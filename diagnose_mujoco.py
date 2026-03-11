"""
diagnose_mujoco.py
==================
Run this to find exactly what's going wrong with the MuJoCo + acados integration.
It tests each piece independently: state reading, force sign, dynamics match, solver.
"""

import numpy as np
import shutil
import os
import glob

# ================================================================
# STEP 0: Clean stale acados build artifacts
# ================================================================
print("=" * 70)
print("STEP 0: Cleaning stale acados build artifacts")
print("=" * 70)

for pattern in ["c_generated_code", "cartpole_ocp.json", "cartpole_sim.json",
                "acados_ocp_*.json", "acados_sim_*.json"]:
    for path in glob.glob(pattern):
        if os.path.isdir(path):
            shutil.rmtree(path)
            print(f"  Removed directory: {path}")
        elif os.path.isfile(path):
            os.remove(path)
            print(f"  Removed file: {path}")
print("  Done.\n")

# ================================================================
# STEP 1: Load MuJoCo and check initial state
# ================================================================
print("=" * 70)
print("STEP 1: MuJoCo initial state")
print("=" * 70)

import mujoco

XML_PATH = "cartpole.xml"
mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
mj_data  = mujoco.MjData(mj_model)

print(f"  Model timestep:  {mj_model.opt.timestep}")
print(f"  Gravity:         {mj_model.opt.gravity}")
print(f"  N joints:        {mj_model.njnt}")
print(f"  N actuators:     {mj_model.nu}")
print(f"  N qpos:          {mj_model.nq}")
print(f"  N qvel:          {mj_model.nv}")

# Print joint info
for i in range(mj_model.njnt):
    jnt_name = mj_model.joint(i).name
    jnt_type = mj_model.joint(i).type  # 2=slide, 3=hinge
    jnt_axis = mj_model.joint(i).axis
    print(f"  Joint {i}: name='{jnt_name}' type={jnt_type} axis={jnt_axis}")

# Reset to keyframe
print(f"\n  Number of keyframes: {mj_model.nkey}")
mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
mujoco.mj_forward(mj_model, mj_data)

print(f"  After keyframe reset:")
print(f"    qpos = {mj_data.qpos}")
print(f"    qvel = {mj_data.qvel}")

x0_mj = np.array([mj_data.qpos[0], mj_data.qpos[1], 
                   mj_data.qvel[0], mj_data.qvel[1]])
print(f"    State [px, theta, vx, omega] = {x0_mj}")
print(f"    Expected:                      [0, {np.pi:.6f}, 0, 0]")
print(f"    theta/pi = {x0_mj[1]/np.pi:.6f} (should be 1.0)")

if abs(x0_mj[1] - np.pi) > 0.01:
    print(f"  *** WARNING: theta != pi! This could be the bug. ***")
    print(f"  *** MuJoCo starts with theta={x0_mj[1]:.4f} but acados expects pi ***")
elif abs(x0_mj[1]) < 0.01:
    print(f"  *** CRITICAL: theta ≈ 0 — pendulum is UPRIGHT in MuJoCo! ***")
    print(f"  *** The XML convention doesn't match. theta=0 should be upright ***")
    print(f"  *** but the initial state should be theta=pi (hanging). ***")
else:
    print(f"  ✓ Initial state looks correct")

# ================================================================
# STEP 2: Test force sign convention
# ================================================================
print(f"\n{'=' * 70}")
print("STEP 2: Force sign convention test")
print("=" * 70)
print("  Applying F=+10N for a short time, checking cart acceleration direction")

mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
mujoco.mj_forward(mj_model, mj_data)

mj_data.ctrl[0] = 10.0
for _ in range(10):
    mujoco.mj_step(mj_model, mj_data)

print(f"  After +10N for {10 * mj_model.opt.timestep:.4f}s:")
print(f"    px  = {mj_data.qpos[0]:.6f} (should be positive → force pushes right)")
print(f"    vx  = {mj_data.qvel[0]:.6f} (should be positive)")
print(f"    θ   = {mj_data.qpos[1]:.6f}")
print(f"    ω   = {mj_data.qvel[1]:.6f}")

if mj_data.qvel[0] > 0:
    print(f"  ✓ Positive force → positive cart velocity (matches acados)")
else:
    print(f"  *** WARNING: Positive force gives NEGATIVE velocity! ***")
    print(f"  *** Force convention is flipped between MuJoCo and acados ***")

# ================================================================
# STEP 3: Compare one-step dynamics: MuJoCo vs exercise ODE
# ================================================================
print(f"\n{'=' * 70}")
print("STEP 3: Dynamics comparison (MuJoCo vs ODE model)")
print("=" * 70)

M, m, l, g = 1.0, 0.1, 0.8, 9.81

def ode_rhs(x, u):
    _, theta, _, omega = x
    D       = M + m * (1 - np.cos(theta)**2)
    f_vx    = (-m*l*np.sin(theta)*omega**2 + m*g*np.cos(theta)*np.sin(theta) + u) / D
    f_omega = (-m*l*np.cos(theta)*np.sin(theta)*omega**2 + u*np.cos(theta)
               + (M+m)*g*np.sin(theta)) / (l*D)
    return np.array([x[2], x[3], f_vx, f_omega])

# Test at several states
test_cases = [
    ("Hanging (θ=π), F=0",     [0, np.pi, 0, 0],       0.0),
    ("Hanging (θ=π), F=+50",   [0, np.pi, 0, 0],       50.0),
    ("Hanging (θ=π), F=-50",   [0, np.pi, 0, 0],      -50.0),
    ("Tilted (θ=π-0.3), F=0",  [0, np.pi-0.3, 0, 0],   0.0),
    ("Tilted (θ=π+0.3), F=0",  [0, np.pi+0.3, 0, 0],   0.0),
    ("Moving (θ=π, ω=2), F=0", [0, np.pi, 0, 2.0],     0.0),
]

dt_test = 0.01  # short step for comparison
n_sub = max(1, round(dt_test / mj_model.opt.timestep))

print(f"  Comparing one integration step of dt={dt_test}s (MuJoCo uses {n_sub} substeps)\n")

all_match = True
for name, x0_test, F_test in test_cases:
    # --- ODE (Euler step for comparison) ---
    xdot = ode_rhs(x0_test, F_test)
    x_ode = np.array(x0_test) + dt_test * xdot

    # --- MuJoCo ---
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    mj_data.qpos[0] = x0_test[0]
    mj_data.qpos[1] = x0_test[1]
    mj_data.qvel[0] = x0_test[2]
    mj_data.qvel[1] = x0_test[3]
    mujoco.mj_forward(mj_model, mj_data)
    
    mj_data.ctrl[0] = F_test
    for _ in range(n_sub):
        mujoco.mj_step(mj_model, mj_data)
    
    x_mj = np.array([mj_data.qpos[0], mj_data.qpos[1],
                      mj_data.qvel[0], mj_data.qvel[1]])
    
    diff = x_mj - x_ode
    max_diff = np.max(np.abs(diff))
    
    match = "✓" if max_diff < 0.05 else "✗ MISMATCH"
    if max_diff >= 0.05:
        all_match = False
    
    print(f"  {name}")
    print(f"    ODE  next state: [{x_ode[0]:+.5f}, {x_ode[1]:+.5f}, {x_ode[2]:+.5f}, {x_ode[3]:+.5f}]")
    print(f"    MuJo next state: [{x_mj[0]:+.5f}, {x_mj[1]:+.5f}, {x_mj[2]:+.5f}, {x_mj[3]:+.5f}]")
    print(f"    Max diff: {max_diff:.6f}  {match}")
    
    # Check sign agreement on accelerations
    acc_ode = xdot[2:]  # [a_x, alpha]
    acc_mj = (x_mj[2:] - np.array(x0_test[2:])) / dt_test
    sign_match = np.sign(acc_ode) == np.sign(acc_mj)
    if not np.all(sign_match[np.abs(acc_ode) > 0.1]):  # only check non-tiny
        print(f"    *** SIGN MISMATCH on accelerations! ***")
        print(f"    ODE accelerations:   [{acc_ode[0]:+.3f}, {acc_ode[1]:+.3f}]")
        print(f"    MuJoCo accelerations:[{acc_mj[0]:+.3f}, {acc_mj[1]:+.3f}]")
        all_match = False
    print()

if all_match:
    print("  ✓ Dynamics match well (expected for matched conventions)")
else:
    print("  ✗ Dynamics MISMATCH — check force signs and angle conventions")

# ================================================================
# STEP 4: Test OCP solver directly with the MuJoCo initial state
# ================================================================
print(f"\n{'=' * 70}")
print("STEP 4: OCP solver test from MuJoCo initial state")
print("=" * 70)

from cartpole_ocp import export_cartpole_ocp_solver

# Use same parameters as mujoco script
Q = np.diag([2e2, 2e2, 1e-1, 1e-1])
R = 1e-1 * np.eye(1)
N = 50
T = 2.0

print("  Building solver (this compiles C code)...")
solver = export_cartpole_ocp_solver(N=N, T=T, Fmax=80.0, use_RTI=False, Q=Q, R=R)
print("  Solver built.\n")

# Reset MuJoCo to get initial state
mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
mujoco.mj_forward(mj_model, mj_data)
x0 = np.array([mj_data.qpos[0], mj_data.qpos[1],
                mj_data.qvel[0], mj_data.qvel[1]])

print(f"  x0 from MuJoCo: {x0}")

# --- Try WITHOUT warm-start first (same as acados-only script) ---
print(f"\n  --- Test A: Solve WITHOUT warm-start ---")
solver.set(0, "lbx", x0)
solver.set(0, "ubx", x0)
status = solver.solve()
u0_a = solver.get(0, "u")
print(f"  Status: {status}  (0=converged, 2=max_iter, else=fail)")
print(f"  u0 = {u0_a}")
print(f"  x[1] = {solver.get(1, 'x')}")

# Print full trajectory summary
print(f"  Trajectory preview:")
for k in [0, 5, 10, 25, 49, 50]:
    xk = solver.get(k, "x") if k <= N else None
    uk = solver.get(k, "u") if k < N else None
    if k <= N:
        print(f"    node {k:2d}: x={xk.round(4)}", end="")
    if k < N:
        print(f"  u={uk.round(2)}", end="")
    print()

# --- Try WITH warm-start (as in mujoco script) ---
print(f"\n  --- Test B: Solve WITH RK4 warm-start ---")

# Rebuild solver to clear state
solver2 = export_cartpole_ocp_solver(N=N, T=T, Fmax=80.0, use_RTI=False, Q=Q, R=R)

def rk4(x, u, dt):
    def f(x, u):
        return ode_rhs(x, u)
    k1 = f(x, u)
    k2 = f(x + dt/2*k1, u)
    k3 = f(x + dt/2*k2, u)
    k4 = f(x + dt*k3, u)
    return x + dt/6*(k1 + 2*k2 + 2*k3 + k4)

dt_mpc = T / N
x_roll = x0.copy()
for k in range(N):
    solver2.set(k, "x", x_roll)
    solver2.set(k, "u", np.zeros(1))
    x_roll = rk4(x_roll, 0.0, dt_mpc)
solver2.set(N, "x", x_roll)
print(f"  Rollout end state: {x_roll.round(4)}")

solver2.set(0, "lbx", x0)
solver2.set(0, "ubx", x0)
status2 = solver2.solve()
u0_b = solver2.get(0, "u")
print(f"  Status: {status2}  (0=converged, 2=max_iter, else=fail)")
print(f"  u0 = {u0_b}")

print(f"  Trajectory preview:")
for k in [0, 5, 10, 25, 49, 50]:
    xk = solver2.get(k, "x") if k <= N else None
    uk = solver2.get(k, "u") if k < N else None
    if k <= N:
        print(f"    node {k:2d}: x={xk.round(4)}", end="")
    if k < N:
        print(f"  u={uk.round(2)}", end="")
    print()

# --- For reference: test with hardcoded x0=[0, pi, 0, 0] ---
print(f"\n  --- Test C: Solve with hardcoded x0=[0, π, 0, 0] (sanity check) ---")
solver3 = export_cartpole_ocp_solver(N=N, T=T, Fmax=80.0, use_RTI=False, Q=Q, R=R)
x0_ref = np.array([0.0, np.pi, 0.0, 0.0])
solver3.set(0, "lbx", x0_ref)
solver3.set(0, "ubx", x0_ref)
status3 = solver3.solve()
u0_c = solver3.get(0, "u")
print(f"  Status: {status3}")
print(f"  u0 = {u0_c}")

# ================================================================
# STEP 5: Run 5 MPC steps with MuJoCo plant (micro closed-loop test)
# ================================================================
print(f"\n{'=' * 70}")
print("STEP 5: Mini closed-loop (5 steps) with MuJoCo plant")
print("=" * 70)

solver4 = export_cartpole_ocp_solver(N=N, T=T, Fmax=80.0, use_RTI=False, Q=Q, R=R)

mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
mujoco.mj_forward(mj_model, mj_data)
xcurrent = np.array([mj_data.qpos[0], mj_data.qpos[1],
                      mj_data.qvel[0], mj_data.qvel[1]])

print(f"  Initial state: {xcurrent}")

for step in range(5):
    solver4.set(0, "lbx", xcurrent)
    solver4.set(0, "ubx", xcurrent)
    status = solver4.solve()
    u0 = solver4.get(0, "u")
    
    # Apply to MuJoCo
    n_sub = max(1, round(dt_mpc / mj_model.opt.timestep))
    mj_data.ctrl[0] = u0[0]
    for _ in range(n_sub):
        mujoco.mj_step(mj_model, mj_data)
    
    x_new = np.array([mj_data.qpos[0], mj_data.qpos[1],
                       mj_data.qvel[0], mj_data.qvel[1]])
    
    print(f"  Step {step}: status={status}  u={u0[0]:+7.2f}N  "
          f"x=[{xcurrent[0]:+.4f}, {xcurrent[1]:+.4f}, {xcurrent[2]:+.4f}, {xcurrent[3]:+.4f}] "
          f"→ [{x_new[0]:+.4f}, {x_new[1]:+.4f}, {x_new[2]:+.4f}, {x_new[3]:+.4f}]")
    
    xcurrent = x_new

print(f"\n{'=' * 70}")
print("DIAGNOSIS COMPLETE — share the output above")
print("=" * 70)
