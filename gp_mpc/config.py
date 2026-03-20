"""Shared settings for the GP-MPC experiments."""

import numpy as np

# Control timing (same as the original exercise: N=50, T=2 s)
DT = 0.04
N_HORIZON = 50
FMAX = 80.0

# Model the MPC is built on: textbook point-mass cart-pole
NOMINAL = dict(M=1.0, m=0.1, l=0.8, g=9.81, b_c=0.0, b_p=0.0, gear=1.0)

# "Real" plant (MuJoCo): what the MPC does not know.
# Heavier tip and cart, weaker actuator, viscous + Coulomb friction on
# the rail, damping in the hinge ("medium" level of mismatch_sweep.py).
TRUE = dict(M=1.4, m=0.2, l=0.8, g=9.81,
            b_c=3.0,      # rail viscous damping, N s/m
            f_c=2.5,      # rail Coulomb friction, N
            b_p=0.008,    # hinge damping, N m s/rad
            gear=0.75)    # actuator gain error

# Harder case: the nominal MPC drops the pole on this one.
TRUE_STRONG = dict(M=1.6, m=0.25, l=0.8, g=9.81, b_c=4.0, f_c=3.5,
                   b_p=0.012, gear=0.65)

# Best hand-written physics model: knows masses, gain and the
# viscous terms, but not the Coulomb friction (non-smooth).
ORACLE = dict(M=TRUE["M"], m=TRUE["m"], l=TRUE["l"], g=9.81,
              b_c=TRUE["b_c"], b_p=TRUE["b_p"], gear=TRUE["gear"])

# Rail length: |px| beyond this is a crash and ends the episode
RAIL_END = 3.0
# Every MPC knows the track: soft |px| limit, inside the crash limit
RAIL_SOFT = 2.5

# MPC weights (RTI settings from the original repo)
Q = np.diag([1e2, 1e2, 1e-2, 1e-2])
R = np.diag([1e-1])

# Measurement noise on [px, theta, vx, omega]
MEAS_SIGMA = np.array([1e-3, 1e-3, 1e-2, 1e-2])

# Evaluation task: swing up, then cart-position steps while balancing
T_EPISODE = 14.0
REF_STEPS = [(0.0, 0.0), (5.0, 1.0), (8.0, -1.0), (11.0, 0.0)]  # (t, px_ref)

# GP settings
GP_N_INDUCING = 300      # points kept in the MPC's GP mean
GP_N_HYPER = 800         # points used to fit hyperparameters

BUILD_DIR = "build"
RESULTS_DIR = "results"


def px_ref_at(t, steps=REF_STEPS):
    ref = steps[0][1]
    for ts, r in steps:
        if t >= ts:
            ref = r
    return ref
