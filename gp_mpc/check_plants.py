"""Nominal / oracle MPC on matched and mismatched plants."""
import sys
import numpy as np
import config as C
from dynamics import rk4_ca, rk4_np
from mpc import build_solver, MPC
from plants import MujocoPlant, OdePlant
from episode import run_episode, metrics

def make(name, p, rail=None):
    s = build_solver(name, lambda x, u: rk4_ca(x, u, p, C.DT), rail=rail)
    return MPC(s, lambda x, u: rk4_np(x, u, p, C.DT), name, rail)

nom = make("nominal", C.NOMINAL)
orc = make("oracle", C.ORACLE)
rng = np.random.default_rng(0)
x0 = np.array([0.0, np.pi, 0.0, 0.0])
for label, ctrl, plant in [("nominal | ode nominal", nom, OdePlant(C.NOMINAL, C.DT)),
                           ("nominal | mujoco nominal", nom, MujocoPlant(C.NOMINAL, C.DT)),
                           ("nominal | mujoco TRUE", nom, MujocoPlant(C.TRUE, C.DT)),
                           ("oracle  | mujoco TRUE", orc, MujocoPlant(C.TRUE, C.DT))]:
    ep = run_episode(ctrl, plant, x0, np.random.default_rng(1))
    m = metrics(ep)
    print(f"{label:26s} " + " ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()))
