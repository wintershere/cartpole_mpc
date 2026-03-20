"""How do dead time, motor lag and rail tilt affect the controllers?

Medium-mismatch MuJoCo plant plus one extra hidden effect per condition.
Controllers: nominal MPC and a GP-MPC (by default the medium round-1 GP,
trained WITHOUT delay or tilt). 5 episodes per condition.

Usage: python robustness_sweep.py [--gp results/medium/raw/gp_r1.pkl] [--tag base]
"""

import argparse
import json
import os
import pickle

import numpy as np

import config as C
from dynamics import rk4_ca, rk4_np
from episode import metrics, run_episode, summarize
from mpc import MPC, build_solver
from plants import MujocoPlant

ap = argparse.ArgumentParser()
ap.add_argument("--gp", default="results/medium/raw/gp_r1.pkl")
ap.add_argument("--n", type=int, default=5)
ap.add_argument("--tag", default="base")
a = ap.parse_args()

CONDS = ([("delay", d, dict(delay_steps=d)) for d in (0, 1, 2, 3)]
         + [("tilt", t, dict(tilt=t)) for t in (0.03, 0.06, 0.1)]
         + [("motor lag", 0.05, dict(motor_tau=0.05))])

step_nom = lambda x, u: rk4_np(x, u, C.NOMINAL, C.DT)
gp = pickle.load(open(a.gp, "rb"))
ctrls = {"nominal": MPC(build_solver("nominal_sweep", lambda x, u: rk4_ca(x, u, C.NOMINAL, C.DT),
                                     rail=C.RAIL_SOFT), step_nom, "nominal"),
         "gp_mpc": MPC(build_solver(f"gpmpc_sweep_{a.tag}", gp.xnext_ca, rail=C.RAIL_SOFT),
                       step_nom, "gp")}

rows = []
for kind, val, extra in CONDS:
    for name, ctrl in ctrls.items():
        ms = []
        for i in range(a.n):
            rng = np.random.default_rng(1000 + i)
            x0 = np.array([rng.uniform(-0.2, 0.2), np.pi + rng.uniform(-0.2, 0.2), 0, 0])
            ms.append(metrics(run_episode(ctrl, MujocoPlant({**C.TRUE, **extra}, C.DT), x0, rng)))
        s = summarize(ms)
        rows.append(dict(kind=kind, value=val, ctrl=name, fell=s["fell"]["mean"],
                         settled=s["settled_rms_m"]["mean"], offset=s["ss_offset_m"]["mean"],
                         theta=s["theta_rms_rad"]["mean"]))
        r = rows[-1]
        print(f"{kind:9s} {val:<5} {name:8s} fell {r['fell']:.0%}  settled {r['settled']:.3f} m  "
              f"offset {r['offset']:.3f} m  pole {r['theta']:.3f} rad", flush=True)

out = os.path.join(C.RESULTS_DIR, "robustness")
os.makedirs(out, exist_ok=True)
json.dump(dict(gp=a.gp, n=a.n, rows=rows), open(os.path.join(out, f"sweep_{a.tag}.json"), "w"), indent=1)
