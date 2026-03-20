"""Dead time: identify it, put it in the model, learn the rest.

Plant: medium-mismatch MuJoCo cart-pole plus a hidden dead time (and
optional motor lag).

  1. nominal MPC
  2. data: nominal MPC + exploration
  3. identify the dead time d (and motor lag tau) from that data
  4. delay-aware MPC: identified d, tau in the model, no GP
  5. data from the delay-aware MPC
  6. GP-MPC WITHOUT the delay in the model (just learn it?)
  7. delay-aware GP-MPC
  8. one more round: data from delay-aware GP-MPC, refit, evaluate

Usage: python delay_experiments.py --delay 2 [--tau 0] [--tag name]
"""

import argparse
import json
import os
import time

import numpy as np

import config as C
from delay import AugController, AugModel, build_solver, identify, residual_dataset
from dynamics import rk4_ca, rk4_np
from episode import metrics, run_episode, summarize
from mpc import MPC
from mpc import build_solver as build_plain
from plants import MujocoPlant
from residual import ResidualGP, dataset

ap = argparse.ArgumentParser()
ap.add_argument("--delay", type=int, default=2, help="dead time in control steps (40 ms each)")
ap.add_argument("--tau", type=float, default=0.0, help="motor time constant (s), 0 = none")
ap.add_argument("--n-eval", type=int, default=10)
ap.add_argument("--n-data", type=int, default=12, help="data episodes per collection round")
ap.add_argument("--tag", default="")
ap.add_argument("--plant", default="mujoco", choices=["mujoco", "isaac"],
                help="isaac: start isaac_server.py with the same --delay-steps first")
ap.add_argument("--port", type=int, default=50555)
a = ap.parse_args()

PLANT = {**C.TRUE, "delay_steps": a.delay, "motor_tau": a.tau}
TAG = a.tag or (f"isaac_d{a.delay}" if a.plant == "isaac" else f"d{a.delay}_tau{int(a.tau * 1000)}")
_ISAAC = None


def plant():
    """MuJoCo: fresh plant per episode. Isaac: one client; the hidden
    effects are set by the server's flags."""
    global _ISAAC
    if a.plant == "isaac":
        if _ISAAC is None:
            from plants import IsaacPlant
            _ISAAC = IsaacPlant(port=a.port)
        return _ISAAC
    return MujocoPlant(PLANT, C.DT)
OUT = os.path.join(C.RESULTS_DIR, "delay", TAG)
os.makedirs(OUT, exist_ok=True)
log = open(os.path.join(OUT, "log.txt"), "w")


def say(*s):
    s = " ".join(str(x) for x in s)
    print(s, flush=True)
    log.write(s + "\n")
    log.flush()


def x0_eval(rng):
    return np.array([rng.uniform(-0.2, 0.2), np.pi + rng.uniform(-0.2, 0.2), 0.0, 0.0])


def x0_data(rng):
    if rng.random() < 0.5:
        return np.array([rng.uniform(-0.5, 0.5), rng.uniform(-np.pi, np.pi),
                         rng.uniform(-1.0, 1.0), rng.uniform(-3, 3)])
    return x0_eval(rng)


def random_refs(rng):
    ts = np.sort(rng.uniform(3.0, C.T_EPISODE - 1.5, rng.integers(3, 6)))
    return [(0.0, 0.0)] + [(float(t), float(rng.uniform(-1.2, 1.2))) for t in ts]


def collect(ctrl, seed0, label):
    eps = []
    for i in range(a.n_data):
        rng = np.random.default_rng(seed0 + i)
        eps.append(run_episode(ctrl, plant(), x0_data(rng), rng,
                               ref_steps=random_refs(rng), explore_sigma=10.0))
    say(f"  data from {label}: {sum(len(e['u']) for e in eps)} steps")
    return eps


def evaluate(ctrl, key, label):
    ms, eps = [], []
    for i in range(a.n_eval):
        rng = np.random.default_rng(1000 + i)
        ep = run_episode(ctrl, plant(), x0_eval(rng), rng)
        ms.append(metrics(ep))
        eps.append(ep)
    s = summarize(ms)
    say(f"  {label:34s} fell {s['fell']['mean']:4.0%}  settled {s['settled_rms_m']['mean']:.3f} m  "
        f"offset {s['ss_offset_m']['mean']:.3f} m  pole {s['theta_rms_rad']['mean']:.3f} rad  "
        f"step {s['step_ms_mean']['mean']:.2f} ms")
    res["eval"][key] = s
    episodes0[key] = eps[0]
    return s


res = dict(plant=PLANT, n_eval=a.n_eval, eval={})
episodes0 = {}
t_all = time.perf_counter()
if a.plant == "isaac":
    info = plant().info
    res["plant"] = info
    say(f"=== Isaac Sim cart-pole (hidden: gain {info['gear']}, rail friction "
        f"{info['rail_viscous']} N s/m + {info['rail_coulomb']} N, dead time "
        f"{info.get('delay_steps', 0) * C.DT * 1000:.0f} ms)")
    if info.get("delay_steps", 0) != a.delay:
        say(f"  WARNING: server delay {info.get('delay_steps')} != --delay {a.delay}")
else:
    say(f"=== medium mismatch + dead time {a.delay * C.DT * 1000:.0f} ms"
        + (f" + motor lag {a.tau} s" if a.tau else ""))
step_nom = lambda x, u: rk4_np(x, u, C.NOMINAL, C.DT)

say("[1] nominal MPC")
nom = MPC(build_plain(f"nom_{TAG}", lambda x, u: rk4_ca(x, u, C.NOMINAL, C.DT), rail=C.RAIL_SOFT),
          step_nom, "nominal")
evaluate(nom, "nominal", "nominal MPC")

say("[2] data")
data = collect(nom, 2000, "nominal MPC")

say("[3] identify the dead time (and motor lag)")
(d, tau), scores = identify(data)
top = sorted(scores, key=scores.get)[:3]
say(f"  identified {d} steps = {d * C.DT * 1000:.0f} ms, tau {tau} s  "
    f"(true {a.delay} steps, tau {a.tau} s); best fits: "
    + ", ".join(f"d={k[0]} tau={k[1]}: {scores[k]:.2f}" for k in top))
res["identified"] = dict(d=d, tau=tau, true_d=a.delay, true_tau=a.tau)

say("[4] delay-aware MPC, no GP")
m0 = AugModel(d, tau)
c0 = AugController(build_solver(f"aug_{TAG}", m0), m0)
evaluate(c0, "delay_aware", "delay-aware MPC")

say("[5] data")
data += collect(c0, 3000, "delay-aware MPC")

say("[6] GP-MPC without the delay in its model")
Z, D = dataset(data)
gp_plain = ResidualGP().fit(Z, D)
c_plain = MPC(build_plain(f"gpplain_{TAG}", gp_plain.xnext_ca, rail=C.RAIL_SOFT), step_nom, "gp")
evaluate(c_plain, "gp_no_delay", "GP-MPC, delay not modelled")

say("[7] delay-aware GP-MPC")
Z, D = residual_dataset(data, d, tau)
gp1 = ResidualGP().fit(Z, D)
say(f"  GP on {len(Z)} samples, {gp1.fit_s:.0f} s")
m1 = AugModel(d, tau, gp=gp1)
c1 = AugController(build_solver(f"auggp1_{TAG}", m1), m1)
evaluate(c1, "delay_aware_gp_r1", "delay-aware GP-MPC")

say("[8] one more round: data from delay-aware GP-MPC, refit")
data += collect(c1, 4000, "delay-aware GP-MPC")
Z, D = residual_dataset(data, d, tau)
gp2 = ResidualGP(seed=2).fit(Z, D)
say(f"  GP on {len(Z)} samples, {gp2.fit_s:.0f} s")
m2 = AugModel(d, tau, gp=gp2)
c2 = AugController(build_solver(f"auggp2_{TAG}", m2), m2)
evaluate(c2, "delay_aware_gp_r2", "delay-aware GP-MPC, round 2")

res["total_s"] = time.perf_counter() - t_all
json.dump(res, open(os.path.join(OUT, "metrics.json"), "w"), indent=1)
import pickle
pickle.dump(episodes0, open(os.path.join(OUT, "episode0.pkl"), "wb"))
pickle.dump(dict(d=d, tau=tau, gp1=gp1, gp2=gp2), open(os.path.join(OUT, "models.pkl"), "wb"))
say(f"done in {res['total_s']:.0f} s")
