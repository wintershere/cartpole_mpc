"""GP-MPC pipeline on a mismatched MuJoCo cart-pole.

  1. Nominal MPC (textbook model) and oracle MPC (true masses, gain and
     viscous friction; no Coulomb friction) on the mismatched plant.
  2. Collect data: nominal MPC + exploration noise on the force.
  3. Fit the residual: GP vs boosting vs bagging vs a small neural net,
     scored on held-out episodes (one-step prediction).
  4. GP-MPC: the GP mean inside the acados model, closed loop.
  5. Round 2: collect with GP-MPC, refit on all data, closed loop again.

Usage:  python run_experiments.py [--scenario medium|strong] [--quick]
"""

import argparse
import json
import os
import pickle
import time

import numpy as np

import config as C
from dynamics import rk4_ca, rk4_np
from episode import metrics, run_episode, summarize
from mpc import MPC, build_solver
from plants import MujocoPlant
from gp_local import LocalGP
from residual import ResidualGP, dataset, fit_baselines, predict_any

ap = argparse.ArgumentParser()
ap.add_argument("--scenario", default="medium", choices=["medium", "strong", "isaac"])
ap.add_argument("--port", type=int, default=50555)
ap.add_argument("--quick", action="store_true")
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--explore", type=float, default=10.0, help="force noise std during data collection (N)")
ap.add_argument("--wide-x0", type=float, default=0.5,
                help="fraction of data episodes started from a random state on the whole circle")
ap.add_argument("--tag", default="", help="suffix for the results folder")
ap.add_argument("--n-train", type=int, default=None, help="data episodes (default 12)")
args = ap.parse_args()

TRUE = {"medium": C.TRUE, "strong": C.TRUE_STRONG, "isaac": None}[args.scenario]
# Oracle = true physics minus Coulomb friction; not defined for the Isaac
# asset (distributed masses, not a point-mass model).
ORACLE = None if TRUE is None else {k: TRUE[k] for k in ("M", "m", "l", "g", "b_c", "b_p", "gear")}
N_EVAL = 3 if args.quick else 10
N_TRAIN = args.n_train or (4 if args.quick else 12)
N_TEST = 2 if args.quick else 4
N_ROUND = 3 if args.quick else 6
EXPLORE = args.explore
OUT = os.path.join(C.RESULTS_DIR, args.scenario + args.tag)
RAW = os.path.join(OUT, "raw")
os.makedirs(RAW, exist_ok=True)
log = open(os.path.join(OUT, "log.txt"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.write(s + "\n")
    log.flush()


_ISAAC = None


def plant():
    global _ISAAC
    if args.scenario == "isaac":
        if _ISAAC is None:
            from plants import IsaacPlant
            _ISAAC = IsaacPlant(port=args.port)
            say(f"  isaac plant: {_ISAAC.info}")
        return _ISAAC
    return MujocoPlant(TRUE, C.DT)


def x0_eval(rng):
    return np.array([rng.uniform(-0.2, 0.2), np.pi + rng.uniform(-0.2, 0.2), 0.0, 0.0])


def random_refs(rng, T=C.T_EPISODE):
    """Swing-up, then 3-5 random cart-position steps (data diversity)."""
    ts = np.sort(rng.uniform(3.0, T - 1.5, rng.integers(3, 6)))
    return [(0.0, 0.0)] + [(float(t), float(rng.uniform(-1.2, 1.2))) for t in ts]


def x0_data(rng):
    """Data diversity: a fraction of episodes start anywhere on the circle
    with some velocity, the rest from hanging (like the evaluation)."""
    if rng.random() < args.wide_x0:
        return np.array([rng.uniform(-0.5, 0.5), rng.uniform(-np.pi, np.pi),
                         rng.uniform(-1.0, 1.0), rng.uniform(-3, 3)])
    return x0_eval(rng)


def evaluate(ctrl, label, seed0=1000):
    ms, eps = [], []
    for i in range(N_EVAL):
        rng = np.random.default_rng(seed0 + i)
        ep = run_episode(ctrl, plant(), x0_eval(rng), rng)
        ms.append(metrics(ep))
        eps.append(ep)
    sm = summarize(ms)
    say(f"  {label:22s} " + "  ".join(
        f"{k}={sm[k]['mean']:.3g}±{sm[k]['std']:.2g}" for k in
        ("swing_up_s", "fell", "settled_rms_m", "ss_offset_m", "theta_rms_rad",
         "max_abs_px_m", "u_rms_N", "fb_ms_mean", "fb_ms_max")))
    with open(os.path.join(RAW, f"eval_{label}.pkl"), "wb") as f:
        pickle.dump(dict(metrics=ms, summary=sm, episode0=eps[0]), f)
    return ms, sm, eps[0]


def collect(ctrl, n, seed0, label):
    eps = []
    for i in range(n):
        rng = np.random.default_rng(seed0 + i)
        eps.append(run_episode(ctrl, plant(), x0_data(rng), rng, ref_steps=random_refs(rng),
                               explore_sigma=EXPLORE))
    m = [metrics(e) for e in eps]
    say(f"  collected {n} episodes with {label}: {sum(len(e['u']) for e in eps)} steps, "
        f"{sum(x['fell'] for x in m)} fell")
    return eps


def one_step_scores(models, Z, D):
    rows = {}
    var = D.var(0)
    for name, mdl in models.items():
        P = np.zeros_like(D) if mdl is None else predict_any(mdl, Z)
        rmse = np.sqrt(np.mean((P - D) ** 2, 0))
        r2 = 1 - np.mean((P - D) ** 2, 0) / var
        rows[name] = dict(rmse_dvx=float(rmse[0]), rmse_dw=float(rmse[1]),
                          r2_dvx=float(r2[0]), r2_dw=float(r2[1]))
        say(f"  {name:26s} rmse dvx={rmse[0]:.4f} m/s  dw={rmse[1]:.4f} rad/s   "
            f"R2 dvx={r2[0]:.3f} dw={r2[1]:.3f}")
    return rows


def main():
    t_all = time.perf_counter()
    res = dict(scenario=args.scenario, true=TRUE, nominal=C.NOMINAL, oracle=ORACLE,
               explore_sigma=EXPLORE, n_eval=N_EVAL)
    sfx = args.scenario + args.tag.replace("-", "_")
    say(f"=== scenario {sfx}: true plant {TRUE}")

    nom = MPC(build_solver(f"nominal_{sfx}", lambda x, u: rk4_ca(x, u, C.NOMINAL, C.DT),
                           rail=C.RAIL_SOFT),
              lambda x, u: rk4_np(x, u, C.NOMINAL, C.DT), "nominal")
    orc = None if ORACLE is None else MPC(
        build_solver(f"oracle_{sfx}", lambda x, u: rk4_ca(x, u, ORACLE, C.DT), rail=C.RAIL_SOFT),
        lambda x, u: rk4_np(x, u, ORACLE, C.DT), "oracle")

    say("[1] closed loop, models without learning")
    res["eval"] = {}
    res["eval"]["nominal"] = evaluate(nom, "nominal")[1]
    if orc is not None:
        res["eval"]["oracle"] = evaluate(orc, "oracle")[1]

    say("[2] data from nominal MPC + exploration")
    train = collect(nom, N_TRAIN, 2000, "nominal MPC")
    test = collect(nom, N_TEST, 3000, "nominal MPC (test)")
    Z, D = dataset(train)
    Zt, Dt = dataset(test)
    say(f"  train {len(Z)} samples, test {len(Zt)} samples")

    say("[3] residual models, one-step prediction on held-out episodes")
    gp = ResidualGP().fit(Z, D)
    say(f"  GP fit {gp.fit_s:.1f} s; kernels: {gp.kernels()}")
    base, btimes = fit_baselines(Z, D)
    models = {"nominal (no learning)": None, f"GP ({C.GP_N_INDUCING} pts)": gp, **base}
    res["one_step_round1"] = one_step_scores(models, Zt, Dt)
    res["fit_seconds"] = {"GP": gp.fit_s, **btimes}
    _, sd = gp.predict(Zt, return_std=True)
    P = gp.predict(Zt)
    cover = np.mean(np.abs(P - Dt) <= 2 * sd, 0)
    res["gp_2sigma_coverage"] = cover.tolist()
    say(f"  GP 2-sigma coverage on test: dvx={cover[0]:.2f} dw={cover[1]:.2f}")
    with open(os.path.join(RAW, "data_round1.pkl"), "wb") as f:
        pickle.dump(dict(Z=Z, D=D, Zt=Zt, Dt=Dt, test_ep=test[0], P=P, sd=sd), f)

    say("[4] GP-MPC (round 1)")
    t0 = time.perf_counter()
    with open(os.path.join(RAW, "gp_r1.pkl"), "wb") as f:
        pickle.dump(gp, f)
    gp1 = MPC(build_solver(f"gpmpc_r1_{sfx}", gp.xnext_ca, rail=C.RAIL_SOFT),
              lambda x, u: rk4_np(x, u, C.NOMINAL, C.DT), "gp1")
    res["gp_build_s"] = time.perf_counter() - t0
    say(f"  solver build {res['gp_build_s']:.1f} s")
    res["eval"]["gp_mpc_r1"] = evaluate(gp1, "gp_mpc_r1")[1]

    say("[4b] local GP-MPC (Hewing et al. 2020): 10 inducing points along the predicted trajectory")
    lgp = LocalGP(n_ind=10).fit(Z, D)
    lmpc = MPC(build_solver(f"gplocal_r1_{sfx}", lgp.xnext_ca, rail=C.RAIL_SOFT, n_p=lgp.n_p),
               lambda x, u: rk4_np(x, u, C.NOMINAL, C.DT), "gplocal", param_cb=lgp.params)
    res["eval"]["gp_local_r1"] = evaluate(lmpc, "gp_local_r1")[1]

    all_eps = train
    gp_last = gp
    for r in range(2, args.rounds + 1):
        say(f"[5] round {r}: data from GP-MPC + exploration, refit on all data")
        prev = MPC(build_solver(f"gpmpc_r{r-1}_{sfx}", gp_last.xnext_ca, rail=C.RAIL_SOFT),
                   lambda x, u: rk4_np(x, u, C.NOMINAL, C.DT), "prev") if r > 2 else gp1
        new = collect(prev, N_ROUND, 4000 + 100 * r, f"GP-MPC r{r-1}")
        all_eps = all_eps + new
        Z, D = dataset(all_eps)
        gp_last = ResidualGP(seed=r).fit(Z, D)
        say(f"  GP fit on {len(Z)} samples, {gp_last.fit_s:.1f} s")
        res[f"one_step_round{r}"] = one_step_scores({f"GP round {r}": gp_last}, Zt, Dt)
        with open(os.path.join(RAW, f"data_round{r}.pkl"), "wb") as f:
            pickle.dump(dict(Z=Z, D=D), f)
        ctrl = MPC(build_solver(f"gpmpc_r{r}_{sfx}", gp_last.xnext_ca, rail=C.RAIL_SOFT),
                   lambda x, u: rk4_np(x, u, C.NOMINAL, C.DT), f"gp{r}")
        res["eval"][f"gp_mpc_r{r}"] = evaluate(ctrl, f"gp_mpc_r{r}")[1]

    with open(os.path.join(RAW, "gp_final.pkl"), "wb") as f:
        pickle.dump(gp_last, f)
    res["total_s"] = time.perf_counter() - t_all
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(res, f, indent=1)
    say(f"done in {res['total_s']:.0f} s")


if __name__ == "__main__":
    main()
