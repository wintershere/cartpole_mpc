"""How sensitive is GP-MPC to the GP's prior mean, point selection and seed?
Fits on the round-1 data of a scenario, 3 closed-loop episodes per config."""
import itertools, pickle, sys, json, os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import casadi as ca

SCEN = sys.argv[1] if len(sys.argv) > 1 else "medium"
DATA = sys.argv[2] if len(sys.argv) > 2 else "data_round1.pkl"
N_EP = 4

def job(cfg):
    method, select, env_q, rail, seed = cfg
    import config as C
    from dynamics import rk4_np
    from mpc import build_solver, MPC
    from plants import MujocoPlant
    from episode import run_episode, metrics
    from residual import ResidualGP
    TRUE = C.TRUE if SCEN == "medium" else C.TRUE_STRONG
    d = pickle.load(open(f"results/{SCEN}/raw/{DATA}", "rb"))
    gp = ResidualGP(seed=seed, method=method, select=select).fit(d["Z"], d["D"])
    # CasADi mean == sklearn mean?
    x = ca.SX.sym("x", 4); u = ca.SX.sym("u", 1)
    f = ca.Function("f", [x, u], [gp.xnext_ca(x, u)])
    rng = np.random.default_rng(0); err = 0
    for _ in range(50):
        xx = rng.normal(0, [0.5, 2, 2, 3]); uu = rng.normal(0, 20)
        err = max(err, np.max(np.abs(np.array(f(xx, [uu])).ravel() - gp.xnext_np(xx, uu))))
    env = None if env_q is None else gp.envelope(d["Z"], env_q)
    name = f"rc_{SCEN}_{method}_{select}_{env_q}_{int(rail)}_{seed}".replace(".", "p")
    ctrl = MPC(build_solver(name, gp.xnext_ca, rail=C.RAIL_SOFT if rail else None, envelope=env),
               lambda x, u: rk4_np(x, u, C.NOMINAL, C.DT), name)
    ms = []
    for i in range(N_EP):
        r = np.random.default_rng(1000 + i)
        x0 = np.array([r.uniform(-0.2, 0.2), np.pi + r.uniform(-0.2, 0.2), 0, 0])
        ms.append(metrics(run_episode(ctrl, MujocoPlant(TRUE, C.DT), x0, r)))
    return dict(method=method, select=select, env_q=env_q, rail=rail, seed=seed,
                max_px=float(np.nanmean([m["max_abs_px_m"] for m in ms])),
                swing=float(np.nanmean([m["swing_up_s"] for m in ms])), casadi_err=float(err),
                fell=sum(m["fell"] for m in ms),
                settled=float(np.nanmean([m["settled_rms_m"] for m in ms])),
                ss=float(np.nanmean([m["ss_offset_m"] for m in ms])),
                theta=float(np.nanmean([m["theta_rms_rad"] for m in ms])),
                kern=gp.kernels())

if __name__ == "__main__":
    cfgs = [(m, "fps", q, r, sd) for m in ("dtc", "sod") for q, r in
            [(None, True), (None, False), (100, True), (99.5, True)] for sd in (0, 1, 2)]
    with ProcessPoolExecutor(8) as ex:
        rows = list(ex.map(job, cfgs))
    for r in rows:
        print(f"{r['method']} env={str(r['env_q']):5s} rail={int(r['rail'])} seed={r['seed']}  fell={r['fell']}/{N_EP}  "
              f"swing={r['swing']:.2f} max_px={r['max_px']:.2f}  "
              f"settled={r['settled']:.3f}  ss={r['ss']:.3f}  theta={r['theta']:.3f}  casadi_err={r['casadi_err']:.1e}")
    json.dump(rows, open(f"results/{SCEN}/robust_check_{DATA[:-4]}.json", "w"), indent=1)
