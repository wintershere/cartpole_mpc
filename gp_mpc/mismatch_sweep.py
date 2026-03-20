"""Try mismatch levels: nominal vs oracle MPC on the MuJoCo plant."""
import numpy as np
import config as C
from dynamics import rk4_ca, rk4_np
from mpc import build_solver, MPC
from plants import MujocoPlant
from episode import run_episode, metrics, summarize

levels = {
  "mild":   dict(M=1.25, m=0.16, l=0.8, g=9.81, b_c=2.0, f_c=1.5, b_p=0.004, gear=0.85),
  "medium": dict(M=1.4,  m=0.2,  l=0.8, g=9.81, b_c=3.0, f_c=2.5, b_p=0.008, gear=0.75),
  "strong": dict(M=1.6,  m=0.25, l=0.8, g=9.81, b_c=4.0, f_c=3.5, b_p=0.012, gear=0.65),
}
nom_s = build_solver("nominal", lambda x, u: rk4_ca(x, u, C.NOMINAL, C.DT))
nom = MPC(nom_s, lambda x, u: rk4_np(x, u, C.NOMINAL, C.DT), "nominal")
keys = ["swing_up_s","fell","settled_rms_m","ss_offset_m","theta_rms_rad","max_abs_px_m","u_rms_N"]
for lv, P in levels.items():
    orc_p = {k: P[k] for k in ("M","m","l","g","b_c","b_p","gear")}
    orc = MPC(build_solver(f"oracle_{lv}", lambda x, u: rk4_ca(x, u, orc_p, C.DT)), lambda x, u: rk4_np(x, u, orc_p, C.DT), "o")
    for label, ctrl in (("nominal", nom), ("oracle", orc)):
        ms = []
        for i in range(6):
            rng = np.random.default_rng(100 + i)
            x0 = np.array([rng.uniform(-0.2, 0.2), np.pi + rng.uniform(-0.2, 0.2), 0, 0])
            ms.append(metrics(run_episode(ctrl, MujocoPlant(P, C.DT), x0, rng)))
        sm = summarize(ms)
        print(f"{lv:6s} {label:7s} " + " ".join(f"{k}={sm[k]['mean']:.3g}±{sm[k]['std']:.2g}" for k in keys))
