"""Figures for the dead-time study (results/delay/*/metrics.json).

Usage: python plot_delay.py [--main d2_tau0]
"""

import argparse
import glob
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as C
from episode import wrap

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]      # validated slots 1-4
CTRLS = [("nominal", "textbook MPC"),
         ("gp_no_delay", "GP-MPC, delay not modelled"),
         ("delay_aware", "delay-aware MPC, no GP"),
         ("delay_aware_gp_r1", "delay-aware GP-MPC")]
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False})

ap = argparse.ArgumentParser()
ap.add_argument("--main", default="d2_tau0")
a = ap.parse_args()
OUT = os.path.join(C.RESULTS_DIR, "delay")

# ---- summary across dead times (no motor lag runs) ----
runs = {}
for f in glob.glob(os.path.join(OUT, "d*_tau0", "metrics.json")):
    r = json.load(open(f))
    runs[r["plant"]["delay_steps"]] = r
ds = sorted(runs)
ms_ = [d * C.DT * 1000 for d in ds]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
w = 0.2
for i, (key, label) in enumerate(CTRLS):
    fell = [runs[d]["eval"][key]["fell"]["mean"] * 100 if key in runs[d]["eval"] else np.nan for d in ds]
    ax[0].bar(np.arange(len(ds)) + (i - 1.5) * (w + 0.01), fell, width=w, color=SERIES[i], label=label)
ax[0].set_xticks(range(len(ds)), [f"{m:.0f} ms" for m in ms_])
ax[0].set_ylabel("episodes where the pole fell (%)")
ax[0].set_title("Hidden dead time: who keeps the pole up")
ax[0].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.14), ncol=2)
ax[0].grid(axis="x", visible=False)
for key, label, c, ls in [("delay_aware_gp_r1", "delay-aware GP-MPC, round 1", SERIES[3], "-"),
                          ("delay_aware_gp_r2", "delay-aware GP-MPC, round 2", SERIES[3], "--")]:
    v = [runs[d]["eval"][key]["settled_rms_m"]["mean"] for d in ds]
    nfell = sum(round(runs[d]["eval"][key]["fell"]["mean"] * runs[d]["n_eval"]) for d in ds)
    label += f" (pole fell in {nfell} of {sum(runs[d]['n_eval'] for d in ds)})"
    ax[1].plot(ms_, v, ls, color=c, marker="o", ms=6, lw=2, label=label)
ax[1].set_xlabel("dead time")
ax[1].set_ylabel("settled position error (m)")
ax[1].set_title("Tracking of the controller that works")
ax[1].set_ylim(bottom=0)
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "delay_summary.png"), dpi=150)
plt.close(fig)

# ---- closed loop at the main dead time ----
eps = pickle.load(open(os.path.join(OUT, a.main, "episode0.pkl"), "rb"))
fig, axs = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
for i, (key, label) in enumerate(CTRLS):
    if key not in eps:
        continue
    ep = eps[key]
    t, x = ep["t"], ep["x"][:-1]
    kw = dict(color=SERIES[i], lw=2.2 if key == "delay_aware_gp_r1" else 1.4,
              ls="-" if key == "delay_aware_gp_r1" else "--")
    axs[0].plot(t, x[:, 0], label=label, **kw)
    axs[1].plot(t, wrap(x[:, 1]), **kw)
    if ep.get("crashed", False):
        axs[0].plot([t[-1]], [np.clip(x[-1, 0], -2.9, 2.9)], "x", color=INK, ms=8, mew=2)
axs[0].plot(t, [C.px_ref_at(tt) for tt in t], color=INK2, lw=1, ls=":", label="setpoint")
axs[0].set_ylim(-3.1, 3.1)
axs[0].set_ylabel("cart position (m)")
axs[1].set_ylim(-0.6, 0.6)
axs[1].set_ylabel("pole angle (rad)")
axs[1].set_xlabel("time (s)")
axs[0].legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
d = json.load(open(os.path.join(OUT, a.main, "metrics.json")))["plant"]["delay_steps"]
fig.suptitle(f"Hidden {d * C.DT * 1000:.0f} ms dead time (x = rail crash)", x=0.01, ha="left",
             fontweight="bold", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, a.main, "closed_loop.png"), dpi=150)
plt.close(fig)
print("figures written to", OUT)
