"""Figures for the README, from results/<scenario>/.

Usage: python plots.py [--scenario medium]
"""

import argparse
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
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]      # validated categorical slots 1-3
STYLE = ["-", "--", "-"]
FEAT = ["sin θ", "cos θ", "cart speed", "pole rate", "force"]

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "lines.linewidth": 1.8, "legend.frameon": False,
})


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def best_gp_round(res):
    keys = [k for k in res["eval"] if k.startswith("gp_mpc")]
    ok = [k for k in keys if res["eval"][k]["fell"]["mean"] == 0]
    pool = ok or keys
    return min(pool, key=lambda k: res["eval"][k]["settled_rms_m"]["mean"])


def fig_closed_loop(out, raw, gp_key, scenario="medium"):
    runs = [("nominal MPC", "nominal"), ("oracle MPC", "oracle"), ("GP-MPC", gp_key)]
    runs = [r for r in runs if os.path.exists(os.path.join(raw, f"eval_{r[1]}.pkl"))]
    fig, ax = plt.subplots(3, 1, figsize=(9, 7.2), sharex=True)
    for (label, key) in runs:
        i = ["nominal", "oracle"].index(key) if key in ("nominal", "oracle") else 2
        ep = load(os.path.join(raw, f"eval_{key}.pkl"))["episode0"]
        t, x, u = ep["t"], ep["x"][:-1], ep["u"]
        kw = dict(color=SERIES[i], ls=STYLE[i], lw=1.8 if i != 2 else 2.2)
        ax[0].plot(t, x[:, 0], label=label, **kw)
        ax[1].plot(t, wrap(x[:, 1]), **kw)
        ax[2].plot(t, u, **kw)
        if ep.get("crashed", False):
            ax[0].plot([t[-1]], [np.clip(x[-1, 0], -2.9, 2.9)], "x", color=INK, ms=9, mew=2)
            ax[0].annotate(f"{label}: rail crash", (t[-1], np.clip(x[-1, 0], -2.9, 2.9)),
                           xytext=(8, -4), textcoords="offset points", color=INK, fontsize=9)
    ref = [C.px_ref_at(tt) for tt in t]
    ax[0].plot(t, ref, color=INK2, lw=1.0, ls=":", label="reference")
    ax[0].set_ylabel("cart position (m)")
    ax[1].set_ylabel("pole angle (rad)")
    ax[1].set_ylim(-0.6, 0.6)
    ax[2].set_ylabel("force (N)")
    ax[2].set_xlabel("time (s)")
    ax[0].set_ylim(-3.1, 3.1)
    ax[0].legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, fontsize=9)
    plant = ("Isaac Sim cart-pole, worn motor + rail friction" if scenario == "isaac"
             else f"mismatched MuJoCo plant, {scenario} mismatch")
    fig.suptitle(f"Swing-up, then cart-position steps while balancing ({plant})",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    ax[1].set_title("Pole angle after swing-up (0 = upright)", fontweight="normal", fontsize=10)
    for a in ax:
        a.axvspan(0, 5, color=GRID, alpha=0.35, lw=0)
    ax[0].text(0.1, -2.9, "swing-up", color=INK2, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "closed_loop.png"), dpi=150)
    plt.close(fig)


def fig_one_step(out, res):
    rows = res["one_step_round1"]
    names = [n for n in rows if not n.startswith("nominal")]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for j, (key, unit, title) in enumerate([("rmse_dvx", "m/s", "Cart-speed error per step"),
                                            ("rmse_dw", "rad/s", "Pole-rate error per step")]):
        v = [rows[n][key] for n in names]
        ax[j].barh(range(len(names)), v, color=SERIES[0], height=0.6)
        ax[j].set_yticks(range(len(names)), names)
        ax[j].invert_yaxis()
        ax[j].set_xlabel(f"RMSE on held-out episodes ({unit})")
        nom = rows["nominal (no learning)"][key]
        ax[j].set_title(f"{title}\n(no learning: {nom:.2f} {unit})")
        for k, val in enumerate(v):
            ax[j].text(val, k, f" {val:.3f}", va="center", fontsize=8, color=INK)
        ax[j].set_xlim(0, max(v) * 1.25)
        ax[j].grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "one_step.png"), dpi=150)
    plt.close(fig)


def fig_gp_band(out, raw):
    d = load(os.path.join(raw, "data_round1.pkl"))
    n = min(len(d["test_ep"]["u"]), int(4.0 / C.DT))     # first 4 s: swing-up
    P, S, D = d["P"][:n], d["sd"][:n], d["Dt"][:n]
    t = np.arange(n) * C.DT
    fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    for j, (lab, unit) in enumerate([("cart speed", "m/s"), ("pole rate", "rad/s")]):
        ax[j].fill_between(t, P[:, j] - 2 * S[:, j], P[:, j] + 2 * S[:, j], color=SERIES[0],
                           alpha=0.18, lw=0, label="GP ±2σ")
        ax[j].plot(t, D[:, j], color=INK2, lw=0, marker="o", ms=3.5, label="measured error")
        ax[j].plot(t, P[:, j], color=SERIES[0], lw=1.6, label="GP mean")
        ax[j].set_ylabel(f"{lab} error\nper step ({unit})")
    ax[0].legend(loc="upper right", ncol=3, fontsize=9)
    ax[0].set_title("What the textbook model gets wrong in one step, and the GP's prediction\n"
                    "(held-out episode, first 4 s)")
    ax[1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "gp_prediction.png"), dpi=150)
    plt.close(fig)


def fig_relevance(out, raw):
    gp = load(os.path.join(raw, "gp_r1.pkl"))
    rel = np.array([1.0 / e for e in gp.ell])          # 1 / length scale
    x = np.arange(len(FEAT))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    for j, lab in enumerate(["cart-speed error", "pole-rate error"]):
        ax.bar(x + (j - 0.5) * (w + 0.02), rel[j], width=w, color=SERIES[j], label=lab)
    ax.set_xticks(x, FEAT)
    ax.set_ylabel("relevance (1 / length scale)")
    ax.set_title("Which inputs the GP uses (ARD length scales)")
    ax.legend(fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "gp_relevance.png"), dpi=150)
    plt.close(fig)


def fig_rounds(out, res):
    base = [k for k in ("nominal", "oracle") if k in res["eval"]]
    keys = base + sorted(k for k in res["eval"] if k.startswith("gp_mpc"))
    labels = base + [f"GP r{k[-1]}" for k in keys[len(base):]]
    mets = [("settled_rms_m", "settled position error (m)"),
            ("ss_offset_m", "steady-state offset (m)"),
            ("theta_rms_rad", "pole angle RMS (rad)")]
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))
    for j, (m, title) in enumerate(mets):
        mu = [res["eval"][k][m]["mean"] for k in keys]
        mu = [0.0 if v != v else v for v in mu]         # NaN: never swung up
        sd = [res["eval"][k][m]["std"] for k in keys]
        ax[j].bar(range(len(keys)), mu, color=SERIES[0], width=0.6)
        ax[j].set_ylim(0, max(mu) * 1.25)
        ax[j].set_xticks(range(len(keys)), labels, rotation=0, fontsize=8)
        ax[j].set_title(title, fontsize=10)
        ax[j].grid(axis="x", visible=False)
        for k, k2 in enumerate(keys):
            f = res["eval"][k2]["fell"]["mean"]
            if f > 0:
                ax[j].annotate(f"fell in\n{f:.0%}", (k, mu[k]), xytext=(0, 4),
                               textcoords="offset points", ha="center", va="bottom",
                               fontsize=7, color=INK)
    fig.suptitle(f"Closed loop, mean over {res['n_eval']} episodes (lower is better)", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "rounds.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="medium")
    ap.add_argument("--gp", default="gp_mpc_r1", help="GP controller to plot ('best' = lowest settled error)")
    a = ap.parse_args()
    out = os.path.join(C.RESULTS_DIR, a.scenario)
    raw = os.path.join(out, "raw")
    res = json.load(open(os.path.join(out, "metrics.json")))
    gk = best_gp_round(res) if a.gp == "best" else a.gp
    fig_closed_loop(out, raw, gk, a.scenario)
    fig_one_step(out, res)
    fig_gp_band(out, raw)
    fig_relevance(out, raw)
    fig_rounds(out, res)
    print("figures written to", out, "| GP-MPC shown:", gk)
