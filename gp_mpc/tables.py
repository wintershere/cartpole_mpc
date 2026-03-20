"""Print the README result tables (markdown) from results/<scenario>/metrics.json."""

import json
import sys

NAMES = {"nominal": "Nominal MPC (textbook model)",
         "oracle": "Oracle MPC (true masses, gain, viscous friction)",
         "gp_local_r1": "GP-MPC, local GP (Hewing et al.), round 1"}


def name(k):
    if k in NAMES:
        return NAMES[k]
    if k.startswith("gp_mpc_r"):
        return f"GP-MPC, sparse GP, round {k[-1]}"
    return k


def fmt(s, k, nd=3, pct=False):
    m, sd = s[k]["mean"], s[k]["std"]
    if pct:
        return f"{m:.0%}"
    if m != m:  # nan
        return "n/a"
    return f"{m:.{nd}f} ± {sd:.{nd}f}"


for scen in sys.argv[1:] or ["medium", "strong"]:
    res = json.load(open(f"results/{scen}/metrics.json"))
    print(f"\n### {scen} mismatch: closed loop, {res['n_eval']} episodes each\n")
    print("| Controller | Pole fell | Swing-up (s) | Settled position error (m) | "
          "Steady-state offset (m) | Pole angle RMS (rad) | Feedback solve (ms) |")
    print("|---|---|---|---|---|---|---|")
    order = ["nominal", "oracle"] + sorted(k for k in res["eval"] if k.startswith("gp_mpc")) \
        + [k for k in res["eval"] if k.startswith("gp_local")]
    for k in order:
        if k not in res["eval"]:
            continue
        s = res["eval"][k]
        print(f"| {name(k)} | {fmt(s, 'fell', pct=True)} | {fmt(s, 'swing_up_s', 2)} | "
              f"{fmt(s, 'settled_rms_m')} | {fmt(s, 'ss_offset_m')} | {fmt(s, 'theta_rms_rad')} | "
              f"{s['fb_ms_mean']['mean']:.2f} (max {s['fb_ms_max']['mean']:.1f}) |")
    print(f"\n### {scen} mismatch: one-step prediction of the residual (held-out episodes)\n")
    print("| Model | Cart-speed error RMSE (m/s) | Pole-rate error RMSE (rad/s) | R² cart | R² pole |")
    print("|---|---|---|---|---|")
    for m, r in res["one_step_round1"].items():
        print(f"| {m} | {r['rmse_dvx']:.4f} | {r['rmse_dw']:.4f} | {r['r2_dvx']:.3f} | {r['r2_dw']:.3f} |")
    print(f"\nGP 2-sigma coverage on held-out data: {res['gp_2sigma_coverage']}")
