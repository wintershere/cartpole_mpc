"""Closed-loop episodes and metrics."""

import numpy as np

import config as C


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def run_episode(ctrl, plant, x0, rng, T=C.T_EPISODE, ref_steps=C.REF_STEPS,
                explore_sigma=0.0, meas_sigma=C.MEAS_SIGMA, on_step=None):
    n = int(round(T / C.DT))
    xs = np.zeros((n + 1, 4))
    ys = np.zeros((n + 1, 4))
    us = np.zeros(n)
    refs = np.zeros(n)
    t_fb = np.zeros(n)
    t_prep = np.zeros(n)
    stat = np.zeros(n, int)

    crashed = False
    plant.reset(x0)
    xs[0] = plant.state()
    ys[0] = xs[0] + rng.normal(0, meas_sigma)
    ctrl.reset(ys[0])
    for k in range(n):
        t = k * C.DT
        refs[k] = C.px_ref_at(t, ref_steps)
        if on_step is not None:
            on_step(ctrl, ys[k], refs[k])
        u, stat[k], t_fb[k], t_prep[k] = ctrl(ys[k], refs[k])
        if explore_sigma > 0:
            u = u + rng.normal(0, explore_sigma)
        us[k] = np.clip(u, -C.FMAX, C.FMAX)
        if hasattr(ctrl, "applied"):        # delay-aware controller: true command
            ctrl.applied(us[k])
        xs[k + 1] = plant.step(us[k])
        ys[k + 1] = xs[k + 1] + rng.normal(0, meas_sigma)
        if not np.all(np.isfinite(xs[k + 1])) or abs(xs[k + 1][0]) > C.RAIL_END:
            crashed = True
            us, refs = us[:k + 1], refs[:k + 1]
            xs, ys = xs[:k + 2], ys[:k + 2]
            t_fb, t_prep, stat = t_fb[:k + 1], t_prep[:k + 1], stat[:k + 1]
            break
    return dict(t=np.arange(len(us)) * C.DT, x=xs, y=ys, u=us, ref=refs,
                t_fb=t_fb, t_prep=t_prep, status=stat, crashed=crashed)


def metrics(ep, t_track=5.0):
    t, x, ref = ep["t"], ep["x"][:-1], ep["ref"]
    th = wrap(x[:, 1])
    up = (np.abs(th) < 0.1) & (np.abs(x[:, 3]) < 1.0)
    swing = float(t[np.argmax(up)]) if up.any() else np.nan
    fell = bool(ep.get("crashed", False) or np.isnan(swing)
                or np.any(np.abs(th[t >= swing + 0.5]) > np.pi / 2))
    tr = t >= t_track
    e = x[tr, 0] - ref[tr]
    # steady-state offset: last 1 s of each reference step after t_track
    ss = []
    steps = [s for s in C.REF_STEPS if s[0] >= t_track] + [(t[-1] + C.DT, None)]
    for (ts, r), (te, _) in zip(steps[:-1], steps[1:]):
        w = (t >= te - 1.0) & (t < te)
        if w.any():
            ss.append(abs(np.mean(x[w, 0]) - r))
    # settled error: skip 1.5 s after each reference change
    st = []
    for (ts, r), (te, _) in zip(steps[:-1], steps[1:]):
        w = (t >= ts + 1.5) & (t < te)
        st.append(x[w, 0] - r)
    st = np.concatenate(st) if st else np.array([np.nan])
    return dict(
        swing_up_s=swing,
        fell=fell,
        track_rms_m=float(np.sqrt(np.mean(e ** 2))) if tr.any() else np.nan,
        ss_offset_m=float(np.mean(ss)) if ss else np.nan,
        settled_rms_m=float(np.sqrt(np.mean(st ** 2))),
        theta_rms_rad=float(np.sqrt(np.mean(th[tr] ** 2))) if tr.any() else np.nan,
        max_abs_px_m=float(np.nanmax(np.abs(x[:, 0]))),
        u_rms_N=float(np.sqrt(np.mean(ep["u"] ** 2))),
        fb_ms_mean=float(1e3 * np.mean(ep["t_fb"])),
        fb_ms_max=float(1e3 * np.max(ep["t_fb"])),
        step_ms_mean=float(1e3 * np.mean(ep["t_fb"] + ep["t_prep"])),
        bad_status=int(np.sum((ep["status"] != 0) & (ep["status"] != 2))),
    )


def summarize(ms):
    keys = ms[0].keys()
    out = {}
    for k in keys:
        v = np.array([m[k] for m in ms], float)
        out[k] = dict(mean=float(np.nanmean(v)), std=float(np.nanstd(v)),
                      n_nan=int(np.isnan(v).sum()))
    return out
