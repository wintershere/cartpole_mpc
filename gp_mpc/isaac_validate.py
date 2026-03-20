"""Validation in Isaac Sim with full RTX rendering: nominal MPC vs GP-MPC on
the same evaluation episodes as run_experiments.py (same seeds and start
states), using the GP trained on Isaac data. Episode 0 of each controller
is filmed from the real renders and turned into a side-by-side GIF.

    # terminal 1
    cd ~/isaacsim6 && ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1 ./python.sh \\
        ~/cartpole_mpc/gp_mpc/isaac_server.py --render \\
        --gear 0.65 --rail-viscous 4.0 --rail-coulomb 3.5
    # terminal 2
    cd ~/cartpole_mpc/gp_mpc && ../.venv/bin/python isaac_validate.py
"""

import argparse
import glob
import json
import os
import pickle

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config as C
from dynamics import rk4_ca, rk4_np
from episode import metrics, run_episode, summarize
from mpc import MPC, build_solver
from plants import IsaacPlant

ap = argparse.ArgumentParser()
ap.add_argument("--gp", default="results/isaac/raw/gp_r1.pkl")
ap.add_argument("--n-eval", type=int, default=10)
ap.add_argument("--port", type=int, default=50555)
ap.add_argument("--out", default="results/isaac_render")
ap.add_argument("--gif-only", action="store_true", help="rebuild the GIF from saved frames")
ap.add_argument("--scale", type=float, default=0.72, help="GIF size relative to the crop")
ap.add_argument("--delay-run", default=None,
                help="results/delay/<run>: film textbook MPC vs delay-aware GP-MPC from that run "
                     "(server needs the same --delay-steps)")
a = ap.parse_args()
if a.delay_run and a.out == "results/isaac_render":
    a.out = "results/isaac_render_delay"
FRAMES = os.path.abspath(os.path.join(C.BUILD_DIR, "isaac_render_frames" + ("_delay" if a.delay_run else "")))
os.makedirs(a.out, exist_ok=True)


def x0_eval(rng):
    return np.array([rng.uniform(-0.2, 0.2), np.pi + rng.uniform(-0.2, 0.2), 0.0, 0.0])


LABELS = ([("textbook MPC", "nominal"), ("delay-aware GP-MPC", "gp_mpc")] if a.delay_run
          else [("nominal MPC", "nominal"), ("GP-MPC", "gp_mpc")])
if not a.gif_only:
    step_nom = lambda x, u: rk4_np(x, u, C.NOMINAL, C.DT)
    nom = MPC(build_solver("nominal_isaacval", lambda x, u: rk4_ca(x, u, C.NOMINAL, C.DT),
                           rail=C.RAIL_SOFT), step_nom, "nominal")
    if a.delay_run:
        from delay import AugController, AugModel
        from delay import build_solver as build_aug
        mods = pickle.load(open(os.path.join(a.delay_run, "models.pkl"), "rb"))
        am = AugModel(mods["d"], mods["tau"], gp=mods["gp1"])
        second = AugController(build_aug("auggp_isaacval", am), am)
        a.gp = os.path.join(a.delay_run, "models.pkl")
    else:
        gp = pickle.load(open(a.gp, "rb"))
        second = MPC(build_solver("gpmpc_isaacval", gp.xnext_ca, rail=C.RAIL_SOFT), step_nom, "gp")
    ctrls = [(LABELS[0][0], "nominal", nom), (LABELS[1][0], "gp_mpc", second)]

    plant = IsaacPlant(port=a.port)
    print("isaac plant:", plant.info, flush=True)
    res, eps0 = {}, {}
    for label, key, ctrl in ctrls:
        ms = []
        for i in range(a.n_eval):
            rng = np.random.default_rng(1000 + i)
            if i == 0:
                plant.record(os.path.join(FRAMES, key))
            film = (lambda c, y, r: plant.target(r)) if i == 0 else None
            ep = run_episode(ctrl, plant, x0_eval(rng), rng, on_step=film)
            if i == 0:
                plant.record(None)
                eps0[key] = ep
            ms.append(metrics(ep))
            print(f"  {label:12s} ep{i}: fell={ms[-1]['fell']} settled={ms[-1]['settled_rms_m']:.3f}", flush=True)
        res[key] = summarize(ms)
    plant.close()

    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(dict(n_eval=a.n_eval, gp=a.gp, plant=plant.info, eval=res), f, indent=1)
    with open(os.path.join(a.out, "episode0.pkl"), "wb") as f:
        pickle.dump(eps0, f)
    for key, s in res.items():
        print(f"{key:8s} fell {s['fell']['mean']:.0%}  settled {s['settled_rms_m']['mean']:.3f} m  "
              f"pole RMS {s['theta_rms_rad']['mean']:.3f} rad  swing-up {s['swing_up_s']['mean']:.2f} s")

else:
    eps0 = pickle.load(open(os.path.join(a.out, "episode0.pkl"), "rb"))

# ---------------- stacked GIF from the rendered frames ----------------
try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    small = ImageFont.truetype("DejaVuSans.ttf", 15)
except OSError:
    font = small = ImageFont.load_default()

files = {k: sorted(glob.glob(os.path.join(FRAMES, k, "*.png"))) for _, k in LABELS}
# crop to the working range (+-3.2 m) around the rail, found from its colour
f0 = np.asarray(Image.open(files["gp_mpc"][0]).convert("RGB")).astype(int)
rail = (f0[..., 0] > 170) & (f0[..., 1] > 150) & (f0[..., 2] < 140)
ry, rx = np.nonzero(rail)
row, c0, c1 = int(np.median(ry)), rx.min(), rx.max()
ppm = (c1 - c0) / 8.0                                  # rail is 8 m long
cx = (c0 + c1) / 2
box = (int(cx - 3.2 * ppm), row - int(1.15 * ppm), int(cx + 3.2 * ppm), row + int(0.95 * ppm))


def panel(path, title, t, note):
    im = Image.open(path).convert("RGB").crop(box)
    d = ImageDraw.Draw(im)
    d.text((10, 6), title, fill=(11, 11, 11), font=font)
    if title:
        d.text((im.width - 290, 10), "red block under the rail = setpoint", fill=(82, 81, 78), font=small)
    d.text((10, im.height - 24), f"t = {t:4.1f} s   {note}", fill=(11, 11, 11), font=small)
    return im


n = max(len(v) for v in files.values())
frames = []
for i in range(0, n, 2):
    tiles = []
    for label, key in LABELS:
        fl = files[key]
        j = min(i, len(fl) - 1)
        crashed = eps0[key].get("crashed", False) and i >= len(fl) - 1
        tiles.append(panel(fl[j], label, j * C.DT, "crashed into the rail end" if crashed else ""))
    w, h = tiles[0].size
    canvas = Image.new("RGB", (w, 2 * h + 6), (252, 252, 251))
    canvas.paste(tiles[0], (0, 0))
    canvas.paste(tiles[1], (0, h + 6))
    canvas = canvas.resize((int(w * a.scale), int((2 * h + 6) * a.scale)), Image.LANCZOS)
    frames.append(canvas.quantize(colors=64, method=Image.Quantize.FASTOCTREE))   # keeps the red marker
gif = os.path.join(a.out, "isaac_render.gif")
frames[0].save(gif, save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)
print(f"wrote {gif} ({os.path.getsize(gif) / 1e6:.1f} MB, {len(frames)} frames, {frames[0].size})")

# ---------------- MP4 at full crop size and 25 fps (ffmpeg) ----------------
import shutil
import subprocess
if shutil.which("ffmpeg"):
    w, h = panel(files["gp_mpc"][0], "", 0, "").size
    W2, H2 = w - w % 2, (2 * h + 6) - (2 * h + 6) % 2
    mp4 = os.path.join(a.out, "isaac_render.mp4")
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{W2}x{H2}", "-r", str(int(round(1 / C.DT))), "-i", "-",
                             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", mp4],
                            stdin=subprocess.PIPE)
    for i in range(n):
        tiles = []
        for label, key in LABELS:
            fl = files[key]
            j = min(i, len(fl) - 1)
            crashed = eps0[key].get("crashed", False) and i >= len(fl) - 1
            tiles.append(panel(fl[j], label, j * C.DT, "crashed into the rail end" if crashed else ""))
        canvas = Image.new("RGB", (w, 2 * h + 6), (252, 252, 251))
        canvas.paste(tiles[0], (0, 0))
        canvas.paste(tiles[1], (0, h + 6))
        proc.stdin.write(canvas.crop((0, 0, W2, H2)).tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"wrote {mp4} ({os.path.getsize(mp4) / 1e6:.1f} MB)")
