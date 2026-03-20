"""Isaac Sim cart-pole as a plant server.

Runs under Isaac Sim's own Python; the MPC and GP run in the project venv
and talk to this process over a local socket (one JSON line per request),
the way a controller talks to a robot. Isaac's Python needs nothing extra.

    cd ~/isaacsim6 && ROS_DOMAIN_ID=77 ./python.sh ~/cartpole_mpc/gp_mpc/isaac_server.py

Requests:  {"cmd": "info"} | {"cmd": "reset", "x": [px, th, vx, w]}
           {"cmd": "step", "u": F} | {"cmd": "quit"}
           {"cmd": "record", "dir": path or null}   (--render: one RTX
           frame per control step saved as PNG; null stops)
           {"cmd": "target", "px": r}   (--render: move the red setpoint
           marker; visual only, no physics)
Reply:     {"x": [px, th, vx, w]} (or info fields)

The plant is NVIDIA's cart-pole asset. Its masses, inertia and pole
geometry are not the textbook point-mass model the MPC uses, so the
mismatch is real, not hand-picked. --gear, --rail-viscous and
--rail-coulomb add a worn motor and rail friction on top.
"""

import argparse
import json
import socket

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=50555)
ap.add_argument("--ctrl-dt", type=float, default=0.04)
ap.add_argument("--physics-dt", type=float, default=0.005)
ap.add_argument("--gui", action="store_true")
ap.add_argument("--render", action="store_true", help="RTX rendering + frame recording")
ap.add_argument("--no-scene", action="store_true", help="debug: no ground plane or light")
ap.add_argument("--scene-only", action="store_true", help="debug: ground + light, no rendering")
ap.add_argument("--width", type=int, default=960)
ap.add_argument("--height", type=int, default=540)
ap.add_argument("--gear", type=float, default=1.0, help="actuator gain (worn motor < 1)")
ap.add_argument("--rail-viscous", type=float, default=0.0, help="rail viscous friction, N s/m")
ap.add_argument("--rail-coulomb", type=float, default=0.0, help="rail Coulomb friction, N")
ap.add_argument("--delay-steps", type=int, default=0, help="dead time: commands act this many control steps late")
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": not args.gui, "width": args.width, "height": args.height})
RENDER = (args.render or args.gui) and not args.scene_only

import numpy as np  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402


def cartpole_usd():
    try:
        from isaacsim.storage.native import get_assets_root_path
        return get_assets_root_path() + "/Isaac/Robots/IsaacSim/Cartpole/cartpole.usd"
    except Exception:
        return ("https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
                "Assets/Isaac/5.1/Isaac/Robots/IsaacSim/Cartpole/cartpole.usd")


world = World(stage_units_in_meters=1.0, physics_dt=args.physics_dt,
              rendering_dt=args.ctrl_dt)
add_reference_to_stage(usd_path=cartpole_usd(), prim_path="/World/CartPole")
if (RENDER or args.scene_only) and not args.no_scene:
    # filming only: a ground plane 1 m below the rail and a light. The
    # cart-pole stays exactly where the asset puts it (moving the
    # articulation changed its dynamics), so physics match headless runs.
    from isaacsim.core.utils.prims import create_prim
    world.scene.add_default_ground_plane(z_position=-1.0)
    create_prim("/World/DomeLight", "DomeLight", attributes={"inputs:intensity": 800.0})
cp = world.scene.add(SingleArticulation(prim_path="/World/CartPole", name="cartpole"))
world.reset()
ann, rec_dir, rec_i = None, None, 0
target_op = None
if RENDER:
    # red setpoint marker under the rail: plain USD geometry, no physics APIs
    import omni.usd
    from pxr import Gf, UsdGeom
    _cube = UsdGeom.Cube.Define(omni.usd.get_context().get_stage(), "/World/SetpointMarker")
    _cube.CreateDisplayColorAttr([Gf.Vec3f(0.89, 0.29, 0.28)])
    _xf = UsdGeom.XformCommonAPI(_cube)
    _xf.SetScale(Gf.Vec3f(0.035, 0.035, 0.11))
    _xf.SetTranslate(Gf.Vec3d(0.0, 0.0, -0.2))
    target_op = _xf
    from isaacsim.core.utils.viewports import set_camera_view
    set_camera_view(eye=[7.5, 0.0, 0.7], target=[0.0, 0.0, 0.15])   # rail runs along world y
if args.render:
    import omni.replicator.core as orep
    from PIL import Image
    _rp = orep.create.render_product("/OmniverseKit_Persp", (args.width, args.height))
    ann = orep.AnnotatorRegistry.get_annotator("rgb")
    ann.attach([_rp])
names = list(cp.dof_names)
SL, HI = names.index("slider_to_cart"), names.index("cart_to_pole")
n_sub = max(1, round(args.ctrl_dt / args.physics_dt))



def info():
    out = dict(dof_names=names, n_sub=n_sub, physics_dt=args.physics_dt, ctrl_dt=args.ctrl_dt,
               gear=args.gear, rail_viscous=args.rail_viscous, rail_coulomb=args.rail_coulomb,
               delay_steps=args.delay_steps)
    try:
        v = cp._articulation_view
        out["body_names"] = list(v.body_names)
        out["body_masses"] = [float(m) for m in np.asarray(v.get_body_masses()).ravel()]
        out["dof_limits"] = np.asarray(v.get_dof_limits()).reshape(-1, 2).tolist()
    except Exception as e:
        out["note"] = f"no body info: {e}"
    return out


def state():
    p, v = cp.get_joint_positions(), cp.get_joint_velocities()
    return [float(p[SL]), float(p[HI]), float(v[SL]), float(v[HI])]


def reset(x):
    pos = np.zeros(len(names))
    vel = np.zeros(len(names))
    pos[SL], pos[HI], vel[SL], vel[HI] = x
    cp.set_joint_positions(pos)
    cp.set_joint_velocities(vel)
    cp.apply_action(ArticulationAction(joint_efforts=np.zeros(len(names))))
    queue[:] = [0.0] * args.delay_steps
    if RENDER:
        world.render()
    return state()


queue = [0.0] * args.delay_steps            # commands still in the pipe


def step(u):
    """Hidden plant effects: dead time (whole control steps), then every
    physics substep actuator gain and rail friction (viscous + Coulomb,
    smoothed over 1 cm/s)."""
    queue.append(float(u))
    u = queue.pop(0)
    eff = np.zeros(len(names))
    for k in range(n_sub):
        v = float(cp.get_joint_velocities()[SL])
        eff[SL] = (args.gear * float(u) - args.rail_viscous * v
                   - args.rail_coulomb * np.tanh(v / 0.01))
        cp.apply_action(ArticulationAction(joint_efforts=eff))
        world.step(render=False)
    if RENDER:
        # render only; world.step(render=True) would also advance physics
        # by a whole rendering_dt (8 substeps), not by one
        world.render()
    if rec_dir is not None and ann is not None:
        save_frame()
    return state()


def save_frame():
    global rec_i
    img = np.asarray(ann.get_data())
    if img.size:
        Image.fromarray(img[..., :3]).save(f"{rec_dir}/{rec_i:04d}.png")
        rec_i += 1


def record(path):
    global rec_dir, rec_i
    import os
    rec_dir, rec_i = path, 0
    if path:
        os.makedirs(path, exist_ok=True)
        save_frame()
    return {"recording": path}


srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", args.port))
srv.listen(1)
print(f"isaac plant server on 127.0.0.1:{args.port}  {info()}", flush=True)

running = True
while running and app.is_running():
    conn, _ = srv.accept()
    f = conn.makefile("rw")
    for line in f:
        req = json.loads(line)
        cmd = req.get("cmd")
        if cmd == "step":
            rep = {"x": step(req["u"])}
        elif cmd == "reset":
            rep = {"x": reset(req["x"])}
        elif cmd == "info":
            rep = info()
        elif cmd == "record":
            rep = record(req.get("dir"))
        elif cmd == "target":
            if target_op is not None:      # rail (cart px) runs along world y
                target_op.SetTranslate(Gf.Vec3d(0.0, float(req["px"]), -0.2))
            rep = {"ok": True}
        elif cmd == "quit":
            running = False
            rep = {"ok": True}
        else:
            rep = {"error": f"unknown cmd {cmd}"}
        f.write(json.dumps(rep) + "\n")
        f.flush()
        if not running:
            break
    conn.close()

srv.close()
app.close()
