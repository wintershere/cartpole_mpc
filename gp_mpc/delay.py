"""Dead time (and motor lag) for GP-MPC.

1. identify(): find the dead time d (in control steps) and the motor time
   constant tau from logged data. For each (d, tau) on a small grid:
   rebuild the force that actually reached the cart, least-squares fit
   the measured cart acceleration on a few physical terms, and keep the
   pair with the smallest fit error.

2. AugModel: the MPC's model with the delay built in, exactly. The state
   is extended with the commands that are still "in the pipe":
       xa = [px, th, vx, w,  f,  b1 .. bd]
       f        force acting on the cart (motor state; equals the command
                when tau = 0)
       b1..bd   the last d commands, b1 newest; bd reaches the motor now
   One step with command u:
       c   = bd  (u if d = 0)                  command leaving the delay line
       f+  = a f + (1 - a) c,   a = exp(-dt/tau)    motor lag
       F   = average force over the step
       x+  = RK4_nom(x, F) + Bd GP(z),   z = [sin th, cos th, vx, w, F]
       b+  = [u, b1 .. b(d-1)]
   The MPC then plans knowing that what it sends now acts d steps later.
"""

import os
import time

import casadi as ca
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

import config as C
from dynamics import rk4_ca, rk4_np
from residual import BD


# ---------------------------------------------------------------- identify
def _avg_force(f, c, a, tau, dt=C.DT):
    """Exact average over a step of a first-order lag that starts at f and
    moves towards c: c + (f - c) * (tau / dt) * (1 - a)."""
    return c + (f - c) * (tau / dt) * (1 - a)


def applied_force(u, d, tau, dt=C.DT):
    """Average force on the cart in each step, for a command sequence u
    (one episode), dead time d steps and motor lag tau."""
    a = np.exp(-dt / tau) if tau > 0 else 0.0
    f, out = 0.0, np.zeros(len(u))
    for k in range(len(u)):
        c = u[k - d] if k - d >= 0 else 0.0
        out[k] = _avg_force(f, c, a, tau, dt) if tau > 0 else c
        f = a * f + (1 - a) * c
    return out


def identify(episodes, d_grid=range(0, 6), tau_grid=(0.0, 0.02, 0.035, 0.05, 0.07, 0.1, 0.15)):
    """For each (d, tau): least-squares fit of the measured cart
    acceleration on
        F, vx, tanh(vx / 0.02), sin(th) w^2, sin(th) cos(th), 1
    (F = applied force under that d, tau). The fitted coefficients absorb
    the unknown motor gain, friction and masses, so they don't bias the
    delay. The (d, tau) with the smallest fit error wins."""
    scores = {}
    for d in d_grid:
        for tau in tau_grid:
            rows, tgt = [], []
            for ep in episodes:
                y, F = ep["y"], applied_force(ep["u"], d, tau)
                for k in range(len(F)):
                    if not (np.all(np.isfinite(y[k])) and np.all(np.isfinite(y[k + 1]))):
                        continue
                    th, vx, w = y[k, 1], y[k, 2], y[k, 3]
                    rows.append([F[k], vx, np.tanh(vx / 0.02), np.sin(th) * w * w,
                                 np.sin(th) * np.cos(th), 1.0])
                    tgt.append((y[k + 1, 2] - y[k, 2]) / C.DT)
            A, b = np.array(rows), np.array(tgt)
            coef, *_ = np.linalg.lstsq(A, b, rcond=None)
            scores[(d, tau)] = float(np.mean((A @ coef - b) ** 2))
    best = min(scores, key=scores.get)
    return best, scores


def residual_dataset(episodes, d, tau, p_nom=C.NOMINAL):
    """(z, velocity error) pairs, with the applied force as the force input."""
    Z, D = [], []
    for ep in episodes:
        y, F = ep["y"], applied_force(ep["u"], d, tau)
        for k in range(len(F)):
            if np.all(np.isfinite(y[k])) and np.all(np.isfinite(y[k + 1])):
                D.append((y[k + 1] - rk4_np(y[k], F[k], p_nom, C.DT))[2:])
                Z.append([np.sin(y[k, 1]), np.cos(y[k, 1]), y[k, 2], y[k, 3], F[k]])
    return np.array(Z), np.array(D)


# ---------------------------------------------------------------- model
class AugModel:
    def __init__(self, d, tau, gp=None, p_nom=C.NOMINAL):
        self.d, self.tau, self.gp, self.p = d, tau, gp, p_nom
        self.a = float(np.exp(-C.DT / tau)) if tau > 0 else 0.0
        self.nx = 4 + 1 + d

    def _force(self, xa, u):
        f = xa[4]
        c = xa[4 + self.d] if self.d > 0 else u
        f_new = self.a * f + (1 - self.a) * c
        F = _avg_force(f, c, self.a, self.tau) if self.tau > 0 else c
        return f_new, F

    def next_ca(self, xa, u):
        x = xa[:4]
        f_new, F = self._force(xa, u[0])
        xn = rk4_ca(x, ca.vertcat(F), self.p, C.DT)
        if self.gp is not None:
            z = ca.vertcat(ca.sin(x[1]), ca.cos(x[1]), x[2], x[3], F)
            xn = xn + ca.DM(BD) @ self.gp.mean_ca(z)
        buf = [u[0]] + [xa[4 + i] for i in range(1, self.d)] if self.d > 0 else []
        return ca.vertcat(xn, f_new, *buf)

    def next_np(self, xa, u):
        xa = np.asarray(xa, float)
        x = xa[:4]
        f_new, F = self._force(xa, float(u))
        xn = rk4_np(x, F, self.p, C.DT)
        if self.gp is not None:
            z = np.array([[np.sin(x[1]), np.cos(x[1]), x[2], x[3], F]])
            xn = xn + BD @ self.gp.predict(z)[0]
        buf = [float(u)] + [xa[4 + i] for i in range(1, self.d)] if self.d > 0 else []
        return np.concatenate([xn, [f_new], buf])


def build_solver(name, model, rail=C.RAIL_SOFT, slack_w=1e3):
    """Same OCP as mpc.build_solver, on the augmented state (cost and rail
    constraint on the first 4 states only)."""
    x = ca.SX.sym("x", model.nx)
    u = ca.SX.sym("u", 1)
    am = AcadosModel()
    am.name = name
    am.x, am.u = x, u
    am.disc_dyn_expr = model.next_ca(x, u)
    am.cost_y_expr = ca.vertcat(x[:4], u)
    am.cost_y_expr_e = x[:4]
    ocp = AcadosOcp()
    ocp.model = am
    N = C.N_HORIZON
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = N * C.DT
    ocp.cost.cost_type = ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W = ca.diagcat(ca.DM(C.Q), ca.DM(C.R)).full()
    ocp.cost.W_e = C.Q
    ocp.cost.yref, ocp.cost.yref_e = np.zeros(5), np.zeros(4)
    ocp.constraints.lbu, ocp.constraints.ubu = np.array([-C.FMAX]), np.array([C.FMAX])
    ocp.constraints.idxbu = np.array([0])
    ocp.constraints.x0 = np.zeros(model.nx)
    for s in ("", "_e"):
        setattr(ocp.constraints, "idxbx" + s, np.array([0]))
        setattr(ocp.constraints, "lbx" + s, np.array([-rail]))
        setattr(ocp.constraints, "ubx" + s, np.array([rail]))
        setattr(ocp.constraints, "idxsbx" + s, np.array([0]))
        for zname in ("zl", "zu", "Zl", "Zu"):
            setattr(ocp.cost, zname + s, np.array([slack_w]))
    so = ocp.solver_options
    so.integrator_type = "DISCRETE"
    so.nlp_solver_type = "SQP_RTI"
    so.hessian_approx = "GAUSS_NEWTON"
    so.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    so.qp_solver_iter_max = 100
    so.levenberg_marquardt = 1e-3
    so.print_level = 0
    os.makedirs(C.BUILD_DIR, exist_ok=True)
    ocp.code_export_directory = os.path.join(C.BUILD_DIR, f"c_{name}")
    return AcadosOcpSolver(ocp, json_file=os.path.join(C.BUILD_DIR, f"{name}.json"), verbose=False)


# ---------------------------------------------------------------- controller
class AugController:
    """MPC on the augmented model. It keeps its own command history (the
    controller knows what it sent) to fill in the extra states."""

    def __init__(self, solver, model, name="aug"):
        self.s, self.m, self.name = solver, model, name
        self.N = C.N_HORIZON

    def _xa(self, y):
        return np.concatenate([y, [self.f], self.hist[:self.m.d]])

    def reset(self, y0):
        self.f, self.hist = 0.0, [0.0] * max(self.m.d, 1)
        xa = self._xa(y0)
        x = xa.copy()
        for k in range(self.N):
            self.s.set(k, "x", x)
            self.s.set(k, "u", np.zeros(1))
            x = self.m.next_np(x, 0.0)
        self.s.set(self.N, "x", x)
        self.s.set(0, "lbx", xa)
        self.s.set(0, "ubx", xa)
        self.s.options_set("rti_phase", 1)
        self.s.solve()

    def __call__(self, y, px_ref):
        xa = self._xa(y)
        yref = np.array([px_ref, 0, 0, 0, 0.0])
        for k in range(self.N):
            self.s.cost_set(k, "yref", yref)
        self.s.cost_set(self.N, "yref", yref[:4])
        self.s.set(0, "lbx", xa)
        self.s.set(0, "ubx", xa)
        t0 = time.perf_counter()
        self.s.options_set("rti_phase", 2)
        status = self.s.solve()
        t_fb = time.perf_counter() - t0
        u0 = float(self.s.get(0, "u")[0])
        self.record(u0)
        for k in range(self.N - 1):
            self.s.set(k, "x", self.s.get(k + 1, "x"))
            self.s.set(k, "u", self.s.get(k + 1, "u"))
        self.s.set(self.N - 1, "x", self.s.get(self.N, "x"))
        t0 = time.perf_counter()
        self.s.options_set("rti_phase", 1)
        self.s.solve()
        return u0, status, t_fb, time.perf_counter() - t0

    def record(self, u_sent):
        """Advance the controller's own copy of the delay line and motor."""
        a = self.m.a
        self._f_prev = self.f
        c = self.hist[self.m.d - 1] if self.m.d > 0 else u_sent
        self.f = a * self.f + (1 - a) * c
        if self.m.d > 0:
            self.hist = [u_sent] + self.hist[:-1]

    def applied(self, u):
        """The plant got u (planned force + exploration noise): fix the copy."""
        if self.m.d > 0:
            self.hist[0] = u
        else:
            self.f = self.m.a * self._f_prev + (1 - self.m.a) * u
