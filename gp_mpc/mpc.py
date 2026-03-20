"""acados MPC with a discrete-time model x+ = F(x, u).

The model is passed in as a CasADi expression builder, so the same code
builds the nominal MPC, the oracle MPC and the GP-MPC:
    nominal:  F = RK4(f_nominal)
    GP-MPC:   F = RK4(f_nominal) + Bd * mu_GP(z(x, u))
"""

import os
import time

import casadi as ca
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

import config as C


def build_solver(name, xnext_builder, rail=None, rti=True, slack_w=1e3, envelope=None, n_p=0):
    """xnext_builder(x, u) -> CasADi SX expression of the next state.
    rail: optional |px| limit as a soft constraint (per-stage bounds can be
    tightened later with set_rail).
    envelope: optional {state index: limit} soft bounds, e.g. the speeds
    the learning data covers (a trust region for a learned model).
    n_p > 0: the model has runtime parameters p and the builder is
    xnext_builder(x, u, p) (used by the local GP)."""
    x = ca.SX.sym("x", 4)
    u = ca.SX.sym("u", 1)

    model = AcadosModel()
    model.name = name
    model.x, model.u = x, u
    if n_p:
        p = ca.SX.sym("p", n_p)
        model.p = p
        model.disc_dyn_expr = xnext_builder(x, u, p)
    else:
        model.disc_dyn_expr = xnext_builder(x, u)

    ocp = AcadosOcp()
    ocp.model = model
    if n_p:
        ocp.parameter_values = np.zeros(n_p)
    N = C.N_HORIZON
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = N * C.DT

    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    model.cost_y_expr = ca.vertcat(x, u)
    model.cost_y_expr_e = x
    ocp.cost.W = ca.diagcat(ca.DM(C.Q), ca.DM(C.R)).full()
    ocp.cost.W_e = C.Q
    ocp.cost.yref = np.zeros(5)
    ocp.cost.yref_e = np.zeros(4)

    ocp.constraints.lbu = np.array([-C.FMAX])
    ocp.constraints.ubu = np.array([C.FMAX])
    ocp.constraints.idxbu = np.array([0])
    ocp.constraints.x0 = np.array([0.0, np.pi, 0.0, 0.0])

    bounds = {}
    if rail is not None:
        bounds[0] = rail
    if envelope:
        bounds.update(envelope)
    if bounds:
        idx = np.array(sorted(bounds))
        lim = np.array([bounds[i] for i in idx], float)
        ns = len(idx)
        for s in ("", "_e"):
            setattr(ocp.constraints, "idxbx" + s, idx)
            setattr(ocp.constraints, "lbx" + s, -lim)
            setattr(ocp.constraints, "ubx" + s, lim)
            setattr(ocp.constraints, "idxsbx" + s, np.arange(ns))
            for z in ("zl", "zu", "Zl", "Zu"):
                setattr(ocp.cost, z + s, slack_w * np.ones(ns))

    so = ocp.solver_options
    so.integrator_type = "DISCRETE"
    so.nlp_solver_type = "SQP_RTI" if rti else "SQP"
    so.hessian_approx = "GAUSS_NEWTON"
    so.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    so.qp_solver_iter_max = 100
    so.levenberg_marquardt = 1e-3
    so.print_level = 0
    if not rti:
        so.nlp_solver_max_iter = 100

    os.makedirs(C.BUILD_DIR, exist_ok=True)
    ocp.code_export_directory = os.path.join(C.BUILD_DIR, f"c_{name}")
    return AcadosOcpSolver(ocp, json_file=os.path.join(C.BUILD_DIR, f"{name}.json"),
                           verbose=False)


class MPC:
    """RTI loop: feedback phase on the new measurement, then shift and
    preparation for the next step (as in the original repo)."""

    def __init__(self, solver, rollout_step, name, rail=None, param_cb=None):
        """param_cb(X_guess, U_guess) -> p, called before each preparation
        phase with the shifted guess (local GP: inducing points along the
        previous predicted trajectory)."""
        self.s, self.rollout_step, self.name, self.rail = solver, rollout_step, name, rail
        self.param_cb = param_cb
        self.N = C.N_HORIZON
        self.t_param = 0.0

    def _update_params(self):
        if self.param_cb is None:
            return
        t0 = time.perf_counter()
        X, U = self.predicted()
        p = self.param_cb(X, U)
        for k in range(self.N + 1):
            self.s.set(k, "p", p)
        self.t_param = time.perf_counter() - t0

    def reset(self, x0):
        x = np.array(x0, float)
        for k in range(self.N):
            self.s.set(k, "x", x)
            self.s.set(k, "u", np.zeros(1))
            x = self.rollout_step(x, 0.0)
        self.s.set(self.N, "x", x)
        self._set_x0(x0)
        self._update_params()
        self.s.options_set("rti_phase", 1)
        self.s.solve()

    def _set_x0(self, x):
        self.s.set(0, "lbx", x)
        self.s.set(0, "ubx", x)

    def set_ref(self, px_ref):
        y = np.array([px_ref, 0, 0, 0, 0.0])
        for k in range(self.N):
            self.s.cost_set(k, "yref", y)
        self.s.cost_set(self.N, "yref", y[:4])

    def set_rail(self, lo, hi):
        """Per-stage px bounds for k = 1..N (used by the cautious variant)."""
        for k in range(1, self.N):
            self.s.constraints_set(k, "lbx", np.array([lo[k]]))
            self.s.constraints_set(k, "ubx", np.array([hi[k]]))
        self.s.constraints_set(self.N, "lbx", np.array([lo[self.N]]))
        self.s.constraints_set(self.N, "ubx", np.array([hi[self.N]]))

    def predicted(self):
        X = np.array([self.s.get(k, "x") for k in range(self.N + 1)])
        U = np.array([self.s.get(k, "u") for k in range(self.N)])
        return X, U

    def __call__(self, x, px_ref):
        self.set_ref(px_ref)
        self._set_x0(x)
        t0 = time.perf_counter()
        self.s.options_set("rti_phase", 2)
        status = self.s.solve()
        t_fb = time.perf_counter() - t0
        u0 = float(self.s.get(0, "u")[0])
        # shift + preparation for next step
        for k in range(self.N - 1):
            self.s.set(k, "x", self.s.get(k + 1, "x"))
            self.s.set(k, "u", self.s.get(k + 1, "u"))
        self.s.set(self.N - 1, "x", self.s.get(self.N, "x"))
        t0 = time.perf_counter()
        self._update_params()
        self.s.options_set("rti_phase", 1)
        self.s.solve()
        t_prep = time.perf_counter() - t0
        return u0, status, t_fb, t_prep
