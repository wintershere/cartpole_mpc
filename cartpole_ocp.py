"""
cartpole_ocp.py
===============
Formulates and exports the Optimal Control Problem (OCP) for the cart-pole
swing-up task using acados.

The OCP from the exercise PDF (equation 2):
    min   ∫[0,T] 0.5*x'Qx + 0.5*u'Ru dt  +  0.5*x(T)'Qx(T)
    s.t.  x(0) = x0_bar
          x_dot = f(x, u)
          -Fmax <= u <= Fmax
"""

import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from cartpole_model import export_cartpole_model


def export_cartpole_ocp_solver(
    N: int = 50,
    T: float = 2.0,
    Q: np.ndarray = None,
    R: np.ndarray = None,
    Fmax: float = 80.0,
    use_RTI: bool = False,
) -> AcadosOcpSolver:

    # ----------------------------------------------------------------
    # Cost matrices
    #
    # The original Q=diag(2e3,2e3,...) caused ACADOS_MINSTEP failures
    # because the Gauss-Newton Hessian J'WJ becomes ill-conditioned
    # when the weight ratio between states and controls is ~2e4.
    # HPIPM's internal regularisation can't fix this when the condition
    # number of W itself is already ~2e4.
    #
    # Rule of thumb: keep max(diag(Q))/min(diag(R)) < ~1e3.
    # Q=diag(2e3)/R=1e-1 gives ratio 2e4 — too large.
    # Q=diag(2e2)/R=1e-1 gives ratio 2e3 — borderline.
    # Q=diag(5e1)/R=1e-1 gives ratio 5e2 — safe.
    #
    # For full SQP we use Q=2e2 (stronger, can still swing up well).
    # For RTI the caller passes Q=1e2 (softer, one SQP step is enough).
    # ----------------------------------------------------------------
    if Q is None:
        Q = np.diag([2e2, 2e2, 1e-1, 1e-1])

    if R is None:
        R = 1e-1 * np.eye(1)

    # ----------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------
    model = export_cartpole_model()
    nx = model.x.rows()
    nu = model.u.rows()

    ocp = AcadosOcp()
    ocp.model = model

    # Horizon
    ocp.solver_options.N_horizon = N

    ny   = nx + nu   # 5  (stage output: [x; u])
    ny_e = nx        # 4  (terminal output: x only)

    # ----------------------------------------------------------------
    # Cost: NONLINEAR_LS
    # stage:    0.5 * ||[x;u] - 0||^2_{blockdiag(Q,R)}
    # terminal: 0.5 * ||x - 0||^2_Q
    # ----------------------------------------------------------------
    ocp.cost.cost_type   = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    import casadi as ca
    ocp.model.cost_y_expr   = ca.vertcat(model.x, model.u)
    ocp.model.cost_y_expr_e = model.x

    ocp.cost.W   = np.block([[Q, np.zeros((nx, nu))],
                              [np.zeros((nu, nx)), R]])
    ocp.cost.W_e = Q

    ocp.cost.yref   = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)

    # ----------------------------------------------------------------
    # Constraints
    # ----------------------------------------------------------------
    ocp.constraints.lbu   = np.array([-Fmax])
    ocp.constraints.ubu   = np.array([ Fmax])
    ocp.constraints.idxbu = np.array([0])
    ocp.constraints.x0    = np.array([0.0, np.pi, 0.0, 0.0])

    # ----------------------------------------------------------------
    # Solver options — clean config, no conflicting regularisation
    # ----------------------------------------------------------------
    ocp.solver_options.tf              = T
    ocp.solver_options.integrator_type = "IRK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI" if use_RTI else "SQP"
    ocp.solver_options.hessian_approx  = "GAUSS_NEWTON"

    # PARTIAL_CONDENSING_HPIPM: fast, standard for MPC, has its own
    # internal regularisation — do NOT add CONVEXIFY on top of it.
    ocp.solver_options.qp_solver          = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_iter_max = 100

    # Small Levenberg-Marquardt term only — adds epsilon*I to Hessian.
    # Much gentler than CONVEXIFY. Keeps QP well-posed near equilibrium
    # without distorting the problem structure.
    ocp.solver_options.levenberg_marquardt = 1e-3

    ocp.solver_options.print_level = 0

    if not use_RTI:
        ocp.solver_options.nlp_solver_max_iter = 150

    # ----------------------------------------------------------------
    # Compile
    # ----------------------------------------------------------------
    solver = AcadosOcpSolver(ocp, json_file="cartpole_ocp.json")
    return solver


if __name__ == "__main__":
    print("Building OCP solver...")
    solver = export_cartpole_ocp_solver()
    print("Solver built successfully.")

    x0 = np.array([0.0, np.pi, 0.0, 0.0])
    solver.set(0, "lbx", x0)
    solver.set(0, "ubx", x0)
    status = solver.solve()
    print(f"Solver status: {status}  (0 = success)")
    print(f"u0 = {solver.get(0, 'u')} N")
