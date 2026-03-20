"""Cart-pole dynamics with optional damping and actuator gain.

State x = [px, theta, vx, omega], theta = 0 upright.
Input u = commanded cart force F (N).

Lagrangian with tip x = p - l sin(theta), z = l cos(theta):
  D     = M + m sin^2(theta)
  Q_p   = gear*F - b_c*vx            (generalised force on the cart)
  Q_th  = -b_p*omega                 (torque on the hinge)
  p_dd  = (Q_p - m l sin(th) w^2 + m g sin(th) cos(th) + cos(th) Q_th / l) / D
  th_dd = (Q_th / (m l) + g sin(th) + cos(th) p_dd) / l
With b_c = b_p = 0 and gear = 1 this is exactly the exercise model.
"""

import casadi as ca
import numpy as np


def _accel(th, vx, w, F, p, sin, cos):
    M, m, l, g = p["M"], p["m"], p["l"], p["g"]
    Qp = p.get("gear", 1.0) * F - p.get("b_c", 0.0) * vx
    Qth = -p.get("b_p", 0.0) * w
    D = M + m * sin(th) ** 2
    p_dd = (Qp - m * l * sin(th) * w ** 2 + m * g * sin(th) * cos(th)
            + cos(th) * Qth / l) / D
    th_dd = (Qth / (m * l) + g * sin(th) + cos(th) * p_dd) / l
    return p_dd, th_dd


def f_np(x, u, p):
    _, th, vx, w = x
    p_dd, th_dd = _accel(th, vx, w, float(u), p, np.sin, np.cos)
    return np.array([vx, w, p_dd, th_dd])


def rk4_np(x, u, p, dt, substeps=2):
    h = dt / substeps
    for _ in range(substeps):
        k1 = f_np(x, u, p)
        k2 = f_np(x + h / 2 * k1, u, p)
        k3 = f_np(x + h / 2 * k2, u, p)
        k4 = f_np(x + h * k3, u, p)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def f_ca(x, u, p):
    th, vx, w = x[1], x[2], x[3]
    p_dd, th_dd = _accel(th, vx, w, u[0], p, ca.sin, ca.cos)
    return ca.vertcat(vx, w, p_dd, th_dd)


def rk4_ca(x, u, p, dt, substeps=2):
    h = dt / substeps
    for _ in range(substeps):
        k1 = f_ca(x, u, p)
        k2 = f_ca(x + h / 2 * k1, u, p)
        k3 = f_ca(x + h / 2 * k2, u, p)
        k4 = f_ca(x + h * k3, u, p)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def step_fn(p, dt, substeps=2):
    """CasADi function x_next = F(x, u) for a parameter set."""
    x = ca.SX.sym("x", 4)
    u = ca.SX.sym("u", 1)
    return ca.Function("F", [x, u], [rk4_ca(x, u, p, dt, substeps)])
