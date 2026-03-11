"""
cartpole_model.py
=================
Defines the cart-pole system dynamics as an AcadosModel.

STATE:  x = [px, theta, vx, omega]
         px    = cart horizontal position (m)
         theta = pendulum angle (rad), 0 = upright
         vx    = cart velocity (m/s)
         omega = pendulum angular velocity (rad/s)

CONTROL: u = F  (horizontal force on cart, Newtons)

The dynamics come directly from equation (1) in the exercise PDF.
"""

import casadi as ca
from acados_template import AcadosModel


def export_cartpole_model() -> AcadosModel:
    """
    Build and return the AcadosModel for the cart-pole system.
    
    acados needs the dynamics in IMPLICIT form:
        f_impl(x, x_dot, u) = x_dot - f(x, u) = 0
    where f(x, u) is the explicit RHS from the PDF.
    """

    model_name = "cartpole"

    # ----------------------------------------------------------------
    # Physical parameters (same as exercise defaults)
    # ----------------------------------------------------------------
    M = 1.0   # cart mass (kg)
    m = 0.1   # pendulum tip mass (kg)
    l = 0.8   # pendulum length (m)
    g = 9.81  # gravitational acceleration (m/s^2)

    # ----------------------------------------------------------------
    # CasADi symbolic state variables
    # Each is a scalar symbol; we'll stack them into a vector below.
    # ----------------------------------------------------------------
    px    = ca.SX.sym("px")     # cart position
    theta = ca.SX.sym("theta")  # pendulum angle
    vx    = ca.SX.sym("vx")     # cart velocity
    omega = ca.SX.sym("omega")  # pendulum angular velocity

    # Stack into state vector x (4x1)
    x = ca.vertcat(px, theta, vx, omega)

    # Symbolic derivatives (what acados calls x_dot)
    px_dot    = ca.SX.sym("px_dot")
    theta_dot = ca.SX.sym("theta_dot")
    vx_dot    = ca.SX.sym("vx_dot")
    omega_dot = ca.SX.sym("omega_dot")

    x_dot = ca.vertcat(px_dot, theta_dot, vx_dot, omega_dot)

    # ----------------------------------------------------------------
    # Control input
    # ----------------------------------------------------------------
    F = ca.SX.sym("F")   # force applied to cart (N)
    u = ca.vertcat(F)

    # ----------------------------------------------------------------
    # Explicit dynamics: x_dot = f(x, u)
    # Taken directly from equation (1) in the exercise PDF.
    #
    # Common denominator: D = M + m*(1 - cos^2(theta))
    # Note: 1 - cos^2(θ) = sin^2(θ), but we keep the PDF form exactly.
    # ----------------------------------------------------------------
    D = M + m * (1 - ca.cos(theta)**2)

    # d(px)/dt = vx  (trivial - velocity is derivative of position)
    f_px = vx

    # d(theta)/dt = omega  (trivial)
    f_theta = omega

    # d(vx)/dt  — translational acceleration of cart (from PDF eq. 1)
    f_vx = (-m * l * ca.sin(theta) * omega**2
             + m * g * ca.cos(theta) * ca.sin(theta)
             + F) / D

    # d(omega)/dt — angular acceleration of pendulum (from PDF eq. 1)
    f_omega = (-m * l * ca.cos(theta) * ca.sin(theta) * omega**2
               + F * ca.cos(theta)
               + (M + m) * g * ca.sin(theta)) / (l * D)

    # Stack into f(x, u)
    f_expl = ca.vertcat(f_px, f_theta, f_vx, f_omega)

    # ----------------------------------------------------------------
    # Implicit dynamics for acados:
    #   f_impl = x_dot - f(x, u) = 0
    # acados solves this internally using Runge-Kutta integrators.
    # ----------------------------------------------------------------
    f_impl = x_dot - f_expl

    # ----------------------------------------------------------------
    # Build the AcadosModel object
    # ----------------------------------------------------------------
    model = AcadosModel()
    model.name = model_name

    model.x    = x        # state vector (4x1)
    model.xdot = x_dot    # symbolic derivatives (needed for implicit form)
    model.u    = u        # control vector (1x1)

    model.f_expl_expr = f_expl   # explicit RHS: used by ERK integrators
    model.f_impl_expr = f_impl   # implicit form: used by IRK integrators (default in acados)

    return model


if __name__ == "__main__":
    # Quick sanity check — just make sure it builds without errors
    model = export_cartpole_model()
    print(f"Model '{model.name}' created successfully.")
    print(f"  States:  {model.x}")
    print(f"  Control: {model.u}")
    print(f"  f_expl:  {model.f_expl_expr}")
