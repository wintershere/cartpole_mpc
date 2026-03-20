"""Learning the model error and putting it into the MPC.

Model:  x+ = F_nom(x, u) + Bd d(z),   z = [sin th, cos th, vx, omega, u]
  d   = velocity error over one step, [dvx, domega]  (what the data shows
        minus what the nominal model predicts)
  Bd  = [[dt/2, 0], [0, dt/2], [1, 0], [0, 1]]: a constant acceleration
        error a over a step changes velocity by a*dt and position by
        a*dt^2/2 = (a*dt)*dt/2.
Cart position px is left out of z: the dynamics don't depend on where the
cart is on the rail.
"""

import time
import warnings

import casadi as ca
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore", category=ConvergenceWarning)

import config as C
from dynamics import rk4_ca, rk4_np

BD = np.array([[C.DT / 2, 0], [0, C.DT / 2], [1, 0], [0, 1]])


def features_np(x, u):
    x = np.atleast_2d(x)
    u = np.atleast_1d(u).reshape(-1)
    return np.column_stack([np.sin(x[:, 1]), np.cos(x[:, 1]), x[:, 2], x[:, 3], u])


def features_ca(x, u):
    return ca.vertcat(ca.sin(x[1]), ca.cos(x[1]), x[2], x[3], u[0])


def dataset(episodes, p_nom=C.NOMINAL):
    """(z, d) pairs from measured transitions y_k, u_k -> y_k+1."""
    Z, D = [], []
    for ep in episodes:
        y, u = ep["y"], ep["u"]
        for k in range(len(u)):
            if not (np.all(np.isfinite(y[k])) and np.all(np.isfinite(y[k + 1]))):
                continue
            pred = rk4_np(y[k], u[k], p_nom, C.DT)
            D.append((y[k + 1] - pred)[2:])
            Z.append(features_np(y[k], u[k])[0])
    return np.array(Z), np.array(D)


def farthest_points(X, n, rng):
    """Greedy farthest-point subset: spreads the GP's points over the
    visited state space instead of piling them where the robot spends
    most time (balancing)."""
    n = min(n, len(X))
    idx = [int(rng.integers(len(X)))]
    d = np.linalg.norm(X - X[idx[0]], axis=1)
    for _ in range(n - 1):
        i = int(np.argmax(d))
        idx.append(i)
        d = np.minimum(d, np.linalg.norm(X - X[i], axis=1))
    return np.array(idx)


def se_kernel(A, B, c, ell):
    """c * exp(-0.5 |(a - b) / ell|^2) for all row pairs."""
    A, B = A / ell, B / ell
    d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return c * np.exp(-0.5 * np.maximum(d2, 0.0))


class ResidualGP:
    """One GP per output (dvx, domega), ARD squared-exponential kernel,
    zero prior mean (away from data the model falls back to physics).

    Hyperparameters (c, ell, noise) by maximum marginal likelihood on a
    random subset of n_hyper points. The MPC evaluates the mean through
    n_ind inducing points U picked farthest-first, so its cost does not
    grow with the data:
        mu(z) = k(z, U) alpha
    method "dtc": alpha from ALL data (sparse GP, DTC / projected process)
        alpha = (s2 K_uu + K_uf K_fu)^-1 K_uf y
    method "sod": exact GP on the inducing points only (subset of data)
        alpha = (K_uu + s2 I)^-1 y_u
    """

    def __init__(self, n_ind=C.GP_N_INDUCING, n_hyper=C.GP_N_HYPER, seed=0,
                 method="dtc", select="fps"):
        self.n_ind, self.n_hyper, self.rng = n_ind, n_hyper, np.random.default_rng(seed)
        self.method, self.select = method, select

    def fit(self, Z, D):
        t0 = time.perf_counter()
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-9
        Zs = (Z - self.mu) / self.sd
        self.dscale = D.std(0)
        Ds = D / self.dscale
        hyp = self.rng.choice(len(Zs), min(self.n_hyper, len(Zs)), replace=False)
        if self.select == "fps":
            ind = farthest_points(Zs, self.n_ind, self.rng)
        else:
            ind = self.rng.choice(len(Zs), min(self.n_ind, len(Zs)), replace=False)
        self.U = Zs[ind]
        self.c, self.ell, self.s2, self.alpha, self._Ainv, self._Kuu_inv = [], [], [], [], [], []
        self._kern_str = []
        for j in range(D.shape[1]):
            k = (ConstantKernel(1.0, (1e-3, 1e3))
                 * RBF(np.ones(Zs.shape[1]), (1e-2, 1e2))
                 + WhiteKernel(1e-2, (1e-6, 1e1)))
            g = GaussianProcessRegressor(k, normalize_y=False, n_restarts_optimizer=2,
                                         random_state=j).fit(Zs[hyp], Ds[hyp, j])
            c = g.kernel_.k1.k1.constant_value
            ell = np.atleast_1d(g.kernel_.k1.k2.length_scale).astype(float)
            s2 = g.kernel_.k2.noise_level
            Kuu = se_kernel(self.U, self.U, c, ell) + 1e-8 * c * np.eye(len(self.U))
            if self.method == "dtc":
                Kuf = se_kernel(self.U, Zs, c, ell)
                A = s2 * Kuu + Kuf @ Kuf.T
                alpha = np.linalg.solve(A, Kuf @ Ds[:, j])
                self._Ainv.append(np.linalg.inv(A))
            else:
                A = Kuu + s2 * np.eye(len(self.U))
                alpha = np.linalg.solve(A, Ds[ind, j])
                self._Ainv.append(np.linalg.inv(A))
            self._Kuu_inv.append(np.linalg.inv(Kuu))
            self.c.append(c); self.ell.append(ell); self.s2.append(s2); self.alpha.append(alpha)
            self._kern_str.append(str(g.kernel_))
        self.env = self.envelope(Z)
        self.fit_s = time.perf_counter() - t0
        return self

    def predict(self, Z, return_std=False):
        Zs = (np.atleast_2d(Z) - self.mu) / self.sd
        means, stds = [], []
        for j in range(len(self.alpha)):
            Ksu = se_kernel(Zs, self.U, self.c[j], self.ell[j])
            means.append(Ksu @ self.alpha[j])
            if return_std:
                if self.method == "dtc":
                    # DTC: k** - q** + s2 k*u A^-1 ku*   (+ s2 observation noise)
                    q = np.einsum("ij,jk,ik->i", Ksu, self._Kuu_inv[j], Ksu)
                    a = np.einsum("ij,jk,ik->i", Ksu, self._Ainv[j], Ksu)
                    var = self.c[j] - q + self.s2[j] * a
                else:
                    var = self.c[j] - np.einsum("ij,jk,ik->i", Ksu, self._Ainv[j], Ksu)
                stds.append(np.sqrt(np.maximum(var, 0) + self.s2[j]))
        M = np.column_stack(means) * self.dscale
        if return_std:
            return M, np.column_stack(stds) * self.dscale
        return M

    def kernels(self):
        return self._kern_str

    def envelope(self, Z, q=99.5):
        """Soft speed bounds at the edge of the data: the model is only
        trusted where it has seen data."""
        return {2: float(np.percentile(np.abs(Z[:, 2]), q)),
                3: float(np.percentile(np.abs(Z[:, 3]), q))}

    def mean_ca(self, z):
        """GP mean as a CasADi expression (goes into the acados model)."""
        zs = (z - ca.DM(self.mu)) / ca.DM(self.sd)
        outs = []
        for j in range(len(self.alpha)):
            Ui = self.U / self.ell[j]
            zz = zs / ca.DM(self.ell[j])
            diff = ca.repmat(zz.T, Ui.shape[0], 1) - ca.DM(Ui)
            kvec = self.c[j] * ca.exp(-0.5 * ca.sum2(diff ** 2))
            outs.append(self.dscale[j] * ca.mtimes(kvec.T, ca.DM(self.alpha[j])))
        return ca.vertcat(*outs)

    def xnext_ca(self, x, u, p_nom=C.NOMINAL):
        return rk4_ca(x, u, p_nom, C.DT) + ca.DM(BD) @ self.mean_ca(features_ca(x, u))

    def xnext_np(self, x, u, p_nom=C.NOMINAL):
        return rk4_np(x, u, p_nom, C.DT) + BD @ self.predict(features_np(x, u))[0]


def fit_baselines(Z, D, seed=0):
    """Boosting, bagging and a small neural net on the same targets."""
    out = {}
    t0 = time.perf_counter()
    out["boosting (HistGB)"] = [HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                              random_state=seed).fit(Z, D[:, j])
                                for j in range(D.shape[1])]
    t1 = time.perf_counter()
    out["bagging (random forest)"] = [RandomForestRegressor(200, min_samples_leaf=3, n_jobs=4,
                                                            random_state=seed).fit(Z, D[:, j])
                                      for j in range(D.shape[1])]
    t2 = time.perf_counter()
    mu, sd = Z.mean(0), Z.std(0) + 1e-9
    dsd = D.std(0) + 1e-12
    mlp = MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh", max_iter=2000, early_stopping=True,
                       random_state=seed).fit((Z - mu) / sd, D / dsd)
    t3 = time.perf_counter()

    class _MLP:
        def predict(self, Zq):
            return mlp.predict((Zq - mu) / sd) * dsd

    out["neural net (MLP 64x64)"] = _MLP()
    times = {"boosting (HistGB)": t1 - t0, "bagging (random forest)": t2 - t1,
             "neural net (MLP 64x64)": t3 - t2}
    return out, times


def predict_any(model, Z):
    if isinstance(model, list):
        return np.column_stack([m.predict(Z) for m in model])
    return model.predict(Z)
