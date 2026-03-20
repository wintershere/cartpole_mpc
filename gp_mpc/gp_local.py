"""Local sparse GP for GP-MPC, following Hewing, Kabzan & Zeilinger,
"Cautious Model Predictive Control using Gaussian Process Regression",
IEEE TCST 2020 (arXiv:1705.10702):

  * residual on the velocity states only (Bd), inputs without position;
  * FITC sparse GP whose few inducing points are placed along the MPC's
    previous predicted trajectory, so the model is accurate where the
    controller is about to go; weights are recomputed every step from
    the whole data set;
  * hyperparameters by maximum marginal likelihood.

In acados the inducing inputs and weights are runtime parameters p, so
the solver is generated once and the GP changes every step.
"""

import time
import warnings

import casadi as ca
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

import config as C
from dynamics import rk4_ca, rk4_np
from residual import BD, features_ca, features_np, se_kernel

warnings.filterwarnings("ignore", category=ConvergenceWarning)
NZ = 5


class LocalGP:
    def __init__(self, n_ind=10, n_hyper=C.GP_N_HYPER, n_dict=3000, seed=0):
        self.m, self.n_hyper, self.n_dict = n_ind, n_hyper, n_dict
        self.rng = np.random.default_rng(seed)

    # ---------------- training ----------------
    def fit(self, Z, D):
        t0 = time.perf_counter()
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-9
        Zs = (Z - self.mu) / self.sd
        self.dscale = D.std(0)
        Ds = D / self.dscale
        hyp = self.rng.choice(len(Zs), min(self.n_hyper, len(Zs)), replace=False)
        self.c, self.ell, self.s2, self._kern = [], [], [], []
        for j in range(D.shape[1]):
            k = (ConstantKernel(1.0, (1e-3, 1e3)) * RBF(np.ones(NZ), (1e-2, 1e2))
                 + WhiteKernel(1e-2, (1e-6, 1e1)))
            g = GaussianProcessRegressor(k, normalize_y=False, n_restarts_optimizer=2,
                                         random_state=j).fit(Zs[hyp], Ds[hyp, j])
            self.c.append(g.kernel_.k1.k1.constant_value)
            self.ell.append(np.atleast_1d(g.kernel_.k1.k2.length_scale).astype(float))
            self.s2.append(g.kernel_.k2.noise_level)
            self._kern.append(str(g.kernel_))
        dct = self.rng.choice(len(Zs), min(self.n_dict, len(Zs)), replace=False)
        self.Zd, self.Dd = Zs[dct], Ds[dct]
        self.Zall = Z
        self.fit_s = time.perf_counter() - t0
        return self

    def kernels(self):
        return self._kern

    # ---------------- FITC on given inducing inputs ----------------
    def fitc(self, U):
        """U: m x 5 scaled inducing inputs. Returns per-output
        (alpha, Kuu^-1, Sigma^-1) for the FITC posterior."""
        out = []
        for j in range(len(self.c)):
            c, ell, s2 = self.c[j], self.ell[j], self.s2[j]
            Kuu = se_kernel(U, U, c, ell) + 1e-6 * c * np.eye(len(U))
            Kuf = se_kernel(U, self.Zd, c, ell)
            Kuu_inv = np.linalg.inv(Kuu)
            qff = np.einsum("ij,ik,kj->j", Kuf, Kuu_inv, Kuf)
            lam = np.maximum(c - qff, 0) + s2              # FITC diagonal
            KL = Kuf / lam
            Sig = Kuu + KL @ Kuf.T
            Sig_inv = np.linalg.inv(Sig)
            alpha = Sig_inv @ (KL @ self.Dd[:, j])
            out.append((alpha, Kuu_inv, Sig_inv))
        return out

    def inducing_from_traj(self, X, U):
        """Inducing inputs along a predicted trajectory (stages 0..N-1),
        evenly spaced, as in Hewing et al. 2020."""
        idx = np.linspace(0, len(U) - 1, self.m).round().astype(int)
        Zt = features_np(X[idx], U[idx].reshape(-1))
        return (Zt - self.mu) / self.sd

    def params(self, X, U):
        Ui = self.inducing_from_traj(X, U)
        self._last = (Ui, self.fitc(Ui))
        return self.pack(Ui, self._last[1])

    def pack(self, Ui, post):
        return np.concatenate([Ui.ravel()] + [a for a, _, _ in post])

    @property
    def n_p(self):
        return self.m * NZ + 2 * self.m

    # ---------------- prediction (numpy) ----------------
    def predict(self, Z, U=None, return_std=False):
        """Predict with inducing inputs U (scaled); default: inducing
        points from the query itself (dense check of the FITC model)."""
        Zs = (np.atleast_2d(Z) - self.mu) / self.sd
        if U is None:
            U = Zs[np.linspace(0, len(Zs) - 1, min(self.m, len(Zs))).round().astype(int)]
        post = self.fitc(U)
        M, S = [], []
        for j, (alpha, Kuu_inv, Sig_inv) in enumerate(post):
            Ksu = se_kernel(Zs, U, self.c[j], self.ell[j])
            M.append(Ksu @ alpha)
            if return_std:
                v = (self.c[j] - np.einsum("ij,jk,ik->i", Ksu, Kuu_inv, Ksu)
                     + np.einsum("ij,jk,ik->i", Ksu, Sig_inv, Ksu))
                S.append(np.sqrt(np.maximum(v, 0) + self.s2[j]))
        M = np.column_stack(M) * self.dscale
        return (M, np.column_stack(S) * self.dscale) if return_std else M

    # ---------------- CasADi (inside acados, p = runtime params) -------
    def mean_ca(self, z, p):
        zs = (z - ca.DM(self.mu)) / ca.DM(self.sd)
        m = self.m
        outs = []
        for j in range(len(self.c)):
            alpha = p[m * NZ + j * m: m * NZ + (j + 1) * m]
            acc = 0
            for i in range(m):
                d = (zs - p[i * NZ:(i + 1) * NZ]) / ca.DM(self.ell[j])
                acc = acc + alpha[i] * self.c[j] * ca.exp(-0.5 * ca.sumsqr(d))
            outs.append(self.dscale[j] * acc)
        return ca.vertcat(*outs)

    def xnext_ca(self, x, u, p, p_nom=C.NOMINAL):
        return rk4_ca(x, u, p_nom, C.DT) + ca.DM(BD) @ self.mean_ca(features_ca(x, u), p)

    def xnext_np(self, x, u, p, p_nom=C.NOMINAL):
        """Same model in numpy for checks (p as packed by params())."""
        m = self.m
        Ui = p[:m * NZ].reshape(m, NZ)
        zs = (features_np(x, u) - self.mu) / self.sd
        d = []
        for j in range(len(self.c)):
            alpha = p[m * NZ + j * m: m * NZ + (j + 1) * m]
            d.append(float(se_kernel(zs, Ui, self.c[j], self.ell[j]) @ alpha) * self.dscale[j])
        return rk4_np(x, u, p_nom, C.DT) + BD @ np.array(d)
