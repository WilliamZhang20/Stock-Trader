"""
Black-Litterman mean-variance portfolio model.

Standalone module preserved from the former `hft_in_cvx/portfolio_models/`
exploration. It implements the Black-Litterman posterior expected returns and a
DPP-compliant factor-model QP. Not currently wired into a live trader, but kept
as a reusable building block (it pairs naturally with mean_variance_trader.py:
feed `black_litterman_posterior(...)` as the `mu` input to a mean-variance solve).
"""
from __future__ import annotations

import numpy as np
import cvxpy as cp


def black_litterman_posterior(
    Sigma: np.ndarray,
    w_mkt: np.ndarray,
    P: np.ndarray,
    q: np.ndarray,
    Omega: np.ndarray,
    tau: float = 0.05,
) -> np.ndarray:
    """Posterior expected returns pi_bl (He-Litterman style)."""
    n = Sigma.shape[0]
    pi = tau * Sigma @ w_mkt
    tau_sigma_inv = np.linalg.inv(tau * Sigma)
    omega_inv = np.linalg.inv(Omega)
    middle = np.linalg.inv(tau_sigma_inv + P.T @ omega_inv @ P)
    rhs = tau_sigma_inv @ pi + P.T @ omega_inv @ q
    return middle @ rhs


def build_black_litterman_problem(n: int, m: int):
    """
    Maximize posterior alpha^T w - w^T Sigma w using a factor covariance model.

    Parameters match the CVXPYgen portfolio example (DPP-compliant QP).
    """
    w = cp.Variable(n, name="w")
    delta_w = cp.Variable(n, name="delta_w")
    f = cp.Variable(m, name="f")

    a = cp.Parameter(n, name="a")
    F = cp.Parameter((n, m), name="F")
    sigma_f_root = cp.Parameter((m, m), name="Sigma_f_root")
    d_root = cp.Parameter(n, name="d_root")
    k_tc = cp.Parameter(n, nonneg=True, name="k_tc")
    k_sh = cp.Parameter(n, nonneg=True, name="k_sh")
    w_prev = cp.Parameter(n, name="w_prev")
    L = cp.Parameter(name="L")

    obj = cp.Maximize(
        a @ w
        - cp.sum_squares(sigma_f_root @ f)
        - cp.sum_squares(cp.multiply(d_root, w))
        - k_tc @ cp.abs(delta_w)
        - k_sh @ cp.neg(w)
    )
    constr = [
        f == F.T @ w,
        cp.sum(w) == 1,
        cp.norm1(w) <= L,
        delta_w == w - w_prev,
    ]
    problem = cp.Problem(obj, constr)
    return problem, {
        "w": w,
        "a": a,
        "F": F,
        "sigma_f_root": sigma_f_root,
        "d_root": d_root,
        "k_tc": k_tc,
        "k_sh": k_sh,
        "w_prev": w_prev,
        "L": L,
    }
