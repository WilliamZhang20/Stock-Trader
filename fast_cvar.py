"""
Fast enhanced-CVaR portfolio solver via CVXPYgen-compiled C code.

The trader's enhanced-CVaR objective is recast as a DPP-compliant parametric
problem so CVXPYgen can emit a compiled solver that is re-used across the
hundreds of solves in a walk-forward backtest. Falls back transparently to
plain CVXPY/ECOS if code generation or compilation is unavailable.

DPP notes:
  * The regime lambdas (lambda_ret, lambda_momentum) are folded into ONE linear
    coefficient `lin = lambda_ret*weighted_mean + lambda_momentum*momentum` that
    is computed in Python and passed as a single Parameter, avoiding illegal
    parameter*parameter products.
  * Turnover uses an auxiliary variable `delta == w - w_prev` (a standard DPP
    reformulation) so the L1 term stays parameter-free.
"""
from __future__ import annotations

import contextlib
import os
import platform
import sys
from pathlib import Path

import numpy as np
import cvxpy as cp

# Windows Store Python stubs break juliapkg's libc probe during cvxpygen import.
_orig_libc_ver = platform.libc_ver


def _safe_libc_ver(executable=sys.executable, lib="", version="", chunksize=16384):
    try:
        return _orig_libc_ver(executable, lib, version, chunksize)
    except OSError:
        return ("", "")


platform.libc_ver = _safe_libc_ver

# Match the constraint constants used by cvar_trader.solve_enhanced_cvar_portfolio
MAX_WEIGHT = 0.2
MAX_INVEST = 0.99

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Cache of compiled solvers keyed by (N, T, alpha)
_CACHE: dict[tuple, dict] = {}
_DISABLED = False  # set True after a hard failure so we stop retrying


@contextlib.contextmanager
def _chdir(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _build_problem(N: int, T: int, alpha: float):
    """Construct the DPP-compliant parametric enhanced-CVaR problem."""
    w = cp.Variable(N, name="w")
    z = cp.Variable(name="z")
    u = cp.Variable(T, name="u")
    delta = cp.Variable(N, name="delta")

    R = cp.Parameter((T, N), name="R")
    lin = cp.Parameter(N, name="lin")
    lam_cvar = cp.Parameter(nonneg=True, name="lam_cvar")
    lam_turn = cp.Parameter(nonneg=True, name="lam_turn")
    w_prev = cp.Parameter(N, name="w_prev")

    k = 1.0 / ((1.0 - alpha) * T)
    cvar_expr = z + k * cp.sum(u)

    objective = cp.Maximize(
        lin @ w
        - lam_cvar * cvar_expr
        - lam_turn * cp.norm1(delta)
    )
    constraints = [
        u >= -(R @ w) - z,
        u >= 0,
        z >= 0,
        delta == w - w_prev,
        cp.sum(w) == MAX_INVEST,
        w >= 0,
        w <= MAX_WEIGHT,
    ]
    prob = cp.Problem(objective, constraints)
    params = dict(R=R, lin=lin, lam_cvar=lam_cvar, lam_turn=lam_turn, w_prev=w_prev, w=w)
    return prob, params


def _get_solver(N: int, T: int, alpha: float):
    """Return (problem, params) with a registered compiled solver, or None on failure."""
    global _DISABLED
    key = (N, T, round(alpha, 4))
    if key in _CACHE:
        return _CACHE[key]
    if _DISABLED:
        return None

    prob, params = _build_problem(N, T, alpha)
    # CVXPYgen treats code_dir as the generated package's module name, so it
    # must be a bare relative name resolvable from the project root.
    mod_name = f"enhanced_cvar_code_{N}_{T}"

    def _register_from(module):
        prob.register_solve("CPG", module.cpg_solve)

    # Try to reuse a previously compiled solver first.
    try:
        module = __import__(f"{mod_name}.cpg_solver", fromlist=["cpg_solve"])
        _register_from(module)
        _CACHE[key] = (prob, params)
        return _CACHE[key]
    except Exception:
        pass

    # Generate + compile (from the project root so the package is importable).
    try:
        from cvxpygen import cpg
        print(f"[fast_cvar] compiling solver for N={N}, T={T} (one-time, ~30-60s)...")
        # CVXPYgen builds the C-extension via `pip ... --target`, which conflicts
        # with a global `user=true` pip config on this machine. Force it off.
        prev_pip_user = os.environ.get("PIP_USER")
        os.environ["PIP_USER"] = "0"
        try:
            with _chdir(_PROJECT_ROOT):
                cpg.generate_code(prob, code_dir=mod_name, solver="ECOS", wrapper=True)
        finally:
            if prev_pip_user is None:
                os.environ.pop("PIP_USER", None)
            else:
                os.environ["PIP_USER"] = prev_pip_user
        module = __import__(f"{mod_name}.cpg_solver", fromlist=["cpg_solve"])
        _register_from(module)
        _CACHE[key] = (prob, params)
        print(f"[fast_cvar] solver ready: {mod_name}")
        return _CACHE[key]
    except Exception as e:
        print(f"[fast_cvar] code generation failed ({type(e).__name__}: {e}); "
              f"falling back to plain CVXPY for the rest of this run.")
        _DISABLED = True
        return None


def available(N: int, T: int, alpha: float = 0.95) -> bool:
    return _get_solver(N, T, alpha) is not None


def solve(returns_values, lin_vec, lam_cvar, lam_turn, w_prev, alpha=0.95):
    """
    Solve the enhanced-CVaR problem with the compiled solver.

    Returns the optimal weight vector, or None if the fast solver is unavailable
    (caller should fall back to plain CVXPY).
    """
    T, N = returns_values.shape
    got = _get_solver(N, T, alpha)
    if got is None:
        return None
    prob, params = got

    params["R"].value = np.ascontiguousarray(returns_values, dtype=float)
    params["lin"].value = np.asarray(lin_vec, dtype=float)
    params["lam_cvar"].value = float(lam_cvar)
    params["lam_turn"].value = float(lam_turn)
    params["w_prev"].value = np.asarray(w_prev, dtype=float)

    try:
        prob.solve(method="CPG")
    except Exception as e:
        print(f"[fast_cvar] compiled solve failed ({e}); caller should fall back.")
        return None

    return None if params["w"].value is None else np.asarray(params["w"].value)
