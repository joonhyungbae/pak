"""IPF — Iterative Proportional Fitting.

Iteratively normalize a multidimensional joint distribution (seed) to match
known marginal distributions (targets).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def ipf(
    seed: np.ndarray,
    targets: list[np.ndarray],
    *,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> tuple[np.ndarray, dict]:
    """Multidimensional IPF.

    Args:
        seed: Initial joint distribution (n-d array). Every entry must be positive or 0.
        targets: 1-d array marginal distribution for each axis. len(targets) == seed.ndim.
                 If the target sum for axis i does not equal every marginal sum of seed,
                 it is affected by the other axes' adjustments during iteration. Standard
                 IPF assumes that all targets share the same grand total.
        max_iter: Maximum number of iterations.
        tol: Convergence threshold (|observed - target| / target < tol on every axis).

    Returns:
        (adjusted joint distribution, info dict).
    """
    if seed.ndim != len(targets):
        raise ValueError(f"seed.ndim={seed.ndim} != len(targets)={len(targets)}")
    arr = seed.astype(np.float64).copy()
    if (arr < 0).any():
        raise ValueError("seed must be non-negative")

    # small epsilon on zero cells (smoothing)
    arr = np.where(arr == 0, 1e-12, arr)

    grand_totals = [float(t.sum()) for t in targets]
    if not all(abs(grand_totals[0] - g) < 1e-6 * grand_totals[0] for g in grand_totals):
        logger.warning(
            "target grand totals differ across axes: %s. IPF may not converge cleanly.",
            grand_totals,
        )

    converged = False
    iters = 0
    max_rel_err_history: list[float] = []
    for it in range(1, max_iter + 1):
        iters = it
        max_rel_err = 0.0
        for axis, target in enumerate(targets):
            current = arr.sum(axis=tuple(a for a in range(arr.ndim) if a != axis))
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(current > 0, target / current, 1.0)
            shape = [1] * arr.ndim
            shape[axis] = arr.shape[axis]
            arr = arr * ratio.reshape(shape)
            # convergence check: relative error between every target and marginal
            new_marginal = arr.sum(axis=tuple(a for a in range(arr.ndim) if a != axis))
            err = np.max(np.abs(new_marginal - target) / np.where(target > 0, target, 1.0))
            if err > max_rel_err:
                max_rel_err = err

        max_rel_err_history.append(float(max_rel_err))
        if max_rel_err < tol:
            converged = True
            break

    info = {
        "converged": converged,
        "iterations": iters,
        "final_max_rel_err": max_rel_err_history[-1] if max_rel_err_history else None,
        "tol": tol,
    }
    return arr, info
