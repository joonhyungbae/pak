"""Distribution goodness-of-fit tests (chi-square / KS test / Cramer's V).

Check whether the marginal distributions of generated personas match the grounding
distributions. Used in both Phase 05 (preview) and Phase 07 (full).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class DistributionTest:
    variable: str
    test: str  # "chi-square" | "ks"
    statistic: float
    p_value: float
    effect_size: float  # Cramer's V (categorical) or KS D (continuous)
    n_observed: int
    passed: bool  # p > 0.05 OR effect < 0.05


def chi_square_test(
    observed: dict[str, int], expected_pct: dict[str, float], *, alpha: float = 0.05
) -> DistributionTest:
    """Chi-square goodness-of-fit test for a categorical distribution."""
    keys = list(observed.keys())
    n = sum(observed.values())
    obs_arr = np.array([observed.get(k, 0) for k in keys], dtype=float)
    exp_arr = np.array([expected_pct.get(k, 0.0) for k in keys], dtype=float) * n
    # Drop zero-expected entries (chi-square is undefined for them)
    mask = exp_arr > 0
    obs_arr = obs_arr[mask]
    exp_arr = exp_arr[mask]
    if obs_arr.size == 0:
        return DistributionTest(
            variable="?",
            test="chi-square",
            statistic=0.0,
            p_value=1.0,
            effect_size=0.0,
            n_observed=n,
            passed=True,
        )
    chi2, p = stats.chisquare(obs_arr, exp_arr)
    # Cramer's V (1-dimensional, so sqrt(chi2 / (n * (k-1))))
    k = obs_arr.size
    cramer_v = float(np.sqrt(chi2 / max(n * (k - 1), 1)))
    return DistributionTest(
        variable="?",
        test="chi-square",
        statistic=float(chi2),
        p_value=float(p),
        effect_size=cramer_v,
        n_observed=n,
        passed=(p > alpha) or (cramer_v < 0.05),
    )


def ks_test(
    observed: list[float], expected: list[float], *, alpha: float = 0.05
) -> DistributionTest:
    """KS test for a continuous variable."""
    if not observed or not expected:
        return DistributionTest(
            variable="?",
            test="ks",
            statistic=0.0,
            p_value=1.0,
            effect_size=0.0,
            n_observed=len(observed),
            passed=True,
        )
    d, p = stats.ks_2samp(observed, expected)
    return DistributionTest(
        variable="?",
        test="ks",
        statistic=float(d),
        p_value=float(p),
        effect_size=float(d),  # the KS D statistic itself is the effect size
        n_observed=len(observed),
        passed=(p > alpha) or (d < 0.05),
    )


def check_marginal_categorical(
    variable: str,
    generated_values: list[str],
    grounding_pct: dict[str, float],
    *,
    alpha: float = 0.05,
) -> DistributionTest:
    obs = {k: 0 for k in grounding_pct}
    for v in generated_values:
        if v in obs:
            obs[v] += 1
    result = chi_square_test(obs, grounding_pct, alpha=alpha)
    result.variable = variable
    return result
