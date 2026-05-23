"""Post-verification helper unit tests."""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pandas as pd
import pytest

from pak.samplers import build_chain_from_spec
from pak.generate import sample_full_quant
from pak.verification.post_check import (
    age_crosstab_checks,
    check_age_career_crosstab,
    fact_checks,
)


@pytest.fixture(scope="module")
def dry_run_10000_df() -> pd.DataFrame:
    chain = build_chain_from_spec()
    rng = random.Random(2026)
    np_rng = np.random.default_rng(2026)
    return pd.DataFrame([sample_full_quant(chain, rng, np_rng) for _ in range(10000)])


def test_fact_checks_use_sample_size_adjusted_tolerance_for_small_preview() -> None:
    rows = 50
    df = pd.DataFrame(
        {
            "employment_type": ["전업"] * 27 + ["겸업"] * 23,
            "has_contract_experience": [True] * 28 + [False] * 22,
            "has_copyright": [True] * 12 + [False] * 38,
            "had_career_break": [True] * 11 + [False] * 39,
            "has_overseas_experience": [True] * 9 + [False] * 41,
        }
    )
    checks = {check.name: check for check in fact_checks(df)}
    assert len(df) == rows
    assert checks["pct_copyright"].passed
    assert checks["pct_copyright"].tolerance > 0.05


def test_age_crosstab_checks_fail_on_old_1000_pilot() -> None:
    """The 1000-row pilot generated with pre-06j grounding should fail on some age cross checks."""
    path = Path("data/synthetic/post_patch_pilot_1000_max_agefix_20260508/personas.parquet")
    df = pd.read_parquet(path)
    checks = age_crosstab_checks(df)
    failed = [check for check in checks if not check.passed]
    assert failed, "old pilot should expose at least one age crosstab failure"
    assert any("age_career_crosstab" in check.name for check in failed)


def test_age_crosstab_checks_pass_on_new_10000_dry_run(
    dry_run_10000_df: pd.DataFrame,
) -> None:
    """The sampler v0.3.0 10,000-row dry run passes all T8~T15 age cross checks."""
    checks = age_crosstab_checks(dry_run_10000_df)
    assert checks
    assert all(check.passed for check in checks)


def test_age_crosstab_checks_handle_empty_dataframe() -> None:
    df = pd.DataFrame(columns=["age", "career_band"])
    checks = check_age_career_crosstab(df)
    assert len(checks) == 1
    assert not checks[0].passed
    assert "empty dataframe" in checks[0].note
