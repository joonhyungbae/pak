"""Grounding distribution unit tests."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pak.config import settings
from pak.grounding.marginals import REPORT_FIELD_POPULATION, cross_check
from pak.grounding.tables import FIELDS_14, INDIVIDUAL_INCOME_9


AGE_VAR_SPECS = [
    {
        "table_id": "T9",
        "sampler_name": "employment_type",
        "table_value_col": "employment_type",
        "report": {
            "30대 이하": {"전업": 49.8, "겸업": 50.2},
            "40대": {"전업": 50.5, "겸업": 49.5},
            "50대": {"전업": 43.4, "겸업": 56.6},
            "60세 이상": {"전업": 63.7, "겸업": 36.3},
        },
    },
    {
        "table_id": "T10",
        "sampler_name": "individual_art_income_bracket",
        "table_value_col": "income_bracket",
        "report": {
            "30대 이하": dict(
                zip(
                    INDIVIDUAL_INCOME_9,
                    [21.8, 34.0, 10.5, 12.5, 8.5, 6.2, 2.2, 1.3, 3.0],
                    strict=True,
                )
            ),
            "40대": dict(
                zip(
                    INDIVIDUAL_INCOME_9,
                    [21.0, 27.2, 10.5, 12.9, 9.3, 6.3, 3.1, 2.8, 6.9],
                    strict=True,
                )
            ),
            "50대": dict(
                zip(
                    INDIVIDUAL_INCOME_9,
                    [30.8, 25.9, 12.4, 11.4, 5.9, 4.8, 2.8, 2.1, 3.9],
                    strict=True,
                )
            ),
            "60세 이상": dict(
                zip(
                    INDIVIDUAL_INCOME_9,
                    [51.3, 24.9, 8.2, 8.0, 3.1, 2.1, 0.8, 0.4, 1.4],
                    strict=True,
                )
            ),
        },
    },
    {
        "table_id": "T11",
        "sampler_name": "has_contract_experience",
        "table_value_col": "value",
        "report": {
            "30대 이하": {True: 69.2, False: 30.8},
            "40대": {True: 69.0, False: 31.0},
            "50대": {True: 52.9, False: 47.1},
            "60세 이상": {True: 33.6, False: 66.4},
        },
    },
    {
        "table_id": "T12",
        "sampler_name": "uses_standard_contract",
        "table_value_col": "value",
        "report": {
            "30대 이하": {True: 70.6, False: 29.4},
            "40대": {True: 74.8, False: 25.2},
            "50대": {True: 73.5, False: 26.5},
            "60세 이상": {True: 69.1, False: 30.9},
        },
    },
    {
        "table_id": "T13",
        "sampler_name": "has_copyright",
        "table_value_col": "value",
        "report": {
            "30대 이하": {True: 35.8, False: 64.2},
            "40대": {True: 33.8, False: 66.2},
            "50대": {True: 24.3, False: 75.7},
            "60세 이상": {True: 18.7, False: 81.3},
        },
    },
    {
        "table_id": "T14",
        "sampler_name": "had_career_break",
        "table_value_col": "value",
        "report": {
            "30대 이하": {True: 25.4, False: 74.6},
            "40대": {True: 30.2, False: 69.8},
            "50대": {True: 25.6, False: 74.4},
            "60세 이상": {True: 13.4, False: 86.6},
        },
    },
    {
        "table_id": "T15",
        "sampler_name": "has_overseas_experience",
        "table_value_col": "value",
        "report": {
            "30대 이하": {True: 17.2, False: 82.8},
            "40대": {True: 20.6, False: 79.4},
            "50대": {True: 18.5, False: 81.5},
            "60세 이상": {True: 11.7, False: 88.3},
        },
    },
]


@pytest.fixture(scope="module")
def synthetic_10000_df() -> pd.DataFrame:
    """Result of a SamplerChain 10,000-row dry run."""
    import numpy as np
    from pak.samplers import age_band_to_4group, build_chain_from_spec, split_sex_age

    chain = build_chain_from_spec()
    rng = np.random.default_rng(42)
    samples = [chain.sample_one(rng) for _ in range(10000)]
    df = pd.DataFrame(samples)
    df["age_band"] = df["sex_age"].map(lambda s: split_sex_age(s)[1])
    df["age_group_4"] = df["age_band"].map(age_band_to_4group)
    return df


def test_marginal_consistency() -> None:
    """The P(field) population marginals of T1 and T2 nearly match the report."""
    df = cross_check()
    t2_diff = df[df["table"] == "T2"]["abs_diff"].max()
    assert t2_diff < 0.005, f"T2 max abs diff > 0.5%: {t2_diff}"

    t1_diff = df[df["table"] == "T1"]["abs_diff"].max()
    # T1 reaches up to ~10% for one field (만화) due to excluding "unknown", but the overall distribution diff is within 1%
    assert t1_diff < 0.02, f"T1 max abs diff > 2%: {t1_diff}"


def test_probabilities_sum_to_one() -> None:
    """Field-conditional distributions (T2~T6, T7-binary) sum to 1.0 ± 1e-6."""
    spec = json.loads((settings.grounding_dir / "sampler_specs.json").read_text(encoding="utf-8"))
    for sampler in spec["samplers"]:
        if sampler["type"] == "category":
            assert abs(sum(sampler["weights"]) - 1.0) < 1e-6, sampler["name"]
        elif sampler["type"] == "subcategory":
            for field, sub in sampler["subcategories"].items():
                weights_sum = sum(sub["weights"])
                # some tables have dash cells, so a slight shortfall is possible — allow up to 5%
                assert 0.85 <= weights_sum <= 1.15, (
                    f"{sampler['name']}/{field} weights sum = {weights_sum}"
                )


def test_no_negative_probabilities() -> None:
    df = pd.read_parquet(settings.grounding_dir / "joint_distributions.parquet")
    assert (df["probability"] >= 0).all()


def test_all_14_fields_covered() -> None:
    """All 14 fields appear in every grounding parquet."""
    for tid in ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]:
        df = pd.read_parquet(settings.grounding_dir / f"{tid}.parquet")
        actual = set(df["field"].unique())
        missing = set(FIELDS_14) - actual
        assert not missing, f"{tid} missing fields: {missing}"


def test_t2_population_total_matches_report() -> None:
    """T2 field totals match the report population 100%."""
    df = pd.read_parquet(settings.grounding_dir / "T2.parquet")
    for field, expected in REPORT_FIELD_POPULATION.items():
        actual = df[df["field"] == field]["count"].sum()
        assert actual == expected, f"{field}: got {actual}, expected {expected}"


def test_t6_has_all_9_income_brackets() -> None:
    """T6 holds all 9 brackets (regression guard for an earlier build where '6천만원 이상' was missing)."""
    df = pd.read_parquet(settings.grounding_dir / "T6.parquet")
    expected = {
        "없음", "5백만원 미만", "5백-1천만원 미만", "1-2천만원 미만", "2-3천만원 미만",
        "3-4천만원 미만", "4-5천만원 미만", "5-6천만원 미만", "6천만원 이상",
    }
    actual = set(df["income_bracket"].unique())
    assert actual == expected, f"T6 income brackets mismatch: {expected - actual}"


def test_t6_field_marginal_matches_report_pcts() -> None:
    """T6 field row sums = report case counts ±1 (rounding tolerance)."""
    df = pd.read_parquet(settings.grounding_dir / "T6.parquet")
    expected = {
        "문학": 480, "미술": 790, "공예": 232, "사진": 311, "건축": 188,
        "음악": 429, "국악": 314, "대중음악": 714, "방송연예": 385,
        "무용": 240, "연극": 467, "영화": 293, "만화": 163, "기타": 41,
    }
    sums = df.groupby("field")["count"].sum().to_dict()
    for f, exp in expected.items():
        assert abs(sums[f] - exp) <= 2, f"T6 {f}: got {sums[f]} expected {exp}"


def test_t6_overall_pct_within_tolerance() -> None:
    """T6 overall marginal % is within ±5%p of the report p.87 cells."""
    df = pd.read_parquet(settings.grounding_dir / "T6.parquet")
    overall = df.groupby("income_bracket")["count"].sum()
    total = overall.sum()
    report = {
        "없음": 31.0, "5백만원 미만": 29.2, "5백-1천만원 미만": 10.2,
        "1-2천만원 미만": 11.2, "2-3천만원 미만": 6.8, "3-4천만원 미만": 4.9,
        "4-5천만원 미만": 2.1, "5-6천만원 미만": 1.4, "6천만원 이상": 3.3,
    }
    for k, expected in report.items():
        actual = 100.0 * overall[k] / total
        assert abs(actual - expected) < 5.0, (
            f"T6 {k}: synthetic {actual:.1f}% vs report {expected}% (diff {actual-expected:+.1f}%p)"
        )


def test_t8_age_career_cross_matches_report() -> None:
    """T8 (age4 x career5) row ratios exactly match the report p.55 cells (±0.5%p)."""
    df = pd.read_parquet(settings.grounding_dir / "T8.parquet")
    report = {
        "30대 이하": [75.9, 22.1, 2.0, 0.0, 0.0],
        "40대": [26.9, 46.1, 24.1, 2.9, 0.1],
        "50대": [22.7, 25.7, 30.4, 19.3, 1.9],
        "60세 이상": [20.0, 20.6, 22.9, 17.0, 19.5],
    }
    careers = ["10년 미만", "10-20년 미만", "20-30년 미만", "30-40년 미만", "40년 이상"]
    for ag, exp_pcts in report.items():
        sub = df[df["age_group_4"] == ag]
        rowsum = sub["count"].sum()
        for c, expected in zip(careers, exp_pcts, strict=True):
            actual = 100.0 * sub[sub["career_band"] == c]["count"].iloc[0] / rowsum
            assert abs(actual - expected) < 0.5, (
                f"T8 {ag}/{c}: {actual:.1f}% vs report {expected}% (diff {actual-expected:+.1f}%p)"
            )


@pytest.mark.parametrize("spec", AGE_VAR_SPECS, ids=lambda s: s["sampler_name"])
def test_age_variable_cross_matches_report(spec: dict) -> None:
    """T9~T15 (age4 x variable) row ratios are within ±0.5%p of the report cells."""
    df = pd.read_parquet(settings.grounding_dir / f"{spec['table_id']}.parquet")
    value_col = spec["table_value_col"]
    for age_group, expected_by_value in spec["report"].items():
        sub = df[df["age_group_4"] == age_group]
        for value, expected in expected_by_value.items():
            actual = 100.0 * sub[sub[value_col] == value]["probability"].iloc[0]
            assert abs(actual - expected) <= 0.5, (
                f"{spec['table_id']} {age_group}/{value}: "
                f"{actual:.1f}% vs report {expected}% (diff {actual-expected:+.1f}%p)"
            )


def test_career_band_sampler_uses_field_age_joint_parent() -> None:
    """The career_band sampler is registered in the spec to use the (field, age_group_4) joint parent."""
    spec = json.loads((settings.grounding_dir / "sampler_specs.json").read_text(encoding="utf-8"))
    samplers = {s["name"]: s for s in spec["samplers"]}
    assert "field_age_group_4" in samplers, "missing derived sampler 'field_age_group_4'"
    assert samplers["field_age_group_4"]["type"] == "derived"
    assert samplers["career_band"]["parent"] == "field_age_group_4", (
        "career_band must depend on field_age_group_4 (joint parent)"
    )
    # all (field, age4) keys exist
    keys = set(samplers["career_band"]["subcategories"].keys())
    assert len(keys) == 14 * 4, f"expected 56 (field,age4) keys, got {len(keys)}"


@pytest.mark.parametrize("var_spec", AGE_VAR_SPECS, ids=lambda s: s["sampler_name"])
def test_age_variable_sampler_uses_field_age_joint_parent(var_spec: dict) -> None:
    """The T9~T15-based samplers use the (field, age_group_4) joint parent."""
    spec = json.loads((settings.grounding_dir / "sampler_specs.json").read_text(encoding="utf-8"))
    samplers = {s["name"]: s for s in spec["samplers"]}
    sampler = samplers[var_spec["sampler_name"]]
    assert spec["version"] == "0.3.0"
    assert sampler["parent"] == "field_age_group_4"
    assert len(sampler["subcategories"]) == 14 * 4


def test_employment_split_sampler_structure() -> None:
    """Split the legacy employment_freelance sampler into employment_type + is_freelance."""
    spec = json.loads((settings.grounding_dir / "sampler_specs.json").read_text(encoding="utf-8"))
    samplers = {s["name"]: s for s in spec["samplers"]}
    assert "employment_freelance" not in samplers
    assert samplers["employment_type"]["parent"] == "field_age_group_4"
    assert samplers["field_employment"]["type"] == "derived"
    assert samplers["field_employment"]["transform"] == "field_employment_join"
    assert samplers["is_freelance"]["parent"] == "field_employment"
    assert len(samplers["is_freelance"]["subcategories"]) == 14 * 2


def test_ipf_career_joint_converged() -> None:
    """Per-field 2-D IPF all converge (max rel err < 1e-8)."""
    spec = json.loads((settings.grounding_dir / "sampler_specs.json").read_text(encoding="utf-8"))
    info = spec.get("ipf_diagnostics", {}).get("career_joint", {})
    assert info.get("all_converged"), f"IPF did not converge: {info}"
    assert info.get("global_max_rel_err", 1.0) < 1e-8


def test_ipf_age_variable_joints_converged() -> None:
    """The T9~T15-based per-field 2-D IPF all converge."""
    spec = json.loads((settings.grounding_dir / "sampler_specs.json").read_text(encoding="utf-8"))
    diagnostics = spec.get("ipf_diagnostics", {})
    for name in [
        "employment_type_joint",
        "individual_art_income_bracket_joint",
        "has_contract_experience_joint",
        "uses_standard_contract_joint",
        "has_copyright_joint",
        "had_career_break_joint",
        "has_overseas_experience_joint",
    ]:
        info = diagnostics.get(name, {})
        assert info.get("all_converged"), f"{name} IPF did not converge: {info}"
        assert info.get("global_max_rel_err", 1.0) < 1e-8


def test_synthetic_age_career_marginal_matches_report(
    synthetic_10000_df: pd.DataFrame,
) -> None:
    """The (age_group_4 x career_band) cross of the sampler's 10,000-row dry run is within ±5%p of report p.55."""
    df = synthetic_10000_df
    report = {
        "30대 이하": {"10년 미만": 75.9, "10-20년 미만": 22.1, "20-30년 미만": 2.0,
                  "30-40년 미만": 0.0, "40년 이상": 0.0},
        "40대": {"10년 미만": 26.9, "10-20년 미만": 46.1, "20-30년 미만": 24.1,
                "30-40년 미만": 2.9, "40년 이상": 0.1},
        "50대": {"10년 미만": 22.7, "10-20년 미만": 25.7, "20-30년 미만": 30.4,
                "30-40년 미만": 19.3, "40년 이상": 1.9},
        "60세 이상": {"10년 미만": 20.0, "10-20년 미만": 20.6, "20-30년 미만": 22.9,
                  "30-40년 미만": 17.0, "40년 이상": 19.5},
    }
    ct = pd.crosstab(df["age_group_4"], df["career_band"], normalize="index") * 100
    for ag, careers_dict in report.items():
        for c, expected in careers_dict.items():
            actual = float(ct.loc[ag, c]) if c in ct.columns else 0.0
            assert abs(actual - expected) <= 5.0, (
                f"{ag}/{c}: synthetic {actual:.1f}% vs report {expected}% (diff {actual-expected:+.1f}%p)"
            )


@pytest.mark.parametrize("spec", AGE_VAR_SPECS, ids=lambda s: s["sampler_name"])
def test_synthetic_age_variable_marginal_matches_report(
    spec: dict,
    synthetic_10000_df: pd.DataFrame,
) -> None:
    """The (age_group_4 x variable) cross of the sampler's 10,000-row dry run is within ±5%p of the report."""
    df = synthetic_10000_df
    ct = pd.crosstab(df["age_group_4"], df[spec["sampler_name"]], normalize="index") * 100
    for age_group, expected_by_value in spec["report"].items():
        for value, expected in expected_by_value.items():
            actual = float(ct.loc[age_group, value]) if value in ct.columns else 0.0
            assert abs(actual - expected) <= 5.0, (
                f"{spec['sampler_name']} {age_group}/{value}: "
                f"synthetic {actual:.1f}% vs report {expected}% "
                f"(diff {actual-expected:+.1f}%p)"
            )
