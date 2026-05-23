"""Preview / gate helper unit tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pak.config_dataset import get_default_pak_core_dataset_config
from pak.preview import (
    PreviewThresholds,
    assert_create_ready,
    default_preview_thresholds,
    district_placeholder_stats,
    evaluate_preview_gate,
    exact_duplicate_narrative_rows,
    exposed_column_fill_stats,
    hobby_atom_frequency_stats,
    hobby_set_duplicate_stats,
    infer_preview_gate_profile,
)


def _sample_row(*, persona_suffix: str, hobbies: str) -> dict[str, str]:
    long = "충분한 길이의 한국어 문장입니다. " * 6
    return {
        "persona": f"이 사람은 한국에서 활동하는 예술인입니다 {persona_suffix}.",
        "professional_persona": long + persona_suffix,
        "sports_persona": long,
        "arts_persona": long,
        "travel_persona": long,
        "culinary_persona": long,
        "family_persona": long,
        "cultural_background": long,
        "skills_and_expertise": long,
        "skills_and_expertise_list": "['항목 하나', '항목 둘', '항목 셋']",
        "hobbies_and_interests": long,
        "hobbies_and_interests_list": hobbies,
        "career_goals_and_ambitions": long,
        "creative_world_persona": long,
        "network_persona": long,
        "living_persona": long,
        "support_persona": long,
    }


def test_exact_duplicate_narrative_rows_counts_duplicate_rows() -> None:
    row = _sample_row(persona_suffix="A", hobbies="['독서', '산책']")
    df = pd.DataFrame([row, dict(row), _sample_row(persona_suffix="B", hobbies="['독서', '영화']")])
    assert exact_duplicate_narrative_rows(df) == 1


def test_hobby_set_duplicate_stats_normalizes_order() -> None:
    df = pd.DataFrame(
        {
            "hobbies_and_interests_list": [
                "['독서', '산책']",
                "['산책', '독서']",
                "['영화 감상', '걷기']",
            ]
        }
    )
    stats = hobby_set_duplicate_stats(df)
    assert stats["duplicate_rows"] == 2
    assert stats["duplicate_rate"] == 2 / 3
    assert stats["top_duplicate_sets"][0]["items"] == ["독서", "산책"]


def test_hobby_atom_frequency_stats_counts_row_share() -> None:
    df = pd.DataFrame(
        {
            "hobbies_and_interests_list": [
                "['독서', '산책']",
                "['독서', '영화 감상']",
                "['수영', '걷기']",
            ]
        }
    )
    stats = hobby_atom_frequency_stats(df)
    assert stats["top_atoms"][0]["atom"] == "독서"
    assert stats["top1_share"] == 2 / 3


def test_district_placeholder_stats_detects_placeholder_rows() -> None:
    df = pd.DataFrame({"district": ["서울-시군구미상", "경기-성남시 분당구", "부산-지역미상"]})
    stats = district_placeholder_stats(df)
    assert stats["placeholder_rows"] == 2
    assert stats["placeholder_rate"] == 2 / 3


def test_exposed_column_fill_stats_counts_zero_fill_columns() -> None:
    long = "충분한 길이의 한국어 문장입니다. " * 6
    row = {
        "pak_uuid": "u-1",
        "sex": "여자",
        "age": 30,
        "province": "서울",
        "district": None,
        "country": "대한민국",
        "education_level": "4년제 대학교",
        "occupation": None,
        "marital_status": None,
        "military_status": None,
        "family_type": None,
        "housing_type": None,
        "bachelors_field": None,
        "age_band": "30대",
        "education_level_pak": "대졸 이하",
        "art_field_primary": "문학",
        "art_field_secondary": None,
        "career_years": 8,
        "career_band": "10년 미만",
        "employment_type": "전업",
        "is_freelance": True,
        "has_secondary_job": False,
        "individual_art_income_bracket": "5백-1천만원 미만",
        "household_income_bracket": "3-4천만원 미만",
        "has_contract_experience": True,
        "uses_standard_contract": True,
        "has_copyright": True,
        "had_career_break": False,
        "has_overseas_experience": False,
        "persona": long,
        "professional_persona": long,
        "sports_persona": long,
        "arts_persona": long,
        "travel_persona": long,
        "culinary_persona": long,
        "family_persona": long,
        "cultural_background": long,
        "skills_and_expertise": long,
        "skills_and_expertise_list": "['항목 하나', '항목 둘', '항목 셋']",
        "hobbies_and_interests": long,
        "hobbies_and_interests_list": "['독서', '산책', '수영']",
        "career_goals_and_ambitions": long,
        "creative_world_persona": long,
        "network_persona": long,
        "living_persona": long,
        "support_persona": long,
    }
    stats = exposed_column_fill_stats(pd.DataFrame([row]))
    assert "occupation" in stats["zero_fill_columns"]
    assert "district" not in stats["zero_fill_columns"]
    assert "district" in stats["ignored_unsupported_columns"]
    assert "marital_status" in stats["ignored_unsupported_columns"]
    assert stats["zero_fill_count"] == 1


def test_evaluate_preview_gate_passes_when_metrics_clear_thresholds() -> None:
    post_report = {
        "n_personas": 100,
        "schema_pass_rate": 1.0,
        "field_marginal": {"passed": True},
        "joint_field_sex_age": {"passed": True},
        "row_validation": {"n_with_errors": 0, "n_with_warnings": 5},
        "fact_checks": [{"name": "a", "passed": True}, {"name": "b", "passed": True}],
    }
    extra_checks = {
        "exact_duplicate_narrative_rows": 0,
        "hobby_set_duplicates": {"duplicate_rate": 0.01},
        "hobby_atom_frequency": {"top1_share": 0.2},
        "district_placeholder": {"placeholder_rate": 0.0},
        "exposed_column_fill": {"zero_fill_count": 0},
    }
    gate = evaluate_preview_gate(
        post_report=post_report,
        extra_checks=extra_checks,
        thresholds=PreviewThresholds(),
    )
    assert gate.passed
    assert all(check.passed for check in gate.checks)


def test_evaluate_preview_gate_fails_on_warning_and_duplicates() -> None:
    post_report = {
        "n_personas": 100,
        "schema_pass_rate": 1.0,
        "field_marginal": {"passed": True},
        "joint_field_sex_age": {"passed": False},
        "row_validation": {"n_with_errors": 1, "n_with_warnings": 20},
        "fact_checks": [{"name": "a", "passed": True}, {"name": "b", "passed": False}],
    }
    extra_checks = {
        "exact_duplicate_narrative_rows": 2,
        "hobby_set_duplicates": {"duplicate_rate": 0.05},
        "hobby_atom_frequency": {"top1_share": 0.8},
        "district_placeholder": {"placeholder_rate": 1.0},
        "exposed_column_fill": {"zero_fill_count": 3},
    }
    gate = evaluate_preview_gate(
        post_report=post_report,
        extra_checks=extra_checks,
        thresholds=PreviewThresholds(),
    )
    assert not gate.passed
    failed = {check.name for check in gate.checks if not check.passed}
    assert "validation_error_rate" in failed
    assert "validation_warning_rate" in failed
    assert "exact_duplicate_narrative_rows" in failed
    assert "hobby_set_duplicate_rate" in failed
    assert "hobby_atom_top1_share" in failed
    assert "district_placeholder_rate" in failed
    assert "null_only_exposed_columns" in failed
    assert "joint_field_sex_age_pass" in failed
    assert "fact_checks_passed" in failed


def test_evaluate_preview_gate_fails_when_generation_incomplete() -> None:
    post_report = {
        "n_personas": 80,
        "schema_pass_rate": 1.0,
        "field_marginal": {"passed": True},
        "joint_field_sex_age": {"passed": True},
        "row_validation": {"n_with_errors": 0, "n_with_warnings": 0},
        "fact_checks": [{"name": "a", "passed": True}],
    }
    extra_checks = {
        "generation_completion": {
            "n_requested": 100,
            "n_generated": 80,
            "n_failed": 20,
            "success_rate": 0.8,
        },
        "exact_duplicate_narrative_rows": 0,
        "hobby_set_duplicates": {"duplicate_rate": 0.01},
        "hobby_atom_frequency": {"top1_share": 0.2},
        "district_placeholder": {"placeholder_rate": 0.0},
        "exposed_column_fill": {"zero_fill_count": 0},
    }
    gate = evaluate_preview_gate(
        post_report=post_report,
        extra_checks=extra_checks,
        thresholds=PreviewThresholds(),
    )
    assert not gate.passed
    failed = {check.name for check in gate.checks if not check.passed}
    assert "generation_success_rate" in failed


def test_infer_preview_gate_profile_uses_fixed_eval_metadata() -> None:
    metadata = {"eval_set": {"path": "data/eval/pak_core_fixed_eval_set_v1.json"}}
    assert infer_preview_gate_profile(metadata) == "fixed_eval_set"
    assert infer_preview_gate_profile({}) == "release_preview"


def test_default_preview_thresholds_disable_distribution_checks_for_fixed_eval() -> None:
    thresholds = default_preview_thresholds("fixed_eval_set")
    assert thresholds.enforce_field_marginal_gate is False
    assert thresholds.enforce_joint_gate is False
    assert thresholds.enforce_fact_check_gate is False


def test_evaluate_preview_gate_skips_distribution_checks_when_disabled() -> None:
    post_report = {
        "n_personas": 20,
        "schema_pass_rate": 1.0,
        "field_marginal": {"passed": False},
        "joint_field_sex_age": {"passed": False},
        "row_validation": {"n_with_errors": 0, "n_with_warnings": 0},
        "fact_checks": [{"name": "a", "passed": False}, {"name": "b", "passed": False}],
    }
    extra_checks = {
        "exact_duplicate_narrative_rows": 0,
        "hobby_set_duplicates": {"duplicate_rate": 0.0},
        "hobby_atom_frequency": {"top1_share": 0.2},
        "district_placeholder": {"placeholder_rate": 0.0},
        "exposed_column_fill": {"zero_fill_count": 0},
    }
    gate = evaluate_preview_gate(
        post_report=post_report,
        extra_checks=extra_checks,
        thresholds=default_preview_thresholds("fixed_eval_set"),
    )
    assert gate.passed
    names = {check.name for check in gate.checks}
    assert "field_marginal_pass" not in names
    assert "joint_field_sex_age_pass" not in names
    assert "fact_checks_passed" not in names


def test_assert_create_ready_accepts_matching_preview_report(tmp_path: Path) -> None:
    fingerprint = get_default_pak_core_dataset_config().config.fingerprint()
    report = {
        "dataset_config": fingerprint,
        "gate": {"passed": True, "checks": []},
    }
    path = tmp_path / "preview_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    loaded = assert_create_ready(path)
    assert loaded["gate"]["passed"] is True
