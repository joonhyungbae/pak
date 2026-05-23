"""Phase 06 — generate.py unit tests (no LLM calls)."""

from __future__ import annotations

from collections import Counter
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pak.config_dataset import get_default_pak_core_dataset_config
from pak.generate import (
    _family_genericity_issues,
    _hobby_list_atomicity_issues,
    _load_quant_rows_from_path,
    _living_genericity_issues,
    _network_genericity_issues,
    _normalized_hobby_family_set,
    _normalized_hobby_set,
    _occupation_from_field,
    _sample_district,
    _sample_household_income,
    _support_genericity_issues,
    _sample_npk_education,
    GenerateConfig,
    build_single_call_prompt,
    generate_one,
    parse_narrative_response,
    sample_full_quant,
)
from pak.samplers import build_chain_from_spec, min_career_start_age
from pak.schema import PAKPersonaNarrative, PAKPersonaQuant
from pak.validators import ValidationPipeline


def _sample_narrative_payload(
    *,
    professional_persona: str | None = None,
    creative_world_persona: str | None = None,
) -> dict[str, str]:
    long = "충분한 길이의 한국어 문장입니다. " * 6
    return {
        "persona": "이 사람은 한국에서 활동하는 예술인으로 작업과 생활의 균형을 꾸준히 만들어가고 있습니다.",
        "professional_persona": professional_persona or long,
        "sports_persona": long,
        "arts_persona": long,
        "travel_persona": long,
        "culinary_persona": long,
        "family_persona": long,
        "cultural_background": long,
        "skills_and_expertise": long,
        "skills_and_expertise_list": "['항목 하나', '항목 둘', '항목 셋']",
        "hobbies_and_interests": long,
        "hobbies_and_interests_list": (
            "['아침 카페에서 기록 정리', '필라테스', '생활사 에세이 읽기', "
            "'핸드드립 커피 내리기', '작은 전시 공간 방문']"
        ),
        "career_goals_and_ambitions": long,
        "creative_world_persona": creative_world_persona or long,
        "network_persona": long,
        "living_persona": long,
        "support_persona": long,
    }


def test_occupation_per_field() -> None:
    rng = random.Random(0)
    for field in ["문학", "미술", "공예", "음악", "영화", "만화"]:
        occ = _occupation_from_field(field, rng)
        assert isinstance(occ, str) and len(occ) > 0


def test_npk_education_breakdown_matches_pak() -> None:
    rng = random.Random(0)
    for level in ["고졸 이하", "대졸 이하", "대학원 이상"]:
        npk = _sample_npk_education(level, rng)
        if level == "대학원 이상":
            assert npk == "대학원"
        elif level == "대졸 이하":
            assert npk in {"2~3년제 전문대학", "4년제 대학교"}
        else:
            assert npk in {"무학", "초등학교", "중학교", "고등학교"}


def test_district_sampler_returns_none_under_pdf_only_grounding() -> None:
    rng = random.Random(0)
    d = _sample_district("서울", rng)
    assert d is None


def test_household_income_returns_valid_value() -> None:
    np_rng = np.random.default_rng(0)
    v = _sample_household_income("문학", np_rng)
    assert v in {
        "1천만원 미만",
        "1-2천만원 미만",
        "2-3천만원 미만",
        "3-4천만원 미만",
        "4-5천만원 미만",
        "5-6천만원 미만",
        "6-7천만원 미만",
        "7-8천만원 미만",
        "8천만원 이상",
    }


def test_sample_full_quant_passes_pydantic() -> None:
    chain = build_chain_from_spec()
    rng = random.Random(0)
    np_rng = np.random.default_rng(0)
    quant = sample_full_quant(chain, rng, np_rng)
    PAKPersonaQuant.model_validate(quant)
    assert quant["career_years"] <= quant["age"] - min_career_start_age(quant["art_field_primary"])


def test_sample_full_quant_normalizes_standard_contract_flag() -> None:
    chain = build_chain_from_spec()
    rng = random.Random(7)
    np_rng = np.random.default_rng(7)
    for _ in range(200):
        quant = sample_full_quant(chain, rng, np_rng)
        if not quant["has_contract_experience"]:
            assert quant["uses_standard_contract"] is None
            return
    pytest.skip("sample did not include a no-contract persona within 200 draws")


def test_npk_compatible_quant_columns_present_in_sample() -> None:
    """sample_full_quant only generates quantitative fields; narrative columns are excluded."""
    chain = build_chain_from_spec()
    rng = random.Random(0)
    np_rng = np.random.default_rng(0)
    quant = sample_full_quant(chain, rng, np_rng)
    npk_quant_cols = {
        "sex",
        "age",
        "marital_status",
        "military_status",
        "family_type",
        "housing_type",
        "education_level",
        "bachelors_field",
        "occupation",
        "district",
        "province",
        "country",
    }
    for col in npk_quant_cols:
        assert col in quant, f"missing NPK quant col: {col}"


def test_pak_domain_quant_columns_present_in_sample() -> None:
    chain = build_chain_from_spec()
    rng = random.Random(0)
    np_rng = np.random.default_rng(0)
    quant = sample_full_quant(chain, rng, np_rng)
    pak_quant_cols = {
        "age_band",
        "education_level_pak",
        "art_field_primary",
        "art_field_secondary",
        "career_years",
        "career_band",
        "employment_type",
        "is_freelance",
        "has_secondary_job",
        "individual_art_income_bracket",
        "household_income_bracket",
        "has_contract_experience",
        "uses_standard_contract",
        "has_copyright",
        "had_career_break",
        "has_overseas_experience",
    }
    for col in pak_quant_cols:
        assert col in quant, f"missing PAK quant col: {col}"


def test_build_single_call_prompt() -> None:
    chain = build_chain_from_spec()
    rng = random.Random(0)
    np_rng = np.random.default_rng(0)
    quant = sample_full_quant(chain, rng, np_rng)
    sys, user = build_single_call_prompt(quant, rng)
    dataset_cfg = get_default_pak_core_dataset_config()
    assert "JSON" in sys
    assert quant["art_field_primary"] in user
    assert quant["occupation"] in user
    assert "global_constraints" in user
    assert "[persona blueprint]" in user
    assert "[취미 계획 앵커]" in user
    assert "[필드별 역할 계약]" in user
    assert "space_anchor" in user
    assert "expense_anchor" in user
    assert "recovery_anchor" in user
    assert "creative_world_persona" in user
    assert "support_persona" in user
    assert "output_contract" in user
    for label in dataset_cfg.narrative_spec.anchor_labels.values():
        assert label in user
    expected_keys = dataset_cfg.narrative_spec.output_fields
    for k in expected_keys:
        assert k in sys, f"key {k} missing from system prompt"


def test_build_single_call_prompt_adds_full_time_no_side_job_guard() -> None:
    chain = build_chain_from_spec()
    rng = random.Random(0)
    np_rng = np.random.default_rng(0)
    quant = sample_full_quant(chain, rng, np_rng)
    quant["art_field_primary"] = "대중음악"
    quant["occupation"] = "음악 프로듀서"
    quant["province"] = "서울"
    quant["employment_type"] = "전업"
    quant["has_secondary_job"] = False
    quant["career_years"] = 5
    _, user = build_single_call_prompt(quant, rng)
    assert "직업명 '음악 프로듀서'" in user
    assert "전업이며 부업 없음" in user
    assert "활동/거주 중심 지역은 '서울'" in user


def test_build_single_call_prompt_includes_retry_feedback() -> None:
    chain = build_chain_from_spec()
    rng = random.Random(0)
    np_rng = np.random.default_rng(0)
    quant = sample_full_quant(chain, rng, np_rng)
    _, user = build_single_call_prompt(
        quant,
        rng,
        retry_feedback=["EMPLOYMENT_MISMATCH: 전업인데 알바 서술 제거"],
    )
    assert "[이전 시도에서 반드시 고칠 점]" in user
    assert "EMPLOYMENT_MISMATCH: 전업인데 알바 서술 제거" in user


def test_generate_config_max_profile_applies_strict_defaults() -> None:
    cfg = GenerateConfig(quality_profile="max")
    assert cfg.fail_on_warnings is True
    assert cfg.max_retries >= 4
    assert cfg.max_warning_revisions >= 2
    assert cfg.temperature <= 0.45
    assert "AGE_MISMATCH" in cfg.blocking_warning_codes
    assert "EMPLOYMENT_MISMATCH" in cfg.blocking_warning_codes
    assert cfg.enforce_living_persona_specificity is True
    assert cfg.enforce_family_persona_specificity is True
    assert cfg.enforce_network_persona_specificity is True
    assert cfg.enforce_support_persona_specificity is True
    assert cfg.hobby_exact_atom_cap_per_batch == 25
    assert cfg.hobby_family_cap_per_batch is None
    assert cfg.coerce_hobbies_to_plan is True


def test_generate_config_max_profile_scales_hobby_cap_with_batch_size() -> None:
    small_cfg = GenerateConfig(n=20, quality_profile="max")
    large_cfg = GenerateConfig(n=1000, quality_profile="max")
    explicit_cfg = GenerateConfig(
        n=100,
        quality_profile="max",
        hobby_exact_atom_cap_per_batch=7,
    )

    assert small_cfg.hobby_exact_atom_cap_per_batch == 5
    assert large_cfg.hobby_exact_atom_cap_per_batch == 250
    assert explicit_cfg.hobby_exact_atom_cap_per_batch == 7


def test_load_quant_rows_from_selector_json(tmp_path: Path) -> None:
    chain = build_chain_from_spec()
    rng = random.Random(11)
    np_rng = np.random.default_rng(11)
    rows = [sample_full_quant(chain, rng, np_rng) for _ in range(2)]
    rows[0]["district"] = "서울-시군구미상"
    rows[1]["district"] = "경기-성남시 분당구"
    parquet_path = tmp_path / "anchors.parquet"
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)

    selector_path = tmp_path / "eval_set.json"
    selector_path.write_text(
        json.dumps(
            {
                "name": "tmp_eval",
                "source_parquet": str(parquet_path),
                "entries": [
                    {"label": "second", "pak_uuid": rows[1]["pak_uuid"]},
                    {"label": "first", "pak_uuid": rows[0]["pak_uuid"]},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    loaded_rows, info = _load_quant_rows_from_path(selector_path)
    assert info["mode"] == "parquet_selector"
    assert info["n_rows"] == 2
    assert [row["pak_uuid"] for row in loaded_rows] == [rows[1]["pak_uuid"], rows[0]["pak_uuid"]]
    for row in loaded_rows:
        PAKPersonaQuant.model_validate(row)
        assert row["district"] is None


def test_normalized_hobby_family_set_collapses_near_duplicates() -> None:
    value = "['작은 전시 공간 방문', '독립서점 둘러보기', '야간 산책하며 생각 정리']"
    normalized = _normalized_hobby_family_set(value)
    assert "전시 공간 방문" in normalized
    assert "서점 방문" in normalized
    assert "산책" in normalized


def test_living_genericity_issues_detects_template_heavy_living_persona() -> None:
    issues = _living_genericity_issues(
        living_text=(
            "작업실과 거주 공간을 병행하고 있다. 주중에는 작업 시간을 길게 확보하고, "
            "주말에는 회복 시간을 챙긴다. 생활 동선을 안정적으로 묶고 일상의 균형을 유지한다."
        ),
        prompt_context={
            "space_anchor": "작업대와 수납 동선을 자주 다시 짜는 편",
            "expense_anchor": "재료비와 생활비를 나눠 계산하는 편",
            "recovery_anchor": "짧은 산책으로 리듬을 다시 맞추는 편",
        },
    )
    assert issues is not None
    assert any("LIVING_GENERICITY" in issue for issue in issues)


def test_family_genericity_issues_detects_template_heavy_family_persona() -> None:
    issues = _family_genericity_issues(
        family_text=(
            "가족과의 시간 조율을 통해 생활 책임과 작업 지속성을 함께 챙기는 편이다. "
            "가족과의 관계를 유지하며 주변 돌봄과 자기 작업 시간을 함께 조율한다."
        ),
        prompt_context={
            "family_rhythm": "작업 블록을 먼저 잡아 두고 관계 약속은 그 빈칸에 맞추는 편",
            "family_boundary": "작업 얘기는 길게 가져가기보다 필요한 일정만 짧게 공유하는 편",
            "family_responsibility": "돌봄 요청이 겹치면 작업 블록을 앞뒤로 옮겨 대응하는 편",
        },
    )
    assert issues is not None
    assert any("FAMILY_GENERICITY" in issue for issue in issues)


def test_hobby_list_atomicity_issues_detects_compound_item() -> None:
    issues = _hobby_list_atomicity_issues(
        hobby_items=(
            "아침 카페에서 기록 정리",
            "수영과 산책",
            "생활사 에세이 읽기",
            "핸드드립 커피 내리기",
            "작은 전시 공간 방문",
        ),
        prompt_context={"hobby_plan_items": []},
    )
    assert issues is not None
    assert any("HOBBY_LIST_ATOMICITY" in issue for issue in issues[0])


def test_network_genericity_issues_detects_overclaim() -> None:
    issues = _network_genericity_issues(
        network_text="서울의 예술 생태계에서 중심적인 역할을 하며 인맥을 넓히고 있다.",
        prompt_context={
            "network_scope": "소수 협업자 중심의 연결이 이어지는 편",
            "network_role": "실무 조율을 맡는 중간 역할에 가까운 편",
            "network_friction": "일정 조율과 응답 속도에서 마찰이 생기는 편",
        },
    )
    assert issues is not None
    assert any("NETWORK_GENERICITY" in issue for issue in issues)


def test_network_genericity_issues_detects_fixed_scaffold() -> None:
    issues = _network_genericity_issues(
        network_text=(
            "요청이 들어오면 연결과 조율을 맡는 중간 역할을 자주 한다. "
            "일정 충돌과 응답 속도를 맞추는 데 에너지를 쏟는다."
        ),
        prompt_context={
            "network_scope": "반복 협업자 중심의 실무 연결이 이어지는 편",
            "network_role": "파트 사이 누락을 줄이기 위해 연락선을 챙기는 편",
            "network_friction": "파일 전달과 일정 확인에서 마찰이 생기기 쉬운 편",
        },
    )
    assert issues is not None
    assert any("NETWORK_GENERICITY" in issue for issue in issues)


def test_support_genericity_issues_detects_general_support_language() -> None:
    issues = _support_genericity_issues(
        support_text="지원 제도를 중요하게 여기며 실질적인 도움을 기대한다.",
        prompt_context={
            "support_path": "작은 제작비 지원을 우선 살피는 편",
            "support_friction": "서류와 정산 부담이 크면 미루는 편",
            "support_effect": "작업 시간을 벌어 주는 지원에 의미를 두는 편",
        },
    )
    assert issues is not None
    assert any("SUPPORT_GENERICITY" in issue for issue in issues)


def test_support_genericity_issues_detects_fixed_scaffold() -> None:
    issues = _support_genericity_issues(
        support_text=(
            "지원 제도는 필요할 때 쓰되 행정 부담이 크면 쉽게 지치는 편이다. "
            "경력 공백 이후 다시 리듬을 회복하는 데 도움이 되는 지원을 중요하게 여긴다."
        ),
        prompt_context={
            "support_path": "복귀 초반에 작은 발표 흐름으로 다시 들어가는 지원을 먼저 찾는 편",
            "support_friction": "서류와 정산 시간이 길어지면 지원을 미루는 편",
            "support_effect": "멈췄던 작업 감각을 다시 꺼내 볼 여지를 크게 보는 편",
        },
    )
    assert issues is not None
    assert any("SUPPORT_GENERICITY" in issue for issue in issues)


def test_parse_narrative_response_valid() -> None:
    raw = json.dumps(_sample_narrative_payload(), ensure_ascii=False)
    out = parse_narrative_response(raw)
    PAKPersonaNarrative.model_validate(out)
    assert out["persona"].startswith("이 사람은")


def test_parse_narrative_response_with_fence() -> None:
    raw = "여기:\n```json\n" + json.dumps(_sample_narrative_payload(), ensure_ascii=False) + "\n```\n끝"
    out = parse_narrative_response(raw)
    assert out["skills_and_expertise_list"].startswith("[")


def test_parse_narrative_response_accepts_unsuffixed_persona_aliases() -> None:
    payload = _sample_narrative_payload()
    payload["creative_world"] = payload.pop("creative_world_persona")
    payload["network"] = payload.pop("network_persona")
    payload["living"] = payload.pop("living_persona")
    payload["support"] = payload.pop("support_persona")
    raw = json.dumps(payload, ensure_ascii=False)
    out = parse_narrative_response(raw)
    assert out["creative_world_persona"]
    assert out["network_persona"]
    assert out["living_persona"]
    assert out["support_persona"]


def test_parse_narrative_response_accepts_common_field_typo_alias() -> None:
    payload = _sample_narrative_payload()
    payload["skills_and_experteise"] = payload.pop("skills_and_expertise")
    payload["skills_and_experteise_list"] = payload.pop("skills_and_expertise_list")
    raw = json.dumps(payload, ensure_ascii=False)
    out = parse_narrative_response(raw)
    assert out["skills_and_expertise"]
    assert out["skills_and_expertise_list"]


def test_parse_narrative_response_accepts_json_list_for_list_fields() -> None:
    payload = _sample_narrative_payload()
    payload["hobbies_and_interests_list"] = ["아침 카페 기록 정리", "필라테스", "생활사 에세이 읽기"]
    payload["skills_and_expertise_list"] = ["항목 하나", "항목 둘", "항목 셋"]
    raw = json.dumps(payload, ensure_ascii=False)
    out = parse_narrative_response(raw)
    assert out["hobbies_and_interests_list"].startswith("[")
    assert "필라테스" in out["hobbies_and_interests_list"]
    assert out["skills_and_expertise_list"].startswith("[")


def test_parse_narrative_response_rejects_partial_payload() -> None:
    with pytest.raises(ValueError):
        parse_narrative_response('{"persona": "짧음", "professional_persona": "부족"}')


def test_generate_one_retries_until_validation_passes() -> None:
    class _Resp:
        def __init__(self, text: str) -> None:
            self.text = text
            self.usage = type("Usage", (), {"input_tokens": 10, "output_tokens": 20})()

    class _Client:
        def __init__(self, responses: list[str]) -> None:
            self._responses = responses

        def chat(self, **kwargs):
            return _Resp(self._responses.pop(0))

    chain = build_chain_from_spec()
    rng = random.Random(0)
    np_rng = np.random.default_rng(0)
    quant = sample_full_quant(chain, rng, np_rng)
    bad = json.dumps(
        _sample_narrative_payload(
            creative_world_persona=(
                "그는 가난하지만 자유로운 보헤미안으로 불린다. "
                "충분한 길이의 한국어 문장입니다. 충분한 길이의 한국어 문장입니다. 충분한 길이의 한국어 문장입니다."
            )
        ),
        ensure_ascii=False,
    )
    good = json.dumps(
        _sample_narrative_payload(
            professional_persona=(
                f"{quant['occupation']}로 활동하는 예술인이다. "
                "충분한 길이의 한국어 문장입니다. 충분한 길이의 한국어 문장입니다. "
                "충분한 길이의 한국어 문장입니다. 충분한 길이의 한국어 문장입니다."
            )
        ),
        ensure_ascii=False,
    )
    client = _Client([bad, good])
    row, log, result = generate_one(
        quant,
        client=client,
        cfg=type("Cfg", (), {"model": "test-model", "max_retries": 1, "max_tokens": 1000, "temperature": 0.8, "skip_validation": False})(),
        rng=random.Random(0),
        pipeline=ValidationPipeline(),
    )
    assert row is not None
    assert log["attempts"] == 2
    assert log["validation_failed_attempts"] == 1
    assert result is not None and not result.has_errors


def test_generate_one_repairs_missing_narrative_fields_within_same_attempt() -> None:
    class _Resp:
        def __init__(self, text: str) -> None:
            self.text = text
            self.usage = type("Usage", (), {"input_tokens": 10, "output_tokens": 20})()

    class _Client:
        def __init__(self, responses: list[str]) -> None:
            self._responses = responses

        def chat(self, **kwargs):
            return _Resp(self._responses.pop(0))

    chain = build_chain_from_spec()
    rng = random.Random(1)
    np_rng = np.random.default_rng(1)
    quant = sample_full_quant(chain, rng, np_rng)
    partial = _sample_narrative_payload()
    partial.pop("creative_world_persona")
    partial.pop("network_persona")
    partial.pop("living_persona")
    partial.pop("support_persona")
    repair = {
        "creative_world_persona": "충분한 길이의 한국어 문장입니다. " * 6,
        "network_persona": "충분한 길이의 한국어 문장입니다. " * 6,
        "living_persona": "충분한 길이의 한국어 문장입니다. " * 6,
        "support_persona": "충분한 길이의 한국어 문장입니다. " * 6,
    }
    client = _Client(
        [
            json.dumps(partial, ensure_ascii=False),
            json.dumps(repair, ensure_ascii=False),
        ]
    )
    row, log, result = generate_one(
        quant,
        client=client,
        cfg=type("Cfg", (), {"model": "test-model", "max_retries": 0, "max_tokens": 1000, "temperature": 0.8, "skip_validation": False})(),
        rng=random.Random(1),
        pipeline=ValidationPipeline(),
    )
    assert row is not None
    assert log["attempts"] == 1
    assert log["repair_calls"] == 1
    assert result is not None and not result.has_errors


def test_generate_one_revises_blocking_warnings_in_max_profile() -> None:
    class _Resp:
        def __init__(self, text: str) -> None:
            self.text = text
            self.usage = type("Usage", (), {"input_tokens": 10, "output_tokens": 20})()

    class _Client:
        def __init__(self, responses: list[str]) -> None:
            self._responses = responses

        def chat(self, **kwargs):
            return _Resp(self._responses.pop(0))

    chain = build_chain_from_spec()
    rng = random.Random(2)
    np_rng = np.random.default_rng(2)
    quant = sample_full_quant(chain, rng, np_rng)
    quant["employment_type"] = "전업"
    quant["has_secondary_job"] = False
    quant["occupation"] = "음악 프로듀서"
    initial_payload = _sample_narrative_payload(
        professional_persona=(
            "음악 프로듀서로 활동하며 충분한 길이의 한국어 문장입니다. "
            "충분한 길이의 한국어 문장입니다. 충분한 길이의 한국어 문장입니다. "
            "충분한 길이의 한국어 문장입니다."
        )
    )
    initial_payload["living_persona"] = (
        "전업으로 활동하지만 세션 알바를 통해 생계를 보완한다. "
        "충분한 길이의 한국어 문장입니다. 충분한 길이의 한국어 문장입니다. "
        "충분한 길이의 한국어 문장입니다."
    )
    revised_payload = dict(initial_payload)
    revised_payload["living_persona"] = (
        "전업으로 활동하며 작업실 운영과 창작 일정에 집중한다. "
        "충분한 길이의 한국어 문장입니다. 충분한 길이의 한국어 문장입니다. "
        "충분한 길이의 한국어 문장입니다."
    )
    revised_payload["age"] = 99
    client = _Client(
        [
            json.dumps(initial_payload, ensure_ascii=False),
            json.dumps(revised_payload, ensure_ascii=False),
        ]
    )
    row, log, result = generate_one(
        quant,
        client=client,
        cfg=GenerateConfig(
            model="test-model",
            max_retries=0,
            max_tokens=1000,
            temperature=0.8,
            skip_validation=False,
            quality_profile="max",
            max_warning_revisions=1,
            enforce_hobby_plan_alignment=False,
        ),
        rng=random.Random(2),
        pipeline=ValidationPipeline(),
    )
    assert row is not None
    assert log["attempts"] == 1
    assert log["warning_revision_calls"] == 1
    assert result is not None and not result.has_errors and not result.has_warnings


def test_generate_one_deterministically_coerces_age_warning_after_failed_revision() -> None:
    class _Resp:
        def __init__(self, text: str) -> None:
            self.text = text
            self.usage = type("Usage", (), {"input_tokens": 10, "output_tokens": 20})()

    class _Client:
        def __init__(self, responses: list[str]) -> None:
            self._responses = responses

        def chat(self, **kwargs):
            return _Resp(self._responses.pop(0))

    chain = build_chain_from_spec()
    rng = random.Random(23)
    np_rng = np.random.default_rng(23)
    quant = sample_full_quant(chain, rng, np_rng)
    quant["age"] = 90
    quant["age_band"] = "70대 이상"
    payload = _sample_narrative_payload(
        professional_persona=(
            f"{quant['occupation']}로 활동하는 70대 예술인이다. "
            "충분한 길이의 한국어 문장입니다. 충분한 길이의 한국어 문장입니다. "
            "충분한 길이의 한국어 문장입니다."
        )
    )
    revised_payload = {
        "professional_persona": payload["professional_persona"],
    }
    client = _Client(
        [
            json.dumps(payload, ensure_ascii=False),
            json.dumps(revised_payload, ensure_ascii=False),
            json.dumps(revised_payload, ensure_ascii=False),
        ]
    )

    row, log, result = generate_one(
        quant,
        client=client,
        cfg=GenerateConfig(
            model="test-model",
            max_retries=0,
            max_tokens=1000,
            temperature=0.8,
            skip_validation=False,
            quality_profile="max",
            max_warning_revisions=1,
            enforce_hobby_plan_alignment=False,
        ),
        rng=random.Random(23),
        pipeline=ValidationPipeline(),
    )

    assert row is not None
    assert result is not None and not result.has_errors and not result.has_warnings
    assert log["warning_revision_calls"] == 2
    assert log["deterministic_warning_coercions"] == 1
    assert "90대 예술인" in row["professional_persona"]
    assert "70대 예술인" not in row["professional_persona"]


def test_generate_one_coerces_duplicate_hobby_set_in_max_profile_without_revision() -> None:
    class _Resp:
        def __init__(self, text: str) -> None:
            self.text = text
            self.usage = type("Usage", (), {"input_tokens": 10, "output_tokens": 20})()

    class _Client:
        def __init__(self, responses: list[str]) -> None:
            self._responses = responses

        def chat(self, **kwargs):
            return _Resp(self._responses.pop(0))

    chain = build_chain_from_spec()
    rng = random.Random(32)
    np_rng = np.random.default_rng(32)
    quant = sample_full_quant(chain, rng, np_rng)
    duplicate_payload = _sample_narrative_payload()
    duplicate_payload["hobbies_and_interests_list"] = (
        "['독서', '산책', '음악 감상', '필드 레코딩', '신시사이저 수집']"
    )
    client = _Client([json.dumps(duplicate_payload, ensure_ascii=False)])
    seen_set = _normalized_hobby_set(duplicate_payload["hobbies_and_interests_list"])
    seen_family_set = _normalized_hobby_family_set(duplicate_payload["hobbies_and_interests_list"])

    row, log, result = generate_one(
        quant,
        client=client,
        cfg=GenerateConfig(
            model="test-model",
            max_retries=0,
            max_tokens=1000,
            temperature=0.8,
            skip_validation=True,
            quality_profile="max",
        ),
        rng=random.Random(32),
        pipeline=None,
        seen_hobby_sets={seen_set},
        seen_hobby_family_sets={seen_family_set},
    )

    assert row is not None
    assert result is None
    assert log["attempts"] == 1
    assert log["duplicate_hobby_retries"] == 1
    assert log["hobby_plan_coercions"] == 1
    assert log["hobby_revision_calls"] == 0
    assert _normalized_hobby_set(row["hobbies_and_interests_list"]) != seen_set
    assert _normalized_hobby_family_set(row["hobbies_and_interests_list"]) != seen_family_set
    assert row["hobbies_and_interests"].startswith("취미와 관심사는")


def test_generate_one_rewrites_living_persona_when_template_is_too_generic() -> None:
    class _Resp:
        def __init__(self, text: str) -> None:
            self.text = text
            self.usage = type("Usage", (), {"input_tokens": 10, "output_tokens": 20})()

    class _Client:
        def __init__(self, responses: list[str]) -> None:
            self._responses = responses

        def chat(self, **kwargs):
            return _Resp(self._responses.pop(0))

    chain = build_chain_from_spec()
    rng = random.Random(21)
    np_rng = np.random.default_rng(21)
    quant = sample_full_quant(chain, rng, np_rng)
    initial_payload = _sample_narrative_payload()
    initial_payload["living_persona"] = (
        "작업실과 거주 공간을 병행하고 있다. 주중에는 작업 시간을 길게 확보하고, "
        "주말에는 회복 시간을 챙긴다. 생활 동선을 안정적으로 묶고 일상의 균형을 유지한다."
    )
    revised_payload = {
        "living_persona": (
            "촬영 장비 가방과 보정용 책상이 생활 공간 일부를 차지해, 외부 촬영이 없는 날에는 "
            "집 안 동선을 줄여 데이터 정리와 보정 작업을 몰아서 한다. 장비 유지비가 큰 달에는 "
            "다른 소비를 먼저 줄이고, 현장 일정 뒤에는 바로 보정보다 짧은 정리 시간을 두며 눈을 쉬게 한다."
        )
    }
    client = _Client(
        [
            json.dumps(initial_payload, ensure_ascii=False),
            json.dumps(revised_payload, ensure_ascii=False),
        ]
    )
    row, log, result = generate_one(
        quant,
        client=client,
        cfg=GenerateConfig(
            model="test-model",
            max_retries=0,
            max_tokens=1000,
            temperature=0.8,
            skip_validation=True,
            quality_profile="balanced",
            enforce_living_persona_specificity=True,
        ),
        rng=random.Random(21),
        pipeline=None,
    )
    assert row is not None
    assert log["living_revision_calls"] == 1
    assert result is None
    assert "장비 유지비" in row["living_persona"]


def test_generate_one_rewrites_hobby_when_exact_atom_quota_is_exceeded() -> None:
    class _Resp:
        def __init__(self, text: str) -> None:
            self.text = text
            self.usage = type("Usage", (), {"input_tokens": 10, "output_tokens": 20})()

    class _Client:
        def __init__(self, responses: list[str]) -> None:
            self._responses = responses

        def chat(self, **kwargs):
            return _Resp(self._responses.pop(0))

    chain = build_chain_from_spec()
    rng = random.Random(31)
    np_rng = np.random.default_rng(31)
    quant = sample_full_quant(chain, rng, np_rng)
    initial_payload = _sample_narrative_payload()
    initial_payload["hobbies_and_interests_list"] = (
        "['아침 카페 기록 정리', '필라테스', '로컬 식문화 탐색', '향 관련 소도구 모으기', '작은 전시 공간 방문']"
    )
    revised_payload = {
        "hobbies_and_interests": "충분한 길이의 한국어 문장입니다. " * 6,
        "hobbies_and_interests_list": (
            "['짧은 메모 산책', '자전거 타기', '생활사 에세이 읽기', '핸드드립 커피 내리기', '근린 공원 짧은 산책']"
        ),
    }
    client = _Client(
        [
            json.dumps(initial_payload, ensure_ascii=False),
            json.dumps(revised_payload, ensure_ascii=False),
        ]
    )
    row, log, result = generate_one(
        quant,
        client=client,
        cfg=GenerateConfig(
            model="test-model",
            max_retries=0,
            max_tokens=1000,
            temperature=0.8,
            skip_validation=True,
            quality_profile="balanced",
            enforce_hobby_plan_alignment=False,
            hobby_exact_atom_cap_per_batch=1,
            hobby_family_cap_per_batch=None,
        ),
        rng=random.Random(31),
        pipeline=None,
        hobby_item_counts=Counter({"아침 카페 기록 정리": 1}),
    )
    assert row is not None
    assert log["hobby_revision_calls"] == 1
    assert result is None
    assert "아침 카페 기록 정리" not in row["hobbies_and_interests_list"]


def test_generate_one_normalizes_non_atomic_hobby_list_entry() -> None:
    class _Resp:
        def __init__(self, text: str) -> None:
            self.text = text
            self.usage = type("Usage", (), {"input_tokens": 10, "output_tokens": 20})()

    class _Client:
        def __init__(self, responses: list[str]) -> None:
            self._responses = responses

        def chat(self, **kwargs):
            return _Resp(self._responses.pop(0))

    chain = build_chain_from_spec()
    rng = random.Random(41)
    np_rng = np.random.default_rng(41)
    quant = sample_full_quant(chain, rng, np_rng)
    initial_payload = _sample_narrative_payload()
    initial_payload["hobbies_and_interests_list"] = (
        "['아침 카페에서 기록 정리', '수영과 산책', '생활사 에세이 읽기', '핸드드립 커피 내리기', '작은 전시 공간 방문']"
    )
    revised_payload = {
        "hobbies_and_interests": "충분한 길이의 한국어 문장입니다. " * 6,
        "hobbies_and_interests_list": (
            "['아침 카페에서 기록 정리', '수영과 산책', '생활사 에세이 읽기', '핸드드립 커피 내리기', '작은 전시 공간 방문']"
        ),
    }
    client = _Client(
        [
            json.dumps(initial_payload, ensure_ascii=False),
            json.dumps(revised_payload, ensure_ascii=False),
        ]
    )
    row, log, result = generate_one(
        quant,
        client=client,
        cfg=GenerateConfig(
            model="test-model",
            max_retries=0,
            max_tokens=1000,
            temperature=0.8,
            skip_validation=True,
            quality_profile="balanced",
        ),
        rng=random.Random(41),
        pipeline=None,
    )
    assert row is not None
    assert log["hobby_revision_calls"] >= 1
    assert result is None
    assert row["hobbies_and_interests_list"].startswith("[")
    assert "수영과 산책" not in row["hobbies_and_interests_list"]


def test_generate_one_rewrites_generic_network_persona() -> None:
    class _Resp:
        def __init__(self, text: str) -> None:
            self.text = text
            self.usage = type("Usage", (), {"input_tokens": 10, "output_tokens": 20})()

    class _Client:
        def __init__(self, responses: list[str]) -> None:
            self._responses = responses

        def chat(self, **kwargs):
            return _Resp(self._responses.pop(0))

    chain = build_chain_from_spec()
    rng = random.Random(51)
    np_rng = np.random.default_rng(51)
    quant = sample_full_quant(chain, rng, np_rng)
    initial_payload = _sample_narrative_payload()
    initial_payload["network_persona"] = (
        "지역 예술 생태계에서 중심적인 역할을 하며 인맥을 넓히고 있다. "
        "충분한 길이의 한국어 문장입니다. 충분한 길이의 한국어 문장입니다."
    )
    revised_payload = {
        "network_persona": (
            "세션 동료와 엔지니어 사이에서 파일 전달과 일정 조율을 맡는 편이며, "
            "새로운 연결보다 자주 함께 일하는 협업선을 유지하는 데 더 신경을 쓴다. "
            "관계 자체보다 응답 속도와 역할 구분에서 마찰이 생기지 않도록 조심한다."
        )
    }
    client = _Client(
        [
            json.dumps(initial_payload, ensure_ascii=False),
            json.dumps(revised_payload, ensure_ascii=False),
        ]
    )
    row, log, result = generate_one(
        quant,
        client=client,
        cfg=GenerateConfig(
            model="test-model",
            max_retries=0,
            max_tokens=1000,
            temperature=0.8,
            skip_validation=True,
            quality_profile="balanced",
            enforce_network_persona_specificity=True,
        ),
        rng=random.Random(51),
        pipeline=None,
    )
    assert row is not None
    assert log["network_revision_calls"] == 1
    assert result is None
    assert "중심적인 역할" not in row["network_persona"]


def test_generate_one_rewrites_generic_support_persona() -> None:
    class _Resp:
        def __init__(self, text: str) -> None:
            self.text = text
            self.usage = type("Usage", (), {"input_tokens": 10, "output_tokens": 20})()

    class _Client:
        def __init__(self, responses: list[str]) -> None:
            self._responses = responses

        def chat(self, **kwargs):
            return _Resp(self._responses.pop(0))

    chain = build_chain_from_spec()
    rng = random.Random(52)
    np_rng = np.random.default_rng(52)
    quant = sample_full_quant(chain, rng, np_rng)
    initial_payload = _sample_narrative_payload()
    initial_payload["support_persona"] = (
        "지원 제도를 중요하게 여기며 실질적인 도움을 기대한다. "
        "충분한 길이의 한국어 문장입니다. 충분한 길이의 한국어 문장입니다."
    )
    revised_payload = {
        "support_persona": (
            "작은 발표 지원은 꾸준히 살펴보지만, 서류와 정산에 들어가는 시간이 길어지면 신청을 미루는 편이다. "
            "선정되더라도 제작비보다 작업 시간을 조금 벌어 주는 효과가 있을 때만 체감이 크다고 느낀다."
        )
    }
    client = _Client(
        [
            json.dumps(initial_payload, ensure_ascii=False),
            json.dumps(revised_payload, ensure_ascii=False),
        ]
    )
    row, log, result = generate_one(
        quant,
        client=client,
        cfg=GenerateConfig(
            model="test-model",
            max_retries=0,
            max_tokens=1000,
            temperature=0.8,
            skip_validation=True,
            quality_profile="balanced",
            enforce_support_persona_specificity=True,
        ),
        rng=random.Random(52),
        pipeline=None,
    )
    assert row is not None
    assert log["support_revision_calls"] == 1
    assert result is None
    assert "실질적인 도움을 기대한다" not in row["support_persona"]


def test_generate_one_rewrites_generic_family_persona() -> None:
    class _Resp:
        def __init__(self, text: str) -> None:
            self.text = text
            self.usage = type("Usage", (), {"input_tokens": 10, "output_tokens": 20})()

    class _Client:
        def __init__(self, responses: list[str]) -> None:
            self._responses = responses

        def chat(self, **kwargs):
            return _Resp(self._responses.pop(0))

    chain = build_chain_from_spec()
    rng = random.Random(53)
    np_rng = np.random.default_rng(53)
    quant = sample_full_quant(chain, rng, np_rng)
    initial_payload = _sample_narrative_payload()
    initial_payload["family_persona"] = (
        "가족과의 시간 조율을 통해 생활 책임과 작업 지속성을 함께 챙긴다. "
        "가족과의 관계를 유지하며 주변 돌봄과 자기 작업 시간을 함께 조율한다."
    )
    revised_payload = {
        "family_persona": (
            "마감이 가까운 주에는 연락 빈도를 줄이고, 끝난 뒤 약속을 몰아 잡는 편이다. "
            "생활 심부름은 한 번에 처리한 뒤 작업 시간을 길게 확보하려 하며, "
            "가까운 사람에게는 필요한 일정만 먼저 공유하는 쪽에 가깝다."
        )
    }
    client = _Client(
        [
            json.dumps(initial_payload, ensure_ascii=False),
            json.dumps(revised_payload, ensure_ascii=False),
        ]
    )
    row, log, result = generate_one(
        quant,
        client=client,
        cfg=GenerateConfig(
            model="test-model",
            max_retries=0,
            max_tokens=1000,
            temperature=0.8,
            skip_validation=True,
            quality_profile="balanced",
            enforce_family_persona_specificity=True,
        ),
        rng=random.Random(53),
        pipeline=None,
    )
    assert row is not None
    assert log["family_revision_calls"] == 1
    assert result is None
    assert "가족과의 시간 조율" not in row["family_persona"]


def test_generate_one_retries_on_duplicate_hobby_set_in_max_profile() -> None:
    class _Resp:
        def __init__(self, text: str) -> None:
            self.text = text
            self.usage = type("Usage", (), {"input_tokens": 10, "output_tokens": 20})()

    class _Client:
        def __init__(self, responses: list[str]) -> None:
            self._responses = responses

        def chat(self, **kwargs):
            return _Resp(self._responses.pop(0))

    chain = build_chain_from_spec()
    rng = random.Random(3)
    np_rng = np.random.default_rng(3)
    quant = sample_full_quant(chain, rng, np_rng)
    quant["occupation"] = "음악 프로듀서"
    duplicate_payload = _sample_narrative_payload(
        professional_persona=(
            "음악 프로듀서로 활동하며 충분한 길이의 한국어 문장입니다. "
            "충분한 길이의 한국어 문장입니다. 충분한 길이의 한국어 문장입니다."
        )
    )
    duplicate_payload["hobbies_and_interests_list"] = (
        "['독서', '산책', '음악 감상', '필드 레코딩', '신시사이저 수집']"
    )
    revised_payload = {
        "hobbies_and_interests": "충분한 길이의 한국어 문장입니다. " * 6,
        "hobbies_and_interests_list": (
            "['야간 산책', '생활사 에세이 읽기', '핸드드립 커피 내리기', '자전거 타기', '작은 전시 공간 방문']"
        ),
    }
    client = _Client(
        [
            json.dumps(duplicate_payload, ensure_ascii=False),
            json.dumps(revised_payload, ensure_ascii=False),
            json.dumps(revised_payload, ensure_ascii=False),
        ]
    )
    row, log, result = generate_one(
        quant,
        client=client,
        cfg=GenerateConfig(
            model="test-model",
            max_retries=1,
            max_tokens=1000,
            temperature=0.8,
            skip_validation=True,
            quality_profile="balanced",
        ),
        rng=random.Random(3),
        pipeline=None,
        seen_hobby_sets={_normalized_hobby_set(duplicate_payload["hobbies_and_interests_list"])},
    )
    assert row is not None
    assert log["attempts"] == 1
    assert log["duplicate_hobby_retries"] == 1
    assert log["hobby_revision_calls"] >= 1
    assert result is None
    assert row["hobbies_and_interests_list"] == (
        "['야간 산책', '생활사 에세이 읽기', '핸드드립 커피 내리기', '자전거 타기', '작은 전시 공간 방문']"
    )


def test_generate_one_rewrites_near_duplicate_hobby_family_set() -> None:
    class _Resp:
        def __init__(self, text: str) -> None:
            self.text = text
            self.usage = type("Usage", (), {"input_tokens": 10, "output_tokens": 20})()

    class _Client:
        def __init__(self, responses: list[str]) -> None:
            self._responses = responses

        def chat(self, **kwargs):
            return _Resp(self._responses.pop(0))

    chain = build_chain_from_spec()
    rng = random.Random(4)
    np_rng = np.random.default_rng(4)
    quant = sample_full_quant(chain, rng, np_rng)
    initial_payload = _sample_narrative_payload()
    initial_payload["hobbies_and_interests_list"] = (
        "['작은 전시 공간 방문', '독립서점 둘러보기', '야간 산책하며 생각 정리', '핸드드립 커피 내리기', '생활사 에세이 읽기']"
    )
    revised_payload = {
        "hobbies_and_interests": "충분한 길이의 한국어 문장입니다. " * 6,
        "hobbies_and_interests_list": (
            "['자전거 타기', '생활사 에세이 읽기', '핸드드립 커피 내리기', '문구류 정리', '작은 문화 공간 들르기']"
        ),
    }
    client = _Client(
        [
            json.dumps(initial_payload, ensure_ascii=False),
            json.dumps(revised_payload, ensure_ascii=False),
            json.dumps(revised_payload, ensure_ascii=False),
        ]
    )
    row, log, result = generate_one(
        quant,
        client=client,
        cfg=GenerateConfig(
            model="test-model",
            max_retries=1,
            max_tokens=1000,
            temperature=0.8,
            skip_validation=True,
            quality_profile="balanced",
        ),
        rng=random.Random(4),
        pipeline=None,
        seen_hobby_family_sets={_normalized_hobby_family_set(initial_payload["hobbies_and_interests_list"])},
    )
    assert row is not None
    assert log["attempts"] == 1
    assert log["duplicate_hobby_retries"] == 1
    assert log["hobby_revision_calls"] >= 1
    assert result is None
    assert row["hobbies_and_interests_list"] == (
        "['자전거 타기', '생활사 에세이 읽기', '핸드드립 커피 내리기', '문구류 정리', '작은 문화 공간 들르기']"
    )
