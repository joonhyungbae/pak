"""Phase 03 — schema / sampler / prompt_builder unit tests."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from pak.prompt_builder import (
    COMMON_NARRATIVES,
    DOMAIN_NARRATIVES,
    render_all_for_persona,
    render_narrative_prompt,
)
from pak.samplers import build_chain_from_spec, sample_age_in_band, sample_career_in_band
from pak.schema import (
    PAKPersona,
    PAKPersonaNarrative,
    PAKPersonaQuant,
    write_json_schema,
)

# ----------------------------------------------------------------------------
# Schema round-trip
# ----------------------------------------------------------------------------


def _sample_quant_dict() -> dict:
    return {
        "pak_uuid": "00000000-0000-0000-0000-000000000001",
        # NPK compatible
        "sex": "여자",
        "age": 38,
        "province": "서울",
        "district": "서울-강남구",
        "country": "대한민국",
        "education_level": "대학원",
        "occupation": "시인",
        "marital_status": None,
        "military_status": None,
        "family_type": None,
        "housing_type": None,
        "bachelors_field": None,
        # PAK domain
        "age_band": "30대",
        "education_level_pak": "대학원 이상",
        "art_field_primary": "문학",
        "career_years": 12,
        "career_band": "10-20년 미만",
        "employment_type": "겸업",
        "is_freelance": True,
        "has_secondary_job": True,
        "individual_art_income_bracket": "5백-1천만원 미만",
        "household_income_bracket": "4-5천만원 미만",
        "has_contract_experience": True,
        "uses_standard_contract": False,
        "has_copyright": True,
        "had_career_break": False,
        "has_overseas_experience": True,
    }


def _sample_narrative_dict() -> dict:
    long = "이것은 충분히 길어 검증 통과를 위한 예시 문장입니다. " * 5
    list_str = "['항목 하나', '항목 둘', '항목 셋']"
    return {
        # NPK 7 main
        "persona": "이 사람은 30대 문학인입니다. 충분히 긴 한 줄 요약 페르소나입니다.",
        "professional_persona": long,
        "sports_persona": long,
        "arts_persona": long,
        "travel_persona": long,
        "culinary_persona": long,
        "family_persona": long,
        # NPK 6 attribute (including list variants)
        "cultural_background": long,
        "skills_and_expertise": long,
        "skills_and_expertise_list": list_str,
        "hobbies_and_interests": long,
        "hobbies_and_interests_list": list_str,
        "career_goals_and_ambitions": long,
        # PAK domain 4
        "creative_world_persona": long,
        "network_persona": long,
        "living_persona": long,
        "support_persona": long,
    }


def test_quant_round_trip() -> None:
    q = PAKPersonaQuant.model_validate(_sample_quant_dict())
    s = q.model_dump()
    q2 = PAKPersonaQuant.model_validate(s)
    assert q == q2


def test_narrative_round_trip() -> None:
    n = PAKPersonaNarrative.model_validate(_sample_narrative_dict())
    n2 = PAKPersonaNarrative.model_validate(n.model_dump())
    assert n == n2


def test_persona_combined_round_trip() -> None:
    full = {**_sample_quant_dict(), **_sample_narrative_dict()}
    p = PAKPersona.model_validate(full)
    assert p.art_field_primary == "문학"
    assert len(p.professional_persona) > 80


def test_invalid_field_rejected() -> None:
    bad = _sample_quant_dict()
    bad["art_field_primary"] = "조각"  # value not in the 14-item enum
    with pytest.raises(Exception):
        PAKPersonaQuant.model_validate(bad)


def test_invalid_age_rejected() -> None:
    bad = _sample_quant_dict()
    bad["age"] = 200
    with pytest.raises(Exception):
        PAKPersonaQuant.model_validate(bad)


def test_json_schema_writes(tmp_path: Path) -> None:
    out = write_json_schema(target=tmp_path / "schema.json")
    schema = json.loads(out.read_text(encoding="utf-8"))
    assert "properties" in schema
    assert "art_field_primary" in schema["properties"]


# ----------------------------------------------------------------------------
# Sampler chain
# ----------------------------------------------------------------------------


def test_sampler_chain_loads() -> None:
    chain = build_chain_from_spec()
    samples = chain.sample_many(20, seed=42)
    assert len(samples) == 20
    fields = {s["art_field_primary"] for s in samples}
    # With ~10,000 rows all 14 appear; with only 20 the sample is small, so just a subset.
    assert fields.issubset(
        {
            "문학",
            "미술",
            "공예",
            "사진",
            "건축",
            "음악",
            "국악",
            "대중음악",
            "방송연예",
            "무용",
            "연극",
            "영화",
            "만화",
            "기타",
        }
    )


def test_sampler_deterministic_with_seed() -> None:
    a = build_chain_from_spec().sample_many(5, seed=20260502)
    b = build_chain_from_spec().sample_many(5, seed=20260502)
    assert a == b


def test_sample_age_band_helpers() -> None:
    import numpy as np

    rng_np = np.random.default_rng(0)
    age = sample_age_in_band(rng_np, "30대")
    assert 30 <= age <= 39
    car = sample_career_in_band(rng_np, "10-20년 미만")
    assert 10 <= car <= 19


# ----------------------------------------------------------------------------
# Prompt builder
# ----------------------------------------------------------------------------


def test_prompt_builder_fallback_works_for_all_fields() -> None:
    rng = random.Random(0)
    base = _sample_quant_dict()
    fields = [
        "문학",
        "미술",
        "공예",
        "사진",
        "건축",
        "음악",
        "국악",
        "대중음악",
        "방송연예",
        "무용",
        "연극",
        "영화",
        "만화",
        "기타",
    ]
    for f in fields:
        q = {**base, "art_field_primary": f}
        for cat in DOMAIN_NARRATIVES:
            text = render_narrative_prompt(cat, q, rng=rng)
            assert f in text, f"{cat}/{f}: field name not in rendered prompt"


def test_prompt_builder_renders_all_12_categories() -> None:
    rng = random.Random(0)
    out = render_all_for_persona(_sample_quant_dict(), rng=rng)
    expected = set(DOMAIN_NARRATIVES) | set(COMMON_NARRATIVES)
    assert set(out.keys()) == expected
    for k, v in out.items():
        assert len(v) > 50, f"{k} too short"


def test_prompt_builder_required_quant_vars_present() -> None:
    rng = random.Random(0)
    q = _sample_quant_dict()
    out = render_all_for_persona(q, rng=rng)
    # core quantitative variables appear in at least one prompt
    combined = "\n".join(out.values())
    for var in ["art_field_primary", "sex", "age", "career_years", "province"]:
        # check the value, not the variable name itself
        assert str(q[var]) in combined, f"{var}={q[var]!r} not appearing in any prompt"
