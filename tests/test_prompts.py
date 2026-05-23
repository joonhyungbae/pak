"""Phase 04 — per-field prompt template / linter unit tests."""

from __future__ import annotations

import random
from pathlib import Path

from pak.config import settings
from pak.prompt_builder import (
    DOMAIN_NARRATIVES,
    has_field_specific_template,
    render_narrative_prompt,
    template_coverage_report,
)
from pak.prompt_validator import (
    Issue,
    check_fallback_completeness,
    check_field_template_coverage,
    lint_all,
)
from pak.prompts_data import FIELD_META

PROMPTS_ROOT: Path = settings.project_root / "data" / "prompts"


# ----------------------------------------------------------------------------
# 70 field-specific templates + 12 fallback/common = all 82 exist / are loadable
# ----------------------------------------------------------------------------


def test_all_70_field_templates_exist() -> None:
    cov = template_coverage_report()
    for cat in DOMAIN_NARRATIVES:
        for f in FIELD_META:
            assert cov[cat][f], f"missing field-specific template: {cat}/{f}"


def test_all_5_fallbacks_exist() -> None:
    for cat in DOMAIN_NARRATIVES:
        path = PROMPTS_ROOT / cat / "_fallback.j2"
        assert path.exists(), f"missing fallback: {cat}"


def test_field_templates_render_without_error() -> None:
    rng = random.Random(0)
    base_quant = {
        "art_field_primary": "문학",
        "sex": "여자",
        "age": 38,
        "age_band": "30대",
        "province": "서울",
        "education_level": "대학원 이상",
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
    for field in FIELD_META:
        for cat in DOMAIN_NARRATIVES:
            q = {**base_quant, "art_field_primary": field}
            text = render_narrative_prompt(cat, q, rng=rng)
            assert len(text) > 200, f"{cat}/{field}: too short"
            # the field name appears in the text
            assert field in text, f"{cat}/{field}: field name not present"


# ----------------------------------------------------------------------------
# Linter self-checks
# ----------------------------------------------------------------------------


def test_linter_finds_no_errors() -> None:
    issues: list[Issue] = lint_all()
    errors = [i for i in issues if i.severity == "error"]
    assert not errors, "\n".join(str(i) for i in errors)


def test_linter_detects_jinja_syntax_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.j2"
    bad.write_text("{% if x %}", encoding="utf-8")  # not closed
    # path can't be made relative_to PROMPTS_ROOT, so trick: call directly
    from pak.prompt_validator import _check_jinja_syntax

    err = _check_jinja_syntax(bad.read_text(encoding="utf-8"))
    assert err is not None


def test_linter_detects_cliche_outside_forbidden() -> None:
    from pak.prompt_validator import _find_cliches_outside_forbidden_section

    src = "이 사람은 가난하지만 자유로운 보헤미안이다."
    hits = _find_cliches_outside_forbidden_section(src)
    assert len(hits) >= 2, f"got {hits}"


def test_linter_allows_cliches_in_forbidden_section() -> None:
    from pak.prompt_validator import _find_cliches_outside_forbidden_section

    src = "[금지]\n다음 표현 금지: 가난하지만 자유로운, 고독한 천재, 보헤미안\n"
    hits = _find_cliches_outside_forbidden_section(src)
    assert hits == []


# ----------------------------------------------------------------------------
# Coverage / fallback completeness
# ----------------------------------------------------------------------------


def test_field_template_coverage_full() -> None:
    issues = check_field_template_coverage()
    assert issues == [], "\n".join(str(i) for i in issues)


def test_fallback_completeness() -> None:
    assert check_fallback_completeness() == []


def test_has_field_specific_template_helper() -> None:
    assert has_field_specific_template("professional", "문학")
    assert not has_field_specific_template("professional", "조각")  # nonexistent field
