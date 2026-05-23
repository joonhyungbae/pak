"""Phase 05 — validation module unit tests."""

from __future__ import annotations

import pytest

from pak.validators import ValidationPipeline, summarize_batch
from pak.validators.cliche import CLICHE_PATTERNS, cliche_frequency, detect_cliches
from pak.validators.consistency import (
    check_age_consistency,
    check_age_career_feasibility,
    check_career_consistency,
    check_contract_consistency,
    check_cultural_background_timeline,
    check_employment_duration_conflation,
    check_employment_consistency,
    check_field_vocabulary,
    check_income_consistency,
    check_late_debut_awareness,
    check_occupation_consistency,
    check_region_consistency,
)
from pak.validators.distribution import (
    check_marginal_categorical,
    chi_square_test,
    ks_test,
)
from pak.validators.diversity import all_unique, jaccard, pairwise_similarity_token
from pak.validators.llm_judge import (
    Judgment,
    JudgmentScores,
    extract_json_object,
    parse_judgment,
)

# ----------------------------------------------------------------------------
# Consistency
# ----------------------------------------------------------------------------


def test_age_consistency_detects_mismatch() -> None:
    quant = {"age": 25, "age_band": "20대"}
    nar = {"professional_persona": "60대 후반의 노년 작가"}
    issues = check_age_consistency(quant, nar)
    assert any(i.code == "AGE_MISMATCH" for i in issues)


def test_age_consistency_ok_when_matched() -> None:
    quant = {"age": 35, "age_band": "30대"}
    nar = {"professional_persona": "30대 중반의 작가"}
    assert check_age_consistency(quant, nar) == []


def test_age_consistency_detects_elder_band_mismatch() -> None:
    quant = {"age": 90, "age_band": "70대 이상"}
    nar = {"professional_persona": "70대 남자 판화가로 지역 전시에 꾸준히 참여한다."}
    issues = check_age_consistency(quant, nar)
    assert any(i.code == "AGE_MISMATCH" for i in issues)


def test_age_consistency_detects_seventy_plus_phrase_for_eighty_persona() -> None:
    quant = {"age": 87, "age_band": "70대 이상"}
    nar = {"professional_persona": "70대 이상의 판화가로 작업실을 오래 지켜 왔다."}
    issues = check_age_consistency(quant, nar)
    assert any(i.code == "AGE_MISMATCH" for i in issues)


def test_age_consistency_ok_for_eighty_band_expression() -> None:
    quant = {"age": 87, "age_band": "70대 이상"}
    nar = {"professional_persona": "80대 후반의 판화가로 작업실을 오래 지켜 왔다."}
    assert check_age_consistency(quant, nar) == []


def test_age_consistency_ignores_past_debut_age_reference() -> None:
    quant = {"age": 65, "age_band": "60대"}
    nar = {"professional_persona": "20대 후반에 신춘문예 당선으로 등단한 뒤 지금까지 활동해왔다."}
    assert check_age_consistency(quant, nar) == []


def test_age_consistency_ignores_family_member_age_reference() -> None:
    quant = {"age": 55, "age_band": "50대"}
    nar = {"family_persona": "이 사람은 20대 자녀와 함께 살며 가족의 응원을 받고 있다."}
    assert check_age_consistency(quant, nar) == []


def test_age_consistency_ignores_taste_age_reference() -> None:
    quant = {"age": 46, "age_band": "40대"}
    nar = {"arts_persona": "그는 30대의 취향에 가까운 클래식과 독립영화를 즐긴다."}
    assert check_age_consistency(quant, nar) == []


def test_age_career_feasibility_detects_impossible_combo() -> None:
    quant = {"art_field_primary": "문학", "age": 21, "career_years": 15}
    issues = check_age_career_feasibility(quant, {})
    assert any(i.code == "AGE_CAREER_IMPOSSIBLE" for i in issues)


def test_age_career_feasibility_uses_activity_career_floor_for_gugak() -> None:
    quant = {"art_field_primary": "국악", "age": 26, "career_years": 20}
    issues = check_age_career_feasibility(quant, {})
    assert any(i.code == "AGE_CAREER_IMPOSSIBLE" for i in issues)


def test_career_consistency_veteran_for_new_artist() -> None:
    quant = {"career_years": 2}
    nar = {"professional_persona": "후배 양성에 헌신하는 원로"}
    issues = check_career_consistency(quant, nar)
    assert any(i.code == "CAREER_MISMATCH_VETERAN" for i in issues)


def test_career_consistency_new_for_old_artist() -> None:
    quant = {"career_years": 30}
    nar = {"professional_persona": "데뷔 직후의 신진 작가"}
    issues = check_career_consistency(quant, nar)
    assert any(i.code == "CAREER_MISMATCH_NEW" for i in issues)


def test_career_consistency_ignores_shinjin_exhibition_phrase() -> None:
    quant = {"career_years": 30}
    nar = {"professional_persona": "갤러리 신진작가전에 참여하며 활동 반경을 넓혀왔다."}
    issues = check_career_consistency(quant, nar)
    assert not any(i.code == "CAREER_MISMATCH_NEW" for i in issues)


def test_career_consistency_ignores_shininsang_entry_phrase() -> None:
    quant = {"career_years": 23}
    nar = {"professional_persona": "신인상 당선 경로로 입문한 뒤 오랜 기간 활동해왔다."}
    issues = check_career_consistency(quant, nar)
    assert not any(i.code == "CAREER_MISMATCH_NEW" for i in issues)


def test_late_debut_awareness_flags_tradition_depth_for_late_debut() -> None:
    quant = {"age": 85, "career_years": 6, "art_field_primary": "국악"}
    nar = {"creative_world_persona": "전통의 깊이와 동시대성 사이의 균형을 추구한다."}
    issues = check_late_debut_awareness(quant, nar)
    assert any(i.code == "LATE_DEBUT_TONE_MISMATCH" for i in issues)


def test_late_debut_awareness_allows_tradition_depth_for_lifetime_career() -> None:
    quant = {"age": 85, "career_years": 70, "art_field_primary": "국악"}
    nar = {"creative_world_persona": "전통의 깊이와 동시대성 사이의 균형을 추구한다."}
    assert check_late_debut_awareness(quant, nar) == []


def test_late_debut_awareness_allows_tradition_depth_for_young_debut_age() -> None:
    quant = {"age": 35, "career_years": 5, "art_field_primary": "국악"}
    nar = {"creative_world_persona": "전통의 깊이와 동시대성 사이의 균형을 추구한다."}
    assert check_late_debut_awareness(quant, nar) == []


def test_late_debut_awareness_allows_recent_start_phrase() -> None:
    quant = {"age": 60, "career_years": 2, "art_field_primary": "무용"}
    nar = {"professional_persona": "최근에 시작해 지역 워크숍과 발표회를 차근히 넓히고 있다."}
    assert check_late_debut_awareness(quant, nar) == []


def test_late_debut_awareness_flags_decades_phrase() -> None:
    quant = {"age": 70, "career_years": 3, "art_field_primary": "무용"}
    nar = {"network_persona": "수십 년 동안 이어 온 협업선을 바탕으로 공연을 조율한다."}
    issues = check_late_debut_awareness(quant, nar)
    assert any(i.code == "LATE_DEBUT_TONE_MISMATCH" for i in issues)


def test_pipeline_prioritizes_late_debut_warning_over_generic_veteran_warning() -> None:
    pipe = ValidationPipeline()
    res = pipe.validate_one(
        pak_uuid="late",
        quant={"age": 70, "career_years": 3, "art_field_primary": "무용"},
        narratives={"network_persona": "수십 년 동안 이어 온 협업선을 바탕으로 공연을 조율한다."},
    )
    codes = [issue.code for issue in res.consistency_issues]
    assert "LATE_DEBUT_TONE_MISMATCH" in codes
    assert "CAREER_MISMATCH_VETERAN" not in codes


def test_cultural_background_timeline_flags_early_interest_for_late_debut() -> None:
    quant = {"age": 62, "career_years": 4, "art_field_primary": "문학"}
    nar = {"cultural_background": "어릴 적부터 문학과 창작에 관심을 두고 지냈다."}
    issues = check_cultural_background_timeline(quant, nar)
    assert any(i.code == "EARLY_INTEREST_VS_LATE_DEBUT" for i in issues)


def test_cultural_background_timeline_flags_education_start_for_late_debut() -> None:
    quant = {"age": 65, "career_years": 5, "art_field_primary": "문학"}
    nar = {"cultural_background": "고등학교를 졸업한 후 문학을 시작했으며 지역에서 자랐다."}
    issues = check_cultural_background_timeline(quant, nar)
    assert any(i.code == "EDUCATION_DEBUT_TIMELINE_MISMATCH" for i in issues)


def test_cultural_background_timeline_allows_same_text_for_early_debut() -> None:
    quant = {"age": 65, "career_years": 45, "art_field_primary": "문학"}
    nar = {"cultural_background": "어릴 적부터 문학에 관심이 있었고 대학 졸업 후 문학 활동을 시작했다."}
    assert check_cultural_background_timeline(quant, nar) == []


def test_cultural_background_timeline_ignores_childhood_interest_without_field() -> None:
    quant = {"age": 62, "career_years": 4, "art_field_primary": "문학"}
    nar = {"cultural_background": "어릴 적부터 책을 읽었다. 50대 후반에 문학 활동을 시작했다."}
    issues = check_cultural_background_timeline(quant, nar)
    assert not any(i.code == "EARLY_INTEREST_VS_LATE_DEBUT" for i in issues)


def test_cultural_background_timeline_detects_phase_06l_case_2() -> None:
    quant = {"age": 58, "career_years": 3, "art_field_primary": "공예"}
    nar = {
        "cultural_background": (
            "강원 지역에서 자랐으며, 어릴 적부터 공예 관련 활동에 관심을 가졌다. "
            "대학에서 공예 관련 전공을 하며, 그 후 공예 분야에 진출하였다."
        )
    }
    codes = {i.code for i in check_cultural_background_timeline(quant, nar)}
    assert codes == {"EARLY_INTEREST_VS_LATE_DEBUT", "EDUCATION_DEBUT_TIMELINE_MISMATCH"}


def test_cultural_background_timeline_detects_phase_06l_case_4() -> None:
    quant = {"age": 65, "career_years": 5, "art_field_primary": "문학"}
    nar = {
        "cultural_background": (
            "고등학교를 졸업한 후 문학을 시작했으며, 대전에서 자랐다. "
            "어릴 적부터 책을 읽고, 문학을 통해 감정을 표현하는 방식에 흥미를 느꼈다. "
            "문학은 일상의 일면을 보완하는 도구로 자리 잡았다."
        )
    }
    codes = {i.code for i in check_cultural_background_timeline(quant, nar)}
    assert codes == {"EARLY_INTEREST_VS_LATE_DEBUT", "EDUCATION_DEBUT_TIMELINE_MISMATCH"}


def test_cultural_background_timeline_allows_explicit_debut_age_in_text() -> None:
    """Phase 06n v3 false positive: '4년제 대학을 졸업한 후 61세 무렵에 사진으로 전환' (graduated from a 4-year college then switched to photography around age 61)."""
    quant = {"age": 90, "career_years": 29, "art_field_primary": "사진"}
    nar = {
        "cultural_background": (
            "4년제 대학을 졸업한 후 61세 무렵에 사진 작업으로 전환한 여자는, "
            "이전에는 지역 사회에서의 활동과 가족 생활을 이어가며 사진에 대한 열정을 키워왔다."
        )
    }
    assert check_cultural_background_timeline(quant, nar) == []


def test_cultural_background_timeline_allows_education_with_concession_clause() -> None:
    """Phase 06n v3 false positive: '대학원에서 건축을 전공했으나, ... 52세 무렵 ... 시작' (majored in architecture in grad school, but ... started around age 52)."""
    quant = {"age": 82, "career_years": 30, "art_field_primary": "건축"}
    nar = {
        "cultural_background": (
            "52세 무렵에 건축으로 전환한 건축가로, 이전에는 다른 직업을 하며 생활했다. "
            "대학원에서 건축을 전공했으나, 일찍이 다른 분야에서 경험을 쌓은 뒤 "
            "서울에서 건축사로 시작했다."
        )
    }
    assert check_cultural_background_timeline(quant, nar) == []


def test_cultural_background_timeline_allows_negation_of_field_major() -> None:
    """Phase 06n v3 false positive: '미술 전공을 하지 않았으나' (did not major in fine art, but)."""
    quant = {"age": 59, "career_years": 19, "art_field_primary": "미술"}
    nar = {
        "cultural_background": (
            "울산에서 태어나 4년제 대학교를 졸업한 59세 여성은 40대 초반까지 다른 직업을 "
            "하며 생활했으나, 미술에 대한 열정으로 40대 후반에 예술가로 전환했다. "
            "대학 시절 미술 전공을 하지 않았으나, 개인적인 경험과 작업을 통해 "
            "예술의 세계에 입문했다."
        )
    }
    assert check_cultural_background_timeline(quant, nar) == []


def test_cultural_background_timeline_decade_match_counts_as_explicit() -> None:
    """Also allow the case where debut_age is 50 and the narrative only specifies it in decade units, such as '50대 초반에 시작' (started in the early 50s)."""
    quant = {"age": 65, "career_years": 15, "art_field_primary": "미술"}
    nar = {
        "cultural_background": (
            "전북에서 자랐으며, 고등학교를 졸업한 후 다른 일을 하다가 "
            "50세 무렵 미술로 전환했다."
        )
    }
    assert check_cultural_background_timeline(quant, nar) == []


def test_employment_duration_conflation_detects_years_before_full_time() -> None:
    quant = {"career_years": 41, "employment_type": "전업"}
    nar = {"professional_persona": "41년 전업 금속공예가로 주문 제작과 공방 직판을 이어간다."}
    issues = check_employment_duration_conflation(quant, nar)
    assert any(i.code == "EMPLOYMENT_DURATION_CONFLATION" for i in issues)


def test_employment_duration_conflation_allows_current_status_split_from_career() -> None:
    quant = {"career_years": 12, "employment_type": "전업"}
    nar = {"professional_persona": "활동 경력은 12년이며, 현재는 전업으로 작업 시간을 확보한다."}
    issues = check_employment_duration_conflation(quant, nar)
    assert not any(i.code == "EMPLOYMENT_DURATION_CONFLATION" for i in issues)


def test_field_vocabulary_foreign_term_caught() -> None:
    quant = {"art_field_primary": "문학"}
    nar = {"creative_world_persona": "안무를 연구하며 BIM 도면을 그린다"}
    issues = check_field_vocabulary(quant, nar)
    assert any(i.code == "FIELD_FOREIGN_VOCAB" for i in issues)


def test_occupation_consistency_detects_missing_anchor() -> None:
    quant = {"occupation": "지휘자"}
    nar = {
        "persona": "이 사람은 40대 음악가다.",
        "professional_persona": "현악기 연주자로 오케스트라와 실내악 무대에 선다.",
    }
    issues = check_occupation_consistency(quant, nar)
    assert any(i.code == "OCCUPATION_MISMATCH" for i in issues)


def test_occupation_consistency_allows_field_level_anchor_for_hoehwa_artist() -> None:
    quant = {"occupation": "회화 작가"}
    nar = {
        "persona": "이 사람은 50대 미술 작가다.",
        "professional_persona": "개인전과 단체전에서 작품을 발표하며 미술 활동을 이어간다.",
    }
    issues = check_occupation_consistency(quant, nar)
    assert not any(i.code == "OCCUPATION_MISMATCH" for i in issues)


def test_occupation_consistency_allows_space_variant_for_film_director() -> None:
    quant = {"occupation": "영화감독"}
    nar = {
        "persona": "이 사람은 20대 영화 감독이다.",
        "professional_persona": "독립영화와 장편 연출을 중심으로 활동한다.",
    }
    issues = check_occupation_consistency(quant, nar)
    assert not any(i.code == "OCCUPATION_MISMATCH" for i in issues)


def test_region_consistency_detects_other_region() -> None:
    quant = {"province": "경기"}
    nar = {"persona": "이 사람은 경기에서 활동하며, 서울 대학로를 주 활동 무대로 삼는다."}
    issues = check_region_consistency(quant, nar)
    assert any(i.code in {"REGION_MISMATCH", "LANDMARK_REGION_MISMATCH"} for i in issues)


def test_employment_consistency_detects_secondary_job_phrase() -> None:
    quant = {"employment_type": "전업", "has_secondary_job": False}
    nar = {"living_persona": "그는 전업이지만 강의 수입과 외주 업무를 병행하며 생계를 유지한다."}
    issues = check_employment_consistency(quant, nar)
    assert any(i.code == "EMPLOYMENT_MISMATCH" for i in issues)


def test_employment_consistency_ignores_generic_workshop_activity() -> None:
    quant = {"employment_type": "전업", "has_secondary_job": False}
    nar = {"living_persona": "그는 전업으로 활동하며 지역 워크숍과 강연 프로그램을 작품 연계 활동으로 진행한다."}
    issues = check_employment_consistency(quant, nar)
    assert not any(i.code == "EMPLOYMENT_MISMATCH" for i in issues)


def test_region_consistency_allows_daehakro_without_landmark_warning() -> None:
    quant = {"province": "경기"}
    nar = {"professional_persona": "경기에서 활동하며 대학로 소극장 무대에도 자주 오른다."}
    issues = check_region_consistency(quant, nar)
    assert not any(i.code == "LANDMARK_REGION_MISMATCH" for i in issues)


def test_region_consistency_ignores_festival_region_reference() -> None:
    quant = {"province": "서울"}
    nar = {"professional_persona": "서울을 기반으로 활동하며 부산국제영화제와 여러 영화제 프로그램에 참여한다."}
    issues = check_region_consistency(quant, nar)
    assert not any(i.code == "REGION_MISMATCH" for i in issues)


def test_income_consistency_rich_phrase_in_low_bracket() -> None:
    quant = {"individual_art_income_bracket": "없음"}
    nar = {"living_persona": "전업 수입만으로 안정적 수입을 유지한다"}
    issues = check_income_consistency(quant, nar)
    assert any(i.code == "INCOME_RICH_BUT_LOW" for i in issues)


def test_contract_consistency_no_contract_but_mention() -> None:
    quant = {"has_contract_experience": False, "has_copyright": False}
    nar = {"professional_persona": "표준계약서를 매번 작성한다"}
    issues = check_contract_consistency(quant, nar)
    assert any(i.code == "CONTRACT_MISMATCH" for i in issues)


# ----------------------------------------------------------------------------
# Cliche
# ----------------------------------------------------------------------------


def test_detect_cliches_finds_known_patterns() -> None:
    nar = {"creative_world_persona": "그는 가난하지만 자유로운 보헤미안이다."}
    hits = detect_cliches(nar)
    labels = {h.label for h in hits}
    assert "가난하지만 자유로운" in labels
    assert "보헤미안" in labels


def test_detect_cliches_clean_text() -> None:
    nar = {"professional_persona": "30대 여성 작가가 시집을 출간했다."}
    assert detect_cliches(nar) == []


def test_cliche_frequency_returns_all_patterns() -> None:
    personas = [
        {"creative_world_persona": "고독한 천재의 작업"},
        {"creative_world_persona": "보헤미안적 삶"},
        {"creative_world_persona": "성실한 작업"},
    ]
    freq = cliche_frequency(personas)
    assert set(freq.keys()) == {label for label, _ in CLICHE_PATTERNS}
    assert freq["고독한 천재"] == pytest.approx(1 / 3)
    assert freq["보헤미안"] == pytest.approx(1 / 3)


# ----------------------------------------------------------------------------
# Diversity
# ----------------------------------------------------------------------------


def test_jaccard_identical() -> None:
    assert jaccard("문학 작가의 일상", "문학 작가의 일상") == 1.0


def test_jaccard_disjoint() -> None:
    assert jaccard("문학 작가", "BIM 도면") == 0.0


def test_pairwise_similarity_returns_report() -> None:
    texts = [
        "문학 작가가 시집을 출간했다",
        "음악가가 콩쿠르에서 입상했다",
        "건축가가 사무소를 운영한다",
    ]
    rep = pairwise_similarity_token(texts, sample_pairs=3)
    assert rep.n == 3
    assert rep.pairs_compared == 3
    assert 0 <= rep.mean_similarity <= 1


def test_all_unique_detects_duplicates() -> None:
    same = "동일한 텍스트"
    nars = [{"a": same}, {"a": same}]
    assert not all_unique(nars)


# ----------------------------------------------------------------------------
# Distribution
# ----------------------------------------------------------------------------


def test_chi_square_passes_when_close() -> None:
    expected = {"문학": 0.5, "미술": 0.5}
    observed = {"문학": 50, "미술": 50}
    res = chi_square_test(observed, expected)
    assert res.passed
    assert res.p_value > 0.05


def test_chi_square_fails_when_far_off() -> None:
    expected = {"문학": 0.5, "미술": 0.5}
    observed = {"문학": 90, "미술": 10}
    res = chi_square_test(observed, expected)
    assert not res.passed or res.effect_size > 0.05


def test_check_marginal_categorical() -> None:
    expected = {"문학": 0.3, "미술": 0.7}
    generated = ["문학"] * 30 + ["미술"] * 70
    res = check_marginal_categorical("art_field_primary", generated, expected)
    assert res.passed
    assert res.variable == "art_field_primary"


def test_ks_test_same_distribution() -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 200).tolist()
    b = rng.normal(0, 1, 200).tolist()
    res = ks_test(a, b)
    assert res.passed


# ----------------------------------------------------------------------------
# LLM judge — no calls made, only the parser is verified
# ----------------------------------------------------------------------------


def test_extract_json_from_fenced_response() -> None:
    text = """여기 결과:
```json
{"a": 1, "b": "x"}
```
끝.
"""
    out = extract_json_object(text)
    assert "a" in out and "b" in out


def test_parse_judgment_minimal() -> None:
    raw = (
        '{"scores": {"realism": 8, "consistency": 9, "field_appropriateness": 7,'
        ' "diversity": 7, "policy_utility": 7}, "overall_score": 7.7,'
        ' "detected_issues": ["boring_generic"], "suggestions": ["..."],'
        ' "summary": "괜찮음"}'
    )
    j = parse_judgment(raw)
    assert isinstance(j, Judgment)
    assert isinstance(j.scores, JudgmentScores)
    assert j.scores.realism == 8


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------


def test_pipeline_validate_one() -> None:
    pipe = ValidationPipeline()
    quant = {
        "art_field_primary": "문학",
        "age": 35,
        "career_years": 10,
        "individual_art_income_bracket": "1-2천만원 미만",
        "has_contract_experience": True,
        "has_copyright": True,
    }
    nar = {"professional_persona": "30대 중반의 작가가 단편집을 출간했다."}
    res = pipe.validate_one(pak_uuid="u1", quant=quant, narratives=nar)
    assert res.pak_uuid == "u1"
    assert not res.has_errors


def test_pipeline_summarize_batch() -> None:
    pipe = ValidationPipeline()
    items = [
        (
            "u1",
            {
                "art_field_primary": "문학",
                "age": 30,
                "career_years": 5,
                "individual_art_income_bracket": "1-2천만원 미만",
                "has_contract_experience": True,
                "has_copyright": True,
            },
            {"creative_world_persona": "성실한 작업"},
        ),
        (
            "u2",
            {
                "art_field_primary": "문학",
                "age": 30,
                "career_years": 5,
                "individual_art_income_bracket": "1-2천만원 미만",
                "has_contract_experience": True,
                "has_copyright": True,
            },
            {"creative_world_persona": "보헤미안의 작품 세계"},
        ),
    ]
    results = pipe.validate_batch(items)
    summary = summarize_batch(results)
    assert summary["n"] == 2
    assert summary["cliche_count"] >= 1
