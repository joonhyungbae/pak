"""Per-field narrative prompt builder (Jinja2 + fallback).

5 domain narratives x 14 fields + fallback = 70+5 templates planned for Phase 04.
v0.1 by default uses only the fallback templates. Once per-field templates are
added they are automatically preferred.

The 7 common narratives use a single ``_common/`` template with no per-field branching.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound

from pak.config import settings

logger = logging.getLogger(__name__)


PROMPTS_ROOT = settings.project_root / "data" / "prompts"

DOMAIN_NARRATIVES: tuple[str, ...] = (
    "professional",
    "creative_world",
    "network",
    "living",
    "support",
)

COMMON_NARRATIVES: tuple[str, ...] = (
    # NPK-compatible (qualitative)
    "persona",
    "professional",  # NOTE: name clash with the domain narrative — domain takes priority, so not included here
    "sports_persona",
    "arts_persona",
    "travel_persona",
    "culinary_persona",
    "family_persona",
    "cultural_background",
    "skills_and_expertise",
    "skills_and_expertise_list",
    "hobbies_and_interests",
    "hobbies_and_interests_list",
    "career_goals_and_ambitions",
)


# Remove the item that clashes with a domain category ("professional" is a domain category)
COMMON_NARRATIVES = tuple(c for c in COMMON_NARRATIVES if c != "professional")


# Per-field diversity seed pools (to be expanded in Phase 04)
SEED_POOLS: dict[str, list[str]] = {
    "문학": [
        "기억",
        "도시",
        "가족",
        "이주",
        "노동",
        "병",
        "잔존",
        "정적",
        "방언",
        "유년",
        "부재",
        "복구",
        "일상",
        "전환",
        "돌봄",
    ],
    "미술": [
        "선",
        "공간",
        "신체",
        "물질",
        "흔적",
        "겹침",
        "비움",
        "흙",
        "도구",
        "수공",
        "측량",
        "복제",
        "사라짐",
        "낯섦",
    ],
    "공예": [
        "손",
        "결",
        "재료",
        "오랜 시간",
        "균열",
        "수리",
        "도자",
        "목리",
        "실험",
        "전통",
        "균형",
    ],
    "사진": ["순간", "노출", "거리", "익명", "골목", "동시대", "기록", "사라짐", "잔상", "구조"],
    "건축": ["공간", "구조", "재료", "빛", "바람", "동선", "비례", "땅", "공공"],
    "음악": [
        "호흡",
        "긴장",
        "여백",
        "공명",
        "리듬",
        "기억",
        "민요",
        "현대",
        "협주",
        "독주",
        "지속음",
        "침묵",
    ],
    "국악": ["가락", "장단", "사랑가", "정가", "소리", "전승", "재해석", "기억", "대지", "물"],
    "대중음악": ["야간", "도시", "고향", "관계", "작업실", "투어", "녹음", "세션", "열기", "긴장"],
    "방송연예": ["카메라 앞", "대기실", "리허설", "객석", "대본", "캐릭터", "휴식기", "복귀"],
    "무용": ["몸", "중력", "균형", "관계", "동선", "호흡", "공간", "리허설", "무대"],
    "연극": ["대본", "리허설", "객석", "분장", "암전", "첫공", "관계", "희극", "비극"],
    "영화": ["프레임", "롱테이크", "현장", "후반작업", "기다림", "첫 상영", "촬영지", "기록"],
    "만화": ["연재", "마감", "캐릭터", "콘티", "톤", "댓글", "휴재", "장르", "설정"],
    "기타": ["경계", "융합", "매개", "행정", "잡일", "사이"],
}

_PROMPT_DEFAULTS: dict[str, object] = {
    "workspace_mode": "작업 방식이 생활 리듬에 영향을 주는 편",
    "weekly_rhythm": "한 주 단위로 작업과 회복 시간을 조정하는 편",
    "housing_pressure": "생활 기반과 작업 지속성을 함께 살피는 편",
    "space_anchor": "작업 공간과 거주 공간의 배치를 계속 손보는 편",
    "expense_anchor": "고정 지출과 작업 비용이 겹치는 시기를 자주 계산하는 편",
    "recovery_anchor": "짧게 쉬는 방식까지 작업 리듬 안에 포함하는 편",
    "family_contact_style": "가까운 사람들과 일정 조율을 해 가며 지내는 편",
    "family_rhythm": "작업 블록을 먼저 잡아 두고 가까운 약속은 그 빈칸에 맞추는 편",
    "family_boundary": "작업 얘기는 길게 가져가기보다 필요한 일정만 짧게 공유하는 편",
    "family_responsibility": "생활 책임은 몰아서 처리한 뒤 작업 시간을 지키려는 편",
    "local_routine_hint": "동네 산책과 서점 방문 같은 소규모 루틴",
    "support_need": "시간과 비용을 아껴 주는 실질 지원을 중요하게 여기는 편",
    "creative_tension": "자기 작업의 기준과 긴장을 조용히 밀고 가는 편",
    "network_exchange_mode": "주변 협업자와 실무적인 피드백을 주고받는 편",
    "network_scope": "소수 협업자와 느슨하지만 꾸준한 연결을 유지하는 편",
    "network_role": "실무 요청이 들어오면 조용히 조율하는 역할을 맡는 편",
    "network_friction": "관계는 넓히기보다 유지 비용을 줄이는 방식으로 관리하는 편",
    "living_tradeoff": "작업 시간을 지키기 위해 생활 동선을 자주 조정하는 편",
    "support_attitude": "지원 제도는 실제 작업 시간을 벌어 주는지부터 따지는 편",
    "support_decision": "공고를 보면 조건과 준비 시간을 먼저 가늠해 맞는 것만 남기는 편",
    "support_path": "작은 규모의 창작·발표 지원을 우선 살피는 편",
    "support_friction": "서류와 정산 부담이 커지면 지원 자체를 미루는 편",
    "support_effect": "작업 시간을 조금이라도 벌어 주는 지원에 의미를 두는 편",
    "persona_focus": "작업 리듬과 생활 태도가 함께 드러나는 편",
    "hobby_plan_text": "동네 산책, 서점 방문, 메모 정리, 가벼운 운동",
    "blocked_hobby_family_text": "",
    "blocked_hobby_item_text": "",
    "family_scope_guard": (
        "배우자, 자녀, 동거인 유무를 사실처럼 단정하지 말고 가까운 관계와 일정 조율의 톤으로만 서술"
    ),
    "family_genericity_guard": (
        "\"가족과의 시간 조율\", \"생활 책임과 작업 지속성\", \"관계를 유지한다\" 같은 반복 뼈대를 피하고 관계 운영의 방식이 보이게 쓸 것"
    ),
    "hobby_genericity_guard": (
        "음악 감상, 책 읽기, 산책 같은 범용 항목은 단독으로 반복하지 말고 구체적 맥락을 붙일 것"
    ),
    "living_genericity_guard": (
        "\"주중에는 ... 주말에는 ...\", \"생활 동선을 안정적으로 묶고 있다\", "
        "\"일상의 균형을 유지한다\" 같은 뼈대 문장을 그대로 복제하지 말 것"
    ),
    "network_genericity_guard": (
        "\"생태계에서 중심적인 역할\", \"인맥을 넓히고 있다\", "
        "\"요청이 들어오면 연결과 조율을 맡는다\" 같은 고정 뼈대를 피할 것"
    ),
    "support_genericity_guard": (
        "\"실질적인 도움을 기대한다\", \"지원 제도는 필요할 때 쓰되...\" 같은 총론 문장을 반복하지 말고 실제 마찰과 효용을 드러낼 것"
    ),
}


def _augment_prompt_for_category(
    category: str,
    rendered: str,
    context: Mapping[str, object],
) -> str:
    if category == "network":
        network_lines = [
            "",
            "[network fact anchors]",
            f"- network_exchange_mode: {context['network_exchange_mode']}",
            f"- network_scope: {context['network_scope']}",
            f"- network_role: {context['network_role']}",
            f"- network_friction: {context['network_friction']}",
            "",
            "[network_persona 추가 지시]",
            "- 협업 상대 2종 이상과 무엇을 주고받는지가 보이게 쓸 것",
            "- 문두를 추상적 역할 설명으로 시작하지 말고, 반복 협업 상대나 주고받는 실무 장면에서 시작할 것",
            "- '중심적인 역할/위치' 같은 생태계 과장 표현은 쓰지 말 것",
            f"- {context['network_genericity_guard']}",
        ]
        return rendered.rstrip() + "\n" + "\n".join(network_lines)

    if category == "support":
        support_lines = [
            "",
            "[support fact anchors]",
            f"- support_need: {context['support_need']}",
            f"- support_attitude: {context['support_attitude']}",
            f"- support_decision: {context['support_decision']}",
            f"- support_path: {context['support_path']}",
            f"- support_friction: {context['support_friction']}",
            f"- support_effect: {context['support_effect']}",
            "",
            "[support_persona 추가 지시]",
            "- 신청/미신청/탈락/수혜 중 어떤 결인지 한쪽으로 분명히 잡을 것",
            "- support_decision을 따라 공고를 어떻게 걸러 보는지 또는 어디서 접는지가 보이게 쓸 것",
            "- 지원 제도의 실제 효용 1개와 마찰 1개를 반드시 같이 쓸 것",
            "- '지원 제도는...' 같은 총론 문장보다 신청 판단이나 체감 효과에서 바로 시작할 것",
            f"- {context['support_genericity_guard']}",
        ]
        return rendered.rstrip() + "\n" + "\n".join(support_lines)

    if category == "family":
        family_lines = [
            "",
            "[family fact anchors]",
            f"- family_contact_style: {context['family_contact_style']}",
            f"- family_rhythm: {context['family_rhythm']}",
            f"- family_boundary: {context['family_boundary']}",
            f"- family_responsibility: {context['family_responsibility']}",
            "",
            "[family_persona 추가 지시]",
            "- 가족 구조를 상상해 채우지 말고, 가까운 관계를 어떻게 운영하는지가 보이게 쓸 것",
            "- 연락 빈도, 약속 배치, 생활 책임 분담 중 최소 2개가 직접 보이게 쓸 것",
            f"- {context['family_genericity_guard']}",
        ]
        return rendered.rstrip() + "\n" + "\n".join(family_lines)

    if category != "living":
        return rendered

    living_lines = [
        "",
        "[생활 fact 앵커]",
        f"- workspace_mode: {context['workspace_mode']}",
        f"- weekly_rhythm: {context['weekly_rhythm']}",
        f"- housing_pressure: {context['housing_pressure']}",
        f"- living_tradeoff: {context['living_tradeoff']}",
        f"- space_anchor: {context['space_anchor']}",
        f"- expense_anchor: {context['expense_anchor']}",
        f"- recovery_anchor: {context['recovery_anchor']}",
        f"- local_routine_hint: {context['local_routine_hint']}",
        "",
        "[living_persona 추가 지시]",
        "- 위 앵커 중 최소 4개를 직접 반영하고, 그중 1개는 시간 운영, 1개는 비용/공간, 1개는 회복 방식이어야 함",
        "- 실제로 돈과 공간을 어떻게 버티는지 보이게 쓰고, 추상적인 균형론으로 마무리하지 말 것",
        f"- {context['living_genericity_guard']}",
    ]
    return rendered.rstrip() + "\n" + "\n".join(living_lines)


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(PROMPTS_ROOT)),
        undefined=StrictUndefined,
        trim_blocks=False,
        lstrip_blocks=False,
        keep_trailing_newline=True,
    )


def _resolve_template(category: str, field: str | None) -> str:
    """Use category/{field}.j2 if it exists, otherwise _fallback.j2."""
    candidates = []
    if field:
        candidates.append(f"{category}/{field}.j2")
    candidates.append(f"{category}/_fallback.j2")
    for c in candidates:
        if (PROMPTS_ROOT / c).exists():
            return c
    raise TemplateNotFound(
        f"no template for category={category!r}, field={field!r}. tried {candidates}"
    )


def render_narrative_prompt(
    category: str,
    quant: Mapping[str, object],
    *,
    seed_word: str | None = None,
    rng: random.Random | None = None,
) -> str:
    """Render a single narrative prompt.

    Args:
        category: "professional" / "creative_world" / ... / "cultural_background" / ...
        quant: quantitative-variable dict (PAKPersonaQuant or sampler output)
        seed_word: explicit seed. If None, chosen at random from the field pool.
        rng: an external random.Random can be injected for determinism.
    """
    if rng is None:
        rng = random.Random()

    field = str(quant.get("art_field_primary", ""))
    if category in DOMAIN_NARRATIVES:
        if seed_word is None:
            seed_word = rng.choice(SEED_POOLS.get(field) or SEED_POOLS["기타"])
        template_name = _resolve_template(category, field)
    elif category in COMMON_NARRATIVES:
        template_name = f"_common/{category}.j2"
        if not (PROMPTS_ROOT / template_name).exists():
            raise TemplateNotFound(template_name)
    else:
        raise ValueError(f"unknown narrative category: {category}")

    env = _env()
    template = env.get_template(template_name)
    context = dict(_PROMPT_DEFAULTS)
    context.update(dict(quant))
    rendered = template.render(seed_word=seed_word, **context)
    return _augment_prompt_for_category(category, rendered, context)


def render_all_for_persona(
    quant: Mapping[str, object],
    *,
    rng: random.Random | None = None,
) -> dict[str, str]:
    """Render all 12 narrative prompts for a single persona."""
    if rng is None:
        rng = random.Random()
    out: dict[str, str] = {}
    for cat in DOMAIN_NARRATIVES:
        out[cat] = render_narrative_prompt(cat, quant, rng=rng)
    for cat in COMMON_NARRATIVES:
        out[cat] = render_narrative_prompt(cat, quant, rng=rng)
    return out


def has_field_specific_template(category: str, field: str) -> bool:
    return (PROMPTS_ROOT / category / f"{field}.j2").exists()


def template_coverage_report() -> dict[str, dict[str, bool]]:
    """In the 5-domain x 14-field matrix, which cells have a field-specific template."""
    from pak.schema import ArtField

    fields: list[str] = list(ArtField.__args__)  # type: ignore[attr-defined]
    out: dict[str, dict[str, bool]] = {}
    for cat in DOMAIN_NARRATIVES:
        out[cat] = {f: has_field_specific_template(cat, f) for f in fields}
    return out


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    sample_quant = {
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
    rng = random.Random(20260502)
    prompts = render_all_for_persona(sample_quant, rng=rng)
    print(f"=== rendered {len(prompts)} narrative prompts ===\n")
    for k, p in prompts.items():
        print(f"--- {k} ---")
        print(p)
        print()

    cov = template_coverage_report()
    print("\n=== template coverage (분야별 전용 템플릿 존재 여부) ===")
    fields = list(next(iter(cov.values())).keys())
    print("category | " + " | ".join(fields))
    for cat in cov:
        cells = ["✓" if cov[cat][f] else "·" for f in fields]
        print(f"{cat:14s} | " + " | ".join(cells))
