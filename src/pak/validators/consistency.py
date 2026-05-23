"""Quantitative-qualitative consistency validation.

Rule-based check that a persona's quantitative variables do not contradict the
expressions in the narrative text.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Literal

from pak.samplers import min_career_start_age

logger = logging.getLogger(__name__)


Severity = Literal["error", "warning", "info"]


@dataclass
class Issue:
    severity: Severity
    code: str
    field: str  # which narrative or quantitative field
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.code} ({self.field}): {self.message}"


# ----------------------------------------------------------------------------
# 1. Age / generation expression consistency
# ----------------------------------------------------------------------------

_AGE_GENERATION_PATTERNS: dict[str, tuple[int, int]] = {
    "10대": (10, 19),
    "20대": (20, 29),
    "30대": (30, 39),
    "40대": (40, 49),
    "50대": (50, 59),
    "60대": (60, 69),
    "70대": (70, 79),
    "80대": (80, 89),
    "90대": (90, 99),
}

_AGE_PAST_PHASE_CONTEXT = re.compile(
    r"(데뷔|입문|시작|등단|당선|진출|처음|첫\s*(전시|공연|활동|작업))"
)
_AGE_SELF_REFERENCE_AFTER = re.compile(
    r"(?:\s*(초반|중반|후반))?(?:의)?\s*"
    r"(?:신인|중견|원로|노년|젊은|청년)?\s*"
    r"(남자|여자|사람|예술인|작가|감독|배우|연주자|창작자|예술가|예능인|"
    r"화가|판화가|도예가|공예가|사진가|건축가|작곡가|시인|소설가|무용가|안무가|연출가|만화가)"
)
_AGE_NON_SELF_HINT_AFTER = re.compile(
    r"(?:\s*(초반|중반|후반))?(?:의)?\s*"
    r"(아들|딸|자녀|남편|아내|배우자|부모|어머니|아버지|취향)"
)


def _is_self_age_reference(text: str, match_start: int, match_end: int) -> bool:
    after = text[match_end : min(len(text), match_end + 18)]
    if _AGE_NON_SELF_HINT_AFTER.match(after):
        return False
    if re.match(r"\s*이상", after):
        return True
    return bool(_AGE_SELF_REFERENCE_AFTER.match(after))


def check_age_consistency(quant: dict, narratives: dict[str, str]) -> list[Issue]:
    issues: list[Issue] = []
    age = int(quant.get("age", 0))
    for field, text in narratives.items():
        for label, (lo, hi) in _AGE_GENERATION_PATTERNS.items():
            if label not in text:
                continue
            for match in re.finditer(label, text):
                window = text[match.start() : min(len(text), match.end() + 24)]
                # Past-history phrasing like "debuted in late 20s" / "entered in early 30s" must be kept separate from current age.
                if _AGE_PAST_PHASE_CONTEXT.search(window):
                    continue
                if not _is_self_age_reference(text, match.start(), match.end()):
                    continue
                if not (lo <= age <= hi):
                    issues.append(
                        Issue(
                            "warning",
                            "AGE_MISMATCH",
                            field,
                            f"narrative contains '{label}' but age={age} is outside range [{lo},{hi}]",
                        )
                    )
                    return issues
    return issues


# ----------------------------------------------------------------------------
# 2. Age <-> career number consistency
# ----------------------------------------------------------------------------


def check_age_career_feasibility(quant: dict, narratives: dict[str, str]) -> list[Issue]:
    del narratives  # numeric-based check
    issues: list[Issue] = []
    if "age" not in quant or "career_years" not in quant:
        return issues

    age = int(quant.get("age", 0))
    career_years = int(quant.get("career_years", 0))
    field = str(quant.get("art_field_primary", ""))
    start_age_floor = min_career_start_age(field)
    max_career_years = max(age - start_age_floor, 0)
    if career_years > max_career_years:
        issues.append(
            Issue(
                "error",
                "AGE_CAREER_IMPOSSIBLE",
                "quant",
                f"field={field}, age={age}, career_years={career_years} is "
                f"impossible given minimum start age {start_age_floor}",
            )
        )
    return issues


# ----------------------------------------------------------------------------
# 3. Career-stage expression consistency
# ----------------------------------------------------------------------------

_CAREER_NEW_PATTERNS = [
    r"신진\s*단계",
    r"신진\s*예술인",
    r"신진\s*(작가|감독|배우|연주자|창작자|예술가)(?!전)",
    r"신인\s*(작가|감독|배우|연주자|창작자|예술가)(?!전)",
    r"새로\s*시작한\s*(작가|감독|배우|연주자|창작자|예술가)",
]
_CAREER_VETERAN_PATTERNS = [
    r"원로",
    r"중견",
    r"대가",
    r"수십\s*년",
    r"오랜\s*세월",
    r"후배\s*양성",
    r"제자\s*육성",
]


def check_career_consistency(quant: dict, narratives: dict[str, str]) -> list[Issue]:
    issues: list[Issue] = []
    career_years = int(quant.get("career_years", 0))
    text = "\n".join(narratives.values())

    if career_years >= 5:
        for pat in _CAREER_NEW_PATTERNS:
            if re.search(pat, text):
                issues.append(
                    Issue(
                        "warning",
                        "CAREER_MISMATCH_NEW",
                        "narrative",
                        f"career_years={career_years} but newcomer expression '{pat}' appears",
                    )
                )
                break

    if career_years <= 7:
        for pat in _CAREER_VETERAN_PATTERNS:
            if re.search(pat, text):
                issues.append(
                    Issue(
                        "warning",
                        "CAREER_MISMATCH_VETERAN",
                        "narrative",
                        f"career_years={career_years} but veteran expression '{pat}' appears",
                    )
                )
                break

    return issues


_LATE_DEBUT_NARRATIVE_FIELDS = (
    "professional_persona",
    "creative_world_persona",
    "network_persona",
    "arts_persona",
)
_LATE_DEBUT_TONE_PATTERNS = [
    r"전통의\s*깊이",
    r"평생(?:을| 동안)?",
    r"수십\s*년(?:\s*동안)?",
    r"오랜\s*세월",
    r"긴\s*세월",
    r"젊었을\s*때",
    r"젊은\s*시절",
    r"긴\s*경력",
    r"오랜\s*경력",
    r"원로",
    r"대가",
    r"거장",
    r"한평생",
    r"오랜\s*시간",
]


def _compute_debut_age(quant: dict) -> int | None:
    """Compute debut/entry age from current age and years of activity."""
    try:
        age = float(quant.get("age"))
        career_years = float(quant.get("career_years"))
    except (TypeError, ValueError):
        return None
    if math.isnan(age) or math.isnan(career_years):
        return None
    return int(round(age - career_years))


def check_late_debut_awareness(quant: dict, narratives: dict[str, str]) -> list[Issue]:
    """Check whether a persona who entered after age 50 mixes in long-lineage / veteran tone."""
    debut_age = _compute_debut_age(quant)
    if debut_age is None or debut_age < 50:
        return []

    age = quant.get("age")
    career_years = quant.get("career_years")
    art_field = quant.get("art_field_primary")
    issues: list[Issue] = []
    for field in _LATE_DEBUT_NARRATIVE_FIELDS:
        text = str(narratives.get(field, "") or "")
        if not text:
            continue
        for pat in _LATE_DEBUT_TONE_PATTERNS:
            for match in re.finditer(pat, text):
                issues.append(
                    Issue(
                        "warning",
                        "LATE_DEBUT_TONE_MISMATCH",
                        field,
                        (
                            f"age={age}, career_years={career_years}, "
                            f"debut_age={debut_age}, art_field_primary={art_field!r} but "
                            f"{field} contains expression "
                            f"{match.group(0)!r} that does not match the late-debut tone"
                        ),
                    )
                )
    return issues


_CULTURAL_TIMELINE_DEBUT_AGE_THRESHOLD = 40
_CULTURAL_EARLY_INTEREST_PATTERN = re.compile(
    r"(어릴\s*적부터|어렸을\s*때부터|어린\s*시절부터)"
)
_CULTURAL_EDUCATION_CUE_PATTERN = re.compile(
    r"("
    r"고등학교(?:를)?\s*졸업(?:한\s*후|후)?|"
    r"고졸\s*후|"
    r"대학교(?:를)?\s*졸업(?:한\s*후|후)?|"
    r"대학(?:을)?\s*(?:졸업(?:한\s*후|후)?|마치고)|"
    r"전공"
    r")"
)
_CULTURAL_FIELD_VERB_PATTERN = re.compile(r"(시작|활동|진출|전공)")

# Avoid false positives: negation / concessive clauses (e.g. "전공하지 않", "전공했으나", "전공이 아닌").
# If the narrative is timeline-consistent, like "전공했으나 다른 일을 하다가 전환", it should not trigger.
_CULTURAL_TIMELINE_NEGATION_PATTERN = re.compile(
    r"("
    r"전공(?:을|이|이라고)?\s*(?:하지\s*않|안\s*했|안\s*하|아니[고였]|이\s*아닌|"
    r"하지\s*못)|"
    r"전공(?:을|만)?\s*(?:했|하)\s*(?:으나|지만|음에도)|"
    r"전공\s*(?:은|이)\s*(?:다른|아니|별도)|"
    r"(?:다른|별도의?)\s*(?:일|직업|분야)\s*(?:을|를)?\s*(?:하다가|하며|거쳐)"
    r")"
)


def _explicit_debut_age_in_window(window: str, debut_age: int) -> bool:
    """Treat the timeline as consistent if the sentence contains an explicit debut-age expression.

    If the narrative explicitly states an expression close to the actual debut_age (within +/-3 years),
    such as `61세 무렵`, `52세에`, `40대 후반`, then it is not a contradiction even when an
    education cue and a field cue appear together.
    """
    for m in re.finditer(r"(\d{1,2})\s*세\s*(?:무렵|즈음|쯤|경|에|때|부터|이후)", window):
        try:
            n = int(m.group(1))
        except ValueError:
            continue
        if abs(n - debut_age) <= 3:
            return True
    decade_target = (debut_age // 10) * 10
    for m in re.finditer(r"(\d{1,2})\s*대(?:\s*(?:초반|중반|후반|들어))?", window):
        try:
            decade = int(m.group(1))
        except ValueError:
            continue
        if abs(decade - decade_target) <= 10:
            return True
    return False


def _cultural_timeline_field_terms(art_field: object) -> list[str]:
    """Build the field cues used by the cultural-background timeline check."""
    field = str(art_field or "").strip()
    terms = ["예술", "창작"]
    if field:
        terms.insert(0, field)
    return terms


def _sentence_windows(text: str) -> list[str]:
    """Split loosely into sentences so only the co-occurrence of an education cue and a field verb is checked."""
    return [
        chunk.strip()
        for chunk in re.split(r"(?<=[.!?。])\s+|\n+", text)
        if chunk.strip()
    ]


def _sentence_tail(text: str, start: int, max_chars: int = 80) -> str:
    """Slice from start only up to the end of the same sentence."""
    end = min(len(text), start + max_chars)
    sentence_ends = [
        pos
        for marker in (".", "!", "?", "。", "\n")
        if (pos := text.find(marker, start, end)) != -1
    ]
    if sentence_ends:
        end = min(sentence_ends) + 1
    return text[start:end]


def check_cultural_background_timeline(
    quant: dict,
    narratives: dict[str, str],
) -> list[Issue]:
    """Check for conflicts between a late-entry timeline and early/education-period descriptions in cultural_background."""
    debut_age = _compute_debut_age(quant)
    if debut_age is None or debut_age < _CULTURAL_TIMELINE_DEBUT_AGE_THRESHOLD:
        return []

    text = str(narratives.get("cultural_background", "") or "")
    if not text:
        return []

    age = quant.get("age")
    career_years = quant.get("career_years")
    art_field = quant.get("art_field_primary")
    field_terms = _cultural_timeline_field_terms(art_field)
    field_pattern = re.compile("|".join(re.escape(term) for term in field_terms if term))
    issues: list[Issue] = []

    early_match = _CULTURAL_EARLY_INTEREST_PATTERN.search(text)
    if early_match:
        tail = _sentence_tail(text, early_match.start())
        field_match = field_pattern.search(tail)
        if field_match:
            # If the narrative also has an explicit debut age or a negation/transition clause, treat as consistent.
            tail_full = _sentence_tail(text, early_match.start(), max_chars=160)
            if not (
                _explicit_debut_age_in_window(tail_full, debut_age)
                or _CULTURAL_TIMELINE_NEGATION_PATTERN.search(tail_full)
            ):
                issues.append(
                    Issue(
                        "warning",
                        "EARLY_INTEREST_VS_LATE_DEBUT",
                        "cultural_background",
                        (
                            f"age={age}, career_years={career_years}, "
                            f"debut_age={debut_age}, art_field_primary={art_field!r} but "
                            f"cultural_background has early-interest expression "
                            f"{early_match.group(0)!r} together with field cue {field_match.group(0)!r}"
                        ),
                    )
                )

    for window in _sentence_windows(text):
        cue_match = _CULTURAL_EDUCATION_CUE_PATTERN.search(window)
        if not cue_match:
            continue
        field_match = field_pattern.search(window)
        verb_match = _CULTURAL_FIELD_VERB_PATTERN.search(window)
        if not field_match or not verb_match:
            continue
        # If the same sentence has an explicit debut age (e.g. "61세 무렵", "40대 후반") or a
        # negation/transition clause, treat the timeline as consistent and do not trigger.
        if _explicit_debut_age_in_window(window, debut_age):
            continue
        if _CULTURAL_TIMELINE_NEGATION_PATTERN.search(window):
            continue
        issues.append(
            Issue(
                "warning",
                "EDUCATION_DEBUT_TIMELINE_MISMATCH",
                "cultural_background",
                (
                    f"age={age}, career_years={career_years}, "
                    f"debut_age={debut_age}, art_field_primary={art_field!r} but "
                    f"education-period cue {cue_match.group(0)!r}, field cue "
                    f"{field_match.group(0)!r}, and activity verb {verb_match.group(0)!r} "
                    "appear in the same cultural_background sentence"
                ),
            )
        )
        break

    return issues


def check_employment_duration_conflation(
    quant: dict,
    narratives: dict[str, str],
) -> list[Issue]:
    """Check whether the current full-time/secondary-job status is written as if it were the duration of career_years."""
    issues: list[Issue] = []
    career_years = int(quant.get("career_years", 0) or 0)
    if career_years <= 0:
        return issues

    text = "\n".join(narratives.values())
    years = re.escape(str(career_years))
    patterns = [
        rf"{years}\s*년(?:간)?\s*(?:동안\s*)?(?:전업|겸업)",
        rf"(?:전업|겸업)\s*(?:으로서|으로|상태로)?\s*{years}\s*년",
    ]
    for pat in patterns:
        match = re.search(pat, text)
        if match:
            issues.append(
                Issue(
                    "warning",
                    "EMPLOYMENT_DURATION_CONFLATION",
                    "narrative",
                    (
                        f"career_years={career_years} is described as if it were the duration of the "
                        f"current full-time/secondary-job status: '{match.group(0)}'. "
                        f"Separate them, e.g. '활동 경력은 {career_years}년이며, 현재는 전업/겸업 상태다'"
                    ),
                )
            )
            break
    return issues


# ----------------------------------------------------------------------------
# 4. Appearance of out-of-field vocabulary
# ----------------------------------------------------------------------------

_FIELD_FOREIGN_TERMS: dict[str, list[str]] = {
    "문학": ["안무", "BIM", "대본 리딩", "도면", "콘티", "녹음실", "오디오 마스터링"],
    "미술": ["대본", "안무", "BIM", "도면 마감", "콘티"],
    "공예": ["대본", "안무", "BIM", "콘티", "롱테이크"],
    "사진": ["대본", "안무", "BIM", "콘티", "악보"],
    "건축": ["대본", "안무", "콘티", "악보", "낭독회"],
    "음악": ["BIM", "콘티", "도면", "안무", "낭독회"],
    "국악": ["BIM", "도면", "콘티", "롱테이크"],
    "대중음악": ["BIM", "도면", "콘티", "낭독회"],
    "방송연예": ["BIM", "도면", "악보 채보", "낭독회"],
    "무용": ["BIM", "도면", "콘티", "악보 채보"],
    "연극": ["BIM", "도면 마감", "콘티"],
    "영화": ["BIM", "도면", "악보 채보", "낭독회"],
    "만화": ["BIM", "도면 마감", "악보", "안무"],
    "기타": [],
}


def check_field_vocabulary(quant: dict, narratives: dict[str, str]) -> list[Issue]:
    issues: list[Issue] = []
    field = str(quant.get("art_field_primary", ""))
    foreign = _FIELD_FOREIGN_TERMS.get(field, [])
    text = "\n".join(narratives.values())
    for term in foreign:
        if term in text:
            issues.append(
                Issue(
                    "error",
                    "FIELD_FOREIGN_VOCAB",
                    "narrative",
                    f"field={field} but out-of-field vocabulary '{term}' appears",
                )
            )
    return issues


# ----------------------------------------------------------------------------
# 5. occupation <-> main-job narrative consistency
# ----------------------------------------------------------------------------

_OCCUPATION_ANCHORS: dict[str, list[str]] = {
    "시인": ["시", "시집"],
    "소설가": ["소설", "소설집", "장편", "단편"],
    "수필가": ["수필", "산문"],
    "평론가": ["평론", "비평"],
    "동화작가": ["동화", "아동문학", "청소년문학"],
    "번역가": ["번역"],
    "회화 작가": ["회화", "미술", "작품"],
    "조각가": ["조각"],
    "판화가": ["판화"],
    "설치미술 작가": ["설치미술", "설치 작업"],
    "미디어아트 작가": ["미디어아트"],
    "도예가": ["도예", "도자"],
    "금속공예가": ["금속공예", "금속 작업"],
    "섬유공예가": ["섬유공예", "직조", "텍스타일"],
    "목공예가": ["목공예", "목재"],
    "유리공예가": ["유리공예", "유리 작업"],
    "다큐멘터리 사진가": ["다큐멘터리 사진", "기록 사진"],
    "파인아트 사진가": ["파인아트 사진", "예술 사진"],
    "광고 사진가": ["광고 사진", "상업 사진"],
    "건축가": ["건축", "설계"],
    "인테리어 디자이너": ["인테리어", "실내 공간"],
    "조경설계가": ["조경", "설계"],
    "작곡가": ["작곡"],
    "지휘자": ["지휘"],
    "성악가": ["성악", "가창"],
    "기악 연주자": ["기악", "연주"],
    "국악인": ["국악"],
    "판소리 명창": ["판소리", "명창", "창"],
    "기악 연주자(국악)": ["국악", "기악", "연주"],
    "싱어송라이터": ["싱어송라이터", "작사", "작곡"],
    "작곡가(대중음악)": ["작곡", "프로듀싱", "대중음악"],
    "세션 연주자": ["세션", "연주"],
    "음악 프로듀서": ["프로듀서", "프로듀싱"],
    "방송 출연자": ["방송 출연", "출연"],
    "예능인": ["예능", "방송", "게스트", "패널", "방송 출연"],
    "방송작가": ["방송작가", "대본", "구성"],
    "MC": ["MC", "진행", "사회"],
    "무용가": ["무용"],
    "안무가": ["안무"],
    "배우": ["배우", "연기"],
    "연출가": ["연출"],
    "극작가": ["극작", "희곡"],
    "무대미술가": ["무대미술", "무대 디자인", "세트"],
    "영화감독": ["영화감독", "영화 감독", "연출"],
    "시나리오 작가": ["시나리오", "각본"],
    "촬영감독": ["촬영감독", "촬영", "카메라"],
    "편집기사": ["편집", "후반작업"],
    "프로듀서": ["프로듀서", "제작"],
    "만화가": ["만화"],
    "웹툰 작가": ["웹툰"],
    "스토리 작가": ["스토리", "서사"],
    "문화예술 매개자": ["매개", "기획", "연결"],
    "예술 행정": ["행정"],
    "융복합 예술가": ["융복합", "복합", "다학제"],
}


def check_occupation_consistency(quant: dict, narratives: dict[str, str]) -> list[Issue]:
    issues: list[Issue] = []
    occupation = str(quant.get("occupation", "") or "")
    if not occupation:
        return issues

    anchors = _OCCUPATION_ANCHORS.get(occupation)
    if not anchors:
        return issues

    text = "\n".join(
        filter(
            None,
            [
                narratives.get("persona", ""),
                narratives.get("professional_persona", ""),
            ],
        )
    )
    if text and not any(anchor in text for anchor in anchors):
        issues.append(
            Issue(
                "warning",
                "OCCUPATION_MISMATCH",
                "professional_persona",
                f"core cues {anchors} for occupation={occupation!r} are absent from persona/professional_persona",
            )
        )
    return issues


# ----------------------------------------------------------------------------
# 6. Region consistency
# ----------------------------------------------------------------------------

_PROVINCES = (
    "서울",
    "부산",
    "대구",
    "인천",
    "광주",
    "대전",
    "울산",
    "세종",
    "경기",
    "강원",
    "충청북",
    "충청남",
    "전북",
    "전라남",
    "경상북",
    "경상남",
    "제주",
)

_LANDMARK_PROVINCES: dict[str, str] = {
    "무등산": "광주",
    "해운대": "부산",
    "경복궁": "서울",
    "창덕궁": "서울",
}

_REGION_REFERENCE_EXEMPTIONS = [
    r"서울(?:국제)?(?:영화제|도서전|아트페어|비엔날레|페스티벌)",
    r"부산(?:국제)?(?:영화제|비엔날레|아트페어|페스티벌)",
    r"광주(?:비엔날레|아트페어|페스티벌)",
]


def check_region_consistency(quant: dict, narratives: dict[str, str]) -> list[Issue]:
    issues: list[Issue] = []
    province = str(quant.get("province", "") or "")
    if not province:
        return issues

    text = "\n".join(
        filter(
            None,
            [
                narratives.get("persona", ""),
                narratives.get("professional_persona", ""),
                narratives.get("living_persona", ""),
            ],
        )
    )
    if not text:
        return issues

    sanitized_text = text
    for pat in _REGION_REFERENCE_EXEMPTIONS:
        sanitized_text = re.sub(pat, "", sanitized_text)

    mentioned_provinces = [p for p in _PROVINCES if p in sanitized_text]
    if province == "기타":
        if mentioned_provinces:
            issues.append(
                Issue(
                    "warning",
                    "REGION_MISMATCH",
                    "narrative",
                    f"province='기타' but narrative mentions specific region {mentioned_provinces[0]!r}",
                )
            )
    else:
        for other in mentioned_provinces:
            if other != province:
                issues.append(
                    Issue(
                        "warning",
                        "REGION_MISMATCH",
                        "narrative",
                        f"province={province!r} but narrative mentions different region {other!r}",
                    )
                )
                break

    for landmark, expected_province in _LANDMARK_PROVINCES.items():
        if landmark in text and province != expected_province:
            issues.append(
                Issue(
                    "warning",
                    "LANDMARK_REGION_MISMATCH",
                    "narrative",
                    f"province={province!r} but {landmark!r} usually belongs to the {expected_province!r} context",
                )
            )
            break

    return issues


# ----------------------------------------------------------------------------
# 7. Full-time/secondary-job <-> description consistency
# ----------------------------------------------------------------------------

_SECONDARY_JOB_PATTERNS = [
    r"겸업",
    r"부업",
    r"아르바이트|알바",
    r"자영업",
    r"외주\s*(수입|일|업무)",
    r"원고료\s*외에도",
]


def check_employment_consistency(quant: dict, narratives: dict[str, str]) -> list[Issue]:
    issues: list[Issue] = []
    employment_type = str(quant.get("employment_type", "") or "")
    has_secondary_job = bool(quant.get("has_secondary_job"))
    if employment_type != "전업" or has_secondary_job:
        return issues

    text = "\n".join(narratives.values())
    for pat in _SECONDARY_JOB_PATTERNS:
        if re.search(pat, text):
            issues.append(
                Issue(
                    "warning",
                    "EMPLOYMENT_MISMATCH",
                    "narrative",
                    f"employment_type='전업', has_secondary_job=False but side-job/secondary-job cue '{pat}' appears",
                )
            )
            break
    return issues


# ----------------------------------------------------------------------------
# 8. Income <-> living_persona consistency
# ----------------------------------------------------------------------------

_INCOME_RICH_PATTERNS = [
    r"안정적\s*수입",
    r"전업\s*수입만으로",
    r"여유\s*있는",
    r"풍족",
    r"고소득",
    r"많은\s*돈을\s*벌",
]
_INCOME_POOR_PATTERNS = [
    r"가족\s*지원에\s*기대",
    r"부수입\s*없이는",
    r"생계\s*위협",
]


def check_income_consistency(quant: dict, narratives: dict[str, str]) -> list[Issue]:
    issues: list[Issue] = []
    income = str(quant.get("individual_art_income_bracket", ""))
    living = narratives.get("living_persona", "")
    if not living:
        return issues

    if income in ("없음", "5백만원 미만"):
        for pat in _INCOME_RICH_PATTERNS:
            if re.search(pat, living):
                issues.append(
                    Issue(
                        "error",
                        "INCOME_RICH_BUT_LOW",
                        "living_persona",
                        f"income={income} but affluence expression '{pat}' appears",
                    )
                )
                break

    if income in ("5-6천만원 미만", "6천만원 이상"):
        for pat in _INCOME_POOR_PATTERNS:
            if re.search(pat, living):
                issues.append(
                    Issue(
                        "warning",
                        "INCOME_POOR_BUT_HIGH",
                        "living_persona",
                        f"income={income} but poverty expression '{pat}' appears",
                    )
                )
                break

    return issues


# ----------------------------------------------------------------------------
# 9. Contract/copyright consistency
# ----------------------------------------------------------------------------


def check_contract_consistency(quant: dict, narratives: dict[str, str]) -> list[Issue]:
    issues: list[Issue] = []
    text = "\n".join(narratives.values())

    has_contract = bool(quant.get("has_contract_experience"))
    if not has_contract and re.search(r"표준계약서|계약서\s*체결", text):
        issues.append(
            Issue(
                "warning",
                "CONTRACT_MISMATCH",
                "narrative",
                "has_contract_experience=False but narrative mentions signing a contract",
            )
        )

    has_copyright = bool(quant.get("has_copyright"))
    if not has_copyright and re.search(r"저작권\s*수입|인세를\s*받", text):
        issues.append(
            Issue(
                "warning",
                "COPYRIGHT_MISMATCH",
                "narrative",
                "has_copyright=False but narrative mentions copyright income",
            )
        )

    return issues


# ----------------------------------------------------------------------------
# Pipeline entry
# ----------------------------------------------------------------------------


def check_all(quant: dict, narratives: dict[str, str]) -> list[Issue]:
    issues: list[Issue] = []
    issues.extend(check_age_consistency(quant, narratives))
    issues.extend(check_age_career_feasibility(quant, narratives))
    late_debut_issues = check_late_debut_awareness(quant, narratives)
    issues.extend(late_debut_issues)
    issues.extend(check_cultural_background_timeline(quant, narratives))
    career_issues = check_career_consistency(quant, narratives)
    if late_debut_issues:
        career_issues = [
            issue for issue in career_issues if issue.code != "CAREER_MISMATCH_VETERAN"
        ]
    issues.extend(career_issues)
    issues.extend(check_employment_duration_conflation(quant, narratives))
    issues.extend(check_field_vocabulary(quant, narratives))
    issues.extend(check_occupation_consistency(quant, narratives))
    issues.extend(check_region_consistency(quant, narratives))
    issues.extend(check_employment_consistency(quant, narratives))
    issues.extend(check_income_consistency(quant, narratives))
    issues.extend(check_contract_consistency(quant, narratives))
    return issues
