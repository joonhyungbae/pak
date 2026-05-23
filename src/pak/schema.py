"""PAK persona Pydantic v2 schema.

L1+ NPK-compatible superset (~40 columns):
- NPK 26-column superset (sex/age/province/district/country/education_level/occupation/
  marital_status/military_status/family_type/housing_type/bachelors_field +
  7 narrative + 6 attribute)
- PAK 17 domain columns (art_field, career, income, contract/copyright/break/overseas,
  3 domain narratives)

Adopts NPK notation (province "충청남"/"전라남" etc.). The PAK report notation alias
can be converted with `alias_province_pak()`.

L2 (KOSIS supplementary statistics IPF) is v1.x. In v0.1, NPK-style columns not present
in the report are allowed to be `null`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ----------------------------------------------------------------------------
# 0. PAK domain enums (verbatim from the Phase 01 report)
# ----------------------------------------------------------------------------

ArtField = Literal[
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

Sex = Literal["남자", "여자"]

AgeBand = Literal["10대", "20대", "30대", "40대", "50대", "60대", "70대 이상"]


# ----------------------------------------------------------------------------
# 1. Province — adopts NPK notation (충청남/전라남/경상북 etc.)
# ----------------------------------------------------------------------------

ProvinceNPK = Literal[
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
    "기타",  # region unknown in the population (added by PAK)
]

# PAK report notation -> NPK notation mapping
_PAK_TO_NPK_PROVINCE: dict[str, str] = {
    "충북": "충청북",
    "충남": "충청남",
    "전남": "전라남",
    "경북": "경상북",
    "경남": "경상남",
}

# NPK -> PAK reverse mapping
_NPK_TO_PAK_PROVINCE: dict[str, str] = {v: k for k, v in _PAK_TO_NPK_PROVINCE.items()}


def alias_province_to_npk(name: str) -> str:
    """Normalize report notation to NPK notation."""
    return _PAK_TO_NPK_PROVINCE.get(name, name)


def alias_province_to_pak(name: str) -> str:
    """Convert NPK notation to report notation (reverse direction)."""
    return _NPK_TO_PAK_PROVINCE.get(name, name)


# ----------------------------------------------------------------------------
# 2. NPK-style demographic enums (mostly nullable in v0.1, filled by KOSIS IPF in v1.x)
# ----------------------------------------------------------------------------

# NPK education_level 7-category (the report's 3-category is separate)
EducationLevelNPK = Literal[
    "무학",
    "초등학교",
    "중학교",
    "고등학교",
    "2~3년제 전문대학",
    "4년제 대학교",
    "대학원",
]

# 3-category from the report's respondent table (PAK domain — the sampler samples on this)
EducationLevelPAK = Literal["고졸 이하", "대졸 이하", "대학원 이상"]


def npk_education_to_pak(level: str) -> str:
    """NPK 7-category -> PAK report 3-category."""
    if level in {"무학", "초등학교", "중학교", "고등학교"}:
        return "고졸 이하"
    if level in {"2~3년제 전문대학", "4년제 대학교"}:
        return "대졸 이하"
    if level == "대학원":
        return "대학원 이상"
    raise ValueError(f"unknown education_level: {level}")


MaritalStatus = Literal["미혼", "배우자있음", "사별", "이혼"]
MilitaryStatus = Literal["비현역", "현역", "해당없음"]  # women / not applicable to military service
HousingType = Literal[
    "아파트",
    "단독주택",
    "다세대주택",
    "연립주택",
    "오피스텔",
    "비주거용 건물 내 주택",
]

BachelorsField = Literal[
    "공학·제조·건설",
    "사회과학·언론",
    "보건·복지",
    "교육",
    "정보통신기술",
    "서비스",
    "예술·인문",
    "자연과학·수학",
    "경영·행정·법",
    "농림어업·수의학",
    "해당없음",
]


# ----------------------------------------------------------------------------
# 3. PAK domain enums
# ----------------------------------------------------------------------------

CareerBand = Literal[
    "10년 미만",
    "10-20년 미만",
    "20-30년 미만",
    "30-40년 미만",
    "40년 이상",
]

EmploymentType = Literal["전업", "겸업"]

IncomeBracket = Literal[
    "없음",
    "5백만원 미만",
    "5백-1천만원 미만",
    "1-2천만원 미만",
    "2-3천만원 미만",
    "3-4천만원 미만",
    "4-5천만원 미만",
    "5-6천만원 미만",
    "6천만원 이상",
]

HouseholdIncomeBracket = Literal[
    "1천만원 미만",
    "1-2천만원 미만",
    "2-3천만원 미만",
    "3-4천만원 미만",
    "4-5천만원 미만",
    "5-6천만원 미만",
    "6-7천만원 미만",
    "7-8천만원 미만",
    "8천만원 이상",
]


# ----------------------------------------------------------------------------
# 4. Quantitative — NPK superset
# ----------------------------------------------------------------------------


class PAKPersonaQuant(BaseModel):
    """Quantitative columns — NPK 26-column superset + PAK 17 domain = ~31 quant."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True, extra="forbid")

    pak_uuid: str = Field(description="unique persona identifier (UUID v4)")

    # ========== NPK demographics & geography (10) ==========
    sex: Sex
    age: int = Field(ge=13, le=100)
    province: ProvinceNPK
    district: str | None = Field(
        default=None,
        description=(
            'Format: "{province}-{district}". NPK compatible. '
            "In PAK-core v0.1, the district cannot be honestly filled from the single PDF source, so null is allowed."
        ),
    )
    country: Literal["대한민국"] = "대한민국"
    education_level: EducationLevelNPK
    occupation: str = Field(description="free-text occupation name. PAK synthesizes it based on art field")

    # NPK compatible, but nullable in v0.1 since absent from the report
    marital_status: MaritalStatus | None = None
    military_status: MilitaryStatus | None = None
    family_type: str | None = None  # 39 categories, free-form
    housing_type: HousingType | None = None
    bachelors_field: BachelorsField | None = None

    # ========== PAK domain (17) ==========
    age_band: AgeBand
    education_level_pak: EducationLevelPAK = Field(
        description="3-category from the report's respondent statistics. Mapped from education_level"
    )
    art_field_primary: ArtField
    art_field_secondary: ArtField | None = None

    career_years: int = Field(ge=0, le=70)
    career_band: CareerBand

    employment_type: EmploymentType
    is_freelance: bool
    has_secondary_job: bool

    individual_art_income_bracket: IncomeBracket
    household_income_bracket: HouseholdIncomeBracket

    has_contract_experience: bool
    uses_standard_contract: bool | None = Field(
        default=None,
        description="Meaningful only for those with contract experience. None for those without.",
    )
    has_copyright: bool
    had_career_break: bool
    has_overseas_experience: bool


# ----------------------------------------------------------------------------
# 5. Qualitative — NPK 13 narrative + PAK 3 domain = 16
# ----------------------------------------------------------------------------


class PAKPersonaNarrative(BaseModel):
    """LLM-generated narrative — NPK 13 compatible + PAK domain 3."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    # ========== NPK 7 main narrative ==========
    persona: str = Field(min_length=30, max_length=300, description="one-line overall summary")
    professional_persona: str = Field(min_length=60, max_length=600)
    sports_persona: str = Field(min_length=50, max_length=400)
    arts_persona: str = Field(
        min_length=50,
        max_length=400,
        description="(NPK compatible) enjoyment of arts/culture. In PAK this differs from one's own creative world",
    )
    travel_persona: str = Field(min_length=50, max_length=400)
    culinary_persona: str = Field(min_length=50, max_length=400)
    family_persona: str = Field(min_length=50, max_length=400)

    # ========== NPK 6 attribute (narrative + list variants) ==========
    cultural_background: str = Field(min_length=50, max_length=400)
    skills_and_expertise: str = Field(min_length=50, max_length=400)
    skills_and_expertise_list: str = Field(
        min_length=20,
        max_length=300,
        description="list as a string (NPK compatible): \"['item1', 'item2', ...]\"",
    )
    hobbies_and_interests: str = Field(min_length=50, max_length=400)
    hobbies_and_interests_list: str = Field(
        min_length=20,
        max_length=300,
        description="list as a string",
    )
    career_goals_and_ambitions: str = Field(min_length=50, max_length=400)

    # ========== PAK domain 3 narratives (policy analysis) ==========
    creative_world_persona: str = Field(
        min_length=60,
        max_length=500,
        description="one's own creative world and aesthetic orientation (PAK domain). Differs from NPK arts_persona.",
    )
    network_persona: str = Field(min_length=50, max_length=400)
    living_persona: str = Field(min_length=60, max_length=500)
    support_persona: str = Field(min_length=40, max_length=400)


# ----------------------------------------------------------------------------
# 6. Metadata
# ----------------------------------------------------------------------------


class GenerationMetadata(BaseModel):
    pak_version: str = "0.1.0"
    sampler_seed: int
    grounding_tables_sha256: str
    llm_model: str
    llm_backend: str
    llm_temperature: float
    prompt_template_version: str
    generated_at: str
    validation_passed: bool


# ----------------------------------------------------------------------------
# 7. Complete persona
# ----------------------------------------------------------------------------


class PAKPersona(PAKPersonaQuant, PAKPersonaNarrative):
    """Complete PAK persona (quantitative NPK-superset + qualitative NPK-superset + PAK domain)."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


# ----------------------------------------------------------------------------
# 8. PAK-core nullable / unsupported columns
# ----------------------------------------------------------------------------

# Exposed columns that are best left honestly empty in single-PDF-source v0.1.
PAK_CORE_UNSUPPORTED_NULLABLE_COLUMNS: tuple[str, ...] = (
    "district",
    "marital_status",
    "military_status",
    "family_type",
    "housing_type",
    "bachelors_field",
    "art_field_secondary",
)


# ----------------------------------------------------------------------------
# 9. NPK column matrix (for the dataset card)
# ----------------------------------------------------------------------------

# Column identification: same name as NPK + adopts the same enum
NPK_COMPATIBLE_COLUMNS: tuple[str, ...] = (
    # quantitative (12)
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
    # qualitative (13)
    "professional_persona",
    "sports_persona",
    "arts_persona",
    "travel_persona",
    "culinary_persona",
    "family_persona",
    "persona",
    "cultural_background",
    "skills_and_expertise",
    "skills_and_expertise_list",
    "hobbies_and_interests",
    "hobbies_and_interests_list",
    "career_goals_and_ambitions",
)

# PAK domain additional columns
PAK_DOMAIN_COLUMNS: tuple[str, ...] = (
    # quantitative (PAK 17 — excluding uuid)
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
    # qualitative (4)
    "creative_world_persona",
    "network_persona",
    "living_persona",
    "support_persona",
)


# ----------------------------------------------------------------------------
# 10. JSON Schema output
# ----------------------------------------------------------------------------


def write_json_schema(target: Path | None = None) -> Path:
    if target is None:
        target = Path("data/prompts/schema.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    schema = PAKPersona.model_json_schema()
    target.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


if __name__ == "__main__":  # pragma: no cover
    p = write_json_schema()
    print(f"wrote {p}")
    print(f"NPK-compatible columns: {len(NPK_COMPATIBLE_COLUMNS)}")
    print(f"PAK domain columns:     {len(PAK_DOMAIN_COLUMNS)}")
    print(
        f"+ pak_uuid + meta       = total {1 + len(NPK_COMPATIBLE_COLUMNS) + len(PAK_DOMAIN_COLUMNS)}"
    )
