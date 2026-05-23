"""LLM-as-Judge — score persona quality from 0 to 10 using Gemma-4-31B (or a compatible model).

PAK uses the same stack as Nemotron-Personas-Korea, so the judge also runs on the same
OpenAI-compatible backend (vLLM/Ollama/NIM/OpenRouter). Anthropic Claude is optional.

Phase 05 only builds the call interface. Actual inference runs in the Phase 06 pilot.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from pak.config import settings
from pak.llm_client import get_client, parse_json_response

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Pydantic result schema
# ----------------------------------------------------------------------------


class JudgmentScores(BaseModel):
    realism: float = Field(ge=0, le=10)
    consistency: float = Field(ge=0, le=10)
    field_appropriateness: float = Field(ge=0, le=10)
    diversity: float = Field(ge=0, le=10)
    policy_utility: float = Field(ge=0, le=10)


class Judgment(BaseModel):
    scores: JudgmentScores
    overall_score: float = Field(ge=0, le=10)
    detected_issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    summary: str


# ----------------------------------------------------------------------------
# Prompt builders
# ----------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """당신은 합성 데이터 페르소나의 품질 평가자입니다.

다음 한국 문화예술인 페르소나(가상 인물)를 평가해 주세요. 이 페르소나는
「2024 예술인 실태조사」를 기반으로 합성 생성되었으며, 한국어 LLM 학습과
문화예술 정책 연구에 사용될 예정입니다.

[평가 기준 — 각 0~10점]

1. realism — 한국 문화예술계의 실제 모습을 반영하는가?
2. consistency — 정량 변수와 narrative가 모순되지 않는가?
3. field_appropriateness — 분야 어휘와 활동이 잘 표현되었는가?
4. diversity — 클리셰 없이 개성 있는가?
5. policy_utility — living/support narrative가 정책 분석에 활용 가능한가?

[detected_issues — 발견되는 항목 코드 모두 나열]
"cliche", "factual_inconsistency", "field_mismatch", "real_person_reference",
"specific_work_name", "narrative_contradiction", "length_violation",
"boring_generic", "stereotypical", "missing_element"

[출력 형식]
다음 JSON Schema에 맞는 JSON 객체만 반환하세요. 다른 텍스트는 절대 포함 금지.

{
  "scores": {
    "realism": <0~10>, "consistency": <0~10>, "field_appropriateness": <0~10>,
    "diversity": <0~10>, "policy_utility": <0~10>
  },
  "overall_score": <0~10>,
  "detected_issues": ["..."],
  "suggestions": ["...", "..."],
  "summary": "한 문장 요약"
}

overall_score 권장: realism*0.25 + consistency*0.25 + field_appropriateness*0.20
+ diversity*0.20 + policy_utility*0.10.
"""


def render_quant_block(quant: dict) -> str:
    return f"""[페르소나 정량 정보]
- 성별 / 나이: {quant.get("sex", "?")}, {quant.get("age", "?")}세 ({quant.get("age_band", "?")})
- 활동 지역: {quant.get("province", "?")}
- 학력: {quant.get("education_level", "?")}
- 분야: {quant.get("art_field_primary", "?")}
- 경력: {quant.get("career_years", "?")}년 ({quant.get("career_band", "?")})
- 고용: {quant.get("employment_type", "?")}, freelance={quant.get("is_freelance", "?")}
- 예술 개인소득: {quant.get("individual_art_income_bracket", "?")}
- 가구 총소득: {quant.get("household_income_bracket", "?")}
- 계약 경험: {quant.get("has_contract_experience", "?")}
- 표준계약서 사용: {quant.get("uses_standard_contract", "?")}
- 저작권 보유: {quant.get("has_copyright", "?")}
- 경력 단절: {quant.get("had_career_break", "?")}
- 해외 활동: {quant.get("has_overseas_experience", "?")}"""


def render_narrative_block(narratives: dict[str, str]) -> str:
    parts = []
    for k, v in narratives.items():
        parts.append(f"[{k}]\n{v}\n")
    return "\n".join(parts)


def build_judge_user_message(quant: dict, narratives: dict[str, str]) -> str:
    return (
        render_quant_block(quant)
        + "\n\n[페르소나 narrative]\n\n"
        + render_narrative_block(narratives)
    )


# ----------------------------------------------------------------------------
# Response parsing
# ----------------------------------------------------------------------------


def parse_judgment(text: str) -> Judgment:
    return Judgment.model_validate(parse_json_response(text))


# Alias kept for compatibility (imported by other modules/tests)
def extract_json_object(text: str) -> str:
    from pak.llm_client import extract_json_object as _impl

    return _impl(text)


# ----------------------------------------------------------------------------
# Inference call (backend-agnostic)
# ----------------------------------------------------------------------------


@dataclass
class JudgeConfig:
    model: str = field(default_factory=lambda: settings.llm_judge_model)
    max_tokens: int = 1500
    temperature: float = 0.0
    backend: str | None = None  # if None, uses settings.llm_backend
    log_path: Path | None = None


def judge_persona_sync(
    quant: dict, narratives: dict[str, str], config: JudgeConfig | None = None
) -> Judgment:
    """Synchronous single call. backend is OpenAI-compatible (vLLM/Ollama/NIM/OpenRouter)
    or anthropic. Determined by settings.llm_backend."""
    cfg = config or JudgeConfig()
    client = get_client(backend=cfg.backend)
    user_msg = build_judge_user_message(quant, narratives)

    result = client.chat(
        model=cfg.model,
        system=JUDGE_SYSTEM_PROMPT,
        user=user_msg,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        response_format={"type": "json_object"},  # supported by both vLLM and OpenAI-compatible backends
    )
    judgment = parse_judgment(result.text)

    if cfg.log_path:
        cfg.log_path.parent.mkdir(parents=True, exist_ok=True)
        with cfg.log_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "model": cfg.model,
                        "backend": result.backend,
                        "input_tokens": result.usage.input_tokens,
                        "output_tokens": result.usage.output_tokens,
                        "judgment": judgment.model_dump(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return judgment


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------


def aggregate_judgments(judgments: list[Judgment]) -> dict[str, Any]:
    if not judgments:
        return {"n": 0}
    import numpy as np

    arr = {
        k: np.array([getattr(j.scores, k) for j in judgments])
        for k in ("realism", "consistency", "field_appropriateness", "diversity", "policy_utility")
    }
    overall = np.array([j.overall_score for j in judgments])
    issue_counter: dict[str, int] = {}
    for j in judgments:
        for issue in j.detected_issues:
            issue_counter[issue] = issue_counter.get(issue, 0) + 1

    return {
        "n": len(judgments),
        "scores_mean": {k: float(v.mean()) for k, v in arr.items()},
        "scores_pct_below_5": {k: float((v < 5).mean()) for k, v in arr.items()},
        "scores_pct_above_8": {k: float((v >= 8).mean()) for k, v in arr.items()},
        "overall_mean": float(overall.mean()),
        "overall_median": float(np.median(overall)),
        "issue_frequency": {k: v / len(judgments) for k, v in issue_counter.items()},
    }
