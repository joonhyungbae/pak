"""Cliche detection (targeting generated narrative text).

Separate from prompt_validator's FORBIDDEN_CLICHES. Checks whether cliches appear in the
LLM-generated narrative text itself, not in the prompt. The safe-keyword exemption is not
applied (a narrative must not use cliches except when quoting them).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

CLICHE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("가난하지만 자유로운", r"가난하지만\s*자유로"),
    ("고독한 천재", r"고독한\s*천재"),
    ("보헤미안", r"보헤미안"),
    ("예술혼 불태우는", r"예술혼\s*(을\s*)?불태"),
    ("순수한 영혼", r"순수한\s*영혼"),
    ("고뇌하는 예술가", r"고뇌하는\s*예술가"),
    ("타고난 재능", r"타고난\s*재능"),
    ("운명적으로 만난", r"운명적으로\s*만난"),
    ("천부적 재능", r"천부적\s*재능"),
    ("불꽃같은 열정", r"불꽃\s*같은\s*열정"),
)


@dataclass
class ClicheHit:
    label: str
    span_start: int
    span_end: int
    matched_text: str
    in_field: str  # narrative kind (professional / creative_world / ...)


def detect_cliches(narratives: dict[str, str]) -> list[ClicheHit]:
    hits: list[ClicheHit] = []
    for nar_field, text in narratives.items():
        for label, pat in CLICHE_PATTERNS:
            for m in re.finditer(pat, text):
                hits.append(
                    ClicheHit(
                        label=label,
                        span_start=m.start(),
                        span_end=m.end(),
                        matched_text=m.group(0),
                        in_field=nar_field,
                    )
                )
    return hits


def cliche_frequency(personas_narratives: list[dict[str, str]]) -> dict[str, float]:
    """Appearance rate of each cliche across all personas (0 to 1)."""
    n = len(personas_narratives)
    if n == 0:
        return {}
    counter: Counter[str] = Counter()
    for nar in personas_narratives:
        seen: set[str] = set()
        for hit in detect_cliches(nar):
            if hit.label in seen:
                continue
            seen.add(hit.label)
            counter[hit.label] += 1
    return {label: counter[label] / n for label, _ in CLICHE_PATTERNS}
