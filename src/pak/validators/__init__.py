"""Validation pipeline integration (Phase 05).

`ValidationPipeline.validate_one(persona)` -> `ValidationResult`.
Schema validation is handled automatically by Pydantic. This module bundles the other
five (consistency / cliche / diversity / distribution / llm_judge).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

from pak.validators.cliche import ClicheHit, detect_cliches
from pak.validators.consistency import Issue as ConsistencyIssue
from pak.validators.consistency import check_all as check_consistency

logger = logging.getLogger(__name__)


Severity = Literal["error", "warning", "info"]


@dataclass
class ValidationResult:
    pak_uuid: str
    consistency_issues: list[ConsistencyIssue] = field(default_factory=list)
    cliche_hits: list[ClicheHit] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.consistency_issues) or bool(self.cliche_hits)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == "warning" for i in self.consistency_issues)

    def severity_counts(self) -> dict[str, int]:
        c = {"error": 0, "warning": 0, "info": 0, "cliche": len(self.cliche_hits)}
        for i in self.consistency_issues:
            c[i.severity] = c.get(i.severity, 0) + 1
        return c


@dataclass
class ValidationConfig:
    enable_consistency: bool = True
    enable_cliche: bool = True
    # diversity / distribution are called separately at the batch level
    # llm_judge is not triggered directly by ValidationPipeline (cost)


class ValidationPipeline:
    def __init__(self, config: ValidationConfig | None = None) -> None:
        self.config = config or ValidationConfig()

    def validate_one(
        self, *, pak_uuid: str, quant: dict, narratives: dict[str, str]
    ) -> ValidationResult:
        result = ValidationResult(pak_uuid=pak_uuid)
        if self.config.enable_consistency:
            result.consistency_issues = check_consistency(quant, narratives)
        if self.config.enable_cliche:
            result.cliche_hits = detect_cliches(narratives)
        return result

    def validate_batch(
        self,
        personas: Iterable[tuple[str, dict, dict[str, str]]],
    ) -> list[ValidationResult]:
        return [self.validate_one(pak_uuid=u, quant=q, narratives=n) for u, q, n in personas]


def summarize_batch(results: list[ValidationResult]) -> dict[str, float | int]:
    n = len(results)
    if n == 0:
        return {"n": 0}
    pass_count = sum(1 for r in results if not r.has_errors)
    cliche_count = sum(len(r.cliche_hits) for r in results)
    consistency_warning_count = sum(
        sum(1 for i in r.consistency_issues if i.severity == "warning") for r in results
    )
    return {
        "n": n,
        "pass_rate": pass_count / n,
        "cliche_count": cliche_count,
        "cliche_per_persona": cliche_count / n,
        "consistency_warning_count": consistency_warning_count,
    }
