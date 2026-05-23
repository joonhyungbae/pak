"""PAK dataset config column specifications.

A reduced reimplementation of DataDesigner's column config layer, adapted for PAK.
The current scope covers the 4 types needed by the PAK-core generation pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ColumnType(str, Enum):
    SAMPLER = "sampler"
    EXPRESSION = "expression"
    LLM_STRUCTURED = "llm-structured"
    VALIDATION = "validation"


class PAKColumnConfigBase(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=(),
        use_enum_values=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )


class PAKColumnSpec(PAKColumnConfigBase, ABC):
    """Abstract base class for all PAK dataset column specs."""

    name: str
    drop: bool = False
    column_type: str

    @property
    @abstractmethod
    def required_columns(self) -> list[str]:
        """Columns that must exist before this column can be materialized."""

    @property
    def side_effect_columns(self) -> list[str]:
        """Additional output fields produced by this column."""
        return []


class SamplerColumnSpec(PAKColumnSpec):
    """Sampler-backed column.

    sampler_kind:
    - ``grounding-chain``: reads the sampler chain raw output as-is.
    - ``derived``: samples a value based on other already-sampled columns.
    """

    sampler_kind: Literal["grounding-chain", "derived"]
    source_key: str | None = None
    parents: list[str] = Field(default_factory=list)
    operation: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    column_type: Literal[ColumnType.SAMPLER] = ColumnType.SAMPLER

    @property
    def required_columns(self) -> list[str]:
        return list(self.parents)


class ExpressionColumnSpec(PAKColumnSpec):
    """Deterministic derived column."""

    expression_kind: str
    parents: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    column_type: Literal[ColumnType.EXPRESSION] = ColumnType.EXPRESSION

    @property
    def required_columns(self) -> list[str]:
        return list(self.parents)


class StructuredNarrativeColumnSpec(PAKColumnSpec):
    """Single-call structured narrative generation column."""

    output_model: type[BaseModel]
    model_alias: str | None = None
    anchor_columns: list[str] = Field(default_factory=list)
    anchor_labels: dict[str, str] = Field(default_factory=dict)
    context_columns: list[str] = Field(default_factory=list)
    system_rules: list[str] = Field(default_factory=list)
    domain_prompt_categories: list[str] = Field(default_factory=list)
    common_prompt_categories: list[str] = Field(default_factory=list)
    column_type: Literal[ColumnType.LLM_STRUCTURED] = ColumnType.LLM_STRUCTURED

    @property
    def required_columns(self) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for col in [*self.anchor_columns, *self.context_columns]:
            if col not in seen:
                ordered.append(col)
                seen.add(col)
        return ordered

    @property
    def output_fields(self) -> list[str]:
        return list(self.output_model.model_fields)

    @property
    def side_effect_columns(self) -> list[str]:
        return self.output_fields

    @model_validator(mode="after")
    def _validate_anchor_labels(self) -> StructuredNarrativeColumnSpec:
        unknown = set(self.anchor_labels) - set(self.anchor_columns)
        if unknown:
            raise ValueError(
                f"anchor_labels contains unknown columns for {self.name!r}: {sorted(unknown)}"
            )
        return self


class ValidationColumnSpec(PAKColumnSpec):
    """Validation step modeled as a column-like config node."""

    validation_kind: Literal["row-pipeline", "post-check"]
    target_columns: list[str] = Field(default_factory=list)
    blocking_on_error: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
    column_type: Literal[ColumnType.VALIDATION] = ColumnType.VALIDATION

    @property
    def required_columns(self) -> list[str]:
        return list(self.target_columns)

