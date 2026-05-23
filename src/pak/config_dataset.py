"""PAK dataset configuration builder and compiler.

Reproduces DataDesigner's `config -> compile -> execute` structure, adapted for PAK-core.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from pak import __version__
from pak.columns import (
    ExpressionColumnSpec,
    PAKColumnSpec,
    SamplerColumnSpec,
    StructuredNarrativeColumnSpec,
    ValidationColumnSpec,
)
from pak.prompt_builder import COMMON_NARRATIVES, DOMAIN_NARRATIVES
from pak.schema import PAKPersonaNarrative, PAKPersonaQuant


ColumnSpecT = Annotated[
    SamplerColumnSpec | ExpressionColumnSpec | StructuredNarrativeColumnSpec | ValidationColumnSpec,
    Field(discriminator="column_type"),
]


class DatasetProfilerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    profiler_type: str
    target_columns: list[str] = Field(default_factory=list)


class PAKDatasetConfig(BaseModel):
    """Declarative config for the PAK generation pipeline."""

    model_config = ConfigDict(
        protected_namespaces=(),
        use_enum_values=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    name: str
    version: str = __version__
    columns: list[ColumnSpecT] = Field(min_length=1)
    profilers: list[DatasetProfilerSpec] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def fingerprint(self) -> dict[str, str | int]:
        payload = json.dumps(_serialize_for_fingerprint(self), ensure_ascii=False, sort_keys=True)
        return {
            "config_hash": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "config_hash_algo": "sha256",
            "config_hash_version": 1,
        }


@dataclass(frozen=True)
class CompiledPAKDatasetConfig:
    config: PAKDatasetConfig
    columns_by_name: dict[str, PAKColumnSpec]
    topological_order: list[str]
    allowed_references: list[str]
    side_effect_to_producer: dict[str, str]
    narrative_column_name: str

    @property
    def narrative_spec(self) -> StructuredNarrativeColumnSpec:
        spec = self.columns_by_name[self.narrative_column_name]
        assert isinstance(spec, StructuredNarrativeColumnSpec)
        return spec

    @property
    def validation_specs(self) -> list[ValidationColumnSpec]:
        return [
            spec for spec in self.columns_by_name.values() if isinstance(spec, ValidationColumnSpec)
        ]

    def get_column(self, name: str) -> PAKColumnSpec:
        return self.columns_by_name[name]

    def resolve_reference(self, name: str) -> str:
        if name in self.columns_by_name:
            return name
        return self.side_effect_to_producer.get(name, name)


class PAKDatasetConfigBuilder:
    """Mutable builder for PAKDatasetConfig."""

    def __init__(self, *, name: str, version: str = __version__, metadata: dict[str, Any] | None = None):
        self._name = name
        self._version = version
        self._metadata = dict(metadata or {})
        self._columns: dict[str, PAKColumnSpec] = {}
        self._profilers: list[DatasetProfilerSpec] = []

    @property
    def allowed_references(self) -> list[str]:
        refs = set(self._columns)
        for column in self._columns.values():
            refs.update(column.side_effect_columns)
        return sorted(refs)

    def add_column(self, column: PAKColumnSpec) -> PAKDatasetConfigBuilder:
        if column.name in self._columns:
            raise ValueError(f"duplicate column name in builder: {column.name!r}")
        self._columns[column.name] = column
        return self

    def add_profiler(self, profiler: DatasetProfilerSpec) -> PAKDatasetConfigBuilder:
        self._profilers.append(profiler)
        return self

    def build(self) -> PAKDatasetConfig:
        return PAKDatasetConfig(
            name=self._name,
            version=self._version,
            columns=list(self._columns.values()),
            profilers=list(self._profilers),
            metadata=dict(self._metadata),
        )


def compile_pak_dataset_config(config: PAKDatasetConfig) -> CompiledPAKDatasetConfig:
    columns_by_name: dict[str, PAKColumnSpec] = {}
    side_effect_to_producer: dict[str, str] = {}

    for column in config.columns:
        if column.name in columns_by_name:
            raise ValueError(f"duplicate column name: {column.name!r}")
        columns_by_name[column.name] = column
        for side_effect in column.side_effect_columns:
            if side_effect in side_effect_to_producer and side_effect_to_producer[side_effect] != column.name:
                raise ValueError(
                    f"side effect column {side_effect!r} has multiple producers: "
                    f"{side_effect_to_producer[side_effect]!r}, {column.name!r}"
                )
            side_effect_to_producer[side_effect] = column.name

    allowed_references = sorted(set(columns_by_name) | set(side_effect_to_producer))
    edges: dict[str, set[str]] = {name: set() for name in columns_by_name}
    indegree: dict[str, int] = {name: 0 for name in columns_by_name}

    def resolve_reference(name: str) -> str:
        if name in columns_by_name:
            return name
        if name in side_effect_to_producer:
            return side_effect_to_producer[name]
        raise ValueError(f"unknown referenced column: {name!r}")

    narrative_columns: list[str] = []
    for name, column in columns_by_name.items():
        if isinstance(column, StructuredNarrativeColumnSpec):
            narrative_columns.append(name)
        for required in column.required_columns:
            upstream = resolve_reference(required)
            if upstream == name:
                continue
            if name not in edges[upstream]:
                edges[upstream].add(name)
                indegree[name] += 1

    if len(narrative_columns) != 1:
        raise ValueError(
            f"expected exactly one structured narrative column, found {len(narrative_columns)}"
        )

    queue = deque([name for name, degree in indegree.items() if degree == 0])
    topo: list[str] = []
    while queue:
        node = queue.popleft()
        topo.append(node)
        for downstream in sorted(edges[node]):
            indegree[downstream] -= 1
            if indegree[downstream] == 0:
                queue.append(downstream)

    if len(topo) != len(columns_by_name):
        unresolved = sorted(name for name, degree in indegree.items() if degree > 0)
        raise ValueError(f"dataset config contains cyclic dependencies: {unresolved}")

    return CompiledPAKDatasetConfig(
        config=config,
        columns_by_name=columns_by_name,
        topological_order=topo,
        allowed_references=allowed_references,
        side_effect_to_producer=side_effect_to_producer,
        narrative_column_name=narrative_columns[0],
    )


def build_default_pak_core_dataset_config() -> PAKDatasetConfig:
    """Current PAK-core generation graph in declarative config form."""

    quant_fields = [field for field in PAKPersonaQuant.model_fields if field != "pak_uuid"]
    builder = PAKDatasetConfigBuilder(
        name="pak_core_default",
        metadata={
            "grounding": "2024 예술인 실태조사",
            "design_reference": "NVIDIA NeMo Data Designer philosophy",
        },
    )

    builder.add_column(
        ExpressionColumnSpec(
            name="pak_uuid",
            expression_kind="uuid4",
        )
    )

    # Raw grounding-chain outputs
    builder.add_column(
        SamplerColumnSpec(
            name="art_field_primary",
            sampler_kind="grounding-chain",
            source_key="art_field_primary",
        )
    )
    for name in (
        "sex_age",
        "province_raw",
        "education_level_pak",
        "career_band",
        "employment_type",
        "is_freelance",
        "individual_art_income_bracket",
        "has_contract_experience",
        "uses_standard_contract_raw",
        "has_copyright",
        "had_career_break",
        "has_overseas_experience",
    ):
        source_key = {
            "province_raw": "province",
            "education_level_pak": "education_level",
            "uses_standard_contract_raw": "uses_standard_contract",
        }.get(name, name)
        builder.add_column(
            SamplerColumnSpec(
                name=name,
                sampler_kind="grounding-chain",
                source_key=source_key,
                parents=["art_field_primary"] if name != "sex_age" else ["art_field_primary"],
                drop=name in {"sex_age", "province_raw"},
            )
        )

    # Deterministic expressions
    builder.add_column(
        ExpressionColumnSpec(
            name="sex",
            expression_kind="split_sex_age_sex",
            parents=["sex_age"],
        )
    )
    builder.add_column(
        ExpressionColumnSpec(
            name="age_band",
            expression_kind="split_sex_age_age_band",
            parents=["sex_age"],
        )
    )
    builder.add_column(
        ExpressionColumnSpec(
            name="province",
            expression_kind="alias_province_to_npk",
            parents=["province_raw"],
        )
    )
    builder.add_column(
        ExpressionColumnSpec(
            name="has_secondary_job",
            expression_kind="employment_type_eq_겸업",
            parents=["employment_type"],
        )
    )
    builder.add_column(
        ExpressionColumnSpec(
            name="education_level",
            expression_kind="map_pak_education_to_npk",
            parents=["education_level_pak"],
        )
    )
    builder.add_column(
        ExpressionColumnSpec(
            name="uses_standard_contract",
            expression_kind="contract_experience_condition",
            parents=["has_contract_experience", "uses_standard_contract_raw"],
        )
    )
    builder.add_column(
        ExpressionColumnSpec(
            name="country",
            expression_kind="constant",
            metadata={"value": "대한민국"},
        )
    )
    builder.add_column(
        ExpressionColumnSpec(
            name="art_field_secondary",
            expression_kind="constant",
            metadata={"value": None},
        )
    )
    builder.add_column(
        ExpressionColumnSpec(
            name="marital_status",
            expression_kind="constant",
            metadata={"value": None},
        )
    )
    builder.add_column(
        ExpressionColumnSpec(
            name="military_status",
            expression_kind="constant",
            metadata={"value": None},
        )
    )
    builder.add_column(
        ExpressionColumnSpec(
            name="family_type",
            expression_kind="constant",
            metadata={"value": None},
        )
    )
    builder.add_column(
        ExpressionColumnSpec(
            name="housing_type",
            expression_kind="constant",
            metadata={"value": None},
        )
    )
    builder.add_column(
        ExpressionColumnSpec(
            name="bachelors_field",
            expression_kind="constant",
            metadata={"value": None},
        )
    )

    # Derived samplers
    builder.add_column(
        SamplerColumnSpec(
            name="age",
            sampler_kind="derived",
            parents=["art_field_primary", "age_band", "career_band"],
            operation="sample_age_for_career",
        )
    )
    builder.add_column(
        ExpressionColumnSpec(
            name="district",
            expression_kind="constant",
            metadata={"value": None},
        )
    )
    builder.add_column(
        SamplerColumnSpec(
            name="occupation",
            sampler_kind="derived",
            parents=["art_field_primary"],
            operation="sample_occupation",
        )
    )
    builder.add_column(
        SamplerColumnSpec(
            name="household_income_bracket",
            sampler_kind="derived",
            parents=["art_field_primary"],
            operation="sample_household_income",
        )
    )
    builder.add_column(
        SamplerColumnSpec(
            name="career_years",
            sampler_kind="derived",
            parents=["art_field_primary", "age", "career_band"],
            operation="sample_career_years",
        )
    )

    # Structured narrative bundle
    builder.add_column(
        StructuredNarrativeColumnSpec(
            name="narratives",
            drop=True,
            output_model=PAKPersonaNarrative,
            anchor_columns=[
                "occupation",
                "province",
                "art_field_primary",
                "career_years",
                "employment_type",
                "has_secondary_job",
            ],
            anchor_labels={
                "occupation": "canonical occupation",
                "province": "canonical primary region",
                "art_field_primary": "canonical art field",
                "career_years": "canonical career_years",
                "employment_type": "canonical employment_type",
                "has_secondary_job": "has_secondary_job",
            },
            context_columns=quant_fields,
            system_rules=[
                "occupation은 가장 우선되는 본업 정체성이다. 인접 직무나 다른 역할로 바꾸지 말 것.",
                "persona / professional_persona / living_persona의 활동 거점은 primary region과 일치해야 한다.",
                "travel_persona를 제외하고 다른 시도명을 주 활동 지역처럼 쓰지 말 것.",
                "has_secondary_job이 False이면 겸업, 알바, 부업, 외주를 별도 생계축으로 만들지 말 것.",
                "career_years는 조기 훈련 기간이 아니라 예술활동 경력이다. age와 맞지 않게 수십 년 경력, 평생 활동, 지나치게 이른 데뷔를 쓰지 말 것.",
                "employment_type은 현재의 전업/겸업 상태다. 'N년 전업', 'N년 겸업'처럼 career_years가 현재 고용형태의 지속기간인 것처럼 쓰지 말 것.",
                "금지 예시는 '12년 전업 작가', '전업으로 12년', '12년간 겸업 음악가'이다. 필요하면 '활동 경력은 12년이며, 현재는 전업으로 작업 시간을 확보한다'처럼 분리해 쓸 것.",
            ],
            domain_prompt_categories=list(DOMAIN_NARRATIVES),
            common_prompt_categories=list(COMMON_NARRATIVES),
        )
    )

    # Validation node
    builder.add_column(
        ValidationColumnSpec(
            name="row_validation",
            drop=True,
            validation_kind="row-pipeline",
            blocking_on_error=True,
            target_columns=["pak_uuid", *quant_fields, *list(PAKPersonaNarrative.model_fields)],
        )
    )

    builder.add_profiler(
        DatasetProfilerSpec(
            name="post_verification",
            profiler_type="post-check",
            target_columns=["pak_uuid", *quant_fields, *list(PAKPersonaNarrative.model_fields)],
        )
    )
    return builder.build()


@lru_cache(maxsize=1)
def get_default_pak_core_dataset_config() -> CompiledPAKDatasetConfig:
    return compile_pak_dataset_config(build_default_pak_core_dataset_config())


def _serialize_for_fingerprint(config: PAKDatasetConfig) -> dict[str, Any]:
    columns: list[dict[str, Any]] = []
    for column in config.columns:
        payload = column.model_dump(exclude_none=True)
        if isinstance(column, StructuredNarrativeColumnSpec):
            payload["output_model"] = column.output_model.__name__
            payload["output_fields"] = column.output_fields
        columns.append(payload)
    return {
        "name": config.name,
        "version": config.version,
        "columns": columns,
        "profilers": [profiler.model_dump(exclude_none=True) for profiler in config.profilers],
        "metadata": config.metadata,
    }
