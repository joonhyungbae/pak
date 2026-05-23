"""PAK dataset config / compiler unit tests."""

from __future__ import annotations

import pytest

from pak.columns import ExpressionColumnSpec
from pak.config_dataset import (
    PAKDatasetConfigBuilder,
    build_default_pak_core_dataset_config,
    compile_pak_dataset_config,
    get_default_pak_core_dataset_config,
)
from pak.schema import PAKPersonaNarrative, PAKPersonaQuant


def test_default_dataset_config_compiles() -> None:
    compiled = get_default_pak_core_dataset_config()
    assert compiled.config.name == "pak_core_default"
    assert compiled.narrative_column_name == "narratives"
    assert compiled.resolve_reference("persona") == "narratives"
    assert compiled.resolve_reference("hobbies_and_interests_list") == "narratives"


def test_default_dataset_config_includes_full_quant_and_narrative_schema() -> None:
    compiled = get_default_pak_core_dataset_config()
    assert "pak_uuid" in compiled.columns_by_name
    assert compiled.narrative_spec.output_fields == list(PAKPersonaNarrative.model_fields)

    validation = compiled.columns_by_name["row_validation"]
    assert "pak_uuid" in validation.required_columns
    for field in PAKPersonaQuant.model_fields:
        assert field in compiled.allowed_references


def test_topological_order_respects_key_dependencies() -> None:
    compiled = get_default_pak_core_dataset_config()
    order = compiled.topological_order
    assert order.index("sex_age") < order.index("sex")
    assert order.index("sex_age") < order.index("age_band")
    assert order.index("age_band") < order.index("age")
    assert order.index("age") < order.index("career_years")
    assert order.index("narratives") < order.index("row_validation")


def test_dataset_config_fingerprint_shape() -> None:
    fingerprint = get_default_pak_core_dataset_config().config.fingerprint()
    assert fingerprint["config_hash_algo"] == "sha256"
    assert fingerprint["config_hash_version"] == 1
    assert isinstance(fingerprint["config_hash"], str)
    assert len(fingerprint["config_hash"]) == 64


def test_builder_rejects_duplicate_columns() -> None:
    builder = PAKDatasetConfigBuilder(name="dup-test")
    builder.add_column(
        ExpressionColumnSpec(
            name="foo",
            expression_kind="constant",
            metadata={"value": 1},
        )
    )
    with pytest.raises(ValueError, match="duplicate column name"):
        builder.add_column(
            ExpressionColumnSpec(
                name="foo",
                expression_kind="constant",
                metadata={"value": 2},
            )
        )


def test_default_dataset_config_builder_round_trip() -> None:
    config = build_default_pak_core_dataset_config()
    compiled = compile_pak_dataset_config(config)
    assert compiled.topological_order
    assert compiled.validation_specs
