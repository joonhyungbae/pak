"""DataDesigner-style preview / gate / report helpers for PAK-core."""

from __future__ import annotations

import ast
import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pak.config_dataset import get_default_pak_core_dataset_config
from pak.generate import GenerateConfig, run as run_generation
from pak.schema import (
    PAK_CORE_UNSUPPORTED_NULLABLE_COLUMNS,
    PAKPersonaNarrative,
    PAKPersonaQuant,
)
from pak.validators.diversity import pairwise_similarity_token
from pak.verification.post_check import report_to_dict, run_post_verification

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreviewThresholds:
    generation_success_rate_min: float = 1.0
    schema_pass_rate_min: float = 1.0
    validation_error_rate_max: float = 0.0
    validation_warning_rate_max: float = 0.10
    exact_duplicate_narrative_rows_max: int = 0
    hobby_set_duplicate_rate_max: float = 0.02
    null_only_exposed_columns_max: int = 0
    district_placeholder_rate_max: float = 0.0
    hobby_atom_top1_share_max: float = 0.35
    enforce_field_marginal_gate: bool = True
    require_field_marginal_pass: bool = True
    enforce_joint_gate: bool = True
    require_joint_pass: bool = True
    enforce_fact_check_gate: bool = True
    require_all_fact_checks_pass: bool = True


@dataclass(frozen=True)
class PreviewGateCheck:
    name: str
    observed: Any
    comparator: str
    threshold: Any
    passed: bool
    note: str = ""


@dataclass(frozen=True)
class PreviewGateResult:
    passed: bool
    checks: list[PreviewGateCheck]


@dataclass(frozen=True)
class PreviewArtifacts:
    parquet_path: Path
    post_verification_path: Path
    preview_report_json_path: Path
    preview_report_md_path: Path


@dataclass(frozen=True)
class PreviewRunResult:
    artifacts: PreviewArtifacts
    gate: PreviewGateResult
    report: dict[str, Any]


def infer_preview_gate_profile(generation_metadata: dict[str, Any]) -> str:
    eval_set = generation_metadata.get("eval_set")
    if isinstance(eval_set, dict):
        return "fixed_eval_set"
    return "release_preview"


def default_preview_thresholds(profile: str) -> PreviewThresholds:
    thresholds = PreviewThresholds()
    if profile == "fixed_eval_set":
        return replace(
            thresholds,
            enforce_field_marginal_gate=False,
            enforce_joint_gate=False,
            enforce_fact_check_gate=False,
        )
    return thresholds


def _load_generation_metadata(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "generation_metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_list_string(value: Any) -> tuple[str, ...]:
    if not isinstance(value, str) or not value.strip():
        return ()
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        parsed = None

    if isinstance(parsed, list):
        return tuple(sorted({str(item).strip() for item in parsed if str(item).strip()}))

    parts = [part.strip(" []'\"") for part in value.split(",")]
    return tuple(sorted({part for part in parts if part}))


def exact_duplicate_narrative_rows(df: pd.DataFrame) -> int:
    narrative_cols = list(PAKPersonaNarrative.model_fields)
    subset = [col for col in narrative_cols if col in df.columns]
    if not subset:
        return 0
    return int(df.duplicated(subset=subset).sum())


def hobby_set_duplicate_stats(df: pd.DataFrame) -> dict[str, Any]:
    if "hobbies_and_interests_list" not in df.columns:
        return {
            "duplicate_rows": 0,
            "duplicate_rate": 0.0,
            "n_unique_sets": 0,
            "top_duplicate_sets": [],
        }

    normalized = [_parse_list_string(value) for value in df["hobbies_and_interests_list"].tolist()]
    counter: Counter[tuple[str, ...]] = Counter(item for item in normalized if item)
    duplicate_rows = sum(count for count in counter.values() if count > 1)
    top_duplicate_sets = [
        {"items": list(items), "count": count}
        for items, count in counter.most_common()
        if count > 1
    ][:5]
    n = len(df)
    return {
        "duplicate_rows": int(duplicate_rows),
        "duplicate_rate": float(duplicate_rows / max(n, 1)),
        "n_unique_sets": int(len(counter)),
        "top_duplicate_sets": top_duplicate_sets,
    }


def hobby_atom_frequency_stats(df: pd.DataFrame) -> dict[str, Any]:
    if "hobbies_and_interests_list" not in df.columns:
        return {
            "top1_share": 0.0,
            "top3_share_sum": 0.0,
            "n_unique_atoms": 0,
            "top_atoms": [],
        }

    counter: Counter[str] = Counter()
    for value in df["hobbies_and_interests_list"].tolist():
        counter.update(_parse_list_string(value))

    n = max(len(df), 1)
    top_atoms = [
        {"atom": atom, "count": count, "share": float(count / n)}
        for atom, count in counter.most_common(10)
    ]
    return {
        "top1_share": float(top_atoms[0]["share"]) if top_atoms else 0.0,
        "top3_share_sum": float(sum(item["share"] for item in top_atoms[:3])),
        "n_unique_atoms": int(len(counter)),
        "top_atoms": top_atoms,
    }


def district_placeholder_stats(df: pd.DataFrame) -> dict[str, Any]:
    if "district" not in df.columns:
        return {
            "placeholder_rows": 0,
            "placeholder_rate": 0.0,
        }

    patterns = ("시군구미상", "지역미상", "unknown", "미상")
    values = df["district"].fillna("").astype(str).tolist()
    placeholder_rows = sum(1 for value in values if any(pattern in value for pattern in patterns))
    n = max(len(values), 1)
    return {
        "placeholder_rows": int(placeholder_rows),
        "placeholder_rate": float(placeholder_rows / n),
    }


def exposed_column_fill_stats(df: pd.DataFrame) -> dict[str, Any]:
    exposed_columns = list(PAKPersonaQuant.model_fields) + list(PAKPersonaNarrative.model_fields)
    zero_fill_columns: list[str] = []
    fill_rates: dict[str, float] = {}
    ignored_unsupported_columns = [
        column for column in PAK_CORE_UNSUPPORTED_NULLABLE_COLUMNS if column in exposed_columns
    ]
    for column in exposed_columns:
        if column not in df.columns:
            fill_rates[column] = 0.0
            if column not in ignored_unsupported_columns:
                zero_fill_columns.append(column)
            continue
        series = df[column]
        filled = series.notna()
        fill_rate = float(filled.mean()) if len(series) else 0.0
        fill_rates[column] = fill_rate
        if fill_rate == 0.0 and column not in ignored_unsupported_columns:
            zero_fill_columns.append(column)

    return {
        "zero_fill_columns": zero_fill_columns,
        "zero_fill_count": len(zero_fill_columns),
        "fill_rates": fill_rates,
        "ignored_unsupported_columns": ignored_unsupported_columns,
        "ignored_unsupported_count": len(ignored_unsupported_columns),
    }


def professional_persona_diversity(df: pd.DataFrame) -> dict[str, Any]:
    if "professional_persona" not in df.columns:
        return {
            "pairs_compared": 0,
            "mean_similarity": 0.0,
            "max_similarity": 0.0,
            "pct_above_threshold": 0.0,
            "threshold": 0.85,
        }
    report = pairwise_similarity_token(
        df["professional_persona"].astype(str).tolist(),
        sample_pairs=min(500, max(len(df) * 2, 1)),
        threshold=0.85,
        seed=0,
    )
    return {
        "pairs_compared": report.pairs_compared,
        "mean_similarity": report.mean_similarity,
        "max_similarity": report.max_similarity,
        "pct_above_threshold": report.pct_above_threshold,
        "threshold": report.threshold,
    }


def evaluate_preview_gate(
    *,
    post_report: dict[str, Any],
    extra_checks: dict[str, Any],
    thresholds: PreviewThresholds,
) -> PreviewGateResult:
    row_validation = post_report["row_validation"]
    n_personas = max(int(post_report["n_personas"]), 1)
    validation_error_rate = row_validation["n_with_errors"] / n_personas
    validation_warning_rate = row_validation["n_with_warnings"] / n_personas
    fact_checks = post_report["fact_checks"]
    fact_passed = sum(1 for check in fact_checks if check["passed"])
    fact_total = len(fact_checks)

    checks: list[PreviewGateCheck] = []
    generation_completion = extra_checks.get("generation_completion")
    if isinstance(generation_completion, dict) and generation_completion.get("n_requested"):
        checks.append(
            PreviewGateCheck(
                name="generation_success_rate",
                observed=generation_completion["success_rate"],
                comparator=">=",
                threshold=thresholds.generation_success_rate_min,
                passed=(
                    generation_completion["success_rate"]
                    >= thresholds.generation_success_rate_min
                ),
                note="n_generated / n_requested",
            )
        )

    checks.extend(
        [
        PreviewGateCheck(
            name="schema_pass_rate",
            observed=post_report["schema_pass_rate"],
            comparator=">=",
            threshold=thresholds.schema_pass_rate_min,
            passed=post_report["schema_pass_rate"] >= thresholds.schema_pass_rate_min,
        ),
        PreviewGateCheck(
            name="validation_error_rate",
            observed=validation_error_rate,
            comparator="<=",
            threshold=thresholds.validation_error_rate_max,
            passed=validation_error_rate <= thresholds.validation_error_rate_max,
        ),
        PreviewGateCheck(
            name="validation_warning_rate",
            observed=validation_warning_rate,
            comparator="<",
            threshold=thresholds.validation_warning_rate_max,
            passed=validation_warning_rate < thresholds.validation_warning_rate_max,
        ),
        PreviewGateCheck(
            name="exact_duplicate_narrative_rows",
            observed=extra_checks["exact_duplicate_narrative_rows"],
            comparator="<=",
            threshold=thresholds.exact_duplicate_narrative_rows_max,
            passed=(
                extra_checks["exact_duplicate_narrative_rows"]
                <= thresholds.exact_duplicate_narrative_rows_max
            ),
        ),
        PreviewGateCheck(
            name="hobby_set_duplicate_rate",
            observed=extra_checks["hobby_set_duplicates"]["duplicate_rate"],
            comparator="<",
            threshold=thresholds.hobby_set_duplicate_rate_max,
            passed=(
                extra_checks["hobby_set_duplicates"]["duplicate_rate"]
                < thresholds.hobby_set_duplicate_rate_max
            ),
        ),
        PreviewGateCheck(
            name="null_only_exposed_columns",
            observed=extra_checks["exposed_column_fill"]["zero_fill_count"],
            comparator="<=",
            threshold=thresholds.null_only_exposed_columns_max,
            passed=(
                extra_checks["exposed_column_fill"]["zero_fill_count"]
                <= thresholds.null_only_exposed_columns_max
            ),
        ),
        PreviewGateCheck(
            name="district_placeholder_rate",
            observed=extra_checks["district_placeholder"]["placeholder_rate"],
            comparator="<=",
            threshold=thresholds.district_placeholder_rate_max,
            passed=(
                extra_checks["district_placeholder"]["placeholder_rate"]
                <= thresholds.district_placeholder_rate_max
            ),
        ),
        PreviewGateCheck(
            name="hobby_atom_top1_share",
            observed=extra_checks["hobby_atom_frequency"]["top1_share"],
            comparator="<=",
            threshold=thresholds.hobby_atom_top1_share_max,
            passed=(
                extra_checks["hobby_atom_frequency"]["top1_share"]
                <= thresholds.hobby_atom_top1_share_max
            ),
        ),
        ]
    )
    if thresholds.enforce_field_marginal_gate:
        checks.append(
            PreviewGateCheck(
                name="field_marginal_pass",
                observed=post_report["field_marginal"]["passed"],
                comparator="==",
                threshold=thresholds.require_field_marginal_pass,
                passed=(post_report["field_marginal"]["passed"] == thresholds.require_field_marginal_pass),
            )
        )
    if thresholds.enforce_joint_gate:
        checks.append(
            PreviewGateCheck(
                name="joint_field_sex_age_pass",
                observed=post_report["joint_field_sex_age"]["passed"],
                comparator="==",
                threshold=thresholds.require_joint_pass,
                passed=(post_report["joint_field_sex_age"]["passed"] == thresholds.require_joint_pass),
            )
        )
    if thresholds.enforce_fact_check_gate:
        checks.append(
            PreviewGateCheck(
                name="fact_checks_passed",
                observed={"passed": fact_passed, "total": fact_total},
                comparator="all" if thresholds.require_all_fact_checks_pass else "any",
                threshold={"passed": fact_total, "total": fact_total},
                passed=(fact_passed == fact_total) if thresholds.require_all_fact_checks_pass else True,
                note="whether all report-based fact checks pass",
            )
        )
    return PreviewGateResult(
        passed=all(check.passed for check in checks),
        checks=checks,
    )


def build_preview_report(
    parquet_path: Path,
    *,
    thresholds: PreviewThresholds | None = None,
) -> tuple[dict[str, Any], PreviewGateResult]:
    output_dir = parquet_path.parent
    df = pd.read_parquet(parquet_path)
    post = report_to_dict(run_post_verification(parquet_path))
    generation_metadata = _load_generation_metadata(output_dir)
    gate_profile = infer_preview_gate_profile(generation_metadata)
    thresholds = thresholds or default_preview_thresholds(gate_profile)
    extra_checks = {
        "generation_completion": {
            "n_requested": int(generation_metadata.get("n_requested", len(df)) or 0),
            "n_generated": int(generation_metadata.get("n_generated", len(df)) or 0),
            "n_failed": int(generation_metadata.get("n_failed", 0) or 0),
            "success_rate": float(
                int(generation_metadata.get("n_generated", len(df)) or 0)
                / max(int(generation_metadata.get("n_requested", len(df)) or 0), 1)
            ),
        },
        "exact_duplicate_narrative_rows": exact_duplicate_narrative_rows(df),
        "hobby_set_duplicates": hobby_set_duplicate_stats(df),
        "hobby_atom_frequency": hobby_atom_frequency_stats(df),
        "district_placeholder": district_placeholder_stats(df),
        "exposed_column_fill": exposed_column_fill_stats(df),
        "professional_persona_diversity": professional_persona_diversity(df),
    }
    gate = evaluate_preview_gate(
        post_report=post,
        extra_checks=extra_checks,
        thresholds=thresholds,
    )

    report = {
        "preview_generated_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        "parquet_path": str(parquet_path),
        "generation_metadata": generation_metadata,
        "gate_profile": gate_profile,
        "dataset_config": generation_metadata.get(
            "dataset_config",
            {
                "name": get_default_pak_core_dataset_config().config.name,
                **get_default_pak_core_dataset_config().config.fingerprint(),
            },
        ),
        "thresholds": asdict(thresholds),
        "post_verification": post,
        "extra_checks": extra_checks,
        "gate": {
            "passed": gate.passed,
            "checks": [asdict(check) for check in gate.checks],
        },
    }
    return report, gate


def preview_report_to_markdown(report: dict[str, Any]) -> str:
    post = report["post_verification"]
    gate = report["gate"]
    dataset_cfg = report.get("dataset_config", {})
    hobby_dups = report["extra_checks"]["hobby_set_duplicates"]
    hobby_atoms = report["extra_checks"]["hobby_atom_frequency"]
    district_placeholders = report["extra_checks"]["district_placeholder"]
    exposed_fill = report["extra_checks"]["exposed_column_fill"]
    lines = [
        "# PAK Preview Report",
        "",
        f"- generated_at: `{report['preview_generated_at']}`",
        f"- parquet: `{report['parquet_path']}`",
        f"- gate_profile: `{report.get('gate_profile', 'release_preview')}`",
        f"- dataset_config_hash: `{dataset_cfg.get('config_hash', 'unknown')}`",
        f"- gate_passed: `{'yes' if gate['passed'] else 'no'}`",
        "",
        "## Gate Checks",
        "",
        "| check | observed | comparator | threshold | passed |",
        "|---|---:|---|---:|---|",
    ]
    for check in gate["checks"]:
        lines.append(
            f"| {check['name']} | `{check['observed']}` | {check['comparator']} | "
            f"`{check['threshold']}` | {'✅' if check['passed'] else '❌'} |"
        )

    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- n_personas: `{post['n_personas']}`",
            f"- schema_pass_rate: `{post['schema_pass_rate']:.3f}`",
            f"- validation_error_rows: `{post['row_validation']['n_with_errors']}`",
            f"- validation_warning_rows: `{post['row_validation']['n_with_warnings']}`",
            f"- exact_duplicate_narrative_rows: `{report['extra_checks']['exact_duplicate_narrative_rows']}`",
            f"- hobby_set_duplicate_rate: `{hobby_dups['duplicate_rate']:.3%}`",
            f"- hobby_atom_top1_share: `{hobby_atoms['top1_share']:.3%}`",
            f"- district_placeholder_rate: `{district_placeholders['placeholder_rate']:.3%}`",
            f"- null_only_exposed_columns: `{exposed_fill['zero_fill_count']}`",
            f"- ignored_unsupported_nullable_columns: `{exposed_fill['ignored_unsupported_count']}`",
            f"- field_marginal_passed: `{post['field_marginal']['passed']}`",
            f"- joint_field_sex_age_passed: `{post['joint_field_sex_age']['passed']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def write_preview_artifacts(
    *,
    parquet_path: Path,
    report: dict[str, Any],
) -> PreviewArtifacts:
    output_dir = parquet_path.parent
    post_verification_path = output_dir / "post_verification.json"
    preview_report_json_path = output_dir / "preview_report.json"
    preview_report_md_path = output_dir / "preview_report.md"

    post_verification_path.write_text(
        json.dumps(report["post_verification"], ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    preview_report_json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    preview_report_md_path.write_text(
        preview_report_to_markdown(report),
        encoding="utf-8",
    )
    return PreviewArtifacts(
        parquet_path=parquet_path,
        post_verification_path=post_verification_path,
        preview_report_json_path=preview_report_json_path,
        preview_report_md_path=preview_report_md_path,
    )


def run_preview(
    cfg: GenerateConfig,
    *,
    thresholds: PreviewThresholds | None = None,
) -> PreviewRunResult:
    parquet_path = run_generation(cfg)
    report, gate = build_preview_report(parquet_path, thresholds=thresholds)
    artifacts = write_preview_artifacts(parquet_path=parquet_path, report=report)
    logger.info(
        "preview gate %s for %s",
        "passed" if gate.passed else "failed",
        parquet_path,
    )
    return PreviewRunResult(artifacts=artifacts, gate=gate, report=report)


def assert_create_ready(preview_report_path: Path) -> dict[str, Any]:
    report = json.loads(preview_report_path.read_text(encoding="utf-8"))
    if not report.get("gate", {}).get("passed", False):
        raise RuntimeError(f"preview gate failed: {preview_report_path}")

    preview_hash = report.get("dataset_config", {}).get("config_hash")
    current_hash = get_default_pak_core_dataset_config().config.fingerprint()["config_hash"]
    if preview_hash != current_hash:
        raise RuntimeError(
            "preview config hash does not match current dataset config: "
            f"{preview_hash!r} != {current_hash!r}"
        )
    return report


def main() -> None:  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260502)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--quant-rows-path", type=Path, default=None)
    ap.add_argument("--checkpoint-every", type=int, default=10)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    cfg = GenerateConfig(
        n=args.n,
        seed=args.seed,
        output_dir=args.output_dir,
        quant_rows_path=args.quant_rows_path,
        checkpoint_every=args.checkpoint_every,
    )
    result = run_preview(cfg)
    print(f"wrote {result.artifacts.preview_report_json_path}")
    print(json.dumps({
        "gate_passed": result.gate.passed,
        "parquet_path": str(result.artifacts.parquet_path),
        "preview_report_json_path": str(result.artifacts.preview_report_json_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
