"""Build or refresh the release-ready PAK-1K-eval judge bundle.

The clean release tree keeps canonical judge scores under
``outputs/pak_1k_eval_release/judges/<annotator>/scores.json`` and omits raw
batch prompts, retry files, and runner-local artifacts. This script refreshes
the tidy csv and per-judge JSONL metadata from those canonical files.

Outputs:
  - outputs/pak_1k_eval_release/annotations.csv
  - outputs/pak_1k_eval_release/judges/{claude,gemini,clova,codex}/...
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
RELEASE_DIR = OUTPUTS / "pak_1k_eval_release"
JUDGES_DIR = RELEASE_DIR / "judges"
HUMAN_CSV = ROOT / "data/eval/human_scores_260520.csv"
HUMAN_AMENDMENTS_CSV = ROOT / "data/eval/human_scores_260520_amendments_260521.csv"
PERSONA_SAMPLE_JSON = ROOT / "web/public/personas_review_sample.json"
CALIBRATION_ANCHORS_JSON = ROOT / "web/public/personas_anchors.json"
MAIN_LICENSE = ROOT / "data/release/pak_v0_1/LICENSE"
PREREGISTRATION_MD = ROOT / "outputs/reports/pak_1k_eval_preregistration.md"
MANIFEST_NAME = "MANIFEST.sha256"

DIMENSIONS: tuple[str, ...] = (
    "groundedness",
    "coherence",
    "plausibility",
    "fluency",
)
OVERALL_DIMENSION = "_overall"
EXPECTED_N = 1000


@dataclass(frozen=True)
class JudgeSpec:
    annotator: str
    display_name: str
    provider: str
    model: str
    api: str
    source_path: Path
    role: str = ""
    allow_null: bool = False


JUDGES: tuple[JudgeSpec, ...] = (
    JudgeSpec(
        annotator="claude",
        display_name="Claude Opus 4.7",
        provider="Anthropic",
        model="Claude Opus 4.7",
        api="Anthropic Claude API",
        source_path=RELEASE_DIR / "judges/claude/scores.json",
    ),
    JudgeSpec(
        annotator="gemini",
        display_name="Gemini 2.5 Pro",
        provider="Google",
        model="Gemini 2.5 Pro",
        api="Google Gemini API",
        source_path=RELEASE_DIR / "judges/gemini/scores.json",
    ),
    JudgeSpec(
        annotator="clova",
        display_name="HyperCLOVA X HCX-007",
        provider="NAVER Cloud",
        model="HyperCLOVA X HCX-007",
        api="NAVER Cloud CLOVA Studio Chat Completions v3",
        source_path=RELEASE_DIR / "judges/clova/scores.json",
    ),
    JudgeSpec(
        annotator="codex",
        display_name="Codex 5.5",
        provider="OpenAI",
        model="Codex 5.5",
        api="Codex CLI",
        source_path=RELEASE_DIR / "judges/codex/scores.json",
    ),
)

# Separate self-preference probe (generator family). Reported against the
# panel, NOT a panel member. Scores may be null (parse failures), so it is
# written to its own judge directory but kept OUT of annotations.csv to
# keep the panel agreement scripts on strictly 1-5 integer data.
QWEN3_PROBE = JudgeSpec(
    annotator="qwen3",
    display_name="qwen3 30B (generator self-preference probe)",
    provider="Alibaba",
    model="qwen3:30b-a3b",
    api="Ollama native (local)",
    source_path=RELEASE_DIR / "judges/qwen3/scores.json",
    role=(
        "Generator-family self-preference probe. Reported against the "
        "Claude-Gemini panel, not a panel member. Some dimension scores are "
        "null from parse failures."
    ),
    allow_null=True,
)

OUTPUT_COLUMNS = ["pak_uuid", "annotator", "dimension", "score", "reasoning", "flag"]
CONTEXT_FRONT_COLUMNS = ["pak_uuid", "order_index", "batch_id", "art_field_primary"]


@dataclass(frozen=True)
class HumanAmendment:
    pak_uuid: str
    order_index: int
    dimension: str
    old_score: int
    new_score: int
    decision_source: str
    rationale: str


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest() -> None:
    manifest_path = RELEASE_DIR / MANIFEST_NAME
    if manifest_path.exists():
        manifest_path.unlink()
    lines: list[str] = []
    for path in sorted(RELEASE_DIR.rglob("*")):
        if path.is_dir() or path.name == MANIFEST_NAME:
            continue
        rel = path.relative_to(RELEASE_DIR).as_posix()
        lines.append(f"{sha256_file(path)}  {rel}")
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_human_batch_map() -> pd.DataFrame:
    human = pd.read_csv(HUMAN_CSV)
    return human[["pak_uuid", "order_index", "batch_id"]].copy()


def order_context_columns(df: pd.DataFrame) -> pd.DataFrame:
    front = [column for column in CONTEXT_FRONT_COLUMNS if column in df.columns]
    rest = [column for column in df.columns if column not in front]
    return df[front + rest]


def load_review_sample() -> pd.DataFrame:
    release_sample = RELEASE_DIR / "sample_personas.csv"
    if PERSONA_SAMPLE_JSON.exists():
        return pd.DataFrame(json.loads(PERSONA_SAMPLE_JSON.read_text(encoding="utf-8")))
    if release_sample.exists():
        log.info(
            "Using existing %s because %s is absent",
            release_sample.relative_to(ROOT),
            PERSONA_SAMPLE_JSON.relative_to(ROOT),
        )
        return pd.read_csv(release_sample)
    raise FileNotFoundError(
        f"Missing {PERSONA_SAMPLE_JSON.relative_to(ROOT)} and {release_sample.relative_to(ROOT)}"
    )


def add_human_batch_context(sample: pd.DataFrame) -> pd.DataFrame:
    if {"order_index", "batch_id"}.issubset(sample.columns) and not sample[
        ["order_index", "batch_id"]
    ].isna().any().any():
        return sample
    sample = sample.drop(columns=["order_index", "batch_id"], errors="ignore")
    return sample.merge(read_human_batch_map(), on="pak_uuid", how="left")


def load_calibration_anchors() -> pd.DataFrame | None:
    release_anchors = RELEASE_DIR / "calibration_anchors.csv"
    if CALIBRATION_ANCHORS_JSON.exists():
        return pd.DataFrame(json.loads(CALIBRATION_ANCHORS_JSON.read_text(encoding="utf-8")))
    if release_anchors.exists():
        log.info(
            "Using existing %s because %s is absent",
            release_anchors.relative_to(ROOT),
            CALIBRATION_ANCHORS_JSON.relative_to(ROOT),
        )
        return pd.read_csv(release_anchors)
    return None


def write_context_tables(reference_uuids: set[str]) -> None:
    sample = load_review_sample()
    sample_uuids = set(sample["pak_uuid"])
    if sample_uuids != reference_uuids:
        raise ValueError(
            "Review sample UUID mismatch: "
            f"extra={len(sample_uuids - reference_uuids)}, missing={len(reference_uuids - sample_uuids)}"
        )
    sample = order_context_columns(add_human_batch_context(sample))
    if sample[["order_index", "batch_id"]].isna().any().any():
        raise ValueError("Review sample is missing order_index or batch_id after human CSV merge")
    sample.to_csv(RELEASE_DIR / "sample_personas.csv", index=False)

    anchors = load_calibration_anchors()
    if anchors is not None:
        anchors = order_context_columns(anchors)
        anchors.to_csv(RELEASE_DIR / "calibration_anchors.csv", index=False)

    if MAIN_LICENSE.exists():
        shutil.copy2(MAIN_LICENSE, RELEASE_DIR / "LICENSE")
    if PREREGISTRATION_MD.exists():
        shutil.copy2(PREREGISTRATION_MD, RELEASE_DIR / "preregistration.md")


def write_schema() -> None:
    schema = {
        "tables": {
            "annotations": {
                "path": "annotations.csv",
                "columns": {
                    "pak_uuid": "string persona identifier",
                    "annotator": "human_anonymous, claude, gemini, clova, or codex",
                    "dimension": "groundedness, coherence, plausibility, fluency, or _overall",
                    "score": "nullable integer score, 1-5 for scoring dimensions",
                    "reasoning": "judge reasoning text; empty for human rows",
                    "flag": "row-level flag text on _overall rows",
                },
            },
            "sample_personas": {
                "path": "sample_personas.csv",
                "description": "The 1,000 stratified PAK personas used for PAK-1K-eval.",
            },
            "calibration_anchors": {
                "path": "calibration_anchors.csv",
                "description": "The 10 calibration anchors used to lock the rubric.",
            },
        },
        "score_scale": {
            "min": 1,
            "max": 5,
            "dimensions": list(DIMENSIONS),
            "overall_dimension": OVERALL_DIMENSION,
        },
        "annotators": {
            "human_anonymous": "Public release label for the single human reference annotation.",
            **{spec.annotator: spec.display_name for spec in JUDGES},
        },
        "probes": {
            QWEN3_PROBE.annotator: (
                f"{QWEN3_PROBE.display_name}. {QWEN3_PROBE.role} "
                "Stored under judges/qwen3/ and not included in annotations.csv."
            ),
        },
    }
    write_json(RELEASE_DIR / "schema.json", schema)


def write_data_card(generated_at: str, human_amendment_count: int) -> None:
    text = f"""# PAK-1K-eval Data Card

Generated: {generated_at}

## Scope

PAK-1K-eval is a 1,000-persona scoring layer for the Persona Arts Korea
release. It contains one human reference annotation and four LLM judge
runs over four dimensions: groundedness, coherence, plausibility, and Korean
fluency. It also ships a separate generator-family self-preference probe
(`judges/qwen3/`) that is reported against the panel and is not part of the
panel `annotations.csv`.

## Self-preference probe

`judges/qwen3/` holds scores from qwen3:30b-a3b, the model that generated the
PAK narratives, run as a judge under the same per-dimension protocol as the
panel. Because it shares the generator family it measures self-preference
rather than serving as an independent judge, so it is kept out of
`annotations.csv`. Some dimension scores are null where the judge output
failed to parse. See `judges/qwen3/metadata.json` for per-dimension null counts
and `outputs/reports/qwen3_self_preference_260523.md` for the analysis.

## Human Reference

The public label `human_anonymous` denotes the single human reference used in
the paper's agreement analysis. It is anonymized only at the schema level and
does not denote an independently recruited external rater. The release applies `{HUMAN_AMENDMENTS_CSV.relative_to(ROOT)}` as a
transparent amendment overlay ({human_amendment_count} score corrections);
the original human CSV is retained outside this bundle for audit.

## Score Scale

Scores are integers from 1 to 5. The rubric anchors levels 1, 3, and 5 for
each dimension. Row-level flags are stored once using `dimension="_overall"`
instead of being duplicated across dimensions.

## Sample and Reproducibility

`sample_personas.csv` stores the 1,000 stratified personas with
`pak_uuid`, `order_index`, `batch_id`, `art_field_primary`, all quantitative
anchors, and narrative fields. `calibration_anchors.csv` stores the 10
rubric calibration personas. `preregistration.md` records the current
PAK-1K-eval plan, including the 2026-05-21 correction that no same-rater
retest is part of the canonical design.

## License

Released under CC-BY-4.0, matching the main PAK release.

## Limitations

This is a seed human-reference audit plus model-panel agreement layer. It is not
inter-annotator agreement, and it should not be treated as final ground truth.
Additional independent human annotations are invited against the same 1,000
personas.
"""
    (RELEASE_DIR / "DATA_CARD.md").write_text(text, encoding="utf-8")


def load_human_amendments() -> dict[tuple[str, str], HumanAmendment]:
    if not HUMAN_AMENDMENTS_CSV.exists():
        return {}

    amendments: dict[tuple[str, str], HumanAmendment] = {}
    with HUMAN_AMENDMENTS_CSV.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            pak_uuid = clean_text(row["pak_uuid"])
            dimension = clean_text(row["dimension"])
            if dimension not in DIMENSIONS:
                raise ValueError(f"Invalid amendment dimension {dimension!r} for {pak_uuid}")
            amendment = HumanAmendment(
                pak_uuid=pak_uuid,
                order_index=int(row["order_index"]),
                dimension=dimension,
                old_score=int(row["old_score"]),
                new_score=int(row["new_score"]),
                decision_source=clean_text(row.get("decision_source")),
                rationale=clean_text(row.get("rationale")),
            )
            if amendment.new_score < 1 or amendment.new_score > 5:
                raise ValueError(f"Invalid amendment score {amendment.new_score} for {pak_uuid}")
            key = (pak_uuid, dimension)
            if key in amendments:
                raise ValueError(f"Duplicate human amendment for {pak_uuid} {dimension}")
            amendments[key] = amendment
    return amendments


def load_human_rows() -> tuple[list[dict[str, Any]], set[str], int]:
    rows: list[dict[str, Any]] = []
    uuids: set[str] = set()
    amendments = load_human_amendments()
    applied_amendments: set[tuple[str, str]] = set()

    with HUMAN_CSV.open(encoding="utf-8", newline="") as fh:
        for source_row in csv.DictReader(fh):
            pak_uuid = clean_text(source_row["pak_uuid"])
            uuids.add(pak_uuid)
            for dimension in DIMENSIONS:
                score = int(source_row[dimension])
                key = (pak_uuid, dimension)
                amendment = amendments.get(key)
                if amendment is not None:
                    if score != amendment.old_score:
                        raise ValueError(
                            f"Human amendment old_score mismatch for {pak_uuid} {dimension}: "
                            f"CSV has {score}, amendment expected {amendment.old_score}"
                        )
                    score = amendment.new_score
                    applied_amendments.add(key)
                rows.append(
                    {
                        "pak_uuid": pak_uuid,
                        "annotator": "human_anonymous",
                        "dimension": dimension,
                        "score": score,
                        "reasoning": "",
                        "flag": "",
                    }
                )
            rows.append(
                {
                    "pak_uuid": pak_uuid,
                    "annotator": "human_anonymous",
                    "dimension": OVERALL_DIMENSION,
                    "score": None,
                    "reasoning": "",
                    "flag": clean_text(source_row.get("flags")),
                }
            )

    unapplied = set(amendments) - applied_amendments
    if unapplied:
        raise ValueError(f"Human amendments not applied: {sorted(unapplied)}")
    if amendments:
        log.info(
            "Applied %d human annotation amendments from %s",
            len(amendments),
            HUMAN_AMENDMENTS_CSV.relative_to(ROOT),
        )
    return rows, uuids, len(amendments)


def validate_entry(spec: JudgeSpec, pak_uuid: str, entry: dict[str, Any]) -> None:
    scores = entry.get("scores")
    reasoning = entry.get("reasoning")
    if not isinstance(scores, dict):
        raise ValueError(f"{spec.annotator} {pak_uuid}: scores is not an object")
    if not isinstance(reasoning, dict):
        raise ValueError(f"{spec.annotator} {pak_uuid}: reasoning is not an object")

    missing_scores = [dimension for dimension in DIMENSIONS if dimension not in scores]
    missing_reasoning = [dimension for dimension in DIMENSIONS if dimension not in reasoning]
    if missing_scores:
        raise ValueError(f"{spec.annotator} {pak_uuid}: missing scores {missing_scores}")
    if missing_reasoning:
        raise ValueError(f"{spec.annotator} {pak_uuid}: missing reasoning {missing_reasoning}")

    for dimension in DIMENSIONS:
        score = int(scores[dimension])
        if score < 1 or score > 5:
            raise ValueError(f"{spec.annotator} {pak_uuid} {dimension}: score {score} outside 1-5")


def build_judge_release(spec: JudgeSpec, reference_uuids: set[str]) -> list[dict[str, Any]]:
    log.info("Preparing %s from %s", spec.annotator, spec.source_path.relative_to(ROOT))
    raw = json.loads(spec.source_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{spec.source_path} must contain a dict keyed by pak_uuid")
    if len(raw) != EXPECTED_N:
        raise ValueError(f"{spec.annotator}: expected {EXPECTED_N} personas, got {len(raw)}")

    raw_uuids = set(raw)
    extra = raw_uuids - reference_uuids
    missing = reference_uuids - raw_uuids
    if extra or missing:
        raise ValueError(
            f"{spec.annotator}: UUID mismatch, extra={len(extra)}, missing={len(missing)}"
        )

    judge_dir = JUDGES_DIR / spec.annotator
    judge_dir.mkdir(parents=True, exist_ok=True)

    score_rows: list[dict[str, Any]] = []
    flag_rows: list[dict[str, Any]] = []
    global_rows: list[dict[str, Any]] = []

    for pak_uuid in sorted(raw):
        entry = raw[pak_uuid]
        if not isinstance(entry, dict):
            raise ValueError(f"{spec.annotator} {pak_uuid}: entry is not an object")
        validate_entry(spec, pak_uuid, entry)

        scores = entry["scores"]
        reasoning = entry["reasoning"]
        for dimension in DIMENSIONS:
            score_row = {
                "pak_uuid": pak_uuid,
                "annotator": spec.annotator,
                "model": spec.model,
                "dimension": dimension,
                "score": int(scores[dimension]),
                "reasoning": clean_text(reasoning.get(dimension)),
            }
            score_rows.append(score_row)
            global_rows.append(
                {
                    "pak_uuid": pak_uuid,
                    "annotator": spec.annotator,
                    "dimension": dimension,
                    "score": int(scores[dimension]),
                    "reasoning": score_row["reasoning"],
                    "flag": "",
                }
            )

        flag = clean_text(entry.get("flag"))
        flag_rows.append(
            {
                "pak_uuid": pak_uuid,
                "annotator": spec.annotator,
                "model": spec.model,
                "flag": flag,
            }
        )
        global_rows.append(
            {
                "pak_uuid": pak_uuid,
                "annotator": spec.annotator,
                "dimension": OVERALL_DIMENSION,
                "score": None,
                "reasoning": "",
                "flag": flag,
            }
        )

    metadata = {
        "annotator": spec.annotator,
        "display_name": spec.display_name,
        "provider": spec.provider,
        "model": spec.model,
        "api": spec.api,
        "source_path": str(spec.source_path.relative_to(RELEASE_DIR)),
        "source_note": "Clean release package. Original raw run folder removed after consolidation.",
        "n_personas": len(raw),
        "dimensions": list(DIMENSIONS),
        "files": {
            "scores_wide": "scores.json",
            "scores_long": "scores_long.jsonl",
            "overall_flags": "overall_flags.jsonl",
        },
    }

    write_json(judge_dir / "metadata.json", metadata)
    write_json(judge_dir / "scores.json", raw)
    write_jsonl(judge_dir / "scores_long.jsonl", score_rows)
    write_jsonl(judge_dir / "overall_flags.jsonl", flag_rows)
    return global_rows


def build_probe_release(spec: JudgeSpec, reference_uuids: set[str]) -> dict[str, int]:
    """Write a separate probe judge directory (generator self-preference probe).

    Unlike build_judge_release, scores may be null and the rows are NOT added to
    the panel annotations table, so downstream agreement scripts keep operating
    on strictly 1-5 integer data for the four panel judges.
    """
    log.info("Preparing probe %s from %s", spec.annotator, spec.source_path.relative_to(ROOT))
    raw = json.loads(spec.source_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{spec.source_path} must contain a dict keyed by pak_uuid")
    if len(raw) != EXPECTED_N:
        raise ValueError(f"{spec.annotator}: expected {EXPECTED_N} personas, got {len(raw)}")
    if set(raw) - reference_uuids:
        raise ValueError(f"{spec.annotator}: UUIDs outside the reference set")

    judge_dir = JUDGES_DIR / spec.annotator
    judge_dir.mkdir(parents=True, exist_ok=True)
    score_rows: list[dict[str, Any]] = []
    flag_rows: list[dict[str, Any]] = []
    null_counts = {dimension: 0 for dimension in DIMENSIONS}

    for pak_uuid in sorted(raw):
        entry = raw[pak_uuid]
        scores = entry.get("scores", {}) or {}
        reasoning = entry.get("reasoning", {}) or {}
        for dimension in DIMENSIONS:
            value = scores.get(dimension)
            if value is None:
                null_counts[dimension] += 1
            else:
                value = int(value)
                if value < 1 or value > 5:
                    raise ValueError(f"{spec.annotator} {pak_uuid} {dimension}: {value} outside 1-5")
            score_rows.append(
                {
                    "pak_uuid": pak_uuid,
                    "annotator": spec.annotator,
                    "model": spec.model,
                    "dimension": dimension,
                    "score": value,
                    "reasoning": clean_text(reasoning.get(dimension)),
                }
            )
        flag_rows.append(
            {
                "pak_uuid": pak_uuid,
                "annotator": spec.annotator,
                "model": spec.model,
                "flag": clean_text(entry.get("flag")),
            }
        )

    metadata = {
        "annotator": spec.annotator,
        "display_name": spec.display_name,
        "provider": spec.provider,
        "model": spec.model,
        "api": spec.api,
        "role": spec.role,
        "panel_member": False,
        "source_path": str(spec.source_path.relative_to(RELEASE_DIR)),
        "n_personas": len(raw),
        "dimensions": list(DIMENSIONS),
        "null_scores": null_counts,
        "files": {
            "scores_wide": "scores.json",
            "scores_long": "scores_long.jsonl",
            "overall_flags": "overall_flags.jsonl",
        },
    }
    write_json(judge_dir / "metadata.json", metadata)
    write_json(judge_dir / "scores.json", raw)
    write_jsonl(judge_dir / "scores_long.jsonl", score_rows)
    write_jsonl(judge_dir / "overall_flags.jsonl", flag_rows)
    log.info("Probe %s null scores per dimension: %s", spec.annotator, null_counts)
    return null_counts


def write_readme(generated_at: str, human_amendment_count: int) -> None:
    judge_lines = "\n".join(
        f"- `{spec.annotator}`: {spec.display_name} ({spec.api})" for spec in JUDGES
    )
    amendment_note = ""
    if human_amendment_count:
        amendment_note = (
            "\nHuman reference scores apply the amendment overlay at "
            f"`{HUMAN_AMENDMENTS_CSV.relative_to(ROOT)}` "
            f"({human_amendment_count} score corrections). The original human CSV is retained "
            "unchanged for audit.\n"
        )
    text = f"""# PAK-1K-eval Judge Release Bundle

Generated by `scripts/prepare_eval_judge_release.py` at {generated_at}.

This directory is the release-facing surface for the four model judges and the
human reference used in the PAK-1K-eval agreement analysis.
{amendment_note}

The annotator label `human_anonymous` is the public release label for the
single human reference used in the paper's agreement analysis. It is anonymized
only at the schema level so that downstream code treats all annotators
uniformly; it does not denote a separate external rater.

## Judges

{judge_lines}

The internal label `clova` refers to HyperCLOVA X HCX-007 via NAVER Cloud CLOVA
Studio Chat Completions v3. It is kept as `clova` for compatibility with earlier
analysis files.

## Self-preference probe

`judges/qwen3/` holds the generator-family self-preference probe (qwen3:30b-a3b,
the model that generated the narratives). It is scored under the same
per-dimension protocol as the panel but is reported against the panel rather
than joining it, so it is not part of `annotations.csv`. Some dimension
scores are null where the judge output failed to parse; see
`judges/qwen3/metadata.json` for per-dimension null counts.

## Files

- `annotations.csv`: tidy long table with columns
  `pak_uuid`, `annotator`, `dimension`, `score`, `reasoning`, `flag`.
- `dimension="_overall"` rows preserve row-level flags without duplicating them
  across the four scoring dimensions.
- `sample_personas.csv`: the 1,000 stratified
  personas used for the audit, with `order_index`, `batch_id`, art field,
  quantitative anchors, and narrative fields.
- `calibration_anchors.csv`: the 10 rubric
  calibration anchors.
- `DATA_CARD.md`, `schema.json`, `preregistration.md`, `LICENSE`, and
  `MANIFEST.sha256`: release documentation, schema, plan record, license, and
  file hashes.
- `judges/<annotator>/scores.json`: canonical wide judge output keyed by
  `pak_uuid`.
- `judges/<annotator>/scores_long.jsonl`: one score row per persona and
  dimension.
- `judges/<annotator>/overall_flags.jsonl`: one row-level flag row per persona.

This clean release package intentionally omits raw batch prompts, retry files,
and runner-local artifacts. The files in this directory are the canonical
release interface for downstream analysis.
"""
    (RELEASE_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    generated_at = datetime.now().isoformat(timespec="seconds")
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    JUDGES_DIR.mkdir(parents=True, exist_ok=True)

    rows, reference_uuids, human_amendment_count = load_human_rows()
    if len(reference_uuids) != EXPECTED_N:
        raise ValueError(f"human reference: expected {EXPECTED_N}, got {len(reference_uuids)}")

    for spec in JUDGES:
        rows.extend(build_judge_release(spec, reference_uuids))
    build_probe_release(QWEN3_PROBE, reference_uuids)
    write_context_tables(reference_uuids)

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    df["score"] = pd.array(df["score"], dtype="Int64")
    for column in ["pak_uuid", "annotator", "dimension", "reasoning", "flag"]:
        df[column] = df[column].fillna("").astype("string")

    df.to_csv(RELEASE_DIR / "annotations.csv", index=False)

    metadata = {
        "generated_at": generated_at,
        "n_reference_personas": len(reference_uuids),
        "human_source_path": str(HUMAN_CSV.relative_to(ROOT)),
        "human_reference_note": (
            "`human_anonymous` is the release label for the single human reference annotation."
        ),
        "human_amendment_path": (
            str(HUMAN_AMENDMENTS_CSV.relative_to(ROOT))
            if human_amendment_count
            else None
        ),
        "human_amendment_count": human_amendment_count,
        "sample_personas_path": "sample_personas.csv",
        "calibration_anchors_path": "calibration_anchors.csv",
        "schema_path": "schema.json",
        "data_card_path": "DATA_CARD.md",
        "manifest_path": MANIFEST_NAME,
        "license": "CC-BY-4.0",
        "score_scale": {"min": 1, "max": 5, "dimensions": list(DIMENSIONS)},
        "preregistration": {
            "path": "preregistration.md",
            "canonical_note": "No same-rater retest is part of the current PAK-1K-eval plan.",
            "latest_correction_date": "2026-05-21",
        },
        "dimensions": list(DIMENSIONS),
        "annotators": ["human_anonymous", *[spec.annotator for spec in JUDGES]],
        "judge_metadata": {
            spec.annotator: {
                "display_name": spec.display_name,
                "provider": spec.provider,
                "model": spec.model,
                "api": spec.api,
            }
            for spec in JUDGES
        },
        "probes": {
            QWEN3_PROBE.annotator: {
                "display_name": QWEN3_PROBE.display_name,
                "provider": QWEN3_PROBE.provider,
                "model": QWEN3_PROBE.model,
                "api": QWEN3_PROBE.api,
                "role": QWEN3_PROBE.role,
                "panel_member": False,
                "in_annotations_csv": False,
            },
        },
    }
    write_json(RELEASE_DIR / "metadata.json", metadata)
    write_schema()
    write_data_card(generated_at, human_amendment_count)
    write_readme(generated_at, human_amendment_count)
    write_manifest()

    coverage = (
        df[df["dimension"].isin(DIMENSIONS)]
        .groupby(["annotator", "dimension"])["pak_uuid"]
        .nunique()
        .unstack(fill_value=0)
    )
    log.info("Wrote release bundle: %s", RELEASE_DIR.relative_to(ROOT))
    log.info("Coverage:\n%s", coverage.to_string())


if __name__ == "__main__":
    main()
