"""Refresh the tidy PAK-1K-eval annotation table from the release bundle.

The raw judge folders were intentionally removed from the clean distribution.
The canonical annotation source is now
``outputs/pak_1k_eval_release/annotations.csv``.
"""

from __future__ import annotations

import logging
import json
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_EVAL = ROOT / "data" / "eval"
RELEASE_DIR = ROOT / "outputs" / "pak_1k_eval_release"
RELEASE_ANNOTATIONS = RELEASE_DIR / "annotations.csv"
RELEASE_METADATA = RELEASE_DIR / "metadata.json"

CSV_OUT = DATA_EVAL / "pak_1k_eval_annotations.csv"
README_OUT = DATA_EVAL / "pak_1k_eval_annotations.README.md"
REPORT_OUT = ROOT / "outputs" / "reports" / "annotation_consolidation_report.md"

DIMENSIONS: tuple[str, ...] = (
    "groundedness",
    "coherence",
    "plausibility",
    "fluency",
)
OVERALL_DIMENSION = "_overall"
OUTPUT_COLUMNS = ["pak_uuid", "annotator", "dimension", "score", "reasoning", "flag"]


def load_release_annotations() -> pd.DataFrame:
    if not RELEASE_ANNOTATIONS.exists():
        raise FileNotFoundError(
            f"Missing {RELEASE_ANNOTATIONS.relative_to(ROOT)}. "
            "Run scripts/prepare_eval_judge_release.py first."
        )

    df = pd.read_csv(RELEASE_ANNOTATIONS)
    missing = set(OUTPUT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"{RELEASE_ANNOTATIONS.relative_to(ROOT)} missing columns: {sorted(missing)}"
        )

    df = df[OUTPUT_COLUMNS].copy()
    df["score"] = pd.array(df["score"], dtype="Int64")
    for column in ["pak_uuid", "annotator", "dimension", "reasoning", "flag"]:
        df[column] = df[column].fillna("").astype("string")
    return df


def validate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows = df[df["dimension"].isin(DIMENSIONS)].copy()
    coverage = (
        score_rows.groupby(["annotator", "dimension"])["pak_uuid"]
        .nunique()
        .unstack(fill_value=0)
    )

    bad_scores = score_rows[~score_rows["score"].between(1, 5)]
    duplicate_scores = score_rows[
        score_rows.duplicated(["pak_uuid", "annotator", "dimension"], keep=False)
    ]
    if not bad_scores.empty:
        raise ValueError(f"Found {len(bad_scores)} scores outside 1-5")
    if not duplicate_scores.empty:
        raise ValueError(f"Found {len(duplicate_scores)} duplicated score rows")
    return coverage, df[df["dimension"].eq(OVERALL_DIMENSION)].copy()


def load_release_metadata() -> dict[str, object]:
    if not RELEASE_METADATA.exists():
        return {}
    return json.loads(RELEASE_METADATA.read_text(encoding="utf-8"))


def amendment_lines(metadata: dict[str, object]) -> list[str]:
    count = int(metadata.get("human_amendment_count") or 0)
    path = metadata.get("human_amendment_path")
    if not count or not path:
        return []
    return [
        "",
        f"Human scores apply `{path}` as an amendment overlay ({count} score corrections).",
        "The original human CSV is retained unchanged for audit.",
    ]


def write_readme() -> None:
    metadata = load_release_metadata()
    lines = [
        "# PAK-1K-eval Annotations",
        "",
        f"Generated from `{RELEASE_ANNOTATIONS.relative_to(ROOT)}`.",
        *amendment_lines(metadata),
        "",
        "Columns: `pak_uuid`, `annotator`, `dimension`, `score`, `reasoning`, `flag`.",
        "",
        "Scoring dimensions are `groundedness`, `coherence`, `plausibility`, and `fluency`.",
        f"`dimension=\"{OVERALL_DIMENSION}\"` rows store persona-level flags once rather than "
        "duplicating them across scoring dimensions.",
        "",
        "Annotator labels:",
        "- `human_anonymous`: public release label for the first-author expert annotation.",
        "- `claude`: Claude Opus 4.7.",
        "- `gemini`: Gemini 2.5 Pro.",
        "- `clova`: HyperCLOVA X HCX-007 via NAVER Cloud CLOVA Studio Chat Completions v3.",
        "- `codex`: Codex 5.5.",
        "",
    ]
    README_OUT.write_text("\n".join(lines), encoding="utf-8")


def write_report(df: pd.DataFrame, coverage: pd.DataFrame, overall_rows: pd.DataFrame) -> None:
    metadata = load_release_metadata()
    human_amendment_count = int(metadata.get("human_amendment_count") or 0)
    human_amendment_path = metadata.get("human_amendment_path")
    lines = [
        "# Annotation Consolidation Report",
        "",
        f"- Source: `{RELEASE_ANNOTATIONS.relative_to(ROOT)}`",
        f"- Output CSV: `{CSV_OUT.relative_to(ROOT)}`",
        f"- Rows: {len(df)}",
        f"- Unique personas: {df['pak_uuid'].nunique()}",
        "- Clean-release note: raw judge batch prompts, retry files, and runner-local "
        "artifacts were removed after consolidation.",
        "- Human reference annotator label: `human_anonymous` "
        "(public release label for first-author expert annotation).",
        f"- Human score amendments applied: {human_amendment_count}"
        + (f" (`{human_amendment_path}`)." if human_amendment_path else "."),
        f"- Row-level flags are stored once as `dimension=\"{OVERALL_DIMENSION}\"` rows.",
        "",
        "## Coverage",
        "",
        coverage.to_markdown(),
        "",
        "## Validation",
        "",
        "- Scores outside 1-5: 0",
        "- Duplicate `(pak_uuid, annotator, dimension)` score rows: 0",
        f"- Overall flag rows: {len(overall_rows)}",
        "",
    ]
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = load_release_annotations()
    coverage, overall_rows = validate(df)

    DATA_EVAL.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_OUT, index=False)
    write_readme()
    write_report(df, coverage, overall_rows)

    log.info("Wrote %s", CSV_OUT.relative_to(ROOT))
    log.info("Coverage:\n%s", coverage.to_string())


if __name__ == "__main__":
    main()
