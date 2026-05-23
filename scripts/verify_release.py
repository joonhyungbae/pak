"""Verify release-facing PAK artifacts.

The script is intentionally read-only. It validates checksum manifests, row
counts, score coverage, and score ranges for the main PAK release and the
PAK-1K-eval annotation bundle.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
MAIN_RELEASE = ROOT / "data/release/pak_v0_1"
EVAL_RELEASE = ROOT / "outputs/pak_1k_eval_release"
MAIN_MANIFEST = MAIN_RELEASE / "MANIFEST.sha256"
EVAL_MANIFEST = EVAL_RELEASE / "MANIFEST.sha256"
EXPECTED_PERSONAS = 30_000
EXPECTED_EVAL_PERSONAS = 1_000
EXPECTED_CALIBRATION_ANCHORS = 10
ANNOTATORS = ("human_anonymous", "claude", "gemini", "clova", "codex")
DIMENSIONS = ("groundedness", "coherence", "plausibility", "fluency")
OVERALL_DIMENSION = "_overall"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(manifest_path: Path, base_dir: Path) -> None:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path.relative_to(ROOT)}")

    checked = 0
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            expected_hash, rel_path = line.split(maxsplit=1)
        except ValueError as exc:
            raise ValueError(f"{manifest_path}:{line_number}: invalid manifest line") from exc
        path = base_dir / rel_path.strip()
        if not path.exists():
            raise FileNotFoundError(f"{manifest_path}:{line_number}: missing {rel_path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"{manifest_path}:{line_number}: hash mismatch for {rel_path}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        checked += 1
    log.info("Manifest OK: %s (%d files)", manifest_path.relative_to(ROOT), checked)


def verify_main_release() -> None:
    personas_path = MAIN_RELEASE / "personas.parquet"
    if not personas_path.exists():
        raise FileNotFoundError(f"Missing {personas_path.relative_to(ROOT)}")

    df = pd.read_parquet(personas_path)
    if len(df) != EXPECTED_PERSONAS:
        raise ValueError(f"Expected {EXPECTED_PERSONAS:,} personas, got {len(df):,}")
    if "pak_uuid" not in df.columns:
        raise ValueError("Main release is missing pak_uuid")
    duplicate_count = int(df["pak_uuid"].duplicated().sum())
    if duplicate_count:
        raise ValueError(f"Main release has {duplicate_count} duplicate pak_uuid values")
    log.info("Main release OK: %d personas, %d columns", len(df), df.shape[1])


def verify_eval_release() -> None:
    annotations_path = EVAL_RELEASE / "annotations.csv"
    sample_path = EVAL_RELEASE / "sample_personas.csv"
    anchors_path = EVAL_RELEASE / "calibration_anchors.csv"
    for path in (annotations_path, sample_path, anchors_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path.relative_to(ROOT)}")

    annotations = pd.read_csv(annotations_path)
    sample = pd.read_csv(sample_path)
    anchors = pd.read_csv(anchors_path)

    if sample["pak_uuid"].nunique() != EXPECTED_EVAL_PERSONAS:
        raise ValueError(
            f"Expected {EXPECTED_EVAL_PERSONAS} eval personas, got {sample['pak_uuid'].nunique()}"
        )
    if anchors["pak_uuid"].nunique() != EXPECTED_CALIBRATION_ANCHORS:
        raise ValueError(
            "Expected "
            f"{EXPECTED_CALIBRATION_ANCHORS} calibration anchors, got {anchors['pak_uuid'].nunique()}"
        )

    score_rows = annotations[annotations["dimension"].isin(DIMENSIONS)].copy()
    coverage = score_rows.groupby(["annotator", "dimension"])["pak_uuid"].nunique()
    for annotator in ANNOTATORS:
        for dimension in DIMENSIONS:
            observed = int(coverage.get((annotator, dimension), 0))
            if observed != EXPECTED_EVAL_PERSONAS:
                raise ValueError(
                    f"Coverage mismatch for {annotator}/{dimension}: "
                    f"expected {EXPECTED_EVAL_PERSONAS}, got {observed}"
                )

    score_values = pd.to_numeric(score_rows["score"], errors="coerce")
    invalid_scores = score_rows[score_values.isna() | ~score_values.between(1, 5)]
    if not invalid_scores.empty:
        raise ValueError(f"Found {len(invalid_scores)} scores outside 1-5")

    overall_rows = annotations[annotations["dimension"].eq(OVERALL_DIMENSION)]
    expected_overall = len(ANNOTATORS) * EXPECTED_EVAL_PERSONAS
    if len(overall_rows) != expected_overall:
        raise ValueError(f"Expected {expected_overall} overall flag rows, got {len(overall_rows)}")
    log.info(
        "PAK-1K-eval OK: %d score rows, %d overall rows",
        len(score_rows),
        len(overall_rows),
    )


def main() -> None:
    if not MAIN_MANIFEST.exists():
        log.warning(
            "Skipping main-release checks: %s not found "
            "(the 30k dataset is hosted on HF/Zenodo and is not bundled in the repo).",
            MAIN_MANIFEST.relative_to(ROOT),
        )
    else:
        verify_manifest(MAIN_MANIFEST, MAIN_RELEASE)
        verify_main_release()
    verify_manifest(EVAL_MANIFEST, EVAL_RELEASE)
    verify_eval_release()
    log.info("Release verification complete.")


if __name__ == "__main__":
    main()
