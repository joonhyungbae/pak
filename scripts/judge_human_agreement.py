"""Compute model-judge agreement against the human PAK-1K-eval reference.

This clean-release version reads the canonical judge bundle at
``outputs/pak_1k_eval_release`` rather than raw batch retry folders. The report
is therefore reproducible from the release package alone.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RELEASE_DIR = ROOT / "outputs" / "pak_1k_eval_release"
ANNOTATIONS_PATH = RELEASE_DIR / "annotations.csv"
REPORT_PATH = ROOT / "outputs" / "reports" / "judge_human_agreement_260520.md"
REFERENCE_CSV = ROOT / "data" / "eval" / "human_scores_260520.csv"

DIMENSIONS: tuple[str, ...] = (
    "groundedness",
    "coherence",
    "plausibility",
    "fluency",
)
JUDGES: tuple[str, ...] = ("claude", "gemini", "clova", "codex")
JUDGE_LABELS: dict[str, str] = {
    "claude": "claude / Claude Opus 4.7",
    "gemini": "gemini / Gemini 2.5 Pro",
    "clova": "clova / HyperCLOVA X HCX-007",
    "codex": "codex / Codex 5.5",
}
SCORE_MIN, SCORE_MAX = 1, 5


def quadratic_weighted_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """QWK for 1-5 ordinal integer labels."""
    labels = list(range(SCORE_MIN, SCORE_MAX + 1))
    k = len(labels)
    idx = {value: i for i, value in enumerate(labels)}
    observed = np.zeros((k, k))
    for x, y in zip(a, b):
        observed[idx[int(x)], idx[int(y)]] += 1

    weights = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            weights[i, j] = ((i - j) ** 2) / ((k - 1) ** 2)

    hist_a = observed.sum(axis=1)
    hist_b = observed.sum(axis=0)
    expected = np.outer(hist_a, hist_b) / observed.sum()
    denom = (weights * expected).sum()
    if denom == 0:
        return float("nan")
    return float(1 - (weights * observed).sum() / denom)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Tie-naive Spearman, matching the original PAK-1K-eval report script."""
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def fmt(x: float) -> str:
    return "--" if x != x else f"{x:.3f}"


def load_scores() -> pd.DataFrame:
    if not ANNOTATIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {ANNOTATIONS_PATH.relative_to(ROOT)}. "
            "Run scripts/prepare_eval_judge_release.py first."
        )

    df = pd.read_csv(ANNOTATIONS_PATH)
    score_rows = df[df["dimension"].isin(DIMENSIONS)].copy()
    missing = {"pak_uuid", "annotator", "dimension", "score"} - set(score_rows.columns)
    if missing:
        raise ValueError(f"{ANNOTATIONS_PATH.relative_to(ROOT)} missing columns: {sorted(missing)}")

    bad_scores = score_rows[~score_rows["score"].between(SCORE_MIN, SCORE_MAX)]
    if not bad_scores.empty:
        raise ValueError(f"Found {len(bad_scores)} scores outside {SCORE_MIN}-{SCORE_MAX}")

    duplicates = score_rows.duplicated(["pak_uuid", "annotator", "dimension"]).sum()
    if duplicates:
        raise ValueError(f"Found {duplicates} duplicate score rows")

    return score_rows


def load_reference_order() -> list[str]:
    if not REFERENCE_CSV.exists():
        raise FileNotFoundError(f"Missing {REFERENCE_CSV.relative_to(ROOT)}")
    ref = pd.read_csv(REFERENCE_CSV, usecols=["pak_uuid"])
    return ref["pak_uuid"].astype(str).tolist()


def compute_metrics(
    df: pd.DataFrame,
    judge: str,
    dimension: str,
    reference_order: list[str],
) -> dict[str, float]:
    subset = df[df["dimension"].eq(dimension)]
    wide = subset.pivot(index="pak_uuid", columns="annotator", values="score")
    wide = wide.reindex(reference_order)
    aligned = wide[["human_anonymous", judge]].dropna()
    if aligned.empty:
        raise ValueError(f"No overlap for judge={judge}, dimension={dimension}")

    ref = aligned["human_anonymous"].to_numpy(dtype=float)
    pred = aligned[judge].to_numpy(dtype=float)
    diff = pred - ref
    return {
        "N": float(len(aligned)),
        "exact": float(np.mean(pred == ref)),
        "within1": float(np.mean(np.abs(diff) <= 1)),
        "MAE": float(np.mean(np.abs(diff))),
        "bias": float(np.mean(diff)),
        "spearman": spearman(pred, ref),
        "qwk": quadratic_weighted_kappa(pred, ref),
    }


def write_report(df: pd.DataFrame) -> None:
    reference_order = load_reference_order()
    coverage = (
        df.groupby(["annotator", "dimension"])["pak_uuid"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(index=["human_anonymous", *JUDGES], columns=list(DIMENSIONS))
    )
    log.info("Coverage:\n%s", coverage.to_string())

    lines: list[str] = [
        "# Judge ↔ Reference agreement report",
        "",
        "- Generated: 2026-05-21",
        f"- Source: `{ANNOTATIONS_PATH.relative_to(ROOT)}`.",
        "- Reference annotator: `human_anonymous` (anonymous human scoring).",
        "- Primary metric: **QWK** (Quadratic Weighted Kappa). "
        "bias = judge_mean - ref_mean (positive means judge is more lenient than human).",
        f"- Spearman tie ordering follows `{REFERENCE_CSV.relative_to(ROOT)}` for "
        "compatibility with the locked 2026-05-20 report.",
        "- Model labels: `clova` refers to the HyperCLOVA X HCX-007 run on "
        "NAVER Cloud CLOVA Studio Chat Completions v3.",
        "",
    ]

    for judge in JUDGES:
        covered = int(coverage.loc[judge].min())
        lines.append(f"## {JUDGE_LABELS[judge]}  (aggregated {covered} personas)")
        lines.append("")
        lines.append("| dim | N | exact | within1 | MAE | bias | spearman | QWK |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for dimension in DIMENSIONS:
            metrics = compute_metrics(df, judge, dimension, reference_order)
            lines.append(
                f"| {dimension} | {int(metrics['N'])} | {fmt(metrics['exact'])} "
                f"| {fmt(metrics['within1'])} | {fmt(metrics['MAE'])} "
                f"| {metrics['bias']:+.3f} | {fmt(metrics['spearman'])} "
                f"| {fmt(metrics['qwk'])} |"
            )
        lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote report: %s", REPORT_PATH.relative_to(ROOT))


def main() -> None:
    df = load_scores()
    write_report(df)


if __name__ == "__main__":
    main()
