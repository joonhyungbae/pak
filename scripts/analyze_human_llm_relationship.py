"""Analyze relationships between human PAK-1K-eval annotations and LLM judges."""

from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS_PATH = ROOT / "outputs" / "pak_1k_eval_release" / "annotations.csv"
REFERENCE_CSV = ROOT / "data" / "eval" / "human_scores_260520.csv"
REPORT_PATH = ROOT / "outputs" / "reports" / "human_llm_relationship_260521.md"
DISAGREEMENT_CSV = ROOT / "outputs" / "reports" / "human_llm_large_disagreements_260521.csv"

DIMENSIONS: tuple[str, ...] = (
    "groundedness",
    "coherence",
    "plausibility",
    "fluency",
)
MODELS: tuple[str, ...] = ("claude", "gemini", "clova", "codex")
MODEL_LABELS: dict[str, str] = {
    "claude": "Claude Opus 4.7",
    "gemini": "Gemini 2.5 Pro",
    "clova": "HyperCLOVA X HCX-007",
    "codex": "Codex 5.5",
}
PANELS: dict[str, tuple[str, ...]] = {
    "claude_gemini": ("claude", "gemini"),
    "all_four": MODELS,
}
SCORE_MIN, SCORE_MAX = 1, 5


def read_annotations() -> pd.DataFrame:
    if not ANNOTATIONS_PATH.exists():
        raise FileNotFoundError(f"Missing {ANNOTATIONS_PATH.relative_to(ROOT)}")
    df = pd.read_csv(ANNOTATIONS_PATH)
    required = {"pak_uuid", "annotator", "dimension", "score", "reasoning", "flag"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{ANNOTATIONS_PATH.relative_to(ROOT)} missing columns: {sorted(missing)}")
    return df


def read_reference_order() -> list[str]:
    if not REFERENCE_CSV.exists():
        raise FileNotFoundError(f"Missing {REFERENCE_CSV.relative_to(ROOT)}")
    ref = pd.read_csv(REFERENCE_CSV, usecols=["pak_uuid"])
    return ref["pak_uuid"].astype(str).tolist()


def round_half_up(values: pd.Series | np.ndarray) -> np.ndarray:
    rounded = np.floor(np.asarray(values, dtype=float) + 0.5)
    return np.clip(rounded, SCORE_MIN, SCORE_MAX).astype(int)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman_compat(a: np.ndarray, b: np.ndarray) -> float:
    """Tie-naive Spearman used by the locked judge-human report."""
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def quadratic_weighted_kappa(a: np.ndarray, b: np.ndarray) -> float:
    labels = list(range(SCORE_MIN, SCORE_MAX + 1))
    index = {value: i for i, value in enumerate(labels)}
    observed = np.zeros((len(labels), len(labels)))
    for x, y in zip(a, b):
        observed[index[int(x)], index[int(y)]] += 1

    weights = np.zeros_like(observed)
    for i in range(len(labels)):
        for j in range(len(labels)):
            weights[i, j] = ((i - j) ** 2) / ((len(labels) - 1) ** 2)

    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    denom = (weights * expected).sum()
    if denom == 0:
        return float("nan")
    return float(1 - (weights * observed).sum() / denom)


def score_matrix(df: pd.DataFrame, dimension: str, reference_order: list[str]) -> pd.DataFrame:
    score_rows = df[df["dimension"].eq(dimension)]
    wide = score_rows.pivot(index="pak_uuid", columns="annotator", values="score")
    wide = wide.reindex(reference_order)
    expected = ["human_anonymous", *MODELS]
    missing = [column for column in expected if column not in wide.columns]
    if missing:
        raise ValueError(f"{dimension}: missing annotator columns {missing}")
    return wide[expected].dropna().astype(float)


def metric_record(
    *,
    name: str,
    dimension: str,
    pred: np.ndarray,
    human: np.ndarray,
    qwk_pred: np.ndarray | None = None,
) -> dict[str, Any]:
    diff = pred - human
    if qwk_pred is None:
        qwk_pred = pred
    return {
        "name": name,
        "dimension": dimension,
        "N": int(len(human)),
        "human_mean": float(np.mean(human)),
        "pred_mean": float(np.mean(pred)),
        "bias": float(np.mean(diff)),
        "MAE": float(np.mean(np.abs(diff))),
        "exact": float(np.mean(qwk_pred == human)),
        "within1": float(np.mean(np.abs(qwk_pred - human) <= 1)),
        "Pearson": pearson(pred, human),
        "Spearman": spearman_compat(pred, human),
        "QWK": quadratic_weighted_kappa(qwk_pred, human),
    }


def build_metric_tables(
    df: pd.DataFrame,
    reference_order: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    individual_rows: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []

    for dimension in DIMENSIONS:
        wide = score_matrix(df, dimension, reference_order)
        human = wide["human_anonymous"].to_numpy(dtype=float)
        for model in MODELS:
            pred = wide[model].to_numpy(dtype=float)
            individual_rows.append(
                metric_record(
                    name=model,
                    dimension=dimension,
                    pred=pred,
                    human=human,
                )
            )

        for panel_name, members in PANELS.items():
            pred = wide[list(members)].mean(axis=1).to_numpy(dtype=float)
            panel_rows.append(
                metric_record(
                    name=panel_name,
                    dimension=dimension,
                    pred=pred,
                    human=human,
                    qwk_pred=round_half_up(pred),
                )
            )

    return pd.DataFrame(individual_rows), pd.DataFrame(panel_rows)


def build_distribution_table(df: pd.DataFrame) -> pd.DataFrame:
    score_rows = df[df["dimension"].isin(DIMENSIONS)].copy()
    return (
        score_rows.groupby(["dimension", "annotator"])["score"]
        .agg(mean="mean", sd="std")
        .reset_index()
    )


def build_calibration_table(df: pd.DataFrame, reference_order: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dimension in DIMENSIONS:
        wide = score_matrix(df, dimension, reference_order)
        human = wide["human_anonymous"]
        panel = wide[list(MODELS)].mean(axis=1)
        residual = panel - human
        tmp = pd.DataFrame(
            {
                "human_score": human.astype(int),
                "llm_mean": panel,
                "residual": residual,
            }
        )
        grouped = tmp.groupby("human_score")
        for human_score, group in grouped:
            rows.append(
                {
                    "dimension": dimension,
                    "human_score": int(human_score),
                    "N": int(len(group)),
                    "llm_mean": float(group["llm_mean"].mean()),
                    "bias": float(group["residual"].mean()),
                    "MAE": float(group["residual"].abs().mean()),
                }
            )
    return pd.DataFrame(rows)


def build_consensus_table(df: pd.DataFrame, reference_order: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dimension in DIMENSIONS:
        wide = score_matrix(df, dimension, reference_order)
        human = wide["human_anonymous"]
        panel = wide[list(MODELS)].mean(axis=1)
        rounded = pd.Series(round_half_up(panel), index=wide.index)
        llm_range = wide[list(MODELS)].max(axis=1) - wide[list(MODELS)].min(axis=1)
        tmp = pd.DataFrame(
            {
                "range": llm_range.astype(int),
                "abs_err": (panel - human).abs(),
                "exact": rounded.eq(human),
                "within1": (rounded - human).abs().le(1),
            }
        )
        grouped = tmp.groupby("range")
        for llm_range_value, group in grouped:
            rows.append(
                {
                    "dimension": dimension,
                    "llm_range": int(llm_range_value),
                    "N": int(len(group)),
                    "MAE": float(group["abs_err"].mean()),
                    "exact": float(group["exact"].mean()),
                    "within1": float(group["within1"].mean()),
                }
            )
    return pd.DataFrame(rows)


def build_pairwise_table(df: pd.DataFrame, reference_order: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dimension in DIMENSIONS:
        wide = score_matrix(df, dimension, reference_order)
        human_qwks = [
            quadratic_weighted_kappa(
                wide["human_anonymous"].to_numpy(dtype=float),
                wide[model].to_numpy(dtype=float),
            )
            for model in MODELS
        ]
        llm_pair_qwks = [
            quadratic_weighted_kappa(
                wide[a].to_numpy(dtype=float),
                wide[b].to_numpy(dtype=float),
            )
            for a, b in combinations(MODELS, 2)
        ]
        rows.append(
            {
                "dimension": dimension,
                "human_llm_avg_qwk": float(np.mean(human_qwks)),
                "llm_pair_avg_qwk": float(np.mean(llm_pair_qwks)),
                "llm_pair_min_qwk": float(np.min(llm_pair_qwks)),
                "llm_pair_max_qwk": float(np.max(llm_pair_qwks)),
            }
        )
    return pd.DataFrame(rows)


def human_flag_series(df: pd.DataFrame) -> pd.Series:
    flag_rows = df[
        df["annotator"].eq("human_anonymous") & df["dimension"].eq("_overall")
    ][["pak_uuid", "flag"]].copy()
    flag_rows["human_flagged"] = flag_rows["flag"].fillna("").astype(str).str.strip().ne("")
    return flag_rows.set_index("pak_uuid")["human_flagged"]


def build_flag_table(df: pd.DataFrame, reference_order: list[str]) -> pd.DataFrame:
    flags = human_flag_series(df)
    rows: list[pd.DataFrame] = []
    for dimension in DIMENSIONS:
        wide = score_matrix(df, dimension, reference_order)
        panel = wide[list(MODELS)].mean(axis=1)
        tmp = pd.DataFrame(
            {
                "dimension": dimension,
                "abs_err": (panel - wide["human_anonymous"]).abs(),
            },
            index=wide.index,
        )
        tmp["human_flagged"] = flags.reindex(tmp.index).fillna(False)
        rows.append(tmp.reset_index(drop=True))

    all_rows = pd.concat(rows, ignore_index=True)
    return (
        all_rows.groupby(["dimension", "human_flagged"])["abs_err"]
        .agg(N="size", MAE="mean", p95=lambda x: float(x.quantile(0.95)), large=lambda x: float((x >= 1.5).mean()))
        .reset_index()
    )


def build_large_disagreements(df: pd.DataFrame, reference_order: list[str]) -> pd.DataFrame:
    human_flags = (
        df[df["annotator"].eq("human_anonymous") & df["dimension"].eq("_overall")]
        .set_index("pak_uuid")["flag"]
        .fillna("")
        .astype(str)
    )
    rows: list[pd.DataFrame] = []
    for dimension in DIMENSIONS:
        wide = score_matrix(df, dimension, reference_order)
        out = wide.reset_index().copy()
        out["dimension"] = dimension
        out["llm_mean"] = wide[list(MODELS)].mean(axis=1).to_numpy(dtype=float)
        out["llm_rounded"] = round_half_up(out["llm_mean"])
        out["residual"] = out["llm_mean"] - out["human_anonymous"].astype(float)
        out["abs_residual"] = out["residual"].abs()
        out["llm_range"] = (
            wide[list(MODELS)].max(axis=1) - wide[list(MODELS)].min(axis=1)
        ).to_numpy(dtype=int)
        out["human_flag"] = human_flags.reindex(out["pak_uuid"]).fillna("").to_numpy()
        rows.append(out)

    combined = pd.concat(rows, ignore_index=True)
    cols = [
        "pak_uuid",
        "dimension",
        "human_anonymous",
        *MODELS,
        "llm_mean",
        "llm_rounded",
        "residual",
        "abs_residual",
        "llm_range",
        "human_flag",
    ]
    large = combined[combined["abs_residual"].ge(1.5)][cols].copy()
    large = large.sort_values(["abs_residual", "dimension", "pak_uuid"], ascending=[False, True, True])
    return large


def markdown_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)


def write_report(
    individual: pd.DataFrame,
    panel: pd.DataFrame,
    distributions: pd.DataFrame,
    calibration: pd.DataFrame,
    consensus: pd.DataFrame,
    pairwise: pd.DataFrame,
    flag_table: pd.DataFrame,
    large: pd.DataFrame,
) -> None:
    individual_view = individual[
        ["name", "dimension", "N", "exact", "within1", "MAE", "bias", "Pearson", "Spearman", "QWK"]
    ].copy()
    panel_view = panel[
        ["name", "dimension", "N", "exact", "within1", "MAE", "bias", "Pearson", "Spearman", "QWK"]
    ].copy()

    judge_summary = (
        individual.groupby("name")[["QWK", "Pearson", "MAE", "bias"]]
        .mean()
        .reset_index()
        .rename(columns={"name": "judge", "bias": "mean_bias"})
    )
    dimension_summary = (
        individual.groupby("dimension")[["QWK", "Pearson", "MAE", "bias"]]
        .mean()
        .reset_index()
        .rename(columns={"bias": "mean_bias"})
    )
    distribution_view = distributions.pivot_table(
        index="dimension",
        columns="annotator",
        values="mean",
        aggfunc="first",
    ).reset_index()

    calibration_view = calibration[calibration["human_score"].isin([1, 2, 3, 4, 5])].copy()
    consensus_view = consensus.copy()
    pairwise_view = pairwise.copy()
    flag_view = flag_table.copy()
    large_counts = large["dimension"].value_counts().rename_axis("dimension").reset_index(name="N")
    top_large = large.head(15).copy()

    lines = [
        "# Human Annotation and LLM Judge Relationship",
        "",
        f"- Source annotations: `{ANNOTATIONS_PATH.relative_to(ROOT)}`",
        f"- Human reference order: `{REFERENCE_CSV.relative_to(ROOT)}`",
        f"- Large-disagreement CSV: `{DISAGREEMENT_CSV.relative_to(ROOT)}`",
        "- Definition: residual = LLM all-four mean score - human score.",
        "- QWK for panels uses half-up rounded panel means clipped to the 1-5 scale.",
        "",
        "## Executive Read",
        "",
        "- The LLM judges track the human annotation well enough to be useful as an agreement layer, but not as a substitute for the human reference.",
        "- Plausibility is the most stable dimension. Groundedness is the weakest and accounts for most large disagreements.",
        "- Averaging judges improves the relationship with the human scores. The all-four panel is stronger than the Claude-Gemini panel on every dimension by Pearson and rounded QWK.",
        "- Bias is small at the dimension mean level, but calibration shows regression toward the middle: low human scores are lifted and high human scores are pulled down.",
        "- HyperCLOVA X HCX-007 is slightly less clustered with the other LLMs than Claude, Gemini, and Codex, especially outside plausibility.",
        "",
        "## Average Individual Relationship",
        "",
        markdown_table(judge_summary),
        "",
        "## Dimension-Level Relationship",
        "",
        markdown_table(dimension_summary),
        "",
        "## Individual Judge Metrics",
        "",
        markdown_table(individual_view),
        "",
        "## Panel Mean Metrics",
        "",
        markdown_table(panel_view),
        "",
        "## Mean Score Calibration",
        "",
        markdown_table(distribution_view),
        "",
        "## All-Four Panel Calibration by Human Score",
        "",
        markdown_table(calibration_view),
        "",
        "## LLM Consensus and Human Error",
        "",
        markdown_table(consensus_view),
        "",
        "## LLM-to-LLM Versus Human-to-LLM QWK",
        "",
        markdown_table(pairwise_view),
        "",
        "## Human Flag Relation",
        "",
        markdown_table(flag_view),
        "",
        "## Large Disagreements",
        "",
        "Rows where `abs(residual) >= 1.5` are written to the CSV above.",
        "",
        markdown_table(large_counts),
        "",
        "Top cases by absolute residual:",
        "",
        markdown_table(top_large),
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = read_annotations()
    reference_order = read_reference_order()

    individual, panel = build_metric_tables(df, reference_order)
    distributions = build_distribution_table(df)
    calibration = build_calibration_table(df, reference_order)
    consensus = build_consensus_table(df, reference_order)
    pairwise = build_pairwise_table(df, reference_order)
    flag_table = build_flag_table(df, reference_order)
    large = build_large_disagreements(df, reference_order)

    DISAGREEMENT_CSV.parent.mkdir(parents=True, exist_ok=True)
    large.to_csv(DISAGREEMENT_CSV, index=False)
    write_report(
        individual=individual,
        panel=panel,
        distributions=distributions,
        calibration=calibration,
        consensus=consensus,
        pairwise=pairwise,
        flag_table=flag_table,
        large=large,
    )

    log.info("Wrote report: %s", REPORT_PATH.relative_to(ROOT))
    log.info("Wrote disagreement CSV: %s", DISAGREEMENT_CSV.relative_to(ROOT))
    log.info("Large disagreements: %d", len(large))


if __name__ == "__main__":
    main()
