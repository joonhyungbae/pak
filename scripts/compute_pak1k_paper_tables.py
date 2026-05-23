"""Compute PAK-1K-eval paper tables from the canonical release annotations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS_PATH = ROOT / "outputs" / "pak_1k_eval_release" / "annotations.csv"
HUMAN_ORIGINAL_PATH = ROOT / "data" / "eval" / "human_scores_260520.csv"
PERSONA_SAMPLE_PATH = ROOT / "outputs" / "pak_1k_eval_release" / "sample_personas.csv"
REPORT_PATH = ROOT / "outputs" / "reports" / "pak_1k_eval_paper_tables_260521.md"

DIMENSIONS: tuple[str, ...] = (
    "groundedness",
    "coherence",
    "plausibility",
    "fluency",
)
DIMENSION_LABELS: dict[str, str] = {
    "groundedness": "Groundedness",
    "coherence": "Coherence",
    "plausibility": "Plausibility",
    "fluency": "Fluency",
}
JUDGES: tuple[str, ...] = ("claude", "gemini", "clova", "codex")
PRIMARY_PANEL: tuple[str, ...] = ("claude", "gemini")
BOOTSTRAP_SEED = 20260520
BOOTSTRAP_N = 10_000
SCORE_MIN, SCORE_MAX = 1, 5


def read_annotations() -> pd.DataFrame:
    if not ANNOTATIONS_PATH.exists():
        raise FileNotFoundError(f"Missing {ANNOTATIONS_PATH.relative_to(ROOT)}")
    df = pd.read_csv(ANNOTATIONS_PATH)
    return df[df["dimension"].isin(DIMENSIONS)].copy()


def read_original_human_scores() -> pd.DataFrame:
    if not HUMAN_ORIGINAL_PATH.exists():
        raise FileNotFoundError(f"Missing {HUMAN_ORIGINAL_PATH.relative_to(ROOT)}")
    human = pd.read_csv(HUMAN_ORIGINAL_PATH)
    rows: list[dict[str, Any]] = []
    for dimension in DIMENSIONS:
        for row in human[["pak_uuid", dimension]].itertuples(index=False):
            rows.append(
                {
                    "pak_uuid": row.pak_uuid,
                    "annotator": "human_anonymous",
                    "dimension": dimension,
                    "score": getattr(row, dimension),
                }
            )
    return pd.DataFrame(rows)


def read_human_batch_map() -> pd.DataFrame:
    if not HUMAN_ORIGINAL_PATH.exists():
        raise FileNotFoundError(f"Missing {HUMAN_ORIGINAL_PATH.relative_to(ROOT)}")
    human = pd.read_csv(HUMAN_ORIGINAL_PATH)
    return human[["pak_uuid", "order_index", "batch_id"]].copy()


def with_original_human_scores(df: pd.DataFrame) -> pd.DataFrame:
    original = read_original_human_scores()
    mask = df["annotator"].eq("human_anonymous") & df["dimension"].isin(DIMENSIONS)
    without_human = df.loc[~mask].copy()
    human_rows = (
        df.loc[mask]
        .drop(columns=["score"])
        .merge(original, on=["pak_uuid", "annotator", "dimension"], how="left")
    )
    if human_rows["score"].isna().any():
        missing = human_rows.loc[human_rows["score"].isna(), ["pak_uuid", "dimension"]]
        raise ValueError(f"Original human score lookup failed for {len(missing)} rows")
    return pd.concat([without_human, human_rows], ignore_index=True)


def read_art_fields() -> pd.DataFrame:
    personas = pd.read_csv(PERSONA_SAMPLE_PATH)
    return personas[["pak_uuid", "art_field_primary"]].copy()


def round_half_up(values: np.ndarray) -> np.ndarray:
    rounded = np.floor(np.asarray(values, dtype=float) + 0.5)
    return np.clip(rounded, SCORE_MIN, SCORE_MAX).astype(int)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman_tie_aware(a: np.ndarray, b: np.ndarray) -> float:
    """Tie-aware Spearman correlation using average ranks."""
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    result = spearmanr(a, b)
    return float(result.statistic)


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


def dimension_wide(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    wide = df[df["dimension"].eq(dimension)].pivot(
        index="pak_uuid",
        columns="annotator",
        values="score",
    )
    required = ["human_anonymous", *JUDGES]
    missing = [column for column in required if column not in wide.columns]
    if missing:
        raise ValueError(f"{dimension}: missing {missing}")
    return wide[required].dropna().astype(float)


def bootstrap_pearson_ci(
    table: pd.DataFrame,
    *,
    rng: np.random.Generator,
) -> tuple[float, float]:
    strata = {
        field: group.index.to_numpy()
        for field, group in table.groupby("art_field_primary", sort=True)
    }
    values = np.empty(BOOTSTRAP_N, dtype=float)
    human = table["human"].to_numpy(dtype=float)
    pred = table["panel_mean"].to_numpy(dtype=float)
    index_positions = {idx: pos for pos, idx in enumerate(table.index)}

    for i in range(BOOTSTRAP_N):
        sampled_positions: list[int] = []
        for indices in strata.values():
            sampled = rng.choice(indices, size=len(indices), replace=True)
            sampled_positions.extend(index_positions[idx] for idx in sampled)
        pos = np.asarray(sampled_positions, dtype=int)
        values[i] = pearson(pred[pos], human[pos])

    low, high = np.nanquantile(values, [0.025, 0.975])
    return float(low), float(high)


def bootstrap_metric_cis(
    table: pd.DataFrame,
    *,
    rng: np.random.Generator,
) -> dict[str, tuple[float, float]]:
    strata = {
        field: group.index.to_numpy()
        for field, group in table.groupby("art_field_primary", sort=True)
    }
    index_positions = {idx: pos for pos, idx in enumerate(table.index)}
    human = table["human"].to_numpy(dtype=float)
    pred = table["panel_mean"].to_numpy(dtype=float)

    values: dict[str, np.ndarray] = {
        "Pearson": np.empty(BOOTSTRAP_N, dtype=float),
        "Spearman": np.empty(BOOTSTRAP_N, dtype=float),
        "QWK": np.empty(BOOTSTRAP_N, dtype=float),
        "MAE": np.empty(BOOTSTRAP_N, dtype=float),
        "Bias": np.empty(BOOTSTRAP_N, dtype=float),
    }
    for i in range(BOOTSTRAP_N):
        sampled_positions: list[int] = []
        for indices in strata.values():
            sampled = rng.choice(indices, size=len(indices), replace=True)
            sampled_positions.extend(index_positions[idx] for idx in sampled)
        pos = np.asarray(sampled_positions, dtype=int)
        sample_human = human[pos]
        sample_pred = pred[pos]
        sample_diff = sample_pred - sample_human
        values["Pearson"][i] = pearson(sample_pred, sample_human)
        values["Spearman"][i] = spearman_tie_aware(sample_pred, sample_human)
        values["QWK"][i] = quadratic_weighted_kappa(round_half_up(sample_pred), sample_human)
        values["MAE"][i] = float(np.mean(np.abs(sample_diff)))
        values["Bias"][i] = float(np.mean(sample_diff))

    return {
        name: tuple(float(x) for x in np.nanquantile(metric_values, [0.025, 0.975]))
        for name, metric_values in values.items()
    }


def panel_metrics(
    df: pd.DataFrame,
    art_fields: pd.DataFrame,
) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    for dimension in DIMENSIONS:
        wide = dimension_wide(df, dimension)
        table = wide.merge(art_fields, left_index=True, right_on="pak_uuid").set_index("pak_uuid")
        table["human"] = table["human_anonymous"].astype(float)
        table["panel_mean"] = table[list(PRIMARY_PANEL)].mean(axis=1)
        rounded = round_half_up(table["panel_mean"].to_numpy(dtype=float))
        human = table["human"].to_numpy(dtype=float)
        pred = table["panel_mean"].to_numpy(dtype=float)
        diff = pred - human
        cis = bootstrap_metric_cis(table, rng=rng)
        rows.append(
            {
                "dimension": dimension,
                "label": DIMENSION_LABELS[dimension],
                "N": int(len(table)),
                "Pearson": pearson(pred, human),
                "Pearson_CI_low": cis["Pearson"][0],
                "Pearson_CI_high": cis["Pearson"][1],
                "Spearman": spearman_tie_aware(pred, human),
                "Spearman_CI_low": cis["Spearman"][0],
                "Spearman_CI_high": cis["Spearman"][1],
                "QWK": quadratic_weighted_kappa(rounded, human),
                "QWK_CI_low": cis["QWK"][0],
                "QWK_CI_high": cis["QWK"][1],
                "MAE": float(np.mean(np.abs(diff))),
                "MAE_CI_low": cis["MAE"][0],
                "MAE_CI_high": cis["MAE"][1],
                "Bias": float(np.mean(diff)),
                "Bias_CI_low": cis["Bias"][0],
                "Bias_CI_high": cis["Bias"][1],
            }
        )
    return pd.DataFrame(rows)


def individual_qwk(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dimension in DIMENSIONS:
        wide = dimension_wide(df, dimension)
        row: dict[str, Any] = {
            "dimension": dimension,
            "label": DIMENSION_LABELS[dimension],
        }
        human = wide["human_anonymous"].to_numpy(dtype=float)
        for judge in JUDGES:
            row[judge] = quadratic_weighted_kappa(wide[judge].to_numpy(dtype=float), human)
        rows.append(row)
    return pd.DataFrame(rows)


def human_quality_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dimension in DIMENSIONS:
        wide = dimension_wide(df, dimension)
        scores = wide["human_anonymous"].astype(int)
        counts = scores.value_counts().reindex(range(SCORE_MIN, SCORE_MAX + 1), fill_value=0)
        n = int(len(scores))
        row: dict[str, Any] = {
            "dimension": dimension,
            "label": DIMENSION_LABELS[dimension],
            "N": n,
            "Mean": float(scores.mean()),
            "Median": float(scores.median()),
            "Le2_pct": float((scores <= 2).mean() * 100),
            "Le3_pct": float((scores <= 3).mean() * 100),
            "Ge4_pct": float((scores >= 4).mean() * 100),
        }
        for score in range(SCORE_MIN, SCORE_MAX + 1):
            row[f"Score{score}_pct"] = float((counts.loc[score] / n) * 100)
        rows.append(row)
    return pd.DataFrame(rows)


def amendment_sensitivity(amended_panel: pd.DataFrame, original_panel: pd.DataFrame) -> pd.DataFrame:
    merged = amended_panel.merge(
        original_panel,
        on=["dimension", "label", "N"],
        suffixes=("_amended", "_original"),
    )
    rows: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        rows.append(
            {
                "dimension": row.dimension,
                "label": row.label,
                "N": row.N,
                "Pearson_original": row.Pearson_original,
                "Pearson_amended": row.Pearson_amended,
                "Delta_Pearson": row.Pearson_amended - row.Pearson_original,
                "QWK_original": row.QWK_original,
                "QWK_amended": row.QWK_amended,
                "Delta_QWK": row.QWK_amended - row.QWK_original,
            }
        )
    return pd.DataFrame(rows)


def panel_long_table(
    df: pd.DataFrame,
    art_fields: pd.DataFrame,
    batch_map: pd.DataFrame,
    dimension: str,
    panel_members: tuple[str, ...] = PRIMARY_PANEL,
) -> pd.DataFrame:
    wide = dimension_wide(df, dimension)
    table = wide.merge(art_fields, left_index=True, right_on="pak_uuid").set_index("pak_uuid")
    table = table.merge(batch_map, left_index=True, right_on="pak_uuid").set_index("pak_uuid")
    table["dimension"] = dimension
    table["label"] = DIMENSION_LABELS[dimension]
    table["human"] = table["human_anonymous"].astype(float)
    table["panel_mean"] = table[list(panel_members)].mean(axis=1)
    table["rounded_panel"] = round_half_up(table["panel_mean"].to_numpy(dtype=float))
    table["diff"] = table["panel_mean"] - table["human"]
    table["mean_score"] = (table["panel_mean"] + table["human"]) / 2
    return table


def bland_altman(df: pd.DataFrame, art_fields: pd.DataFrame, batch_map: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dimension in DIMENSIONS:
        table = panel_long_table(df, art_fields, batch_map, dimension)
        diff = table["diff"].to_numpy(dtype=float)
        mean_bias = float(np.mean(diff))
        sd = float(np.std(diff, ddof=1))
        rows.append(
            {
                "dimension": dimension,
                "label": DIMENSION_LABELS[dimension],
                "Bias": mean_bias,
                "Diff_SD": sd,
                "LoA_low": mean_bias - 1.96 * sd,
                "LoA_high": mean_bias + 1.96 * sd,
            }
        )
    return pd.DataFrame(rows)


def batch_drift(df: pd.DataFrame, art_fields: pd.DataFrame, batch_map: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dimension in DIMENSIONS:
        table = panel_long_table(df, art_fields, batch_map, dimension)
        x = table["batch_id"].to_numpy(dtype=float)
        y = table["diff"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        pred = intercept + slope * x
        residual = y - pred
        n = len(y)
        sse = float(np.sum(residual**2))
        sxx = float(np.sum((x - np.mean(x)) ** 2))
        se = float(np.sqrt((sse / (n - 2)) / sxx)) if n > 2 and sxx > 0 else float("nan")
        t_value = float(slope / se) if se and not np.isnan(se) else float("nan")
        batch_means = table.groupby("batch_id")["diff"].mean().sort_index()
        rows.append(
            {
                "dimension": dimension,
                "label": DIMENSION_LABELS[dimension],
                "Slope_per_batch": float(slope),
                "Slope_SE": se,
                "T": t_value,
                "Batch1_bias": float(batch_means.loc[1]),
                "Batch5_bias": float(batch_means.loc[5]),
                "Max_batch_abs_bias": float(batch_means.abs().max()),
            }
        )
    return pd.DataFrame(rows)


def panel_drop_robustness(df: pd.DataFrame, art_fields: pd.DataFrame, batch_map: pd.DataFrame) -> pd.DataFrame:
    panel_specs: dict[str, tuple[str, ...]] = {
        "Claude+Gemini": ("claude", "gemini"),
        "Claude only": ("claude",),
        "Gemini only": ("gemini",),
        "Claude+Gemini+HCX": ("claude", "gemini", "clova"),
        "Claude+Gemini+Codex": ("claude", "gemini", "codex"),
        "All four": ("claude", "gemini", "clova", "codex"),
    }
    rows: list[dict[str, Any]] = []
    for dimension in DIMENSIONS:
        for panel_label, members in panel_specs.items():
            table = panel_long_table(df, art_fields, batch_map, dimension, panel_members=members)
            human = table["human"].to_numpy(dtype=float)
            pred = table["panel_mean"].to_numpy(dtype=float)
            rows.append(
                {
                    "dimension": dimension,
                    "label": DIMENSION_LABELS[dimension],
                    "Panel": panel_label,
                    "Pearson": pearson(pred, human),
                    "QWK": quadratic_weighted_kappa(round_half_up(pred), human),
                    "MAE": float(np.mean(np.abs(pred - human))),
                }
            )
    return pd.DataFrame(rows)


def fmt(value: float) -> str:
    return f"{value:.3f}"


def signed(value: float) -> str:
    return f"{value:+.3f}"


def latex_panel_rows(panel: pd.DataFrame) -> list[str]:
    rows: list[str] = []
    for row in panel.itertuples(index=False):
        rows.append(
            f"{row.label} & {row.N} & {fmt(row.Pearson)} & "
            f"[{fmt(row.Pearson_CI_low)}, {fmt(row.Pearson_CI_high)}] & {fmt(row.Spearman)} & "
            f"{fmt(row.QWK)} & {fmt(row.MAE)} & {signed(row.Bias)} \\\\"
        )
    return rows


def latex_qwk_rows(qwk: pd.DataFrame) -> list[str]:
    rows: list[str] = []
    for row in qwk.itertuples(index=False):
        rows.append(
            f"{row.label} & {fmt(row.claude)} & {fmt(row.gemini)} & "
            f"{fmt(row.clova)} & {fmt(row.codex)} \\\\"
        )
    return rows


def latex_human_quality_rows(quality: pd.DataFrame) -> list[str]:
    rows: list[str] = []
    for row in quality.itertuples(index=False):
        rows.append(
            f"{row.label} & {row.N} & {fmt(row.Mean)} & {row.Le2_pct:.1f} & "
            f"{row.Score1_pct:.1f} & {row.Score2_pct:.1f} & {row.Score3_pct:.1f} & "
            f"{row.Score4_pct:.1f} & {row.Score5_pct:.1f} \\\\"
        )
    return rows


def latex_sensitivity_rows(sensitivity: pd.DataFrame) -> list[str]:
    rows: list[str] = []
    for row in sensitivity.itertuples(index=False):
        rows.append(
            f"{row.label} & {fmt(row.Pearson_original)} & {fmt(row.Pearson_amended)} & "
            f"{signed(row.Delta_Pearson)} & {fmt(row.QWK_original)} & "
            f"{fmt(row.QWK_amended)} & {signed(row.Delta_QWK)} \\\\"
        )
    return rows


def latex_interval_rows(panel: pd.DataFrame) -> list[str]:
    rows: list[str] = []
    for row in panel.itertuples(index=False):
        rows.append(
            f"{row.label} & [{fmt(row.Spearman_CI_low)}, {fmt(row.Spearman_CI_high)}] & "
            f"[{fmt(row.QWK_CI_low)}, {fmt(row.QWK_CI_high)}] & "
            f"[{fmt(row.MAE_CI_low)}, {fmt(row.MAE_CI_high)}] & "
            f"[{signed(row.Bias_CI_low)}, {signed(row.Bias_CI_high)}] \\\\"
        )
    return rows


def latex_bland_altman_rows(ba: pd.DataFrame) -> list[str]:
    rows: list[str] = []
    for row in ba.itertuples(index=False):
        rows.append(
            f"{row.label} & {signed(row.Bias)} & {fmt(row.Diff_SD)} & "
            f"[{signed(row.LoA_low)}, {signed(row.LoA_high)}] \\\\"
        )
    return rows


def latex_drift_rows(drift: pd.DataFrame) -> list[str]:
    rows: list[str] = []
    for row in drift.itertuples(index=False):
        rows.append(
            f"{row.label} & {signed(row.Slope_per_batch)} & {fmt(row.Slope_SE)} & "
            f"{signed(row.T)} & {signed(row.Batch1_bias)} & {signed(row.Batch5_bias)} \\\\"
        )
    return rows


def latex_panel_drop_rows(robustness: pd.DataFrame) -> list[str]:
    rows: list[str] = []
    selected = robustness[robustness["Panel"].isin(["Claude+Gemini", "Claude only", "Gemini only", "All four"])]
    for label, group in selected.groupby("label", sort=False):
        lookup = group.set_index("Panel")
        rows.append(
            f"{label} & {fmt(lookup.loc['Claude+Gemini', 'QWK'])} & "
            f"{fmt(lookup.loc['Claude only', 'QWK'])} & "
            f"{fmt(lookup.loc['Gemini only', 'QWK'])} & "
            f"{fmt(lookup.loc['All four', 'QWK'])} \\\\"
        )
    return rows


def write_report(
    panel: pd.DataFrame,
    quality: pd.DataFrame,
    qwk: pd.DataFrame,
    sensitivity: pd.DataFrame,
    ba: pd.DataFrame,
    drift: pd.DataFrame,
    robustness: pd.DataFrame,
) -> None:
    lines = [
        "# PAK-1K-eval Paper Tables",
        "",
        f"- Source annotations: `{ANNOTATIONS_PATH.relative_to(ROOT)}`",
        f"- Original human scores for amendment sensitivity: `{HUMAN_ORIGINAL_PATH.relative_to(ROOT)}`",
        f"- Art-field strata: `{PERSONA_SAMPLE_PATH.relative_to(ROOT)}`",
        f"- Bootstrap: {BOOTSTRAP_N} stratified resamples within art field, seed {BOOTSTRAP_SEED}.",
        "- Spearman uses tie-aware average ranks (`scipy.stats.spearmanr`).",
        "- Primary panel: Claude Opus 4.7 + Gemini 2.5 Pro.",
        "- Human reference includes `data/eval/human_scores_260520_amendments_260521.csv` overlay via the release annotations.",
        "",
        "## Human Quality Distribution",
        "",
        quality[
            [
                "label",
                "N",
                "Mean",
                "Median",
                "Le2_pct",
                "Le3_pct",
                "Ge4_pct",
                "Score1_pct",
                "Score2_pct",
                "Score3_pct",
                "Score4_pct",
                "Score5_pct",
            ]
        ].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Panel Table",
        "",
        panel[
            [
                "label",
                "N",
                "Pearson",
                "Pearson_CI_low",
                "Pearson_CI_high",
                "Spearman",
                "QWK",
                "MAE",
                "Bias",
            ]
        ].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Secondary Bootstrap Intervals",
        "",
        panel[
            [
                "label",
                "Spearman_CI_low",
                "Spearman_CI_high",
                "QWK_CI_low",
                "QWK_CI_high",
                "MAE_CI_low",
                "MAE_CI_high",
                "Bias_CI_low",
                "Bias_CI_high",
            ]
        ].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Individual QWK Table",
        "",
        qwk[["label", "claude", "gemini", "clova", "codex"]].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Bland-Altman Summary",
        "",
        ba[["label", "Bias", "Diff_SD", "LoA_low", "LoA_high"]].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Batch Drift",
        "",
        drift[
            [
                "label",
                "Slope_per_batch",
                "Slope_SE",
                "T",
                "Batch1_bias",
                "Batch5_bias",
                "Max_batch_abs_bias",
            ]
        ].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Panel-Drop Robustness",
        "",
        robustness[["label", "Panel", "Pearson", "QWK", "MAE"]].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Amendment Sensitivity Table",
        "",
        sensitivity[
            [
                "label",
                "Pearson_original",
                "Pearson_amended",
                "Delta_Pearson",
                "QWK_original",
                "QWK_amended",
                "Delta_QWK",
            ]
        ].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## LaTeX Panel Rows",
        "",
        "```tex",
        *latex_panel_rows(panel),
        "```",
        "",
        "## LaTeX Human Quality Rows",
        "",
        "```tex",
        *latex_human_quality_rows(quality),
        "```",
        "",
        "## LaTeX Individual QWK Rows",
        "",
        "```tex",
        *latex_qwk_rows(qwk),
        "```",
        "",
        "## LaTeX Secondary Interval Rows",
        "",
        "```tex",
        *latex_interval_rows(panel),
        "```",
        "",
        "## LaTeX Bland-Altman Rows",
        "",
        "```tex",
        *latex_bland_altman_rows(ba),
        "```",
        "",
        "## LaTeX Batch Drift Rows",
        "",
        "```tex",
        *latex_drift_rows(drift),
        "```",
        "",
        "## LaTeX Panel-Drop Rows",
        "",
        "```tex",
        *latex_panel_drop_rows(robustness),
        "```",
        "",
        "## LaTeX Amendment Sensitivity Rows",
        "",
        "```tex",
        *latex_sensitivity_rows(sensitivity),
        "```",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    annotations = read_annotations()
    art_fields = read_art_fields()
    batch_map = read_human_batch_map()
    panel = panel_metrics(annotations, art_fields)
    quality = human_quality_distribution(annotations)
    qwk = individual_qwk(annotations)
    original_annotations = with_original_human_scores(annotations)
    original_panel = panel_metrics(original_annotations, art_fields)
    sensitivity = amendment_sensitivity(panel, original_panel)
    ba = bland_altman(annotations, art_fields, batch_map)
    drift = batch_drift(annotations, art_fields, batch_map)
    robustness = panel_drop_robustness(annotations, art_fields, batch_map)
    write_report(panel, quality, qwk, sensitivity, ba, drift, robustness)
    log.info("Wrote %s", REPORT_PATH.relative_to(ROOT))
    log.info("Human quality distribution:\n%s", quality.to_string(index=False))
    log.info("Panel table:\n%s", panel.to_string(index=False))
    log.info("Individual QWK:\n%s", qwk.to_string(index=False))
    log.info("Bland-Altman:\n%s", ba.to_string(index=False))
    log.info("Batch drift:\n%s", drift.to_string(index=False))
    log.info("Panel-drop robustness:\n%s", robustness.to_string(index=False))
    log.info("Amendment sensitivity:\n%s", sensitivity.to_string(index=False))


if __name__ == "__main__":
    main()
