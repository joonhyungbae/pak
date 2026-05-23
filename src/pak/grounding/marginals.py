"""Marginal distribution extraction and cross-validation.

Compare the population P(field) from T1, T2 with the respondent-based P(field)
from T3~T7.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pak.config import settings

logger = logging.getLogger(__name__)


# Quote from the report body: 「3) 표본추출틀 ... 모집단은 ... 334,036명」
REPORT_POPULATION_TOTAL = 334_036
REPORT_RESPONDENTS_TOTAL = 5_059

# Per-field population from report tables 표 1-3 / 1-4 / 1-5 (values verified to fully match T2)
REPORT_FIELD_POPULATION: dict[str, int] = {
    "문학": 30_401,
    "미술": 81_062,
    "공예": 6_245,
    "사진": 15_560,
    "건축": 5_633,
    "음악": 37_221,
    "국악": 12_195,
    "대중음악": 58_452,
    "방송연예": 31_388,
    "무용": 7_887,
    "연극": 26_876,
    "영화": 9_152,
    "만화": 4_183,
    "기타": 7_781,
}


def _field_marginal(df: pd.DataFrame, count_col: str = "count") -> pd.Series:
    if "field" not in df.columns or count_col not in df.columns:
        return pd.Series(dtype=float)
    s = df.groupby("field")[count_col].sum()
    if s.sum() == 0:
        return s.astype(float)
    return s / s.sum()


def field_marginal_from_report() -> pd.Series:
    s = pd.Series(REPORT_FIELD_POPULATION, name="probability", dtype=float)
    return s / s.sum()


def cross_check() -> pd.DataFrame:
    """Compare each P(field) marginal from T1, T2, T3~T7 against the report population."""
    rows: list[dict[str, Any]] = []
    report = field_marginal_from_report()

    for tid in ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]:
        path = settings.grounding_dir / f"{tid}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        marginal = _field_marginal(df)
        for f in REPORT_FIELD_POPULATION:
            obs = float(marginal.get(f, 0.0))
            exp = float(report.get(f, 0.0))
            rows.append(
                {
                    "table": tid,
                    "field": f,
                    "P_field_observed": obs,
                    "P_field_report": exp,
                    "abs_diff": abs(obs - exp),
                    "rel_diff_pct": (obs - exp) / exp * 100 if exp > 0 else None,
                }
            )

    return pd.DataFrame(rows)


def write_marginal_report() -> Path:
    df = cross_check()
    out_dir = settings.grounding_dir / "marginals"
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "field_marginal_crosscheck.parquet"
    df.to_parquet(parquet_path, index=False)

    summary: dict[str, Any] = {
        "generated_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        "report_population_total": REPORT_POPULATION_TOTAL,
        "report_respondents_total": REPORT_RESPONDENTS_TOTAL,
        "tables_checked": sorted(df["table"].unique().tolist()),
        "max_abs_diff_per_table": (df.groupby("table")["abs_diff"].max().to_dict()),
        "max_rel_diff_pct_per_table": (
            df.groupby("table")["rel_diff_pct"].apply(lambda s: s.abs().max()).to_dict()
        ),
    }
    summary_path = out_dir / "field_marginal_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("marginal cross-check: %s, %s", parquet_path, summary_path)
    return parquet_path


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    write_marginal_report()
    df = cross_check()
    print(df.pivot(index="field", columns="table", values="P_field_observed").round(4))
