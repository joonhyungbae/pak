"""Leave-one-table-out external validation: hold out T8 (age_group_4 x career_band).

Tests whether PAK's per-field max-entropy (IPF) reconstruction recovers the real
age-by-career joint WITHOUT using T8 as a constraint. With T8 removed, the only
inputs are P(age_group_4 | field) from T1 (population) and P(career_band | field)
from T4 (respondent), so within each field career is independent of age. Cross-
field aggregation by population field weights can still induce an age-career
association. We compare this holdout reconstruction R to the held-out T8 table.

T8 is the one age-conditional table whose information is not also pinned by a
field-level table (T9-T15 are additionally constrained by T4/T6/T7, so dropping
them is near-tautological). For reference we also report M, a pure marginal-
independence reconstruction P(age) (x) P(career) built from T8's own marginals;
R closer to T8 than M shows that field composition explains part of the age-
career association, and the residual R-vs-T8 is the within-field structure that
only T8 supplies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from pak.grounding.joint import _AGE_BAND_TO_4GROUP  # noqa: E402

G = ROOT / "data/grounding"
REPORT = ROOT / "outputs/reports/holdout_t8_validation_260522.md"


def main() -> None:
    t1 = pd.read_parquet(G / "T1.parquet")
    t4 = pd.read_parquet(G / "T4.parquet")
    t8 = pd.read_parquet(G / "T8.parquet")

    t1 = t1.copy()
    t1["age_group_4"] = t1["age_band"].map(_AGE_BAND_TO_4GROUP)
    fa = t1.groupby(["field", "age_group_4"])["count"].sum().unstack(fill_value=0.0)
    field_pop = fa.sum(axis=1)
    w_f = field_pop / field_pop.sum()
    p_age_given_f = fa.div(fa.sum(axis=1), axis=0)

    fc = t4.groupby(["field", "career_band"])["count"].sum().unstack(fill_value=0.0)
    p_career_given_f = fc.div(fc.sum(axis=1), axis=0)

    age_levels = list(p_age_given_f.columns)
    career_levels = list(p_career_given_f.columns)
    fields = [f for f in w_f.index if f in p_career_given_f.index]

    # Holdout reconstruction R = sum_f w_f P(age|f) (x) P(career|f)
    R = np.zeros((len(age_levels), len(career_levels)))
    for f in fields:
        pa = p_age_given_f.loc[f, age_levels].to_numpy().reshape(-1, 1)
        pc = p_career_given_f.loc[f, career_levels].to_numpy().reshape(1, -1)
        R += float(w_f[f]) * (pa @ pc)
    R = R / R.sum()
    R = pd.DataFrame(R, index=age_levels, columns=career_levels)

    # Held-out target T8
    t8p = t8.groupby(["age_group_4", "career_band"])["count"].sum().unstack(fill_value=0.0)
    t8p = t8p.reindex(index=age_levels, columns=career_levels, fill_value=0.0)
    t8p = t8p / t8p.to_numpy().sum()

    # Pure marginal-independence reference M from T8's own marginals
    pa_t8 = t8p.sum(axis=1).to_numpy().reshape(-1, 1)
    pc_t8 = t8p.sum(axis=0).to_numpy().reshape(1, -1)
    M = pd.DataFrame(pa_t8 @ pc_t8, index=age_levels, columns=career_levels)

    def metrics(approx: pd.DataFrame, target: pd.DataFrame) -> dict[str, float]:
        d = (approx - target).abs()
        return {
            "total_variation": float(0.5 * d.to_numpy().sum()),
            "max_abs_cell_dev_pp": float(d.to_numpy().max() * 100),
            "mean_abs_cell_dev_pp": float(d.to_numpy().mean() * 100),
        }

    mR = metrics(R, t8p)
    mM = metrics(M, t8p)

    lines = [
        "# Holdout external validation: T8 (age_group_4 x career_band)",
        "",
        "T8 removed from the grounding. R is the per-field max-entropy reconstruction",
        "(within-field age-career independence, population field weights). M is pure",
        "marginal independence from T8's own marginals. Lower is closer to the held-out T8.",
        "",
        "| Reconstruction | Total variation | Max abs cell dev (pp) | Mean abs cell dev (pp) |",
        "|---|---|---|---|",
        f"| R (field comp + within-field independence) | {mR['total_variation']:.4f} | {mR['max_abs_cell_dev_pp']:.2f} | {mR['mean_abs_cell_dev_pp']:.2f} |",
        f"| M (pure marginal independence) | {mM['total_variation']:.4f} | {mM['max_abs_cell_dev_pp']:.2f} | {mM['mean_abs_cell_dev_pp']:.2f} |",
        "",
        "## Held-out T8 (target) probabilities",
        "",
        t8p.to_markdown(floatfmt=".4f"),
        "",
        "## R holdout reconstruction",
        "",
        R.to_markdown(floatfmt=".4f"),
        "",
        "## Per-cell absolute deviation R vs T8 (pp)",
        "",
        ((R - t8p).abs() * 100).to_markdown(floatfmt=".2f"),
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT.relative_to(ROOT))
    print("R vs T8:", mR)
    print("M vs T8:", mM)
    print("\nT8 target:\n", t8p.round(4).to_string())
    print("\nR holdout:\n", R.round(4).to_string())


if __name__ == "__main__":
    main()
