"""Unified joint distribution + sampler spec builder.

Corresponds to Phase 02 Step 4-5. Converts T1~T7 into field-conditional
distributions and emits them in a form the NeMo Data Designer sampler chain can
use directly.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pak.config import settings
from pak.grounding.ipf import ipf as ipf_solve
from pak.grounding.marginals import (
    field_marginal_from_report,
)
from pak.grounding.tables import (
    AGE_BANDS_7,
    AGE_GROUP_4,
    CAREER_BANDS_5,
    EDUCATION_3,
    FIELDS_14,
    HOUSEHOLD_INCOME_9,
    INDIVIDUAL_INCOME_9,
    PROVINCES_18,
    SEXES_2,
)

_AGE_BAND_TO_4GROUP: dict[str, str] = {
    "10대": "30대 이하",
    "20대": "30대 이하",
    "30대": "30대 이하",
    "40대": "40대",
    "50대": "50대",
    "60대": "60세 이상",
    "70대 이상": "60세 이상",
}

logger = logging.getLogger(__name__)


def _conditional_table(
    df: pd.DataFrame, condition_col: str, value_col: str
) -> dict[str, dict[str, float]]:
    """Convert to P(value_col | condition_col) form.

    Returns: {field: {category: probability, ...}, ...}
    """
    out: dict[str, dict[str, float]] = {}
    for cond_value, sub in df.groupby(condition_col):
        total = float(sub["count"].sum())
        if total <= 0:
            continue
        out[str(cond_value)] = {
            str(row[value_col]): float(row["count"]) / total for _, row in sub.iterrows()
        }
    return out


def _conditional_t1_with_ipf() -> dict[str, dict[str, float]]:
    """T1: P(sex, age | field).

    T1 is already a (field, sex, age) population count, so it is itself a joint
    distribution. Normalize within each field to force the field marginal to the
    population (REPORT_FIELD_POPULATION).
    """
    df = pd.read_parquet(settings.grounding_dir / "T1.parquet")
    out: dict[str, dict[str, float]] = {}
    for f, sub in df.groupby("field"):
        # No IPF needed — T1 is already a 3-way joint. Just normalize per field.
        total = float(sub["count"].sum())
        if total <= 0:
            continue
        joint: dict[str, float] = {}
        for _, row in sub.iterrows():
            key = f"{row['sex']}|{row['age_band']}"
            joint[key] = float(row["count"]) / total
        out[str(f)] = joint
    return out


def _field_age4_probability_matrix() -> np.ndarray:
    """P(age_group_4 | field) matrix based on the T1 population."""
    fields = list(FIELDS_14)
    ages_4 = list(AGE_GROUP_4)
    t1 = pd.read_parquet(settings.grounding_dir / "T1.parquet")
    t1_proj = t1.copy()
    t1_proj["age_group_4"] = t1_proj["age_band"].map(_AGE_BAND_TO_4GROUP)
    if t1_proj["age_group_4"].isna().any():
        bad = t1_proj[t1_proj["age_group_4"].isna()]["age_band"].unique().tolist()
        raise ValueError(f"unknown age_band(s) in T1: {bad}")
    fa_pop = (
        t1_proj.groupby(["field", "age_group_4"], as_index=False)["count"]
        .sum()
        .pivot(index="field", columns="age_group_4", values="count")
        .fillna(0.0)
        .reindex(index=fields, columns=ages_4, fill_value=0.0)
        .values.astype(float)
    )
    field_pop_total = fa_pop.sum(axis=1, keepdims=True)
    return np.divide(
        fa_pop, field_pop_total, out=np.zeros_like(fa_pop), where=field_pop_total > 0
    )


def _conditional_matrix_from_table(
    table: pd.DataFrame,
    *,
    index_col: str,
    variable_col: str,
    index_values: list[Any],
    variable_categories: list[Any],
) -> np.ndarray:
    """Convert the table's probability/count into a per-index conditional probability matrix."""
    if table.empty:
        raise RuntimeError(f"empty table for {variable_col}")

    if "probability" in table.columns:
        grouped = (
            table.groupby([index_col, variable_col], as_index=False)["probability"]
            .sum()
            .pivot(index=index_col, columns=variable_col, values="probability")
            .fillna(0.0)
            .reindex(index=index_values, columns=variable_categories, fill_value=0.0)
            .values.astype(float)
        )
    else:
        grouped = (
            table.groupby([index_col, variable_col], as_index=False)["count"]
            .sum()
            .pivot(index=index_col, columns=variable_col, values="count")
            .fillna(0.0)
            .reindex(index=index_values, columns=variable_categories, fill_value=0.0)
            .values.astype(float)
        )
    row_total = grouped.sum(axis=1, keepdims=True)
    return np.divide(grouped, row_total, out=np.zeros_like(grouped), where=row_total > 0)


def _conditional_var_given_field_age4_via_ipf(
    *,
    field_var_table: pd.DataFrame,
    age_var_table: pd.DataFrame,
    variable_col: str,
    variable_categories: list[Any],
    field_total_by_field: dict[str, float] | None = None,
    diagnostic_label: str | None = None,
) -> tuple[dict[str, dict[Any, float]], dict[str, Any]]:
    """Compute P(variable | field, age_group_4) via 2-D IPF within each field.

    For each field f:
      seed = the report's age_group_4 × variable prior scaled to N_f
      row target = P(age_group_4 | f, T1 population) × N_f
      col target = P(variable | f, field_var_table) × N_f

    If ``field_total_by_field`` is given, that N is used instead of the subset N
    from field_var_table. This is a mechanism for tables whose denominator is a
    subset (such as 표 3-23/3-26): use only the report's conditional probability
    and take the per-field total N from a table based on N=5,059.
    """
    fields = list(FIELDS_14)
    ages_4 = list(AGE_GROUP_4)
    cats = list(variable_categories)

    p_age4_given_field = _field_age4_probability_matrix()
    p_var_given_field = _conditional_matrix_from_table(
        field_var_table,
        index_col="field",
        variable_col=variable_col,
        index_values=fields,
        variable_categories=cats,
    )

    if field_total_by_field is None:
        n_field = (
            field_var_table.groupby("field")["count"]
            .sum()
            .reindex(fields, fill_value=0.0)
            .values.astype(float)
        )
    else:
        n_field = np.asarray([float(field_total_by_field.get(f, 0.0)) for f in fields])

    age_var_counts = (
        age_var_table.groupby(["age_group_4", variable_col], as_index=False)["count"]
        .sum()
        .pivot(index="age_group_4", columns=variable_col, values="count")
        .fillna(0.0)
        .reindex(index=ages_4, columns=cats, fill_value=0.0)
        .values.astype(float)
    )
    age_var_total = float(age_var_counts.sum())
    if age_var_total <= 0:
        raise RuntimeError(f"{variable_col}: age_var_table has zero total")
    age_var_prior = age_var_counts / age_var_total

    arr = np.zeros((len(fields), len(ages_4), len(cats)), dtype=np.float64)
    info_per_field: list[dict[str, Any]] = []

    for fi, f in enumerate(fields):
        n_f = float(n_field[fi])
        if n_f <= 0:
            arr[fi] = 0.0
            info_per_field.append({"field": f, "N_f": 0, "skipped": True})
            continue

        target_age = p_age4_given_field[fi] * n_f
        target_var = p_var_given_field[fi] * n_f
        seed = np.where(age_var_prior * n_f == 0.0, 1e-12, age_var_prior * n_f)

        arr_f = seed.copy()
        max_err = float("inf")
        for it in range(1, 201):
            row = arr_f.sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio_r = np.where(row > 0, target_age / row, 0.0)
            arr_f = arr_f * ratio_r[:, None]

            col = arr_f.sum(axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio_c = np.where(col > 0, target_var / col, 0.0)
            arr_f = arr_f * ratio_c[None, :]

            row_err = float(
                np.max(
                    np.abs(arr_f.sum(axis=1) - target_age)
                    / np.where(target_age > 0, target_age, 1.0)
                )
            )
            col_err = float(
                np.max(
                    np.abs(arr_f.sum(axis=0) - target_var)
                    / np.where(target_var > 0, target_var, 1.0)
                )
            )
            max_err = max(row_err, col_err)
            if max_err < 1e-10:
                break
        if max_err >= 1e-8:
            raise RuntimeError(
                f"{variable_col} IPF did not converge for field={f}: "
                f"iterations={it}, max_rel_err={max_err:.3e}"
            )
        arr[fi] = arr_f
        info_per_field.append(
            {"field": f, "N_f": n_f, "iterations": it, "final_max_rel_err": max_err}
        )

    info = {
        "method": "per_field_2d_IPF",
        "variable": diagnostic_label or variable_col,
        "marginals_used": [
            "T1_field_age4 (population, within-field P(age4|field))",
            f"field_{variable_col} (report field sub-block conditional)",
            f"age4_{variable_col} (report age sub-block prior)",
        ],
        "field_total_basis": (
            "override_field_total_by_field" if field_total_by_field is not None else "field_var_table_row_sum"
        ),
        "fields": info_per_field,
        "global_max_rel_err": float(
            max(d.get("final_max_rel_err", 0.0) for d in info_per_field if not d.get("skipped"))
        ),
        "all_converged": all(
            d.get("final_max_rel_err", 1.0) < 1e-8
            for d in info_per_field
            if not d.get("skipped")
        ),
    }

    out: dict[str, dict[Any, float]] = {}
    for fi, f in enumerate(fields):
        for ai, age_group in enumerate(ages_4):
            slice_fa = arr[fi, ai, :]
            total = float(slice_fa.sum())
            if total <= 0:
                out[f"{f}|{age_group}"] = {cat: 0.0 for cat in cats}
                continue
            out[f"{f}|{age_group}"] = {
                cat: float(slice_fa[ci]) / total for ci, cat in enumerate(cats)
            }
    return out, info


def _conditional_career_given_field_age4_via_ipf() -> tuple[
    dict[str, dict[str, float]], dict[str, Any]
]:
    """Compute P(career_band | field, age_group_4) via IPF within each field.

    PAK design consistency:
    - P(field): population weights (REPORT_FIELD_POPULATION) — representative of the
      Korean artist population.
    - P(age|field): T1 (within-field age distribution from the population).
    - P(career|field): T4 (within-field career distribution from respondents).
    - field-independent (age4 × career) prior: T8.

    With these inputs, run 2-D IPF per field f:
        seed = T8 (age4 × career, respondent marginal scaled to field N_f)
        targets = [P(age4|f, population) × N_f, P(career|f, respondents) × N_f]
    → arr_f[a, c] = N(field=f, age=a, career=c). Satisfies both marginals within the field.

    Synthesis flow:
        field ~ population P(field)
        sex_age | field ~ T1 (population)
        age_group_4 = bucket(age_band)
        career | field, age_group_4 ~ arr_f[age, :] / arr_f[age, :].sum()

    This route gives:
      - synthetic (field × age_group_4) marginal = population (T1) ✓
      - synthetic within-field P(career|f) = respondents (T4) ✓
      - synthetic (age_group_4 × career) marginal ≈ T8 (not an exact match because the
        field weighting is the population, but informed by the T8 prior)

    The output dict keys are the ``"<field>|<age_group_4>"`` joint key.
    """
    fields = list(FIELDS_14)
    ages_4 = list(AGE_GROUP_4)
    careers = list(CAREER_BANDS_5)

    t1 = pd.read_parquet(settings.grounding_dir / "T1.parquet")
    t4 = pd.read_parquet(settings.grounding_dir / "T4.parquet")
    t8 = pd.read_parquet(settings.grounding_dir / "T8.parquet")

    # T1 → within-field population P(age4|field)
    t1_proj = t1.copy()
    t1_proj["age_group_4"] = t1_proj["age_band"].map(_AGE_BAND_TO_4GROUP)
    if t1_proj["age_group_4"].isna().any():
        bad = t1_proj[t1_proj["age_group_4"].isna()]["age_band"].unique().tolist()
        raise ValueError(f"unknown age_band(s) in T1: {bad}")
    fa_pop = (
        t1_proj.groupby(["field", "age_group_4"], as_index=False)["count"]
        .sum()
        .pivot(index="field", columns="age_group_4", values="count")
        .fillna(0.0)
        .reindex(index=fields, columns=ages_4, fill_value=0.0)
        .values.astype(float)
    )
    field_pop_total = fa_pop.sum(axis=1, keepdims=True)
    p_age4_given_field = np.divide(
        fa_pop, field_pop_total, out=np.zeros_like(fa_pop), where=field_pop_total > 0
    )

    # T4 → within-field respondent (career|field) marginal (counts)
    fc_resp = (
        t4.pivot(index="field", columns="career_band", values="count")
        .fillna(0.0)
        .reindex(index=fields, columns=careers, fill_value=0.0)
        .values.astype(float)
    )

    # T8 → field-independent respondent (age4 × career) prior. row=age, col=career.
    ac_resp = (
        t8.pivot(index="age_group_4", columns="career_band", values="count")
        .fillna(0.0)
        .reindex(index=ages_4, columns=careers, fill_value=0.0)
        .values.astype(float)
    )
    # normalize T8 into probabilities (so it can be scaled by per-field N_f)
    ac_total = ac_resp.sum()
    if ac_total <= 0:
        raise RuntimeError("T8 has zero total")
    ac_prob = ac_resp / ac_total

    n_field = fc_resp.sum(axis=1, keepdims=False)  # (14,)

    arr = np.zeros((len(fields), len(ages_4), len(careers)), dtype=np.float64)
    info_per_field: list[dict[str, Any]] = []

    for fi, f in enumerate(fields):
        N_f = float(n_field[fi])
        if N_f <= 0:
            arr[fi] = 0.0
            info_per_field.append({"field": f, "N_f": 0, "skipped": True})
            continue
        target_age = p_age4_given_field[fi] * N_f  # (4,)
        target_career = fc_resp[fi]  # (5,)
        # seed: T8 prior scaled to N_f
        seed = ac_prob * N_f  # (4, 5)
        seed = np.where(seed == 0, 1e-12, seed)

        # 2-D IPF: targets = [row(age) sum, col(career) sum]
        arr_f = seed.copy()
        max_err = float("inf")
        for it in range(1, 201):
            row = arr_f.sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio_r = np.where(row > 0, target_age / row, 0.0)
            arr_f = arr_f * ratio_r[:, None]
            col = arr_f.sum(axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio_c = np.where(col > 0, target_career / col, 0.0)
            arr_f = arr_f * ratio_c[None, :]
            row_err = float(
                np.max(
                    np.abs(arr_f.sum(axis=1) - target_age)
                    / np.where(target_age > 0, target_age, 1.0)
                )
            )
            col_err = float(
                np.max(
                    np.abs(arr_f.sum(axis=0) - target_career)
                    / np.where(target_career > 0, target_career, 1.0)
                )
            )
            max_err = max(row_err, col_err)
            if max_err < 1e-10:
                break
        arr[fi] = arr_f
        info_per_field.append(
            {"field": f, "N_f": N_f, "iterations": it, "final_max_rel_err": max_err}
        )

    info = {
        "method": "per_field_2d_IPF",
        "marginals_used": [
            "T1_field_age4 (population, within-field P(age4|field))",
            "T4_field_career (respondents, within-field P(career|field))",
            "T8_age4_career (field-independent prior; used as seed)",
        ],
        "fields": info_per_field,
        "global_max_rel_err": float(
            max(d.get("final_max_rel_err", 0.0) for d in info_per_field if not d.get("skipped"))
        ),
        "all_converged": all(
            d.get("final_max_rel_err", 1.0) < 1e-8
            for d in info_per_field
            if not d.get("skipped")
        ),
    }

    # extract P(career | field, age_group_4)
    out: dict[str, dict[str, float]] = {}
    for fi, f in enumerate(fields):
        for ai, a in enumerate(ages_4):
            slice_fa = arr[fi, ai, :]
            total = float(slice_fa.sum())
            if total <= 0:
                out[f"{f}|{a}"] = {c: 0.0 for c in careers}
                continue
            out[f"{f}|{a}"] = {
                c: float(slice_fa[ci]) / total for ci, c in enumerate(careers)
            }
    return out, info


def build_unified_distributions() -> dict[str, Any]:
    """Integrate the 7 tables into field-conditional joint distributions.

    Output structure:
        {
            "P_field": {field: prob, ...},     # based on the population
            "P_sex_age_given_field": {field: {"남자|30대": prob, ...}, ...},
            "P_province_given_field": {field: {province: prob, ...}, ...},
            "P_education_given_field": {...},
            "P_career_band_given_field": {...},
            "P_employment_freelance_given_field": {field: {"전업|True": prob, ...}, ...},
            "P_income_bracket_given_field": {...},
            "P_T7_given_field": {field: {var: {True: prob, False: prob}, ...}, ...},
        }
    """
    out: dict[str, Any] = {}

    # field marginal: based on the population
    out["P_field"] = field_marginal_from_report().to_dict()

    # T1: P(sex, age | field) — direct joint
    out["P_sex_age_given_field"] = _conditional_t1_with_ipf()

    # T2: P(province | field)
    t2 = pd.read_parquet(settings.grounding_dir / "T2.parquet")
    out["P_province_given_field"] = _conditional_table(t2, "field", "province")

    # T3: P(education | field) — already long-format with implied counts
    t3 = pd.read_parquet(settings.grounding_dir / "T3.parquet")
    out["P_education_given_field"] = _conditional_table(t3, "field", "education")

    # T4: P(career_band | field) — legacy 1-way conditional. Emitted alongside the
    # (field × career × age) 3-way joint for backward-compat.
    t4 = pd.read_parquet(settings.grounding_dir / "T4.parquet")
    out["P_career_band_given_field"] = _conditional_table(t4, "field", "career_band")

    # T1+T4+T8 IPF: P(career_band | field, age_group_4)
    p_career_joint, ipf_info = _conditional_career_given_field_age4_via_ipf()
    out["P_career_band_given_field_and_age4"] = p_career_joint
    out["_ipf_info_career"] = ipf_info

    # T5: P(employment, freelance | field) — preserve legacy joint categories
    t5 = pd.read_parquet(settings.grounding_dir / "T5.parquet")
    p_t5: dict[str, dict[str, float]] = {}
    for f, sub in t5.groupby("field"):
        total = float(sub["count"].sum())
        if total <= 0:
            continue
        p_t5[str(f)] = {
            f"{row['employment_type']}|{bool(row['is_freelance'])}": float(row["count"]) / total
            for _, row in sub.iterrows()
        }
    out["P_employment_freelance_given_field"] = p_t5

    # Split T5 into direct field conditionals:
    # - employment_type is age-jointed through T9.
    # - is_freelance remains conditioned on field + employment_type because no age cross exists.
    t5_emp = (
        t5.groupby(["field", "employment_type"], as_index=False)
        .agg(count=("count", "sum"), probability=("probability", "sum"))
        .reset_index(drop=True)
    )
    out["P_employment_type_given_field"] = _conditional_table(
        t5_emp, "field", "employment_type"
    )

    p_free: dict[str, dict[bool, float]] = {}
    for (f, emp), sub in t5.groupby(["field", "employment_type"]):
        total = float(sub["probability"].sum())
        if total <= 0:
            continue
        p_free[f"{f}|{emp}"] = {
            bool(row["is_freelance"]): float(row["probability"]) / total
            for _, row in sub.iterrows()
        }
    out["P_is_freelance_given_field_and_employment_type"] = p_free

    t9 = pd.read_parquet(settings.grounding_dir / "T9.parquet")
    p_employment_age, ipf_employment = _conditional_var_given_field_age4_via_ipf(
        field_var_table=t5_emp,
        age_var_table=t9,
        variable_col="employment_type",
        variable_categories=["전업", "겸업"],
        diagnostic_label="employment_type",
    )
    out["P_employment_type_given_field_and_age4"] = p_employment_age
    out["_ipf_info_employment_type"] = ipf_employment

    # T6: P(income_bracket | field)
    t6 = pd.read_parquet(settings.grounding_dir / "T6.parquet")
    out["P_income_bracket_given_field"] = _conditional_table(t6, "field", "income_bracket")
    t10 = pd.read_parquet(settings.grounding_dir / "T10.parquet")
    p_income_age, ipf_income = _conditional_var_given_field_age4_via_ipf(
        field_var_table=t6,
        age_var_table=t10,
        variable_col="income_bracket",
        variable_categories=list(INDIVIDUAL_INCOME_9),
        diagnostic_label="individual_art_income_bracket",
    )
    out["P_individual_art_income_bracket_given_field_and_age4"] = p_income_age
    out["_ipf_info_individual_art_income_bracket"] = ipf_income

    # T7: P(var=True | field) for each binary variable
    t7 = pd.read_parquet(settings.grounding_dir / "T7.parquet")
    p_t7: dict[str, dict[str, dict[bool, float]]] = {}
    for f, sub in t7.groupby("field"):
        var_dict: dict[str, dict[bool, float]] = {}
        for var, sub2 in sub.groupby("variable"):
            true_row = sub2[sub2["value"]]
            false_row = sub2[~sub2["value"]]
            if true_row.empty or false_row.empty:
                continue
            p_true = float(true_row.iloc[0]["probability"])
            var_dict[str(var)] = {True: p_true, False: 1 - p_true}
        if var_dict:
            p_t7[str(f)] = var_dict
    out["P_T7_given_field"] = p_t7

    t4_field_totals = t4.groupby("field")["count"].sum().to_dict()
    age_table_by_var = {
        "has_contract_experience": "T11",
        "uses_standard_contract": "T12",
        "has_copyright": "T13",
        "had_career_break": "T14",
        "has_overseas_experience": "T15",
    }
    for var, table_id in age_table_by_var.items():
        field_var = t7[t7["variable"] == var].copy()
        age_var = pd.read_parquet(settings.grounding_dir / f"{table_id}.parquet")
        field_totals = t4_field_totals if var in {"has_contract_experience", "uses_standard_contract"} else None
        p_var_age, ipf_var = _conditional_var_given_field_age4_via_ipf(
            field_var_table=field_var,
            age_var_table=age_var,
            variable_col="value",
            variable_categories=[True, False],
            field_total_by_field=field_totals,
            diagnostic_label=var,
        )
        out[f"P_{var}_given_field_and_age4"] = p_var_age
        out[f"_ipf_info_{var}"] = ipf_var

    return out


def write_unified_parquet(unified: dict[str, Any]) -> Path:
    """Also save the unified distributions as long-format parquet (for analysis/validation)."""
    rows: list[dict[str, Any]] = []
    for f, prob in unified["P_field"].items():
        rows.append({"distribution": "P_field", "field": f, "key": "", "probability": prob})

    for dist_name, _key_attr in [
        ("P_sex_age_given_field", "sex_age"),
        ("P_province_given_field", "province"),
        ("P_education_given_field", "education"),
        ("P_career_band_given_field", "career_band"),
        ("P_career_band_given_field_and_age4", "career_band"),
        ("P_employment_freelance_given_field", "employment_freelance"),
        ("P_employment_type_given_field", "employment_type"),
        ("P_employment_type_given_field_and_age4", "employment_type"),
        ("P_is_freelance_given_field_and_employment_type", "is_freelance"),
        ("P_income_bracket_given_field", "income_bracket"),
        ("P_individual_art_income_bracket_given_field_and_age4", "income_bracket"),
        ("P_has_contract_experience_given_field_and_age4", "value"),
        ("P_uses_standard_contract_given_field_and_age4", "value"),
        ("P_has_copyright_given_field_and_age4", "value"),
        ("P_had_career_break_given_field_and_age4", "value"),
        ("P_has_overseas_experience_given_field_and_age4", "value"),
    ]:
        if dist_name not in unified:
            continue
        block = unified[dist_name]
        for f, kv in block.items():
            for k, v in kv.items():
                rows.append(
                    {
                        "distribution": dist_name,
                        "field": str(f),
                        "key": str(k),
                        "probability": float(v),
                    }
                )

    for f, var_dict in unified["P_T7_given_field"].items():
        for var, tf in var_dict.items():
            for value, prob in tf.items():
                rows.append(
                    {
                        "distribution": "P_T7_given_field",
                        "field": f,
                        "key": f"{var}={value}",
                        "probability": float(prob),
                    }
                )

    df = pd.DataFrame(rows)
    out = settings.grounding_dir / "joint_distributions.parquet"
    df.to_parquet(out, index=False)
    logger.info("wrote %s (%d rows)", out, len(df))
    return out


def build_sampler_specs(unified: dict[str, Any]) -> dict[str, Any]:
    """Sampler chain definition that NeMo Data Designer can read."""
    field_probs = unified["P_field"]

    def _subcat(block: dict[str, dict[str, float]]) -> dict[str, dict[str, list[Any]]]:
        out: dict[str, dict[str, list[Any]]] = {}
        for f, kv in block.items():
            keys = list(kv.keys())
            weights = [kv[k] for k in keys]
            out[f] = {"values": keys, "weights": weights}
        return out

    samplers: list[dict[str, Any]] = [
        {
            "name": "art_field_primary",
            "type": "category",
            "values": list(field_probs.keys()),
            "weights": list(field_probs.values()),
        },
        {
            "name": "sex_age",
            "type": "subcategory",
            "parent": "art_field_primary",
            "subcategories": _subcat(unified["P_sex_age_given_field"]),
            "schema": {"keys": "sex|age_band"},
        },
        {
            "name": "province",
            "type": "subcategory",
            "parent": "art_field_primary",
            "subcategories": _subcat(unified["P_province_given_field"]),
        },
        {
            "name": "education_level",
            "type": "subcategory",
            "parent": "art_field_primary",
            "subcategories": _subcat(unified["P_education_given_field"]),
            "note": "Based on respondents N=5,059 (no population education data available)",
        },
        {
            "name": "field_age_group_4",
            "type": "derived",
            "sources": ["art_field_primary", "sex_age"],
            "transform": "field_age_group_4_join",
            "schema": {"keys": "art_field_primary|age_group_4"},
            "note": (
                "Joint parent key used in P(career_band | field, age_group_4). "
                "Derived from the art_field_primary and sex_age results (matches report cells directly)."
            ),
        },
        {
            "name": "career_band",
            "type": "subcategory",
            "parent": "field_age_group_4",
            "subcategories": _subcat(unified["P_career_band_given_field_and_age4"]),
            "note": (
                "Using T1·T4·T8 (all from report p.21/p.55) and generalized IPF, estimate the "
                "(field × age4 × career) 3-way joint, then take the P(career_band | field, age_group_4) "
                "conditional. The previous 1-way conditional (P(career|field)) is preserved in "
                "P_career_band_given_field."
            ),
        },
        {
            "name": "employment_type",
            "type": "subcategory",
            "parent": "field_age_group_4",
            "subcategories": _subcat(unified["P_employment_type_given_field_and_age4"]),
            "note": (
                "Compute P(employment_type | field, age_group_4) via IPF based on T5/T9. "
                "Freelance status is conditioned on field+employment_type in the separate is_freelance sampler."
            ),
        },
        {
            "name": "field_employment",
            "type": "derived",
            "sources": ["art_field_primary", "employment_type"],
            "transform": "field_employment_join",
            "schema": {"keys": "art_field_primary|employment_type"},
        },
        {
            "name": "is_freelance",
            "type": "subcategory",
            "parent": "field_employment",
            "subcategories": _subcat(unified["P_is_freelance_given_field_and_employment_type"]),
            "note": "표 3-20/3-21 have no age cross, so P(is_freelance | field, employment_type) is kept.",
        },
        {
            "name": "individual_art_income_bracket",
            "type": "subcategory",
            "parent": "field_age_group_4",
            "subcategories": _subcat(
                unified["P_individual_art_income_bracket_given_field_and_age4"]
            ),
        },
    ]

    # T7 5 binary variables — all use the field+age4 joint parent
    for var in [
        "has_contract_experience",
        "uses_standard_contract",
        "has_copyright",
        "had_career_break",
        "has_overseas_experience",
    ]:
        samplers.append(
            {
                "name": var,
                "type": "subcategory",
                "parent": "field_age_group_4",
                "subcategories": _subcat(unified[f"P_{var}_given_field_and_age4"]),
            }
        )

    return {
        "version": "0.3.0",
        "generated_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        "field_taxonomy": list(FIELDS_14),
        "enums": {
            "sex": list(SEXES_2),
            "age_band": list(AGE_BANDS_7),
            "age_group_4": list(AGE_GROUP_4),
            "province": list(PROVINCES_18),
            "education": list(EDUCATION_3),
            "career_band": list(CAREER_BANDS_5),
            "individual_income_bracket": list(INDIVIDUAL_INCOME_9),
            "household_income_bracket": list(HOUSEHOLD_INCOME_9),
        },
        "samplers": samplers,
        "ipf_diagnostics": {
            "career_joint": unified.get("_ipf_info_career"),
            "employment_type_joint": unified.get("_ipf_info_employment_type"),
            "individual_art_income_bracket_joint": unified.get(
                "_ipf_info_individual_art_income_bracket"
            ),
            "has_contract_experience_joint": unified.get("_ipf_info_has_contract_experience"),
            "uses_standard_contract_joint": unified.get("_ipf_info_uses_standard_contract"),
            "has_copyright_joint": unified.get("_ipf_info_has_copyright"),
            "had_career_break_joint": unified.get("_ipf_info_had_career_break"),
            "has_overseas_experience_joint": unified.get("_ipf_info_has_overseas_experience"),
        },
    }


def write_all() -> dict[str, Path]:
    unified = build_unified_distributions()
    parquet = write_unified_parquet(unified)
    spec = build_sampler_specs(unified)
    spec_path = settings.grounding_dir / "sampler_specs.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("wrote %s (%d samplers)", spec_path, len(spec["samplers"]))
    return {"joint_parquet": parquet, "sampler_specs": spec_path}


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    write_all()
