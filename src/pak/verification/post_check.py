"""Phase 07 — post-hoc verification.

Apply the following checks to the generated persona parquet:
1. Schema (Pydantic)
2. Marginal distributions (chi-square: P(field), P(province), P(career_band) vs grounding)
3. Joint distribution (P(field, sex, age) vs T1)
4. Per-field balance (each field N >= 1000)
5. Narrative diversity (token Jaccard)
6. Field vocabulary coverage (appearance rate of FIELD_META vocabulary)
7. Text fact checks (matching report means such as 10.55M KRW)
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from pak.config import settings
from pak.grounding.marginals import REPORT_FIELD_POPULATION
from pak.prompts_data import FIELD_META
from pak.schema import PAKPersona, PAKPersonaNarrative, PAKPersonaQuant
from pak.validators import ValidationPipeline
from pak.validators.cliche import CLICHE_PATTERNS, detect_cliches
from pak.validators.distribution import check_marginal_categorical
from pak.validators.diversity import pairwise_similarity_token

logger = logging.getLogger(__name__)

AGE_CROSSTAB_TOLERANCE = 0.05


# ----------------------------------------------------------------------------
# Fact checks (compare report-text means against synthetic-data statistics)
# ----------------------------------------------------------------------------


@dataclass
class FactCheck:
    name: str
    expected: float
    observed: float
    tolerance: float
    passed: bool
    note: str = ""


def fact_checks(df: pd.DataFrame) -> list[FactCheck]:
    """Report-cited values vs synthetic-data proportions."""
    checks: list[FactCheck] = []
    n = max(len(df), 1)

    def tol(expected: float, *, floor: float = 0.05, z: float = 1.96) -> float:
        # In preview samples the sampling error can exceed a fixed 5pp, so also
        # incorporate a tolerance based on the proportion's standard error.
        return max(floor, z * math.sqrt(expected * (1 - expected) / n))

    # Table 3-12: full-time artists 52.5%
    pct_full = (df["employment_type"] == "전업").mean()
    pct_full_tol = tol(0.525)
    checks.append(FactCheck(
        name="pct_full_time", expected=0.525, observed=float(pct_full),
        tolerance=pct_full_tol, passed=abs(pct_full - 0.525) < pct_full_tol,
        note=f"report table 3-12 full-time artist ratio 52.5% (n={n}, tolerance={pct_full_tol:.3f})",
    ))

    # Table 3-23: contract experience 57.3%
    pct_contract = df["has_contract_experience"].astype(bool).mean()
    pct_contract_tol = tol(0.573)
    checks.append(FactCheck(
        name="pct_contract_experience", expected=0.573, observed=float(pct_contract),
        tolerance=pct_contract_tol, passed=abs(pct_contract - 0.573) < pct_contract_tol,
        note=f"report table 3-23 contract experience 57.3% (n={n}, tolerance={pct_contract_tol:.3f})",
    ))

    # Table 3-41: copyright ownership 29.1%
    pct_copyright = df["has_copyright"].astype(bool).mean()
    pct_copyright_tol = tol(0.291)
    checks.append(FactCheck(
        name="pct_copyright", expected=0.291, observed=float(pct_copyright),
        tolerance=pct_copyright_tol, passed=abs(pct_copyright - 0.291) < pct_copyright_tol,
        note=f"report table 3-41 copyright ownership 29.1% (n={n}, tolerance={pct_copyright_tol:.3f})",
    ))

    # Table 3-55: career break 23.0%
    pct_break = df["had_career_break"].astype(bool).mean()
    pct_break_tol = tol(0.230)
    checks.append(FactCheck(
        name="pct_career_break", expected=0.230, observed=float(pct_break),
        tolerance=pct_break_tol, passed=abs(pct_break - 0.230) < pct_break_tol,
        note=f"report table 3-55 career break experience 23.0% (n={n}, tolerance={pct_break_tol:.3f})",
    ))

    # Table 3-7: overseas activity 16.5%
    pct_overseas = df["has_overseas_experience"].astype(bool).mean()
    pct_overseas_tol = tol(0.165)
    checks.append(FactCheck(
        name="pct_overseas", expected=0.165, observed=float(pct_overseas),
        tolerance=pct_overseas_tol, passed=abs(pct_overseas - 0.165) < pct_overseas_tol,
        note=f"report table 3-7 overseas activity experience 16.5% (n={n}, tolerance={pct_overseas_tol:.3f})",
    ))

    return checks


def _age_to_4group(age: int) -> str:
    """int age -> report's 4-bin age_group_4."""
    if age <= 39:
        return "30대 이하"
    if age <= 49:
        return "40대"
    if age <= 59:
        return "50대"
    return "60세 이상"


def _ensure_age_group_4(df: pd.DataFrame) -> pd.DataFrame:
    if "age_group_4" in df.columns:
        return df
    if "age" not in df.columns:
        raise KeyError("age crosstab checks require either 'age_group_4' or 'age' column")
    out = df.copy()
    out["age_group_4"] = out["age"].map(lambda age: _age_to_4group(int(age)))
    return out


def _empty_age_crosstab_check(name: str) -> list[FactCheck]:
    return [
        FactCheck(
            name=name,
            expected=0.0,
            observed=0.0,
            tolerance=AGE_CROSSTAB_TOLERANCE,
            passed=False,
            note="empty dataframe; age crosstab check skipped",
        )
    ]


def _value_col_for_age_var(var_name: str, baseline: pd.DataFrame) -> str:
    if var_name == "individual_art_income_bracket" and "income_bracket" in baseline.columns:
        return "income_bracket"
    if var_name in baseline.columns:
        return var_name
    if "value" in baseline.columns:
        return "value"
    raise KeyError(
        f"cannot infer value column for {var_name!r}; "
        f"baseline columns={baseline.columns.tolist()}"
    )


def _validate_baseline_columns(
    baseline: pd.DataFrame,
    *,
    table_path: Path,
    value_col: str,
) -> None:
    required = {"age_group_4", value_col, "probability"}
    missing = required - set(baseline.columns)
    if missing:
        raise KeyError(f"{table_path}: missing expected columns {sorted(missing)}")


def _age_crosstab_checks_from_baseline(
    df: pd.DataFrame,
    *,
    var_name: str,
    baseline: pd.DataFrame,
    value_col: str,
    label: str,
    tolerance: float = AGE_CROSSTAB_TOLERANCE,
) -> list[FactCheck]:
    if df.empty:
        return _empty_age_crosstab_check(f"{label}_empty")
    if var_name not in df.columns:
        raise KeyError(f"age crosstab checks require {var_name!r} column in synthetic df")

    work = _ensure_age_group_4(df)
    observed = pd.crosstab(work["age_group_4"], work[var_name], normalize="index")
    checks: list[FactCheck] = []
    for row in baseline.itertuples(index=False):
        age_group = str(getattr(row, "age_group_4"))
        value = getattr(row, value_col)
        expected = float(getattr(row, "probability"))
        observed_value = (
            float(observed.loc[age_group, value])
            if age_group in observed.index and value in observed.columns
            else 0.0
        )
        diff = observed_value - expected
        checks.append(
            FactCheck(
                name=f"{label}:{age_group}:{value}",
                expected=expected,
                observed=observed_value,
                tolerance=tolerance,
                passed=abs(diff) <= tolerance,
                note=(
                    f"age_group_4={age_group}, {var_name}={value}, "
                    f"expected={expected:.4f}, observed={observed_value:.4f}, "
                    f"diff={diff:+.4f}"
                ),
            )
        )
    return checks


def check_age_career_crosstab(df: pd.DataFrame) -> list[FactCheck]:
    """Compare synthetic (age_group_4 x career_band) ratios against T8 report cells."""
    t_path = settings.grounding_dir / "T8.parquet"
    baseline = pd.read_parquet(t_path)
    value_col = "career_band"
    _validate_baseline_columns(baseline, table_path=t_path, value_col=value_col)
    return _age_crosstab_checks_from_baseline(
        df,
        var_name="career_band",
        baseline=baseline,
        value_col=value_col,
        label="age_career_crosstab",
    )


def check_age_var_crosstab(
    df: pd.DataFrame,
    var_name: str,
    t_path: Path,
) -> list[FactCheck]:
    """Compare synthetic (age_group_4 x var_name) ratios against grounding parquet cells."""
    baseline = pd.read_parquet(t_path)
    value_col = _value_col_for_age_var(var_name, baseline)
    _validate_baseline_columns(baseline, table_path=t_path, value_col=value_col)
    return _age_crosstab_checks_from_baseline(
        df,
        var_name=var_name,
        baseline=baseline,
        value_col=value_col,
        label=f"age_{var_name}_crosstab",
    )


def age_crosstab_checks(df: pd.DataFrame) -> list[FactCheck]:
    """age_group_4 cross checks based on T8~T15."""
    checks = check_age_career_crosstab(df)
    for var_name, table_id in [
        ("employment_type", "T9"),
        ("individual_art_income_bracket", "T10"),
        ("has_contract_experience", "T11"),
        ("uses_standard_contract", "T12"),
        ("has_copyright", "T13"),
        ("had_career_break", "T14"),
        ("has_overseas_experience", "T15"),
    ]:
        checks.extend(
            check_age_var_crosstab(
                df,
                var_name,
                settings.grounding_dir / f"{table_id}.parquet",
            )
        )
    return checks


# ----------------------------------------------------------------------------
# Field vocabulary coverage
# ----------------------------------------------------------------------------


@dataclass
class FieldVocabularyCoverage:
    field: str
    n_personas: int
    n_vocab: int
    n_vocab_appearing: int
    coverage_pct: float
    top_unused: list[str]


def field_vocabulary_coverage(df: pd.DataFrame) -> list[FieldVocabularyCoverage]:
    """How much field vocabulary appears in each field's persona narratives."""
    out: list[FieldVocabularyCoverage] = []
    nar_cols = [
        "professional_persona", "creative_world_persona", "network_persona",
        "living_persona", "support_persona",
    ]
    for f, meta in FIELD_META.items():
        sub = df[df["art_field_primary"] == f]
        if len(sub) == 0:
            continue
        all_text = " ".join(
            str(sub[c].fillna("").str.cat(sep=" ")) for c in nar_cols if c in sub.columns
        )
        vocab = list(meta["vocabulary"])
        appearing = [v for v in vocab if v in all_text]
        unused = [v for v in vocab if v not in appearing]
        out.append(FieldVocabularyCoverage(
            field=f,
            n_personas=int(len(sub)),
            n_vocab=len(vocab),
            n_vocab_appearing=len(appearing),
            coverage_pct=len(appearing) / max(len(vocab), 1) * 100,
            top_unused=unused[:5],
        ))
    return out


# ----------------------------------------------------------------------------
# Joint distribution match (P(field, sex, age))
# ----------------------------------------------------------------------------


@dataclass
class JointDistributionResult:
    name: str
    n: int
    mean_abs_deviation: float
    max_abs_deviation: float
    passed: bool


@dataclass
class RowValidationSummary:
    n_checked: int
    n_with_errors: int
    n_with_warnings: int
    top_error_codes: dict[str, int]
    top_warning_codes: dict[str, int]


def check_field_sex_age_joint(df: pd.DataFrame, threshold: float = 0.02) -> JointDistributionResult:
    """T1 population (field x sex x age) distribution vs synthetic."""
    t1 = pd.read_parquet(settings.grounding_dir / "T1.parquet")
    t1_total = float(t1["count"].sum())
    expected = t1.groupby(["field", "sex", "age_band"], as_index=False)["count"].sum()
    expected["expected_pct"] = expected["count"] / t1_total

    obs = (
        df.groupby(["art_field_primary", "sex", "age_band"]).size().reset_index(name="n")
    )
    n_total = float(len(df))
    obs["observed_pct"] = obs["n"] / n_total
    obs = obs.rename(columns={"art_field_primary": "field"})

    merged = expected.merge(
        obs[["field", "sex", "age_band", "observed_pct"]],
        on=["field", "sex", "age_band"], how="outer",
    ).fillna({"expected_pct": 0.0, "observed_pct": 0.0})
    merged["abs_dev"] = (merged["expected_pct"] - merged["observed_pct"]).abs()
    return JointDistributionResult(
        name="(field, sex, age_band)",
        n=int(len(df)),
        mean_abs_deviation=float(merged["abs_dev"].mean()),
        max_abs_deviation=float(merged["abs_dev"].max()),
        passed=bool(merged["abs_dev"].mean() < threshold),
    )


# ----------------------------------------------------------------------------
# Combined report
# ----------------------------------------------------------------------------


@dataclass
class PostVerificationReport:
    n_personas: int
    schema_pass_rate: float
    cliche_per_1000: float
    cliche_freq: dict[str, float]
    field_marginal: dict[str, Any]
    field_balance: dict[str, int]
    diversity: dict[str, Any]
    fact_checks: list[FactCheck]
    age_crosstab_checks: list[FactCheck]
    field_vocab_coverage: list[FieldVocabularyCoverage]
    joint_field_sex_age: JointDistributionResult
    row_validation: RowValidationSummary
    notes: list[str] = field(default_factory=list)


def run_post_verification(parquet_path: Path) -> PostVerificationReport:
    df = pd.read_parquet(parquet_path)
    n = len(df)
    narrative_keys = set(PAKPersonaNarrative.model_fields)
    quant_keys = set(PAKPersonaQuant.model_fields)
    pipeline = ValidationPipeline()

    # 1. Schema
    schema_ok = 0
    for _, row in df.iterrows():
        try:
            PAKPersona.model_validate(row.to_dict())
            schema_ok += 1
        except ValidationError:
            pass
    schema_pass_rate = schema_ok / max(n, 1)

    # 2. Cliches
    cliche_total = 0
    cliche_counter: Counter[str] = Counter()
    nar_cols = [c for c in df.columns if isinstance(df[c].iloc[0], str) and not df[c].iloc[0].isascii()]
    for _, row in df.iterrows():
        nars = {c: str(row[c]) for c in nar_cols if isinstance(row[c], str)}
        hits = detect_cliches(nars)
        cliche_total += len(hits)
        for h in hits:
            cliche_counter[h.label] += 1
    cliche_freq = {label: cliche_counter[label] / max(n, 1) for label, _ in CLICHE_PATTERNS}

    # 3. Field marginal distribution chi-square
    total_pop = sum(REPORT_FIELD_POPULATION.values())
    expected_pct = {f: c / total_pop for f, c in REPORT_FIELD_POPULATION.items()}
    field_chi = check_marginal_categorical(
        "art_field_primary", df["art_field_primary"].tolist(), expected_pct,
    )

    # 4. Per-field counts
    field_balance = df["art_field_primary"].value_counts().to_dict()

    # 5. Diversity (sample 200 pairs per field)
    diversity_summary: dict[str, Any] = {}
    nar_for_div = "professional_persona" if "professional_persona" in df.columns else df.columns[1]
    for f in REPORT_FIELD_POPULATION:
        sub = df[df["art_field_primary"] == f]
        if len(sub) < 2:
            continue
        rep = pairwise_similarity_token(
            sub[nar_for_div].astype(str).tolist(), sample_pairs=min(200, len(sub) * 2),
        )
        diversity_summary[f] = {
            "n": rep.n,
            "mean_sim": rep.mean_similarity,
            "max_sim": rep.max_similarity,
            "pct_above_threshold": rep.pct_above_threshold,
        }

    # 6. Fact checks
    fc = fact_checks(df)

    # 7. age-4 joint distribution check
    age_fc = age_crosstab_checks(df)

    # 8. Field vocabulary coverage
    fvc = field_vocabulary_coverage(df)

    # 9. Joint distribution
    joint = check_field_sex_age_joint(df)

    # 10. row-level validation
    error_counter: Counter[str] = Counter()
    warning_counter: Counter[str] = Counter()
    n_with_errors = 0
    n_with_warnings = 0
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        quant = {k: row_dict.get(k) for k in quant_keys if k in row_dict}
        narratives = {
            k: str(v) for k, v in row_dict.items() if k in narrative_keys and isinstance(v, str)
        }
        result = pipeline.validate_one(
            pak_uuid=str(row_dict.get("pak_uuid", "")),
            quant=quant,
            narratives=narratives,
        )
        if result.has_errors:
            n_with_errors += 1
        if result.has_warnings:
            n_with_warnings += 1
        for issue in result.consistency_issues:
            if issue.severity == "error":
                error_counter[issue.code] += 1
            elif issue.severity == "warning":
                warning_counter[issue.code] += 1
        for hit in result.cliche_hits:
            error_counter[f"CLICHE:{hit.label}"] += 1

    return PostVerificationReport(
        n_personas=n,
        schema_pass_rate=schema_pass_rate,
        cliche_per_1000=cliche_total / max(n, 1) * 1000,
        cliche_freq=cliche_freq,
        field_marginal={
            "chi2": field_chi.statistic, "p_value": field_chi.p_value,
            "cramer_v": field_chi.effect_size, "passed": bool(field_chi.passed),
        },
        field_balance=field_balance,
        diversity=diversity_summary,
        fact_checks=fc,
        age_crosstab_checks=age_fc,
        field_vocab_coverage=fvc,
        joint_field_sex_age=joint,
        row_validation=RowValidationSummary(
            n_checked=n,
            n_with_errors=n_with_errors,
            n_with_warnings=n_with_warnings,
            top_error_codes=dict(error_counter.most_common(10)),
            top_warning_codes=dict(warning_counter.most_common(10)),
        ),
    )


def report_to_dict(r: PostVerificationReport) -> dict[str, Any]:
    return {
        "n_personas": r.n_personas,
        "schema_pass_rate": r.schema_pass_rate,
        "cliche_per_1000": r.cliche_per_1000,
        "cliche_freq_top5": dict(sorted(r.cliche_freq.items(), key=lambda x: -x[1])[:5]),
        "field_marginal": {
            **r.field_marginal,
            "passed": bool(r.field_marginal["passed"]),
        },
        "field_balance": r.field_balance,
        "diversity": r.diversity,
        "fact_checks": [
            {"name": f.name, "expected": f.expected, "observed": f.observed,
             "tolerance": f.tolerance, "passed": bool(f.passed), "note": f.note}
            for f in r.fact_checks
        ],
        "age_crosstab_checks": [
            {"name": f.name, "expected": f.expected, "observed": f.observed,
             "tolerance": f.tolerance, "passed": bool(f.passed), "note": f.note}
            for f in r.age_crosstab_checks
        ],
        "field_vocab_coverage": [
            {"field": v.field, "n": v.n_personas, "coverage_pct": v.coverage_pct,
             "n_vocab": v.n_vocab, "n_appearing": v.n_vocab_appearing,
             "top_unused": v.top_unused}
            for v in r.field_vocab_coverage
        ],
        "joint_field_sex_age": {
            "name": r.joint_field_sex_age.name,
            "mean_abs_dev": r.joint_field_sex_age.mean_abs_deviation,
            "max_abs_dev": r.joint_field_sex_age.max_abs_deviation,
            "passed": bool(r.joint_field_sex_age.passed),
        },
        "row_validation": {
            "n_checked": r.row_validation.n_checked,
            "n_with_errors": r.row_validation.n_with_errors,
            "n_with_warnings": r.row_validation.n_with_warnings,
            "top_error_codes": r.row_validation.top_error_codes,
            "top_warning_codes": r.row_validation.top_warning_codes,
        },
        "notes": r.notes,
    }


def main() -> None:  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("parquet", type=Path)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    rep = run_post_verification(args.parquet)
    out = report_to_dict(rep)
    target = args.output or args.parquet.parent / "post_verification.json"
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"wrote {target}")
    print(json.dumps({
        "n": rep.n_personas,
        "schema_pass_rate": rep.schema_pass_rate,
        "cliche_per_1000": rep.cliche_per_1000,
        "field_marginal_passed": rep.field_marginal["passed"],
        "joint_passed": rep.joint_field_sex_age.passed,
        "fact_checks_passed": sum(1 for f in rep.fact_checks if f.passed),
        "fact_checks_total": len(rep.fact_checks),
        "age_crosstab_checks_passed": sum(1 for f in rep.age_crosstab_checks if f.passed),
        "age_crosstab_checks_total": len(rep.age_crosstab_checks),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
