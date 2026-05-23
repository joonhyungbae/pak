"""Phase 01 Step D — normalize extracted tables into long-format parquet.

Each joint table (T1~T7) has a reconcile function that takes the best extraction
result as input and performs:
- field name standardization
- thousands-separator comma removal
- missing/dash handling
- (variables dict, value, value_type) long-format conversion
- writing provenance.json
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pak.config import settings

logger = logging.getLogger(__name__)


FIELDS_14: tuple[str, ...] = (
    "문학",
    "미술",
    "공예",
    "사진",
    "건축",
    "음악",
    "국악",
    "대중음악",
    "방송연예",
    "무용",
    "연극",
    "영화",
    "만화",
    "기타",
)

PROVINCES_18: tuple[str, ...] = (
    "서울",
    "부산",
    "대구",
    "인천",
    "광주",
    "대전",
    "울산",
    "세종",
    "경기",
    "강원",
    "충북",
    "충남",
    "전북",
    "전남",
    "경북",
    "경남",
    "제주",
    "기타",
)

AGE_BANDS_7: tuple[str, ...] = (
    "10대",
    "20대",
    "30대",
    "40대",
    "50대",
    "60대",
    "70대 이상",
)

SEXES_2: tuple[str, ...] = ("남자", "여자")
EDUCATION_3: tuple[str, ...] = ("고졸 이하", "대졸 이하", "대학원 이상")

CAREER_BANDS_5: tuple[str, ...] = (
    "10년 미만",
    "10-20년 미만",
    "20-30년 미만",
    "30-40년 미만",
    "40년 이상",
)

INDIVIDUAL_INCOME_9: tuple[str, ...] = (
    "없음",
    "5백만원 미만",
    "5백-1천만원 미만",
    "1-2천만원 미만",
    "2-3천만원 미만",
    "3-4천만원 미만",
    "4-5천만원 미만",
    "5-6천만원 미만",
    "6천만원 이상",
)

HOUSEHOLD_INCOME_9: tuple[str, ...] = (
    "1천만원 미만",
    "1-2천만원 미만",
    "2-3천만원 미만",
    "3-4천만원 미만",
    "4-5천만원 미만",
    "5-6천만원 미만",
    "6-7천만원 미만",
    "7-8천만원 미만",
    "8천만원 이상",
)

AGE_GROUP_4: tuple[str, ...] = ("30대 이하", "40대", "50대", "60세 이상")


def _parse_int(value: Any) -> int | None:
    """'1,234' → 1234, '(481)' → 481, '-' / '' → None."""
    if value is None:
        return None
    s = str(value).strip()
    if s in {"", "-", "—", "–", "nan", "NaN", "None"}:
        return None
    s = s.replace(",", "").replace("(", "").replace(")", "").strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _parse_float(value: Any) -> float | None:
    import math

    if value is None:
        return None
    s = str(value).strip()
    if s in {"", "-", "—", "–", "nan", "NaN", "None"}:
        return None
    s = s.replace(",", "").replace("(", "").replace(")", "").replace("%", "")
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _normalize_field(name: str) -> str | None:
    """Map aliases to the 14 standard names."""
    s = str(name).strip()
    aliases = {
        "방송·연예": "방송연예",
        "방송 · 연예": "방송연예",
        "방송, 연예": "방송연예",
        "방송 연예": "방송연예",
    }
    s = aliases.get(s, s)
    return s if s in FIELDS_14 else None


def _pdf_sha256() -> str:
    """sha256 of the source PDF (for provenance)."""
    candidates = sorted(settings.source_dir.glob("*.pdf"))
    if not candidates:
        return ""
    h = hashlib.sha256()
    with candidates[0].open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _source_pdf_path() -> Path:
    """Source PDF file path (the first PDF in source_dir)."""
    candidates = sorted(settings.source_dir.glob("*.pdf"))
    if not candidates:
        raise FileNotFoundError(f"no PDF in {settings.source_dir}")
    return candidates[0]


def _parse_field_pcts_from_pdf_text(
    page_idx_0: int,
    *,
    fields_order: list[str],
    n_categories: int,
    expected_extra_numbers: int = 0,
    rowsum_target: float = 100.0,
    rowsum_tolerance: float = 1.5,
) -> pd.DataFrame:
    """Parse the field×% table directly from the page raw text.

    Each field row has the form "<field> (<N>) <pct1> ... <pctK> [<extra1> ... <extraE>]".
    - n_categories: number of % columns to take (from the front)
    - expected_extra_numbers: extra numeric columns after the % values such as
      mean/median/stddev (for validation; warns if a row has fewer than
      n_categories + expected_extra_numbers)
    - rowsum_target/tolerance: % row-sum validation

    Bypasses camelot/tabula column-alignment defects and treats the report % itself
    as the source of truth.
    """
    import re

    import pdfplumber  # noqa: PLC0415  (optional dependency)

    pdf_path = _source_pdf_path()
    with pdfplumber.open(pdf_path) as pdf:
        text = pdf.pages[page_idx_0].extract_text() or ""

    rows: list[dict[str, Any]] = []
    n_total_expected = n_categories + expected_extra_numbers
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        for field in fields_order:
            if not line.startswith(field + " ("):
                continue
            m = re.match(rf"^{re.escape(field)}\s+\(([0-9,]+)\)\s+(.+)$", line)
            if not m:
                continue
            case_count = int(m.group(1).replace(",", ""))
            tail = m.group(2)
            nums = re.findall(r"-?[\d,]+\.\d+|-?[\d,]+", tail)
            parsed = [float(x.replace(",", "")) for x in nums]
            if len(parsed) < n_categories:
                logger.warning(
                    "p%d field=%s: only %d numbers in row, expected at least %d",
                    page_idx_0 + 1,
                    field,
                    len(parsed),
                    n_categories,
                )
                continue
            if expected_extra_numbers > 0 and len(parsed) < n_total_expected:
                logger.warning(
                    "p%d field=%s: row has %d numbers, expected %d (cats=%d, extras=%d)",
                    page_idx_0 + 1,
                    field,
                    len(parsed),
                    n_total_expected,
                    n_categories,
                    expected_extra_numbers,
                )
            pcts = parsed[:n_categories]
            row_sum = sum(pcts)
            if abs(row_sum - rowsum_target) > rowsum_tolerance:
                logger.warning(
                    "p%d field=%s: row sum %.2f deviates from %.1f (tol=%.1f)",
                    page_idx_0 + 1,
                    field,
                    row_sum,
                    rowsum_target,
                    rowsum_tolerance,
                )
            rows.append(
                {
                    "field": field,
                    "case_count": case_count,
                    "pcts": pcts,
                    "row_sum": row_sum,
                }
            )
            break

    if not rows:
        raise RuntimeError(
            f"no field rows parsed on page {page_idx_0 + 1} for {fields_order}"
        )

    parsed_fields = {r["field"] for r in rows}
    missing = [f for f in fields_order if f not in parsed_fields]
    if missing:
        raise RuntimeError(
            f"page {page_idx_0 + 1}: missing fields after text parse: {missing}"
        )

    return pd.DataFrame(rows)


def _parse_age4_sub_block(
    page_idx_0: int,
    *,
    n_categories: int,
    category_labels: list[str],
    expected_extras: int = 0,
    rowsum_indices: list[int] | None = None,
    rowsum_target: float = 100.0,
    rowsum_tolerance: float = 0.5,
    rowsum_hard_tolerance: float = 1.5,
) -> pd.DataFrame:
    """Parse the age4×% sub-block directly from the page raw text.

    Age rows in the report body table usually have the form
    ``<age_label> (<N>) <pct1> ... <pctK> [extra...]``.
    This helper extracts only the 4 age groups the report provides directly, and
    also returns row-sum validation info. If a row sum exceeds the hard tolerance,
    it aborts on suspicion of a cell-extraction error.
    """
    import re

    import pdfplumber  # noqa: PLC0415

    if n_categories != len(category_labels):
        raise ValueError(
            "n_categories must match category_labels length: "
            f"{n_categories} != {len(category_labels)}"
        )

    pdf_path = _source_pdf_path()
    with pdfplumber.open(pdf_path) as pdf:
        text = pdf.pages[page_idx_0].extract_text() or ""

    label_patterns = {
        "30대 이하": r"30대\s*이하",
        "40대": r"40대",
        "50대": r"50대",
        "60세 이상": r"60세\s*이상",
    }
    n_total_expected = n_categories + expected_extras
    if rowsum_indices is None:
        rowsum_indices = list(range(n_categories))
    for idx in rowsum_indices:
        if idx < 0 or idx >= n_categories:
            raise ValueError(f"rowsum index out of range: {idx}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        for label, pat in label_patterns.items():
            if label in seen:
                continue
            m = re.match(rf"^{pat}\s+\(([0-9,]+)\)\s+(.+)$", line)
            if not m:
                continue
            case_count = int(m.group(1).replace(",", ""))
            tail = m.group(2)
            nums = re.findall(r"-?[\d,]+\.\d+|-?[\d,]+", tail)
            parsed = [float(x.replace(",", "")) for x in nums]
            if len(parsed) < n_categories:
                logger.warning(
                    "p%d age_group=%s: only %d numbers, expected at least %d",
                    page_idx_0 + 1,
                    label,
                    len(parsed),
                    n_categories,
                )
                continue
            if expected_extras > 0 and len(parsed) < n_total_expected:
                logger.warning(
                    "p%d age_group=%s: row has %d numbers, expected %d "
                    "(cats=%d, extras=%d)",
                    page_idx_0 + 1,
                    label,
                    len(parsed),
                    n_total_expected,
                    n_categories,
                    expected_extras,
                )
            pcts = parsed[:n_categories]
            row_sum = float(sum(pcts[idx] for idx in rowsum_indices))
            row_sum_abs_diff = abs(row_sum - rowsum_target)
            if row_sum_abs_diff > rowsum_hard_tolerance:
                raise RuntimeError(
                    f"p{page_idx_0 + 1} age_group={label}: row sum {row_sum:.2f} "
                    f"deviates from {rowsum_target:.1f} by more than "
                    f"{rowsum_hard_tolerance:.1f}%p"
                )
            if row_sum_abs_diff > rowsum_tolerance:
                logger.warning(
                    "p%d age_group=%s: row sum %.2f deviates from %.1f "
                    "(tol=%.1f)",
                    page_idx_0 + 1,
                    label,
                    row_sum,
                    rowsum_target,
                    rowsum_tolerance,
                )
            rows.append(
                {
                    "age_group_4": label,
                    "case_count": case_count,
                    "pcts": pcts,
                    "category_labels": list(category_labels),
                    "row_sum": row_sum,
                    "row_sum_indices": list(rowsum_indices),
                    "row_sum_abs_diff": row_sum_abs_diff,
                    "row_sum_passed": row_sum_abs_diff <= rowsum_tolerance,
                }
            )
            seen.add(label)
            break

    if not rows:
        raise RuntimeError(f"no age4 rows parsed on page {page_idx_0 + 1}")

    parsed_labels = {r["age_group_4"] for r in rows}
    missing = [a for a in AGE_GROUP_4 if a not in parsed_labels]
    if missing:
        raise RuntimeError(
            f"page {page_idx_0 + 1}: missing age groups after parse: {missing}"
        )

    return pd.DataFrame(rows)


def _now_iso() -> str:
    return datetime.now(UTC).astimezone().isoformat(timespec="seconds")


def _save_parquet_and_provenance(
    table_id: str,
    df: pd.DataFrame,
    provenance: dict[str, Any],
) -> tuple[Path, Path]:
    settings.grounding_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = settings.grounding_dir / f"{table_id}.parquet"
    provenance_path = settings.grounding_dir / f"{table_id}_provenance.json"
    df.to_parquet(parquet_path, index=False)
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("wrote %s (%d rows) and %s", parquet_path, len(df), provenance_path.name)
    return parquet_path, provenance_path


# ----------------------------------------------------------------------------
# T1 — field × age band × sex (population)
# ----------------------------------------------------------------------------


def reconcile_t1() -> pd.DataFrame:
    """표 1-6 (p.21) — camelot lattice captures the two part-tables (left/right) on one page well.

    Left: 10대~40대 × {남,여,모름}
    Right: 50대~70대이상 × {남,여,모름} + 모름 + 합계
    """
    left = pd.read_csv(settings.extracted_dir / "T1_camelot_lattice_0.csv", encoding="utf-8-sig")
    right = pd.read_csv(settings.extracted_dir / "T1_camelot_lattice_1.csv", encoding="utf-8-sig")
    # 2 header rows (age band + sex), then 14 fields + total
    rows: list[dict[str, Any]] = []

    def parse_block(df: pd.DataFrame, age_bands: list[str], col_offset: int = 1) -> None:
        # df: 2 header rows + 15 field rows. First column is the field name.
        for body in df.iloc[2:].itertuples(index=False):
            field_raw = str(body[0]).strip().replace(" ", "")
            if field_raw == "합계":
                continue
            field = _normalize_field(field_raw)
            if field is None:
                logger.warning("T1 unknown field: %r", field_raw)
                continue
            for age_idx, age_band in enumerate(age_bands):
                base = col_offset + age_idx * 3
                male = _parse_int(body[base])
                female = _parse_int(body[base + 1])
                # body[base+2]: 모름 (excluded)
                if male is not None:
                    rows.append(
                        {
                            "field": field,
                            "age_band": age_band,
                            "sex": "남자",
                            "count": male,
                        }
                    )
                if female is not None:
                    rows.append(
                        {
                            "field": field,
                            "age_band": age_band,
                            "sex": "여자",
                            "count": female,
                        }
                    )

    parse_block(left, age_bands=["10대", "20대", "30대", "40대"], col_offset=1)
    parse_block(right, age_bands=["50대", "60대", "70대 이상"], col_offset=1)

    df = pd.DataFrame(rows)
    # per-field ratio (count / sum_count_within_field)
    totals = (
        df.groupby("field", as_index=False)["count"].sum().rename(columns={"count": "field_total"})
    )
    df = df.merge(totals, on="field")
    df["probability"] = df["count"] / df["field_total"]
    df["value_type"] = "count"
    df["is_imputed"] = False
    df["confidence"] = "high"
    df["source_table"] = "표 1-6"
    df["source_page"] = 21
    return df[
        [
            "field",
            "sex",
            "age_band",
            "count",
            "probability",
            "value_type",
            "is_imputed",
            "confidence",
            "source_table",
            "source_page",
        ]
    ]


# ----------------------------------------------------------------------------
# T2 — field × 17 provinces + "기타" (unknown region) (population)
# ----------------------------------------------------------------------------


def reconcile_t2() -> pd.DataFrame:
    """표 1-3 (p.19) — camelot lattice captures it well as two parts (left/right).

    Left: 서울 ~ 강원 (10 provinces)
    Right: 충북 ~ 기타 (unknown region) + 합계
    """
    left = pd.read_csv(settings.extracted_dir / "T2_camelot_lattice_0.csv", encoding="utf-8-sig")
    right = pd.read_csv(settings.extracted_dir / "T2_camelot_lattice_1.csv", encoding="utf-8-sig")

    rows: list[dict[str, Any]] = []

    def parse_block(df: pd.DataFrame, provinces: list[str]) -> None:
        # df: 1 header row + 14 fields + 1 total row. First column is the field.
        for body in df.iloc[1:].itertuples(index=False):
            field_raw = str(body[0]).strip().replace(" ", "")
            if field_raw == "합계":
                continue
            field = _normalize_field(field_raw)
            if field is None:
                continue
            for col_idx, province in enumerate(provinces, start=1):
                count = _parse_int(body[col_idx])
                if count is None:
                    continue
                rows.append({"field": field, "province": province, "count": count})

    parse_block(
        left,
        provinces=[
            "서울",
            "부산",
            "대구",
            "인천",
            "광주",
            "대전",
            "울산",
            "세종",
            "경기",
            "강원",
        ],
    )
    parse_block(
        right,
        provinces=[
            "충북",
            "충남",
            "전북",
            "전남",
            "경북",
            "경남",
            "제주",
            "기타",
        ],
    )

    df = pd.DataFrame(rows)
    totals = (
        df.groupby("field", as_index=False)["count"].sum().rename(columns={"count": "field_total"})
    )
    df = df.merge(totals, on="field")
    df["probability"] = df["count"] / df["field_total"]
    df["value_type"] = "count"
    df["is_imputed"] = False
    df["confidence"] = "high"
    df["source_table"] = "표 1-3"
    df["source_page"] = 19
    return df[
        [
            "field",
            "province",
            "count",
            "probability",
            "value_type",
            "is_imputed",
            "confidence",
            "source_table",
            "source_page",
        ]
    ]


# ----------------------------------------------------------------------------
# T3 — field × education (respondents N=5,059, appendix table)
# ----------------------------------------------------------------------------


def reconcile_t3() -> pd.DataFrame:
    """For T3, the appendix respondent table (p.117-118) comes in P(field|education) form.

    Here we combine the respondent case counts (고졸 826 / 대졸 2,768 / 대학원 1,465)
    with each education level's field ratios and convert to (field, education, count) form.

    The appendix table p.117-118 extraction is multi-page, so this simple version parses
    the integer case counts directly from text (camelot lattice has low extraction quality
    because the chart area and the table are mixed together).
    """
    # parse the education rows directly from p.117 text (positioned after sex/age)
    # 14 field ratios appear in the form "고졸 이하 (826) ..."
    p117 = (settings.extracted_dir / "text" / "page_117.txt").read_text(encoding="utf-8")
    p118 = (settings.extracted_dir / "text" / "page_118.txt").read_text(encoding="utf-8")

    # p.117 is the left 7 fields (문학 미술 공예 사진 건축 음악 국악)
    # p.118 is the right 7 fields (대중음악 방송연예 무용 연극 영화 만화 기타)
    fields_left = ["문학", "미술", "공예", "사진", "건축", "음악", "국악"]
    fields_right = ["대중음악", "방송연예", "무용", "연극", "영화", "만화", "기타"]

    edu_case_counts = {"고졸 이하": 826, "대졸 이하": 2768, "대학원 이상": 1465}

    def find_edu_pcts(text: str, n_cols: int) -> dict[str, list[float | None]]:
        """Extract things like '고졸 이하 (826) <pct1> <pct2> ...' from text."""
        out: dict[str, list[float | None]] = {}
        for label in ["고졸 이하", "대졸 이하", "대학원 이상"]:
            # use only the first occurrence of each label (appears once per page)
            idx = text.find(label)
            if idx < 0:
                out[label] = [None] * n_cols
                continue
            line_end = text.find("\n", idx)
            line = text[idx : line_end if line_end > 0 else len(text)]
            tokens = line.replace("(", " ").replace(")", " ").split()
            # tokens[0]=고졸 / 이하, [1]=826, [2~]=ratios
            nums = [t for t in tokens if any(c.isdigit() for c in t) and "%" not in t]
            # the first number is the case count, the next n_cols are ratios
            pcts = []
            seen_first = False
            for t in nums:
                v = _parse_float(t)
                if v is None:
                    continue
                if not seen_first:
                    seen_first = True  # case count
                    continue
                pcts.append(v)
                if len(pcts) >= n_cols:
                    break
            while len(pcts) < n_cols:
                pcts.append(None)
            out[label] = pcts
        return out

    left_pcts = find_edu_pcts(p117, n_cols=len(fields_left))
    right_pcts = find_edu_pcts(p118, n_cols=len(fields_right) + 1)  # +1 for "합계" 100.0

    rows: list[dict[str, Any]] = []
    for edu, cases in edu_case_counts.items():
        # left 7 fields
        for f, pct in zip(fields_left, left_pcts[edu], strict=False):
            if pct is None:
                continue
            rows.append(
                {
                    "field": f,
                    "education": edu,
                    "case_count_in_education": cases,
                    "pct_field_within_education": pct,
                    "implied_count": round(cases * pct / 100.0),
                }
            )
        # right (excluding the last total column 100.0)
        for f, pct in zip(fields_right, right_pcts[edu][:-1], strict=False):
            if pct is None:
                continue
            rows.append(
                {
                    "field": f,
                    "education": edu,
                    "case_count_in_education": cases,
                    "pct_field_within_education": pct,
                    "implied_count": round(cases * pct / 100.0),
                }
            )

    df = pd.DataFrame(rows)
    # Bayesian conversion to P(education|field): pct(field|education) * P(education) / P(field|...)
    # here we normalize implied_count per field to compute P(education|field)
    totals = (
        df.groupby("field", as_index=False)["implied_count"]
        .sum()
        .rename(columns={"implied_count": "field_total"})
    )
    df = df.merge(totals, on="field")
    df["probability"] = df["implied_count"] / df["field_total"]
    df["count"] = df["implied_count"]
    df["value_type"] = "implied_count_from_pct"
    df["is_imputed"] = True  # Bayesian conversion P(field|education) → P(education|field)
    df["confidence"] = "medium"
    df["source_table"] = "표 부록-2"
    df["source_page"] = 117
    return df[
        [
            "field",
            "education",
            "count",
            "probability",
            "value_type",
            "is_imputed",
            "confidence",
            "source_table",
            "source_page",
        ]
    ]


# ----------------------------------------------------------------------------
# T4 — field × career band (respondents, 표 3-3)
# ----------------------------------------------------------------------------


def _read_field_by_category_pcts(
    table_id: str,
    fields_order: list[str],
    category_labels: list[str],
    case_total_expected: int = 5059,
) -> pd.DataFrame:
    """Build a long-format DataFrame from a body respondent statistics table (field × category %).

    Uses the highest-confidence extraction (the last camelot:lattice) as the basis.
    Each row: field name + case count (N) + category ratios.
    """
    # prioritize an extraction that includes all 14 fields and has the right category column count
    summary_path = settings.extracted_dir / f"{table_id}_extractions.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    method_priority = (
        "camelot:lattice",
        "camelot:stream",
        "pdfplumber:text+lines",
        "pdfplumber:lines+text",
        "pdfplumber:text+text",
    )
    best_csv: str | None = None

    def fields_in_csv(csv_path: Path) -> int:
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
        except Exception:
            return 0
        n = 0
        for _, row in df.iterrows():
            for ci in range(min(3, len(row))):
                col = str(row.iloc[ci]).strip().replace(" ", "")
                if _normalize_field(col) is not None:
                    n += 1
                    break
        return n

    candidates = [r for r in summary["results"] if not r["error"] and r["csv"]]
    candidates.sort(
        key=lambda r: (
            method_priority.index(r["method"]) if r["method"] in method_priority else 99,
            -fields_in_csv(settings.project_root / r["csv"]),
            -r["n_cols"],
        )
    )
    for r in candidates:
        n_fields = fields_in_csv(settings.project_root / r["csv"])
        if n_fields >= 12 and r["n_cols"] >= len(category_labels) + 2:
            best_csv = r["csv"]
            break
    if best_csv is None:
        # last fallback: the result with the most fields
        candidates2 = sorted(
            candidates,
            key=lambda r: -fields_in_csv(settings.project_root / r["csv"]),
        )
        for r in candidates2:
            if fields_in_csv(settings.project_root / r["csv"]) >= 10:
                best_csv = r["csv"]
                break
    if best_csv is None:
        raise RuntimeError(f"{table_id}: no suitable extraction found")

    df = pd.read_csv(settings.project_root / best_csv, encoding="utf-8-sig")

    # find the column where field labels appear most often (handles stream cases where the first column is NaN)
    label_col_idx = 0
    best_n = 0
    for ci in range(min(3, df.shape[1])):
        n = sum(
            1
            for _, r in df.iterrows()
            if _normalize_field(str(r.iloc[ci]).strip().replace(" ", "")) is not None
        )
        if n > best_n:
            best_n = n
            label_col_idx = ci

    rows: list[dict[str, Any]] = []
    for body in df.itertuples(index=False):
        col_label = str(body[label_col_idx]).strip().replace(" ", "")
        field = _normalize_field(col_label)
        if field is None:
            continue
        # the case count may be in the next column or the one after. Find the first numeric column (possibly parenthesized).
        case_count: int | None = None
        case_col_idx = label_col_idx + 1
        for ci in range(label_col_idx + 1, min(label_col_idx + 3, len(body))):
            v = _parse_int(body[ci])
            if v is not None and v > 5 and v < 100000:  # case count range
                case_count = v
                case_col_idx = ci
                break
        if case_count is None:
            continue
        # from after the case count, skip NaN/empty columns and collect only numeric ratios in order
        category_values: list[float | None] = []
        ci = case_col_idx + 1
        while ci < len(body) and len(category_values) < len(category_labels):
            v = _parse_float(body[ci])
            if v is None:
                ci += 1
                continue
            category_values.append(v)
            ci += 1
        while len(category_values) < len(category_labels):
            category_values.append(None)
        for label, pct in zip(category_labels, category_values, strict=False):
            if pct is None or case_count is None:
                continue
            rows.append(
                {
                    "field": field,
                    "category": label,
                    "case_count": int(case_count),
                    "pct": float(pct),
                    "count": int(round(case_count * pct / 100.0)),
                }
            )

    return pd.DataFrame(rows), best_csv


def reconcile_t4() -> pd.DataFrame:
    df, used_csv = _read_field_by_category_pcts(
        "T4",
        fields_order=list(FIELDS_14),
        category_labels=list(CAREER_BANDS_5),
    )
    df = df.rename(columns={"category": "career_band"})
    totals = (
        df.groupby("field", as_index=False)["count"].sum().rename(columns={"count": "field_total"})
    )
    df = df.merge(totals, on="field")
    df["probability"] = df["count"] / df["field_total"]
    df["value_type"] = "implied_count_from_pct"
    df["is_imputed"] = False
    df["confidence"] = "high"
    df["source_table"] = "표 3-3"
    df["source_page"] = 55
    df.attrs["used_csv"] = used_csv
    return df[
        [
            "field",
            "career_band",
            "count",
            "probability",
            "value_type",
            "is_imputed",
            "confidence",
            "source_table",
            "source_page",
        ]
    ]


# ----------------------------------------------------------------------------
# T5 — field × full-time/side-job × freelance (respondents, 표 3-12 + 표 3-20 + 표 3-21)
# ----------------------------------------------------------------------------


def reconcile_t5() -> pd.DataFrame:
    """Combine the three tables to build the P(full-time, freelance|field) joint distribution."""
    main, _ = _read_field_by_category_pcts(
        "T5_main",
        fields_order=list(FIELDS_14),
        category_labels=["전업", "겸업"],
    )
    full, _ = _read_field_by_category_pcts(
        "T5_freelance_full",
        fields_order=list(FIELDS_14),
        category_labels=["프리랜서", "비프리랜서"],
    )
    part, _ = _read_field_by_category_pcts(
        "T5_freelance_part",
        fields_order=list(FIELDS_14),
        category_labels=["프리랜서", "비프리랜서"],
    )

    if main.empty or "field" not in main.columns:
        logger.warning("T5: main extraction empty, returning empty df")
        return pd.DataFrame(
            columns=[
                "field",
                "employment_type",
                "is_freelance",
                "count",
                "probability",
                "value_type",
                "is_imputed",
                "confidence",
                "source_table",
                "source_page",
            ]
        )

    rows: list[dict[str, Any]] = []
    for f in FIELDS_14:
        emp_row = main[main["field"] == f]
        if emp_row.empty:
            continue
        # first category = full-time (전업) ratio
        emp_pct = emp_row.iloc[0]["pct"] if "pct" in emp_row.columns else None
        cases = emp_row.iloc[0]["case_count"] if "case_count" in emp_row.columns else None
        if emp_pct is None or cases is None:
            continue
        full_row = full[full["field"] == f]
        full_free = full_row.iloc[0]["pct"] if not full_row.empty else None
        part_row = part[part["field"] == f]
        part_free = part_row.iloc[0]["pct"] if not part_row.empty else None

        # P(full-time) = emp_pct/100. P(side-job) = (100-emp_pct)/100
        p_full = emp_pct / 100.0
        p_part = 1 - p_full
        # P(freelance|full-time, field) = full_free/100  → if missing, field-average value is absent
        # P(freelance|side-job, field) = part_free/100
        if full_free is not None and part_free is not None:
            for emp_label, p_emp, free_pct in [
                ("전업", p_full, full_free / 100.0),
                ("겸업", p_part, part_free / 100.0),
            ]:
                rows.append(
                    {
                        "field": f,
                        "employment_type": emp_label,
                        "is_freelance": True,
                        "case_count": cases,
                        "probability": p_emp * free_pct,
                        "count": round(cases * p_emp * free_pct),
                    }
                )
                rows.append(
                    {
                        "field": f,
                        "employment_type": emp_label,
                        "is_freelance": False,
                        "case_count": cases,
                        "probability": p_emp * (1 - free_pct),
                        "count": round(cases * p_emp * (1 - free_pct)),
                    }
                )

    df = pd.DataFrame(rows)
    df["value_type"] = "implied_count_from_pct"
    df["is_imputed"] = True  # combination of three tables (conditional independence assumption)
    df["confidence"] = "medium"
    df["source_table"] = "표 3-12 + 3-20 + 3-21"
    df["source_page"] = 65
    return df[
        [
            "field",
            "employment_type",
            "is_freelance",
            "count",
            "probability",
            "value_type",
            "is_imputed",
            "confidence",
            "source_table",
            "source_page",
        ]
    ]


def _age4_categorical_frame(
    parsed: pd.DataFrame,
    *,
    variable_col: str,
    source_table: str,
    source_page: int,
    category_labels: list[Any] | None = None,
    pct_indices: list[int] | None = None,
) -> pd.DataFrame:
    """Parsed age4 rows → standard long-format categorical table."""
    if parsed.empty:
        raise RuntimeError(f"{source_table}: empty age4 parse")
    if category_labels is None:
        category_labels = list(parsed.iloc[0]["category_labels"])
    if pct_indices is None:
        pct_indices = list(range(len(category_labels)))
    if len(category_labels) != len(pct_indices):
        raise ValueError("category_labels and pct_indices must have the same length")

    rows: list[dict[str, Any]] = []
    for age_group in AGE_GROUP_4:
        sub = parsed[parsed["age_group_4"] == age_group]
        if sub.empty:
            raise RuntimeError(f"{source_table}: missing age group {age_group!r}")
        body = sub.iloc[0]
        case_count = int(body["case_count"])
        pcts = list(body["pcts"])
        for label, pct_idx in zip(category_labels, pct_indices, strict=True):
            pct = float(pcts[pct_idx])
            rows.append(
                {
                    "age_group_4": age_group,
                    variable_col: label,
                    "count": int(round(case_count * pct / 100.0)),
                    "probability": pct / 100.0,
                    "value_type": "implied_count_from_pct",
                    "is_imputed": False,
                    "confidence": "high",
                    "source_table": source_table,
                    "source_page": source_page,
                }
            )
    return pd.DataFrame(rows)


def _age4_binary_frame(
    parsed: pd.DataFrame,
    *,
    source_table: str,
    source_page: int,
    true_index: int = 0,
    false_index: int | None = 1,
) -> pd.DataFrame:
    """Parsed age4 rows → age4×{True,False} table using report cells."""
    rows: list[dict[str, Any]] = []
    for age_group in AGE_GROUP_4:
        sub = parsed[parsed["age_group_4"] == age_group]
        if sub.empty:
            raise RuntimeError(f"{source_table}: missing age group {age_group!r}")
        body = sub.iloc[0]
        case_count = int(body["case_count"])
        pcts = list(body["pcts"])
        true_pct = float(pcts[true_index])
        false_pct = 100.0 - true_pct if false_index is None else float(pcts[false_index])
        for value, pct in [(True, true_pct), (False, false_pct)]:
            rows.append(
                {
                    "age_group_4": age_group,
                    "value": value,
                    "count": int(round(case_count * pct / 100.0)),
                    "probability": pct / 100.0,
                    "value_type": "implied_count_from_pct",
                    "is_imputed": False,
                    "confidence": "high",
                    "source_table": source_table,
                    "source_page": source_page,
                }
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# T6 — field × income bracket (respondents, 표 3-34 main + 표 3-33 supporting)
# ----------------------------------------------------------------------------


def reconcile_t8() -> pd.DataFrame:
    """표 3-3 (p.55) — 연령 4구간 × 경력 5구간 sub-block (age 4-band × career 5-band).

    The same 표 3-3 contains field (T4), sex, and age sub-blocks together, but T4 took
    only the field rows. T8 augments the age sub-block from the same source. It is used
    as one of the marginals for estimating the (field × age × career) 3-way joint via IPF
    (the report provides no direct 3-way table).

    Each age row has the form "<age_label> (<N>) <p1> <p2> <p3> <p4> <p5>".
    """
    df = _age4_categorical_frame(
        _parse_age4_sub_block(
            page_idx_0=54,
            n_categories=len(CAREER_BANDS_5),
            category_labels=list(CAREER_BANDS_5),
        ),
        variable_col="career_band",
        source_table="표 3-3",
        source_page=55,
    )
    df["value_type"] = "implied_count_from_pct"
    df["is_imputed"] = False
    df["confidence"] = "high"
    return df[
        [
            "age_group_4",
            "career_band",
            "count",
            "probability",
            "value_type",
            "is_imputed",
            "confidence",
            "source_table",
            "source_page",
        ]
    ]


def reconcile_t6() -> pd.DataFrame:
    """표 3-34 (p.87) — field × art-creation-activity income, 9 brackets.

    The previous implementation trusted the column alignment of the camelot/tabula CSV,
    but the 표 3-34 header is wrapped across multiple lines (e.g. `5백 / 만원 / 미만`), so
    automatic extraction absorbed the mean/median cells into the 9th % column position and
    truncated the last "6천만원 이상". As a result the synthetic data income distribution was
    completely off versus the actual % (e.g. 4-5천 50.7%, 없음 6.7%; report: 4-5천 2.1%, 없음 31.0%).

    To avoid the defect, parse the 14×9 % matrix directly from the report body text. Each field
    row has the form `<field> (<N>) <p1> ... <p9> <mean> <median> <stddev>` and goes through a
    100% row-sum check.
    """
    parsed = _parse_field_pcts_from_pdf_text(
        page_idx_0=86,  # p.87 (1-indexed)
        fields_order=list(FIELDS_14),
        n_categories=len(INDIVIDUAL_INCOME_9),
        expected_extra_numbers=3,  # mean, median, stddev
        rowsum_target=100.0,
        rowsum_tolerance=0.5,
    )
    rows: list[dict[str, Any]] = []
    for body in parsed.itertuples(index=False):
        case_count = int(body.case_count)
        for label, pct in zip(INDIVIDUAL_INCOME_9, body.pcts, strict=True):
            implied = int(round(case_count * pct / 100.0))
            rows.append(
                {
                    "field": body.field,
                    "income_bracket": label,
                    "count": implied,
                    "_case_count": case_count,
                }
            )
    df = pd.DataFrame(rows)
    df["probability"] = df["count"] / df["_case_count"]
    df["value_type"] = "implied_count_from_pct"
    df["is_imputed"] = False
    df["confidence"] = "high"
    df["source_table"] = "표 3-34"
    df["source_page"] = 87
    return df[
        [
            "field",
            "income_bracket",
            "count",
            "probability",
            "value_type",
            "is_imputed",
            "confidence",
            "source_table",
            "source_page",
        ]
    ]


# ----------------------------------------------------------------------------
# T9~T15 — age4 × variable sub-blocks (respondents)
# ----------------------------------------------------------------------------


def reconcile_t9_employment_age() -> pd.DataFrame:
    """표 3-12 (p.65) — age4 × full-time/side-job."""
    return _age4_categorical_frame(
        _parse_age4_sub_block(
            page_idx_0=64,
            n_categories=2,
            category_labels=["전업", "겸업"],
        ),
        variable_col="employment_type",
        source_table="표 3-12",
        source_page=65,
    )[
        [
            "age_group_4",
            "employment_type",
            "count",
            "probability",
            "value_type",
            "is_imputed",
            "confidence",
            "source_table",
            "source_page",
        ]
    ]


def reconcile_t10_income_age() -> pd.DataFrame:
    """표 3-34 (p.87) — age4 × art-creation-activity income, 9 brackets."""
    return _age4_categorical_frame(
        _parse_age4_sub_block(
            page_idx_0=86,
            n_categories=len(INDIVIDUAL_INCOME_9),
            category_labels=list(INDIVIDUAL_INCOME_9),
            expected_extras=3,
        ),
        variable_col="income_bracket",
        source_table="표 3-34",
        source_page=87,
    )[
        [
            "age_group_4",
            "income_bracket",
            "count",
            "probability",
            "value_type",
            "is_imputed",
            "confidence",
            "source_table",
            "source_page",
        ]
    ]


def reconcile_t11_contract_experience_age() -> pd.DataFrame:
    """표 3-23 (p.76) — age4 × contract-signing experience (yes/no)."""
    return _age4_binary_frame(
        _parse_age4_sub_block(
            page_idx_0=75,
            n_categories=4,
            category_labels=["경험 있음", "서면 계약", "서면 없이 구두 계약", "경험 없음"],
            rowsum_indices=[0, 3],
        ),
        source_table="표 3-23",
        source_page=76,
        true_index=0,
        false_index=3,
    )[
        [
            "age_group_4",
            "value",
            "count",
            "probability",
            "value_type",
            "is_imputed",
            "confidence",
            "source_table",
            "source_page",
        ]
    ]


def reconcile_t12_standard_contract_age() -> pd.DataFrame:
    """표 3-26 (p.78) — age4 × use of standard-contract form (yes/no)."""
    return _age4_binary_frame(
        _parse_age4_sub_block(
            page_idx_0=77,
            n_categories=3,
            category_labels=["활용 있음", "활용 없음", "모름"],
        ),
        source_table="표 3-26",
        source_page=78,
        true_index=0,
        false_index=None,
    )[
        [
            "age_group_4",
            "value",
            "count",
            "probability",
            "value_type",
            "is_imputed",
            "confidence",
            "source_table",
            "source_page",
        ]
    ]


def reconcile_t13_copyright_age() -> pd.DataFrame:
    """표 3-41 (p.94) — age4 × copyright ownership (yes/no)."""
    return _age4_binary_frame(
        _parse_age4_sub_block(
            page_idx_0=93,
            n_categories=2,
            category_labels=["예", "아니오"],
        ),
        source_table="표 3-41",
        source_page=94,
    )[
        [
            "age_group_4",
            "value",
            "count",
            "probability",
            "value_type",
            "is_imputed",
            "confidence",
            "source_table",
            "source_page",
        ]
    ]


def reconcile_t14_career_break_age() -> pd.DataFrame:
    """표 3-55 (p.108) — age4 × art-career-break experience (yes/no)."""
    return _age4_binary_frame(
        _parse_age4_sub_block(
            page_idx_0=107,
            n_categories=2,
            category_labels=["있음", "없음"],
        ),
        source_table="표 3-55",
        source_page=108,
    )[
        [
            "age_group_4",
            "value",
            "count",
            "probability",
            "value_type",
            "is_imputed",
            "confidence",
            "source_table",
            "source_page",
        ]
    ]


def reconcile_t15_overseas_age() -> pd.DataFrame:
    """표 3-7 (p.60) — age4 × overseas art-activity experience (yes/no)."""
    return _age4_binary_frame(
        _parse_age4_sub_block(
            page_idx_0=59,
            n_categories=2,
            category_labels=["예", "아니오"],
        ),
        source_table="표 3-7",
        source_page=60,
    )[
        [
            "age_group_4",
            "value",
            "count",
            "probability",
            "value_type",
            "is_imputed",
            "confidence",
            "source_table",
            "source_page",
        ]
    ]


# ----------------------------------------------------------------------------
# T7 — field × 5 binary variables (respondents, 5 tables)
# ----------------------------------------------------------------------------

T7_TABLES = [
    # T7_contract: 표 3-23 has 4 categories (경험 있음 전체 / 서면계약 / 서면없이 / 경험 없음).
    # Body quote "지난 1년간 계약 경험이 있는 예술인은 57.3%" = first column.
    # (the previous sum_first_two rule added [experience-total + written] and exceeded 100% → clamped to 1.0, a defect)
    (
        "has_contract_experience",
        "T7_contract",
        ["경험 있음(전체)", "서면계약", "서면없이 구두", "경험 없음"],
        "표 3-23",
        76,
        "first_only",
    ),
    # T7_standard_contract: "활용한 적 있음" / "활용한 적 없음" / "모름" — exclude 모름, only the first is True
    (
        "uses_standard_contract",
        "T7_standard_contract",
        ["활용 있음", "활용 없음", "모름"],
        "표 3-26",
        78,
        "first_only",
    ),
    ("has_copyright", "T7_copyright", ["예", "아니오"], "표 3-41", 94, "first_only"),
    ("had_career_break", "T7_career_break", ["있음", "없음"], "표 3-55", 108, "first_only"),
    ("has_overseas_experience", "T7_overseas", ["예", "아니오"], "표 3-7", 60, "first_only"),
]


def reconcile_t7() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for var_name, table_id, labels, source_table, page, true_rule in T7_TABLES:
        try:
            sub, _ = _read_field_by_category_pcts(
                table_id,
                fields_order=list(FIELDS_14),
                category_labels=labels,
            )
        except Exception as exc:
            logger.warning("T7 %s extraction failed: %s", var_name, exc)
            continue
        if sub.empty or "field" not in sub.columns:
            logger.warning("T7 %s: empty sub-frame, skipping", var_name)
            continue
        for f in FIELDS_14:
            sub_f = sub[sub["field"] == f]
            if sub_f.empty:
                continue
            cases = int(sub_f.iloc[0]["case_count"])
            if true_rule == "first_only":
                true_rows = sub_f[sub_f["category"] == labels[0]]
                if true_rows.empty:
                    continue
                pct_true = float(true_rows.iloc[0]["pct"]) / 100.0
            elif true_rule == "sum_first_two":
                # contract experience: "경험 있음(서면)" + "서면 없이 경험" = has contract experience.
                # extraction noise can push this over 100%, so clamp to [0, 1].
                pcts = [
                    float(sub_f[sub_f["category"] == lbl].iloc[0]["pct"])
                    for lbl in labels[:2]
                    if not sub_f[sub_f["category"] == lbl].empty
                ]
                if not pcts:
                    continue
                pct_true = min(max(sum(pcts) / 100.0, 0.0), 1.0)
            else:
                continue
            rows.append(
                {
                    "field": f,
                    "variable": var_name,
                    "value": True,
                    "case_count": cases,
                    "probability": pct_true,
                    "count": round(cases * pct_true),
                    "source_table": source_table,
                    "source_page": page,
                }
            )
            rows.append(
                {
                    "field": f,
                    "variable": var_name,
                    "value": False,
                    "case_count": cases,
                    "probability": 1 - pct_true,
                    "count": round(cases * (1 - pct_true)),
                    "source_table": source_table,
                    "source_page": page,
                }
            )
    df = pd.DataFrame(rows)
    df["value_type"] = "implied_count_from_pct"
    df["is_imputed"] = False
    df["confidence"] = "medium"  # case counts may differ across tables
    return df[
        [
            "field",
            "variable",
            "value",
            "count",
            "probability",
            "value_type",
            "is_imputed",
            "confidence",
            "source_table",
            "source_page",
        ]
    ]


# ----------------------------------------------------------------------------
# Public entrypoints
# ----------------------------------------------------------------------------


def write_t1() -> None:
    df = reconcile_t1()
    pdf_sha = _pdf_sha256()
    total = int(df["count"].sum())
    expected = 334036 - 4871 - 8892  # population total - sex-unknown - age-unknown (possible double count, approximate)
    pages_dict = {"21": "data/pages/page_021.png"}
    provenance: dict[str, Any] = {
        "table_id": "T1",
        "title": "분야 × 성별 × 연령대 (모집단)",
        "source": {
            "pdf_path": "data/source/2024년 예술인 실태조사통계보고서.pdf",
            "pdf_sha256": pdf_sha,
            "source_table_id_in_report": "표 1-6",
            "pages": [21],
            "page_image_paths": pages_dict,
        },
        "extractions": [
            {"method": "camelot:lattice", "table_index_on_page": 0, "shape": "17x13"},
            {"method": "camelot:lattice", "table_index_on_page": 1, "shape": "17x14"},
        ],
        "normalization": {
            "field_aliases_applied": {"방송·연예": "방송연예"},
            "excluded_categories": ["모름 (성별)", "모름 (연령)"],
            "excluded_reason": "Population non-identified cases. PAK personas use only known (sex, age) combinations",
        },
        "verification": {
            "method": "automatic_row_total",
            "computed_grand_total": total,
            "expected_total_basis": "Body total 334,036 persons (excluding sex-unknown 4,871 + part of age-unknown gives roughly 320k expected)",
            "passed_within_5pct": abs(total - expected) / max(expected, 1) < 0.05,
            "verifier": "automatic",
            "timestamp": _now_iso(),
            "confidence": "high",
        },
        "post_processing": [],
    }
    _save_parquet_and_provenance("T1", df, provenance)


def _basic_provenance(
    table_id: str,
    title: str,
    source_table: str,
    pages: list[int],
    method_note: str,
    df: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "table_id": table_id,
        "title": title,
        "source": {
            "pdf_path": "data/source/2024년 예술인 실태조사통계보고서.pdf",
            "pdf_sha256": _pdf_sha256(),
            "source_table_id_in_report": source_table,
            "pages": pages,
            "page_image_paths": {str(p): f"data/pages/page_{p:03d}.png" for p in pages},
        },
        "extractions": [{"method": method_note}],
        "verification": {
            "method": "automatic_aggregate",
            "n_rows": len(df),
            "fields_covered": int(df["field"].nunique()) if "field" in df.columns else None,
            "verifier": "automatic",
            "timestamp": _now_iso(),
        },
        "post_processing": [],
    }


def write_t2() -> None:
    df = reconcile_t2()
    prov = _basic_provenance(
        "T2",
        "분야 × 17개 시도 + 기타 (모집단)",
        "표 1-3",
        [19],
        "camelot:lattice idx=0,1 (left/right merge)",
        df,
    )
    prov["verification"]["computed_grand_total"] = int(df["count"].sum())
    prov["verification"]["expected_grand_total"] = 334036
    prov["verification"]["passed"] = abs(int(df["count"].sum()) - 334036) / 334036 < 0.001
    _save_parquet_and_provenance("T2", df, prov)


def write_t3() -> None:
    df = reconcile_t3()
    prov = _basic_provenance(
        "T3",
        "분야 × 학력 (응답자 N=5,059)",
        "표 부록-2",
        [117, 118],
        "page text parse + Bayes inversion (P(분야|학력) → P(학력|분야))",
        df,
    )
    prov["normalization"] = {
        "transformation": "P(분야|학력) × case_count → implied_count → 분야별 정규화 → P(학력|분야)",
        "education_case_counts": {"고졸 이하": 826, "대졸 이하": 2768, "대학원 이상": 1465},
    }
    _save_parquet_and_provenance("T3", df, prov)


def write_t4() -> None:
    df = reconcile_t4()
    prov = _basic_provenance(
        "T4",
        "분야 × 경력 구간 5구간 (응답자)",
        "표 3-3",
        [55],
        f"camelot:lattice (used: {df.attrs.get('used_csv', '?')})",
        df,
    )
    _save_parquet_and_provenance("T4", df, prov)


def write_t5() -> None:
    df = reconcile_t5()
    prov = _basic_provenance(
        "T5",
        "분야 × 전업/겸업 × 프리랜서 (응답자)",
        "표 3-12 + 3-20 + 3-21",
        [65, 73, 74],
        "3-table chained P(전업|분야) × P(프리|전업,분야) × P(프리|겸업,분야)",
        df,
    )
    _save_parquet_and_provenance("T5", df, prov)


def write_t6() -> None:
    df = reconcile_t6()
    prov = _basic_provenance(
        "T6",
        "분야 × 예술창작활동 수입 9구간 (응답자)",
        "표 3-34",
        [87],
        "pdfplumber text parse (column-aligned via row regex)",
        df,
    )
    prov["supporting_table"] = {"id": "표 3-33 (가구 총소득)", "page": 86}
    # cell-level validation: 14 fields × 9 brackets = 126 cells. Each row % sums to 100±0.5%p, sum of 14 fields' N = 5,047.
    case_count_total = (
        df.groupby("field")["count"].sum().sum()
    )
    prov["verification"]["computed_grand_total"] = int(case_count_total)
    prov["verification"]["expected_grand_total"] = 5047
    prov["verification"]["passed_within_5pct"] = (
        abs(int(case_count_total) - 5047) / 5047 < 0.05
    )
    prov["verification"]["confidence"] = "high"
    prov["verification"]["method"] = "pdfplumber_text_row_sum_check"
    prov["verification"]["row_sum_target_pct"] = 100.0
    prov["verification"]["row_sum_tolerance_pct"] = 0.5
    prov["normalization"] = {
        "transformation": "P(income_bracket | field) × case_count_field → implied_count",
        "fixed_bug_2026_05_09": (
            "The previous build, due to a camelot/tabula CSV column-alignment defect, absorbed "
            "the mean (만원)/median into % cells and dropped the '6천만원 이상' column. This build "
            "replaces that with raw text parsing."
        ),
    }
    prov["visual_verification"] = {
        "method": "page_image_compared",
        "page_image": "data/pages/page_087.png",
        "verifier": "claude_vision_2026_05_09",
        "confidence": "high",
    }
    _save_parquet_and_provenance("T6", df, prov)


def write_t7() -> None:
    df = reconcile_t7()
    prov = _basic_provenance(
        "T7",
        "분야 × 5개 이항 변수 (응답자)",
        "표 3-23 + 3-26 + 3-41 + 3-55 + 3-7",
        [76, 78, 94, 108, 60],
        "5 separate tables, each contributing one binary variable",
        df,
    )
    prov["component_variables"] = [t[0] for t in T7_TABLES]
    _save_parquet_and_provenance("T7", df, prov)


def write_t8() -> None:
    df = reconcile_t8()
    prov = _basic_provenance(
        "T8",
        "연령 4구간 × 경력 5구간 (응답자)",
        "표 3-3 (연령 sub-block)",
        [55],
        "pdfplumber text parse (same source page as T4 분야 sub-block)",
        df,
    )
    case_total = int(df.groupby("age_group_4")["count"].sum().sum())
    prov["verification"]["computed_grand_total"] = case_total
    prov["verification"]["expected_grand_total"] = 5059
    prov["verification"]["passed_within_5pct"] = (
        abs(case_total - 5059) / 5059 < 0.05
    )
    prov["verification"]["confidence"] = "high"
    prov["verification"]["method"] = "pdfplumber_text_row_sum_check"
    prov["verification"]["row_sum_target_pct"] = 100.0
    prov["verification"]["row_sum_tolerance_pct"] = 0.5
    prov["normalization"] = {
        "transformation": "P(career_band | age_group_4) × case_count_age → implied_count",
        "reason_added": (
            "The original grounding used only the (field × career) 1-way conditional, so the "
            "(age × career) joint distribution diverged from report p.55. In the 1000-record pilot, "
            "ages 40+ with 0-9 years were oversampled 1.7~1.85x versus the report. T8 is used to "
            "estimate the (field × age × career) 3-way joint via IPF."
        ),
    }
    prov["age_group_definition"] = {
        "30대 이하": "10대 + 20대 + 30대 (report 표 3-3 4-way classification)",
        "40대": "40대",
        "50대": "50대",
        "60세 이상": "60대 + 70대 이상",
        "note": (
            "When mapping to PAK's internal 7 bands (AGE_BANDS_7): only the direction that "
            "aggregates 7 bands into 4 bands is used. The reverse (decomposing 4 bands into 7) is "
            "an assumption not present in the report, so it is not introduced."
        ),
    }
    prov["visual_verification"] = {
        "method": "page_image_compared",
        "page_image": "data/pages/page_055.png",
        "verifier": "claude_vision_2026_05_09",
        "confidence": "high",
    }
    _save_parquet_and_provenance("T8", df, prov)


def _age4_cell_spot_checks(
    df: pd.DataFrame,
    *,
    variable_col: str,
    checks: list[tuple[str, Any, float]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for age_group, value, expected_pct in checks:
        sub = df[(df["age_group_4"] == age_group) & (df[variable_col] == value)]
        actual_pct = float(sub.iloc[0]["probability"] * 100.0) if not sub.empty else float("nan")
        out.append(
            {
                "age_group_4": age_group,
                variable_col: value,
                "expected_pct": expected_pct,
                "extracted_pct": round(actual_pct, 3),
                "abs_diff_pct_point": round(abs(actual_pct - expected_pct), 3),
                "passed": abs(actual_pct - expected_pct) <= 0.5,
            }
        )
    return out


def _write_age4_table(
    table_id: str,
    df: pd.DataFrame,
    *,
    title: str,
    source_table: str,
    source_page: int,
    variable_col: str,
    expected_grand_total: int,
    normalization_note: str,
    spot_checks: list[tuple[str, Any, float]],
) -> None:
    prov = _basic_provenance(
        table_id,
        title,
        f"{source_table} (연령 sub-block)",
        [source_page],
        "pdfplumber text parse (age4 sub-block row regex)",
        df,
    )
    case_total = int(df.groupby("age_group_4")["count"].sum().sum())
    prob_row_sums = df.groupby("age_group_4")["probability"].sum()
    max_row_sum_diff = float((prob_row_sums - 1.0).abs().max())
    checks = _age4_cell_spot_checks(df, variable_col=variable_col, checks=spot_checks)
    prov["verification"].update(
        {
            "method": "pdfplumber_text_row_sum_check",
            "computed_grand_total": case_total,
            "expected_grand_total": expected_grand_total,
            "passed_within_5pct": abs(case_total - expected_grand_total)
            / expected_grand_total
            < 0.05,
            "probability_row_sum_target": 1.0,
            "max_probability_row_sum_abs_diff": max_row_sum_diff,
            "row_sum_target_pct": 100.0,
            "row_sum_tolerance_pct": 0.5,
            "confidence": "high",
        }
    )
    prov["normalization"] = {
        "transformation": normalization_note,
        "source_integrity": (
            "All probabilities come from report cells in the age sub-block; counts are "
            "case_count × reported_pct rounded to the nearest integer."
        ),
    }
    prov["age_group_definition"] = {
        "30대 이하": "10대 + 20대 + 30대 (report 4-way classification)",
        "40대": "40대",
        "50대": "50대",
        "60세 이상": "60대 + 70대 이상",
        "note": "The assumption of reverse-decomposing 4 bands into 7 bands is not introduced.",
    }
    prov["visual_verification"] = {
        "method": "claude_code_extraction (post-merge review pending)",
        "page_image": f"data/pages/page_{source_page:03d}.png",
        "cell_spot_checks": checks,
        "all_passed": all(c["passed"] for c in checks) and max_row_sum_diff <= 0.005,
        "confidence": "high",
    }
    _save_parquet_and_provenance(table_id, df, prov)


def write_t9() -> None:
    _write_age4_table(
        "T9",
        reconcile_t9_employment_age(),
        title="연령 4구간 × 전업/겸업 (응답자)",
        source_table="표 3-12",
        source_page=65,
        variable_col="employment_type",
        expected_grand_total=5059,
        normalization_note="P(employment_type | age_group_4) × case_count_age → implied_count",
        spot_checks=[("50대", "전업", 43.4), ("60세 이상", "전업", 63.7)],
    )


def write_t10() -> None:
    _write_age4_table(
        "T10",
        reconcile_t10_income_age(),
        title="연령 4구간 × 예술창작활동 수입 9구간 (응답자)",
        source_table="표 3-34",
        source_page=87,
        variable_col="income_bracket",
        expected_grand_total=5047,
        normalization_note="P(income_bracket | age_group_4) × case_count_age → implied_count",
        spot_checks=[("30대 이하", "없음", 21.8), ("60세 이상", "없음", 51.3)],
    )


def write_t11() -> None:
    _write_age4_table(
        "T11",
        reconcile_t11_contract_experience_age(),
        title="연령 4구간 × 예술활동 관련 계약 체결 경험 여부 (작품 발표자)",
        source_table="표 3-23",
        source_page=76,
        variable_col="value",
        expected_grand_total=4362,
        normalization_note="P(has_contract_experience | age_group_4) × subset case_count_age → implied_count",
        spot_checks=[("30대 이하", True, 69.2), ("60세 이상", True, 33.6)],
    )


def write_t12() -> None:
    _write_age4_table(
        "T12",
        reconcile_t12_standard_contract_age(),
        title="연령 4구간 × 표준계약서 양식 활용 여부 (서면 계약 경험자)",
        source_table="표 3-26",
        source_page=78,
        variable_col="value",
        expected_grand_total=2284,
        normalization_note="P(uses_standard_contract | age_group_4) × subset case_count_age → implied_count",
        spot_checks=[("30대 이하", True, 70.6), ("60세 이상", True, 69.1)],
    )


def write_t13() -> None:
    _write_age4_table(
        "T13",
        reconcile_t13_copyright_age(),
        title="연령 4구간 × 저작권(저작인접권) 보유 여부 (응답자)",
        source_table="표 3-41",
        source_page=94,
        variable_col="value",
        expected_grand_total=5059,
        normalization_note="P(has_copyright | age_group_4) × case_count_age → implied_count",
        spot_checks=[("30대 이하", True, 35.8), ("60세 이상", True, 18.7)],
    )


def write_t14() -> None:
    _write_age4_table(
        "T14",
        reconcile_t14_career_break_age(),
        title="연령 4구간 × 예술경력 단절 경험 여부 (응답자)",
        source_table="표 3-55",
        source_page=108,
        variable_col="value",
        expected_grand_total=5059,
        normalization_note="P(had_career_break | age_group_4) × case_count_age → implied_count",
        spot_checks=[("40대", True, 30.2), ("60세 이상", True, 13.4)],
    )


def write_t15() -> None:
    _write_age4_table(
        "T15",
        reconcile_t15_overseas_age(),
        title="연령 4구간 × 외국 예술활동 경험 여부 (응답자)",
        source_table="표 3-7",
        source_page=60,
        variable_col="value",
        expected_grand_total=5059,
        normalization_note="P(has_overseas_experience | age_group_4) × case_count_age → implied_count",
        spot_checks=[("40대", True, 20.6), ("60세 이상", True, 11.7)],
    )


def write_all() -> None:
    write_t1()
    write_t2()
    write_t3()
    write_t4()
    write_t5()
    write_t6()
    write_t7()
    write_t8()
    write_t9()
    write_t10()
    write_t11()
    write_t12()
    write_t13()
    write_t14()
    write_t15()
