"""Run downstream persona-use tasks on the PAK release.

The suite is intentionally lightweight and fully reproducible. It treats the
existing field-recovery task as a sanity check, then adds leakage-audited tasks
that use the persona structure more directly:

1. Narrative-to-anchor recovery for non-field attributes.
2. Hard persona-anchor consistency detection and 10-way retrieval, with field
   and occupation strings masked and candidate negatives drawn from the same
   art field and age band whenever possible.

Each task is run under direct masking and aggressive cue masking. The latter
removes selected anchor-value strings, numeric age/time/money expressions, and
common lexical cues for employment, income, contract, copyright, career-break,
and overseas-activity attributes. Pair and retrieval settings also include a
lexical-similarity baseline so that simple anchor/narrative word overlap is not
mistaken for learned persona reasoning.
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data/release/pak_v0_1/personas.parquet"
REPORT_PATH = ROOT / "outputs/reports/downstream_persona_tasks_260521.md"
ANCHOR_RECOVERY_CSV = ROOT / "outputs/reports/downstream_persona_tasks_anchor_recovery_260521.csv"
PAIR_PREDICTIONS_CSV = ROOT / "outputs/reports/downstream_persona_tasks_pair_predictions_260521.csv"
RETRIEVAL_CSV = ROOT / "outputs/reports/downstream_persona_tasks_retrieval_260521.csv"

RANDOM_SEED = 20260521
TEST_SIZE = 0.20
CANDIDATE_K = 10
MASKING_SETTINGS: tuple[str, ...] = ("direct", "aggressive")

ART_FIELDS: tuple[str, ...] = (
    "대중음악",
    "방송연예",
    "문학",
    "미술",
    "공예",
    "사진",
    "건축",
    "음악",
    "국악",
    "무용",
    "연극",
    "영화",
    "만화",
    "기타",
)

NARRATIVE_COLUMNS: tuple[str, ...] = (
    "persona",
    "professional_persona",
    "creative_world_persona",
    "network_persona",
    "living_persona",
    "support_persona",
    "family_persona",
    "sports_persona",
    "arts_persona",
    "travel_persona",
    "culinary_persona",
    "cultural_background",
    "skills_and_expertise",
    "hobbies_and_interests",
    "career_goals_and_ambitions",
)

ANCHOR_RECOVERY_TARGETS: tuple[str, ...] = (
    "art_field_primary",
    "age_band",
    "career_band",
    "employment_type",
    "is_freelance",
    "income_coarse",
    "has_contract_experience",
    "has_copyright",
    "had_career_break",
    "has_overseas_experience",
)

PAIR_ANCHOR_COLUMNS: tuple[str, ...] = (
    "sex",
    "age_band",
    "province",
    "education_level_pak",
    "career_band",
    "employment_type",
    "is_freelance",
    "has_secondary_job",
    "individual_art_income_bracket",
    "household_income_bracket",
    "has_contract_experience",
    "uses_standard_contract",
    "has_copyright",
    "had_career_break",
    "has_overseas_experience",
)

ANCHOR_LABELS: dict[str, str] = {
    "sex": "sex",
    "age_band": "age band",
    "province": "province",
    "education_level_pak": "education",
    "career_band": "career band",
    "employment_type": "employment",
    "is_freelance": "freelance",
    "has_secondary_job": "secondary job",
    "individual_art_income_bracket": "individual art income",
    "household_income_bracket": "household income",
    "has_contract_experience": "contract experience",
    "uses_standard_contract": "standard contract",
    "has_copyright": "copyright",
    "had_career_break": "career break",
    "has_overseas_experience": "overseas activity",
}

AGGRESSIVE_CUE_TERMS: tuple[str, ...] = (
    "10대",
    "20대",
    "30대",
    "40대",
    "50대",
    "60대",
    "70대 이상",
    "70대",
    "80대",
    "90대",
    "청년",
    "중년",
    "장년",
    "노년",
    "경력",
    "년 차",
    "년차",
    "년째",
    "데뷔",
    "입문",
    "전환",
    "전업",
    "겸업",
    "프리랜서",
    "외주",
    "부업",
    "생계",
    "수입",
    "소득",
    "생활비",
    "작업비",
    "창작비",
    "판매",
    "입금",
    "금액",
    "계약",
    "계약서",
    "표준계약",
    "저작권",
    "저작물",
    "저작인접권",
    "권리",
    "인세",
    "라이선스",
    "경력 단절",
    "경력단절",
    "공백",
    "휴식",
    "휴직",
    "복귀",
    "재진입",
    "중단",
    "돌봄",
    "육아",
    "건강",
    "해외",
    "국외",
    "외국",
    "국제",
    "레지던시",
    "투어",
    "페스티벌",
    "교류",
    "초청",
    "순회",
    "일본",
    "중국",
    "미국",
    "유럽",
    "프랑스",
    "독일",
)

AGGRESSIVE_CUE_PATTERN = re.compile(
    "|".join(re.escape(term) for term in sorted(AGGRESSIVE_CUE_TERMS, key=len, reverse=True))
)
NUMERIC_CUE_PATTERN = re.compile(
    r"\d+(?:\.\d+)?\s*[-~–]\s*\d+(?:\.\d+)?\s*(?:년|세|대|만원|천만원|백만원|원)"
    r"|\d+(?:\.\d+)?\s*(?:년째|년차|년간|개월|천만원|백만원|만원|세|대|년|억|원)"
)


def read_release() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH.relative_to(ROOT)}")
    df = pd.read_parquet(DATA_PATH)
    required = set(NARRATIVE_COLUMNS + PAIR_ANCHOR_COLUMNS + ("pak_uuid", "art_field_primary", "occupation"))
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{DATA_PATH.relative_to(ROOT)} missing columns: {missing}")
    df = df.copy()
    df["income_coarse"] = df["individual_art_income_bracket"].map(coarsen_income)
    if df["income_coarse"].isna().any():
        bad = sorted(df.loc[df["income_coarse"].isna(), "individual_art_income_bracket"].unique())
        raise ValueError(f"Unmapped income brackets: {bad}")
    return df


def coarsen_income(value: Any) -> str:
    mapping = {
        "없음": "none",
        "5백만원 미만": "low",
        "5백-1천만원 미만": "low",
        "1-2천만원 미만": "middle",
        "2-3천만원 미만": "middle",
        "3-4천만원 미만": "middle",
        "4-5천만원 미만": "high",
        "5-6천만원 미만": "high",
        "6천만원 이상": "high",
    }
    return mapping.get(str(value), "")


def join_narratives(df: pd.DataFrame) -> pd.Series:
    return df.loc[:, list(NARRATIVE_COLUMNS)].fillna("").astype(str).agg(" ".join, axis=1)


def mask_direct_labels(texts: pd.Series, df: pd.DataFrame) -> pd.Series:
    field_pattern = re.compile("|".join(re.escape(label) for label in sorted(ART_FIELDS, key=len, reverse=True)))
    occupations = df.set_index("pak_uuid")["occupation"].fillna("").astype(str).to_dict()

    def mask_one(pak_uuid: str, text: str) -> str:
        masked = field_pattern.sub("[FIELD]", text)
        occupation = occupations.get(pak_uuid, "")
        if occupation:
            masked = masked.replace(occupation, "[OCCUPATION]")
        return masked

    return pd.Series(
        [mask_one(str(pak_uuid), text) for pak_uuid, text in texts.items()],
        index=texts.index,
    )


def collect_anchor_value_terms(df: pd.DataFrame) -> tuple[str, ...]:
    value_columns = (
        "age_band",
        "career_band",
        "employment_type",
        "individual_art_income_bracket",
        "household_income_bracket",
    )
    terms: set[str] = set()
    for column in value_columns:
        for value in df[column].dropna().astype(str).unique():
            if value and value != "None":
                terms.add(value)
    return tuple(sorted(terms, key=len, reverse=True))


def mask_aggressive_cues(texts: pd.Series, df: pd.DataFrame) -> pd.Series:
    anchor_terms = collect_anchor_value_terms(df)
    anchor_pattern = re.compile("|".join(re.escape(term) for term in anchor_terms)) if anchor_terms else None

    def mask_one(text: str) -> str:
        masked = text
        if anchor_pattern is not None:
            masked = anchor_pattern.sub("[ANCHOR_VALUE]", masked)
        masked = NUMERIC_CUE_PATTERN.sub("[NUMERIC_CUE]", masked)
        masked = AGGRESSIVE_CUE_PATTERN.sub("[ANCHOR_CUE]", masked)
        return masked

    return texts.map(mask_one)


def make_masked_texts(df: pd.DataFrame, masking: str) -> pd.Series:
    all_texts = join_narratives(df)
    all_texts.index = df["pak_uuid"].astype(str)
    masked = mask_direct_labels(all_texts, df)
    if masking == "direct":
        return masked
    if masking == "aggressive":
        return mask_aggressive_cues(masked, df)
    raise ValueError(f"Unknown masking setting: {masking}")


def make_vectorizer(max_features: int = 60_000) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=max_features,
        sublinear_tf=True,
    )


def make_classifier() -> SGDClassifier:
    return SGDClassifier(
        class_weight="balanced",
        loss="log_loss",
        alpha=1e-5,
        max_iter=40,
        tol=1e-3,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )


def top_k_accuracy(probabilities: np.ndarray, classes: np.ndarray, y_true: pd.Series, k: int) -> float:
    top = np.argsort(probabilities, axis=1)[:, -k:]
    labels = classes[top]
    truth = y_true.to_numpy()
    return float(np.mean([truth[i] in labels[i] for i in range(len(truth))]))


def split_personas(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df["art_field_primary"],
    )
    return df.loc[train_idx].copy(), df.loc[test_idx].copy()


def evaluate_anchor_recovery(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    masked_text: pd.Series,
    masking: str,
) -> pd.DataFrame:
    train_texts = masked_text.loc[train_df["pak_uuid"].astype(str)]
    test_texts = masked_text.loc[test_df["pak_uuid"].astype(str)]

    vectorizer = make_vectorizer()
    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)
    rows: list[dict[str, Any]] = []

    for target in ANCHOR_RECOVERY_TARGETS:
        y_train = train_df[target].astype(str)
        y_test = test_df[target].astype(str)

        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(x_train, y_train)
        baseline_pred = baseline.predict(x_test)

        stratified = DummyClassifier(strategy="stratified", random_state=RANDOM_SEED)
        stratified.fit(x_train, y_train)
        stratified_pred = stratified.predict(x_test)

        model = make_classifier()
        model.fit(x_train, y_train)
        pred = model.predict(x_test)

        row = {
            "masking": masking,
            "task": target,
            "classes": int(y_train.nunique()),
            "train_n": int(len(y_train)),
            "test_n": int(len(y_test)),
            "majority_accuracy": float(accuracy_score(y_test, baseline_pred)),
            "majority_macro_f1": float(f1_score(y_test, baseline_pred, average="macro")),
            "stratified_accuracy": float(accuracy_score(y_test, stratified_pred)),
            "stratified_macro_f1": float(f1_score(y_test, stratified_pred, average="macro")),
            "model_accuracy": float(accuracy_score(y_test, pred)),
            "model_macro_f1": float(f1_score(y_test, pred, average="macro")),
            "model_top3_accuracy": float("nan"),
        }
        if hasattr(model, "predict_proba") and y_train.nunique() > 2:
            probabilities = model.predict_proba(x_test)
            row["model_top3_accuracy"] = top_k_accuracy(probabilities, model.classes_, y_test, k=min(3, y_train.nunique()))
        rows.append(row)

    result = pd.DataFrame(rows)
    return result


def value_to_text(value: Any) -> str:
    if pd.isna(value):
        return "not applicable"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    if str(value) == "None":
        return "not applicable"
    return str(value)


def anchor_block(row: pd.Series) -> str:
    parts = [f"{ANCHOR_LABELS[col]}={value_to_text(row[col])}" for col in PAIR_ANCHOR_COLUMNS]
    return "; ".join(parts)


def build_group_map(df: pd.DataFrame, columns: tuple[str, ...]) -> dict[tuple[str, ...], list[str]]:
    grouped: dict[tuple[str, ...], list[str]] = {}
    groupby_key: str | list[str] = columns[0] if len(columns) == 1 else list(columns)
    for values, group in df.groupby(groupby_key, dropna=False):
        key = tuple(str(v) for v in (values if isinstance(values, tuple) else (values,)))
        grouped[key] = group["pak_uuid"].astype(str).tolist()
    return grouped


def choose_partner_uuid(
    row: pd.Series,
    same_field_age: dict[tuple[str, ...], list[str]],
    same_field: dict[tuple[str, ...], list[str]],
    all_uuids: list[str],
    rng: np.random.Generator,
) -> str:
    own = str(row["pak_uuid"])
    keys = [
        (str(row["art_field_primary"]), str(row["age_band"])),
        (str(row["art_field_primary"]),),
    ]
    candidate_sets = [
        same_field_age.get(keys[0], []),
        same_field.get(keys[1], []),
        all_uuids,
    ]
    for candidates in candidate_sets:
        filtered = [uuid for uuid in candidates if uuid != own]
        if filtered:
            return str(rng.choice(filtered))
    raise RuntimeError("Could not choose a negative partner")


def build_pair_dataset(df: pd.DataFrame, masked_text: pd.Series, rng: np.random.Generator) -> pd.DataFrame:
    by_uuid = df.set_index("pak_uuid", drop=False)
    all_uuids = df["pak_uuid"].astype(str).tolist()
    same_field_age = build_group_map(df, ("art_field_primary", "age_band"))
    same_field = build_group_map(df, ("art_field_primary",))
    rows: list[dict[str, Any]] = []

    for row in df.itertuples(index=False):
        row_s = pd.Series(row._asdict())
        uuid = str(row_s["pak_uuid"])
        positive_anchor = anchor_block(row_s)
        positive_narrative = masked_text.loc[uuid]
        positive_text = f"ANCHORS: {positive_anchor}\nNARRATIVE: {positive_narrative}"
        rows.append(
            {
                "query_uuid": uuid,
                "candidate_uuid": uuid,
                "label": 1,
                "anchor_text": positive_anchor,
                "narrative_text": positive_narrative,
                "text": positive_text,
                "negative_type": "positive",
            }
        )

        partner_uuid = choose_partner_uuid(row_s, same_field_age, same_field, all_uuids, rng)
        negative_narrative = masked_text.loc[partner_uuid]
        negative_text = f"ANCHORS: {positive_anchor}\nNARRATIVE: {negative_narrative}"
        same_age = by_uuid.loc[partner_uuid, "age_band"] == row_s["age_band"]
        rows.append(
            {
                "query_uuid": uuid,
                "candidate_uuid": partner_uuid,
                "label": 0,
                "anchor_text": positive_anchor,
                "narrative_text": negative_narrative,
                "text": negative_text,
                "negative_type": "same_field_age" if same_age else "same_field",
            }
        )

    pair_df = pd.DataFrame(rows)
    return pair_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)


def rowwise_cosine(left: Any, right: Any) -> np.ndarray:
    return np.asarray(left.multiply(right).sum(axis=1)).ravel()


def best_threshold(scores: np.ndarray, labels: pd.Series) -> float:
    thresholds = np.unique(np.quantile(scores, np.linspace(0.05, 0.95, 91)))
    if len(thresholds) == 0:
        return 0.0
    y_true = labels.astype(int).to_numpy()
    best_score = -1.0
    best_value = float(thresholds[0])
    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)
        score = f1_score(y_true, pred, average="macro")
        if score > best_score:
            best_score = score
            best_value = float(threshold)
    return best_value


def lexical_similarity_baseline(train_pairs: pd.DataFrame, test_pairs: pd.DataFrame) -> dict[str, float]:
    vectorizer = make_vectorizer(max_features=90_000)
    vectorizer.fit(pd.concat([train_pairs["anchor_text"], train_pairs["narrative_text"]], ignore_index=True))
    train_scores = rowwise_cosine(
        vectorizer.transform(train_pairs["anchor_text"]),
        vectorizer.transform(train_pairs["narrative_text"]),
    )
    test_scores = rowwise_cosine(
        vectorizer.transform(test_pairs["anchor_text"]),
        vectorizer.transform(test_pairs["narrative_text"]),
    )
    threshold = best_threshold(train_scores, train_pairs["label"])
    pred = (test_scores >= threshold).astype(int)
    y_test = test_pairs["label"].astype(int)
    return {
        "lexical_similarity_accuracy": float(accuracy_score(y_test, pred)),
        "lexical_similarity_macro_f1": float(f1_score(y_test, pred, average="macro")),
        "lexical_similarity_auc": float(roc_auc_score(y_test, test_scores)),
        "lexical_similarity_threshold": threshold,
    }


def evaluate_pair_consistency(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    masked_text: pd.Series,
    masking: str,
) -> tuple[dict[str, Any], SGDClassifier, TfidfVectorizer, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    train_pairs = build_pair_dataset(train_df, masked_text, rng)
    test_pairs = build_pair_dataset(test_df, masked_text, rng)

    vectorizer = make_vectorizer(max_features=90_000)
    x_train = vectorizer.fit_transform(train_pairs["text"])
    x_test = vectorizer.transform(test_pairs["text"])
    y_train = train_pairs["label"].astype(int)
    y_test = test_pairs["label"].astype(int)

    model = make_classifier()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    score = model.predict_proba(x_test)[:, 1]

    rng_baseline = np.random.default_rng(RANDOM_SEED)
    baseline_pred = rng_baseline.integers(0, 2, size=len(y_test))
    row = {
        "masking": masking,
        "task": "hard_pair_consistency",
        "train_n": int(len(y_train)),
        "test_n": int(len(y_test)),
        "random_accuracy": float(accuracy_score(y_test, baseline_pred)),
        "random_macro_f1": float(f1_score(y_test, baseline_pred, average="macro")),
        "model_accuracy": float(accuracy_score(y_test, pred)),
        "model_macro_f1": float(f1_score(y_test, pred, average="macro")),
        "model_auc": float(roc_auc_score(y_test, score)),
        "same_field_age_negative_pct": float((test_pairs["negative_type"].eq("same_field_age")).mean() * 100),
    }
    row.update(lexical_similarity_baseline(train_pairs, test_pairs))
    predictions = test_pairs.loc[:, ["query_uuid", "candidate_uuid", "label", "negative_type"]].copy()
    predictions["masking"] = masking
    predictions["score"] = score
    predictions["prediction"] = pred
    return row, model, vectorizer, predictions


def choose_retrieval_candidates(
    row: pd.Series,
    df: pd.DataFrame,
    same_field_age: dict[tuple[str, ...], list[str]],
    same_field: dict[tuple[str, ...], list[str]],
    all_uuids: list[str],
    rng: np.random.Generator,
) -> list[str]:
    own = str(row["pak_uuid"])
    pools = [
        same_field_age.get((str(row["art_field_primary"]), str(row["age_band"])), []),
        same_field.get((str(row["art_field_primary"]),), []),
        all_uuids,
    ]
    decoys: list[str] = []
    for pool in pools:
        candidates = [uuid for uuid in pool if uuid != own and uuid not in decoys]
        needed = CANDIDATE_K - 1 - len(decoys)
        if needed <= 0:
            break
        if len(candidates) <= needed:
            decoys.extend(candidates)
        else:
            decoys.extend(rng.choice(candidates, size=needed, replace=False).astype(str).tolist())
    while len(decoys) < CANDIDATE_K - 1:
        sampled = str(rng.choice(all_uuids))
        if sampled != own and sampled not in decoys:
            decoys.append(sampled)
    candidates = [own] + decoys[: CANDIDATE_K - 1]
    rng.shuffle(candidates)
    return candidates


def evaluate_retrieval(
    test_df: pd.DataFrame,
    masked_text: pd.Series,
    pair_model: SGDClassifier,
    pair_vectorizer: TfidfVectorizer,
    masking: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    same_field_age = build_group_map(test_df, ("art_field_primary", "age_band"))
    same_field = build_group_map(test_df, ("art_field_primary",))
    all_uuids = test_df["pak_uuid"].astype(str).tolist()

    rows: list[dict[str, Any]] = []
    pair_texts: list[str] = []
    anchor_texts: list[str] = []
    narrative_texts: list[str] = []
    pair_meta: list[tuple[str, str, int]] = []

    for query_rank, row in enumerate(test_df.itertuples(index=False)):
        row_s = pd.Series(row._asdict())
        query_uuid = str(row_s["pak_uuid"])
        query_anchor = anchor_block(row_s)
        candidates = choose_retrieval_candidates(row_s, test_df, same_field_age, same_field, all_uuids, rng)
        for candidate_uuid in candidates:
            candidate_narrative = masked_text.loc[candidate_uuid]
            pair_texts.append(f"ANCHORS: {query_anchor}\nNARRATIVE: {candidate_narrative}")
            anchor_texts.append(query_anchor)
            narrative_texts.append(candidate_narrative)
            pair_meta.append((query_uuid, candidate_uuid, query_rank))

    scores = pair_model.predict_proba(pair_vectorizer.transform(pair_texts))[:, 1]
    lexical_vectorizer = make_vectorizer(max_features=90_000)
    lexical_vectorizer.fit(pd.Series([*anchor_texts, *narrative_texts]))
    lexical_scores = rowwise_cosine(
        lexical_vectorizer.transform(anchor_texts),
        lexical_vectorizer.transform(narrative_texts),
    )
    score_df = pd.DataFrame(pair_meta, columns=["query_uuid", "candidate_uuid", "query_rank"])
    score_df["score"] = scores
    score_df["lexical_similarity_score"] = lexical_scores

    ranks_arr = retrieval_ranks(score_df, "score")
    lexical_ranks_arr = retrieval_ranks(score_df, "lexical_similarity_score")
    for query_uuid, group in score_df.groupby("query_uuid", sort=False):
        sorted_group = group.sort_values("score", ascending=False).reset_index(drop=True)
        lexical_sorted_group = group.sort_values("lexical_similarity_score", ascending=False).reset_index(drop=True)
        lexical_rank_map = {
            candidate_uuid: int(rank) + 1
            for rank, candidate_uuid in enumerate(lexical_sorted_group["candidate_uuid"].tolist())
        }
        for candidate_rank, candidate in sorted_group.iterrows():
            rows.append(
                {
                    "masking": masking,
                    "query_uuid": query_uuid,
                    "candidate_uuid": candidate["candidate_uuid"],
                    "rank": int(candidate_rank) + 1,
                    "lexical_similarity_rank": lexical_rank_map[candidate["candidate_uuid"]],
                    "score": float(candidate["score"]),
                    "lexical_similarity_score": float(candidate["lexical_similarity_score"]),
                    "is_correct": candidate["candidate_uuid"] == query_uuid,
                }
            )

    harmonic = sum(1.0 / i for i in range(1, CANDIDATE_K + 1)) / CANDIDATE_K
    metrics = {
        "masking": masking,
        "task": f"{CANDIDATE_K}_way_anchor_to_narrative_retrieval",
        "queries": int(len(ranks_arr)),
        "candidates_per_query": CANDIDATE_K,
        "random_top1": 1 / CANDIDATE_K,
        "random_top3": min(3, CANDIDATE_K) / CANDIDATE_K,
        "random_mrr": harmonic,
        "lexical_similarity_top1": float(np.mean(lexical_ranks_arr == 1)),
        "lexical_similarity_top3": float(np.mean(lexical_ranks_arr <= 3)),
        "lexical_similarity_mrr": float(np.mean(1 / lexical_ranks_arr)),
        "model_top1": float(np.mean(ranks_arr == 1)),
        "model_top3": float(np.mean(ranks_arr <= 3)),
        "model_mrr": float(np.mean(1 / ranks_arr)),
        "median_rank": float(np.median(ranks_arr)),
    }
    return metrics, pd.DataFrame(rows)


def retrieval_ranks(score_df: pd.DataFrame, score_column: str) -> np.ndarray:
    ranks: list[int] = []
    for query_uuid, group in score_df.groupby("query_uuid", sort=False):
        sorted_group = group.sort_values(score_column, ascending=False).reset_index(drop=True)
        rank = int(sorted_group.index[sorted_group["candidate_uuid"].eq(query_uuid)][0]) + 1
        ranks.append(rank)
    return np.array(ranks)


def write_report(
    anchor_metrics: pd.DataFrame,
    pair_metrics: pd.DataFrame,
    retrieval_metrics: pd.DataFrame,
) -> None:
    summary_rows: list[dict[str, Any]] = []
    for masking in MASKING_SETTINGS:
        masked_anchor = anchor_metrics[anchor_metrics["masking"].eq(masking)]
        non_field = masked_anchor[masked_anchor["task"].ne("art_field_primary")]
        field = masked_anchor[masked_anchor["task"].eq("art_field_primary")].iloc[0]
        pair = pair_metrics[pair_metrics["masking"].eq(masking)].iloc[0]
        retrieval = retrieval_metrics[retrieval_metrics["masking"].eq(masking)].iloc[0]
        summary_rows.extend(
            [
                {
                    "masking": masking,
                    "task": "field_recovery_sanity",
                    "random_or_majority": field["majority_macro_f1"],
                    "stratified_or_lexical": field["stratified_macro_f1"],
                    "model": field["model_macro_f1"],
                    "metric": "macro-F1",
                },
                {
                    "masking": masking,
                    "task": "non_field_anchor_recovery_mean",
                    "random_or_majority": non_field["majority_macro_f1"].mean(),
                    "stratified_or_lexical": non_field["stratified_macro_f1"].mean(),
                    "model": non_field["model_macro_f1"].mean(),
                    "metric": "macro-F1",
                },
                {
                    "masking": masking,
                    "task": "hard_pair_consistency",
                    "random_or_majority": pair["random_macro_f1"],
                    "stratified_or_lexical": pair["lexical_similarity_macro_f1"],
                    "model": pair["model_macro_f1"],
                    "metric": "macro-F1",
                },
                {
                    "masking": masking,
                    "task": f"{CANDIDATE_K}_way_anchor_to_narrative_retrieval",
                    "random_or_majority": retrieval["random_mrr"],
                    "stratified_or_lexical": retrieval["lexical_similarity_mrr"],
                    "model": retrieval["model_mrr"],
                    "metric": "MRR",
                },
            ]
        )
    summary = pd.DataFrame(summary_rows)
    leakage = (
        anchor_metrics[anchor_metrics["task"].ne("art_field_primary")]
        .pivot(index="task", columns="masking", values="model_macro_f1")
        .reset_index()
    )
    leakage["delta_aggressive_minus_direct"] = leakage["aggressive"] - leakage["direct"]
    anchor_display = anchor_metrics.copy()
    anchor_display["model_top3_accuracy"] = anchor_display["model_top3_accuracy"].map(
        lambda value: "--" if pd.isna(value) else f"{value:.3f}"
    )

    lines = [
        "# PAK Downstream Persona-Use Tasks",
        "",
        f"- Source: `{DATA_PATH.relative_to(ROOT)}`",
        f"- Split: stratified {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)} train/test, seed {RANDOM_SEED}.",
        "- Text model: character 3-5 gram TF-IDF plus balanced logistic regression.",
        "- Direct masking: literal Korean art-field labels and row-level occupation strings are masked for all task inputs.",
        "- Aggressive masking: direct masking plus age/career/income/employment/contract/copyright/career-break/overseas cue terms, selected categorical anchor values, and numeric age/time/money expressions.",
        "- Hard pair/retrieval anchors exclude `art_field_primary` and `occupation`; decoys are drawn from the same art field and age band whenever possible.",
        "- Pair/retrieval baselines include random ranking and a lexical-similarity baseline between the structured anchor block and candidate narrative.",
        "- Interpretation: downstream persona-use benchmark and leakage audit for structured persona grounding, not a general Korean text-classification leaderboard.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".3f"),
        "",
        "Column note: `random_or_majority` is majority macro-F1 for anchor recovery, random macro-F1 for hard-pair consistency, and random-ranking MRR for retrieval. `stratified_or_lexical` is stratified-random macro-F1 for anchor recovery and lexical-similarity performance for pair/retrieval.",
        "",
        "## Leakage Audit: Anchor Recovery",
        "",
        leakage.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Anchor Recovery",
        "",
        anchor_display.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Hard Pair Consistency",
        "",
        pair_metrics.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## 10-Way Anchor-to-Narrative Retrieval",
        "",
        retrieval_metrics.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## LaTeX Summary Rows",
        "",
        "```tex",
    ]
    for row in summary.itertuples(index=False):
        lexical = "--" if math.isnan(row.stratified_or_lexical) else f"{row.stratified_or_lexical:.3f}"
        lines.append(
            f"{row.masking} & {row.task} & {row.metric} & "
            f"{row.random_or_majority:.3f} & {lexical} & {row.model:.3f} \\\\"
        )
    lines.extend(
        [
            "```",
            "",
            "## LaTeX Anchor Recovery Rows",
            "",
            "```tex",
        ]
    )
    for row in anchor_metrics.itertuples(index=False):
        top3 = "--" if math.isnan(row.model_top3_accuracy) else f"{row.model_top3_accuracy:.3f}"
        lines.append(
            f"{row.masking} & {row.task} & {row.classes} & {row.majority_macro_f1:.3f} & "
            f"{row.stratified_macro_f1:.3f} & {row.model_macro_f1:.3f} & {row.model_accuracy:.3f} & {top3} \\\\"
        )
    lines.extend(["```", ""])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = read_release()
    train_df, test_df = split_personas(df)
    all_df = pd.concat([train_df, test_df])

    anchor_frames: list[pd.DataFrame] = []
    pair_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    pair_prediction_frames: list[pd.DataFrame] = []
    retrieval_frames: list[pd.DataFrame] = []

    for masking in MASKING_SETTINGS:
        log.info("Running %s-masked anchor recovery tasks", masking)
        masked_text = make_masked_texts(all_df, masking)
        anchor_frames.append(evaluate_anchor_recovery(train_df, test_df, masked_text, masking))

        log.info("Running %s-masked hard pair consistency task", masking)
        pair_metrics, pair_model, pair_vectorizer, pair_predictions = evaluate_pair_consistency(
            train_df,
            test_df,
            masked_text,
            masking,
        )
        pair_rows.append(pair_metrics)
        pair_prediction_frames.append(pair_predictions)

        log.info("Running %s-masked %d-way anchor-to-narrative retrieval task", masking, CANDIDATE_K)
        retrieval_metrics, retrieval_predictions = evaluate_retrieval(
            test_df,
            masked_text,
            pair_model,
            pair_vectorizer,
            masking,
        )
        retrieval_rows.append(retrieval_metrics)
        retrieval_frames.append(retrieval_predictions)

    anchor_metrics = pd.concat(anchor_frames, ignore_index=True)
    pair_metrics_df = pd.DataFrame(pair_rows)
    retrieval_metrics_df = pd.DataFrame(retrieval_rows)
    anchor_metrics.to_csv(ANCHOR_RECOVERY_CSV, index=False)
    pd.concat(pair_prediction_frames, ignore_index=True).to_csv(PAIR_PREDICTIONS_CSV, index=False)
    pd.concat(retrieval_frames, ignore_index=True).to_csv(RETRIEVAL_CSV, index=False)

    write_report(anchor_metrics, pair_metrics_df, retrieval_metrics_df)
    log.info("Wrote %s", REPORT_PATH.relative_to(ROOT))
    log.info("Wrote %s", ANCHOR_RECOVERY_CSV.relative_to(ROOT))
    log.info("Wrote %s", PAIR_PREDICTIONS_CSV.relative_to(ROOT))
    log.info("Wrote %s", RETRIEVAL_CSV.relative_to(ROOT))
    log.info("Anchor recovery:\n%s", anchor_metrics.to_string(index=False))
    log.info("Pair consistency:\n%s", pair_metrics_df.to_string(index=False))
    log.info("Retrieval:\n%s", retrieval_metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
