"""Run extended downstream tasks for the PAK release.

This script complements ``run_downstream_persona_tasks.py`` with three
additional use cases that are closer to how a structured persona resource is
used in evaluation:

1. Held-out narrative-field retrieval: given the other persona fields, retrieve
   the matching held-out paragraph among hard same-field candidates.
2. Targeted counterfactual consistency: change one structured anchor and test
   whether a model can detect the mismatch with the narrative.
3. PAK-1K quality filtering: predict human-reference low-quality labels from
   the released persona text and anchors.

The tasks are lightweight, deterministic, and use only released project
artifacts.
"""

from __future__ import annotations

import logging
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import ConvergenceWarning

import run_downstream_persona_tasks as base

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parent.parent
ANNOTATION_PATH = ROOT / "data/eval/pak_1k_eval_annotations.csv"
REPORT_PATH = ROOT / "outputs/reports/downstream_extended_persona_tasks_260521.md"
HELDOUT_CSV = ROOT / "outputs/reports/downstream_heldout_field_retrieval_260521.csv"
COUNTERFACTUAL_CSV = ROOT / "outputs/reports/downstream_counterfactual_consistency_260521.csv"
QUALITY_CSV = ROOT / "outputs/reports/downstream_pak_1k_quality_filtering_260521.csv"

RANDOM_SEED = base.RANDOM_SEED
CANDIDATE_K = base.CANDIDATE_K
MASKING_SETTINGS = base.MASKING_SETTINGS

HELDOUT_TARGET_FIELDS: tuple[str, ...] = (
    "professional_persona",
    "creative_world_persona",
    "support_persona",
    "cultural_background",
    "skills_and_expertise",
    "career_goals_and_ambitions",
)

COUNTERFACTUAL_TARGETS: tuple[str, ...] = (
    "career_band",
    "employment_type",
    "individual_art_income_bracket",
    "has_contract_experience",
    "had_career_break",
    "has_overseas_experience",
)

QUALITY_DIMENSIONS: tuple[str, ...] = (
    "groundedness",
    "coherence",
    "plausibility",
    "fluency",
)

COUNTERFACTUAL_TRAIN_PER_VALUE_CAP = 1_000
COUNTERFACTUAL_TEST_PER_VALUE_CAP = 300
COUNTERFACTUAL_MIN_PER_VALUE = 50


def make_text_model(max_features: int = 90_000) -> tuple[TfidfVectorizer, SGDClassifier]:
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=max_features,
        sublinear_tf=True,
    )
    classifier = SGDClassifier(
        class_weight="balanced",
        loss="log_loss",
        alpha=1e-5,
        max_iter=30,
        tol=1e-3,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    return vectorizer, classifier


def make_masked_field_table(df: pd.DataFrame, masking: str) -> pd.DataFrame:
    fields = pd.DataFrame(index=df["pak_uuid"].astype(str))
    for column in base.NARRATIVE_COLUMNS:
        series = df.set_index(df["pak_uuid"].astype(str))[column].fillna("").astype(str)
        masked = base.mask_direct_labels(series, df)
        if masking == "aggressive":
            masked = base.mask_aggressive_cues(masked, df)
        elif masking != "direct":
            raise ValueError(f"Unknown masking setting: {masking}")
        fields[column] = masked
    return fields


def context_without_target(fields: pd.DataFrame, target: str) -> pd.Series:
    columns = [column for column in base.NARRATIVE_COLUMNS if column != target]
    return fields.loc[:, columns].fillna("").astype(str).agg(" ".join, axis=1)


def choose_same_field_candidates(
    row: pd.Series,
    split_df: pd.DataFrame,
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


def evaluate_heldout_field_retrieval(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fields: pd.DataFrame,
    masking: str,
) -> pd.DataFrame:
    same_field_age = base.build_group_map(test_df, ("art_field_primary", "age_band"))
    same_field = base.build_group_map(test_df, ("art_field_primary",))
    all_uuids = test_df["pak_uuid"].astype(str).tolist()
    uuid_to_row = test_df.set_index("pak_uuid", drop=False)
    rows: list[dict[str, Any]] = []
    train_corpus = fields.loc[train_df["pak_uuid"].astype(str), list(base.NARRATIVE_COLUMNS)].stack()
    vectorizer = base.make_vectorizer(max_features=90_000)
    vectorizer.fit(train_corpus)
    test_field_vectors = {
        column: vectorizer.transform(fields.loc[test_df["pak_uuid"].astype(str), column])
        for column in base.NARRATIVE_COLUMNS
    }

    for target_index, target in enumerate(HELDOUT_TARGET_FIELDS):
        log.info("Held-out retrieval target=%s masking=%s", target, masking)
        rng = np.random.default_rng(RANDOM_SEED + target_index)
        context_vectors = None
        for column, matrix in test_field_vectors.items():
            if column == target:
                continue
            context_vectors = matrix.copy() if context_vectors is None else context_vectors + matrix
        if context_vectors is None:
            raise RuntimeError("No context fields available for held-out retrieval")
        target_vectors = test_field_vectors[target]
        test_uuids = test_df["pak_uuid"].astype(str).tolist()
        uuid_to_position = {uuid: position for position, uuid in enumerate(test_uuids)}
        ranks: list[int] = []

        for query_position, uuid in enumerate(test_uuids):
            row = uuid_to_row.loc[uuid]
            candidates = choose_same_field_candidates(
                row=row,
                split_df=test_df,
                same_field_age=same_field_age,
                same_field=same_field,
                all_uuids=all_uuids,
                rng=rng,
            )
            candidate_positions = [uuid_to_position[candidate] for candidate in candidates]
            candidate_vectors = target_vectors[candidate_positions]
            scores = np.asarray(candidate_vectors.multiply(context_vectors[query_position]).sum(axis=1)).ravel()
            sorted_indices = np.argsort(-scores)
            rank = int(np.where(np.array(candidates)[sorted_indices] == uuid)[0][0]) + 1
            ranks.append(rank)

        ranks_arr = np.array(ranks)
        harmonic = sum(1.0 / i for i in range(1, CANDIDATE_K + 1)) / CANDIDATE_K
        rows.append(
            {
                "masking": masking,
                "target_field": target,
                "queries": int(len(ranks_arr)),
                "candidates_per_query": CANDIDATE_K,
                "random_top1": 1 / CANDIDATE_K,
                "random_top3": min(3, CANDIDATE_K) / CANDIDATE_K,
                "random_mrr": harmonic,
                "tfidf_top1": float(np.mean(ranks_arr == 1)),
                "tfidf_top3": float(np.mean(ranks_arr <= 3)),
                "tfidf_mrr": float(np.mean(1 / ranks_arr)),
                "median_rank": float(np.median(ranks_arr)),
            }
        )
    return pd.DataFrame(rows)


def unique_values(values: pd.Series) -> list[Any]:
    return [value for value in values.dropna().unique().tolist()]


def build_counterfactual_value_pools(split_df: pd.DataFrame, target: str) -> dict[str, Any]:
    by_field_age: dict[tuple[str, str], list[Any]] = {}
    by_field: dict[str, list[Any]] = {}
    for (field, age), group in split_df.groupby(["art_field_primary", "age_band"], dropna=False):
        by_field_age[(str(field), str(age))] = unique_values(group[target])
    for field, group in split_df.groupby("art_field_primary", dropna=False):
        by_field[str(field)] = unique_values(group[target])
    return {
        "by_field_age": by_field_age,
        "by_field": by_field,
        "all": unique_values(split_df[target]),
    }


def draw_different_value(pool: list[Any], current: Any, rng: np.random.Generator) -> Any | None:
    candidates = [value for value in pool if str(value) != str(current)]
    if not candidates:
        return None
    return rng.choice(candidates)


def choose_counterfactual_value(
    row: dict[str, Any],
    pools: dict[str, Any],
    target: str,
    rng: np.random.Generator,
) -> Any:
    current = row[target]
    if isinstance(current, (bool, np.bool_)):
        return not bool(current)

    candidate_pools = [
        pools["by_field_age"].get((str(row["art_field_primary"]), str(row["age_band"])), []),
        pools["by_field"].get(str(row["art_field_primary"]), []),
        pools["all"],
    ]
    for pool in candidate_pools:
        replacement = draw_different_value(pool, current, rng)
        if replacement is not None:
            return replacement
    raise ValueError(f"No counterfactual value available for {target}={current!r}")


def anchor_block_from_record(record: dict[str, Any]) -> str:
    parts = [
        f"{base.ANCHOR_LABELS[column]}={base.value_to_text(record.get(column))}"
        for column in base.PAIR_ANCHOR_COLUMNS
    ]
    return "; ".join(parts)


def counterfactual_anchor_block(row: dict[str, Any], target: str, replacement: Any | None = None) -> str:
    updated = dict(row)
    if replacement is not None:
        updated[target] = replacement
    return anchor_block_from_record(updated)


def build_counterfactual_pairs(
    split_df: pd.DataFrame,
    masked_text: pd.Series,
    target: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pools = build_counterfactual_value_pools(split_df, target)
    text_lookup = masked_text.to_dict()
    for row in split_df.to_dict("records"):
        uuid = str(row["pak_uuid"])
        narrative = text_lookup[uuid]
        positive_anchor = counterfactual_anchor_block(row, target)
        replacement = choose_counterfactual_value(row, pools, target, rng)
        negative_anchor = counterfactual_anchor_block(row, target, replacement)
        rows.append(
            {
                "pak_uuid": uuid,
                "target": target,
                "label": 1,
                "anchor_text": positive_anchor,
                "narrative_text": narrative,
                "text": f"ANCHORS: {positive_anchor}\nNARRATIVE: {narrative}",
                "replacement_value": "",
            }
        )
        rows.append(
            {
                "pak_uuid": uuid,
                "target": target,
                "label": 0,
                "anchor_text": negative_anchor,
                "narrative_text": narrative,
                "text": f"ANCHORS: {negative_anchor}\nNARRATIVE: {narrative}",
                "replacement_value": str(replacement),
            }
        )
    return pd.DataFrame(rows).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)


def fit_predict_binary(
    train_text: pd.Series,
    train_label: pd.Series,
    test_text: pd.Series,
    test_label: pd.Series,
    max_features: int = 90_000,
) -> dict[str, float]:
    vectorizer, classifier = make_text_model(max_features=max_features)
    x_train = vectorizer.fit_transform(train_text)
    x_test = vectorizer.transform(test_text)
    classifier.fit(x_train, train_label)
    pred = classifier.predict(x_test)
    score = classifier.predict_proba(x_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(test_label, pred)),
        "macro_f1": float(f1_score(test_label, pred, average="macro")),
        "auc": float(roc_auc_score(test_label, score)),
        "average_precision": float(average_precision_score(test_label, score)),
    }


def evaluate_counterfactual_consistency(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    masked_text: pd.Series,
    masking: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target_index, target in enumerate(COUNTERFACTUAL_TARGETS):
        log.info("Counterfactual target=%s masking=%s", target, masking)
        rng = np.random.default_rng(RANDOM_SEED + target_index)
        target_train_df = balanced_counterfactual_source(
            train_df,
            target,
            cap_per_value=COUNTERFACTUAL_TRAIN_PER_VALUE_CAP,
            seed_offset=target_index,
        )
        target_test_df = balanced_counterfactual_source(
            test_df,
            target,
            cap_per_value=COUNTERFACTUAL_TEST_PER_VALUE_CAP,
            seed_offset=100 + target_index,
        )
        train_pairs = build_counterfactual_pairs(target_train_df, masked_text, target, rng)
        test_pairs = build_counterfactual_pairs(target_test_df, masked_text, target, rng)
        y_train = train_pairs["label"].astype(int)
        y_test = test_pairs["label"].astype(int)

        dummy = DummyClassifier(strategy="stratified", random_state=RANDOM_SEED + target_index)
        dummy.fit(train_pairs[["anchor_text"]], y_train)
        dummy_pred = dummy.predict(test_pairs[["anchor_text"]])

        anchor_only = fit_predict_binary(
            train_pairs["anchor_text"],
            y_train,
            test_pairs["anchor_text"],
            y_test,
            max_features=15_000,
        )
        full = fit_predict_binary(train_pairs["text"], y_train, test_pairs["text"], y_test, max_features=35_000)
        rows.append(
            {
                "masking": masking,
                "target": target,
                "train_n": int(len(train_pairs)),
                "test_n": int(len(test_pairs)),
                "random_macro_f1": float(f1_score(y_test, dummy_pred, average="macro")),
                "anchor_only_macro_f1": anchor_only["macro_f1"],
                "anchor_only_auc": anchor_only["auc"],
                "model_accuracy": full["accuracy"],
                "model_macro_f1": full["macro_f1"],
                "model_auc": full["auc"],
                "model_average_precision": full["average_precision"],
                "gain_over_anchor_only": full["macro_f1"] - anchor_only["macro_f1"],
            }
        )
    return pd.DataFrame(rows)


def balanced_counterfactual_source(
    df: pd.DataFrame,
    target: str,
    cap_per_value: int,
    seed_offset: int,
) -> pd.DataFrame:
    counts = df[target].value_counts(dropna=False)
    per_value = int(min(cap_per_value, counts.min()))
    if per_value < COUNTERFACTUAL_MIN_PER_VALUE:
        raise ValueError(
            f"Target {target} has only {per_value} rows per value after balancing; "
            f"minimum is {COUNTERFACTUAL_MIN_PER_VALUE}"
        )
    frames: list[pd.DataFrame] = []
    for value, group in df.groupby(target, dropna=False):
        frames.append(group.sample(n=per_value, random_state=RANDOM_SEED + seed_offset))
    return pd.concat(frames, ignore_index=True).sample(frac=1.0, random_state=RANDOM_SEED + seed_offset)


def load_quality_table(df: pd.DataFrame) -> pd.DataFrame:
    if not ANNOTATION_PATH.exists():
        raise FileNotFoundError(f"Missing {ANNOTATION_PATH.relative_to(ROOT)}")
    annotations = pd.read_csv(ANNOTATION_PATH)
    human = annotations[
        annotations["annotator"].eq("human_anonymous")
        & annotations["dimension"].isin(QUALITY_DIMENSIONS)
    ].copy()
    score_wide = human.pivot(index="pak_uuid", columns="dimension", values="score")
    missing_dimensions = sorted(set(QUALITY_DIMENSIONS) - set(score_wide.columns))
    if missing_dimensions:
        raise ValueError(f"Missing human quality dimensions: {missing_dimensions}")
    quality_df = df[df["pak_uuid"].isin(score_wide.index)].copy()
    quality_df = quality_df.merge(score_wide.reset_index(), on="pak_uuid", how="inner")
    if len(quality_df) != 1000:
        raise ValueError(f"Expected 1000 PAK-1K rows, found {len(quality_df)}")
    quality_df["any_low_score"] = quality_df.loc[:, list(QUALITY_DIMENSIONS)].le(2).any(axis=1)
    return quality_df


def quality_feature_text(df: pd.DataFrame) -> pd.Series:
    anchors = df.apply(base.anchor_block, axis=1)
    narratives = base.join_narratives(df)
    return "ANCHORS: " + anchors.astype(str) + "\nNARRATIVE: " + narratives.astype(str)


def cross_validated_quality_metrics(
    text: pd.Series,
    label: pd.Series,
    target: str,
) -> dict[str, Any]:
    y = label.astype(int).to_numpy()
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives < 5 or negatives < 5:
        raise ValueError(f"Target {target} has too few examples: {positives} positive, {negatives} negative")

    splits = min(5, positives, negatives)
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=RANDOM_SEED)
    model_pred = np.zeros(len(y), dtype=int)
    model_score = np.zeros(len(y), dtype=float)
    dummy_pred = np.zeros(len(y), dtype=int)

    for fold, (train_idx, test_idx) in enumerate(cv.split(text, y)):
        x_train = text.iloc[train_idx]
        x_test = text.iloc[test_idx]
        y_train = y[train_idx]

        vectorizer, classifier = make_text_model(max_features=60_000)
        train_matrix = vectorizer.fit_transform(x_train)
        test_matrix = vectorizer.transform(x_test)
        classifier.fit(train_matrix, y_train)
        model_pred[test_idx] = classifier.predict(test_matrix)
        model_score[test_idx] = classifier.predict_proba(test_matrix)[:, 1]

        dummy = DummyClassifier(strategy="stratified", random_state=RANDOM_SEED + fold)
        dummy.fit(np.zeros((len(train_idx), 1)), y_train)
        dummy_pred[test_idx] = dummy.predict(np.zeros((len(test_idx), 1)))

    majority_label = int(np.round(np.mean(y)))
    majority_pred = np.full(len(y), majority_label)
    return {
        "target": target,
        "n": int(len(y)),
        "positives": positives,
        "prevalence": positives / len(y),
        "majority_macro_f1": float(f1_score(y, majority_pred, average="macro")),
        "stratified_macro_f1": float(f1_score(y, dummy_pred, average="macro")),
        "model_macro_f1": float(f1_score(y, model_pred, average="macro")),
        "model_auc": float(roc_auc_score(y, model_score)),
        "model_average_precision": float(average_precision_score(y, model_score)),
    }


def evaluate_quality_filtering(df: pd.DataFrame) -> pd.DataFrame:
    quality_df = load_quality_table(df)
    text = quality_feature_text(quality_df)
    rows: list[dict[str, Any]] = []
    targets = {"any_low_score": quality_df["any_low_score"]}
    for dimension in QUALITY_DIMENSIONS:
        targets[f"{dimension}_low"] = quality_df[dimension].le(2)
    for target, label in targets.items():
        rows.append(cross_validated_quality_metrics(text, label, target))
    return pd.DataFrame(rows)


def write_report(
    heldout: pd.DataFrame,
    counterfactual: pd.DataFrame,
    quality: pd.DataFrame,
) -> None:
    heldout_summary = (
        heldout.groupby("masking", as_index=False)
        .agg(
            random_mrr=("random_mrr", "mean"),
            tfidf_mrr=("tfidf_mrr", "mean"),
            tfidf_top1=("tfidf_top1", "mean"),
            tfidf_top3=("tfidf_top3", "mean"),
        )
        .assign(task="heldout_narrative_field_retrieval")
    )
    counterfactual_summary = (
        counterfactual.groupby("masking", as_index=False)
        .agg(
            random_macro_f1=("random_macro_f1", "mean"),
            anchor_only_macro_f1=("anchor_only_macro_f1", "mean"),
            model_macro_f1=("model_macro_f1", "mean"),
            model_auc=("model_auc", "mean"),
            gain_over_anchor_only=("gain_over_anchor_only", "mean"),
        )
        .assign(task="targeted_counterfactual_consistency")
    )

    lines = [
        "# PAK Extended Downstream Persona-Use Tasks",
        "",
        f"- Source personas: `{base.DATA_PATH.relative_to(ROOT)}`",
        f"- PAK-1K annotations: `{ANNOTATION_PATH.relative_to(ROOT)}`",
        f"- Split: stratified {int((1 - base.TEST_SIZE) * 100)}/{int(base.TEST_SIZE * 100)} train/test for release-scale tasks, seed {RANDOM_SEED}.",
        "- Text model: character 3-5 gram TF-IDF plus balanced logistic regression/SGD classifier.",
        "- Direct and aggressive masking follow `scripts/run_downstream_persona_tasks.py`.",
        f"- Counterfactual consistency uses value-balanced source personas per target anchor (up to {COUNTERFACTUAL_TRAIN_PER_VALUE_CAP:,} train and {COUNTERFACTUAL_TEST_PER_VALUE_CAP:,} test personas per target value).",
        "- Interpretation: use-case diagnostics for persona internal consistency, counterfactual evaluation, and quality filtering; not a general Korean NLP leaderboard.",
        "",
        "## Summary: Held-out Narrative-Field Retrieval",
        "",
        heldout_summary.loc[:, ["masking", "task", "random_mrr", "tfidf_mrr", "tfidf_top1", "tfidf_top3"]].to_markdown(
            index=False,
            floatfmt=".3f",
        ),
        "",
        "The query contains all narrative fields except the target field. Candidates contain the true held-out field plus nine decoys drawn from the same art field and age band whenever possible.",
        "",
        "## Summary: Targeted Counterfactual Consistency",
        "",
        counterfactual_summary.loc[
            :,
            [
                "masking",
                "task",
                "random_macro_f1",
                "anchor_only_macro_f1",
                "model_macro_f1",
                "model_auc",
                "gain_over_anchor_only",
            ],
        ].to_markdown(index=False, floatfmt=".3f"),
        "",
        "The negative example changes one structured anchor while keeping the narrative fixed. The anchor-only baseline is trained on the counterfactual anchor block without the narrative.",
        "The value-balanced counterfactual task is retained as a negative stress test: the simple TF-IDF model does not beat the anchor-only control, so these rows should not be interpreted as solved counterfactual reasoning.",
        "",
        "## Summary: PAK-1K Quality Filtering",
        "",
        quality.to_markdown(index=False, floatfmt=".3f"),
        "",
        "Quality filtering uses five-fold stratified cross-validation over the 1,000 PAK-1K personas. Low quality means a human-reference score of 1 or 2; `any_low_score` is positive when any of the four dimensions is low.",
        "",
        "## Held-out Narrative-Field Retrieval",
        "",
        heldout.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Targeted Counterfactual Consistency",
        "",
        counterfactual.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## LaTeX Summary Rows",
        "",
        "```tex",
    ]
    for row in heldout_summary.itertuples(index=False):
        lines.append(
            f"Held-out field retrieval & {row.masking.title()} & {row.random_mrr:.3f} & "
            f"{row.tfidf_mrr:.3f} & MRR \\\\"
        )
    for row in counterfactual_summary.itertuples(index=False):
        lines.append(
            f"Counterfactual consistency & {row.masking.title()} & {row.anchor_only_macro_f1:.3f} & "
            f"{row.model_macro_f1:.3f} & Macro-F1 \\\\"
        )
    any_low = quality[quality["target"].eq("any_low_score")].iloc[0]
    lines.append(
        f"PAK-1K low-quality filtering & 5-fold CV & {any_low.majority_macro_f1:.3f} & "
        f"{any_low.model_macro_f1:.3f} & Macro-F1 \\\\"
    )
    lines.extend(["```", ""])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = base.read_release()
    train_df, test_df = base.split_personas(df)
    all_df = pd.concat([train_df, test_df])

    heldout_frames: list[pd.DataFrame] = []
    counterfactual_frames: list[pd.DataFrame] = []

    for masking in MASKING_SETTINGS:
        log.info("Preparing %s-masked field table", masking)
        fields = make_masked_field_table(all_df, masking)
        log.info("Running %s-masked held-out narrative-field retrieval", masking)
        heldout_frames.append(evaluate_heldout_field_retrieval(train_df, test_df, fields, masking))

        log.info("Running %s-masked targeted counterfactual consistency", masking)
        masked_text = base.make_masked_texts(all_df, masking)
        counterfactual_frames.append(
            evaluate_counterfactual_consistency(
                train_df,
                test_df,
                masked_text,
                masking,
            )
        )

    log.info("Running PAK-1K quality filtering")
    quality = evaluate_quality_filtering(df)

    heldout = pd.concat(heldout_frames, ignore_index=True)
    counterfactual = pd.concat(counterfactual_frames, ignore_index=True)
    HELDOUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    heldout.to_csv(HELDOUT_CSV, index=False)
    counterfactual.to_csv(COUNTERFACTUAL_CSV, index=False)
    quality.to_csv(QUALITY_CSV, index=False)
    write_report(heldout, counterfactual, quality)

    log.info("Wrote %s", REPORT_PATH.relative_to(ROOT))
    log.info("Wrote %s", HELDOUT_CSV.relative_to(ROOT))
    log.info("Wrote %s", COUNTERFACTUAL_CSV.relative_to(ROOT))
    log.info("Wrote %s", QUALITY_CSV.relative_to(ROOT))
    log.info("Held-out retrieval:\n%s", heldout.to_string(index=False))
    log.info("Counterfactual consistency:\n%s", counterfactual.to_string(index=False))
    log.info("Quality filtering:\n%s", quality.to_string(index=False))


if __name__ == "__main__":
    main()
