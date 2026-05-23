"""Strong-encoder rerun of the hard downstream persona-use tasks.

This script reuses the exact data pipeline of ``run_downstream_persona_tasks.py``
and ``run_downstream_extended_tasks.py`` (release loading, train/test split,
direct/aggressive masking, pair/retrieval/counterfactual dataset construction)
and swaps ONLY the model layer from character TF-IDF to dragonkue/BGE-m3-ko
sentence embeddings. Baselines (random, majority, stratified, lexical-similarity,
anchor-only) are left untouched and are read from the existing TF-IDF runs, so the
comparison is honest: only the "Model" column changes.

Tasks covered (the ones where TF-IDF does not clearly beat the baselines):
  * non-field anchor recovery (mean macro-F1)
  * field-recovery sanity check
  * hard pair consistency (macro-F1)
  * 10-way anchor-to-narrative retrieval (MRR)
  * targeted counterfactual consistency (per-target macro-F1 and gain over the
    anchor-only control)

Embedding-based pair scoring uses a standard sentence-pair head: features are the
concatenation [a, n, a*n, |a-n|] of the L2-normalised anchor-block embedding a and
narrative embedding n, fed to a balanced logistic regression. Retrieval reuses the
trained pair head. Counterfactual reuses it per target, with an anchor-only control
that embeds the anchor block alone.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_downstream_persona_tasks as base  # noqa: E402
import run_downstream_extended_tasks as ext  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT = base.ROOT
REPORT_PATH = ROOT / "outputs/reports/downstream_bge_strong_model_260522.md"
EMBED_MODEL = "dragonkue/BGE-m3-ko"
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 64
RANDOM_SEED = base.RANDOM_SEED
CANDIDATE_K = base.CANDIDATE_K


def load_encoder():
    from sentence_transformers import SentenceTransformer

    log.info("Loading encoder %s (max_seq_length=%d)", EMBED_MODEL, MAX_SEQ_LENGTH)
    model = SentenceTransformer(EMBED_MODEL, local_files_only=True)
    model.max_seq_length = MAX_SEQ_LENGTH
    return model


class Embedder:
    """Encode-once cache keyed by raw text string."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.cache: dict[str, np.ndarray] = {}

    def encode(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if missing:
            vectors = self.model.encode(
                missing,
                batch_size=BATCH_SIZE,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for text, vec in zip(missing, vectors):
                self.cache[text] = vec.astype(np.float32)
        return np.vstack([self.cache[t] for t in texts])


def pair_features(anchor_emb: np.ndarray, narrative_emb: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [anchor_emb, narrative_emb, anchor_emb * narrative_emb, np.abs(anchor_emb - narrative_emb)],
        axis=1,
    )


def logistic() -> LogisticRegression:
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        C=1.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


# --------------------------------------------------------------------------- #
# Task 1: anchor recovery (non-field mean + field sanity)
# --------------------------------------------------------------------------- #
def run_anchor_recovery(
    embedder: Embedder,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    masked_text: pd.Series,
    masking: str,
) -> pd.DataFrame:
    train_uuids = train_df["pak_uuid"].astype(str).tolist()
    test_uuids = test_df["pak_uuid"].astype(str).tolist()
    x_train = embedder.encode([masked_text.loc[u] for u in train_uuids])
    x_test = embedder.encode([masked_text.loc[u] for u in test_uuids])

    rows: list[dict[str, Any]] = []
    for target in base.ANCHOR_RECOVERY_TARGETS:
        y_train = train_df[target].astype(str)
        y_test = test_df[target].astype(str)
        model = logistic()
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        rows.append(
            {
                "masking": masking,
                "task": target,
                "bge_accuracy": float(accuracy_score(y_test, pred)),
                "bge_macro_f1": float(f1_score(y_test, pred, average="macro")),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Task 2 + 3: hard pair consistency + 10-way retrieval
# --------------------------------------------------------------------------- #
def fit_pair_model(
    embedder: Embedder, train_pairs: pd.DataFrame
) -> LogisticRegression:
    a = embedder.encode(train_pairs["anchor_text"].tolist())
    n = embedder.encode(train_pairs["narrative_text"].tolist())
    model = logistic()
    model.fit(pair_features(a, n), train_pairs["label"].astype(int))
    return model


def run_pair_consistency(
    embedder: Embedder,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    masked_text: pd.Series,
    masking: str,
) -> tuple[dict[str, Any], LogisticRegression]:
    rng = np.random.default_rng(RANDOM_SEED)
    train_pairs = base.build_pair_dataset(train_df, masked_text, rng)
    test_pairs = base.build_pair_dataset(test_df, masked_text, rng)
    model = fit_pair_model(embedder, train_pairs)

    a = embedder.encode(test_pairs["anchor_text"].tolist())
    n = embedder.encode(test_pairs["narrative_text"].tolist())
    score = model.predict_proba(pair_features(a, n))[:, 1]
    pred = (score >= 0.5).astype(int)
    y_test = test_pairs["label"].astype(int)
    row = {
        "masking": masking,
        "task": "hard_pair_consistency",
        "bge_accuracy": float(accuracy_score(y_test, pred)),
        "bge_macro_f1": float(f1_score(y_test, pred, average="macro")),
        "bge_auc": float(roc_auc_score(y_test, score)),
    }
    return row, model


def run_retrieval(
    embedder: Embedder,
    test_df: pd.DataFrame,
    masked_text: pd.Series,
    pair_model: LogisticRegression,
    masking: str,
) -> dict[str, Any]:
    rng = np.random.default_rng(RANDOM_SEED)
    same_field_age = base.build_group_map(test_df, ("art_field_primary", "age_band"))
    same_field = base.build_group_map(test_df, ("art_field_primary",))
    all_uuids = test_df["pak_uuid"].astype(str).tolist()

    anchor_texts: list[str] = []
    narrative_texts: list[str] = []
    meta: list[tuple[int, str, str]] = []
    for query_rank, row in enumerate(test_df.itertuples(index=False)):
        row_s = pd.Series(row._asdict())
        query_uuid = str(row_s["pak_uuid"])
        query_anchor = base.anchor_block(row_s)
        candidates = base.choose_retrieval_candidates(
            row_s, test_df, same_field_age, same_field, all_uuids, rng
        )
        for candidate_uuid in candidates:
            anchor_texts.append(query_anchor)
            narrative_texts.append(masked_text.loc[candidate_uuid])
            meta.append((query_rank, query_uuid, candidate_uuid))

    a = embedder.encode(anchor_texts)
    n = embedder.encode(narrative_texts)
    scores = pair_model.predict_proba(pair_features(a, n))[:, 1]
    score_df = pd.DataFrame(meta, columns=["query_rank", "query_uuid", "candidate_uuid"])
    score_df["score"] = scores

    ranks: list[int] = []
    for _, group in score_df.groupby("query_uuid", sort=False):
        ordered = group.sort_values("score", ascending=False).reset_index(drop=True)
        rank = int(ordered.index[ordered["candidate_uuid"].eq(ordered["query_uuid"])][0]) + 1
        ranks.append(rank)
    ranks_arr = np.array(ranks)
    return {
        "masking": masking,
        "task": f"{CANDIDATE_K}_way_retrieval",
        "bge_top1": float(np.mean(ranks_arr == 1)),
        "bge_top3": float(np.mean(ranks_arr <= 3)),
        "bge_mrr": float(np.mean(1 / ranks_arr)),
    }


# --------------------------------------------------------------------------- #
# Task 4: targeted counterfactual consistency
# --------------------------------------------------------------------------- #
def run_counterfactual(
    embedder: Embedder,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    masked_text: pd.Series,
    masking: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target_index, target in enumerate(ext.COUNTERFACTUAL_TARGETS):
        rng = np.random.default_rng(RANDOM_SEED + target_index)
        train_src = ext.balanced_counterfactual_source(
            train_df, target, cap_per_value=ext.COUNTERFACTUAL_TRAIN_PER_VALUE_CAP, seed_offset=target_index
        )
        test_src = ext.balanced_counterfactual_source(
            test_df, target, cap_per_value=ext.COUNTERFACTUAL_TEST_PER_VALUE_CAP, seed_offset=100 + target_index
        )
        train_pairs = ext.build_counterfactual_pairs(train_src, masked_text, target, rng)
        test_pairs = ext.build_counterfactual_pairs(test_src, masked_text, target, rng)
        y_train = train_pairs["label"].astype(int)
        y_test = test_pairs["label"].astype(int)

        a_tr = embedder.encode(train_pairs["anchor_text"].tolist())
        n_tr = embedder.encode(train_pairs["narrative_text"].tolist())
        a_te = embedder.encode(test_pairs["anchor_text"].tolist())
        n_te = embedder.encode(test_pairs["narrative_text"].tolist())

        full = logistic()
        full.fit(pair_features(a_tr, n_tr), y_train)
        full_pred = full.predict(pair_features(a_te, n_te))
        full_f1 = float(f1_score(y_test, full_pred, average="macro"))

        anchor_only = logistic()
        anchor_only.fit(a_tr, y_train)
        anchor_pred = anchor_only.predict(a_te)
        anchor_f1 = float(f1_score(y_test, anchor_pred, average="macro"))

        rows.append(
            {
                "masking": masking,
                "target": target,
                "bge_anchor_only_macro_f1": anchor_f1,
                "bge_model_macro_f1": full_f1,
                "bge_gain_over_anchor_only": full_f1 - anchor_f1,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    embedder = Embedder(load_encoder())
    df = base.read_release()
    train_df, test_df = base.split_personas(df)
    all_df = pd.concat([train_df, test_df])

    anchor_frames: list[pd.DataFrame] = []
    pair_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    cf_frames: list[pd.DataFrame] = []

    for masking in base.MASKING_SETTINGS:
        log.info("=== masking=%s ===", masking)
        masked_text = base.make_masked_texts(all_df, masking)

        log.info("[%s] anchor recovery", masking)
        anchor_frames.append(run_anchor_recovery(embedder, train_df, test_df, masked_text, masking))

        log.info("[%s] hard pair consistency", masking)
        pair_row, pair_model = run_pair_consistency(embedder, train_df, test_df, masked_text, masking)
        pair_rows.append(pair_row)

        log.info("[%s] 10-way retrieval", masking)
        retrieval_rows.append(run_retrieval(embedder, test_df, masked_text, pair_model, masking))

        log.info("[%s] counterfactual consistency", masking)
        cf_frames.append(run_counterfactual(embedder, train_df, test_df, masked_text, masking))

    anchor = pd.concat(anchor_frames, ignore_index=True)
    pair = pd.DataFrame(pair_rows)
    retrieval = pd.DataFrame(retrieval_rows)
    counterfactual = pd.concat(cf_frames, ignore_index=True)

    non_field = (
        anchor[anchor["task"].ne("art_field_primary")]
        .groupby("masking", as_index=False)["bge_macro_f1"]
        .mean()
        .rename(columns={"bge_macro_f1": "non_field_anchor_recovery_mean"})
    )
    field = anchor[anchor["task"].eq("art_field_primary")].loc[:, ["masking", "bge_macro_f1"]].rename(
        columns={"bge_macro_f1": "field_recovery_macro_f1"}
    )
    cf_summary = counterfactual.groupby("masking", as_index=False).agg(
        bge_anchor_only_macro_f1=("bge_anchor_only_macro_f1", "mean"),
        bge_model_macro_f1=("bge_model_macro_f1", "mean"),
        bge_gain_over_anchor_only=("bge_gain_over_anchor_only", "mean"),
    )

    lines = [
        "# PAK Downstream Hard Tasks — BGE-m3-ko strong-encoder rerun",
        "",
        f"- Encoder: `{EMBED_MODEL}`, max_seq_length={MAX_SEQ_LENGTH}, L2-normalised.",
        "- Same data pipeline, split (seed %d), and masking as the TF-IDF runs; only the model changed." % RANDOM_SEED,
        "- Pair head: logistic on [a, n, a*n, |a-n|]. Retrieval reuses the pair head. Counterfactual anchor-only control embeds the anchor block alone.",
        "- Note: narratives are truncated at %d tokens; baselines are unchanged from the TF-IDF runs." % MAX_SEQ_LENGTH,
        "",
        "## Field recovery sanity",
        "",
        field.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Non-field anchor recovery (mean macro-F1)",
        "",
        non_field.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Hard pair consistency (macro-F1)",
        "",
        pair.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## 10-way anchor-to-narrative retrieval (MRR)",
        "",
        retrieval.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Counterfactual consistency (mean over targets)",
        "",
        cf_summary.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Counterfactual consistency (per target)",
        "",
        counterfactual.to_markdown(index=False, floatfmt=".3f"),
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote %s", REPORT_PATH.relative_to(ROOT))
    log.info("Field recovery:\n%s", field.to_string(index=False))
    log.info("Non-field anchor recovery:\n%s", non_field.to_string(index=False))
    log.info("Pair consistency:\n%s", pair.to_string(index=False))
    log.info("Retrieval:\n%s", retrieval.to_string(index=False))
    log.info("Counterfactual summary:\n%s", cf_summary.to_string(index=False))
    log.info("Counterfactual per target:\n%s", counterfactual.to_string(index=False))


if __name__ == "__main__":
    main()
