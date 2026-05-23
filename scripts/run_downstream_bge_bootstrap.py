"""Bootstrap 95% confidence intervals for the BGE-m3-ko hard downstream tasks.

This adds uncertainty quantification to the point estimates reported by
``run_downstream_bge.py`` (Table 6 of the paper) without changing the data
pipeline, the split, the masking, or the model. The design is the standard
"fixed model, resampled test set" bootstrap: every model is trained exactly
once on the original training split, per-instance predictions are computed once
on the original test split, and only the test instances are resampled with
replacement to estimate the sampling variance of each aggregate metric.

Covered rows (the load-bearing BGE numbers):
  * non-field anchor recovery mean macro-F1 (resample test personas)
  * hard pair consistency macro-F1 (resample test pairs)
  * 10-way anchor-to-narrative retrieval MRR (resample test queries)
  * targeted counterfactual mean model macro-F1 and mean gain over the
    anchor-only control (resample test pairs within each target)

Bootstrap: B replicates, percentile 95% interval [2.5, 97.5]. Seeded from the
base RANDOM_SEED so the intervals are reproducible.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_downstream_persona_tasks as base  # noqa: E402
import run_downstream_extended_tasks as ext  # noqa: E402
import run_downstream_bge as bge  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT = base.ROOT
REPORT_PATH = ROOT / "outputs/reports/downstream_bge_bootstrap_ci_260522.md"
N_BOOTSTRAP = 1000
CI_LOW, CI_HIGH = 2.5, 97.5
RANDOM_SEED = base.RANDOM_SEED
NON_FIELD_TARGETS = [t for t in base.ANCHOR_RECOVERY_TARGETS if t != "art_field_primary"]


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def percentile_ci(samples: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(samples, CI_LOW)), float(np.percentile(samples, CI_HIGH))


# --------------------------------------------------------------------------- #
# Per-instance prediction collection (model trained once, predicted once)
# --------------------------------------------------------------------------- #
def collect_anchor_recovery(
    embedder: bge.Embedder,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    masked_text: pd.Series,
) -> dict[str, dict[str, np.ndarray]]:
    train_uuids = train_df["pak_uuid"].astype(str).tolist()
    test_uuids = test_df["pak_uuid"].astype(str).tolist()
    x_train = embedder.encode([masked_text.loc[u] for u in train_uuids])
    x_test = embedder.encode([masked_text.loc[u] for u in test_uuids])

    out: dict[str, dict[str, np.ndarray]] = {}
    for target in NON_FIELD_TARGETS:
        y_train = train_df[target].astype(str)
        model = bge.logistic()
        model.fit(x_train, y_train)
        y_true = test_df[target].astype(str).to_numpy()
        y_pred = model.predict(x_test)
        out[target] = {
            "y_true": y_true,
            "y_pred": np.asarray(y_pred),
            "labels": np.unique(y_true),
        }
    return out


def collect_pair(
    embedder: bge.Embedder,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    masked_text: pd.Series,
) -> tuple[dict[str, np.ndarray], Any]:
    rng = np.random.default_rng(RANDOM_SEED)
    train_pairs = base.build_pair_dataset(train_df, masked_text, rng)
    test_pairs = base.build_pair_dataset(test_df, masked_text, rng)
    model = bge.fit_pair_model(embedder, train_pairs)

    a = embedder.encode(test_pairs["anchor_text"].tolist())
    n = embedder.encode(test_pairs["narrative_text"].tolist())
    pred = (model.predict_proba(bge.pair_features(a, n))[:, 1] >= 0.5).astype(int)
    y_true = test_pairs["label"].astype(int).to_numpy()
    return {"y_true": y_true, "y_pred": pred, "labels": np.unique(y_true)}, model


def collect_retrieval_ranks(
    embedder: bge.Embedder,
    test_df: pd.DataFrame,
    masked_text: pd.Series,
    pair_model: Any,
) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_SEED)
    same_field_age = base.build_group_map(test_df, ("art_field_primary", "age_band"))
    same_field = base.build_group_map(test_df, ("art_field_primary",))
    all_uuids = test_df["pak_uuid"].astype(str).tolist()

    anchor_texts: list[str] = []
    narrative_texts: list[str] = []
    meta: list[tuple[str, str]] = []
    for row in test_df.itertuples(index=False):
        row_s = pd.Series(row._asdict())
        query_uuid = str(row_s["pak_uuid"])
        query_anchor = base.anchor_block(row_s)
        candidates = base.choose_retrieval_candidates(
            row_s, test_df, same_field_age, same_field, all_uuids, rng
        )
        for candidate_uuid in candidates:
            anchor_texts.append(query_anchor)
            narrative_texts.append(masked_text.loc[candidate_uuid])
            meta.append((query_uuid, candidate_uuid))

    a = embedder.encode(anchor_texts)
    n = embedder.encode(narrative_texts)
    scores = pair_model.predict_proba(bge.pair_features(a, n))[:, 1]
    score_df = pd.DataFrame(meta, columns=["query_uuid", "candidate_uuid"])
    score_df["score"] = scores

    ranks: list[int] = []
    for _, group in score_df.groupby("query_uuid", sort=False):
        ordered = group.sort_values("score", ascending=False).reset_index(drop=True)
        rank = int(ordered.index[ordered["candidate_uuid"].eq(ordered["query_uuid"])][0]) + 1
        ranks.append(rank)
    return np.asarray(ranks)


def collect_counterfactual(
    embedder: bge.Embedder,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    masked_text: pd.Series,
) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
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
        y_test = test_pairs["label"].astype(int).to_numpy()

        a_tr = embedder.encode(train_pairs["anchor_text"].tolist())
        n_tr = embedder.encode(train_pairs["narrative_text"].tolist())
        a_te = embedder.encode(test_pairs["anchor_text"].tolist())
        n_te = embedder.encode(test_pairs["narrative_text"].tolist())

        full = bge.logistic()
        full.fit(bge.pair_features(a_tr, n_tr), y_train)
        full_pred = full.predict(bge.pair_features(a_te, n_te))

        anchor_only = bge.logistic()
        anchor_only.fit(a_tr, y_train)
        anchor_pred = anchor_only.predict(a_te)

        out[target] = {
            "y_true": y_test,
            "full_pred": np.asarray(full_pred),
            "anchor_pred": np.asarray(anchor_pred),
            "labels": np.unique(y_test),
        }
    return out


# --------------------------------------------------------------------------- #
# Bootstrap aggregation
# --------------------------------------------------------------------------- #
def bootstrap_anchor_mean(collected: dict[str, dict[str, np.ndarray]], rng: np.random.Generator) -> np.ndarray:
    n = len(next(iter(collected.values()))["y_true"])
    reps = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        per_target = [
            macro_f1(d["y_true"][idx], d["y_pred"][idx], d["labels"]) for d in collected.values()
        ]
        reps[b] = float(np.mean(per_target))
    return reps


def bootstrap_pair(collected: dict[str, np.ndarray], rng: np.random.Generator) -> np.ndarray:
    n = len(collected["y_true"])
    reps = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        reps[b] = macro_f1(collected["y_true"][idx], collected["y_pred"][idx], collected["labels"])
    return reps


def bootstrap_mrr(ranks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(ranks)
    reps = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        reps[b] = float(np.mean(1.0 / ranks[idx]))
    return reps


def bootstrap_counterfactual(
    collected: dict[str, dict[str, np.ndarray]], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    model_reps = np.empty(N_BOOTSTRAP)
    gain_reps = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        full_per_target = []
        gain_per_target = []
        for d in collected.values():
            n = len(d["y_true"])
            idx = rng.integers(0, n, n)
            full_f1 = macro_f1(d["y_true"][idx], d["full_pred"][idx], d["labels"])
            anchor_f1 = macro_f1(d["y_true"][idx], d["anchor_pred"][idx], d["labels"])
            full_per_target.append(full_f1)
            gain_per_target.append(full_f1 - anchor_f1)
        model_reps[b] = float(np.mean(full_per_target))
        gain_reps[b] = float(np.mean(gain_per_target))
    return model_reps, gain_reps


def main() -> None:
    embedder = bge.Embedder(bge.load_encoder())
    df = base.read_release()
    train_df, test_df = base.split_personas(df)
    all_df = pd.concat([train_df, test_df])

    rows: list[dict[str, Any]] = []
    for masking in base.MASKING_SETTINGS:
        log.info("=== masking=%s ===", masking)
        masked_text = base.make_masked_texts(all_df, masking)
        rng = np.random.default_rng(RANDOM_SEED)

        log.info("[%s] anchor recovery", masking)
        anchor = collect_anchor_recovery(embedder, train_df, test_df, masked_text)
        reps = bootstrap_anchor_mean(anchor, rng)
        lo, hi = percentile_ci(reps)
        rows.append({"masking": masking, "task": "non_field_anchor_recovery_mean",
                     "metric": "macro_f1", "point": float(np.mean([macro_f1(d["y_true"], d["y_pred"], d["labels"]) for d in anchor.values()])),
                     "ci_low": lo, "ci_high": hi})

        log.info("[%s] hard pair consistency", masking)
        pair, pair_model = collect_pair(embedder, train_df, test_df, masked_text)
        reps = bootstrap_pair(pair, rng)
        lo, hi = percentile_ci(reps)
        rows.append({"masking": masking, "task": "hard_pair_consistency",
                     "metric": "macro_f1", "point": macro_f1(pair["y_true"], pair["y_pred"], pair["labels"]),
                     "ci_low": lo, "ci_high": hi})

        log.info("[%s] 10-way retrieval", masking)
        ranks = collect_retrieval_ranks(embedder, test_df, masked_text, pair_model)
        reps = bootstrap_mrr(ranks, rng)
        lo, hi = percentile_ci(reps)
        rows.append({"masking": masking, "task": f"{base.CANDIDATE_K}_way_retrieval",
                     "metric": "mrr", "point": float(np.mean(1.0 / ranks)),
                     "ci_low": lo, "ci_high": hi})

        log.info("[%s] counterfactual consistency", masking)
        cf = collect_counterfactual(embedder, train_df, test_df, masked_text)
        model_reps, gain_reps = bootstrap_counterfactual(cf, rng)
        lo, hi = percentile_ci(model_reps)
        point_model = float(np.mean([macro_f1(d["y_true"], d["full_pred"], d["labels"]) for d in cf.values()]))
        rows.append({"masking": masking, "task": "counterfactual_mean_model",
                     "metric": "macro_f1", "point": point_model, "ci_low": lo, "ci_high": hi})
        lo, hi = percentile_ci(gain_reps)
        point_gain = float(np.mean([
            macro_f1(d["y_true"], d["full_pred"], d["labels"]) - macro_f1(d["y_true"], d["anchor_pred"], d["labels"])
            for d in cf.values()
        ]))
        rows.append({"masking": masking, "task": "counterfactual_mean_gain",
                     "metric": "macro_f1_gain", "point": point_gain, "ci_low": lo, "ci_high": hi})

    result = pd.DataFrame(rows)
    lines = [
        "# PAK Downstream Hard Tasks — BGE-m3-ko bootstrap 95% confidence intervals",
        "",
        f"- Encoder: `{bge.EMBED_MODEL}`, max_seq_length={bge.MAX_SEQ_LENGTH}, L2-normalised.",
        f"- Bootstrap: B={N_BOOTSTRAP} replicates, percentile {CI_LOW:g}--{CI_HIGH:g} interval, seed {RANDOM_SEED}.",
        "- Fixed model, resampled test set: models are trained once on the original training split; per-instance predictions are computed once on the original test split; only test instances are resampled with replacement.",
        "- Resampling unit: test personas (anchor recovery), test pairs (pair consistency, counterfactual), test queries (retrieval).",
        "- Same data pipeline, split, and masking as the point-estimate run; only uncertainty is added.",
        "",
        result.to_markdown(index=False, floatfmt=".3f"),
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote %s", REPORT_PATH.relative_to(ROOT))
    log.info("\n%s", result.to_string(index=False))


if __name__ == "__main__":
    main()
