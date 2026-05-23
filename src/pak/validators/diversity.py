"""Narrative diversity check.

The default is token-jaccard (no dependencies, fast). Optionally, a Korean
sentence-transformers embedding (`jhgan/ko-sroberta-multitask`) can be used
(recommended for analyzing 50k records).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


_TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text))


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    inter = ta & tb
    union = ta | tb
    return len(inter) / max(len(union), 1)


@dataclass
class DiversityReport:
    n: int
    pairs_compared: int
    mean_similarity: float
    max_similarity: float
    pct_above_threshold: float
    threshold: float


def pairwise_similarity_token(
    texts: list[str],
    *,
    sample_pairs: int | None = 1000,
    threshold: float = 0.85,
    seed: int = 0,
) -> DiversityReport:
    """Jaccard similarity statistics over sample_pairs random pairs from N texts."""
    n = len(texts)
    if n < 2:
        return DiversityReport(
            n=n,
            pairs_compared=0,
            mean_similarity=0.0,
            max_similarity=0.0,
            pct_above_threshold=0.0,
            threshold=threshold,
        )

    rng = np.random.default_rng(seed)
    total_pairs = n * (n - 1) // 2
    target = total_pairs if sample_pairs is None else min(sample_pairs, total_pairs)

    sims: list[float] = []
    seen: set[tuple[int, int]] = set()
    while len(sims) < target:
        i, j = rng.integers(0, n, size=2)
        if i == j or (int(i), int(j)) in seen or (int(j), int(i)) in seen:
            continue
        seen.add((int(i), int(j)))
        sims.append(jaccard(texts[int(i)], texts[int(j)]))

    arr = np.array(sims)
    return DiversityReport(
        n=n,
        pairs_compared=len(sims),
        mean_similarity=float(arr.mean()),
        max_similarity=float(arr.max()),
        pct_above_threshold=float((arr > threshold).mean()),
        threshold=threshold,
    )


def diversity_by_field(
    field_texts: dict[str, list[str]], *, threshold: float = 0.85
) -> dict[str, DiversityReport]:
    """Per-field narrative diversity statistics."""
    return {
        field: pairwise_similarity_token(texts, threshold=threshold)
        for field, texts in field_texts.items()
        if len(texts) >= 2
    }


# ----------------------------------------------------------------------------
# (optional) sentence-transformers embedding based — more accurate at large N
# ----------------------------------------------------------------------------


def pairwise_similarity_embedding(
    texts: list[str],
    *,
    model_name: str = "jhgan/ko-sroberta-multitask",
    sample_pairs: int = 1000,
    threshold: float = 0.85,
    seed: int = 0,
) -> DiversityReport:
    """Embedding cosine similarity via a sentence-transformers model. Raises ImportError if the package is missing."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers not installed. Only token-Jaccard is available."
        ) from exc

    model = SentenceTransformer(model_name)
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    n = len(texts)
    rng = np.random.default_rng(seed)
    target = min(sample_pairs, n * (n - 1) // 2)
    sims: list[float] = []
    while len(sims) < target:
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        sims.append(float(np.dot(embs[int(i)], embs[int(j)])))

    arr = np.array(sims)
    return DiversityReport(
        n=n,
        pairs_compared=len(sims),
        mean_similarity=float(arr.mean()),
        max_similarity=float(arr.max()),
        pct_above_threshold=float((arr > threshold).mean()),
        threshold=threshold,
    )


def all_unique(narratives: Iterable[dict[str, str]]) -> bool:
    """Whether there are no exactly identical narrative pairs (per field)."""
    seen: set[str] = set()
    for n in narratives:
        for v in n.values():
            if v in seen:
                return False
            seen.add(v)
    return True
