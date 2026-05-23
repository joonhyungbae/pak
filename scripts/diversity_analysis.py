"""Quantify narrative diversity for the PAK 30,000-persona release.

Outputs:
    outputs/reports/diversity_analysis.md
    pak_paper/diversity_pairwise_cosine.pdf
"""

from __future__ import annotations

import html
import math
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, brevity_penalty, sentence_bleu
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "release" / "pak_v0_1" / "personas.parquet"
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "diversity_analysis.md"
FIGURE_PATH = PROJECT_ROOT / "pak_paper" / "diversity_pairwise_cosine.pdf"

SAMPLE_SIZE = 1_000
RANDOM_SEED = 20260512
SELF_BLEU_TIMEOUT_SECONDS = 30 * 60

NARRATIVE_FIELDS = [
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
]

PRIMARY_EMBEDDING_MODEL = "dragonkue/BGE-m3-ko"
FALLBACK_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@dataclass(frozen=True)
class TokenizerSpec:
    """Tokeniser metadata and callable."""

    name: str
    limitation_note: str
    tokenize: Callable[[str], list[str]]


@dataclass(frozen=True)
class LexicalRow:
    """Distinct-n metrics for one narrative field."""

    field: str
    tokens: int
    unique_unigrams: int
    distinct_1: float
    distinct_2: float
    distinct_3: float


@dataclass(frozen=True)
class PairRecord:
    """A high- or low-similarity pair for report inspection."""

    cosine: float
    uuid_a: str
    excerpt_a: str
    uuid_b: str
    excerpt_b: str


@dataclass(frozen=True)
class NgramReferenceStats:
    """Per-sentence n-gram counts and corpus max-count metadata."""

    counters: list[Counter[tuple[str, ...]]]
    top_counts: dict[tuple[str, ...], tuple[int, int, int]]


def package_version(package: str) -> str:
    """Return an installed package version when import metadata is available."""
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "unknown"


def log(message: str) -> None:
    """Print a progress line immediately."""
    print(message, flush=True)


def text_value(value: Any) -> str:
    """Convert a nullable field value to text without changing its content."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def build_tokenizer() -> TokenizerSpec:
    """Select the strongest available Korean tokeniser."""
    try:
        from kiwipiepy import Kiwi

        kiwi = Kiwi()

        def tokenize_kiwi(text: str) -> list[str]:
            return [token.form for token in kiwi.tokenize(text_value(text)) if token.form.strip()]

        return TokenizerSpec(
            name=f"kiwipiepy {package_version('kiwipiepy')}",
            limitation_note="None. Korean morpheme tokenisation used.",
            tokenize=tokenize_kiwi,
        )
    except ImportError:
        pass

    try:
        import MeCab  # type: ignore[import-not-found]

        tagger = MeCab.Tagger()

        def tokenize_mecab(text: str) -> list[str]:
            parsed = tagger.parse(text_value(text))
            tokens: list[str] = []
            for line in parsed.splitlines():
                if line == "EOS" or not line.strip():
                    continue
                surface, _, _features = line.partition("\t")
                if surface.strip():
                    tokens.append(surface)
            return tokens

        return TokenizerSpec(
            name=f"MeCab {package_version('mecab-python3')}",
            limitation_note="None. Korean MeCab tokenisation used.",
            tokenize=tokenize_mecab,
        )
    except ImportError:
        pass

    try:
        import mecab  # type: ignore[import-not-found]

        tagger = mecab.MeCab()

        def tokenize_python_mecab(text: str) -> list[str]:
            return [token for token in tagger.morphs(text_value(text)) if token.strip()]

        return TokenizerSpec(
            name=f"python-mecab-ko {package_version('python-mecab-ko')}",
            limitation_note="None. Korean MeCab tokenisation used.",
            tokenize=tokenize_python_mecab,
        )
    except ImportError:
        pass

    def tokenize_whitespace(text: str) -> list[str]:
        return text_value(text).split()

    return TokenizerSpec(
        name="whitespace fallback",
        limitation_note=(
            "Tokenisation limitation: no Korean morphological tokeniser was available. "
            "Whitespace tokenisation can overstate Korean lexical diversity because "
            "particles and endings remain attached to lexical stems."
        ),
        tokenize=tokenize_whitespace,
    )


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Return contiguous token n-grams for one token sequence."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def compute_lexical_diversity(df: pd.DataFrame, tokenizer: TokenizerSpec) -> list[LexicalRow]:
    """Compute corpus-level distinct-1, distinct-2, and distinct-3 per field."""
    rows: list[LexicalRow] = []
    for field in NARRATIVE_FIELDS:
        total_ngrams = {1: 0, 2: 0, 3: 0}
        unique_ngrams: dict[int, set[tuple[str, ...]]] = {1: set(), 2: set(), 3: set()}

        for value in df[field].tolist():
            tokens = tokenizer.tokenize(text_value(value))
            for n in (1, 2, 3):
                field_ngrams = ngrams(tokens, n)
                total_ngrams[n] += len(field_ngrams)
                unique_ngrams[n].update(field_ngrams)

        row = LexicalRow(
            field=field,
            tokens=total_ngrams[1],
            unique_unigrams=len(unique_ngrams[1]),
            distinct_1=safe_ratio(len(unique_ngrams[1]), total_ngrams[1]),
            distinct_2=safe_ratio(len(unique_ngrams[2]), total_ngrams[2]),
            distinct_3=safe_ratio(len(unique_ngrams[3]), total_ngrams[3]),
        )
        rows.append(row)
        log(
            f"3.1 lexical field={field} tokens={row.tokens} "
            f"distinct-1={row.distinct_1:.4f} distinct-3={row.distinct_3:.4f}"
        )
    return rows


def safe_ratio(numerator: int, denominator: int) -> float:
    """Divide two counts and return 0.0 when the denominator is zero."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_self_bleu(sample: pd.DataFrame, tokenizer: TokenizerSpec) -> dict[str, float]:
    """Compute mean smoothed self-BLEU-4 per field on the fixed sample."""
    started = time.monotonic()
    scores: dict[str, float] = {}
    validated = False

    for field in NARRATIVE_FIELDS:
        token_lists = [tokenizer.tokenize(text_value(value)) for value in sample[field].tolist()]
        stats_by_n = {n: build_ngram_reference_stats(token_lists, n) for n in (1, 2, 3, 4)}
        lengths = [len(tokens) for tokens in token_lists]
        if not validated:
            validate_fast_bleu(token_lists[:20])
            validated = True

        field_scores = [
            fast_self_bleu_score(idx, token_lists[idx], stats_by_n, lengths)
            for idx in range(len(token_lists))
        ]
        elapsed = time.monotonic() - started
        if elapsed > SELF_BLEU_TIMEOUT_SECONDS:
            raise TimeoutError(
                "STOP: self-BLEU exceeded 30 minutes. "
                "Sample-size adjustment requires user confirmation."
            )
        scores[field] = float(np.mean(field_scores))
        log(f"3.2 self-BLEU field={field} mean={scores[field]:.4f}")
    return scores


def build_ngram_reference_stats(token_lists: list[list[str]], n: int) -> NgramReferenceStats:
    """Precompute n-gram counts needed for self-BLEU references."""
    counters: list[Counter[tuple[str, ...]]] = []
    top_counts: dict[tuple[str, ...], tuple[int, int, int]] = {}
    for idx, tokens in enumerate(token_lists):
        counter = Counter(ngrams(tokens, n))
        counters.append(counter)
        for gram, count in counter.items():
            top_count, top_idx, second_count = top_counts.get(gram, (0, -1, 0))
            if count > top_count:
                top_counts[gram] = (count, idx, top_count)
            elif count > second_count:
                top_counts[gram] = (top_count, top_idx, count)
    return NgramReferenceStats(counters=counters, top_counts=top_counts)


def fast_self_bleu_score(
    idx: int,
    candidate: list[str],
    stats_by_n: dict[int, NgramReferenceStats],
    lengths: list[int],
) -> float:
    """Compute the NLTK sentence_bleu method1 score using precomputed references."""
    hyp_len = len(candidate)
    if hyp_len == 0:
        return 0.0

    precisions: list[float] = []
    unigram_matches = 0
    epsilon = SmoothingFunction().epsilon
    for n in (1, 2, 3, 4):
        stats = stats_by_n[n]
        candidate_counts = stats.counters[idx]
        denominator = max(1, sum(candidate_counts.values()))
        numerator = 0
        for gram, count in candidate_counts.items():
            top_count, top_idx, second_count = stats.top_counts[gram]
            reference_max = second_count if top_idx == idx else top_count
            numerator += min(count, reference_max)
        if n == 1:
            unigram_matches = numerator
        precisions.append(epsilon / denominator if numerator == 0 else numerator / denominator)

    if unigram_matches == 0:
        return 0.0

    closest_len = closest_reference_length_excluding(lengths, idx, hyp_len)
    bp = brevity_penalty(closest_len, hyp_len)
    weighted_log_precision = math.fsum(0.25 * math.log(precision) for precision in precisions)
    return float(bp * math.exp(weighted_log_precision))


def closest_reference_length_excluding(lengths: list[int], idx: int, hyp_len: int) -> int:
    """Match NLTK's closest_ref_length while excluding the candidate row."""
    return min(
        (length for ref_idx, length in enumerate(lengths) if ref_idx != idx),
        key=lambda ref_len: (abs(ref_len - hyp_len), ref_len),
    )


def validate_fast_bleu(token_lists: list[list[str]]) -> None:
    """Check the fast path against NLTK sentence_bleu on a small deterministic slice."""
    if len(token_lists) < 3:
        return
    stats_by_n = {n: build_ngram_reference_stats(token_lists, n) for n in (1, 2, 3, 4)}
    lengths = [len(tokens) for tokens in token_lists]
    smoothing = SmoothingFunction().method1
    weights = (0.25, 0.25, 0.25, 0.25)
    for idx, candidate in enumerate(token_lists[:5]):
        references = token_lists[:idx] + token_lists[idx + 1 :]
        slow = float(
            sentence_bleu(
                references,
                candidate,
                weights=weights,
                smoothing_function=smoothing,
            )
        )
        fast = fast_self_bleu_score(idx, candidate, stats_by_n, lengths)
        if not np.isclose(slow, fast, atol=1e-12):
            raise AssertionError(
                "Fast self-BLEU validation failed: "
                f"idx={idx}, nltk={slow:.12f}, fast={fast:.12f}"
            )


def select_embedding_model() -> tuple[str, Any]:
    """Load the primary embedding model from cache, then fall back if needed."""
    from sentence_transformers import SentenceTransformer

    try:
        model = SentenceTransformer(PRIMARY_EMBEDDING_MODEL, local_files_only=True)
        return PRIMARY_EMBEDDING_MODEL, model
    except Exception as primary_error:
        log(f"Embedding primary model unavailable: {primary_error}")

    try:
        model = SentenceTransformer(FALLBACK_EMBEDDING_MODEL)
        return FALLBACK_EMBEDDING_MODEL, model
    except Exception as fallback_error:
        raise RuntimeError(
            "STOP: embedding model download or loading failed. "
            "Authentication or network access may be required."
        ) from fallback_error


def combined_narratives(df: pd.DataFrame) -> list[str]:
    """Join the 15 paragraph fields in fixed order using a single space."""
    texts: list[str] = []
    for _, row in df.iterrows():
        parts = [text_value(row[field]).strip() for field in NARRATIVE_FIELDS]
        texts.append(" ".join(part for part in parts if part))
    return texts


def compute_semantic_metrics(sample: pd.DataFrame) -> tuple[str, np.ndarray, np.ndarray, float, float]:
    """Embed combined narratives and compute pairwise cosine values."""
    model_name, model = select_embedding_model()
    texts = combined_narratives(sample)
    embeddings = model.encode(
        texts,
        batch_size=16,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    similarities = cosine_similarity(embeddings)
    upper_i, upper_j = np.triu_indices(len(sample), k=1)
    pairwise = similarities[upper_i, upper_j]
    mean_cosine = float(np.mean(pairwise))
    std_cosine = float(np.std(pairwise))
    write_histogram(pairwise)
    return model_name, embeddings, pairwise, mean_cosine, std_cosine


def write_histogram(pairwise: np.ndarray) -> None:
    """Write a single-image histogram of pairwise cosine similarities."""
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.hist(pairwise, bins=50, range=(0, 1), color="#3b6ea8", edgecolor="white", linewidth=0.35)
    ax.set_xlabel("pairwise cosine similarity")
    ax.set_ylabel("frequency")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=150)
    plt.close(fig)


def compute_silhouette(embeddings: np.ndarray, labels: pd.Series) -> float:
    """Compute cosine-distance silhouette for art-field labels."""
    return float(silhouette_score(embeddings, labels.to_numpy(), metric="cosine"))


def top_pair_records(sample: pd.DataFrame, pairwise: np.ndarray) -> tuple[list[PairRecord], list[PairRecord]]:
    """Return top-10 highest- and lowest-cosine persona pairs."""
    n_rows = len(sample)
    upper_i, upper_j = np.triu_indices(n_rows, k=1)
    narratives = combined_narratives(sample)
    uuids = sample["pak_uuid"].astype(str).tolist()

    high_order = np.argsort(pairwise)[-10:][::-1]
    low_order = np.argsort(pairwise)[:10]

    high = [
        make_pair_record(pairwise[pos], upper_i[pos], upper_j[pos], uuids, narratives)
        for pos in high_order
    ]
    low = [
        make_pair_record(pairwise[pos], upper_i[pos], upper_j[pos], uuids, narratives)
        for pos in low_order
    ]
    return high, low


def make_pair_record(
    cosine: float,
    idx_a: int,
    idx_b: int,
    uuids: list[str],
    narratives: list[str],
) -> PairRecord:
    """Build a report record for a pair of sample rows."""
    return PairRecord(
        cosine=float(cosine),
        uuid_a=uuids[idx_a],
        excerpt_a=excerpt(narratives[idx_a]),
        uuid_b=uuids[idx_b],
        excerpt_b=excerpt(narratives[idx_b]),
    )


def excerpt(text: str, limit: int = 100) -> str:
    """Return a one-line excerpt with at most limit characters."""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit]


def md_escape(value: str) -> str:
    """Escape Markdown table control characters."""
    return html.escape(value).replace("|", "\\|")


def format_float(value: float, digits: int = 4) -> str:
    """Format a float with fixed decimals."""
    return f"{value:.{digits}f}"


def write_report(
    lexical_rows: list[LexicalRow],
    self_bleu: dict[str, float],
    embedding_model: str,
    mean_cosine: float,
    std_cosine: float,
    silhouette: float,
    high_pairs: list[PairRecord],
    low_pairs: list[PairRecord],
    tokenizer: TokenizerSpec,
) -> None:
    """Write the Markdown analysis report."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    distinct_1_values = [row.distinct_1 for row in lexical_rows]
    distinct_2_values = [row.distinct_2 for row in lexical_rows]
    distinct_3_values = [row.distinct_3 for row in lexical_rows]
    self_bleu_values = list(self_bleu.values())

    lines = [
        "# Phase 06s \u2014 Narrative diversity (PAK 30,000 personas)",
        "",
        "## Summary",
        (
            "Across the 15 paragraph narrative fields, distinct-1 ranges from "
            f"{min(distinct_1_values):.4f} to {max(distinct_1_values):.4f}, "
            f"distinct-2 from {min(distinct_2_values):.4f} to {max(distinct_2_values):.4f}, "
            f"and distinct-3 from {min(distinct_3_values):.4f} to {max(distinct_3_values):.4f}. "
            f"Mean self-BLEU ranges from {min(self_bleu_values):.4f} to "
            f"{max(self_bleu_values):.4f} on the fixed 1,000-person sample. "
            f"The combined-narrative embedding space has mean pairwise cosine "
            f"{mean_cosine:.4f} with standard deviation {std_cosine:.4f}, "
            f"and the cosine-distance silhouette score over 14 art-field labels is "
            f"{silhouette:.4f}."
        ),
        "",
        "## Lexical diversity (distinct-n) per narrative field",
        "",
        "| field | tokens | unique 1-gram | distinct-1 | distinct-2 | distinct-3 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in lexical_rows:
        lines.append(
            f"| `{row.field}` | {row.tokens:,} | {row.unique_unigrams:,} | "
            f"{row.distinct_1:.4f} | {row.distinct_2:.4f} | {row.distinct_3:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Self-BLEU per narrative field (n=1,000 sample, seed 20260512)",
            "",
            (
                "Definition: an exact precomputed implementation of NLTK "
                "`sentence_bleu` with uniform 1- to 4-gram weights and "
                "Chen-Cherry method1 smoothing. The fast path is validated against "
                "NLTK at runtime. The field value is the mean over 1,000 candidates, "
                "each compared with the other 999 sampled paragraphs from the same field."
            ),
            "",
            "| field | self-BLEU |",
            "|---|---:|",
        ]
    )
    for field in NARRATIVE_FIELDS:
        lines.append(f"| `{field}` | {self_bleu[field]:.4f} |")

    lines.extend(
        [
            "",
            "## Semantic diversity (n=1,000 sample)",
            "",
            f"- Embedding model: {embedding_model}",
            f"- Mean pairwise cosine: {mean_cosine:.4f}",
            f"- Std: {std_cosine:.4f}",
            "- Figure: pak_paper/diversity_pairwise_cosine.pdf",
            "",
            (
                "Definition: for each sampled persona, the 15 paragraph narrative fields "
                "are joined in the fixed release-schema order with a single space before "
                "sentence embedding."
            ),
            "",
            "## Field separation",
            "",
            f"- Silhouette score (cosine, 14 art fields, n=1,000): {silhouette:.4f}",
            "",
            "## Sanity dump",
            "",
            "### Top-10 highest-cosine pairs",
            "",
        ]
    )
    lines.extend(pair_table(high_pairs))
    lines.extend(
        [
            "",
            "### Top-10 lowest-cosine pairs",
            "",
        ]
    )
    lines.extend(pair_table(low_pairs))
    lines.extend(
        [
            "",
            "## Tokenisation note",
            "",
            f"- Tokeniser actually used: {tokenizer.name}",
            f"- {tokenizer.limitation_note}",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def pair_table(records: list[PairRecord]) -> list[str]:
    """Format pair records as a Markdown table."""
    lines = [
        "| rank | cosine | pak_uuid A | excerpt A | pak_uuid B | excerpt B |",
        "|---:|---:|---|---|---|---|",
    ]
    for rank, record in enumerate(records, start=1):
        lines.append(
            f"| {rank} | {record.cosine:.4f} | `{record.uuid_a}` | "
            f"{md_escape(record.excerpt_a)} | `{record.uuid_b}` | "
            f"{md_escape(record.excerpt_b)} |"
        )
    return lines


def assert_required_columns(df: pd.DataFrame) -> None:
    """Fail fast when a required release column is missing."""
    required = set(NARRATIVE_FIELDS) | {"pak_uuid", "art_field_primary"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def main() -> None:
    """Run the analysis from the release Parquet file."""
    log(f"Loading data: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    assert_required_columns(df)
    sample = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)
    tokenizer = build_tokenizer()
    log(f"Tokeniser: {tokenizer.name}")

    lexical_rows = compute_lexical_diversity(df, tokenizer)
    log(
        "3.1 lexical: "
        f"distinct-1 range={min(row.distinct_1 for row in lexical_rows):.4f}-"
        f"{max(row.distinct_1 for row in lexical_rows):.4f}, "
        f"distinct-3 range={min(row.distinct_3 for row in lexical_rows):.4f}-"
        f"{max(row.distinct_3 for row in lexical_rows):.4f}"
    )
    if tokenizer.name == "whitespace fallback" and max(row.distinct_1 for row in lexical_rows) >= 0.9:
        raise RuntimeError(
            "STOP: whitespace tokenisation produced distinct-1 >= 0.9. "
            "Korean tokenisation distortion requires user confirmation."
        )

    self_bleu = compute_self_bleu(sample, tokenizer)
    log(
        "3.2 self-BLEU: "
        f"range={min(self_bleu.values()):.4f}-{max(self_bleu.values()):.4f}"
    )

    embedding_model, embeddings, pairwise, mean_cosine, std_cosine = compute_semantic_metrics(sample)
    log(
        "3.3 semantic: "
        f"model={embedding_model}, mean={mean_cosine:.4f}, std={std_cosine:.4f}, "
        f"figure={FIGURE_PATH}"
    )

    silhouette = compute_silhouette(embeddings, sample["art_field_primary"])
    log(f"3.4 field separation: silhouette={silhouette:.4f}")

    high_pairs, low_pairs = top_pair_records(sample, pairwise)
    write_report(
        lexical_rows=lexical_rows,
        self_bleu=self_bleu,
        embedding_model=embedding_model,
        mean_cosine=mean_cosine,
        std_cosine=std_cosine,
        silhouette=silhouette,
        high_pairs=high_pairs,
        low_pairs=low_pairs,
        tokenizer=tokenizer,
    )
    log(
        "3.5 sanity dump: "
        f"highest={high_pairs[0].cosine:.4f}, lowest={low_pairs[0].cosine:.4f}, "
        f"report={REPORT_PATH}"
    )


if __name__ == "__main__":
    main()
