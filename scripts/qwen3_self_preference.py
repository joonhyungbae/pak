"""qwen3 self-preference analysis for PAK-1K-eval.

Compares the results of using the generator (qwen3:30b-a3b) as a judge against
the disjoint Claude-Gemini panel and the human reference. The self-preference
hypothesis of Panickssery et al. (2024): a judge from the generator family rates
its own outputs higher than disjoint judges do.

- self-preference statistic: qwen3 - panel (mean of Claude, Gemini), per-dimension
  paired Wilcoxon signed-rank.
- QWK(qwen3, human): dimensions with near-zero variance (coherence/fluency saturation)
  give denom=0 → nan → reported as "saturated".
- bias(qwen3 - human): leniency/strictness relative to the human anchor.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
QWEN3_PATH = ROOT / "outputs" / "pak_1k_eval_release" / "judges" / "qwen3" / "scores.json"
ANNOTATIONS_PATH = ROOT / "outputs" / "pak_1k_eval_release" / "annotations.csv"
REPORT_PATH = ROOT / "outputs" / "reports" / "qwen3_self_preference_260523.md"

DIMENSIONS = ("groundedness", "coherence", "plausibility", "fluency")
SCORE_MIN, SCORE_MAX = 1, 5


def quadratic_weighted_kappa(a: np.ndarray, b: np.ndarray) -> float:
    labels = list(range(SCORE_MIN, SCORE_MAX + 1))
    k = len(labels)
    idx = {v: i for i, v in enumerate(labels)}
    observed = np.zeros((k, k))
    for x, y in zip(a, b):
        observed[idx[int(round(x))], idx[int(round(y))]] += 1
    weights = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            weights[i, j] = ((i - j) ** 2) / ((k - 1) ** 2)
    hist_a = observed.sum(axis=1)
    hist_b = observed.sum(axis=0)
    expected = np.outer(hist_a, hist_b) / observed.sum()
    denom = (weights * expected).sum()
    if denom == 0:
        return float("nan")
    return float(1 - (weights * observed).sum() / denom)


def load_qwen3() -> pd.DataFrame:
    d = json.loads(QWEN3_PATH.read_text(encoding="utf-8"))
    rows = []
    for uuid, rec in d.items():
        for dim in DIMENSIONS:
            s = rec["scores"].get(dim)
            if s is not None:
                rows.append({"pak_uuid": str(uuid), "dimension": dim, "qwen3": float(s)})
    return pd.DataFrame(rows)


def load_panel() -> pd.DataFrame:
    df = pd.read_csv(ANNOTATIONS_PATH)
    df = df[df["dimension"].isin(DIMENSIONS)].copy()
    df["pak_uuid"] = df["pak_uuid"].astype(str)
    wide = df.pivot_table(
        index=["pak_uuid", "dimension"], columns="annotator", values="score"
    ).reset_index()
    return wide


def main() -> None:
    qwen3 = load_qwen3()
    panel = load_panel()
    merged = panel.merge(qwen3, on=["pak_uuid", "dimension"], how="inner")

    lines = [
        "# qwen3 self-preference analysis (PAK-1K-eval)",
        "",
        "- Generated: 2026-05-23",
        "- Judge model: qwen3:30b-a3b (same family as the generator = self-preference probe)",
        "- Disjoint panel: (Claude Opus 4.7 + Gemini 2.5 Pro) / 2",
        "- self-pref Δ = qwen3 mean − panel mean (positive = self-preference). p = paired Wilcoxon signed-rank.",
        "- bias = qwen3 mean − human mean. QWK(qwen3,human): near-zero-variance dimensions are saturated(nan).",
        "",
        "| dim | N | human | panel | qwen3 | Δ(qwen3−panel) | Wilcoxon p | bias(qwen3−human) | QWK(qwen3,human) | qwen3 max-share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    summary = {}
    for dim in DIMENSIONS:
        sub = merged[merged["dimension"].eq(dim)].copy()
        sub["panel"] = sub[["claude", "gemini"]].mean(axis=1)
        sub = sub.dropna(subset=["human_anonymous", "panel", "qwen3"])
        n = len(sub)
        human = sub["human_anonymous"].to_numpy(float)
        pnl = sub["panel"].to_numpy(float)
        q = sub["qwen3"].to_numpy(float)
        delta = q.mean() - pnl.mean()
        # Wilcoxon: qwen3 vs panel paired. zero_method handles ties.
        nonzero = np.abs(q - pnl) > 1e-9
        if nonzero.sum() > 0:
            try:
                _, p = wilcoxon(q[nonzero], pnl[nonzero])
            except ValueError:
                p = float("nan")
        else:
            p = float("nan")
        bias = q.mean() - human.mean()
        qwk = quadratic_weighted_kappa(q, human)
        max_share = max((q == k).mean() for k in range(1, 6))
        qwk_s = "saturated" if np.isnan(qwk) else f"{qwk:.3f}"
        p_s = "--" if np.isnan(p) else (f"{p:.2e}" if p < 1e-3 else f"{p:.3f}")
        lines.append(
            f"| {dim} | {n} | {human.mean():.3f} | {pnl.mean():.3f} | {q.mean():.3f} "
            f"| {delta:+.3f} | {p_s} | {bias:+.3f} | {qwk_s} | {max_share:.2f} |"
        )
        summary[dim] = dict(n=n, human=human.mean(), panel=pnl.mean(), qwen3=q.mean(),
                            delta=delta, p=p, bias=bias, qwk=qwk, max_share=max_share)

    lines += [
        "",
        "## Interpretation",
        "",
        "- **Style dimensions (coherence, fluency)**: qwen3 scores significantly higher than the "
        "panel and gives nearly all 5s (saturation) → strong self-preference. With variance≈0, QWK is undefined.",
        "- **Anchor-contrast dimensions (groundedness, plausibility)**: qwen3 is actually stricter than the "
        "panel and humans (negative Δ) → self-preference reverses in quantitative anchor contrast. QWK is computed normally.",
        "- Conclusion: self-preference manifests only in subjective quality judgments, and in factual "
        "grounding checks it is absent or reversed.",
    ]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
