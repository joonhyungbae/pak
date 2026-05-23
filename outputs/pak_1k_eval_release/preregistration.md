# PAK-1K-eval — pre-registration (Amendment 4 correction, 2026-05-21)

**Locked**: before annotation begins. Any deviation from this plan
must be documented in a separate amendment file committed to the same
repository on or before the day of the deviation, and disclosed in
the paper's §5.5 narrative.

**Human reference**: single human-reference annotator within the author
team. External human annotations are outside the current release scope.

**Companion plan**: [experiment2_plan.md](experiment2_plan.md).

**Amendment history**:
- v0 (initial draft, committed 2026-05-15 morning, hash `b67f386`):
  original protocol with 1,000 Likert + cross-family LLM panel.
- v1 (afternoon, hash `151ce93`): added statistical hardening
  (Wilcoxon signed-rank, weighted κ, BH-FDR, stratified bootstrap,
  mixed-effects drift); operational protocol revisions (30-persona
  pilot, coarse per-batch dwell-time log). Also added a 200-persona ×
  3-perturbation adversarial battery as a validity layer.
- v2 (afternoon, hash `778b33e`): removes the adversarial battery.
  Returns to the Likert layer with all v1 statistical hardening intact.
- **v3 (this file, 2026-05-16)**: judge input convention.
  The two NPK-compatibility list fields
  `skills_and_expertise_list` and `hobbies_and_interests_list` are
  excluded from both the human rater's view (PersonaView) and the
  LLM judge input. Both fields ship in the dataset release for NPK
  compatibility but are not part of the narrative evaluated by
  PAK-1K-eval. Rationale: an audit on the full 30,000 release shows
  that 80.8 percent of skills rows and 58.0 percent of hobbies rows
  have at least one list item that is not surfaced in the paragraph
  (this is a deliberate dual-representation pattern, see
  src/pak/schema.py:282 "NPK-compatible list variant"). Including
  both fields caused early Codex judge runs to flag the missing
  items as coherence violations; the design treats list and
  paragraph as parallel representations of the same content, not as
  cross-validating statements. Excluding the list fields keeps the
  rater and the judge on the same input and removes a known
  false-positive source.
- **v4 correction (2026-05-21)**: clarifies that no same-rater
  retest is part of the canonical PAK-1K-eval plan. Temporary retest
  wording and scaffold files introduced during later review drafting
  are not part of the locked release design.

---

## 1. Research questions

- **RQ1 (agreement)**. Does the single human reference on PAK
  narratives agree with a cross-family LLM panel?
- **RQ2 (self-preference)**. Does the generator-family judge (qwen3)
  systematically score PAK narratives higher than the cross-family
  panel?
- **RQ3 (drift)**. Do human-reference scores drift across batches during
  a multi-batch annotation?

## 2. Sample

- N = 1,000 personas, stratified by `art_field_primary` (14 strata,
  proportional with minimum 10 per stratum). Drawn from
  `data/release/pak_v0_1/personas.parquet`.
- Random seed: 20260515.
- Calibration anchors N=10: drawn with seed 20260516, disjoint from
  the 1,000.

## 3. Measures

For each persona, four 1–5 Likert scores by the human reference and by
three LLM judges (Claude Opus 4.7, Gemini 2.5 Pro, qwen3:30b-a3b),
each judge × dimension repeated 3 times.

Dimensions: Groundedness, Internal Coherence, Plausibility, Korean
Fluency. Definitions are locked in
[web/src/rubric.ts](../../web/src/rubric.ts).

**Narrative fields evaluated**: 15 of the 17 narrative fields. The
two NPK-compatibility list fields (`skills_and_expertise_list`,
`hobbies_and_interests_list`) are excluded from both the human rater
view and the LLM judge input (see Amendment 3 in the amendment
history above). The 15 fields are enumerated in
[web/src/components/PersonaView.tsx](../../web/src/components/PersonaView.tsx)
and must match the field set passed to the LLM judge.

## 4. Hypotheses

| ID | Statement | Direction | Decision rule |
|---|---|---|---|
| H1 | Human-panel Pearson r ≥ 0.6 on Groundedness AND on Internal Coherence (reported per dimension, not as AND-conjunction) | one-sided positive per dimension | reject H₀ per dimension if 95 % stratified-bootstrap CI lower bound ≥ 0.6 |
| H2 | qwen3 mean score > cross-family mean per dimension (Wilcoxon signed-rank, paired) | two-sided per dimension | BH-FDR at q = 0.05 across 4 dimensions |
| H3 | Human-reference scores show no monotonic batch trend | null retained if no trend | mixed-effects model on persona × batch panel (batch as ordered fixed effect), Wald test α = 0.05 |

## 5. Analysis plan

### 5.1 Primary (RQ1)

- Pearson r and Spearman ρ between human-reference and panel mean per
  dimension, with **stratified 10,000-bootstrap** 95 % CI (resampling
  within `art_field_primary`).
- Continuous r reported with the Stureborg et al. (2024) reference
  band overlay so that downstream readers can locate the result
  against published comparators rather than against an absolute
  threshold.

### 5.2 Secondary (RQ1)

- **Weighted Cohen's κ (quadratic weights)** on the full 1–5 scale
  per dimension as the primary κ. Binary κ at the score-≥4 cut is
  reported as a robustness check only.
- Bland-Altman plot per dimension (mean vs difference).
- Mean absolute deviation per dimension.

### 5.3 Calibration drift (RQ3)

- Calibration drift: **mixed-effects model** on the 10-anchor scores
  with batch as an ordered fixed effect, per dimension.

### 5.4 Self-preference (RQ2)

- **Wilcoxon signed-rank** (paired, two-sided) between qwen3 score
  and cross-family panel mean per dimension. Pratt's zero-handling
  for ties.
- BH-FDR at q = 0.05 across the 4 dimensions.
- Cliff's δ per dimension as the effect-size companion.

### 5.5 Stratified (RQ1, descriptive)

- 14 fields × 4 dimensions = 56 cell means, with stratified
  10,000-bootstrap 95 % CI.
- **BH-FDR at q = 0.10** flag for cells whose CI excludes the overall
  dimension mean.

### 5.6 Confounds

- **Length confound**: Pearson r between concatenated narrative
  character count and (a) human-reference score, (b) panel score, per
  dimension.
- **Batch-order confound**: mixed-effects model on batch (see §5.3).
- **Panel-composition robustness**: drop one cross-family judge and
  recompute the §5.1 statistic. Robust if Pearson r changes by ≤ 0.05.

## 6. Exclusion criteria

- Persona whose `flags` field contains the literal token `SKIP` is
  excluded from primary and secondary analyses; counted and reported
  separately.
- Personas where LLM panel produced fewer than 3 successful repeats
  for ≥1 judge × dimension are excluded from §5.1–5.4; the affected
  count is reported.

## 7. Stopping rule

Annotation stops when 1,000 personas have a complete human-reference
judgement across all four dimensions. No early stopping based on
interim agreement statistics.

## 8. Operational requirements

- A **30-persona pilot** is run end-to-end before batch 1 begins,
  to measure realistic per-persona dwell time. The schedule (5
  batches × ≤13 hr) is locked only if the pilot median is ≤4
  minutes per persona; if pilot median is >4 min, the batch size is
  reduced to keep per-batch wall-clock ≤6 hr.
- The React review tool logs **per-batch median and IQR of dwell
  time**, not per-item timestamps.
- The React tool gates `run_judge_panel.py` execution to after CSV
  export of all 1,000 Likert items, so that the human pass remains
  blind to panel scores by construction.
- Anchor re-rating mode: previous anchor scores are hidden from the
  UI and anchor order is shuffled at each batch start.
- A 1,080-call smoke test of `run_judge_panel.py` is run (30
  personas × 4 dims × 3 judges × 3 repeats) before the full
  36,000-call panel run.

## 9. Pre-registration external timestamp

The pre-registration file is committed to the project git history
**and**, before annotation begins, deposited as a Zenodo record so
that the lock is anchored to a tamper-evident third-party DOI.
Subsequent analyses must reference the Zenodo DOI in addition to the
git commit hash.

## 10. Deviations from this plan

If a deviation occurs after annotation begins, the rubric file is
updated, the annotation is restarted from order_index 0 of the
affected dimension, the deviation is logged in this file as
the next amendment, and the paper's §5.5 narrative discloses
the deviation and its rationale.

## 11. Negative-result commitment

If any hypothesis (H1, H2, H3) yields the null direction, the
paper reports the result as-is with 95 % CI. We do not condition
publication of PAK-1K-eval on the hypotheses' direction. For
degenerate cases where a test is non-computable (panel sd zero on a
dimension, fewer than the minimum successful repeats), we report
descriptives and the reason; we do not silently drop the test.

## 12. Human-reference scope

The human reference is produced within the author team by a native
Korean speaker with domain familiarity with the Korean cultural-arts
sector through prior work on the 2024 Survey of Korean Artists. The
release label `human_anonymous` is a schema label for this single
human reference, not an independently recruited external rater. The
study does not claim inter-annotator-grade ground truth; the LLM panel
triangulation, calibration drift checks, and public reference checks
are the agreement and triangulation layers, and the design is
explicitly an *agreement* study, not a *validity* study.

## 13. Lock signature

This file is committed to the git history of the PAK repository
before annotation begins and also deposited as a Zenodo record (DOI
pending) for an external timestamp. Subsequent analyses must
reference both the git commit hash and the Zenodo DOI. Any post-hoc
analysis not listed here is reported in the paper as **exploratory**.
