# PAK-1K-eval Data Card

Generated: 2026-05-23T06:19:44

## Scope

PAK-1K-eval is a 1,000-persona scoring layer for the Persona Arts Korea
release. It contains one human reference annotation and four LLM judge
runs over four dimensions: groundedness, coherence, plausibility, and Korean
fluency. It also ships a separate generator-family self-preference probe
(`judges/qwen3/`) that is reported against the panel and is not part of the
panel `annotations.csv`.

## Self-preference probe

`judges/qwen3/` holds scores from qwen3:30b-a3b, the model that generated the
PAK narratives, run as a judge under the same per-dimension protocol as the
panel. Because it shares the generator family it measures self-preference
rather than serving as an independent judge, so it is kept out of
`annotations.csv`. Some dimension scores are null where the judge output
failed to parse. See `judges/qwen3/metadata.json` for per-dimension null counts
and `outputs/reports/qwen3_self_preference_260523.md` for the analysis.

## Human Reference

The public label `human_anonymous` denotes the single human reference used in
the paper's agreement analysis. It is anonymized only at the schema level and
does not denote an independently recruited external rater. The release applies `data/eval/human_scores_260520_amendments_260521.csv` as a
transparent amendment overlay (4 score corrections);
the original human CSV is retained outside this bundle for audit.

## Score Scale

Scores are integers from 1 to 5. The rubric anchors levels 1, 3, and 5 for
each dimension. Row-level flags are stored once using `dimension="_overall"`
instead of being duplicated across dimensions.

## Sample and Reproducibility

`sample_personas.csv` stores the 1,000 stratified personas with
`pak_uuid`, `order_index`, `batch_id`, `art_field_primary`, all quantitative
anchors, and narrative fields. `calibration_anchors.csv` stores the 10
rubric calibration personas. `preregistration.md` records the current
PAK-1K-eval plan, including the 2026-05-21 correction that no same-rater
retest is part of the canonical design.

## License

Released under CC-BY-4.0, matching the main PAK release.

## Limitations

This is a seed human-reference audit plus model-panel agreement layer. It is not
inter-annotator agreement, and it should not be treated as final ground truth.
Additional independent human annotations are invited against the same 1,000
personas.
