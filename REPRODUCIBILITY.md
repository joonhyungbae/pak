# PAK Reproducibility Guide

This guide records the release-facing commands for rebuilding and validating
Persona Arts Korea artifacts from this repository.

## Environment

Use Python 3.11+.

```bash
uv sync --frozen --extra dev
```

The locked dependency graph is `uv.lock`. GPU/model-serving dependencies are
included because the repository also contains generation code; loading and
validating the released Parquet files only requires the data-analysis subset of
the dependencies.

Generated reports under `outputs/reports/` are not committed. They are written
locally when the reproduction commands below are run; the paper carries the
canonical numbers.

## Artifact Map

| Artifact | Command | Main inputs | Main outputs |
|---|---|---|---|
| Main 30k release | `python scripts/package_release.py` | `data/synthetic/*/personas.parquet`, `data/grounding/` | `data/release/pak_v0_1/`, `data/release/persona-arts-korea.zip` |
| PAK-1K-eval bundle | `python scripts/prepare_eval_judge_release.py` | `data/eval/human_scores_260520.csv`, `outputs/pak_1k_eval_release/judges/*/scores.json` | `outputs/pak_1k_eval_release/annotations.csv`, metadata, manifest |
| Tidy annotation table | `python scripts/consolidate_annotations.py` | human CSV, canonical judge outputs | `data/eval/pak_1k_eval_annotations.csv` |
| Agreement report | `python scripts/judge_human_agreement.py` | consolidated annotations and human reference | `outputs/reports/judge_human_agreement_260520.md` |
| Paper tables | `python scripts/compute_pak1k_paper_tables.py` | PAK-1K-eval bundle | `outputs/reports/pak_1k_eval_paper_tables_260521.md` |
| Downstream diagnostics | `python scripts/run_downstream_persona_tasks.py` | `data/release/pak_v0_1/personas.parquet` | `outputs/reports/downstream_persona_tasks_260521.md`, CSV diagnostics |
| Release verification | `python scripts/verify_release.py` | release directories and manifests | log-only verification; non-zero exit on failure |

## One-Pass Verification

After cloning the repository, run:

```bash
python scripts/verify_release.py
pytest -q
```

On a fresh clone the main 30k-release checks are skipped with a warning, because
the dataset is hosted on Hugging Face and Zenodo rather than committed here; the
PAK-1K-eval checks run against the in-repo bundle. Download the dataset into
`data/release/pak_v0_1/` to verify the full release.

`verify_release.py` checks:

- `data/release/pak_v0_1/MANIFEST.sha256`
- `outputs/pak_1k_eval_release/MANIFEST.sha256`
- 30,000 main-release personas with no duplicate `pak_uuid`
- 1,000 PAK-1K-eval sample personas
- 10 calibration anchors
- complete annotator-by-dimension coverage for `human_anonymous`, `claude`,
  `gemini`, `clova`, and `codex`
- all scoring-dimension values in the 1-5 range

## Rebuilding The Main Release

```bash
python scripts/package_release.py
sha256sum -c data/release/pak_v0_1/MANIFEST.sha256
```

The command expects the two local synthetic generation runs listed in
`scripts/package_release.py`. These large intermediate runs are intentionally
ignored by git and are not part of the lightweight code repository.

The package script writes `code_snapshot/` into the release directory. The
snapshot includes:

- `src/pak/`
- `scripts/`
- `tests/`
- `README.md`
- `REPRODUCIBILITY.md`
- `pyproject.toml`
- `uv.lock`

## Rebuilding PAK-1K-eval

```bash
python scripts/prepare_eval_judge_release.py
sha256sum -c outputs/pak_1k_eval_release/MANIFEST.sha256
```

The script reads the canonical per-judge `scores.json` files under
`outputs/pak_1k_eval_release/judges/` together with the release bundle's
`sample_personas.csv` and `calibration_anchors.csv`.

## Known Non-Release Local State

The following paths are local development state and are intentionally not part
of the release contract:

- `.venv/`, `.pytest_cache/`, `.ruff_cache/`, `__pycache__/`
- `data/synthetic/` generation intermediates
- `data/release/` (the 30k dataset and its archive are hosted on Hugging Face and Zenodo)

Do not treat these paths as required inputs unless a specific script says so.

## Grounding Provenance

The grounding tables ship pre-built as `data/grounding/T1.parquet` through
`T15.parquet`, each paired with a `T*_provenance.json` that records the source
report table id, page number, and the SHA-256 of the source PDF. The source
report (2024 Survey of Korean Artists) is available from the Korea Culture and
Tourism Institute under the Korea Open Government Licence Type 1; the grounding
tables are derived from it and are released here so the sampling pipeline can be
reproduced without redistributing the source document.
