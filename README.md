# PAK: Persona Arts Korea

PAK is a reproducible release package for **Persona Arts Korea**, a
30,000-row synthetic Korean cultural-arts persona dataset grounded in public
statistics from the 2024 Survey of Korean Artists.

The repository contains the sampler, narrative-generation utilities,
validation suite, downstream diagnostic tasks, and release-packaging scripts
used for the dataset and the companion PAK-1K-eval annotation layer.

## Repository Layout

| Path | Purpose |
|---|---|
| `src/pak/` | Core package: schema, samplers, prompts, validators, grounding helpers |
| `scripts/` | Reproducible command-line workflows for packaging, evaluation, reports, and release checks |
| `tests/` | Unit tests for schema, sampling, validators, prompts, and post-generation checks |
| `data/release/pak_v0_1/` | Local build output of `package_release.py` (not committed); the dataset itself is hosted on Hugging Face and Zenodo |
| `outputs/pak_1k_eval_release/` | Release-facing PAK-1K-eval annotation bundle |
| `outputs/reports/` | Local output directory written by the reproduction scripts (not committed; the paper carries the canonical numbers) |

## Environment

Python 3.11+ is required. The lockfile is `uv.lock`.

```bash
uv sync --frozen --extra dev
```

If `uv` is not available, install the package into an equivalent Python 3.11
environment from `pyproject.toml`.

## Verify The Current Release

```bash
python scripts/verify_release.py
```

This verifies SHA-256 manifests, row counts, duplicate IDs, PAK-1K-eval
coverage, and score ranges for the release-facing artifacts.

## Rebuild Release Artifacts

The main dataset package is rebuilt from the local synthetic generation runs
and grounding tables:

```bash
python scripts/package_release.py
```

Required local inputs:

- `data/synthetic/pak_10k_30b_concurrent_20260510/personas.parquet`
- `data/synthetic/pak_extra_20k_30b_concurrent_20260520/personas.parquet`
- `data/grounding/`

The PAK-1K-eval annotation bundle is rebuilt from canonical judge JSON files
and the human-reference CSV:

```bash
python scripts/prepare_eval_judge_release.py
```

The downstream diagnostic suite is rebuilt with:

```bash
python scripts/run_downstream_persona_tasks.py
```

For a fuller command map, see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Release Artifacts

The main dataset release is assembled under:

```text
data/release/pak_v0_1/
data/release/persona-arts-korea.zip
```

The PAK-1K-eval release-facing bundle is assembled under:

```text
outputs/pak_1k_eval_release/
```

Both release directories include `MANIFEST.sha256` files. The release package
also contains a `code_snapshot/` directory with `src/pak`, `scripts`, `tests`,
`pyproject.toml`, `uv.lock`, this README, and the reproducibility guide.

## Tests

```bash
pytest -q
```

Some end-to-end generation paths require local model/runtime assets and are not
needed to load the released Parquet artifacts.

## License

This repository, including the code (`src/`, `scripts/`, `tests/`), the grounding tables, and the PAK-1K-eval annotation layer, is released under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**; see [LICENSE](LICENSE). This matches the PAK dataset release on the Hugging Face Hub. The upstream source statistics from the 2024 Survey of Korean Artists are available under the Korea Open Government Licence Type 1.

## Links and Citation

- Dataset: https://huggingface.co/datasets/joonhyungbae/Persona-Arts-Korea
- Code: https://github.com/joonhyungbae/pak
- Zenodo DOI: assigned on publication (added here once minted)

Please cite:

```bibtex
@misc{bae2026pak,
  author = "Bae, Joonhyung",
  title  = "PAK: A Survey-Grounded Synthetic Persona Dataset of the Korean Cultural Arts Workforce",
  year   = "2026"
}
```

Please also acknowledge the upstream source, the 2024 Survey of Korean Artists (Korea Culture and Tourism Institute, Ministry of Culture, Sports and Tourism), released under the Korea Open Government Licence Type 1. See [CITATION.cff](CITATION.cff).
