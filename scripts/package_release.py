"""Assemble the PAK v0.1 release directory and ZIP it for Zenodo upload.

Outputs:
    data/release/pak_v0_1/personas.parquet         (30,000 rows)
    data/release/pak_v0_1/personas_sample_100.jsonl (curated sample)
    data/release/pak_v0_1/grounding/T*.parquet     (T1~T15)
    data/release/pak_v0_1/grounding/T*_provenance.json
    data/release/pak_v0_1/grounding/sampler_specs.json
    data/release/pak_v0_1/grounding/joint_distributions.parquet
    data/release/pak_v0_1/code_snapshot/                 (src/pak + scripts + tests + docs)
    data/release/pak_v0_1/MANIFEST.sha256          (release file hashes)
    data/release/persona-arts-korea.zip            (full archive)

LICENSE and README.md are assumed pre-written in the same directory.
"""

from __future__ import annotations

import json
import hashlib
import logging
import shutil
import zipfile
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_RUNS: tuple[Path, ...] = (
    PROJECT_ROOT / "data" / "synthetic" / "pak_10k_30b_concurrent_20260510",
    PROJECT_ROOT / "data" / "synthetic" / "pak_extra_20k_30b_concurrent_20260520",
)
EXPECTED_ROWS = 30_000
RELEASE_DIR = PROJECT_ROOT / "data" / "release" / "pak_v0_1"
GROUNDING_SRC = PROJECT_ROOT / "data" / "grounding"
SRC_DIR = PROJECT_ROOT / "src" / "pak"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"
MANIFEST_NAME = "MANIFEST.sha256"
CODE_SNAPSHOT_DOCS: tuple[str, ...] = (
    "README.md",
    "REPRODUCIBILITY.md",
    "pyproject.toml",
    "uv.lock",
)

RELEASE_COLUMNS: tuple[str, ...] = (
    "persona",
    "professional_persona",
    "sports_persona",
    "arts_persona",
    "travel_persona",
    "culinary_persona",
    "family_persona",
    "cultural_background",
    "skills_and_expertise",
    "skills_and_expertise_list",
    "hobbies_and_interests",
    "hobbies_and_interests_list",
    "career_goals_and_ambitions",
    "creative_world_persona",
    "network_persona",
    "living_persona",
    "support_persona",
    "pak_uuid",
    "sex",
    "age",
    "province",
    "country",
    "education_level",
    "occupation",
    "age_band",
    "education_level_pak",
    "art_field_primary",
    "career_years",
    "career_band",
    "employment_type",
    "is_freelance",
    "has_secondary_job",
    "individual_art_income_bracket",
    "household_income_bracket",
    "has_contract_experience",
    "uses_standard_contract",
    "has_copyright",
    "had_career_break",
    "has_overseas_experience",
    "debut_age",
)


def load_release_personas() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_dir in SOURCE_RUNS:
        src = run_dir / "personas.parquet"
        if not src.exists():
            raise FileNotFoundError(f"Missing source run: {src.relative_to(PROJECT_ROOT)}")
        frame = pd.read_parquet(src)
        if "debut_age" not in frame.columns:
            frame["debut_age"] = frame["age"].astype(int) - frame["career_years"].astype(int)
        missing = sorted(set(RELEASE_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"{src.relative_to(PROJECT_ROOT)} missing columns: {missing}")
        frames.append(frame.loc[:, list(RELEASE_COLUMNS)].copy())

    df = pd.concat(frames, ignore_index=True)
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} personas, got {len(df):,}")
    duplicate_uuids = int(df["pak_uuid"].duplicated().sum())
    if duplicate_uuids:
        raise ValueError(f"Release would contain {duplicate_uuids} duplicate pak_uuid values")
    return df


def write_personas() -> None:
    df = load_release_personas()
    dst = RELEASE_DIR / "personas.parquet"
    df.to_parquet(dst, index=False)
    log.info("personas.parquet: %s rows, %s columns", f"{len(df):,}", df.shape[1])


def write_sample_jsonl(n: int = 100, seed: int = 20260512) -> None:
    df = pd.read_parquet(RELEASE_DIR / "personas.parquet")
    sample = df.sample(n=n, random_state=seed).sort_index()
    out = RELEASE_DIR / f"personas_sample_{n}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for _, row in sample.iterrows():
            obj = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in row.items()}
            f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
    log.info("sample %s: %s", n, out.name)


def copy_grounding() -> None:
    target = RELEASE_DIR / "grounding"
    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted(GROUNDING_SRC.glob("T*.parquet")):
        shutil.copy2(path, target / path.name)
        copied += 1
    for path in sorted(GROUNDING_SRC.glob("T*_provenance.json")):
        shutil.copy2(path, target / path.name)
        copied += 1
    for path in ("sampler_specs.json", "joint_distributions.parquet"):
        src = GROUNDING_SRC / path
        if src.exists():
            shutil.copy2(src, target / path)
            copied += 1
    log.info("grounding: %d files", copied)


def copy_code_snapshot() -> None:
    target = RELEASE_DIR / "code_snapshot"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(SRC_DIR, target / "pak", ignore=shutil.ignore_patterns("__pycache__"))
    if TESTS_DIR.exists():
        shutil.copytree(TESTS_DIR, target / "tests", ignore=shutil.ignore_patterns("__pycache__"))
    scripts_dst = target / "scripts"
    scripts_dst.mkdir(parents=True, exist_ok=True)
    for path in SCRIPTS_DIR.glob("*.py"):
        shutil.copy2(path, scripts_dst / path.name)
    copied_docs = 0
    for rel_path in CODE_SNAPSHOT_DOCS:
        src = PROJECT_ROOT / rel_path
        if src.exists():
            shutil.copy2(src, target / src.name)
            copied_docs += 1
    log.info("code_snapshot: src/pak + scripts + tests + %d root docs", copied_docs)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest() -> None:
    manifest_path = RELEASE_DIR / MANIFEST_NAME
    if manifest_path.exists():
        manifest_path.unlink()
    lines: list[str] = []
    for path in sorted(RELEASE_DIR.rglob("*")):
        if path.is_dir() or path.name == MANIFEST_NAME:
            continue
        rel = path.relative_to(RELEASE_DIR).as_posix()
        lines.append(f"{sha256_file(path)}  {rel}")
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("%s: %d files", MANIFEST_NAME, len(lines))


def build_zip() -> Path:
    # File name has no version suffix; Zenodo and HF carry version metadata
    # on the record itself.
    zip_path = RELEASE_DIR.parent / "persona-arts-korea.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root in sorted(RELEASE_DIR.rglob("*")):
            if root.is_dir():
                continue
            arc = root.relative_to(RELEASE_DIR.parent)
            zf.write(root, arcname=str(arc))
    size_mb = zip_path.stat().st_size / 1024 / 1024
    log.info("zip: %s (%.1f MB)", zip_path, size_mb)
    return zip_path


def main() -> None:
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Assembling release at %s", RELEASE_DIR)
    write_personas()
    write_sample_jsonl(n=100)
    copy_grounding()
    copy_code_snapshot()
    write_manifest()
    build_zip()
    log.info("done.")


if __name__ == "__main__":
    main()
