"""PAK-1K-eval — qwen3 self-preference probe judge.

Uses the *same model* as the generator (qwen3:30b-a3b) as a judge to measure
self-preference against the disjoint panel (Claude+Gemini) (Panickssery et al. 2024).

The protocol is **matched exactly** to the existing 4 judges (Claude/Gemini/Clova/Codex):
- Input: web/public/personas_review_sample.json (1,000 personas)
- Uses only the 15 narrative fields (preregistered v3: skills/hobbies _list excluded)
- Includes all 19 quantitative anchors in the reference block
- Separate call per dimension (4 calls per persona), temperature 0.0, max_tokens 600
  → single pass. The other judges are also single-pass, avoiding the 3-repeat asymmetry.
- groundedness/plausibility require inline citation of anchor variable names in the reasoning

Rubric source: groundedness/fluency use the English rubric from audit_prompt/claude.md
verbatim. coherence/plausibility were reconstructed with the same pattern from
web/src/rubric.ts (canonical rubric) because the original run script was deleted uncommitted (noted in the log).

Output: web/public/personas_baseline_qwen3.json (dict keyed by pak_uuid),
atomic partial save per persona.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

# import src from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pak.llm_client import OllamaClient  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("qwen3_judge")

REPO = Path(__file__).resolve().parent.parent
INPUT_PATH = REPO / "web" / "public" / "personas_review_sample.json"
OUTPUT_PATH = REPO / "web" / "public" / "personas_baseline_qwen3.json"
LOG_PATH = REPO / "outputs" / "reports" / "qwen3_baseline_v1_log.md"

GENERATOR_MODEL = "qwen3:30b-a3b"  # paper §5.3 generator model = self-preference judge

ANCHOR_FIELDS = [
    "art_field_primary", "age", "age_band", "sex", "province", "education_level_pak",
    "career_years", "career_band", "employment_type", "is_freelance", "has_secondary_job",
    "individual_art_income_bracket", "household_income_bracket", "has_contract_experience",
    "uses_standard_contract", "has_copyright", "had_career_break", "has_overseas_experience",
    "occupation",
]

NARRATIVE_FIELDS = [
    "persona", "professional_persona", "creative_world_persona", "network_persona",
    "living_persona", "support_persona", "family_persona", "sports_persona", "arts_persona",
    "travel_persona", "culinary_persona", "cultural_background", "skills_and_expertise",
    "hobbies_and_interests", "career_goals_and_ambitions",
]

DIMENSIONS = ["groundedness", "coherence", "plausibility", "fluency"]

# dimensions that require anchor citation
ANCHOR_REQUIRED = {"groundedness", "plausibility"}

SYSTEM_PROMPT = (
    "You are an expert evaluator of synthetic Korean cultural-arts personas. "
    "Score one dimension per call. Respond ONLY with the requested JSON object. /no_think"
)

# per-dimension [Dimension] / [Decision rubric] / [Required reasoning rule]
DIM_HEADER = {
    "groundedness": "GROUNDEDNESS — Does the narrative content match the quantitative anchors?",
    "coherence": "INTERNAL COHERENCE — Do the 15 narrative fields agree as one person?",
    "plausibility": "PLAUSIBILITY — Is the persona a realistic Korean cultural-arts worker (occupation x region x income x activity)?",
    "fluency": "KOREAN FLUENCY — Does the narrative read as native Korean rather than translation or model-typical Korean?",
}

DIM_RUBRIC = {
    # audit_prompt/claude.md verbatim
    "groundedness": (
        "1 = severe contradiction(s)\n"
        "2 = multiple minor contradictions\n"
        "3 = mostly grounded, one minor issue\n"
        "4 = fully grounded, no contradictions\n"
        "5 = grounded *and* uses the anchors in non-trivial, characterful ways"
    ),
    # reconstructed from web/src/rubric.ts (canonical rubric)
    "coherence": (
        "1 = incoherent, reads as multiple different people\n"
        "2 = two or more fields conflict\n"
        "3 = mostly coherent, one field off-key\n"
        "4 = all fields cohere naturally as one person\n"
        "5 = fields reinforce each other into a vivid single person"
    ),
    # reconstructed from web/src/rubric.ts (canonical rubric)
    "plausibility": (
        "1 = implausible (violates geography, economics, or industry practice)\n"
        "2 = possible but very rare, with no explanation\n"
        "3 = within the possible range, slightly atypical\n"
        "4 = a typical case of a Korean cultural-arts worker\n"
        "5 = typical yet richly detailed (real organisation names, practices, regional vocabulary)"
    ),
    # audit_prompt/claude.md verbatim
    "fluency": (
        "5 = no AI tells, native register, character-distinct word choice\n"
        "4 = no AI tells, natural but plain\n"
        "3 = one AI tell or one or two awkward sentences\n"
        "2 = two or more AI tells, or awkward honorifics\n"
        "1 = many AI tells, broken word order or honorifics"
    ),
}

REASON_RULE = {
    "groundedness": (
        "For groundedness, the reasoning MUST inline-cite at least one quantitative anchor "
        "variable by name (e.g., \"career_band=10년 미만 인데 narrative 의 '30년차 베테랑' 과 충돌\"). "
        "Reasoning without an anchor name is incomplete."
    ),
    "plausibility": (
        "For plausibility, the reasoning MUST inline-cite at least one quantitative anchor "
        "variable by name (e.g., \"province=세종 + individual_art_income_bracket=6천만원 이상 = 비현실적\"). "
        "Reasoning without an anchor name is incomplete."
    ),
    "coherence": "",
    "fluency": (
        "AI tell canonical 예: \"황홀한 영역에서\", \"디지털 캔버스라는 무한한 가능성\", "
        "\"이야기를 펼쳐냅니다\", \"예술의 본질을 탐구하며\", 추상명사 + \"의 영역에서\" 패턴."
    ),
}


def render_anchor_block(p: dict) -> str:
    lines = [f"{k}: {p.get(k, '?')}" for k in ANCHOR_FIELDS]
    return "[Quantitative anchors of this persona]\n" + "\n".join(lines)


def render_narrative_block(p: dict) -> str:
    lines = [f"{k}: {p.get(k, '') or ''}" for k in NARRATIVE_FIELDS]
    return "[Narrative (15 fields)]\n" + "\n".join(lines)


def build_user_message(p: dict, dim: str) -> str:
    parts = [
        render_anchor_block(p),
        "",
        render_narrative_block(p),
        "",
        "[Dimension to evaluate]",
        DIM_HEADER[dim],
        "",
        "[Decision rubric]",
        DIM_RUBRIC[dim],
    ]
    rule = REASON_RULE.get(dim, "")
    if rule:
        parts += ["", "[Required reasoning rule]", rule]
    parts += [
        "",
        "[Output format — return ONLY this JSON object, no other text]",
        '{"contradictions": "<bullets or none>", "support": "<bullets>", "score": <integer 1-5>}',
    ]
    return "\n".join(parts)


def parse_response(text: str) -> tuple[int | None, str, str]:
    """Extract (score, contradictions, support) from the qwen3 JSON response."""
    from pak.llm_client import parse_json_response

    try:
        d = parse_json_response(text)
    except (ValueError, json.JSONDecodeError):
        return None, "", ""
    raw = d.get("score")
    score = None
    if isinstance(raw, (int, float)) and 1 <= int(raw) <= 5:
        score = int(raw)
    elif isinstance(raw, str):
        m = re.search(r"[1-5]", raw)
        if m:
            score = int(m.group(0))
    contra = str(d.get("contradictions", "") or "").strip()
    support = str(d.get("support", "") or "").strip()
    return score, contra, support


def has_anchor_citation(reasoning: str) -> bool:
    return any(name in reasoning for name in ANCHOR_FIELDS)


def compress_reason(contra: str, support: str) -> str:
    contra = contra.replace("\n", " ").strip()
    support = support.replace("\n", " ").strip()
    if contra and contra.lower() not in ("none", "none."):
        return f"모순: {contra} / 근거: {support}".strip()[:500]
    return (support or contra)[:500]


_local = threading.local()


def get_thread_client(timeout: float) -> OllamaClient:
    cli = getattr(_local, "client", None)
    if cli is None:
        cli = OllamaClient(base_url="http://localhost:11434", timeout=timeout)
        _local.client = cli
    return cli


def judge_one_dimension(p: dict, dim: str, *, model: str, timeout: float, max_retries: int = 2):
    client = get_thread_client(timeout)
    user_msg = build_user_message(p, dim)
    last_text = ""
    for attempt in range(max_retries + 1):
        res = client.chat(
            model=model, system=SYSTEM_PROMPT, user=user_msg,
            max_tokens=600, temperature=0.0,
            response_format={"type": "json_object"},
        )
        last_text = res.text
        score, contra, support = parse_response(res.text)
        reasoning = compress_reason(contra, support)
        if score is None:
            continue  # PARSE_FAIL → retry
        if dim in ANCHOR_REQUIRED and not has_anchor_citation(contra + " " + support):
            if attempt < max_retries:
                continue  # no anchor → retry
        return score, reasoning
    # failed
    logger.warning("PARSE_FAIL dim=%s uuid=%s text=%.120s", dim, p.get("pak_uuid"), last_text)
    return None, compress_reason(*parse_response(last_text)[1:])


def judge_persona(p: dict, *, model: str, timeout: float) -> dict:
    scores: dict[str, int | None] = {}
    reasoning: dict[str, str] = {}
    for dim in DIMENSIONS:
        s, r = judge_one_dimension(p, dim, model=model, timeout=timeout)
        scores[dim] = s
        reasoning[dim] = r
    return {
        "scores": scores,
        "reasoning": reasoning,
        "flag": "",
        "model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


_save_lock = threading.Lock()


def atomic_save(results: dict, path: Path) -> None:
    with _save_lock:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)


def summarise(results: dict) -> str:
    import numpy as np

    lines = []
    for dim in DIMENSIONS:
        vals = [r["scores"][dim] for r in results.values() if r["scores"].get(dim) is not None]
        if not vals:
            lines.append(f"- {dim}: no valid scores")
            continue
        arr = np.array(vals)
        freq = {k: int((arr == k).sum()) for k in range(1, 6)}
        max_share = max(freq.values()) / len(arr)
        # anchor citation rate (required dimensions only)
        cite = ""
        if dim in ANCHOR_REQUIRED:
            rate = np.mean([
                has_anchor_citation(r["reasoning"][dim]) for r in results.values()
                if r["scores"].get(dim) is not None
            ])
            cite = f" anchor_cite={rate:.2f}"
        lines.append(
            f"- {dim}: mean={arr.mean():.3f} sd={arr.std():.3f} n={len(arr)} "
            f"freq={freq} max_share={max_share:.2f}{cite}"
        )
    return "\n".join(lines)


def write_log_header(model: str, n: int) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = f"""# qwen3 self-preference baseline log

- Start: {datetime.now(timezone.utc).isoformat()}
- Model: {model}
- Family: Alibaba qwen3
- Generator family: Alibaba qwen3 (**SAME family → self-preference probe**, Panickssery et al. 2024)
- Temperature: 0.0, max_tokens: 600
- Pre-reg lock: outputs/reports/pak_1k_eval_preregistration.md (v3)
- Narrative fields: 15 (Amendment 3, _list excluded)
- Per-dim separate inference: yes (single pass, matches the other 4 judges)
- Rubric source: groundedness/fluency verbatim from audit_prompt/claude.md;
  coherence/plausibility reconstructed from web/src/rubric.ts (canonical rubric)
- Target n: {n}

"""
    LOG_PATH.write_text(header, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=GENERATOR_MODEL)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--pilot-stop", action="store_true", help="score only the first 30 personas and stop")
    args = ap.parse_args()

    personas = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    if not isinstance(personas, list):
        personas = list(personas.values())

    if args.pilot_stop:
        batch = personas[0:30]
    else:
        batch = personas[args.start : args.start + args.n]

    # resume from existing results
    results: dict = {}
    if OUTPUT_PATH.exists():
        try:
            results = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
            logger.info("resume: %d existing entries", len(results))
        except json.JSONDecodeError:
            results = {}

    todo = [p for p in batch if p.get("pak_uuid") not in results]
    logger.info("model=%s batch=%d todo=%d concurrency=%d", args.model, len(batch), len(todo), args.concurrency)
    if not (args.pilot_stop and OUTPUT_PATH.exists()):
        write_log_header(args.model, len(batch))

    done = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {
            ex.submit(judge_persona, p, model=args.model, timeout=args.timeout): p
            for p in todo
        }
        for fut in tqdm(as_completed(futs), total=len(futs), desc="judging", unit="persona"):
            p = futs[fut]
            try:
                results[p["pak_uuid"]] = fut.result()
            except Exception as exc:  # noqa: BLE001
                logger.error("persona %s failed: %s", p.get("pak_uuid"), exc)
                continue
            done += 1
            if done % 10 == 0:
                atomic_save(results, OUTPUT_PATH)

    atomic_save(results, OUTPUT_PATH)
    summary = summarise({k: v for k, v in results.items() if k in {p["pak_uuid"] for p in batch}})
    print("\n" + summary)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"\n## {'PILOT' if args.pilot_stop else 'RUN'} done {datetime.now(timezone.utc).isoformat()} "
                f"(n={len(results)})\n\n{summary}\n")

    if args.pilot_stop:
        print("\n[PILOT done] After checking distribution and anchor_cite, say 'go' to run the full scoring:")
        print("  uv run python scripts/run_qwen3_judge.py --start 30 --n 970")


if __name__ == "__main__":
    main()
