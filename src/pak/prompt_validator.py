"""Phase 04 — static validation (linter) for prompt templates.

For each .j2 file:
- Jinja2 syntax validation
- For the 5 domain narratives, check use of field vocabulary / quantitative variables
- Whether forbidden cliches appear
- Whether a fallback exists

CLI: ``uv run python -m pak.prompt_validator``
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, TemplateSyntaxError

from pak.config import settings
from pak.prompt_builder import COMMON_NARRATIVES, DOMAIN_NARRATIVES
from pak.prompts_data import FIELD_META

logger = logging.getLogger(__name__)


PROMPTS_ROOT = settings.project_root / "data" / "prompts"


# Forbidden cliches (regex). It is OK for a prompt to mention a cliche in a
# "forbidden" section rather than telling the model to "use" it — so appearances
# inside a negative context such as `[금지]` or `금지` are allowed.
FORBIDDEN_CLICHES: tuple[str, ...] = (
    r"가난하지만\s*자유로",
    r"고독한\s*천재",
    r"보헤미안",
    r"예술혼\s*불태",
    r"순수한\s*영혼",
    r"고뇌하는\s*예술가",
    r"타고난\s*재능",
    r"운명적으로\s*만난",
)


REQUIRED_VARS_BY_CATEGORY: dict[str, tuple[str, ...]] = {
    "professional": ("art_field_primary", "sex", "age", "career_years", "career_band"),
    "creative_world": ("art_field_primary", "sex", "age", "career_band"),
    "network": ("art_field_primary", "career_band", "province"),
    "living": ("art_field_primary", "sex", "age", "individual_art_income_bracket"),
    "support": ("art_field_primary", "career_band", "career_years"),
}

# Fallback templates may have different required variables, so use a looser set
REQUIRED_VARS_COMMON: tuple[str, ...] = ("art_field_primary",)


@dataclass
class Issue:
    severity: str  # "error" | "warning"
    template_path: str
    code: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.template_path} ({self.code}): {self.message}"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _find_cliches_outside_forbidden_section(src: str) -> list[str]:
    """Treat a cliche as a quotation if it appears near a safe keyword (forbidden/avoid/do-not-use)."""
    hits: list[str] = []
    safe_keywords = ("금지", "회피", "비사용", "지양")
    for pat in FORBIDDEN_CLICHES:
        for m in re.finditer(pat, src):
            window = src[max(0, m.start() - 200) : min(len(src), m.end() + 80)]
            if any(k in window for k in safe_keywords):
                continue
            hits.append(m.group(0))
    return hits


def _check_jinja_syntax(src: str) -> str | None:
    env = Environment(autoescape=False, keep_trailing_newline=True)
    try:
        env.parse(src)
    except TemplateSyntaxError as exc:
        return f"line {exc.lineno}: {exc.message}"
    return None


def _missing_vars(src: str, required: Iterable[str]) -> list[str]:
    missing: list[str] = []
    for var in required:
        # Jinja {{ var }} or {% if var %} / {% for ... in var %}
        pattern = re.compile(rf"{{{{\s*{var}\b|{{%\s*if\s+{var}\b|{{%\s*for\s+\w+\s+in\s+{var}\b")
        if not pattern.search(src):
            missing.append(var)
    return missing


def lint_template(path: Path) -> list[Issue]:
    issues: list[Issue] = []
    rel = str(path.relative_to(PROMPTS_ROOT))
    src = _read(path)

    syn_err = _check_jinja_syntax(src)
    if syn_err:
        issues.append(Issue("error", rel, "JINJA_SYNTAX", syn_err))
        return issues  # other checks run only after syntax passes

    cat = path.parent.name
    if cat in DOMAIN_NARRATIVES:
        required = list(REQUIRED_VARS_BY_CATEGORY.get(cat, ()))
        # Field-specific templates have the field name as the filename, so
        # art_field_primary is hardcoded -> in that case exempt the requirement
        # to use the art_field_primary variable.
        if path.stem != "_fallback" and path.stem in FIELD_META:
            required = [v for v in required if v != "art_field_primary"]
        # Fallbacks are generic, so missing some variables is allowed — enforce only half of required
        if path.stem == "_fallback":
            required = required[:2]
        missing = _missing_vars(src, required)
        if missing:
            issues.append(
                Issue(
                    "error",
                    rel,
                    "MISSING_REQUIRED_VAR",
                    f"missing variables for {cat}: {missing}",
                )
            )
        # For a field-specific template, at least one of field vocabulary / ecosystem / support programs must appear
        if path.stem != "_fallback":
            field = path.stem
            if field not in FIELD_META:
                issues.append(
                    Issue(
                        "error",
                        rel,
                        "UNKNOWN_FIELD",
                        f"template filename {field} not in FIELD_META",
                    )
                )
            else:
                meta = FIELD_META[field]
                pool = (
                    list(meta["vocabulary"])
                    + list(meta["ecosystem"])
                    + list(meta["support_programs"])
                    + list(meta["network_terms"])
                    + list(meta["aesthetic"])
                )
                if not any(v in src for v in pool):
                    issues.append(
                        Issue(
                            "warning",
                            rel,
                            "MISSING_FIELD_VOCABULARY",
                            f"no field-specific vocabulary appears in {cat}/{field}",
                        )
                    )
    elif cat == "_common":
        missing = _missing_vars(src, REQUIRED_VARS_COMMON)
        if missing:
            issues.append(
                Issue("warning", rel, "MISSING_REQUIRED_VAR", f"missing variables: {missing}")
            )

    cliches = _find_cliches_outside_forbidden_section(src)
    if cliches:
        issues.append(
            Issue(
                "error",
                rel,
                "CLICHE_OUTSIDE_FORBIDDEN",
                f"found cliche(s) outside the [금지] (forbidden) context: {cliches}",
            )
        )

    return issues


def check_fallback_completeness() -> list[Issue]:
    """All 5 domain narratives must have a _fallback.j2."""
    issues: list[Issue] = []
    for cat in DOMAIN_NARRATIVES:
        fb = PROMPTS_ROOT / cat / "_fallback.j2"
        if not fb.exists():
            issues.append(
                Issue(
                    "error",
                    f"{cat}/_fallback.j2",
                    "MISSING_FALLBACK",
                    f"{cat} category lacks _fallback.j2",
                )
            )
    for cat in COMMON_NARRATIVES:
        path = PROMPTS_ROOT / "_common" / f"{cat}.j2"
        if not path.exists():
            issues.append(
                Issue(
                    "error",
                    f"_common/{cat}.j2",
                    "MISSING_COMMON_TEMPLATE",
                    f"common narrative {cat} lacks template",
                )
            )
    return issues


def check_field_template_coverage() -> list[Issue]:
    """Whether all 5 domains x 14 fields = 70 field-specific templates exist."""
    issues: list[Issue] = []
    for cat in DOMAIN_NARRATIVES:
        for field in FIELD_META:
            path = PROMPTS_ROOT / cat / f"{field}.j2"
            if not path.exists():
                issues.append(
                    Issue(
                        "warning",
                        f"{cat}/{field}.j2",
                        "MISSING_FIELD_TEMPLATE",
                        "field-specific template missing (fallback used)",
                    )
                )
    return issues


def lint_all() -> list[Issue]:
    issues: list[Issue] = []
    issues.extend(check_fallback_completeness())
    issues.extend(check_field_template_coverage())
    for path in sorted(PROMPTS_ROOT.rglob("*.j2")):
        issues.extend(lint_template(path))
    return issues


def main() -> int:  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    issues = lint_all()
    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]
    for i in issues:
        print(i)
    print(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
