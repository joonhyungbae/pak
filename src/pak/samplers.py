"""sampler_specs.json → PAK 샘플러 체인 변환.

NeMo Data Designer는 PyPI 패키지가 아니라 (NVIDIA 내부 경로), 본 모듈은 PAK 자체의
가벼운 sampler 추상화를 제공한다. NeMo와의 어댑터는 ``to_data_designer_columns``로
별도 노출 (data_designer 가용 시).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pak.config import settings

logger = logging.getLogger(__name__)


@dataclass
class CategorySampler:
    """root 카테고리 샘플러: P(value)."""

    name: str
    values: list[Any]
    weights: list[float]

    def sample(self, rng: np.random.Generator, parent_value: Any | None = None) -> Any:
        # parent_value 무시 — root 샘플러
        idx = rng.choice(len(self.values), p=_normalize(self.weights))
        return self.values[idx]


@dataclass
class SubcategorySampler:
    """parent 조건부 샘플러: P(value | parent)."""

    name: str
    parent: str
    subcategories: dict[str, dict[str, list[Any]]]
    """{parent_value: {"values": [...], "weights": [...]}, ...}"""

    def sample(self, rng: np.random.Generator, parent_value: Any) -> Any:
        sub = self.subcategories.get(str(parent_value))
        if sub is None or not sub.get("values"):
            raise KeyError(f"{self.name}: no subcategory entry for parent={parent_value!r}")
        idx = rng.choice(len(sub["values"]), p=_normalize(sub["weights"]))
        return sub["values"][idx]


_AGE_BAND_TO_4GROUP: dict[str, str] = {
    "10대": "30대 이하",
    "20대": "30대 이하",
    "30대": "30대 이하",
    "40대": "40대",
    "50대": "50대",
    "60대": "60세 이상",
    "70대 이상": "60세 이상",
}


def age_band_to_4group(age_band: str) -> str:
    """7구간 age_band → 보고서 4구간 age_group_4."""
    g = _AGE_BAND_TO_4GROUP.get(age_band)
    if g is None:
        raise KeyError(f"unknown age_band: {age_band!r}")
    return g


def _derive_field_age4(state: dict[str, Any]) -> str:
    """art_field_primary + sex_age → "<field>|<age_group_4>" 결합 키."""
    field = str(state["art_field_primary"])
    sex_age = str(state["sex_age"])
    _sex, age_band = split_sex_age(sex_age)
    return f"{field}|{age_band_to_4group(age_band)}"


def _derive_field_employment(state: dict[str, Any]) -> str:
    """art_field_primary + employment_type → "<field>|<employment_type>" 결합 키."""
    field = str(state["art_field_primary"])
    employment_type = str(state["employment_type"])
    return f"{field}|{employment_type}"


_DERIVED_TRANSFORMS: dict[str, Any] = {
    "field_age_group_4_join": _derive_field_age4,
    "field_employment_join": _derive_field_employment,
}


@dataclass
class DerivedSampler:
    """이미 샘플된 값들로부터 합성 키를 만드는 sampler.

    샘플링이 아니라 결정적 변환이지만 chain의 토폴로지 정렬에 합류시키기 위해
    sampler로 표현. ``career_band``처럼 다중 부모 conditional이 필요한 경우
    (예: P(career | field, age_group_4)) 부모 결합 키를 생성하는 단계로 쓰인다.
    """

    name: str
    sources: list[str]
    transform: str

    @property
    def parent(self) -> str | None:
        # topological_order는 마지막으로 의존성이 해결된 순간에 emit하므로,
        # sources의 모든 항목이 emit돼야 자기 자신이 emit됨. 다중 부모를 표현하기 위해
        # parent property로 sources를 노출해 chain 정렬에 활용.
        return self.sources[-1] if self.sources else None

    def derive(self, state: dict[str, Any]) -> Any:
        fn = _DERIVED_TRANSFORMS[self.transform]
        return fn(state)


SamplerLike = CategorySampler | SubcategorySampler | DerivedSampler


def _normalize(weights: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(weights), dtype=np.float64)
    arr = np.clip(arr, 0.0, None)
    s = arr.sum()
    if s <= 0:
        return np.ones_like(arr) / len(arr)
    return arr / s


@dataclass
class SamplerChain:
    """Sampler 의존 그래프. parent → child 순서로 호출."""

    samplers: list[SamplerLike]
    spec_meta: dict[str, Any]

    def topological_order(self) -> list[SamplerLike]:
        """의존성이 해결된 순서로 정렬. DerivedSampler는 sources 전부가 emit돼야 emit."""
        emitted: set[str] = set()
        ordered: list[SamplerLike] = []
        remaining = list(self.samplers)
        while remaining:
            progressed = False
            for s in list(remaining):
                if isinstance(s, DerivedSampler):
                    deps = list(s.sources)
                else:
                    parent = getattr(s, "parent", None)
                    deps = [parent] if parent else []
                if all((d is None) or (d in emitted) for d in deps):
                    ordered.append(s)
                    emitted.add(s.name)
                    remaining.remove(s)
                    progressed = True
            if not progressed:
                names = [s.name for s in remaining]
                raise ValueError(f"sampler chain has unresolved dependencies: {names}")
        return ordered

    def sample_one(
        self, rng: np.random.Generator, *, extras: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """한 페르소나 분량의 정량 변수 dict 생성."""
        out: dict[str, Any] = dict(extras or {})
        for s in self.topological_order():
            if isinstance(s, DerivedSampler):
                out[s.name] = s.derive(out)
                continue
            parent = getattr(s, "parent", None)
            parent_value = out.get(parent) if parent else None
            out[s.name] = s.sample(rng, parent_value)
        return out

    def sample_many(self, n: int, *, seed: int | None = None) -> list[dict[str, Any]]:
        rng = np.random.default_rng(seed)
        return [self.sample_one(rng) for _ in range(n)]


# ----------------------------------------------------------------------------
# Builders
# ----------------------------------------------------------------------------


def build_chain_from_spec(spec_path: Path | None = None) -> SamplerChain:
    if spec_path is None:
        spec_path = settings.grounding_dir / "sampler_specs.json"
    spec: dict[str, Any] = json.loads(spec_path.read_text(encoding="utf-8"))

    samplers: list[SamplerLike] = []
    for s in spec["samplers"]:
        t = s["type"]
        if t == "category":
            samplers.append(
                CategorySampler(
                    name=s["name"], values=list(s["values"]), weights=list(s["weights"])
                )
            )
        elif t == "subcategory":
            samplers.append(
                SubcategorySampler(
                    name=s["name"],
                    parent=s["parent"],
                    subcategories=s["subcategories"],
                )
            )
        elif t == "derived":
            samplers.append(
                DerivedSampler(
                    name=s["name"],
                    sources=list(s["sources"]),
                    transform=str(s["transform"]),
                )
            )
        else:
            raise ValueError(f"unknown sampler type: {t}")

    meta = {k: v for k, v in spec.items() if k != "samplers"}
    return SamplerChain(samplers=samplers, spec_meta=meta)


# ----------------------------------------------------------------------------
# Helpers (PAK 페르소나 정량 필드 후처리)
# ----------------------------------------------------------------------------


def split_sex_age(joined: str) -> tuple[str, str]:
    """'남자|30대' → ('남자', '30대')."""
    sex, age_band = joined.split("|", 1)
    return sex, age_band


def split_employment_freelance(joined: str) -> tuple[str, bool]:
    """'전업|True' → ('전업', True)."""
    emp, free = joined.split("|", 1)
    return emp, free.strip().lower() == "true"


_AGE_BAND_RANGES: dict[str, tuple[int, int]] = {
    "10대": (13, 19),
    "20대": (20, 29),
    "30대": (30, 39),
    "40대": (40, 49),
    "50대": (50, 59),
    "60대": (60, 69),
    "70대 이상": (70, 90),
}

_CAREER_BAND_RANGES: dict[str, tuple[int, int]] = {
    "10년 미만": (0, 9),
    "10-20년 미만": (10, 19),
    "20-30년 미만": (20, 29),
    "30-40년 미만": (30, 39),
    "40년 이상": (40, 50),
}


def sample_age_in_band(rng: np.random.Generator, age_band: str) -> int:
    lo, hi = _AGE_BAND_RANGES[age_band]
    return int(rng.integers(lo, hi + 1))


def min_career_start_age(field: str) -> int:
    """분야별 가장 이른 합리적 경력 시작 나이.

    T4의 경력 구간은 `활동 경력`이다. 조기 훈련을 받을 수 있는 분야라도
    narrative에서는 전문 활동 경력처럼 읽히므로, 너무 어린 시작 나이가
    만들어지지 않도록 보수적인 guardrail을 둔다.
    """
    return {
        "음악": 15,
        "국악": 15,
        "무용": 15,
        "대중음악": 15,
        "미술": 15,
        "공예": 15,
        "만화": 15,
        "연극": 15,
        "방송연예": 15,
        "사진": 18,
        "영화": 18,
        "문학": 18,
        "기타": 18,
        "건축": 18,
    }.get(field, 15)


def compatible_age_range(
    age_band: str,
    career_band: str,
    *,
    min_start_age: int,
) -> tuple[int, int]:
    """주어진 age_band 안에서 career_band를 만족시킬 수 있는 나이 범위."""
    age_lo, age_hi = _AGE_BAND_RANGES[age_band]
    career_lo, _career_hi = _CAREER_BAND_RANGES[career_band]
    lo = max(age_lo, career_lo + min_start_age)
    hi = age_hi
    if lo > hi:
        raise ValueError(
            "incompatible age/career bands: "
            f"age_band={age_band!r}, career_band={career_band!r}, "
            f"min_start_age={min_start_age}"
        )
    return lo, hi


def can_sample_age_for_career(
    age_band: str,
    career_band: str,
    *,
    min_start_age: int,
) -> bool:
    try:
        compatible_age_range(age_band, career_band, min_start_age=min_start_age)
    except ValueError:
        return False
    return True


def sample_age_in_band_for_career(
    rng: np.random.Generator,
    age_band: str,
    career_band: str,
    *,
    min_start_age: int,
) -> int:
    lo, hi = compatible_age_range(age_band, career_band, min_start_age=min_start_age)
    return int(rng.integers(lo, hi + 1))


def sample_career_in_band(
    rng: np.random.Generator,
    career_band: str,
    *,
    max_years: int | None = None,
) -> int:
    lo, hi = _CAREER_BAND_RANGES[career_band]
    if max_years is not None:
        hi = min(hi, max_years)
        if hi < lo:
            raise ValueError(
                f"career_band={career_band!r} cannot fit within max_years={max_years}"
            )
    return int(rng.integers(lo, hi + 1))


# ----------------------------------------------------------------------------
# (선택) NeMo Data Designer 어댑터
# ----------------------------------------------------------------------------


def to_data_designer_columns(chain: SamplerChain) -> list[Any]:
    """data_designer가 설치된 환경에서만 동작.

    Returns ColumnConfig 리스트. data_designer 미설치 시 ImportError.
    """
    try:
        import data_designer.config as dd  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "NeMo Data Designer not installed. PAK 자체 SamplerChain만으로 충분히 작동합니다."
        ) from exc

    columns: list[Any] = []
    for s in chain.samplers:
        if isinstance(s, CategorySampler):
            columns.append(
                dd.SamplerColumnConfig(  # type: ignore[attr-defined]
                    name=s.name,
                    sampler_type=dd.SamplerType.CATEGORY,
                    params=dd.CategorySamplerParams(values=s.values, weights=s.weights),
                )
            )
        else:
            columns.append(
                dd.SamplerColumnConfig(  # type: ignore[attr-defined]
                    name=s.name,
                    sampler_type=dd.SamplerType.SUBCATEGORY,
                    params=dd.SubcategorySamplerParams(
                        parent=s.parent, subcategories=s.subcategories
                    ),
                )
            )
    return columns


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    chain = build_chain_from_spec()
    samples = chain.sample_many(5, seed=20260502)
    for i, s in enumerate(samples):
        print(f"--- persona {i} ---")
        for k, v in s.items():
            print(f"  {k}: {v}")
