"""Phase 06 — main persona generation entry point.

Workflow:
1. Sample quantitative variables with SamplerChain (based on field-conditional joint distributions).
2. Quantitative -> narrative prompt builder (17 narratives).
3. **single-call** JSON output: generate all 17 narratives in one LLM call.
4. JSON parsing + Pydantic validation.
5. Validation failure -> retry (up to N times) -> discard.
6. Checkpoint save (parquet append).
"""

from __future__ import annotations

import ast
import json
import logging
import random
import re
import time
import uuid
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor

from tqdm.auto import tqdm
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from pak.columns import ExpressionColumnSpec, SamplerColumnSpec
from pak.config import settings
from pak.config_dataset import CompiledPAKDatasetConfig, get_default_pak_core_dataset_config
from pak.llm_client import get_client, parse_json_response
from pak.prompt_builder import (
    SEED_POOLS,
    render_narrative_prompt,
)
from pak.samplers import (
    build_chain_from_spec,
    can_sample_age_for_career,
    min_career_start_age,
    sample_age_in_band_for_career,
    sample_career_in_band,
    split_sex_age,
)
from pak.schema import (
    PAK_CORE_UNSUPPORTED_NULLABLE_COLUMNS,
    PAKPersona,
    PAKPersonaQuant,
    alias_province_to_npk,
)
from pak.validators import ValidationPipeline
from pak.validators.consistency import (
    _AGE_GENERATION_PATTERNS,
    _AGE_PAST_PHASE_CONTEXT,
    _OCCUPATION_ANCHORS,
    _is_self_age_reference,
)

logger = logging.getLogger(__name__)

_STRICT_WARNING_CODES_DEFAULT: tuple[str, ...] = (
    "AGE_MISMATCH",
    "EMPLOYMENT_MISMATCH",
    "OCCUPATION_MISMATCH",
    "REGION_MISMATCH",
    "LANDMARK_REGION_MISMATCH",
    "CAREER_MISMATCH_NEW",
    "CAREER_MISMATCH_VETERAN",
    "EMPLOYMENT_DURATION_CONFLATION",
)


# ----------------------------------------------------------------------------
# Quant sampling
# ----------------------------------------------------------------------------


# Proportionally map from the NPK 7-category to the PAK respondent 3-category distribution
# (report respondents 826 / 2,768 / 1,465 = 16.3% / 54.7% / 28.9%)
_NPK_EDUCATION_BREAKDOWN: dict[str, list[tuple[str, float]]] = {
    "고졸 이하": [("무학", 0.05), ("초등학교", 0.18), ("중학교", 0.18), ("고등학교", 0.59)],
    "대졸 이하": [("2~3년제 전문대학", 0.35), ("4년제 대학교", 0.65)],
    "대학원 이상": [("대학원", 1.0)],
}

_HOUSEHOLD_INCOME_BY_FIELD: dict[str, tuple[float, ...]] = {
    # Table 3-33 artist household total income (page 86), ratios by field
    "문학": (1.6, 17.2, 20.2, 21.1, 9.7, 10.7, 7.2, 2.9, 9.5),
    "미술": (0.4, 11.6, 20.3, 20.2, 15.3, 11.6, 6.4, 4.2, 10.1),
    "공예": (0.8, 9.2, 14.2, 20.5, 11.1, 11.7, 8.7, 8.2, 15.6),
    "사진": (0.4, 12.6, 26.6, 16.2, 14.4, 9.7, 7.6, 4.1, 8.2),
    "건축": (0.0, 3.2, 3.8, 6.6, 3.5, 9.0, 11.0, 7.8, 55.2),
    "음악": (0.4, 8.6, 15.3, 20.9, 12.6, 13.4, 10.6, 5.1, 13.0),
    "국악": (1.9, 11.2, 12.2, 24.6, 15.9, 12.1, 8.8, 6.6, 6.6),
    "대중음악": (0.7, 12.2, 18.9, 20.3, 17.8, 8.1, 5.2, 3.5, 13.3),
    "방송연예": (0.6, 7.1, 17.4, 18.8, 10.7, 14.2, 10.0, 5.5, 15.7),
    "무용": (0.0, 11.4, 16.5, 14.1, 13.8, 9.2, 13.0, 7.6, 14.4),
    "연극": (1.1, 21.9, 23.1, 13.7, 11.2, 11.3, 5.6, 4.2, 7.9),
    "영화": (1.9, 17.5, 19.8, 20.0, 10.4, 7.0, 8.4, 3.3, 11.7),
    "만화": (1.7, 10.6, 16.8, 13.8, 12.3, 7.1, 18.8, 5.1, 13.7),
    "기타": (0.0, 11.4, 24.3, 20.2, 7.4, 3.9, 6.5, 10.2, 16.0),
}

_HOUSEHOLD_INCOME_OPTIONS: tuple[str, ...] = (
    "1천만원 미만",
    "1-2천만원 미만",
    "2-3천만원 미만",
    "3-4천만원 미만",
    "4-5천만원 미만",
    "5-6천만원 미만",
    "6-7천만원 미만",
    "7-8천만원 미만",
    "8천만원 이상",
)

_WORKSPACE_MODE_BY_FIELD: dict[str, tuple[str, ...]] = {
    "문학": (
        "집과 작업 공간의 경계가 느슨한 편",
        "조용한 시간대를 골라 혼자 원고를 다듬는 편",
    ),
    "미술": (
        "재료와 도구 보관이 생활 리듬에 직접 영향을 주는 편",
        "작업실 정리와 제작 준비가 하루 일정의 일부인 편",
    ),
    "공예": (
        "재료 관리와 정리 시간이 작업만큼 중요한 편",
        "손작업 준비와 마감 정돈이 일상에 깊게 들어온 편",
    ),
    "사진": (
        "외부 촬영 일정과 후반 정리 시간이 번갈아 들어오는 편",
        "이동과 보정 작업이 생활 리듬을 좌우하는 편",
    ),
    "건축": (
        "설계 검토와 현장 이동 사이를 오가며 시간을 쓰는 편",
        "도면 검토와 미팅 일정이 생활 리듬을 크게 좌우하는 편",
    ),
    "음악": (
        "연습 시간과 휴식 시간을 분리해 체력을 관리하는 편",
        "반복 연습과 기록 정리가 하루 루틴에 박혀 있는 편",
    ),
    "국악": (
        "연습 시간대와 소리 컨디션 관리가 생활 리듬의 핵심인 편",
        "전승 학습과 개인 연습이 교차하는 일정으로 움직이는 편",
    ),
    "대중음악": (
        "야간 작업과 낮 시간 회복이 교차하는 편",
        "녹음, 편집, 회의가 짧게 끊겨 들어오는 편",
    ),
    "방송연예": (
        "대기 시간과 촬영 시간이 들쭉날쭉한 편",
        "갑작스러운 호출과 준비 시간을 염두에 두고 움직이는 편",
    ),
    "무용": (
        "몸 관리와 리허설 일정이 일상을 강하게 규정하는 편",
        "훈련, 회복, 이동이 생활 리듬을 함께 결정하는 편",
    ),
    "연극": (
        "리허설과 공연 주간에 생활 패턴이 크게 달라지는 편",
        "공연 전후 컨디션 조절이 하루 일정을 좌우하는 편",
    ),
    "영화": (
        "프로젝트 집중기와 공백기가 분명히 나뉘는 편",
        "촬영 현장 일정과 후반 작업 일정이 번갈아 밀려오는 편",
    ),
    "만화": (
        "마감 주간과 비마감 주간의 생활 차이가 큰 편",
        "장시간 앉아 작업한 뒤 짧게 환기하는 루틴이 중요한 편",
    ),
    "기타": (
        "기획, 실행, 정리 업무가 한 주 안에 섞여 있는 편",
        "프로젝트별로 생활 리듬이 달라지는 편",
    ),
}

_WEEKLY_RHYTHM_BY_EMPLOYMENT: dict[str, tuple[str, ...]] = {
    "전업": (
        "주중 작업 시간을 비교적 길게 확보하려고 조정하는 편",
        "작업과 회복을 같은 주 안에서 세밀하게 배분하는 편",
    ),
    "겸업": (
        "생계 일정과 창작 일정을 겹치지 않게 쪼개서 운영하는 편",
        "주중과 주말의 역할이 다르게 나뉘는 편",
    ),
}

_FAMILY_CONTACT_STYLE_BY_AGE: dict[str, tuple[str, ...]] = {
    "10대": ("가까운 보호자나 주변 어른과 일정을 자주 상의하는 편",),
    "20대": (
        "가까운 가족이나 지인과 생활 리듬을 느슨하게 공유하는 편",
        "혼자 시간을 보내도 안부 연락은 비교적 자주 주고받는 편",
    ),
    "30대": (
        "가까운 사람들과 일정 조율이 생활의 일부인 편",
        "작업 일정과 인간관계 사이의 균형을 의식하는 편",
    ),
    "40대": (
        "가까운 관계를 유지하되 작업 시간은 분명히 지키려는 편",
        "주변 돌봄과 자기 작업 시간을 함께 조율하는 편",
    ),
    "50대": (
        "가까운 사람들과의 관계를 안정적으로 유지하려는 편",
        "생활 책임과 작업 지속성을 함께 챙기는 편",
    ),
    "60대": (
        "무리한 사교보다 익숙한 관계를 오래 이어가는 편",
        "생활의 안정감과 작업의 지속성을 같이 살피는 편",
    ),
    "70대 이상": (
        "익숙한 관계와 일상 리듬을 크게 흔들지 않는 편",
        "가까운 사람들과의 안부와 건강 리듬을 함께 챙기는 편",
    ),
}

_SPACE_ANCHOR_BY_FIELD: dict[str, tuple[str, ...]] = {
    "문학": (
        "집 안 작은 책상과 바깥 작업 장소를 번갈아 쓰며 집중 구간을 나누는 편",
        "주거 공간 일부가 자료 더미와 원고 메모로 천천히 작업 공간화되는 편",
    ),
    "미술": (
        "재료 보관과 건조 자리를 확보하느라 거주 공간까지 작업 동선에 묶이는 편",
        "스튜디오 안 수납과 제작 자리를 자주 다시 짜며 공간 밀도를 관리하는 편",
    ),
    "공예": (
        "공방 안 포장, 건조, 재료 정리 자리를 따로 챙기느라 손이 자주 가는 편",
        "작업대 주변 수납과 안전 동선을 직접 손보며 공간을 굴리는 편",
    ),
    "사진": (
        "촬영 장비 가방과 보정용 책상이 생활 공간 일부를 꾸준히 차지하는 편",
        "촬영 준비와 데이터 정리를 위해 이동 가방과 책상을 늘 가동 상태로 두는 편",
    ),
    "건축": (
        "사무소 책상과 현장 이동 가방이 늘 같이 굴러가며 생활 동선을 결정하는 편",
        "도면 검토 자리와 외근 준비 동선이 분리되지 않아 하루 흐름이 자주 바뀌는 편",
    ),
    "음악": (
        "연습 가능한 시간과 소음 부담을 고려해 집과 외부 공간 사용을 나눠 두는 편",
        "악기 보관과 연습 자리를 위해 생활 공간 배치를 자주 손보는 편",
    ),
    "국악": (
        "소리 연습 시간과 생활 소음을 같이 계산하며 공간을 조심스럽게 쓰는 편",
        "연습 자리를 확보하기 위해 거주 공간의 시간대별 쓰임을 나눠 두는 편",
    ),
    "대중음악": (
        "컴퓨터와 장비가 놓인 작업 자리와 쉬는 자리를 의도적으로 분리하려는 편",
        "녹음 장비와 케이블 정리가 생활 공간을 잠식하지 않게 계속 손보는 편",
    ),
    "방송연예": (
        "대기용 짐과 의상 준비가 생활 공간 한쪽을 차지해 출발 동선을 단순하게 두는 편",
        "갑작스러운 호출에 대비한 준비 물품이 늘 눈에 보이는 자리에 놓이는 편",
    ),
    "무용": (
        "연습복과 보호용품, 회복 도구까지 한데 관리하며 공간을 쓰는 편",
        "몸을 풀 수 있는 빈 자리를 남겨 두려다 거주 공간 배치를 자주 바꾸는 편",
    ),
    "연극": (
        "리허설 물품과 대본, 분장 준비를 챙기기 쉬운 배치로 공간을 묶어 두는 편",
        "공연 주간에는 집이 잠깐 창고처럼 바뀔 정도로 준비 물품이 늘어나는 편",
    ),
    "영화": (
        "촬영 장비와 후반 작업 장비가 번갈아 생활 공간을 점유하는 편",
        "프로젝트 기간마다 편집 자리와 보관 자리를 다시 짜며 공간을 굴리는 편",
    ),
    "만화": (
        "책상, 타블렛, 자료 더미가 생활 공간의 중심이 되지 않게 계속 선을 조정하는 편",
        "마감기에는 생활 공간이 곧 작업 공간이 되지 않도록 환기용 자리를 남겨 두는 편",
    ),
    "기타": (
        "프로젝트 문서와 물품 정리가 거주 공간 안쪽까지 들어와 공간 용도를 자주 바꾸는 편",
        "기획 자료와 실행 도구가 섞여 있어 생활 공간을 구획처럼 나눠 쓰는 편",
    ),
}

_EXPENSE_ANCHOR_BY_FIELD: dict[str, tuple[str, ...]] = {
    "문학": (
        "원고료와 강의 수입이 들어오는 간격이 달라 생활비를 달별로 나눠 쓰는 편",
        "기고와 강연 수입의 편차를 고려해 고정 지출을 먼저 묶어 두는 편",
    ),
    "미술": (
        "재료비와 스튜디오 임대료가 한 번에 몰리는 달을 먼저 계산하는 편",
        "판매 수입이 비정기라 제작비를 따로 묶어 두고 생활비를 줄여 가는 편",
    ),
    "공예": (
        "재료 단가와 공방 유지비가 겹치는 시기를 먼저 살펴 주문 속도를 조절하는 편",
        "제작비가 선투입되는 일을 피하려고 공방 비용과 생활비를 촘촘히 나눠 보는 편",
    ),
    "사진": (
        "장비 유지비와 이동비가 한꺼번에 커지는 시즌을 경계하는 편",
        "촬영 수입이 들어와도 장비 교체와 보정 비용을 먼저 떼어 두는 편",
    ),
    "건축": (
        "사무소 운영비와 현장 이동비가 겹치는 달에는 개인 지출을 먼저 줄이는 편",
        "프로젝트 입금 시차를 고려해 사무소 비용과 생활비를 따로 관리하는 편",
    ),
    "음악": (
        "악기 관리비와 레슨 이동비가 겹치는 달에는 생활 지출을 미리 줄이는 편",
        "연주 수입이 들쭉날쭉해 연습 공간 비용부터 먼저 계산하는 편",
    ),
    "국악": (
        "공연 수입과 교육 활동비의 간격이 달라 생활비를 보수적으로 나누는 편",
        "연습과 이동 비용이 커지는 시기엔 다른 지출을 먼저 줄이는 편",
    ),
    "대중음악": (
        "장비 업그레이드와 공간 사용료가 겹치면 생활비를 가장 먼저 조정하는 편",
        "스트리밍보다 프로젝트 수입 비중이 커 입금 시차를 오래 계산하는 편",
    ),
    "방송연예": (
        "출연 간격이 비는 시기를 대비해 생활비를 짧게 나눠 쓰는 편",
        "준비 비용과 이동 비용이 겹치는 달에는 고정 지출부터 다시 손보는 편",
    ),
    "무용": (
        "몸 관리 비용과 연습 공간 비용이 동시에 나가면 다른 지출을 빠르게 줄이는 편",
        "출연료보다 훈련 유지비를 먼저 계산하고 생활비를 남겨 두는 편",
    ),
    "연극": (
        "프로젝트 사이 공백기를 감안해 생활비를 공연 주간과 비공연 주간으로 나눠 쓰는 편",
        "리허설 이동비와 극단 분담금이 겹치면 다른 소비를 먼저 줄이는 편",
    ),
    "영화": (
        "후반 작업비와 촬영 이동비가 겹치는 시기를 길게 보고 생활비를 조정하는 편",
        "프로젝트 입금 간격이 길어 제작 관련 선지출을 따로 관리하는 편",
    ),
    "만화": (
        "마감기 건강 관리비와 어시스트 비용이 겹치는 달을 늘 계산하는 편",
        "플랫폼 정산 주기에 맞춰 생활비와 작업비를 따로 쪼개 쓰는 편",
    ),
    "기타": (
        "프로젝트비 입금 전후로 생활비와 실행비를 따로 잠가 두는 편",
        "행정과 제작 비용이 동시에 생기는 시기를 보고 지출 순서를 자주 바꾸는 편",
    ),
}

_RECOVERY_ANCHOR_BY_FIELD: dict[str, tuple[str, ...]] = {
    "문학": (
        "오래 앉아 쓴 뒤에는 짧은 산책이나 카페 메모 정리로 머리를 비우는 편",
        "원고가 막힐 때는 집 밖 짧은 이동으로 리듬을 다시 맞추는 편",
    ),
    "미술": (
        "제작 뒤에는 정리와 환기 시간을 따로 두며 손의 피로를 식히는 편",
        "재료를 치우는 시간이 곧 회복 시간이 되도록 하루를 마무리하는 편",
    ),
    "공예": (
        "손작업 뒤에는 손을 쉬게 하는 정리 시간을 꼭 두는 편",
        "제작 후 짧은 환기와 청소로 몸의 긴장을 푸는 편",
    ),
    "사진": (
        "촬영 뒤에는 바로 보정보다 짧은 정리 시간을 두며 눈을 쉬게 하는 편",
        "외부 일정 뒤에는 데이터 백업과 짧은 휴식으로 리듬을 다시 세우는 편",
    ),
    "건축": (
        "현장과 미팅이 몰린 날 뒤에는 이동을 줄여 회복 시간을 확보하는 편",
        "도면 검토가 길어진 날은 짧은 산책으로 시선을 환기하는 편",
    ),
    "음악": (
        "연습 뒤에는 몸과 귀를 쉬게 하는 고요한 시간을 일부러 남겨 두는 편",
        "무리한 연습 다음 날은 회복 시간을 먼저 확보하고 일정을 다시 짜는 편",
    ),
    "국악": (
        "소리 컨디션이 무너지지 않게 조용한 회복 시간을 꼭 확보하는 편",
        "집중 연습 뒤에는 목과 몸의 긴장을 풀며 하루 속도를 늦추는 편",
    ),
    "대중음악": (
        "야간 작업 뒤에는 늦은 오전 회복 시간을 비워 두는 편",
        "긴 편집 세션 다음엔 소리 없는 시간으로 귀를 쉬게 하는 편",
    ),
    "방송연예": (
        "촬영 다음 날은 약속을 줄여 몸을 천천히 회복시키는 편",
        "대기 시간이 길었던 날엔 집에 돌아와 말을 줄이며 리듬을 되찾는 편",
    ),
    "무용": (
        "훈련 뒤에는 스트레칭과 냉온 관리까지 회복 루틴으로 묶는 편",
        "몸이 무거운 날엔 이동을 줄이고 회복 시간을 먼저 확보하는 편",
    ),
    "연극": (
        "공연 주간엔 귀가 뒤 말을 줄이며 컨디션을 천천히 되돌리는 편",
        "리허설이 길어진 날은 다음 날 일정을 비워 회복 구간을 만든다",
    ),
    "영화": (
        "현장 집중기가 끝나면 짧게라도 공백 시간을 만들어 리듬을 재정비하는 편",
        "후반 작업이 길어진 뒤에는 바깥 동선 하나로 눈과 몸을 쉬게 하는 편",
    ),
    "만화": (
        "마감 뒤에는 화면을 오래 보지 않는 시간으로 회복 구간을 만드는 편",
        "비마감기에는 생활 리듬을 다시 세우는 데 먼저 시간을 쓰는 편",
    ),
    "기타": (
        "프로젝트 종료 직후에는 약속을 줄이고 빈 시간을 만들어 회복하는 편",
        "행정과 실행이 몰린 뒤에는 생활 동선을 단순하게 줄여 숨을 고르는 편",
    ),
}

_PROVINCE_LOCAL_ROUTINE: dict[str, tuple[str, ...]] = {
    "서울": ("동네 독립서점 둘러보기", "근린 공원 짧은 산책", "작은 전시 공간 방문"),
    "부산": ("해안 근처 짧은 걷기", "동네 시장 한 바퀴 돌기", "작은 공연장 일정 확인"),
    "대구": ("동네 카페에서 메모 정리", "도심 산책로 걷기", "근처 서점 들르기"),
    "인천": ("주말 동네 산책", "근린 시장 방문", "소규모 전시 공간 둘러보기"),
    "광주": ("동네 서점 방문", "근린 산책로 걷기", "작은 문화 공간 들르기"),
    "대전": ("하천변 산책", "동네 카페에서 일정 정리", "근처 서점 방문"),
    "울산": ("주말 산책로 걷기", "근린 체육 공간 이용", "동네 카페에서 정리 시간 보내기"),
    "세종": ("호수공원 근처 걷기", "동네 카페 작업 정리", "근처 서점 둘러보기"),
    "경기": ("동네 산책로 걷기", "주말 독립서점 방문", "근린 카페에서 메모 정리"),
    "강원": ("주변 산책길 걷기", "조용한 카페에서 기록 정리", "지역 소규모 행사 둘러보기"),
    "충청북": ("하천변 산책", "주말 동네 서점 들르기", "작은 문화 행사 확인"),
    "충청남": ("동네 시장 산책", "근린 카페에서 메모 정리", "주말 짧은 드라이브"),
    "전북": ("동네 책방 방문", "천변 산책", "소규모 전시 공간 들르기"),
    "전라남": ("주변 산책길 걷기", "동네 카페에서 일정 정리", "근린 문화 공간 방문"),
    "경상북": ("주말 산책로 걷기", "동네 카페에서 메모 정리", "근린 시장 방문"),
    "경상남": ("근린 체육 공간 이용", "동네 서점 들르기", "주말 짧은 산책"),
    "제주": ("주변 산책길 걷기", "동네 카페에서 메모 정리", "작은 책방 방문"),
    "기타": ("주말 동네 산책", "근처 카페에서 기록 정리", "작은 문화 공간 들르기"),
}

_HOBBY_PLAN_POOLS: dict[str, tuple[str, ...]] = {
    "routine": (
        "아침 카페에서 짧게 기록 정리",
        "야간 산책하며 생각 정리",
        "주말 동네 서점 둘러보기",
        "짧은 메모 산책",
        "한적한 시간에 작은 전시 공간 들르기",
        "아침 차 마시며 하루 계획 세우기",
        "공공도서관 신간 코너 둘러보기",
        "집 근처 벤치에서 호흡 정리",
        "수첩에 하루 지출 기록하기",
        "저녁 시간대 조용한 골목 걷기",
        "주말 빨래 정리하며 음악 듣기",
        "동네 우체국까지 천천히 걷기",
        "휴대폰 사진 폴더 정리하기",
        "아침 스트레칭 루틴 지키기",
        "밤 시간대 따뜻한 차 준비하기",
        "주간 일정표 색깔별로 정리하기",
        "동네 도서관 열람실 들르기",
    ),
    "body_social": (
        "수영",
        "필라테스",
        "배드민턴",
        "저강도 러닝",
        "자전거 타기",
        "가벼운 등산",
        "요가 매트 스트레칭",
        "가벼운 근력 운동",
        "탁구",
        "볼링",
        "실내 클라이밍",
        "줄넘기",
        "느린 홈트레이닝",
        "체력 기록 앱 확인",
        "주말 체육관 이용",
        "계단 오르기",
        "걷기 모임 참석",
        "가벼운 코어 운동",
    ),
    "curiosity": (
        "인문사회 팟캐스트 청취",
        "도시 기록 사진 모으기",
        "로컬 식문화 탐색",
        "생활사 에세이 읽기",
        "라디오 프로그램 아카이브 듣기",
        "식물 관리",
        "지역 뉴스레터 읽기",
        "동네 역사 자료 찾아보기",
        "생활용품 디자인 구경",
        "계절 식재료 기록하기",
        "외국어 단어장 정리",
        "지도 앱으로 골목 탐색",
        "공공도서관 추천 목록 살피기",
        "생활경제 기사 스크랩",
        "동네 간판 사진 분류하기",
        "지역 행사 달력 확인",
        "손글씨 노트 비교하기",
        "생활 소리 녹음 목록 정리",
    ),
    "craft_rest": (
        "핸드드립 커피 내리기",
        "짧은 요리 실험",
        "문구류 정리",
        "작은 소품 수선",
        "향 관련 소도구 모으기",
        "차 우리는 시간 갖기",
        "실내 퍼즐 맞추기",
        "손글씨 연습",
        "천천히 빨래 개기",
        "수첩 꾸미기",
        "계절 향초 정리",
        "간단한 도시락 준비",
        "책갈피 모으기",
        "오래된 파일 라벨 붙이기",
        "집 안 조명 위치 바꾸기",
        "작은 천 가방 손질",
        "간단한 제철 반찬 만들기",
    ),
    "local_extra": (
        "동네 도서관 자료실 들르기",
        "공원 벤치에서 짧게 쉬기",
        "생활권 버스 노선 살피기",
        "작은 생활용품 가게 구경",
        "지역 게시판 일정 확인",
        "동네 빵집 신제품 둘러보기",
        "근처 하천변 잠깐 걷기",
        "아파트 화단 계절 변화 보기",
        "동네 지도에 새 길 표시하기",
        "주말 생활 장보기",
        "지역 커뮤니티 소식 읽기",
        "느린 골목길 산책",
    ),
}

_HOBBY_FAMILY_PATTERNS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("전시 공간", "전시공간", "전시장"), "전시 공간 방문"),
    (("서점", "책방"), "서점 방문"),
    (("산책", "걷기"), "산책"),
    (("카페", "메모 정리"), "카페에서 메모 정리"),
    (("카페", "기록 정리"), "카페에서 기록 정리"),
    (("카페", "일정 정리"), "카페에서 일정 정리"),
    (("메모 정리", "기록 정리", "문구류 정리", "메모 산책"), "기록 정리"),
    (("핸드드립", "커피"), "커피 내리기"),
    (("팟캐스트", "라디오 프로그램 아카이브"), "오디오 콘텐츠 듣기"),
    (("식물",), "식물 관리"),
    (("시장",), "시장 둘러보기"),
    (("문화 공간", "문화 행사"), "문화 공간 방문"),
    (("드라이브",), "드라이브"),
    (("로컬 식문화",), "로컬 식문화 탐색"),
    (("요리",), "요리 실험"),
    (("도시 기록 사진",), "도시 기록 사진 모으기"),
    (("생활사 에세이",), "생활사 에세이 읽기"),
    (("향 관련 소도구",), "향 관련 소도구 모으기"),
    (("소품 수선",), "소품 수선"),
    (("수영",), "수영"),
    (("필라테스",), "필라테스"),
    (("배드민턴",), "배드민턴"),
    (("러닝",), "러닝"),
    (("자전거",), "자전거 타기"),
    (("등산",), "가벼운 등산"),
    (("음악 감상",), "음악 감상"),
)

_LIVING_GENERIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("주중/주말 분할", re.compile(r"주중(?:에는)?[^.]{0,60}주말(?:에는)?")),
    ("작업 시간 길게 확보", re.compile(r"작업 시간을 길게 확보")),
    ("생활 동선 안정", re.compile(r"생활 동선을 안정적으로 묶")),
    ("일상 균형 유지", re.compile(r"일상(?:의)? 균형을 유지|생활과 작업을 조화롭게")),
    ("시간과 공간 타협", re.compile(r"시간과 공간을 타협")),
    ("작업실-거주 병행", re.compile(r"작업실과 거주 공간을 (?:병행|함께 사용|별도로 유지|분리)")),
    ("회복 시간 챙김", re.compile(r"회복 시간을 .*?(?:챙기|확보)")),
)

_FAMILY_GENERIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("가족 시간 조율", re.compile(r"가족과의?\s*(?:시간|일정)\s*조율")),
    ("관계 유지 일반론", re.compile(r"가족과의?\s*관계를?\s*유지")),
    ("생활 책임 반복", re.compile(r"생활\s*책임과\s*작업\s*지속성")),
    ("가까운 가족 공유", re.compile(r"가까운\s*가족과\s*생활\s*리듬")),
    ("돌봄-작업 조율", re.compile(r"주변\s*돌봄과\s*자기\s*작업\s*시간")),
)

_HOBBY_NON_ATOMIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("와 연결", re.compile(r"\S+와\s+\S+")),
    ("과 연결", re.compile(r"\S+과\s+\S+")),
    ("및 연결", re.compile(r"\S+\s+및\s+\S+")),
    ("하고 연결", re.compile(r"\S+\s+하고\s+\S+")),
    ("slash 연결", re.compile(r"/")),
    ("comma 연결", re.compile(r",")),
    ("middle dot 연결", re.compile(r"·")),
)

_NETWORK_GENERIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("중심 역할", re.compile(r"중심적인?\s*(역할|위치)")),
    ("인맥 확장", re.compile(r"인맥을\s*넓히")),
    ("생태계 과장", re.compile(r"생태계에서\s*중심")),
    ("관계 운영 방식", re.compile(r"관계의\s*운영\s*방식")),
    ("중간 조율 고정문", re.compile(r"요청이\s*들어오면\s*연결과\s*조율")),
    ("응답 속도 고정문", re.compile(r"일정\s*충돌과\s*응답\s*속도")),
    ("기존 협업선 고정문", re.compile(r"기존\s*협업선의\s*신뢰")),
)

_SUPPORT_GENERIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("실질 도움 기대", re.compile(r"실질적인?\s*도움.*기대")),
    ("지원 중요 반복", re.compile(r"지원\s*제도를?\s*중요하게\s*여기")),
    ("기대와 거리감", re.compile(r"무엇을\s*기대하고\s*어디서\s*거리감을")),
    ("효율 활용", re.compile(r"자원을\s*효율적으로\s*활용")),
    ("작업시간 확보 일반론", re.compile(r"안정적인?\s*작업\s*시간\s*확보")),
    ("지원제도 스타터", re.compile(r"지원\s*제도는\s*필요할\s*때\s*쓰되")),
)

_CREATIVE_TENSION_BY_FIELD: dict[str, tuple[str, ...]] = {
    "문학": (
        "문장 톤을 과하게 설명하지 않고 정서의 간격으로 밀어붙이는 편",
        "사적인 감정을 바로 고백하기보다 거리감과 리듬으로 돌려 보여 주는 편",
    ),
    "미술": (
        "재료의 질감과 화면의 밀도를 오래 조정하며 기준을 세우는 편",
        "손의 흔적을 남길지 지울지 끝까지 고민하는 편",
    ),
    "공예": (
        "쓰임과 조형감 사이의 균형을 끝까지 붙드는 편",
        "마감의 단정함보다 손에 남는 감각을 더 오래 살피는 편",
    ),
    "사진": (
        "대상을 꾸미기보다 거리와 타이밍으로 인상을 남기는 편",
        "설명적인 장면보다 사라지기 직전의 분위기를 붙잡는 편",
    ),
    "건축": (
        "공간의 효율보다 사람이 머무는 감각을 끝까지 따지는 편",
        "도면의 정합성과 현장의 사용감을 동시에 놓치지 않으려는 편",
    ),
    "음악": (
        "기술적 완성도와 감정의 호흡을 동시에 맞추려는 편",
        "과장된 표현보다 음의 결을 오래 다듬는 편",
    ),
    "국악": (
        "전승의 어법을 지키면서도 자기 호흡으로 다시 세우려는 편",
        "소리의 깊이를 유지하되 낡은 재현으로 보이진 않게 경계하는 편",
    ),
    "대중음악": (
        "즉각적인 반응과 오래 남는 훅 사이의 균형을 자주 고민하는 편",
        "사운드의 세련됨보다 자기 톤이 묻어나는 지점을 찾는 편",
    ),
    "방송연예": (
        "보여지는 캐릭터와 실제 작업 감각 사이의 간격을 계속 조정하는 편",
        "대중적 전달력은 챙기되 얄팍하게 소비되는 이미지는 피하려는 편",
    ),
    "무용": (
        "몸의 선명함보다 움직임이 남기는 여운을 중요하게 여기는 편",
        "기교를 드러내기보다 몸의 긴장과 이완의 흐름을 세밀하게 다루는 편",
    ),
    "연극": (
        "장면의 즉각적인 힘과 인물의 잔향을 함께 살리려는 편",
        "메시지를 앞세우기보다 관계의 온도와 리듬으로 장면을 밀고 가는 편",
    ),
    "영화": (
        "이야기의 설명보다 장면이 남기는 공기를 더 오래 붙드는 편",
        "완결된 메시지보다 인물의 시간감이 화면에 남는 것을 중요하게 여기는 편",
    ),
    "만화": (
        "속도감과 감정선의 밀도를 동시에 챙기려는 편",
        "설정의 화려함보다 컷 사이의 온도 차이를 더 신경 쓰는 편",
    ),
    "기타": (
        "형식보다 전달되는 맥락과 현장의 반응을 함께 보려는 편",
        "낯선 조합을 시도하되 작업의 중심축이 흐려지지 않게 챙기는 편",
    ),
}

_NETWORK_EXCHANGE_BY_FIELD: dict[str, tuple[str, ...]] = {
    "문학": (
        "편집자, 기획자, 동료 작가와 원고의 결을 두고 조심스럽게 의견을 주고받는 편",
        "낭독회, 합평, 원고 청탁 같은 느슨한 연결을 오래 유지하는 편",
    ),
    "미술": (
        "기획자, 갤러리, 설치 보조 인력과 실무적으로 짧고 정확하게 맞추는 편",
        "전시 준비 과정에서 운송, 설치, 일정 협의를 세밀하게 조율하는 편",
    ),
    "공예": (
        "공방, 재료 거래처, 클래스 수강생과 실용적인 대화를 꾸준히 이어가는 편",
        "납품 일정과 제작 속도를 주변 협업자와 자주 맞추는 편",
    ),
    "사진": (
        "촬영 대상자, 편집자, 클라이언트와 결과물의 온도를 먼저 맞추는 편",
        "현장 일정과 후반 작업 인계를 명확하게 나누는 편",
    ),
    "건축": (
        "발주처, 시공, 설계 파트너와 요구사항을 문서와 미팅으로 정리하는 편",
        "현장 판단과 설계 의도를 오가며 협업 언어를 조정하는 편",
    ),
    "음악": (
        "연주자, 지휘, 교육 현장과 합을 맞추며 해석의 차이를 조율하는 편",
        "리허설과 공연 직전의 피드백을 짧고 밀도 있게 주고받는 편",
    ),
    "국악": (
        "선후배 예인, 기획자, 교육 현장과 전승의 맥락을 공유하며 움직이는 편",
        "무대와 수업, 지역 행사 사이에서 관계의 결을 오래 이어가는 편",
    ),
    "대중음악": (
        "세션, 엔지니어, 기획 파트와 파일 단위로 빠르게 실무를 주고받는 편",
        "짧은 피드백 회전 속도 안에서 자기 기준을 지키려는 편",
    ),
    "방송연예": (
        "작가, 연출, 매니지먼트와 타이밍과 톤을 맞추는 편",
        "현장의 대기와 즉흥 대응 속에서도 관계를 무리 없이 이어가는 편",
    ),
    "무용": (
        "안무, 출연진, 스태프와 몸의 감각을 말로 번역해 조율하는 편",
        "리허설 과정에서 반복과 수정의 속도를 함께 맞추는 편",
    ),
    "연극": (
        "배우, 연출, 기술 스태프와 장면의 호흡을 두고 오래 대화하는 편",
        "공연 직전의 작은 수정도 팀과 빠르게 공유하는 편",
    ),
    "영화": (
        "제작, 촬영, 후반 스태프와 공정 단위로 일의 흐름을 맞추는 편",
        "프로젝트 단위로 모였다 흩어지는 관계를 실무적으로 관리하는 편",
    ),
    "만화": (
        "편집자, 플랫폼, 어시스턴트와 마감 리듬을 기준으로 협업하는 편",
        "피드백은 빠르게 받되 작품의 핵심 톤은 쉽게 넘기지 않는 편",
    ),
    "기타": (
        "행정, 기획, 창작 파트 사이의 언어 차이를 중간에서 번역하는 편",
        "프로젝트마다 다른 이해관계자를 무리 없이 연결하는 편",
    ),
}

_NETWORK_SCOPE_BY_FIELD: dict[str, tuple[str, ...]] = {
    "문학": (
        "합평, 편집, 낭독회 같은 느슨한 연결이 이어지는 편",
        "소수 편집자와 동료 작가를 오래 보는 좁은 네트워크를 유지하는 편",
    ),
    "미술": (
        "전시 준비 때만 밀도 높게 붙었다가 평소엔 느슨하게 유지되는 편",
        "기획자와 설치 인력 중심의 실무형 연결이 반복되는 편",
    ),
    "공예": (
        "공방 동료와 거래처, 클래스 수강생이 생활권 안에서 이어지는 편",
        "주문과 납품을 중심으로 작고 반복적인 연결이 쌓이는 편",
    ),
    "사진": (
        "촬영 대상자, 편집자, 클라이언트와 프로젝트 단위로 묶였다 풀리는 편",
        "외부 현장과 후반 작업 인계 중심의 얇은 연결이 이어지는 편",
    ),
    "건축": (
        "사무소 내부와 현장 파트너가 문서와 미팅을 통해 이어지는 편",
        "발주처와 시공 파트의 요구를 오가며 관계가 유지되는 편",
    ),
    "음악": (
        "연습과 공연을 같이 도는 소수 연주자 중심의 연결이 굳는 편",
        "교육 현장과 무대 파트가 겹치며 관계가 천천히 이어지는 편",
    ),
    "국악": (
        "전수, 공연, 지역 행사 축으로 선후배 연결이 겹치는 편",
        "교육 현장과 무대 경험이 함께 네트워크를 굴리는 편",
    ),
    "대중음악": (
        "세션, 엔지니어, 기획 파트가 파일 단위로 빠르게 연결되는 편",
        "작업실 바깥보다 온라인 파일 교환으로 이어지는 비중이 큰 편",
    ),
    "방송연예": (
        "작가, 연출, 매니지먼트와 프로젝트별 접점이 짧게 생겼다 사라지는 편",
        "현장 호출에 따라 인맥보다 타이밍 중심의 연결이 반복되는 편",
    ),
    "무용": (
        "안무가, 출연진, 스태프가 리허설 주기마다 밀집되는 편",
        "공연 시즌마다 가까워졌다가 쉬는 기간엔 느슨해지는 편",
    ),
    "연극": (
        "극단, 외부 스태프, 배우 연결이 작품마다 다시 짜이는 편",
        "반복 협업자 몇 명과 새 프로젝트 인력이 섞이는 편",
    ),
    "영화": (
        "프로젝트가 열릴 때만 밀도 높게 모이는 팀형 연결이 많은 편",
        "제작과 후반 인력이 공정 단위로 이어지는 편",
    ),
    "만화": (
        "편집자, 어시스턴트, 플랫폼 실무자가 마감 주기로 붙는 편",
        "작가 개인 네트워크보다 연재 리듬을 공유하는 실무 연결이 중심인 편",
    ),
    "기타": (
        "행정, 기획, 창작 파트가 프로젝트마다 다르게 섞이는 편",
        "분야 밖 실무자와 느슨한 접점이 자주 생기는 편",
    ),
}

_NETWORK_ROLE_BY_CAREER_BAND: dict[str, tuple[str, ...]] = {
    "신진": (
        "먼저 배우고 소개받는 쪽에 가까운 편",
        "주도하기보다 실무를 받아 적응하는 쪽에 가까운 편",
        "먼저 이름을 알리기보다 작은 실무를 정확히 넘기는 쪽에 가까운 편",
    ),
    "중견": (
        "파트 사이 누락을 줄이기 위해 연락선과 순서를 챙기는 편",
        "앞에서 이끌기보다 필요한 사람을 제때 이어 주는 역할이 잦은 편",
        "실무가 끊기지 않도록 자료 전달과 일정 확인을 맡는 편",
        "주도와 협조 사이를 오가며 각 파트의 속도를 맞추는 편",
    ),
    "원로": (
        "관계 수를 늘리기보다 익숙한 협업선을 오래 유지하는 편",
        "소개와 조언 요청을 받되 직접 개입은 줄이는 편",
        "새 연결을 넓히기보다 오래 본 사람들과의 작업 호흡을 지키는 편",
    ),
}

_NETWORK_FRICTION_DEFAULTS: tuple[str, ...] = (
    "관계 수보다 일정이 어긋나지 않게 확인하는 데 더 에너지가 드는 편",
    "새 인연을 만들기보다 오래 본 협업선의 호흡을 유지하는 일이 더 중요한 편",
    "관계 자체보다 파일 전달, 일정 확인, 역할 구분에서 마찰이 생기기 쉬운 편",
    "연락은 잦지 않아도 한 번 어긋난 일정이 길게 번지는 편이라 확인이 많은 편",
)

_SUPPORT_ATTITUDE_DEFAULTS: tuple[str, ...] = (
    "지원 여부보다 당장 작업 일정이 덜 무너지는지를 먼저 보는 편",
    "지원을 받더라도 행정 에너지보다 남는 시간이 생기는지를 먼저 계산하는 편",
    "규모보다 지금 작업 흐름에 실제 빈틈을 만들어 주는 지원을 더 따지는 편",
)

_SUPPORT_NEED_BREAK_DEFAULTS: tuple[str, ...] = (
    "경력 공백 이후 다시 작업 리듬을 붙잡는 데 도움 되는 지원을 중요하게 여기는 편",
    "끊겼던 작업 흐름을 다시 잇는 데 작은 발판이 되는 지원에 민감한 편",
    "복귀 초반에 시간 감각을 다시 세우는 데 보탬이 되는 지원을 먼저 보는 편",
)

_SUPPORT_ATTITUDE_BREAK_DEFAULTS: tuple[str, ...] = (
    "지원이 경력 리듬을 다시 세우는 데 실제로 쓰일 수 있는지부터 따지는 편",
    "복귀라는 이름보다 다시 움직일 시간을 만들어 주는지가 더 중요한 편",
    "지원 설명보다 끊겼던 흐름을 다시 잇는 데 얼마나 현실적인지부터 보는 편",
)

_SUPPORT_ATTITUDE_SIDEJOB_DEFAULTS: tuple[str, ...] = (
    "지원이 있더라도 생계 일정과 충돌하지 않는 유연성이 있는지부터 보는 편",
    "작은 지원이라도 겸업 시간을 덜 깎아 먹는 구조인지 먼저 따지는 편",
    "선정 여부보다 신청 과정이 생계 리듬을 얼마나 흔드는지를 먼저 보는 편",
)

_SUPPORT_PATH_BY_CONTEXT: dict[str, tuple[str, ...]] = {
    "신진": (
        "작은 제작비나 발표 기회를 주는 초반 지원을 우선 살피는 편",
        "프로필과 포트폴리오를 쌓을 수 있는 초기 지원부터 보는 편",
    ),
    "중견": (
        "창작비보다 다음 프로젝트로 연결되는 지원을 더 따지는 편",
        "발표 이후 이어지는 후속 연결이 있는 지원을 우선 보는 편",
    ),
    "원로": (
        "규모보다 현재 리듬을 크게 흔들지 않는 지원을 우선 보는 편",
        "새 제도보다 익숙한 지원 창구를 선택하는 편",
    ),
}

_SUPPORT_PATH_BREAK_DEFAULTS: tuple[str, ...] = (
    "경력 공백 이후 다시 연결될 수 있는 복귀형 지원을 먼저 보는 편",
    "당장 큰 규모보다 다시 발표 흐름에 올라탈 수 있는 작은 복귀 지원을 먼저 찾는 편",
    "이전 작업 감각을 다시 꺼내 볼 수 있는 재진입형 지원에 더 눈이 가는 편",
)

_SUPPORT_FRICTION_DEFAULTS: tuple[str, ...] = (
    "서류와 정산에 들어가는 시간이 길어지면 지원 자체를 미루는 편",
    "선정 여부보다 일정 고정과 보고 부담이 더 크게 느껴질 때가 있는 편",
    "작은 금액이라도 선지출과 증빙 부담이 겹치면 체감 효용이 떨어지는 편",
)

_SUPPORT_EFFECT_BY_EMPLOYMENT: dict[str, tuple[str, ...]] = {
    "전업": (
        "하루 일정에 숨통이 생겨 작업 블록을 만들 수 있는 효과를 크게 보는 편",
        "제작비 자체보다 일정표에 비어 있는 작업 칸을 확보해 주는 도움이 중요하다고 보는 편",
    ),
    "겸업": (
        "생계 일정과 충돌을 줄여 주는 유연성이 있어야 의미가 생기는 편",
        "작은 지원이라도 겸업 시간 압박을 덜어 주면 체감이 커지는 편",
    ),
}

_SUPPORT_EFFECT_BREAK_DEFAULTS: tuple[str, ...] = (
    "리듬을 다시 세울 수 있는 작은 시간과 비용 여유를 만드는 데 의미를 두는 편",
    "멈췄던 작업 감각을 다시 꺼내 볼 여지를 만드는 효과를 크게 보는 편",
    "다시 시작할 때 필요한 작은 마중물 역할이 생기면 체감이 커지는 편",
)

_FAMILY_RHYTHM_BY_EMPLOYMENT: dict[str, tuple[str, ...]] = {
    "전업": (
        "작업 블록을 먼저 잡아 두고 관계 약속은 그 빈칸에 맞춰 넣는 편",
        "마감 전후로 연락 밀도가 크게 달라지는 편이라 가까운 사람들과 리듬을 미리 맞추는 편",
        "작업 시간이 길어지는 주에는 약속을 줄이고 끝난 뒤 몰아서 시간을 쓰는 편",
    ),
    "겸업": (
        "생계 일정이 먼저 잡히면 남는 시간대에 관계 약속을 얹는 편",
        "주간 일정표를 먼저 공유하고 가능한 시간만 관계 약속으로 남겨 두는 편",
        "고정 근무나 수업 일정이 생기면 가까운 사람들과 약속 시간을 더 일찍 확정하는 편",
    ),
}

_FAMILY_BOUNDARY_DEFAULTS: tuple[str, ...] = (
    "작업 얘기를 길게 가져가기보다 필요한 일정만 짧게 공유하는 편",
    "마감기엔 연락 빈도를 줄이고 끝난 뒤 한 번에 시간을 내는 편",
    "도움이 필요할 때만 구체적으로 부탁하고 평소엔 작업 이야기를 짧게 두는 편",
    "가까운 관계라도 작업 판단 자체는 혼자 정리한 뒤 결과만 공유하는 편",
)

_FAMILY_RESPONSIBILITY_DEFAULTS: tuple[str, ...] = (
    "집안 일과 심부름을 한꺼번에 처리한 뒤 긴 작업 시간을 만드는 편",
    "돌봄이나 약속 요청이 겹치면 작업 블록을 앞뒤로 옮겨 대응하는 편",
    "관계 쪽 일정 변동이 생기면 작업량을 잘게 나눠 맞추는 편",
    "작은 생활 책임은 미리 몰아 처리해 작업 시간을 지키려는 편",
)

_SUPPORT_DECISION_BY_CONTEXT: dict[str, tuple[str, ...]] = {
    "신진": (
        "공고를 보면 자격 조건과 준비 시간부터 빠르게 가늠하고 맞는 것만 남기는 편",
        "일단 메모해 두었다가 제출 부담이 큰 공고는 초반에 접는 편",
        "작은 발표나 제작 기회가 보이는 공고만 추려 끝까지 검토하는 편",
    ),
    "중견": (
        "후속 연결이 보이는 공고만 남기고 애매한 것은 초반에 정리하는 편",
        "정산 부담이 큰 공고는 마감 전에 접고, 이어질 프로젝트가 보이는 것만 끝까지 보는 편",
        "비슷한 지원 중에서도 다음 일로 이어질 가능성이 있는 것만 골라보는 편",
    ),
    "원로": (
        "익숙한 창구만 먼저 확인하고 새 제도는 천천히 살피는 편",
        "제출 구조가 비슷한 지원만 남기고 낯선 형식은 일찍 거르는 편",
        "규모보다 지금 리듬에 맞는 공고만 추려 보는 편",
    ),
}

_SUPPORT_DECISION_BREAK_DEFAULTS: tuple[str, ...] = (
    "복귀 흐름과 직접 맞닿은 공고만 남기고 나머지는 빠르게 접는 편",
    "다시 시작하는 데 바로 도움이 되는 지원만 끝까지 보고 나머지는 미루는 편",
    "재진입에 필요한 작은 발판이 보이는 공고만 고르고 그 외는 일찍 정리하는 편",
)

_SUPPORT_DECISION_SIDEJOB_DEFAULTS: tuple[str, ...] = (
    "신청 준비가 생계 일정과 충돌하면 초반에 접고 유연한 공고만 남기는 편",
    "제출 부담이 큰 공고는 일찍 거르고 일정 조정이 가능한 지원만 끝까지 보는 편",
    "겸업 시간을 덜 깎아 먹는 지원만 추려 검토하는 편",
)


def _sample_npk_education(pak_level: str, rng: random.Random) -> str:
    """Proportionally distribute from the PAK 3-category to the NPK 7-category."""
    options = _NPK_EDUCATION_BREAKDOWN[pak_level]
    weights = [w for _, w in options]
    return rng.choices([c for c, _ in options], weights=weights, k=1)[0]


def _pick_least_used_option(
    options: tuple[str, ...],
    counts: Counter[str] | None,
    rng: random.Random,
) -> str:
    if not options:
        return ""
    if counts is None:
        return rng.choice(options)
    scored = sorted((int(counts.get(option, 0)), rng.random(), option) for option in options)
    return scored[0][2]


def _sample_district(province_npk: str, rng: random.Random) -> str | None:
    """v0.1: with a single PDF source, the district is not filled in.

    Args:
        province_npk: reserved argument for future integration with a district grounding sampler.
        rng: reserved argument for future probabilistic sampling.
    """
    del province_npk, rng
    return None


def _parse_hobby_items(value: Any) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        parsed = None
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    parts = [part.strip(" []'\"") for part in value.split(",")]
    return [part for part in parts if part]


def _canonicalize_hobby_atom(value: str) -> str:
    text = " ".join(str(value).split())
    if ("전시" in text and "공간" in text) or "전시장" in text:
        return "전시 공간 방문"
    if "음악 감상" in text:
        return "음악 감상"
    if "독서" in text or "책 읽기" in text:
        return "독서"
    if "서점" in text or "책방" in text:
        return "서점 방문"
    if "카페" in text and any(token in text for token in ("메모", "기록", "일정 정리")):
        return "카페에서 기록 정리"
    if "산책" in text or "걷기" in text:
        return "산책"
    if any(token in text for token in ("메모 정리", "기록 정리", "문구류 정리", "메모 산책")):
        return "기록 정리"
    if "핸드드립" in text or "커피" in text:
        return "커피 내리기"
    if "팟캐스트" in text or "라디오 프로그램 아카이브" in text:
        return "오디오 콘텐츠 듣기"
    if "식물" in text:
        return "식물 관리"
    if "시장" in text:
        return "시장 둘러보기"
    if "문화 공간" in text or "문화 행사" in text:
        return "문화 공간 방문"
    if "드라이브" in text:
        return "드라이브"
    if "로컬 식문화" in text:
        return "로컬 식문화 탐색"
    if "요리" in text:
        return "요리 실험"
    if "도시 기록 사진" in text:
        return "도시 기록 사진 모으기"
    if "생활사 에세이" in text:
        return "생활사 에세이 읽기"
    if "향 관련 소도구" in text:
        return "향 관련 소도구 모으기"
    if "소품 수선" in text:
        return "소품 수선"
    if "수영" in text:
        return "수영"
    if "필라테스" in text:
        return "필라테스"
    if "배드민턴" in text:
        return "배드민턴"
    if "러닝" in text:
        return "러닝"
    if "자전거" in text:
        return "자전거 타기"
    if "등산" in text:
        return "가벼운 등산"
    return text


def _pick_diverse_hobby_item(
    candidates: tuple[str, ...],
    *,
    hobby_item_counts: Counter[str] | None,
    recent_hobby_family_counts: Counter[str] | None,
    used_families: set[str],
    blocked_items: set[str],
    blocked_families: set[str],
    rng: random.Random,
) -> tuple[str, str]:
    scored: list[tuple[int, int, int, float, str, str]] = []
    for item in candidates:
        family = _canonicalize_hobby_atom(item)
        primary_penalty = 0
        if item in blocked_items:
            primary_penalty += 100
        if family in blocked_families:
            primary_penalty += 100
        if family in used_families:
            primary_penalty += 10
        exact_penalty = int((hobby_item_counts or Counter()).get(item, 0))
        recent_penalty = int((recent_hobby_family_counts or Counter()).get(family, 0))
        scored.append((primary_penalty, exact_penalty, recent_penalty, rng.random(), item, family))
    scored.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
    _, _, _, _, item, family = scored[0]
    used_families.add(family)
    return item, family


def _occupation_from_field(field: str, rng: random.Random) -> str:
    """art_field_primary -> free-text occupation (NPK compatible)."""
    mapping: dict[str, list[str]] = {
        "문학": ["시인", "소설가", "수필가", "평론가", "동화작가", "번역가"],
        "미술": ["회화 작가", "조각가", "판화가", "설치미술 작가", "미디어아트 작가"],
        "공예": ["도예가", "금속공예가", "섬유공예가", "목공예가", "유리공예가"],
        "사진": ["다큐멘터리 사진가", "파인아트 사진가", "광고 사진가"],
        "건축": ["건축가", "인테리어 디자이너", "조경설계가"],
        "음악": ["작곡가", "지휘자", "성악가", "기악 연주자"],
        "국악": ["국악인", "판소리 명창", "기악 연주자(국악)"],
        "대중음악": ["싱어송라이터", "작곡가(대중음악)", "세션 연주자", "음악 프로듀서"],
        "방송연예": ["방송 출연자", "예능인", "방송작가", "MC"],
        "무용": ["무용가", "안무가"],
        "연극": ["배우", "연출가", "극작가", "무대미술가"],
        "영화": ["영화감독", "시나리오 작가", "촬영감독", "편집기사", "프로듀서"],
        "만화": ["만화가", "웹툰 작가", "스토리 작가"],
        "기타": ["문화예술 매개자", "예술 행정", "융복합 예술가"],
    }
    return rng.choice(mapping.get(field, ["예술인"]))


def sample_full_quant(chain, rng: random.Random, np_rng) -> dict[str, Any]:
    """Materialize a quant row following the compiled dataset config."""
    dataset_cfg = get_default_pak_core_dataset_config()
    raw: dict[str, Any] | None = None
    for _ in range(64):
        candidate = chain.sample_one(np_rng)
        field = candidate["art_field_primary"]
        _, age_band = split_sex_age(candidate["sex_age"])
        career_band = candidate["career_band"]
        if can_sample_age_for_career(
            age_band,
            career_band,
            min_start_age=min_career_start_age(field),
        ):
            raw = candidate
            break
    if raw is None:
        raise RuntimeError("failed to sample a compatible age/career combination")

    state: dict[str, Any] = {}
    for name in dataset_cfg.topological_order:
        spec = dataset_cfg.get_column(name)
        if isinstance(spec, SamplerColumnSpec):
            if spec.sampler_kind == "grounding-chain":
                source_key = spec.source_key or spec.name
                if source_key not in raw:
                    raise KeyError(f"missing raw grounding key for {spec.name!r}: {source_key!r}")
                state[name] = raw[source_key]
            elif spec.sampler_kind == "derived":
                state[name] = _execute_derived_sampler(spec, state, rng=rng, np_rng=np_rng)
            else:  # pragma: no cover
                raise ValueError(f"unknown sampler_kind: {spec.sampler_kind}")
        elif isinstance(spec, ExpressionColumnSpec):
            state[name] = _execute_expression(spec, state, rng=rng)

    quant_fields = list(PAKPersonaQuant.model_fields)
    missing = [field for field in quant_fields if field not in state]
    if missing:
        raise KeyError(f"materialized quant missing fields: {missing}")
    return {field: state[field] for field in quant_fields}


def _sample_household_income(field: str, np_rng) -> str:
    """Sample from the by-field household total income distribution in Table 3-33."""
    weights = _HOUSEHOLD_INCOME_BY_FIELD.get(field)
    if weights is None:
        raise KeyError(f"missing household income distribution for field={field!r}")
    probs = [w / sum(weights) for w in weights]
    idx = int(np_rng.choice(len(_HOUSEHOLD_INCOME_OPTIONS), p=probs))
    return _HOUSEHOLD_INCOME_OPTIONS[idx]


# ----------------------------------------------------------------------------
# Single-call narrative generation
# ----------------------------------------------------------------------------


def _sample_lifestyle_facts(quant: dict[str, Any], rng: random.Random) -> dict[str, str]:
    field = str(quant.get("art_field_primary", "기타"))
    age_band = str(quant.get("age_band", "30대"))
    province = str(quant.get("province", "기타"))
    employment = str(quant.get("employment_type", "전업"))
    income = str(quant.get("individual_art_income_bracket", "없음"))
    household_income = str(quant.get("household_income_bracket", "3-4천만원 미만"))
    had_break = bool(quant.get("had_career_break"))

    if income in {"없음", "5백만원 미만", "5백-1천만원 미만"}:
        housing_pressure = rng.choice(
            [
                "생활비와 작업비 사이의 균형을 자주 계산하는 편",
                "지출을 세심하게 조정하며 작업 지속성을 관리하는 편",
            ]
        )
    elif household_income in {"6-7천만원 미만", "7-8천만원 미만", "8천만원 이상"}:
        housing_pressure = rng.choice(
            [
                "주거와 작업 공간을 비교적 안정적으로 유지하려는 편",
                "생활 기반을 크게 흔들지 않으면서 작업 시간을 확보하려는 편",
            ]
        )
    else:
        housing_pressure = rng.choice(
            [
                "생활 리듬을 해치지 않는 범위에서 작업비를 조절하는 편",
                "고정 지출과 작업 비용을 함께 살피며 움직이는 편",
            ]
        )

    support_need = rng.choice(
        [
            "행정 절차와 일정 조율 부담을 줄여 주는 지원에 민감한 편",
            "작업 블록을 비워 주는 지원에 반응하는 편",
            "공간, 시간, 비용을 아껴 주는 실질 지원에 반응하는 편",
        ]
    )
    if had_break:
        support_need = rng.choice(_SUPPORT_NEED_BREAK_DEFAULTS)

    family_contact_style = rng.choice(
        _FAMILY_CONTACT_STYLE_BY_AGE.get(age_band, _FAMILY_CONTACT_STYLE_BY_AGE["30대"])
    )
    family_rhythm = _pick_least_used_option(
        _FAMILY_RHYTHM_BY_EMPLOYMENT.get(
            employment,
            _FAMILY_RHYTHM_BY_EMPLOYMENT["전업"],
        ),
        None,
        rng,
    )
    family_boundary = _pick_least_used_option(_FAMILY_BOUNDARY_DEFAULTS, None, rng)
    family_responsibility = _pick_least_used_option(_FAMILY_RESPONSIBILITY_DEFAULTS, None, rng)
    space_anchor = rng.choice(
        _SPACE_ANCHOR_BY_FIELD.get(field, _SPACE_ANCHOR_BY_FIELD["기타"])
    )
    expense_anchor = rng.choice(
        _EXPENSE_ANCHOR_BY_FIELD.get(field, _EXPENSE_ANCHOR_BY_FIELD["기타"])
    )
    recovery_anchor = rng.choice(
        _RECOVERY_ANCHOR_BY_FIELD.get(field, _RECOVERY_ANCHOR_BY_FIELD["기타"])
    )
    workspace_mode = rng.choice(
        _WORKSPACE_MODE_BY_FIELD.get(field, _WORKSPACE_MODE_BY_FIELD["기타"])
    )
    weekly_rhythm = rng.choice(
        _WEEKLY_RHYTHM_BY_EMPLOYMENT.get(employment, _WEEKLY_RHYTHM_BY_EMPLOYMENT["전업"])
    )
    local_routine = rng.choice(
        _PROVINCE_LOCAL_ROUTINE.get(province, _PROVINCE_LOCAL_ROUTINE["기타"])
    )

    return {
        "family_contact_style": family_contact_style,
        "family_rhythm": family_rhythm,
        "family_boundary": family_boundary,
        "family_responsibility": family_responsibility,
        "housing_pressure": housing_pressure,
        "space_anchor": space_anchor,
        "expense_anchor": expense_anchor,
        "recovery_anchor": recovery_anchor,
        "workspace_mode": workspace_mode,
        "weekly_rhythm": weekly_rhythm,
        "local_routine_hint": local_routine,
        "support_need": support_need,
    }


def _sample_persona_blueprint(
    quant: dict[str, Any],
    lifestyle_facts: dict[str, str],
    rng: random.Random,
    *,
    network_role_counts: Counter[str] | None = None,
    network_friction_counts: Counter[str] | None = None,
    support_attitude_counts: Counter[str] | None = None,
    support_decision_counts: Counter[str] | None = None,
    support_path_counts: Counter[str] | None = None,
    support_friction_counts: Counter[str] | None = None,
    support_effect_counts: Counter[str] | None = None,
    family_rhythm_counts: Counter[str] | None = None,
    family_boundary_counts: Counter[str] | None = None,
    family_responsibility_counts: Counter[str] | None = None,
) -> dict[str, str]:
    field = str(quant.get("art_field_primary", "기타"))
    career_band = str(quant.get("career_band", "중견"))
    employment = str(quant.get("employment_type", "전업"))
    income = str(quant.get("individual_art_income_bracket", "없음"))
    had_break = bool(quant.get("had_career_break"))

    creative_tension = rng.choice(
        _CREATIVE_TENSION_BY_FIELD.get(field, _CREATIVE_TENSION_BY_FIELD["기타"])
    )
    network_exchange_mode = rng.choice(
        _NETWORK_EXCHANGE_BY_FIELD.get(field, _NETWORK_EXCHANGE_BY_FIELD["기타"])
    )
    network_scope = rng.choice(
        _NETWORK_SCOPE_BY_FIELD.get(field, _NETWORK_SCOPE_BY_FIELD["기타"])
    )
    network_role = _pick_least_used_option(
        _NETWORK_ROLE_BY_CAREER_BAND.get(career_band, _NETWORK_ROLE_BY_CAREER_BAND["중견"]),
        network_role_counts,
        rng,
    )
    network_friction = _pick_least_used_option(
        _NETWORK_FRICTION_DEFAULTS,
        network_friction_counts,
        rng,
    )
    family_rhythm = _pick_least_used_option(
        _FAMILY_RHYTHM_BY_EMPLOYMENT.get(
            employment,
            _FAMILY_RHYTHM_BY_EMPLOYMENT["전업"],
        ),
        family_rhythm_counts,
        rng,
    )
    family_boundary = _pick_least_used_option(
        _FAMILY_BOUNDARY_DEFAULTS,
        family_boundary_counts,
        rng,
    )
    family_responsibility = _pick_least_used_option(
        _FAMILY_RESPONSIBILITY_DEFAULTS,
        family_responsibility_counts,
        rng,
    )

    if employment == "겸업":
        living_tradeoff = rng.choice(
            [
                "생계 일정 사이에 작업 시간을 쪼개 넣기 때문에 생활 동선이 자주 잘게 나뉘는 편",
                "수입원 일정에 맞춰 작업 집중 시간을 따로 확보하려고 생활 리듬을 자주 조정하는 편",
            ]
        )
    elif income in {"없음", "5백만원 미만", "5백-1천만원 미만"}:
        living_tradeoff = rng.choice(
            [
                "작업 시간을 지키기 위해 소비와 이동 동선을 최대한 단순하게 줄이는 편",
                "생활비와 작업비의 균형을 맞추느라 일상 리듬을 세밀하게 다듬는 편",
            ]
        )
    else:
        living_tradeoff = rng.choice(
            [
                "작업실 유지와 회복 시간을 함께 챙기기 위해 생활 동선을 안정적으로 묶는 편",
                "작업 효율을 해치지 않는 선에서 생활 패턴을 단정하게 유지하려는 편",
            ]
        )

    if had_break:
        support_attitude = _pick_least_used_option(
            _SUPPORT_ATTITUDE_BREAK_DEFAULTS,
            support_attitude_counts,
            rng,
        )
    elif employment == "겸업":
        support_attitude = _pick_least_used_option(
            _SUPPORT_ATTITUDE_SIDEJOB_DEFAULTS,
            support_attitude_counts,
            rng,
        )
    else:
        support_attitude = _pick_least_used_option(
            _SUPPORT_ATTITUDE_DEFAULTS,
            support_attitude_counts,
            rng,
        )

    if had_break:
        support_decision = _pick_least_used_option(
            _SUPPORT_DECISION_BREAK_DEFAULTS,
            support_decision_counts,
            rng,
        )
    elif employment == "겸업":
        support_decision = _pick_least_used_option(
            _SUPPORT_DECISION_SIDEJOB_DEFAULTS,
            support_decision_counts,
            rng,
        )
    else:
        support_decision = _pick_least_used_option(
            _SUPPORT_DECISION_BY_CONTEXT.get(career_band, _SUPPORT_DECISION_BY_CONTEXT["중견"]),
            support_decision_counts,
            rng,
        )

    support_path = _pick_least_used_option(
        _SUPPORT_PATH_BY_CONTEXT.get(career_band, _SUPPORT_PATH_BY_CONTEXT["중견"]),
        support_path_counts,
        rng,
    )
    support_friction = _pick_least_used_option(
        _SUPPORT_FRICTION_DEFAULTS,
        support_friction_counts,
        rng,
    )
    support_effect = _pick_least_used_option(
        _SUPPORT_EFFECT_BY_EMPLOYMENT.get(employment, _SUPPORT_EFFECT_BY_EMPLOYMENT["전업"]),
        support_effect_counts,
        rng,
    )
    if had_break:
        support_path = _pick_least_used_option(
            _SUPPORT_PATH_BREAK_DEFAULTS,
            support_path_counts,
            rng,
        )
        support_effect = _pick_least_used_option(
            _SUPPORT_EFFECT_BREAK_DEFAULTS,
            support_effect_counts,
            rng,
        )

    persona_focus = rng.choice(
        [
            lifestyle_facts["weekly_rhythm"],
            lifestyle_facts["workspace_mode"],
            living_tradeoff,
            creative_tension,
        ]
    )

    return {
        "creative_tension": creative_tension,
        "network_exchange_mode": network_exchange_mode,
        "network_scope": network_scope,
        "network_role": network_role,
        "network_friction": network_friction,
        "family_rhythm": family_rhythm,
        "family_boundary": family_boundary,
        "family_responsibility": family_responsibility,
        "living_tradeoff": living_tradeoff,
        "support_attitude": support_attitude,
        "support_decision": support_decision,
        "support_path": support_path,
        "support_friction": support_friction,
        "support_effect": support_effect,
        "persona_focus": persona_focus,
    }


def _sample_hobby_plan(
    quant: dict[str, Any],
    lifestyle_facts: dict[str, str],
    rng: random.Random,
    *,
    hobby_item_counts: Counter[str] | None = None,
    recent_hobby_family_counts: Counter[str] | None = None,
    blocked_items: set[str] | None = None,
    blocked_families: set[str] | None = None,
) -> dict[str, Any]:
    province = str(quant.get("province", "기타"))
    blocked_exact = set(blocked_items or set())
    blocked = set(blocked_families or set())
    used_families: set[str] = set()
    category_pools: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("routine", _HOBBY_PLAN_POOLS["routine"]),
        ("body/social", _HOBBY_PLAN_POOLS["body_social"]),
        ("non-art curiosity", _HOBBY_PLAN_POOLS["curiosity"]),
        ("rest/craft", _HOBBY_PLAN_POOLS["craft_rest"]),
        (
            "local",
            _PROVINCE_LOCAL_ROUTINE.get(province, _PROVINCE_LOCAL_ROUTINE["기타"])
            + _HOBBY_PLAN_POOLS["local_extra"],
        ),
    )
    selections: list[tuple[str, str, str]] = []
    for category, pool in category_pools:
        item, family = _pick_diverse_hobby_item(
            pool,
            hobby_item_counts=hobby_item_counts,
            recent_hobby_family_counts=recent_hobby_family_counts,
            used_families=used_families,
            blocked_items=blocked_exact,
            blocked_families=blocked,
            rng=rng,
        )
        selections.append((category, item, family))

    items = [item for _, item, _ in selections]
    plan_lines = [f"- {category}: {item}" for category, item, _ in selections]
    return {
        "hobby_plan_items": items,
        "hobby_plan_families": [family for _, _, family in selections],
        "hobby_plan_text": ", ".join(items),
        "hobby_plan_lines": "\n".join(plan_lines),
        "local_routine_hint": lifestyle_facts["local_routine_hint"],
        "blocked_hobby_items": sorted(blocked_exact),
        "blocked_hobby_item_text": ", ".join(sorted(blocked_exact)),
        "blocked_hobby_families": sorted(blocked),
        "blocked_hobby_family_text": ", ".join(sorted(blocked)),
    }


def _refresh_hobby_prompt_context(
    prompt_context: dict[str, Any],
    quant: dict[str, Any],
    rng: random.Random,
    *,
    hobby_item_counts: Counter[str] | None = None,
    recent_hobby_family_counts: Counter[str] | None = None,
    blocked_items: set[str] | None = None,
    blocked_families: set[str] | None = None,
    seen_hobby_sets: set[tuple[str, ...]] | None = None,
    seen_hobby_family_sets: set[tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    updated = dict(prompt_context)
    best_plan: dict[str, Any] | None = None
    best_score: tuple[int, int, float] | None = None
    for _ in range(24):
        plan = _sample_hobby_plan(
            quant,
            updated,
            rng,
            hobby_item_counts=hobby_item_counts,
            recent_hobby_family_counts=recent_hobby_family_counts,
            blocked_items=blocked_items,
            blocked_families=blocked_families,
        )
        hobby_value = repr(plan["hobby_plan_items"])
        exact_dup = int(
            bool(seen_hobby_sets is not None and _normalized_hobby_set(hobby_value) in seen_hobby_sets)
        )
        family_dup = int(
            bool(
                seen_hobby_family_sets is not None
                and _normalized_hobby_family_set(hobby_value) in seen_hobby_family_sets
            )
        )
        repeat_pressure = sum(
            int((hobby_item_counts or Counter()).get(item, 0))
            for item in plan["hobby_plan_items"]
        )
        score = (family_dup, exact_dup, repeat_pressure + rng.random())
        if best_score is None or score < best_score:
            best_plan = plan
            best_score = score
        if family_dup == 0 and exact_dup == 0:
            break
    updated.update(best_plan or {})
    return updated


def _build_prompt_context(
    quant: dict[str, Any],
    rng: random.Random,
    *,
    hobby_item_counts: Counter[str] | None = None,
    recent_hobby_family_counts: Counter[str] | None = None,
    blocked_items: set[str] | None = None,
    blocked_families: set[str] | None = None,
    seen_hobby_sets: set[tuple[str, ...]] | None = None,
    seen_hobby_family_sets: set[tuple[str, ...]] | None = None,
    network_role_counts: Counter[str] | None = None,
    network_friction_counts: Counter[str] | None = None,
    family_rhythm_counts: Counter[str] | None = None,
    family_boundary_counts: Counter[str] | None = None,
    family_responsibility_counts: Counter[str] | None = None,
    support_attitude_counts: Counter[str] | None = None,
    support_decision_counts: Counter[str] | None = None,
    support_path_counts: Counter[str] | None = None,
    support_friction_counts: Counter[str] | None = None,
    support_effect_counts: Counter[str] | None = None,
) -> dict[str, Any]:
    context = dict(quant)
    lifestyle_facts = _sample_lifestyle_facts(quant, rng)
    persona_blueprint = _sample_persona_blueprint(
        quant,
        lifestyle_facts,
        rng,
        network_role_counts=network_role_counts,
        network_friction_counts=network_friction_counts,
        family_rhythm_counts=family_rhythm_counts,
        family_boundary_counts=family_boundary_counts,
        family_responsibility_counts=family_responsibility_counts,
        support_attitude_counts=support_attitude_counts,
        support_decision_counts=support_decision_counts,
        support_path_counts=support_path_counts,
        support_friction_counts=support_friction_counts,
        support_effect_counts=support_effect_counts,
    )
    context.update(lifestyle_facts)
    context.update(persona_blueprint)
    context = _refresh_hobby_prompt_context(
        context,
        quant,
        rng,
        hobby_item_counts=hobby_item_counts,
        recent_hobby_family_counts=recent_hobby_family_counts,
        blocked_items=blocked_items,
        blocked_families=blocked_families,
        seen_hobby_sets=seen_hobby_sets,
        seen_hobby_family_sets=seen_hobby_family_sets,
    )
    context["family_scope_guard"] = (
        "배우자, 자녀, 동거인 유무를 사실처럼 단정하지 말고 가까운 관계와 일정 조율의 톤으로만 서술"
    )
    context["family_genericity_guard"] = (
        "\"가족과의 시간 조율\", \"생활 책임과 작업 지속성\", \"관계를 유지한다\" 같은 반복 뼈대를 피하고 관계 운영의 방식이 보이게 쓸 것"
    )
    context["hobby_genericity_guard"] = (
        "음악 감상, 책 읽기, 산책 같은 범용 항목은 단독으로 반복하지 말고 구체적 맥락을 붙일 것"
    )
    context["living_genericity_guard"] = (
        "\"주중에는 ... 주말에는 ...\", \"생활 동선을 안정적으로 묶고 있다\", "
        "\"일상의 균형을 유지한다\" 같은 뼈대 문장을 그대로 복제하지 말 것"
    )
    context["network_genericity_guard"] = (
        "\"생태계에서 중심적인 역할\", \"인맥을 넓히고 있다\", "
        "\"요청이 들어오면 연결과 조율을 맡는다\" 같은 고정 뼈대를 피할 것"
    )
    context["support_genericity_guard"] = (
        "\"실질적인 도움을 기대한다\", \"지원 제도는 필요할 때 쓰되...\" 같은 총론 문장을 반복하지 말고 실제 마찰과 효용을 드러낼 것"
    )
    return context


def _quant_only(row: dict[str, Any]) -> dict[str, Any]:
    return {
        field: row[field]
        for field in PAKPersonaQuant.model_fields
        if field in row
    }


def _canonicalize_quant_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    for column in PAK_CORE_UNSUPPORTED_NULLABLE_COLUMNS:
        if column in normalized:
            normalized[column] = None
    return normalized


def _resolve_eval_source_path(base_path: Path, source: str) -> Path:
    source_path = Path(source)
    if source_path.is_absolute():
        return source_path
    candidate = (settings.project_root / source_path).resolve()
    if candidate.exists():
        return candidate
    return (base_path.parent / source_path).resolve()


def _load_quant_rows_from_path(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    info: dict[str, Any] = {"path": str(path)}
    rows: list[dict[str, Any]] = []

    if isinstance(payload, list):
        rows = [_quant_only(dict(item)) for item in payload]
        info["mode"] = "inline_rows"
    elif isinstance(payload, dict):
        info.update({k: v for k, v in payload.items() if k not in {"rows", "entries"}})
        if "rows" in payload:
            rows = [_quant_only(dict(item)) for item in payload["rows"]]
            info["mode"] = "inline_rows"
        elif "source_parquet" in payload and "entries" in payload:
            source_parquet = _resolve_eval_source_path(path, str(payload["source_parquet"]))
            df = pd.read_parquet(source_parquet)
            info["mode"] = "parquet_selector"
            info["source_parquet"] = str(source_parquet)
            for entry in payload["entries"]:
                if isinstance(entry, dict):
                    pak_uuid = str(entry["pak_uuid"])
                else:
                    pak_uuid = str(entry)
                matched = df.loc[df["pak_uuid"] == pak_uuid]
                if matched.empty:
                    raise KeyError(f"pak_uuid not found in eval source parquet: {pak_uuid}")
                rows.append(_quant_only(matched.iloc[0].to_dict()))
        else:
            raise ValueError(
                "quant_rows_path JSON must be a list of quant rows, "
                "an object with 'rows', or an object with 'source_parquet' and 'entries'"
            )
    else:
        raise ValueError("quant_rows_path JSON must be a list or object")

    validated = [
        PAKPersonaQuant.model_validate(_canonicalize_quant_row(row)).model_dump() for row in rows
    ]
    info["n_rows"] = len(validated)
    return validated, info


def _execute_expression(
    spec: ExpressionColumnSpec,
    state: dict[str, Any],
    *,
    rng: random.Random,
) -> Any:
    kind = spec.expression_kind
    if kind == "uuid4":
        return str(uuid.uuid4())
    if kind == "split_sex_age_sex":
        sex, _ = split_sex_age(str(state["sex_age"]))
        return sex
    if kind == "split_sex_age_age_band":
        _, age_band = split_sex_age(str(state["sex_age"]))
        return age_band
    if kind == "alias_province_to_npk":
        return alias_province_to_npk(str(state["province_raw"]))
    if kind == "employment_type_eq_겸업":
        return state["employment_type"] == "겸업"
    if kind == "map_pak_education_to_npk":
        return _sample_npk_education(str(state["education_level_pak"]), rng)
    if kind == "contract_experience_condition":
        if not bool(state["has_contract_experience"]):
            return None
        return bool(state["uses_standard_contract_raw"])
    if kind == "constant":
        return spec.metadata.get("value")
    raise ValueError(f"unknown expression_kind: {kind}")


def _execute_derived_sampler(
    spec: SamplerColumnSpec,
    state: dict[str, Any],
    *,
    rng: random.Random,
    np_rng,
) -> Any:
    operation = spec.operation
    if operation == "sample_age_for_career":
        field = str(state["art_field_primary"])
        return sample_age_in_band_for_career(
            np_rng,
            str(state["age_band"]),
            str(state["career_band"]),
            min_start_age=min_career_start_age(field),
        )
    if operation == "sample_district":
        return _sample_district(str(state["province"]), rng)
    if operation == "sample_occupation":
        return _occupation_from_field(str(state["art_field_primary"]), rng)
    if operation == "sample_household_income":
        return _sample_household_income(str(state["art_field_primary"]), np_rng)
    if operation == "sample_career_years":
        field = str(state["art_field_primary"])
        return sample_career_in_band(
            np_rng,
            str(state["career_band"]),
            max_years=max(int(state["age"]) - min_career_start_age(field), 0),
        )
    raise ValueError(f"unknown derived sampler operation: {operation}")


def build_single_call_prompt(
    quant: dict,
    rng: random.Random,
    dataset_cfg: CompiledPAKDatasetConfig | None = None,
    prompt_context: dict[str, Any] | None = None,
    retry_feedback: list[str] | None = None,
) -> tuple[str, str]:
    """Build the system + user message for a single call."""
    dataset_cfg = dataset_cfg or get_default_pak_core_dataset_config()
    narrative_spec = dataset_cfg.narrative_spec
    field = quant["art_field_primary"]
    seed_word = rng.choice(SEED_POOLS.get(field, SEED_POOLS["기타"]))
    prompt_context = prompt_context or _build_prompt_context(quant, rng)
    field_contracts = {
        "persona": (
            "성격 소개보다 작업 리듬, 생활 태도, 지역 맥락 중 2개를 묶어 한 사람의 분위기가 바로 보이게 쓸 것"
        ),
        "family_persona": (
            "family_contact_style, family_rhythm, family_boundary, family_responsibility 중 최소 2개를 직접 반영하고, 가까운 관계를 어떻게 운영하는지 보이게 쓸 것"
        ),
        "professional_persona": (
            "경력 나열보다 지금 어떤 일을 어떤 방식으로 굴리는지에 집중하고, 운영 디테일 1개와 인과 1개를 포함할 것"
        ),
        "creative_world_persona": (
            "무엇에 끌리고 무엇을 피하는지, 작업의 기준과 긴장을 드러낼 것"
        ),
        "network_persona": (
            "network_scope, network_role, network_friction, network_exchange_mode 중 최소 2개를 직접 반영하고, 누구와 무엇을 주고받는지 구체적으로 드러낼 것"
        ),
        "living_persona": (
            "space_anchor, expense_anchor, recovery_anchor, weekly_rhythm, living_tradeoff 중 최소 3개를 직접 반영하고, 시간·돈·공간을 어디서 타협하는지 구체적으로 드러낼 것"
        ),
        "support_persona": (
            "support_decision, support_path, support_friction, support_effect, support_attitude 중 최소 3개를 직접 반영하고, 신청 판단 흐름과 지원의 실제 효용/거리감을 함께 쓸 것"
        ),
    }

    # Gather the per-category prompts into one
    global_constraints = [
        "=== global_constraints ===",
        "[정합성 앵커]",
        *[
            f"- {narrative_spec.anchor_labels.get(column, column)}: {quant[column]}"
            for column in narrative_spec.anchor_columns
        ],
        "",
        "[persona blueprint]",
        f"- workspace_mode: {prompt_context['workspace_mode']}",
        f"- weekly_rhythm: {prompt_context['weekly_rhythm']}",
        f"- housing_pressure: {prompt_context['housing_pressure']}",
        f"- space_anchor: {prompt_context['space_anchor']}",
        f"- expense_anchor: {prompt_context['expense_anchor']}",
        f"- recovery_anchor: {prompt_context['recovery_anchor']}",
        f"- family_contact_style: {prompt_context['family_contact_style']}",
        f"- family_rhythm: {prompt_context['family_rhythm']}",
        f"- family_boundary: {prompt_context['family_boundary']}",
        f"- family_responsibility: {prompt_context['family_responsibility']}",
        f"- local_routine_hint: {prompt_context['local_routine_hint']}",
        f"- support_need: {prompt_context['support_need']}",
        f"- creative_tension: {prompt_context['creative_tension']}",
        f"- network_exchange_mode: {prompt_context['network_exchange_mode']}",
        f"- network_scope: {prompt_context['network_scope']}",
        f"- network_role: {prompt_context['network_role']}",
        f"- network_friction: {prompt_context['network_friction']}",
        f"- living_tradeoff: {prompt_context['living_tradeoff']}",
        f"- support_attitude: {prompt_context['support_attitude']}",
        f"- support_decision: {prompt_context['support_decision']}",
        f"- support_path: {prompt_context['support_path']}",
        f"- support_friction: {prompt_context['support_friction']}",
        f"- support_effect: {prompt_context['support_effect']}",
        f"- persona_focus: {prompt_context['persona_focus']}",
        f"- family_scope_guard: {prompt_context['family_scope_guard']}",
        "",
        "[취미 계획 앵커]",
        *[f"- {item}" for item in prompt_context["hobby_plan_items"]],
        "",
        "[필드별 역할 계약]",
        *[f"- {key}: {rule}" for key, rule in field_contracts.items()],
        "",
        "[공통 규칙]",
        *[f"- {rule}" for rule in narrative_spec.system_rules],
        *(
            [
                f"- persona 또는 professional_persona에서 직업명 '{quant['occupation']}'을 최소 1회 직접 언급하세요."
            ]
            if quant.get("occupation")
            else []
        ),
        *(
            [
                f"- 활동/거주 중심 지역은 '{quant['province']}'로 유지하세요. 다른 지역명은 영화제, 투어, 외부 기관 맥락이 아닐 때 거주지처럼 쓰지 마세요."
            ]
            if quant.get("province") and quant.get("province") != "기타"
            else []
        ),
        *(
            [
                "- 전업이며 부업 없음: 알바, 아르바이트, 부업, 겸업, 외주 수입, 세션 알바, 강의 수입 같은 보조 일거리나 부수입을 서술하지 마세요."
            ]
            if quant.get("employment_type") == "전업" and not quant.get("has_secondary_job")
            else []
        ),
        *(
            [
                "- 겸업 또는 부업이 있는 경우에만 보조 생계 활동이나 추가 수입원을 언급하세요."
            ]
            if quant.get("employment_type") != "전업" or quant.get("has_secondary_job")
            else []
        ),
        *(
            [
                f"- 이번 persona에서는 다음 취미 표현군을 피하세요: {prompt_context['blocked_hobby_family_text']}"
            ]
            if prompt_context.get("blocked_hobby_family_text")
            else []
        ),
        *(
            [
                f"- 이번 persona에서는 다음 정확한 취미 표현을 피하세요: {prompt_context['blocked_hobby_item_text']}"
            ]
            if prompt_context.get("blocked_hobby_item_text")
            else []
        ),
        "- 배우자, 자녀, 동거인 유무처럼 근거 없는 구조 사실은 단정하지 말고 관계의 거리감이나 시간 조율 방식으로만 서술하세요.",
        "- family_persona에서는 생활/지원 총론을 반복하기보다 연락 빈도, 약속 배치, 생활 책임 분담처럼 관계 운영 방식이 보이게 쓰세요.",
        "- 취미는 위 취미 계획 앵커를 우선 사용하고, 같은 범용 항목을 여러 persona에서 복제한 듯한 표현은 피하세요.",
        "- hobbies_and_interests_list는 취미 계획 앵커 5개만으로 구성하고, 새 generic 항목을 덧붙이지 마세요.",
        "- 유명 랜드마크 이름을 취미 예시로 상투적으로 반복하지 말고 province 수준의 생활 리듬으로만 지역감을 주세요.",
        "- network_persona에서는 '생태계에서 중심적인 역할/위치', '인맥을 넓힌다', '요청이 들어오면 연결과 조율을 맡는다' 같은 고정 표현을 쓰지 마세요.",
        "- support_persona에서는 '실질적인 도움을 기대한다', '지원 제도는 필요할 때 쓰되...' 같은 총론 문장을 반복하지 말고 실제 효용 1개와 마찰 1개를 함께 쓰세요.",
        "- living_persona에서는 '주중에는 ... 주말에는 ...', '생활 동선을 안정적으로 묶고 있다', '일상의 균형을 유지한다' 같은 템플릿 문장을 그대로 복제하지 마세요.",
    ]
    if retry_feedback:
        global_constraints.extend([
            "",
            "[이전 시도에서 반드시 고칠 점]",
            *[f"- {item}" for item in retry_feedback],
        ])

    sections: list[str] = ["\n".join(global_constraints)]
    for cat in narrative_spec.domain_prompt_categories:
        text = render_narrative_prompt(cat, prompt_context, seed_word=seed_word, rng=rng)
        sections.append(f"=== {cat}_persona ===\n{text}")
    for cat in narrative_spec.common_prompt_categories:
        # For _list variants the narrative key stays the same
        out_key = cat
        text = render_narrative_prompt(cat, prompt_context, rng=rng)
        sections.append(f"=== {out_key} ===\n{text}")

    keys = narrative_spec.output_fields
    json_skeleton = json.dumps({key: "..." for key in keys}, ensure_ascii=False, indent=2)

    system_prompt = (
        "당신은 한국 문화예술인 페르소나 narrative 작성자입니다.\n"
        "주어진 정량 변수와 분야별 prompt를 따라 17개 narrative를 생성합니다.\n"
        "occupation, province, employment_type, career_years는 canonical anchor이며 narrative 간 불일치가 없어야 합니다.\n"
        "출력은 다음 키만 포함하는 단일 JSON 객체로만 반환하세요. 다른 텍스트 절대 금지.\n"
        f"키 목록: {keys}\n"
        "특히 마지막 4개 키 creative_world_persona, network_persona, living_persona, support_persona를 절대 생략하지 마세요.\n"
        "각 값은 한국어 narrative 본문(_list 변종은 list 문자열). 클리셰 회피, 3인칭, 실존 인물 비참조."
    )

    sections.append("=== output_contract ===\n" + json_skeleton)
    user_msg = "\n\n".join(sections)
    return system_prompt, user_msg


def _normalize_narrative_obj(
    obj: dict[str, Any],
    dataset_cfg: CompiledPAKDatasetConfig,
) -> dict[str, Any]:
    scalar_aliases = {
        "skills_and_experteise": "skills_and_expertise",
        "skills_and_experteise_list": "skills_and_expertise_list",
    }
    for alias, canonical in scalar_aliases.items():
        if alias in obj and canonical not in obj:
            obj[canonical] = obj.pop(alias)
    for field in dataset_cfg.narrative_spec.output_fields:
        if not field.endswith("_persona") or field in obj:
            continue
        alias = field.removesuffix("_persona")
        if alias in obj:
            obj[field] = obj.pop(alias)
    for field in dataset_cfg.narrative_spec.output_fields:
        if not field.endswith("_list"):
            continue
        value = obj.get(field)
        if isinstance(value, list):
            obj[field] = repr([str(item).strip() for item in value if str(item).strip()])
    return obj


def _missing_narrative_fields(
    obj: dict[str, Any],
    dataset_cfg: CompiledPAKDatasetConfig,
) -> list[str]:
    return [field for field in dataset_cfg.narrative_spec.output_fields if field not in obj]


def parse_narrative_response(
    text: str,
    dataset_cfg: CompiledPAKDatasetConfig | None = None,
) -> dict[str, str]:
    """LLM response -> narrative dict. Raises ValueError on parse/schema failure."""
    dataset_cfg = dataset_cfg or get_default_pak_core_dataset_config()
    obj = parse_json_response(text)
    if not isinstance(obj, dict):
        raise ValueError(f"expected dict, got {type(obj).__name__}")
    obj = _normalize_narrative_obj(obj, dataset_cfg)
    try:
        validated = dataset_cfg.narrative_spec.output_model.model_validate(obj)
    except ValidationError as exc:
        raise ValueError(f"invalid narrative schema: {exc}") from exc
    return validated.model_dump()


def _build_missing_field_repair_prompt(
    *,
    quant: dict[str, Any],
    prompt_context: dict[str, Any],
    partial_obj: dict[str, Any],
    missing_fields: list[str],
) -> tuple[str, str]:
    system_prompt = (
        "당신은 불완전한 JSON 응답을 수리하는 도우미입니다.\n"
        "반드시 누락된 키만 포함하는 단일 JSON 객체만 반환하세요.\n"
        "기존 키를 반복하지 말고, 설명 문장이나 코드블록도 절대 추가하지 마세요."
    )
    user_prompt = "\n".join(
        [
            "[canonical anchors]",
            f"- occupation: {quant['occupation']}",
            f"- sex: {quant['sex']}",
            f"- age: {quant['age']}",
            f"- age_band: {quant['age_band']}",
            f"- province: {quant['province']}",
            f"- art_field_primary: {quant['art_field_primary']}",
            f"- career_years: {quant['career_years']}",
            f"- employment_type: {quant['employment_type']}",
            "",
            "[persona blueprint]",
            f"- persona_focus: {prompt_context['persona_focus']}",
            f"- creative_tension: {prompt_context['creative_tension']}",
            f"- family_contact_style: {prompt_context['family_contact_style']}",
            f"- family_rhythm: {prompt_context['family_rhythm']}",
            f"- family_boundary: {prompt_context['family_boundary']}",
            f"- family_responsibility: {prompt_context['family_responsibility']}",
            f"- network_exchange_mode: {prompt_context['network_exchange_mode']}",
            f"- network_scope: {prompt_context['network_scope']}",
            f"- network_role: {prompt_context['network_role']}",
            f"- network_friction: {prompt_context['network_friction']}",
            f"- living_tradeoff: {prompt_context['living_tradeoff']}",
            f"- space_anchor: {prompt_context['space_anchor']}",
            f"- expense_anchor: {prompt_context['expense_anchor']}",
            f"- recovery_anchor: {prompt_context['recovery_anchor']}",
            f"- support_attitude: {prompt_context['support_attitude']}",
            f"- support_decision: {prompt_context['support_decision']}",
            f"- support_path: {prompt_context['support_path']}",
            f"- support_friction: {prompt_context['support_friction']}",
            f"- support_effect: {prompt_context['support_effect']}",
            "",
            f"[missing keys] {missing_fields}",
            "",
            "[existing partial json]",
            json.dumps(partial_obj, ensure_ascii=False, indent=2),
            "",
            "[instruction]",
            "위 partial json과 정합적으로 이어지는 누락 키만 생성하세요.",
        ]
    )
    return system_prompt, user_prompt


def _target_keys_for_issues(
    issues: list[str],
    dataset_cfg: CompiledPAKDatasetConfig,
) -> list[str]:
    if not issues:
        return list(dataset_cfg.narrative_spec.output_fields)

    mapping = {
        "AGE_MISMATCH": ["persona", "professional_persona", "living_persona"],
        "CAREER_MISMATCH_NEW": ["persona", "professional_persona", "living_persona"],
        "CAREER_MISMATCH_VETERAN": ["persona", "professional_persona", "living_persona"],
        "EMPLOYMENT_DURATION_CONFLATION": ["persona", "professional_persona", "living_persona"],
        "OCCUPATION_MISMATCH": [
            "persona",
            "professional_persona",
            "living_persona",
            "network_persona",
        ],
        "EMPLOYMENT_MISMATCH": ["persona", "professional_persona", "living_persona"],
        "REGION_MISMATCH": ["persona", "professional_persona", "living_persona"],
        "LANDMARK_REGION_MISMATCH": ["persona", "professional_persona", "living_persona"],
        "HOBBY_GENERICITY": ["hobbies_and_interests", "hobbies_and_interests_list"],
        "HOBBY_DUPLICATE": ["hobbies_and_interests", "hobbies_and_interests_list"],
        "HOBBY_NEAR_DUPLICATE": ["hobbies_and_interests", "hobbies_and_interests_list"],
        "HOBBY_PLAN_DRIFT": ["hobbies_and_interests", "hobbies_and_interests_list"],
        "HOBBY_QUOTA_EXCEEDED": ["hobbies_and_interests", "hobbies_and_interests_list"],
        "HOBBY_LIST_ATOMICITY": ["hobbies_and_interests", "hobbies_and_interests_list"],
        "FAMILY_GENERICITY": ["family_persona"],
        "NETWORK_GENERICITY": ["network_persona"],
        "SUPPORT_GENERICITY": ["support_persona"],
        "LIVING_GENERICITY": ["living_persona"],
        "FAMILY_LIVING_COHERENCE": ["family_persona", "living_persona"],
    }
    keys: list[str] = []
    for issue in issues:
        code = issue.split(":", 1)[0].strip()
        keys.extend(mapping.get(code, []))
    if not keys:
        return list(dataset_cfg.narrative_spec.output_fields)
    ordered = list(dict.fromkeys(keys))
    return [key for key in ordered if key in dataset_cfg.narrative_spec.output_fields]


def _build_validation_revision_prompt(
    *,
    quant: dict[str, Any],
    prompt_context: dict[str, Any],
    narratives: dict[str, str],
    issues: list[str],
    dataset_cfg: CompiledPAKDatasetConfig,
    target_keys: list[str],
) -> tuple[str, str]:
    system_prompt = (
        "당신은 품질 검수 결과를 반영해 JSON 초안을 수정하는 도우미입니다.\n"
        "반드시 단일 JSON 객체만 반환하세요.\n"
        "가능하면 문제와 직접 관련된 키만 반환하고, 정량 앵커는 바꾸지 마세요."
    )
    user_prompt = "\n".join(
        [
            "[canonical anchors]",
            f"- occupation: {quant['occupation']}",
            f"- province: {quant['province']}",
            f"- art_field_primary: {quant['art_field_primary']}",
            f"- career_years: {quant['career_years']}",
            f"- employment_type: {quant['employment_type']}",
            f"- has_secondary_job: {quant['has_secondary_job']}",
            "",
            "[persona blueprint]",
            f"- persona_focus: {prompt_context['persona_focus']}",
            f"- creative_tension: {prompt_context['creative_tension']}",
            f"- family_contact_style: {prompt_context['family_contact_style']}",
            f"- family_rhythm: {prompt_context['family_rhythm']}",
            f"- family_boundary: {prompt_context['family_boundary']}",
            f"- family_responsibility: {prompt_context['family_responsibility']}",
            f"- network_exchange_mode: {prompt_context['network_exchange_mode']}",
            f"- network_scope: {prompt_context['network_scope']}",
            f"- network_role: {prompt_context['network_role']}",
            f"- network_friction: {prompt_context['network_friction']}",
            f"- living_tradeoff: {prompt_context['living_tradeoff']}",
            f"- space_anchor: {prompt_context['space_anchor']}",
            f"- expense_anchor: {prompt_context['expense_anchor']}",
            f"- recovery_anchor: {prompt_context['recovery_anchor']}",
            f"- support_attitude: {prompt_context['support_attitude']}",
            f"- support_decision: {prompt_context['support_decision']}",
            f"- support_path: {prompt_context['support_path']}",
            f"- support_friction: {prompt_context['support_friction']}",
            f"- support_effect: {prompt_context['support_effect']}",
            "",
            "[hobby guidance]",
            f"- hobby_plan_text: {prompt_context.get('hobby_plan_text', '')}",
            *(
                [f"- avoid_hobby_families: {prompt_context['blocked_hobby_family_text']}"]
                if prompt_context.get("blocked_hobby_family_text")
                else []
            ),
            *(
                [f"- avoid_hobby_items: {prompt_context['blocked_hobby_item_text']}"]
                if prompt_context.get("blocked_hobby_item_text")
                else []
            ),
            "",
            "[living guidance]",
            f"- workspace_mode: {prompt_context['workspace_mode']}",
            f"- weekly_rhythm: {prompt_context['weekly_rhythm']}",
            f"- housing_pressure: {prompt_context['housing_pressure']}",
            f"- local_routine_hint: {prompt_context['local_routine_hint']}",
            f"- living_genericity_guard: {prompt_context['living_genericity_guard']}",
            "",
            "[family guidance]",
            f"- family_contact_style: {prompt_context['family_contact_style']}",
            f"- family_rhythm: {prompt_context['family_rhythm']}",
            f"- family_boundary: {prompt_context['family_boundary']}",
            f"- family_responsibility: {prompt_context['family_responsibility']}",
            f"- family_genericity_guard: {prompt_context['family_genericity_guard']}",
            "",
            "[network guidance]",
            f"- network_scope: {prompt_context['network_scope']}",
            f"- network_role: {prompt_context['network_role']}",
            f"- network_friction: {prompt_context['network_friction']}",
            f"- network_genericity_guard: {prompt_context['network_genericity_guard']}",
            "",
            "[support guidance]",
            f"- support_decision: {prompt_context['support_decision']}",
            f"- support_path: {prompt_context['support_path']}",
            f"- support_friction: {prompt_context['support_friction']}",
            f"- support_effect: {prompt_context['support_effect']}",
            f"- support_genericity_guard: {prompt_context['support_genericity_guard']}",
            "",
            "[issues to fix]",
            *[f"- {item}" for item in issues],
            "",
            f"[target keys] {target_keys}",
            "",
            "[current draft json]",
            json.dumps(narratives, ensure_ascii=False, indent=2),
            "",
            "[instruction]",
            "위 문제를 해결하는 target keys만 다시 작성하세요.",
            "target keys 바깥의 값은 반환하지 않아도 됩니다.",
            "직업, 지역, 경력 단계, 전업/겸업 정합성을 우선적으로 바로잡으세요.",
            "EMPLOYMENT_DURATION_CONFLATION이 있으면 'N년 전업/겸업', '전업/겸업으로 N년' 문형을 쓰지 말고, '활동 경력은 N년이며 현재는 전업/겸업 상태'처럼 두 사실을 분리하세요.",
            "현재 persona blueprint와 어긋나지 않게 고치고, 필드별 역할이 겹치지 않게 쓰세요.",
            "배우자, 자녀, 동거인 유무처럼 근거 없는 구조 사실은 새로 단정하지 마세요.",
        ]
    )
    return system_prompt, user_prompt


def _repair_missing_narrative_fields(
    *,
    client: Any,
    cfg: GenerateConfig,
    quant: dict[str, Any],
    prompt_context: dict[str, Any],
    partial_obj: dict[str, Any],
    missing_fields: list[str],
    dataset_cfg: CompiledPAKDatasetConfig,
) -> tuple[dict[str, Any], int, int]:
    system, user = _build_missing_field_repair_prompt(
        quant=quant,
        prompt_context=prompt_context,
        partial_obj=partial_obj,
        missing_fields=missing_fields,
    )
    resp = client.chat(
        model=cfg.model,
        system=system,
        user=user,
        max_tokens=max(1200, cfg.max_tokens // 2),
        temperature=min(cfg.temperature, 0.5),
        response_format={"type": "json_object"},
    )
    obj = parse_json_response(resp.text)
    if not isinstance(obj, dict):
        raise ValueError(f"repair expected dict, got {type(obj).__name__}")
    obj = _normalize_narrative_obj(obj, dataset_cfg)
    unexpected = sorted(set(obj) - set(missing_fields))
    if unexpected:
        raise ValueError(f"repair returned unexpected keys: {unexpected}")
    return obj, resp.usage.input_tokens, resp.usage.output_tokens


def _revise_narrative_after_validation(
    *,
    client: Any,
    cfg: Any,
    quant: dict[str, Any],
    prompt_context: dict[str, Any],
    narratives: dict[str, str],
    issues: list[str],
    dataset_cfg: CompiledPAKDatasetConfig,
) -> tuple[dict[str, str], int, int]:
    target_keys = _target_keys_for_issues(issues, dataset_cfg)
    system, user = _build_validation_revision_prompt(
        quant=quant,
        prompt_context=prompt_context,
        narratives=narratives,
        issues=issues,
        dataset_cfg=dataset_cfg,
        target_keys=target_keys,
    )
    resp = client.chat(
        model=cfg.model,
        system=system,
        user=user,
        max_tokens=cfg.max_tokens,
        temperature=min(_retry_temperature(cfg, 1), 0.3),
        response_format={"type": "json_object"},
    )
    obj = parse_json_response(resp.text)
    if not isinstance(obj, dict):
        raise ValueError(f"validation revision expected dict, got {type(obj).__name__}")
    obj = _normalize_narrative_obj(obj, dataset_cfg)
    all_keys = set(dataset_cfg.narrative_spec.output_fields)
    returned_keys = set(obj)
    if not returned_keys.issubset(all_keys):
        unexpected = sorted(returned_keys - all_keys)
        logger.warning("validation revision ignored unexpected keys: %s", unexpected)
        obj = {key: value for key, value in obj.items() if key in all_keys}
        returned_keys = set(obj)
    if not returned_keys:
        raise ValueError("validation revision returned no narrative keys")

    merged = dict(narratives)
    merged.update(obj)
    validated = dataset_cfg.narrative_spec.output_model.model_validate(merged)
    return validated.model_dump(), resp.usage.input_tokens, resp.usage.output_tokens


def _complete_narrative_response(
    text: str,
    *,
    quant: dict[str, Any],
    prompt_context: dict[str, Any],
    client: Any,
    cfg: GenerateConfig,
    dataset_cfg: CompiledPAKDatasetConfig,
) -> tuple[dict[str, str], dict[str, int]]:
    obj = parse_json_response(text)
    if not isinstance(obj, dict):
        raise ValueError(f"expected dict, got {type(obj).__name__}")
    obj = _normalize_narrative_obj(obj, dataset_cfg)
    repair_usage = {
        "repair_calls": 0,
        "repair_input_tokens": 0,
        "repair_output_tokens": 0,
    }
    missing_fields = _missing_narrative_fields(obj, dataset_cfg)
    if missing_fields:
        repaired, repair_in, repair_out = _repair_missing_narrative_fields(
            client=client,
            cfg=cfg,
            quant=quant,
            prompt_context=prompt_context,
            partial_obj=obj,
            missing_fields=missing_fields,
            dataset_cfg=dataset_cfg,
        )
        repair_usage["repair_calls"] += 1
        repair_usage["repair_input_tokens"] += repair_in
        repair_usage["repair_output_tokens"] += repair_out
        obj.update(repaired)

    try:
        validated = dataset_cfg.narrative_spec.output_model.model_validate(obj)
    except ValidationError as exc:
        raise ValueError(f"invalid narrative schema: {exc}") from exc
    return validated.model_dump(), repair_usage


# ----------------------------------------------------------------------------
# Single / batch generation
# ----------------------------------------------------------------------------


@dataclass
class GenerateConfig:
    n: int = 100
    seed: int = 20260502
    model: str = field(default_factory=lambda: settings.llm_default_model)
    base_url: str = field(default_factory=lambda: settings.llm_base_url)
    api_key: str = field(default_factory=lambda: settings.llm_api_key)
    temperature: float = 0.8
    max_tokens: int = 4096
    max_retries: int = 2
    output_dir: Path = field(default_factory=lambda: settings.synthetic_dir / "pilot")
    quant_rows_path: Path | None = None
    checkpoint_every: int = 50
    skip_validation: bool = False
    quality_profile: str = "balanced"
    fail_on_warnings: bool = False
    blocking_warning_codes: tuple[str, ...] = ()
    max_warning_revisions: int = 0
    enforce_hobby_plan_alignment: bool | None = None
    enforce_living_persona_specificity: bool | None = None
    enforce_family_persona_specificity: bool | None = None
    enforce_network_persona_specificity: bool | None = None
    enforce_support_persona_specificity: bool | None = None
    hobby_exact_atom_cap_per_batch: int | None = None
    hobby_family_cap_per_batch: int | None = None
    coerce_hobbies_to_plan: bool | None = None
    retry_temperature_decay: float = 0.0
    min_temperature: float = 0.35
    max_concurrent: int = field(default_factory=lambda: max(1, int(settings.llm_max_concurrent)))

    def __post_init__(self) -> None:
        alignment_unspecified = self.enforce_hobby_plan_alignment is None
        living_unspecified = self.enforce_living_persona_specificity is None
        family_unspecified = self.enforce_family_persona_specificity is None
        network_unspecified = self.enforce_network_persona_specificity is None
        support_unspecified = self.enforce_support_persona_specificity is None
        coerce_hobbies_unspecified = self.coerce_hobbies_to_plan is None
        if self.quality_profile == "max":
            self.temperature = min(self.temperature, 0.45)
            self.max_retries = max(self.max_retries, 4)
            self.fail_on_warnings = True
            if not self.blocking_warning_codes:
                self.blocking_warning_codes = _STRICT_WARNING_CODES_DEFAULT
            self.max_warning_revisions = max(self.max_warning_revisions, 2)
            if alignment_unspecified:
                self.enforce_hobby_plan_alignment = True
            if living_unspecified:
                self.enforce_living_persona_specificity = True
            if family_unspecified:
                self.enforce_family_persona_specificity = True
            if network_unspecified:
                self.enforce_network_persona_specificity = True
            if support_unspecified:
                self.enforce_support_persona_specificity = True
            if self.hobby_exact_atom_cap_per_batch is None:
                self.hobby_exact_atom_cap_per_batch = max(4, (max(self.n, 1) + 3) // 4)
            if coerce_hobbies_unspecified:
                self.coerce_hobbies_to_plan = True
            self.retry_temperature_decay = max(self.retry_temperature_decay, 0.08)
            self.min_temperature = min(self.min_temperature, 0.2)
        if self.enforce_hobby_plan_alignment is None:
            self.enforce_hobby_plan_alignment = False
        if self.enforce_living_persona_specificity is None:
            self.enforce_living_persona_specificity = False
        if self.enforce_family_persona_specificity is None:
            self.enforce_family_persona_specificity = False
        if self.enforce_network_persona_specificity is None:
            self.enforce_network_persona_specificity = False
        if self.enforce_support_persona_specificity is None:
            self.enforce_support_persona_specificity = False
        if self.coerce_hobbies_to_plan is None:
            self.coerce_hobbies_to_plan = False


def _validate_quant(quant: dict) -> dict | None:
    try:
        # Validate only PAKPersonaQuant, with narratives left empty
        validated = PAKPersonaQuant.model_validate(_canonicalize_quant_row(quant))
    except ValidationError as exc:
        logger.warning("quant validation failed: %s", exc)
        return None
    return validated.model_dump()


def _narratives_only(
    row: dict[str, Any],
    dataset_cfg: CompiledPAKDatasetConfig | None = None,
) -> dict[str, str]:
    dataset_cfg = dataset_cfg or get_default_pak_core_dataset_config()
    narrative_keys = set(dataset_cfg.narrative_spec.output_fields)
    return {
        key: str(value)
        for key, value in row.items()
        if key in narrative_keys and isinstance(value, str)
    }


def _retry_temperature(cfg: Any, attempt: int) -> float:
    base = float(getattr(cfg, "temperature", 0.8))
    decay = float(getattr(cfg, "retry_temperature_decay", 0.0) or 0.0)
    minimum = float(getattr(cfg, "min_temperature", base))
    return max(minimum, base - attempt * decay)


def _feedback_lines_from_validation(result: Any, *, include_warnings: bool = True) -> list[str]:
    lines: list[str] = []
    for issue in getattr(result, "consistency_issues", []):
        if issue.severity == "error" or (include_warnings and issue.severity == "warning"):
            lines.append(f"{issue.code}: {issue.message}")
    for hit in getattr(result, "cliche_hits", []):
        lines.append(f"CLICHE: {hit.label}")
    return lines[:8]


def _blocking_warning_issues(result: Any, cfg: Any) -> list[Any]:
    if not getattr(cfg, "fail_on_warnings", False):
        return []
    issues = [issue for issue in getattr(result, "consistency_issues", []) if issue.severity == "warning"]
    blocking_codes = tuple(getattr(cfg, "blocking_warning_codes", ()) or ())
    if not blocking_codes:
        return issues
    blocking_set = set(blocking_codes)
    return [issue for issue in issues if issue.code in blocking_set]


def _normalized_hobby_set(value: Any) -> tuple[str, ...]:
    return tuple(sorted(set(_parse_hobby_items(value))))


def _normalized_hobby_family_set(value: Any) -> tuple[str, ...]:
    items = _parse_hobby_items(value)
    return tuple(sorted({_canonicalize_hobby_atom(item) for item in items if item}))


def _append_recent_hobby_families(
    recent_hobby_families: deque[str],
    recent_hobby_family_counts: Counter[str],
    families: tuple[str, ...],
    *,
    limit: int = 24,
) -> None:
    for family in families:
        if len(recent_hobby_families) >= limit:
            removed = recent_hobby_families.popleft()
            recent_hobby_family_counts[removed] -= 1
            if recent_hobby_family_counts[removed] <= 0:
                del recent_hobby_family_counts[removed]
        recent_hobby_families.append(family)
        recent_hobby_family_counts[family] += 1


def _apply_blocked_hobby_families(
    prompt_context: dict[str, Any],
    blocked_families: set[str] | None,
) -> dict[str, Any]:
    existing = set(prompt_context.get("blocked_hobby_families", []) or [])
    blocked = sorted(existing | set(blocked_families or set()))
    updated = dict(prompt_context)
    updated["blocked_hobby_families"] = blocked
    updated["blocked_hobby_family_text"] = ", ".join(blocked)
    return updated


def _apply_blocked_hobby_items(
    prompt_context: dict[str, Any],
    blocked_items: set[str] | None,
) -> dict[str, Any]:
    existing = set(prompt_context.get("blocked_hobby_items", []) or [])
    blocked = sorted(existing | set(blocked_items or set()))
    updated = dict(prompt_context)
    updated["blocked_hobby_items"] = blocked
    updated["blocked_hobby_item_text"] = ", ".join(blocked)
    return updated


def _blocked_hobby_items_from_counts(
    hobby_item_counts: Counter[str] | None,
    *,
    cap: int | None,
) -> set[str]:
    if hobby_item_counts is None or cap is None or cap <= 0:
        return set()
    return {item for item, count in hobby_item_counts.items() if count >= cap}


def _blocked_hobby_families_from_counts(
    hobby_family_counts: Counter[str] | None,
    *,
    cap: int | None,
) -> set[str]:
    if hobby_family_counts is None or cap is None or cap <= 0:
        return set()
    return {family for family, count in hobby_family_counts.items() if count >= cap}


def _planned_hobby_list_value(prompt_context: dict[str, Any]) -> str:
    items = [str(item).strip() for item in prompt_context.get("hobby_plan_items", []) if str(item).strip()]
    return repr(items[:5])


def _planned_hobby_text(prompt_context: dict[str, Any]) -> str:
    items = [str(item).strip() for item in prompt_context.get("hobby_plan_items", []) if str(item).strip()]
    if not items:
        return ""
    if len(items) == 1:
        body = items[0]
    else:
        body = ", ".join(items[:-1]) + f", 그리고 {items[-1]}"
    return (
        f"취미와 관심사는 {body}를 중심으로 이어진다. "
        "각 활동은 작업 밖에서 몸을 풀고 생각을 정리하는 별도 루틴으로 자리 잡아, "
        "한 주의 리듬을 단조롭지 않게 나누는 역할을 한다."
    )


def _coerce_hobbies_to_plan(
    row: dict[str, Any],
    prompt_context: dict[str, Any],
) -> dict[str, Any]:
    updated = dict(row)
    list_value = _planned_hobby_list_value(prompt_context)
    updated["hobbies_and_interests_list"] = list_value
    text_value = _planned_hobby_text(prompt_context)
    if text_value:
        updated["hobbies_and_interests"] = text_value
    return updated


def _coerce_hobby_list_to_plan(
    row: dict[str, Any],
    prompt_context: dict[str, Any],
) -> dict[str, Any]:
    return _coerce_hobbies_to_plan(row, prompt_context)


def _hobby_plan_drift_issues(
    *,
    hobby_family_set: tuple[str, ...],
    prompt_context: dict[str, Any],
) -> tuple[list[str], set[str]] | None:
    planned_families = set(prompt_context.get("hobby_plan_families", []))
    current_families = set(hobby_family_set)
    if not planned_families or not current_families:
        return None

    overlap = current_families & planned_families
    if len(overlap) >= 3:
        return None

    off_plan = current_families - planned_families
    issues = [
        "HOBBY_PLAN_DRIFT: hobbies_and_interests_list가 planned hobby anchors를 충분히 반영하지 못했습니다.",
        f"HOBBY_PLAN_DRIFT: 현재 취미 표현군 중 계획 밖 항목은 {', '.join(sorted(off_plan or current_families))} 입니다.",
        "HOBBY_PLAN_DRIFT: hobby_plan_text에서 최소 3개 이상의 취미 표현군을 직접 반영해 다시 작성하세요.",
    ]
    return issues, set(off_plan or current_families)


def _hobby_duplication_issues(
    *,
    hobby_set: tuple[str, ...],
    hobby_family_set: tuple[str, ...],
    seen_hobby_sets: set[tuple[str, ...]] | None,
    seen_hobby_family_sets: set[tuple[str, ...]] | None,
) -> tuple[list[str], set[str], str] | None:
    blocked_families = set(hobby_family_set)
    if hobby_set and seen_hobby_sets is not None and hobby_set in seen_hobby_sets:
        return (
            [
                "HOBBY_DUPLICATE: hobbies_and_interests_list가 기존 persona와 exact duplicate입니다.",
                f"HOBBY_DUPLICATE: 피해야 할 취미 표현군은 {', '.join(sorted(blocked_families))} 입니다.",
                "HOBBY_DUPLICATE: 새로운 취미 계획 앵커를 사용해 더 구체적이고 다른 조합으로 다시 작성하세요.",
            ],
            blocked_families,
            f"duplicate hobby set: {hobby_set}",
        )
    if hobby_family_set and seen_hobby_family_sets is not None and hobby_family_set in seen_hobby_family_sets:
        return (
            [
                "HOBBY_NEAR_DUPLICATE: 취미 표현이 기존 persona와 의미상 너무 비슷합니다.",
                f"HOBBY_NEAR_DUPLICATE: 피해야 할 취미 표현군은 {', '.join(sorted(blocked_families))} 입니다.",
                "HOBBY_NEAR_DUPLICATE: 같은 생활 리듬처럼 읽히지 않게 취미 문장과 list를 새로 작성하세요.",
            ],
            blocked_families,
            f"near-duplicate hobby family set: {hobby_family_set}",
        )
    return None


def _hobby_quota_issues(
    *,
    hobby_items: tuple[str, ...],
    hobby_family_set: tuple[str, ...],
    hobby_item_counts: Counter[str] | None,
    hobby_family_counts: Counter[str] | None,
    exact_cap: int | None,
    family_cap: int | None,
) -> tuple[list[str], set[str], set[str], str] | None:
    blocked_items = {
        item
        for item in hobby_items
        if hobby_item_counts is not None and exact_cap is not None and exact_cap > 0
        and hobby_item_counts.get(item, 0) >= exact_cap
    }
    blocked_families = {
        family
        for family in hobby_family_set
        if hobby_family_counts is not None and family_cap is not None and family_cap > 0
        and hobby_family_counts.get(family, 0) >= family_cap
    }
    if not blocked_items and not blocked_families:
        return None

    issues = [
        "HOBBY_QUOTA_EXCEEDED: 이번 batch에서 이미 많이 나온 취미 표현을 다시 사용했습니다.",
        "HOBBY_QUOTA_EXCEEDED: hobby_plan_text의 5개 앵커를 모두 유지하되, 과다 반복된 취미는 다른 planner anchor로 바꿔 다시 작성하세요.",
        "HOBBY_QUOTA_EXCEEDED: hobbies_and_interests_list는 planner anchor 5개만 포함하고 새 generic filler는 추가하지 마세요.",
    ]
    if blocked_items:
        issues.append(
            f"HOBBY_QUOTA_EXCEEDED: 피해야 할 정확한 취미 표현은 {', '.join(sorted(blocked_items))} 입니다."
        )
    if blocked_families:
        issues.append(
            f"HOBBY_QUOTA_EXCEEDED: 피해야 할 취미 표현군은 {', '.join(sorted(blocked_families))} 입니다."
        )
    return (
        issues,
        blocked_items,
        blocked_families,
        "hobby quota exceeded",
    )


def _hobby_list_atomicity_issues(
    *,
    hobby_items: tuple[str, ...],
    prompt_context: dict[str, Any],
) -> tuple[list[str], list[str]] | None:
    if len(hobby_items) != 5:
        issues = [
            "HOBBY_LIST_ATOMICITY: hobbies_and_interests_list는 planner anchor 기준 정확히 5개 항목이어야 합니다.",
            f"HOBBY_LIST_ATOMICITY: 현재 항목 수는 {len(hobby_items)}개입니다.",
            "HOBBY_LIST_ATOMICITY: hobby_plan_text의 5개 앵커를 각각 한 항목씩 분리해 다시 작성하세요.",
        ]
        return issues, list(hobby_items)

    flagged: list[str] = []
    for item in hobby_items:
        for _, pattern in _HOBBY_NON_ATOMIC_PATTERNS:
            if pattern.search(item):
                flagged.append(item)
                break
    if not flagged:
        return None

    issues = [
        "HOBBY_LIST_ATOMICITY: hobbies_and_interests_list 일부 항목이 한 항목 안에 둘 이상의 취미를 묶고 있습니다.",
        f"HOBBY_LIST_ATOMICITY: 분리해야 할 항목은 {', '.join(flagged)} 입니다.",
        "HOBBY_LIST_ATOMICITY: 각 항목은 정확히 한 활동/관심만 담고, 'A와 B', 'A 및 B', 'A/B' 같은 연결을 쓰지 마세요.",
        "HOBBY_LIST_ATOMICITY: hobby_plan_text의 5개 planner anchor를 각각 독립된 항목으로 다시 쓰세요.",
    ]
    return issues, flagged


def _living_genericity_issues(
    *,
    living_text: str,
    prompt_context: dict[str, Any],
) -> list[str] | None:
    matched = [
        label for label, pattern in _LIVING_GENERIC_PATTERNS if pattern.search(living_text or "")
    ]
    if len(matched) < 3:
        return None

    return [
        "LIVING_GENERICITY: living_persona가 반복적인 생활 서술 템플릿에 치우쳐 있습니다.",
        f"LIVING_GENERICITY: 다음 템플릿 표현을 피하세요: {', '.join(matched)}.",
        (
            "LIVING_GENERICITY: workspace_mode, weekly_rhythm, housing_pressure, "
            "living_tradeoff, space_anchor, expense_anchor, recovery_anchor, local_routine_hint "
            "중 최소 4개를 직접 반영해 living_persona만 다시 작성하세요."
        ),
        (
            "LIVING_GENERICITY: 추상적인 균형론 대신 실제 시간 운영 1개, 비용/공간 압박 1개, "
            "회복 방식 1개가 보이게 쓰세요."
        ),
        f"LIVING_GENERICITY: 이번 persona의 공간 앵커는 '{prompt_context['space_anchor']}' 입니다.",
        f"LIVING_GENERICITY: 이번 persona의 비용 앵커는 '{prompt_context['expense_anchor']}' 입니다.",
        f"LIVING_GENERICITY: 이번 persona의 회복 앵커는 '{prompt_context['recovery_anchor']}' 입니다.",
    ]


def _family_genericity_issues(
    *,
    family_text: str,
    prompt_context: dict[str, Any],
) -> list[str] | None:
    matched = [
        label for label, pattern in _FAMILY_GENERIC_PATTERNS if pattern.search(family_text or "")
    ]
    if len(matched) < 2:
        return None

    return [
        "FAMILY_GENERICITY: family_persona가 생활/가족 일반론에 머물고 관계 운영 방식이 약합니다.",
        f"FAMILY_GENERICITY: 다음 표현을 피하세요: {', '.join(matched)}.",
        (
            "FAMILY_GENERICITY: family_contact_style, family_rhythm, family_boundary, "
            "family_responsibility를 반영해 연락 빈도, 약속 배치, 생활 책임 조율이 직접 보이게 다시 쓰세요."
        ),
        f"FAMILY_GENERICITY: 이번 persona의 가족 리듬 앵커는 '{prompt_context['family_rhythm']}' 입니다.",
        f"FAMILY_GENERICITY: 이번 persona의 관계 경계 앵커는 '{prompt_context['family_boundary']}' 입니다.",
        f"FAMILY_GENERICITY: 이번 persona의 생활 책임 앵커는 '{prompt_context['family_responsibility']}' 입니다.",
    ]


def _network_genericity_issues(
    *,
    network_text: str,
    prompt_context: dict[str, Any],
) -> list[str] | None:
    matched = [
        label for label, pattern in _NETWORK_GENERIC_PATTERNS if pattern.search(network_text or "")
    ]
    if not matched:
        return None

    return [
        "NETWORK_GENERICITY: network_persona가 과장된 생태계 위치 설명이나 일반론에 치우쳐 있습니다.",
        f"NETWORK_GENERICITY: 다음 표현을 피하세요: {', '.join(matched)}.",
        (
            "NETWORK_GENERICITY: network_scope, network_role, network_friction, "
            "network_exchange_mode를 반영해 누구와 무엇을 주고받는지 다시 쓰세요."
        ),
        f"NETWORK_GENERICITY: 이번 persona의 네트워크 범위 앵커는 '{prompt_context['network_scope']}' 입니다.",
        f"NETWORK_GENERICITY: 이번 persona의 네트워크 역할 앵커는 '{prompt_context['network_role']}' 입니다.",
        f"NETWORK_GENERICITY: 이번 persona의 네트워크 마찰 앵커는 '{prompt_context['network_friction']}' 입니다.",
    ]


def _support_genericity_issues(
    *,
    support_text: str,
    prompt_context: dict[str, Any],
) -> list[str] | None:
    matched = [
        label for label, pattern in _SUPPORT_GENERIC_PATTERNS if pattern.search(support_text or "")
    ]
    if not matched:
        return None

    return [
        "SUPPORT_GENERICITY: support_persona가 지원 일반론에 머물고 실제 효용/마찰이 약합니다.",
        f"SUPPORT_GENERICITY: 다음 표현을 피하세요: {', '.join(matched)}.",
        (
            "SUPPORT_GENERICITY: support_decision, support_path, support_friction, "
            "support_effect, support_attitude를 반영해 신청 판단 흐름과 수혜/미신청 맥락이 함께 보이게 다시 쓰세요."
        ),
        f"SUPPORT_GENERICITY: 이번 persona의 지원 판단 앵커는 '{prompt_context.get('support_decision', '')}' 입니다.",
        f"SUPPORT_GENERICITY: 이번 persona의 지원 경로 앵커는 '{prompt_context.get('support_path', '')}' 입니다.",
        f"SUPPORT_GENERICITY: 이번 persona의 지원 마찰 앵커는 '{prompt_context.get('support_friction', '')}' 입니다.",
        f"SUPPORT_GENERICITY: 이번 persona의 지원 효용 앵커는 '{prompt_context.get('support_effect', '')}' 입니다.",
    ]


def _format_validation_error_codes(result) -> str:
    parts: list[str] = []
    if result.consistency_issues:
        parts.append(
            "consistency="
            + ",".join(sorted({issue.code for issue in result.consistency_issues if issue.severity == "error"}))
        )
    if result.cliche_hits:
        parts.append("cliches=" + ",".join(sorted({hit.label for hit in result.cliche_hits})))
    return "; ".join(parts) if parts else "validation_failed"


def _warning_codes(result: Any) -> set[str]:
    return {
        str(issue.code)
        for issue in getattr(result, "consistency_issues", [])
        if getattr(issue, "severity", None) == "warning"
    }


def _current_age_label(age: int) -> str:
    if age >= 100:
        return f"{age}세"
    if age >= 10:
        return f"{(age // 10) * 10}대"
    return f"{age}세"


def _coerce_age_mismatch_references(
    row: dict[str, Any],
    quant: dict[str, Any],
    dataset_cfg: CompiledPAKDatasetConfig,
) -> dict[str, Any]:
    age = int(quant.get("age", 0) or 0)
    correct_label = _current_age_label(age)
    mismatched_labels = [
        label
        for label, (lo, hi) in _AGE_GENERATION_PATTERNS.items()
        if not (lo <= age <= hi)
    ]
    if not mismatched_labels:
        return row

    updated = dict(row)
    narrative_fields = dataset_cfg.narrative_spec.output_fields
    for field in narrative_fields:
        text = updated.get(field)
        if not isinstance(text, str) or not text:
            continue
        replacements: list[tuple[int, int, str]] = []
        for label in mismatched_labels:
            for match in re.finditer(re.escape(label), text):
                window = text[match.start() : min(len(text), match.end() + 24)]
                if _AGE_PAST_PHASE_CONTEXT.search(window):
                    continue
                if not _is_self_age_reference(text, match.start(), match.end()):
                    continue
                end = match.end()
                above = re.match(r"\s*이상", text[end:])
                if above:
                    end += above.end()
                replacements.append((match.start(), end, correct_label))
        for start, end, value in sorted(replacements, reverse=True):
            text = text[:start] + value + text[end:]
        updated[field] = text
    return updated


def _coerce_occupation_anchor(
    row: dict[str, Any],
    quant: dict[str, Any],
) -> dict[str, Any]:
    occupation = str(quant.get("occupation", "") or "").strip()
    anchors = _OCCUPATION_ANCHORS.get(occupation)
    if not occupation or not anchors:
        return row

    current = "\n".join(
        str(row.get(field, "") or "")
        for field in ("persona", "professional_persona")
    )
    if any(anchor in current for anchor in anchors):
        return row

    updated = dict(row)
    anchor = anchors[0]
    province = str(quant.get("province", "") or "한국")
    career_years = int(quant.get("career_years", 0) or 0)
    updated["persona"] = (
        f"이 사람은 {province}에서 {occupation}로 활동하며, "
        f"{career_years}년의 경력을 바탕으로 작업과 생활을 조율한다."
    )
    original_professional = str(row.get("professional_persona", "") or "")
    prefix = f"{occupation}로서 {anchor} 관련 작업을 중심에 두고 활동한다. "
    updated["professional_persona"] = (prefix + original_professional)[:600]
    return updated


def _coerce_employment_duration_conflation(
    row: dict[str, Any],
    quant: dict[str, Any],
    dataset_cfg: CompiledPAKDatasetConfig,
) -> dict[str, Any]:
    career_years = int(quant.get("career_years", 0) or 0)
    if career_years <= 0:
        return row
    employment = str(quant.get("employment_type", "") or "전업")
    replacement = f"활동 경력은 {career_years}년이며 현재는 {employment} 상태"
    years = re.escape(str(career_years))
    patterns = (
        re.compile(rf"{years}\s*년(?:간)?\s*(?:동안\s*)?(?:전업|겸업)"),
        re.compile(rf"(?:전업|겸업)\s*(?:으로서|으로|상태로)?\s*{years}\s*년"),
    )

    updated = dict(row)
    for field in dataset_cfg.narrative_spec.output_fields:
        text = updated.get(field)
        if not isinstance(text, str) or not text:
            continue
        for pattern in patterns:
            text = pattern.sub(replacement, text)
        updated[field] = text
    return updated


def _coerce_blocking_warning_issues(
    row: dict[str, Any],
    quant: dict[str, Any],
    result: Any,
    dataset_cfg: CompiledPAKDatasetConfig,
) -> dict[str, Any]:
    codes = _warning_codes(result)
    updated = dict(row)
    if "AGE_MISMATCH" in codes:
        updated = _coerce_age_mismatch_references(updated, quant, dataset_cfg)
    if "OCCUPATION_MISMATCH" in codes:
        updated = _coerce_occupation_anchor(updated, quant)
    if "EMPLOYMENT_DURATION_CONFLATION" in codes:
        updated = _coerce_employment_duration_conflation(updated, quant, dataset_cfg)
    return PAKPersona.model_validate({**quant, **_narratives_only(updated, dataset_cfg)}).model_dump()


def generate_one(
    quant: dict,
    *,
    client: Any,
    cfg: GenerateConfig,
    rng: random.Random,
    pipeline: ValidationPipeline | None = None,
    dataset_cfg: CompiledPAKDatasetConfig | None = None,
    seen_hobby_sets: set[tuple[str, ...]] | None = None,
    seen_hobby_family_sets: set[tuple[str, ...]] | None = None,
    hobby_item_counts: Counter[str] | None = None,
    hobby_family_counts: Counter[str] | None = None,
    recent_hobby_family_counts: Counter[str] | None = None,
    network_role_counts: Counter[str] | None = None,
    network_friction_counts: Counter[str] | None = None,
    family_rhythm_counts: Counter[str] | None = None,
    family_boundary_counts: Counter[str] | None = None,
    family_responsibility_counts: Counter[str] | None = None,
    support_attitude_counts: Counter[str] | None = None,
    support_decision_counts: Counter[str] | None = None,
    support_path_counts: Counter[str] | None = None,
    support_friction_counts: Counter[str] | None = None,
    support_effect_counts: Counter[str] | None = None,
) -> tuple[dict | None, dict[str, Any], Any | None]:
    dataset_cfg = dataset_cfg or get_default_pak_core_dataset_config()
    last_err: str | None = None
    last_validation_result = None
    retry_feedback: list[str] = []
    usage_log: dict[str, Any] = {
        "pak_uuid": quant["pak_uuid"],
        "model": cfg.model,
        "attempt": 0,
        "attempts": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "elapsed_sec": 0.0,
        "validation_failed_attempts": 0,
        "repair_calls": 0,
        "repair_input_tokens": 0,
        "repair_output_tokens": 0,
        "warning_revision_calls": 0,
        "warning_revision_input_tokens": 0,
        "warning_revision_output_tokens": 0,
        "deterministic_warning_coercions": 0,
        "hobby_revision_calls": 0,
        "hobby_revision_input_tokens": 0,
        "hobby_revision_output_tokens": 0,
        "living_revision_calls": 0,
        "living_revision_input_tokens": 0,
        "living_revision_output_tokens": 0,
        "family_revision_calls": 0,
        "family_revision_input_tokens": 0,
        "family_revision_output_tokens": 0,
        "network_revision_calls": 0,
        "network_revision_input_tokens": 0,
        "network_revision_output_tokens": 0,
        "support_revision_calls": 0,
        "support_revision_input_tokens": 0,
        "support_revision_output_tokens": 0,
        "duplicate_hobby_retries": 0,
        "hobby_plan_coercions": 0,
    }
    blocked_hobby_items = _blocked_hobby_items_from_counts(
        hobby_item_counts,
        cap=getattr(cfg, "hobby_exact_atom_cap_per_batch", None),
    )
    blocked_hobby_families = _blocked_hobby_families_from_counts(
        hobby_family_counts,
        cap=getattr(cfg, "hobby_family_cap_per_batch", None),
    )
    prompt_context = _build_prompt_context(
        quant,
        rng,
        hobby_item_counts=hobby_item_counts,
        recent_hobby_family_counts=recent_hobby_family_counts,
        blocked_items=blocked_hobby_items,
        blocked_families=blocked_hobby_families,
        seen_hobby_sets=seen_hobby_sets,
        seen_hobby_family_sets=seen_hobby_family_sets,
        network_role_counts=network_role_counts,
        network_friction_counts=network_friction_counts,
        family_rhythm_counts=family_rhythm_counts,
        family_boundary_counts=family_boundary_counts,
        family_responsibility_counts=family_responsibility_counts,
        support_attitude_counts=support_attitude_counts,
        support_decision_counts=support_decision_counts,
        support_path_counts=support_path_counts,
        support_friction_counts=support_friction_counts,
        support_effect_counts=support_effect_counts,
    )
    usage_log["selected_family_rhythm"] = prompt_context.get("family_rhythm")
    usage_log["selected_family_boundary"] = prompt_context.get("family_boundary")
    usage_log["selected_family_responsibility"] = prompt_context.get("family_responsibility")
    usage_log["selected_network_role"] = prompt_context.get("network_role")
    usage_log["selected_network_friction"] = prompt_context.get("network_friction")
    usage_log["selected_support_attitude"] = prompt_context.get("support_attitude")
    usage_log["selected_support_decision"] = prompt_context.get("support_decision")
    usage_log["selected_support_path"] = prompt_context.get("support_path")
    usage_log["selected_support_friction"] = prompt_context.get("support_friction")
    usage_log["selected_support_effect"] = prompt_context.get("support_effect")
    for attempt in range(cfg.max_retries + 1):
        try:
            system, user = build_single_call_prompt(
                quant,
                rng,
                dataset_cfg,
                prompt_context=prompt_context,
                retry_feedback=retry_feedback,
            )
            t0 = time.time()
            resp = client.chat(
                model=cfg.model,
                system=system,
                user=user,
                max_tokens=cfg.max_tokens,
                temperature=_retry_temperature(cfg, attempt),
                response_format={"type": "json_object"},
            )
            elapsed = time.time() - t0
            usage_log["attempt"] = attempt
            usage_log["attempts"] = attempt + 1
            usage_log["input_tokens"] += resp.usage.input_tokens
            usage_log["output_tokens"] += resp.usage.output_tokens
            usage_log["elapsed_sec"] += elapsed
            narratives, repair_usage = _complete_narrative_response(
                resp.text,
                quant=quant,
                prompt_context=prompt_context,
                client=client,
                cfg=cfg,
                dataset_cfg=dataset_cfg,
            )
            usage_log["repair_calls"] += repair_usage["repair_calls"]
            usage_log["repair_input_tokens"] += repair_usage["repair_input_tokens"]
            usage_log["repair_output_tokens"] += repair_usage["repair_output_tokens"]
            usage_log["input_tokens"] += repair_usage["repair_input_tokens"]
            usage_log["output_tokens"] += repair_usage["repair_output_tokens"]
            persona = PAKPersona.model_validate({**quant, **narratives})
            row = persona.model_dump()

            if seen_hobby_sets is not None or seen_hobby_family_sets is not None:
                hobby_set = _normalized_hobby_set(row.get("hobbies_and_interests_list"))
                hobby_family_set = _normalized_hobby_family_set(row.get("hobbies_and_interests_list"))
                duplication = _hobby_duplication_issues(
                    hobby_set=hobby_set,
                    hobby_family_set=hobby_family_set,
                    seen_hobby_sets=seen_hobby_sets,
                    seen_hobby_family_sets=seen_hobby_family_sets,
                )
                if duplication is not None:
                    issues, blocked_families, duplicate_err = duplication
                    usage_log["duplicate_hobby_retries"] += 1
                    prompt_context = _refresh_hobby_prompt_context(
                        prompt_context,
                        quant,
                        rng,
                        hobby_item_counts=hobby_item_counts,
                        recent_hobby_family_counts=recent_hobby_family_counts,
                        blocked_items=blocked_hobby_items,
                        blocked_families=blocked_hobby_families | blocked_families,
                        seen_hobby_sets=seen_hobby_sets,
                        seen_hobby_family_sets=seen_hobby_family_sets,
                    )
                    if getattr(cfg, "coerce_hobbies_to_plan", False):
                        row = _coerce_hobbies_to_plan(row, prompt_context)
                        row = PAKPersona.model_validate(
                            {**quant, **_narratives_only(row, dataset_cfg)}
                        ).model_dump()
                        usage_log["hobby_plan_coercions"] += 1
                    else:
                        revised_narratives, rev_in, rev_out = _revise_narrative_after_validation(
                            client=client,
                            cfg=cfg,
                            quant=quant,
                            prompt_context=prompt_context,
                            narratives=_narratives_only(row, dataset_cfg),
                            issues=issues,
                            dataset_cfg=dataset_cfg,
                        )
                        usage_log["hobby_revision_calls"] += 1
                        usage_log["hobby_revision_input_tokens"] += rev_in
                        usage_log["hobby_revision_output_tokens"] += rev_out
                        usage_log["input_tokens"] += rev_in
                        usage_log["output_tokens"] += rev_out
                        row = PAKPersona.model_validate({**quant, **revised_narratives}).model_dump()
                    hobby_set = _normalized_hobby_set(row.get("hobbies_and_interests_list"))
                    hobby_family_set = _normalized_hobby_family_set(row.get("hobbies_and_interests_list"))
                    duplication = _hobby_duplication_issues(
                        hobby_set=hobby_set,
                        hobby_family_set=hobby_family_set,
                        seen_hobby_sets=seen_hobby_sets,
                        seen_hobby_family_sets=seen_hobby_family_sets,
                    )
                    if duplication is not None:
                        issues, blocked_families, duplicate_err = duplication
                        usage_log["validation_failed_attempts"] += 1
                        last_err = duplicate_err
                        retry_feedback = issues
                        logger.warning(
                            "generate attempt %d duplicate hobby set for %s: %s",
                            attempt,
                            quant.get("pak_uuid", "?"),
                            last_err,
                        )
                        continue

            if getattr(cfg, "enforce_hobby_plan_alignment", False):
                hobby_family_set = _normalized_hobby_family_set(row.get("hobbies_and_interests_list"))
                hobby_plan_drift = _hobby_plan_drift_issues(
                    hobby_family_set=hobby_family_set,
                    prompt_context=prompt_context,
                )
                if hobby_plan_drift is not None:
                    issues, blocked_families = hobby_plan_drift
                    if getattr(cfg, "coerce_hobbies_to_plan", False):
                        row = _coerce_hobbies_to_plan(row, prompt_context)
                        row = PAKPersona.model_validate(
                            {**quant, **_narratives_only(row, dataset_cfg)}
                        ).model_dump()
                        usage_log["hobby_plan_coercions"] += 1
                    else:
                        prompt_context = _apply_blocked_hobby_families(prompt_context, blocked_families)
                        revised_narratives, rev_in, rev_out = _revise_narrative_after_validation(
                            client=client,
                            cfg=cfg,
                            quant=quant,
                            prompt_context=prompt_context,
                            narratives=_narratives_only(row, dataset_cfg),
                            issues=issues,
                            dataset_cfg=dataset_cfg,
                        )
                        usage_log["hobby_revision_calls"] += 1
                        usage_log["hobby_revision_input_tokens"] += rev_in
                        usage_log["hobby_revision_output_tokens"] += rev_out
                        usage_log["input_tokens"] += rev_in
                        usage_log["output_tokens"] += rev_out
                        row = PAKPersona.model_validate({**quant, **revised_narratives}).model_dump()

            hobby_items = tuple(_parse_hobby_items(row.get("hobbies_and_interests_list")))
            hobby_atomicity = _hobby_list_atomicity_issues(
                hobby_items=hobby_items,
                prompt_context=prompt_context,
            )
            if hobby_atomicity is not None:
                issues, _flagged_items = hobby_atomicity
                if getattr(cfg, "coerce_hobbies_to_plan", False):
                    row = _coerce_hobbies_to_plan(row, prompt_context)
                    row = PAKPersona.model_validate(
                        {**quant, **_narratives_only(row, dataset_cfg)}
                    ).model_dump()
                    usage_log["hobby_plan_coercions"] += 1
                else:
                    revised_narratives, rev_in, rev_out = _revise_narrative_after_validation(
                        client=client,
                        cfg=cfg,
                        quant=quant,
                        prompt_context=prompt_context,
                        narratives=_narratives_only(row, dataset_cfg),
                        issues=issues,
                        dataset_cfg=dataset_cfg,
                    )
                    usage_log["hobby_revision_calls"] += 1
                    usage_log["hobby_revision_input_tokens"] += rev_in
                    usage_log["hobby_revision_output_tokens"] += rev_out
                    usage_log["input_tokens"] += rev_in
                    usage_log["output_tokens"] += rev_out
                    row = PAKPersona.model_validate({**quant, **revised_narratives}).model_dump()
                hobby_items = tuple(_parse_hobby_items(row.get("hobbies_and_interests_list")))
                hobby_atomicity = _hobby_list_atomicity_issues(
                    hobby_items=hobby_items,
                    prompt_context=prompt_context,
                )
                if hobby_atomicity is not None:
                    row = _coerce_hobby_list_to_plan(row, prompt_context)
                    row = PAKPersona.model_validate(
                        {**quant, **_narratives_only(row, dataset_cfg)}
                    ).model_dump()
                    usage_log["hobby_plan_coercions"] += 1
                    hobby_items = tuple(_parse_hobby_items(row.get("hobbies_and_interests_list")))

            hobby_family_set = _normalized_hobby_family_set(row.get("hobbies_and_interests_list"))
            hobby_quota = _hobby_quota_issues(
                hobby_items=hobby_items,
                hobby_family_set=hobby_family_set,
                hobby_item_counts=hobby_item_counts,
                hobby_family_counts=hobby_family_counts,
                exact_cap=getattr(cfg, "hobby_exact_atom_cap_per_batch", None),
                family_cap=getattr(cfg, "hobby_family_cap_per_batch", None),
            )
            if hobby_quota is not None:
                issues, blocked_items, blocked_families, quota_err = hobby_quota
                prompt_context = _refresh_hobby_prompt_context(
                    prompt_context,
                    quant,
                    rng,
                    hobby_item_counts=hobby_item_counts,
                    recent_hobby_family_counts=recent_hobby_family_counts,
                    blocked_items=blocked_hobby_items | blocked_items,
                    blocked_families=blocked_hobby_families | blocked_families,
                    seen_hobby_sets=seen_hobby_sets,
                    seen_hobby_family_sets=seen_hobby_family_sets,
                )
                if getattr(cfg, "coerce_hobbies_to_plan", False):
                    row = _coerce_hobbies_to_plan(row, prompt_context)
                    row = PAKPersona.model_validate(
                        {**quant, **_narratives_only(row, dataset_cfg)}
                    ).model_dump()
                    usage_log["hobby_plan_coercions"] += 1
                else:
                    revised_narratives, rev_in, rev_out = _revise_narrative_after_validation(
                        client=client,
                        cfg=cfg,
                        quant=quant,
                        prompt_context=prompt_context,
                        narratives=_narratives_only(row, dataset_cfg),
                        issues=issues,
                        dataset_cfg=dataset_cfg,
                    )
                    usage_log["hobby_revision_calls"] += 1
                    usage_log["hobby_revision_input_tokens"] += rev_in
                    usage_log["hobby_revision_output_tokens"] += rev_out
                    usage_log["input_tokens"] += rev_in
                    usage_log["output_tokens"] += rev_out
                    row = PAKPersona.model_validate({**quant, **revised_narratives}).model_dump()
                hobby_items = tuple(_parse_hobby_items(row.get("hobbies_and_interests_list")))
                hobby_family_set = _normalized_hobby_family_set(row.get("hobbies_and_interests_list"))
                hobby_quota = _hobby_quota_issues(
                    hobby_items=hobby_items,
                    hobby_family_set=hobby_family_set,
                    hobby_item_counts=hobby_item_counts,
                    hobby_family_counts=hobby_family_counts,
                    exact_cap=getattr(cfg, "hobby_exact_atom_cap_per_batch", None),
                    family_cap=getattr(cfg, "hobby_family_cap_per_batch", None),
                )
                if hobby_quota is not None:
                    usage_log["validation_failed_attempts"] += 1
                    last_err = quota_err
                    retry_feedback = issues
                    logger.warning(
                        "generate attempt %d unresolved hobby quota for %s",
                        attempt,
                        quant.get("pak_uuid", "?"),
                    )
                    continue

            if getattr(cfg, "enforce_living_persona_specificity", False):
                living_genericity = _living_genericity_issues(
                    living_text=str(row.get("living_persona", "")),
                    prompt_context=prompt_context,
                )
                if living_genericity is not None:
                    revised_narratives, rev_in, rev_out = _revise_narrative_after_validation(
                        client=client,
                        cfg=cfg,
                        quant=quant,
                        prompt_context=prompt_context,
                        narratives=_narratives_only(row, dataset_cfg),
                        issues=living_genericity,
                        dataset_cfg=dataset_cfg,
                    )
                    usage_log["living_revision_calls"] += 1
                    usage_log["living_revision_input_tokens"] += rev_in
                    usage_log["living_revision_output_tokens"] += rev_out
                    usage_log["input_tokens"] += rev_in
                    usage_log["output_tokens"] += rev_out
                    row = PAKPersona.model_validate({**quant, **revised_narratives}).model_dump()
                    living_genericity = _living_genericity_issues(
                        living_text=str(row.get("living_persona", "")),
                        prompt_context=prompt_context,
                    )
                    if living_genericity is not None:
                        usage_log["validation_failed_attempts"] += 1
                        last_err = "living genericity unresolved"
                        retry_feedback = living_genericity
                        logger.warning(
                            "generate attempt %d unresolved living genericity for %s",
                            attempt,
                            quant.get("pak_uuid", "?"),
                        )
                        continue

            if getattr(cfg, "enforce_family_persona_specificity", False):
                family_genericity = _family_genericity_issues(
                    family_text=str(row.get("family_persona", "")),
                    prompt_context=prompt_context,
                )
                if family_genericity is not None:
                    revised_narratives, rev_in, rev_out = _revise_narrative_after_validation(
                        client=client,
                        cfg=cfg,
                        quant=quant,
                        prompt_context=prompt_context,
                        narratives=_narratives_only(row, dataset_cfg),
                        issues=family_genericity,
                        dataset_cfg=dataset_cfg,
                    )
                    usage_log["family_revision_calls"] += 1
                    usage_log["family_revision_input_tokens"] += rev_in
                    usage_log["family_revision_output_tokens"] += rev_out
                    usage_log["input_tokens"] += rev_in
                    usage_log["output_tokens"] += rev_out
                    row = PAKPersona.model_validate({**quant, **revised_narratives}).model_dump()
                    family_genericity = _family_genericity_issues(
                        family_text=str(row.get("family_persona", "")),
                        prompt_context=prompt_context,
                    )
                    if family_genericity is not None:
                        usage_log["validation_failed_attempts"] += 1
                        last_err = "family genericity unresolved"
                        retry_feedback = family_genericity
                        logger.warning(
                            "generate attempt %d unresolved family genericity for %s",
                            attempt,
                            quant.get("pak_uuid", "?"),
                        )
                        continue

            if getattr(cfg, "enforce_network_persona_specificity", False):
                network_genericity = _network_genericity_issues(
                    network_text=str(row.get("network_persona", "")),
                    prompt_context=prompt_context,
                )
                if network_genericity is not None:
                    revised_narratives, rev_in, rev_out = _revise_narrative_after_validation(
                        client=client,
                        cfg=cfg,
                        quant=quant,
                        prompt_context=prompt_context,
                        narratives=_narratives_only(row, dataset_cfg),
                        issues=network_genericity,
                        dataset_cfg=dataset_cfg,
                    )
                    usage_log["network_revision_calls"] += 1
                    usage_log["network_revision_input_tokens"] += rev_in
                    usage_log["network_revision_output_tokens"] += rev_out
                    usage_log["input_tokens"] += rev_in
                    usage_log["output_tokens"] += rev_out
                    row = PAKPersona.model_validate({**quant, **revised_narratives}).model_dump()
                    network_genericity = _network_genericity_issues(
                        network_text=str(row.get("network_persona", "")),
                        prompt_context=prompt_context,
                    )
                    if network_genericity is not None:
                        usage_log["validation_failed_attempts"] += 1
                        last_err = "network genericity unresolved"
                        retry_feedback = network_genericity
                        logger.warning(
                            "generate attempt %d unresolved network genericity for %s",
                            attempt,
                            quant.get("pak_uuid", "?"),
                        )
                        continue

            if getattr(cfg, "enforce_support_persona_specificity", False):
                support_genericity = _support_genericity_issues(
                    support_text=str(row.get("support_persona", "")),
                    prompt_context=prompt_context,
                )
                if support_genericity is not None:
                    revised_narratives, rev_in, rev_out = _revise_narrative_after_validation(
                        client=client,
                        cfg=cfg,
                        quant=quant,
                        prompt_context=prompt_context,
                        narratives=_narratives_only(row, dataset_cfg),
                        issues=support_genericity,
                        dataset_cfg=dataset_cfg,
                    )
                    usage_log["support_revision_calls"] += 1
                    usage_log["support_revision_input_tokens"] += rev_in
                    usage_log["support_revision_output_tokens"] += rev_out
                    usage_log["input_tokens"] += rev_in
                    usage_log["output_tokens"] += rev_out
                    row = PAKPersona.model_validate({**quant, **revised_narratives}).model_dump()
                    support_genericity = _support_genericity_issues(
                        support_text=str(row.get("support_persona", "")),
                        prompt_context=prompt_context,
                    )
                    if support_genericity is not None:
                        usage_log["validation_failed_attempts"] += 1
                        last_err = "support genericity unresolved"
                        retry_feedback = support_genericity
                        logger.warning(
                            "generate attempt %d unresolved support genericity for %s",
                            attempt,
                            quant.get("pak_uuid", "?"),
                        )
                        continue

            if not cfg.skip_validation and pipeline is not None:
                v_result = pipeline.validate_one(
                    pak_uuid=row["pak_uuid"],
                    quant=quant,
                    narratives=_narratives_only(row, dataset_cfg),
                )
                last_validation_result = v_result
                if v_result.has_errors:
                    usage_log["validation_failed_attempts"] += 1
                    last_err = _format_validation_error_codes(v_result)
                    retry_feedback = _feedback_lines_from_validation(v_result)
                    logger.warning(
                        "generate attempt %d failed validation for %s: %s",
                        attempt,
                        quant.get("pak_uuid", "?"),
                        last_err,
                    )
                    continue

                warning_issues = _blocking_warning_issues(v_result, cfg)
                if warning_issues:
                    revised_row = row
                    revised_narratives = _narratives_only(row, dataset_cfg)
                    revised_result = v_result
                    for _ in range(int(getattr(cfg, "max_warning_revisions", 0) or 0)):
                        issues = _feedback_lines_from_validation(
                            revised_result,
                            include_warnings=True,
                        )
                        revised_narratives, rev_in, rev_out = _revise_narrative_after_validation(
                            client=client,
                            cfg=cfg,
                            quant=quant,
                            prompt_context=prompt_context,
                            narratives=revised_narratives,
                            issues=issues,
                            dataset_cfg=dataset_cfg,
                        )
                        usage_log["warning_revision_calls"] += 1
                        usage_log["warning_revision_input_tokens"] += rev_in
                        usage_log["warning_revision_output_tokens"] += rev_out
                        usage_log["input_tokens"] += rev_in
                        usage_log["output_tokens"] += rev_out
                        revised_row = PAKPersona.model_validate({**quant, **revised_narratives}).model_dump()
                        revised_result = pipeline.validate_one(
                            pak_uuid=revised_row["pak_uuid"],
                            quant=quant,
                            narratives=_narratives_only(revised_row, dataset_cfg),
                        )
                        last_validation_result = revised_result
                        if not revised_result.has_errors and not _blocking_warning_issues(revised_result, cfg):
                            return revised_row, usage_log, revised_result

                    coerced_row = _coerce_blocking_warning_issues(
                        revised_row,
                        quant,
                        revised_result,
                        dataset_cfg,
                    )
                    coerced_result = pipeline.validate_one(
                        pak_uuid=coerced_row["pak_uuid"],
                        quant=quant,
                        narratives=_narratives_only(coerced_row, dataset_cfg),
                    )
                    last_validation_result = coerced_result
                    if not coerced_result.has_errors and not _blocking_warning_issues(coerced_result, cfg):
                        usage_log["deterministic_warning_coercions"] += 1
                        return coerced_row, usage_log, coerced_result

                    usage_log["validation_failed_attempts"] += 1
                    last_err = "blocking warnings: " + ",".join(
                        issue.code for issue in _blocking_warning_issues(revised_result, cfg)
                    )
                    retry_feedback = _feedback_lines_from_validation(revised_result)
                    logger.warning(
                        "generate attempt %d unresolved warnings for %s: %s",
                        attempt,
                        quant.get("pak_uuid", "?"),
                        last_err,
                    )
                    continue
                return row, usage_log, v_result

            return row, usage_log, None
        except Exception as exc:
            last_err = str(exc)
            retry_feedback = [f"이전 시도 실패: {last_err}"]
            logger.warning(
                "generate attempt %d failed for %s: %s",
                attempt,
                quant.get("pak_uuid", "?"),
                exc,
            )
    usage_log["error"] = last_err
    return None, usage_log, last_validation_result


def run(cfg: GenerateConfig) -> Path:
    """Synchronous single-client generation (simple). Can be extended to asyncio later."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.seed)
    np_rng = __import__("numpy").random.default_rng(cfg.seed)

    chain = build_chain_from_spec()
    client = get_client()
    pipeline = ValidationPipeline()
    dataset_cfg = get_default_pak_core_dataset_config()

    rows: list[dict] = []
    failures: list[dict] = []
    usage_logs: list[dict] = []
    validation_summaries: list[dict] = []
    seen_hobby_sets: set[tuple[str, ...]] = set()
    seen_hobby_family_sets: set[tuple[str, ...]] = set()
    hobby_item_counts: Counter[str] = Counter()
    hobby_family_counts: Counter[str] = Counter()
    recent_hobby_families: deque[str] = deque()
    recent_hobby_family_counts: Counter[str] = Counter()
    network_role_counts: Counter[str] = Counter()
    network_friction_counts: Counter[str] = Counter()
    family_rhythm_counts: Counter[str] = Counter()
    family_boundary_counts: Counter[str] = Counter()
    family_responsibility_counts: Counter[str] = Counter()
    support_attitude_counts: Counter[str] = Counter()
    support_decision_counts: Counter[str] = Counter()
    support_path_counts: Counter[str] = Counter()
    support_friction_counts: Counter[str] = Counter()
    support_effect_counts: Counter[str] = Counter()

    output_parquet = cfg.output_dir / "personas.parquet"
    cost_log_path = cfg.output_dir / "usage_log.jsonl"
    failure_log_path = cfg.output_dir / "failures.jsonl"

    started = datetime.now(UTC).astimezone().isoformat(timespec="seconds")
    eval_set_meta: dict[str, Any] | None = None
    fixed_quants: list[dict[str, Any]] | None = None
    if cfg.quant_rows_path is not None:
        fixed_quants, eval_set_meta = _load_quant_rows_from_path(cfg.quant_rows_path)
        if not fixed_quants:
            raise ValueError(f"quant_rows_path produced no quant rows: {cfg.quant_rows_path}")

    requested_n = cfg.n
    if fixed_quants is not None:
        requested_n = min(cfg.n, len(fixed_quants)) if cfg.n > 0 else len(fixed_quants)

    n_concurrent = max(1, min(int(getattr(cfg, "max_concurrent", 1) or 1), 8))

    def _run_one(valid_q: dict[str, Any]) -> tuple[dict | None, dict, Any]:
        # Closure for batch parallelism. Counters are passed read-only (snapshot at creation time); updates happen on the main thread.
        return generate_one(
            valid_q,
            client=client,
            cfg=cfg,
            rng=rng,
            pipeline=pipeline,
            dataset_cfg=dataset_cfg,
            seen_hobby_sets=seen_hobby_sets if cfg.quality_profile == "max" else None,
            seen_hobby_family_sets=seen_hobby_family_sets if cfg.quality_profile == "max" else None,
            hobby_item_counts=hobby_item_counts if cfg.quality_profile == "max" else None,
            hobby_family_counts=hobby_family_counts if cfg.quality_profile == "max" else None,
            recent_hobby_family_counts=recent_hobby_family_counts,
            network_role_counts=network_role_counts if cfg.quality_profile == "max" else None,
            network_friction_counts=network_friction_counts if cfg.quality_profile == "max" else None,
            family_rhythm_counts=family_rhythm_counts if cfg.quality_profile == "max" else None,
            family_boundary_counts=family_boundary_counts if cfg.quality_profile == "max" else None,
            family_responsibility_counts=family_responsibility_counts if cfg.quality_profile == "max" else None,
            support_attitude_counts=support_attitude_counts if cfg.quality_profile == "max" else None,
            support_decision_counts=support_decision_counts if cfg.quality_profile == "max" else None,
            support_path_counts=support_path_counts if cfg.quality_profile == "max" else None,
            support_friction_counts=support_friction_counts if cfg.quality_profile == "max" else None,
            support_effect_counts=support_effect_counts if cfg.quality_profile == "max" else None,
        )

    executor = (
        ThreadPoolExecutor(max_workers=n_concurrent) if n_concurrent > 1 else None
    )
    if executor is not None:
        logger.info("persona-level concurrency enabled: max_workers=%d", n_concurrent)

    pbar = tqdm(
        total=requested_n,
        desc=f"PAK gen ({cfg.model})",
        unit="persona",
        smoothing=0.05,
        dynamic_ncols=True,
    )

    i = 0
    while i < requested_n:
        # Prepare quant in batches
        batch_size = n_concurrent if executor is not None else 1
        batch_valid: list[dict[str, Any]] = []
        batch_indices: list[int] = []
        while len(batch_valid) < batch_size and i < requested_n:
            quant = fixed_quants[i] if fixed_quants is not None else sample_full_quant(chain, rng, np_rng)
            valid_q = _validate_quant(quant)
            if valid_q is None:
                failures.append({"pak_uuid": quant.get("pak_uuid"), "stage": "quant_validation"})
                i += 1
                continue
            batch_valid.append(valid_q)
            batch_indices.append(i)
            i += 1
        if not batch_valid:
            continue

        if executor is not None and len(batch_valid) > 1:
            futures = [executor.submit(_run_one, q) for q in batch_valid]
            results = [f.result() for f in futures]
        else:
            results = [_run_one(batch_valid[0])]

        for valid_q, (row, log, v_result) in zip(batch_valid, results, strict=True):
            usage_logs.append(log)
            if row is None:
                failures.append(
                    {
                        "pak_uuid": valid_q["pak_uuid"],
                        "stage": "generation_or_validation",
                        "error": log.get("error"),
                        "attempts": log.get("attempts"),
                    }
                )
                continue

            if not cfg.skip_validation and v_result is not None:
                validation_summaries.append(
                    {
                        "pak_uuid": row["pak_uuid"],
                        "has_errors": v_result.has_errors,
                        "has_warnings": v_result.has_warnings,
                        "n_consistency": len(v_result.consistency_issues),
                        "n_cliches": len(v_result.cliche_hits),
                    }
                )

            rows.append(row)
            if log.get("selected_family_rhythm"):
                family_rhythm_counts.update([str(log["selected_family_rhythm"])])
            if log.get("selected_family_boundary"):
                family_boundary_counts.update([str(log["selected_family_boundary"])])
            if log.get("selected_family_responsibility"):
                family_responsibility_counts.update([str(log["selected_family_responsibility"])])
            if log.get("selected_network_role"):
                network_role_counts.update([str(log["selected_network_role"])])
            if log.get("selected_network_friction"):
                network_friction_counts.update([str(log["selected_network_friction"])])
            if log.get("selected_support_attitude"):
                support_attitude_counts.update([str(log["selected_support_attitude"])])
            if log.get("selected_support_decision"):
                support_decision_counts.update([str(log["selected_support_decision"])])
            if log.get("selected_support_path"):
                support_path_counts.update([str(log["selected_support_path"])])
            if log.get("selected_support_friction"):
                support_friction_counts.update([str(log["selected_support_friction"])])
            if log.get("selected_support_effect"):
                support_effect_counts.update([str(log["selected_support_effect"])])
            hobby_set = _normalized_hobby_set(row.get("hobbies_and_interests_list"))
            if hobby_set:
                seen_hobby_sets.add(hobby_set)
                hobby_item_counts.update(hobby_set)
            hobby_family_set = _normalized_hobby_family_set(row.get("hobbies_and_interests_list"))
            if hobby_family_set:
                seen_hobby_family_sets.add(hobby_family_set)
                hobby_family_counts.update(hobby_family_set)
                _append_recent_hobby_families(
                    recent_hobby_families,
                    recent_hobby_family_counts,
                    hobby_family_set,
                )

        # After the batch completes, update progress + decide checkpoint (based on the last processed index)
        last_index = batch_indices[-1] if batch_indices else i - 1
        # tqdm update: by the number of valid personas processed within the batch.
        pbar.update(len(batch_valid))
        pbar.set_postfix(
            ok=len(rows),
            fail=len(failures),
            warn=sum(1 for v in validation_summaries if v["has_warnings"]),
        )

        if (last_index + 1) % cfg.checkpoint_every == 0 or (last_index + 1) >= requested_n:
            df = pd.DataFrame(rows)
            df.to_parquet(output_parquet, index=False)
            logger.info(
                "checkpoint %d/%d → %s", last_index + 1, requested_n, output_parquet
            )

    pbar.close()
    if executor is not None:
        executor.shutdown(wait=True)

    # Flush logs
    with cost_log_path.open("w", encoding="utf-8") as f:
        for log in usage_logs:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")
    if failures:
        with failure_log_path.open("w", encoding="utf-8") as f:
            for fr in failures:
                f.write(json.dumps(fr, ensure_ascii=False) + "\n")

    # Metadata
    finished = datetime.now(UTC).astimezone().isoformat(timespec="seconds")
    meta = {
        "n_requested": requested_n,
        "n_generated": len(rows),
        "n_failed": len(failures),
        "model": cfg.model,
        "started_at": started,
        "finished_at": finished,
        "config": asdict(cfg) | {"output_dir": str(cfg.output_dir)},
        "dataset_config": {
            "name": dataset_cfg.config.name,
            "version": dataset_cfg.config.version,
            "narrative_column": dataset_cfg.narrative_column_name,
            **dataset_cfg.config.fingerprint(),
        },
        "validation_summary": {
            "n_with_errors": sum(1 for v in validation_summaries if v["has_errors"]),
            "n_with_warnings": sum(1 for v in validation_summaries if v["has_warnings"]),
            "total_cliches": sum(v["n_cliches"] for v in validation_summaries),
            "total_consistency_issues": sum(v["n_consistency"] for v in validation_summaries),
        },
        "total_input_tokens": sum(int(log.get("input_tokens", 0) or 0) for log in usage_logs),
        "total_output_tokens": sum(int(log.get("output_tokens", 0) or 0) for log in usage_logs),
    }
    if eval_set_meta is not None:
        meta["eval_set"] = eval_set_meta
    (cfg.output_dir / "generation_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    logger.info("done: %d ok, %d failed", len(rows), len(failures))
    return output_parquet


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _cli() -> None:  # pragma: no cover
    import argparse

    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    preview = sub.add_parser("preview")
    preview.add_argument("--n", type=int, default=100)
    preview.add_argument("--seed", type=int, default=20260502)
    preview.add_argument("--output-dir", type=Path, default=settings.synthetic_dir / "preview_100")
    preview.add_argument("--quant-rows-path", type=Path, default=None)
    preview.add_argument("--checkpoint-every", type=int, default=10)
    preview.add_argument("--quality-profile", choices=["balanced", "max"], default="balanced")

    pilot = sub.add_parser("pilot")
    pilot.add_argument("--n", type=int, default=100)
    pilot.add_argument("--seed", type=int, default=20260502)
    pilot.add_argument("--output-dir", type=Path, default=settings.synthetic_dir / "pilot")
    pilot.add_argument("--quant-rows-path", type=Path, default=None)
    pilot.add_argument("--checkpoint-every", type=int, default=10)
    pilot.add_argument("--quality-profile", choices=["balanced", "max"], default="balanced")

    full = sub.add_parser("full")
    full.add_argument("--n", type=int, default=50_000)
    full.add_argument("--seed", type=int, default=20260502)
    full.add_argument("--output-dir", type=Path, default=settings.synthetic_dir / "v0_1")
    full.add_argument("--quant-rows-path", type=Path, default=None)
    full.add_argument("--checkpoint-every", type=int, default=500)
    full.add_argument("--quality-profile", choices=["balanced", "max"], default="balanced")

    create = sub.add_parser("create")
    create.add_argument("--preview-report", type=Path, required=True)
    create.add_argument("--n", type=int, default=50_000)
    create.add_argument("--seed", type=int, default=20260502)
    create.add_argument("--output-dir", type=Path, default=settings.synthetic_dir / "v0_1")
    create.add_argument("--quant-rows-path", type=Path, default=None)
    create.add_argument("--checkpoint-every", type=int, default=500)
    create.add_argument("--quality-profile", choices=["balanced", "max"], default="balanced")

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    if args.cmd == "preview":
        from pak.preview import run_preview

        cfg = GenerateConfig(
            n=args.n,
            seed=args.seed,
            output_dir=args.output_dir,
            quant_rows_path=args.quant_rows_path,
            checkpoint_every=args.checkpoint_every,
            quality_profile=args.quality_profile,
        )
        result = run_preview(cfg)
        print(f"wrote {result.artifacts.preview_report_json_path}")
        return

    if args.cmd == "create":
        from pak.preview import assert_create_ready

        assert_create_ready(args.preview_report)
        cfg = GenerateConfig(
            n=args.n,
            seed=args.seed,
            output_dir=args.output_dir,
            quant_rows_path=args.quant_rows_path,
            checkpoint_every=args.checkpoint_every,
            quality_profile=args.quality_profile,
        )
        out = run(cfg)
        print(f"wrote {out}")
        return

    cfg = GenerateConfig(
        n=args.n,
        seed=args.seed,
        output_dir=args.output_dir,
        quant_rows_path=args.quant_rows_path,
        checkpoint_every=args.checkpoint_every,
        quality_profile=args.quality_profile,
    )
    out = run(cfg)
    print(f"wrote {out}")


if __name__ == "__main__":  # pragma: no cover
    _cli()
