"""
core/parsers.py - 쿠키런 AI 플랫폼 텍스트 파싱 유틸리티
=====================================================
데브시스터즈 기술혁신 프로젝트
"""
import re
from typing import Any, Optional, Tuple

import pandas as pd

from .utils import safe_str
from .constants import DEFAULT_TOPN, MAX_TOPN


def extract_top_k_from_text(user_text: str, default_k: int = DEFAULT_TOPN) -> int:
    """텍스트에서 top-k 숫자 추출 (예: "상위 10개", "top 5")"""
    t = safe_str(user_text)

    # 1) top 10 / top=10 / top_k=10
    m = re.search(r"(?i)\btop\s*[_-]?\s*k?\s*[:=]?\s*(\d{1,3})", t)
    if not m:
        # 2) 상위 10 / TOP10
        m = re.search(r"(?i)(?:상위|top)\s*(\d{1,3})", t)
    if not m:
        # 3) 10개 / 10곳 / 10개 추천
        m = re.search(r"(\d{1,3})\s*(?:개|곳|건|명)", t)

    k = default_k
    if m:
        try:
            k = int(m.group(1))
        except Exception:
            k = default_k

    k = max(1, min(int(k), MAX_TOPN))
    return k


def parse_date_range_from_text(user_text: str) -> Tuple[Optional[str], Optional[str]]:
    """텍스트에서 날짜 범위 파싱 (예: "2024-01-01부터 2024-03-31까지")"""
    t = safe_str(user_text)
    pats = re.findall(r"(20\d{2})\s*[-./]?\s*(\d{1,2})\s*[-./]?\s*(\d{1,2})?", t)
    if not pats:
        return (None, None)

    dates = []
    for y, m, d in pats:
        mm = int(m)
        dd = int(d) if d else 1
        if 1 <= mm <= 12 and 1 <= dd <= 31:
            dates.append(f"{int(y):04d}-{mm:02d}-{dd:02d}")
    if not dates:
        return (None, None)

    if len(dates) == 1:
        return (dates[0], dates[0])

    dates = sorted(dates)
    return (dates[0], dates[-1])


def extract_user_id(text: str) -> Optional[str]:
    """텍스트에서 유저 ID 추출 (예: U0001)"""
    t = text or ""
    m = re.search(r"(?i)(?<![0-9a-zA-Z])(u\d{4})(?!\d)", t)
    if not m:
        m = re.search(r"(?i)(u\d{4})", t)
        if not m:
            return None
    return m.group(1).upper()


def extract_cookie_id(text: str) -> Optional[str]:
    """텍스트에서 쿠키 ID 추출 (예: CK001)"""
    t = text or ""
    m = re.search(r"(?i)(?<![0-9a-zA-Z])(ck\d{3})(?!\d)", t)
    if not m:
        m = re.search(r"(?i)(ck\d{3})", t)
        if not m:
            return None
    return m.group(1).upper()


def extract_kingdom_id(text: str) -> Optional[str]:
    """텍스트에서 왕국 ID 추출 (예: KD001)"""
    t = text or ""
    m = re.search(r"(?i)(?<![0-9a-zA-Z])(kd\d{3})(?!\d)", t)
    if not m:
        m = re.search(r"(?i)(kd\d{3})", t)
        if not m:
            return None
    return m.group(1).upper()


def _norm_key(s: Any) -> str:
    """문자열 정규화 (공백 제거, 소문자 변환)"""
    return re.sub(r"\s+", "", safe_str(s)).lower().strip()


def extract_cookie_grade_from_text(user_text: str) -> Optional[str]:
    """텍스트에서 쿠키 등급 추출"""
    txt = safe_str(user_text).strip().lower()

    grades = {
        "커먼": "커먼",
        "레어": "레어",
        "슈퍼레어": "슈퍼레어",
        "에픽": "에픽",
        "레전더리": "레전더리",
        "에인션트": "에인션트",
        "common": "커먼",
        "rare": "레어",
        "super rare": "슈퍼레어",
        "epic": "에픽",
        "legendary": "레전더리",
        "ancient": "에인션트",
    }

    for key, value in grades.items():
        if key in txt:
            return value
    return None


def extract_language_from_text(user_text: str) -> Optional[str]:
    """텍스트에서 번역 대상 언어 추출"""
    txt = safe_str(user_text).strip().lower()

    lang_map = {
        "영어": "en",
        "english": "en",
        "일본어": "ja",
        "japanese": "ja",
        "중국어": "zh",
        "chinese": "zh",
        "간체": "zh",
        "번체": "zh-TW",
        "태국어": "th",
        "thai": "th",
        "인도네시아어": "id",
        "indonesian": "id",
        "독일어": "de",
        "german": "de",
        "프랑스어": "fr",
        "french": "fr",
        "스페인어": "es",
        "spanish": "es",
        "포르투갈어": "pt",
        "portuguese": "pt",
    }

    for key, value in lang_map.items():
        if key in txt:
            return value
    return None
