"""
agent/tool_schemas.py - LLM Tool Calling을 위한 쿠키런 AI 도구 정의
===================================================================
데브시스터즈 기술혁신 프로젝트

LangChain @tool 데코레이터를 사용하여 LLM이 호출할 수 있는 도구들을 정의합니다.
"""
from typing import Optional
from langchain_core.tools import tool

from agent.tools import (
    tool_get_cookie_info,
    tool_list_cookies,
    tool_get_cookie_skill,
    tool_get_kingdom_info,
    tool_list_kingdoms,
    tool_translate_text,
    tool_check_translation_quality,
    tool_get_worldview_terms,
    tool_analyze_user,
    tool_get_user_segment,
    tool_detect_user_anomaly,
    tool_get_segment_statistics,
    tool_get_anomaly_statistics,
    tool_get_event_statistics,
    tool_get_user_activity_report,
    tool_classify_text,
    tool_get_translation_statistics,
    tool_get_dashboard_summary,
)
from rag.service import tool_rag_search
import state as st


# ============================================================
# 쿠키 캐릭터 도구
# ============================================================
@tool
def get_cookie_info(cookie_id: str) -> dict:
    """
    특정 쿠키 캐릭터의 상세 정보를 조회합니다.

    Args:
        cookie_id: 쿠키 ID 또는 이름 (예: CK001, 용감한 쿠키)

    Returns:
        쿠키의 이름, 등급, 타입, 스토리 등 상세 정보
    """
    return tool_get_cookie_info(cookie_id)


@tool
def list_cookies(grade: Optional[str] = None, cookie_type: Optional[str] = None) -> dict:
    """
    쿠키 캐릭터 목록을 조회합니다. 등급이나 타입으로 필터링할 수 있습니다.

    Args:
        grade: 등급 필터 (예: 커먼, 레어, 에픽, 레전더리, 에인션트)
        cookie_type: 타입 필터 (예: 돌격, 마법, 사격, 방어, 지원, 치유)

    Returns:
        필터링된 쿠키 목록
    """
    return tool_list_cookies(grade=grade, cookie_type=cookie_type)


@tool
def get_cookie_skill(cookie_id: str) -> dict:
    """
    특정 쿠키의 스킬 정보를 조회합니다.

    Args:
        cookie_id: 쿠키 ID (예: CK001)

    Returns:
        스킬 이름, 설명 (한국어/영어)
    """
    return tool_get_cookie_skill(cookie_id)


# ============================================================
# 왕국/지역 도구
# ============================================================
@tool
def get_kingdom_info(kingdom_id: str) -> dict:
    """
    특정 왕국/지역의 상세 정보를 조회합니다.

    Args:
        kingdom_id: 왕국 ID 또는 이름 (예: KD001, 쿠키 왕국)

    Returns:
        왕국의 이름, 설명 (한국어/영어)
    """
    return tool_get_kingdom_info(kingdom_id)


@tool
def list_kingdoms() -> dict:
    """
    모든 왕국/지역 목록을 조회합니다.

    Returns:
        왕국 목록과 기본 정보
    """
    return tool_list_kingdoms()


# ============================================================
# 번역 도구
# ============================================================
@tool
def translate_text(
    text: str,
    target_lang: str,
    category: str = "dialog",
    preserve_terms: bool = True
) -> dict:
    """
    쿠키런 세계관에 맞춰 텍스트를 번역합니다.
    세계관 고유 용어의 일관성을 유지합니다.

    Args:
        text: 번역할 한국어 텍스트
        target_lang: 번역 대상 언어 (en, ja, zh, zh-TW, th, id, de, fr, es, pt)
        category: 텍스트 카테고리 (UI, story, skill, dialog, item, quest 등)
        preserve_terms: 세계관 용어 보존 여부 (기본값: True)

    Returns:
        번역 컨텍스트와 감지된 세계관 용어
    """
    return tool_translate_text(text, target_lang, category, preserve_terms)


@tool
def check_translation_quality(
    source_text: str,
    translated_text: str,
    target_lang: str,
    category: str = "dialog"
) -> dict:
    """
    번역 품질을 평가합니다.

    Args:
        source_text: 원문 텍스트
        translated_text: 번역된 텍스트
        target_lang: 번역 대상 언어
        category: 텍스트 카테고리

    Returns:
        품질 등급 (excellent/good/acceptable/needs_review), 신뢰도, 권장사항
    """
    return tool_check_translation_quality(source_text, translated_text, target_lang, category)


@tool
def get_worldview_terms(target_lang: Optional[str] = None) -> dict:
    """
    쿠키런 세계관 용어집을 조회합니다.

    Args:
        target_lang: 특정 언어 번역만 조회 (선택사항)

    Returns:
        세계관 용어와 번역 정보
    """
    return tool_get_worldview_terms(target_lang)


@tool
def get_translation_statistics() -> dict:
    """
    번역 데이터 통계를 조회합니다.

    Returns:
        언어별/카테고리별 번역 통계, 품질 분포
    """
    return tool_get_translation_statistics()


# ============================================================
# 유저 분석 도구
# ============================================================
@tool
def analyze_user(user_id: str) -> dict:
    """
    특정 유저의 행동 패턴을 분석합니다.

    Args:
        user_id: 유저 ID (예: U0001)

    Returns:
        유저 세그먼트, 행동 지표, 이상 여부
    """
    return tool_analyze_user(user_id)


@tool
def get_user_segment(user_features: dict) -> dict:
    """
    유저 피처를 기반으로 세그먼트를 분류합니다.

    Args:
        user_features: 유저 행동 피처 (total_events, stage_clears, gacha_pulls, pvp_battles, purchases, vip_level)

    Returns:
        세그먼트 분류 결과 (캐주얼/하드코어/PvP전문가/콘텐츠수집가/신규)
    """
    return tool_get_user_segment(user_features)


@tool
def detect_user_anomaly(user_features: dict) -> dict:
    """
    유저 행동의 이상 여부를 탐지합니다.

    Args:
        user_features: 유저 행동 피처

    Returns:
        이상 여부, 이상 점수, 위험 수준
    """
    return tool_detect_user_anomaly(user_features)


@tool
def get_segment_statistics() -> dict:
    """
    유저 세그먼트별 통계를 조회합니다.

    Returns:
        세그먼트별 유저 수, 평균 활동량, 이상 비율
    """
    return tool_get_segment_statistics()


@tool
def get_anomaly_statistics() -> dict:
    """
    전체 이상 행동 유저 통계를 조회합니다.

    Returns:
        이상 유저 수, 이상 비율, 세그먼트별 이상 분포, 이상 유저 샘플
    """
    return tool_get_anomaly_statistics()


@tool
def get_user_activity_report(user_id: str, days: int = 30) -> dict:
    """
    특정 유저의 활동 리포트를 생성합니다.

    Args:
        user_id: 유저 ID
        days: 조회할 기간 (기본값: 30일)

    Returns:
        활동 요약, 이벤트 타입별 집계
    """
    return tool_get_user_activity_report(user_id, days)


# ============================================================
# 게임 이벤트 도구
# ============================================================
@tool
def get_event_statistics(event_type: Optional[str] = None, days: int = 30) -> dict:
    """
    게임 이벤트 통계를 조회합니다.

    Args:
        event_type: 이벤트 타입 필터 (stage_clear, gacha, pvp, purchase 등)
        days: 조회할 기간 (기본값: 30일)

    Returns:
        이벤트 타입별 집계, 일별 추이
    """
    return tool_get_event_statistics(event_type, days)


# ============================================================
# 텍스트 분류 도구
# ============================================================
@tool
def classify_text(text: str) -> dict:
    """
    텍스트의 카테고리를 분류합니다.

    Args:
        text: 분류할 텍스트

    Returns:
        예측 카테고리, 신뢰도, 상위 3개 카테고리
    """
    return tool_classify_text(text)


# ============================================================
# RAG 검색 도구
# ============================================================
@tool
def search_worldview(query: str, top_k: int = 5) -> dict:
    """
    쿠키런 세계관 지식베이스를 검색합니다.
    캐릭터, 스토리, 용어, 설정 등에 대한 정보를 검색할 때 사용합니다.

    사용 예시:
    - "소울잼이란 무엇인가?" → query="소울잼 정의 설명"
    - "에인션트 쿠키 종류" → query="에인션트 쿠키 목록"
    - "다크엔챈트리스 스토리" → query="다크엔챈트리스 쿠키 배경"

    Args:
        query: 검색 질의
        top_k: 검색 결과 개수 (기본값: 5)

    Returns:
        관련 문서 스니펫과 출처
    """
    return tool_rag_search(query, st.OPENAI_API_KEY, top_k)


# ============================================================
# 대시보드 도구
# ============================================================
@tool
def get_dashboard_summary() -> dict:
    """
    대시보드 요약 정보를 조회합니다.

    Returns:
        쿠키/유저/번역/게임로그 통계 요약
    """
    return tool_get_dashboard_summary()


# ============================================================
# 모든 도구 리스트 (LLM에 바인딩할 때 사용)
# ============================================================
ALL_TOOLS = [
    # 쿠키 정보
    get_cookie_info,
    list_cookies,
    get_cookie_skill,
    # 왕국 정보
    get_kingdom_info,
    list_kingdoms,
    # 번역
    translate_text,
    check_translation_quality,
    get_worldview_terms,
    get_translation_statistics,
    # 유저 분석
    analyze_user,
    get_user_segment,
    detect_user_anomaly,
    get_segment_statistics,
    get_anomaly_statistics,
    get_user_activity_report,
    # 게임 이벤트
    get_event_statistics,
    # 텍스트 분류
    classify_text,
    # RAG 검색
    search_worldview,
    # 대시보드
    get_dashboard_summary,
]
