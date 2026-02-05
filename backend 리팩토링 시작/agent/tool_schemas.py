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
    # ML 모델 예측 도구
    tool_predict_user_churn,
    tool_predict_cookie_win_rate,
    tool_get_cookie_win_rate,
    tool_optimize_investment,
    # 분석 도구
    tool_get_churn_prediction,
    tool_get_cohort_analysis,
    tool_get_trend_analysis,
    tool_get_revenue_prediction,
)
from rag.service import tool_rag_search
from rag.light_rag import lightrag_search_sync, get_lightrag_status
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
    ★ 특정 유저 분석 질문은 이 도구를 사용하세요! ★

    특정 유저의 행동 패턴을 분석합니다.

    ✅ 이 도구를 사용해야 하는 질문:
    - "U0001 유저 분석해줘", "이 유저 정보 알려줘"
    - "U100211 유저 세그먼트가 뭐야?"
    - "특정 유저 행동 패턴"

    Args:
        user_id: 유저 ID (예: U0001, U100211)

    Returns:
        유저 세그먼트, 행동 지표(이벤트 수, 스테이지, 가챠, PvP, 구매), 이상 여부
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
    쿠키런 세계관/스토리/용어/설정에 대한 '지식' 질문에만 사용합니다.
    (기본 임베딩 검색, LightRAG보다 가벼움)

    ⚠️ 사용 금지 케이스 (다른 도구 사용):
    - 유저/이탈 분석 → get_churn_prediction, analyze_user
    - 매출/수익 → get_revenue_prediction
    - 코호트/리텐션 → get_cohort_analysis
    - DAU/KPI → get_trend_analysis
    - 쿠키 승률/스탯 → get_cookie_win_rate

    ✅ 사용해야 하는 케이스:
    - "소울잼이란?", "에인션트 쿠키가 뭐야?"
    - "다크엔챈트리스 스토리", "마녀의 오븐이란?"

    Args:
        query: 검색 질의 (세계관/스토리/용어 관련)
        top_k: 검색 결과 개수 (기본값: 5)

    Returns:
        관련 문서 스니펫과 출처
    """
    return tool_rag_search(query, top_k=top_k, api_key=st.OPENAI_API_KEY)


@tool
def search_worldview_lightrag(query: str, mode: str = "hybrid") -> dict:
    """
    쿠키런 세계관/스토리/캐릭터 배경에 대한 '지식' 질문에만 사용합니다.

    ⚠️ 사용 금지 케이스 (다른 도구 사용):
    - 유저 통계/이탈 예측 → get_churn_prediction 사용
    - 매출/수익 분석 → get_revenue_prediction 사용
    - 코호트/리텐션 → get_cohort_analysis 사용
    - DAU/KPI 트렌드 → get_trend_analysis 사용
    - 특정 유저 분석 → analyze_user, predict_user_churn 사용
    - 쿠키 승률/스탯 → get_cookie_win_rate 사용

    ✅ 사용해야 하는 케이스:
    - "소울잼이란?", "에인션트 쿠키가 뭐야?"
    - "다크카카오 왕국 역사", "쿠키런 세계관 배경"
    - "용감한 쿠키 스토리", "마녀의 저주란?"

    검색 모드:
    - "local": 구체적인 엔티티/캐릭터 중심 (예: "용감한 쿠키 스킬")
    - "global": 추상적인 테마/개념 중심 (예: "쿠키런 세계관 역사")
    - "hybrid": local + global 조합 (권장, 기본값)

    Args:
        query: 검색 질의 (세계관/스토리/캐릭터 관련)
        mode: 검색 모드 ("local", "global", "hybrid", "naive")

    Returns:
        관련 문서 및 엔티티 정보
    """
    # LightRAG 상태 확인
    status = get_lightrag_status()
    if not status.get("ready"):
        return {
            "status": "FAILED",
            "error": "LightRAG가 준비되지 않았습니다. 먼저 인덱스를 빌드해주세요.",
            "lightrag_status": status
        }

    return lightrag_search_sync(query, mode=mode)  # top_k는 state.LIGHTRAG_CONFIG에서 관리


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
# ML 모델 예측 도구
# ============================================================
@tool
def predict_user_churn(user_id: str) -> dict:
    """
    특정 유저의 이탈 확률을 예측합니다.
    ML 모델(LightGBM)과 SHAP Explainer를 사용하여 예측 및 주요 이탈 요인을 분석합니다.

    사용 예시:
    - "U0001 유저 이탈 예측해줘" → predict_user_churn(user_id="U0001")
    - "이 유저가 이탈할 확률은?" → predict_user_churn(user_id="U0001")

    Args:
        user_id: 유저 ID (예: U0001)

    Returns:
        이탈 확률(%), 위험 수준(HIGH/MEDIUM/LOW), 주요 이탈 요인, 권장 조치
    """
    return tool_predict_user_churn(user_id)


@tool
def predict_cookie_win_rate(cookie_stats: dict) -> dict:
    """
    쿠키의 스탯을 기반으로 PvP 승률을 예측합니다.
    LightGBM 회귀 모델을 사용합니다.

    사용 예시:
    - 특정 스탯 조합의 예상 승률 계산
    - 스탯 변경 시 승률 변화 시뮬레이션

    Args:
        cookie_stats: 쿠키 스탯 딕셔너리
            - atk: 공격력
            - hp: 체력
            - def: 방어력
            - skill_dmg: 스킬 데미지
            - cooldown: 쿨다운
            - crit_rate: 치명타 확률
            - crit_dmg: 치명타 데미지

    Returns:
        예측 승률(%), 티어, 스탯 분석
    """
    return tool_predict_cookie_win_rate(cookie_stats)


@tool
def get_cookie_win_rate(cookie_id: str) -> dict:
    """
    특정 쿠키의 현재 스탯을 기반으로 승률을 예측합니다.
    cookie_stats.csv 데이터와 ML 모델을 결합하여 분석합니다.

    사용 예시:
    - "용감한 쿠키 승률 어때?" → get_cookie_win_rate(cookie_id="용감한 쿠키")
    - "CK001 쿠키 승률 예측해줘" → get_cookie_win_rate(cookie_id="CK001")
    - "에픽 쿠키 중 승률 높은 쿠키는?" → 여러 쿠키 조회 후 비교

    Args:
        cookie_id: 쿠키 ID 또는 이름 (예: CK001, 용감한 쿠키)

    Returns:
        쿠키 정보, 실제 승률, 예측 승률, 티어, 상세 스탯
    """
    return tool_get_cookie_win_rate(cookie_id)


@tool
def optimize_investment(
    user_id: str,
    goal: str = "maximize_win_rate",
    resources: Optional[dict] = None
) -> dict:
    """
    유저의 자원을 분석하여 최적의 쿠키 투자 전략을 제안합니다.
    P-PSO(Phasor Particle Swarm Optimization) 알고리즘을 사용합니다.

    사용 예시:
    - "내 쿠키 어디에 투자하면 좋을까?" → optimize_investment(user_id="U000001")
    - "승률 최대화 투자 추천해줘" → optimize_investment(user_id="U000001", goal="maximize_win_rate")
    - "효율적인 투자 방법 알려줘" → optimize_investment(user_id="U000001", goal="maximize_efficiency")

    Args:
        user_id: 유저 ID (예: U000001)
        goal: 최적화 목표
            - maximize_win_rate: 승률 최대화 (기본값)
            - maximize_efficiency: 투자 효율 최대화
            - balanced: 균형 잡힌 투자
        resources: 보유 자원 (선택사항, 없으면 DB에서 조회)
            - exp_jelly: 경험치 젤리
            - coin: 코인
            - skill_powder: 스킬 파우더
            - soul_stone: 소울스톤

    Returns:
        투자 추천 목록 (최대 10개), 예상 총 승률 증가, 필요 자원, 자원 사용률
    """
    return tool_optimize_investment(user_id, goal, resources)


# ============================================================
# 분석 도구 (Analysis Tools)
# ============================================================
@tool
def get_churn_prediction(risk_level: str = None, limit: int = None) -> dict:
    """
    ★ 유저 이탈 관련 질문은 이 도구를 사용하세요! ★

    전체 유저의 이탈 예측 분석을 조회합니다.
    고위험/중위험/저위험 이탈 유저 수와 주요 이탈 요인을 반환합니다.

    ✅ 이 도구를 사용해야 하는 질문:
    - "이탈 유저 현황", "이탈 예측 분석"
    - "고위험 유저 몇 명이야?", "중위험 이탈 유저"
    - "이탈률 알려줘", "이탈 요인이 뭐야?"
    - "어떤 유저가 이탈할 것 같아?"

    Args:
        risk_level: 특정 위험 등급만 필터 ("high", "medium", "low")
            - "high": 고위험 유저만 조회
            - "medium": 중위험 유저만 조회
            - "low": 저위험 유저만 조회
        limit: 상세 유저 목록 반환 시 최대 개수 (기본값: 10)

    사용 예시:
    - "이탈 예측 분석 보여줘" → get_churn_prediction()
    - "고위험 이탈 유저 목록" → get_churn_prediction(risk_level="high")
    - "중위험 이탈 유저 현황" → get_churn_prediction(risk_level="medium")

    Returns:
        고위험/중위험/저위험 유저 수, 예상 이탈률, 주요 이탈 요인 5개, 인사이트
    """
    return tool_get_churn_prediction(risk_level=risk_level, limit=limit)


@tool
def get_cohort_analysis(cohort: str = None, month: str = None) -> dict:
    """
    ★ 코호트/리텐션 관련 질문은 이 도구를 사용하세요! ★

    코호트 리텐션 분석을 조회합니다.
    주간별 리텐션율과 코호트별 잔존율을 반환합니다.

    ✅ 이 도구를 사용해야 하는 질문:
    - "코호트 분석 보여줘", "리텐션 현황 알려줘"
    - "Week1 리텐션 얼마야?", "Week4 잔존율은?"
    - "1월 코호트 분석", "11월 리텐션 현황"
    - "유저 잔존율 분석해줘"

    Args:
        cohort: 특정 코호트명 필터 (예: "2024-11 W1", "2025-01 W2")
        month: 특정 월 필터 (예: "2024-11", "2025-01")

    Returns:
        주별 코호트 리텐션 테이블, 평균 Week1/Week4 리텐션, 인사이트
    """
    return tool_get_cohort_analysis(cohort=cohort, month=month)


@tool
def get_trend_analysis(start_date: str = None, end_date: str = None, days: int = None) -> dict:
    """
    ★ DAU/KPI/트렌드 관련 질문은 이 도구를 사용하세요! ★

    트렌드 KPI 분석을 조회합니다.
    주요 지표(DAU, ARPU, 신규가입, 세션시간 등)의 변화율과 상관관계를 반환합니다.

    ✅ 이 도구를 사용해야 하는 질문:
    - "DAU 얼마야?", "오늘 활성 유저 수"
    - "트렌드 분석 보여줘", "KPI 변화율"
    - "신규 가입자 수", "세션 시간 분석"
    - "지표 변화 추이", "최근 7일 DAU"

    Args:
        start_date: 시작 날짜 (YYYY-MM-DD 형식)
        end_date: 종료 날짜 (YYYY-MM-DD 형식)
        days: 최근 N일 분석 (기본값: 7일)

    Returns:
        KPI별 현재/이전 값, 변화율, 주요 상관관계, 인사이트
    """
    return tool_get_trend_analysis(start_date=start_date, end_date=end_date, days=days)


@tool
def get_revenue_prediction(days: int = None, start_date: str = None, end_date: str = None) -> dict:
    """
    ★ 매출/수익 관련 질문은 이 도구를 사용하세요! ★

    매출 예측 분석을 조회합니다.
    예상 매출, ARPU/ARPPU, 과금 유저 분포를 반환합니다.

    ✅ 이 도구를 사용해야 하는 질문:
    - "매출 예측 보여줘", "이번 달 예상 매출은?"
    - "ARPU 얼마야?", "ARPPU 분석해줘"
    - "고래 유저 몇 명?", "과금 유저 분포"
    - "매출 성장률은?", "수익 분석해줘"

    Args:
        days: 최근 N일 기준 분석 (기본값: 30일)
        start_date: 시작 날짜 (YYYY-MM-DD 형식)
        end_date: 종료 날짜 (YYYY-MM-DD 형식)

    Returns:
        예상 월매출, 성장률, ARPU/ARPPU, whale/dolphin/minnow 분포, 인사이트
    """
    return tool_get_revenue_prediction(days=days, start_date=start_date, end_date=end_date)


# ============================================================
# 에이전트별 도구 분류 (Multi-Agent용)
# ============================================================

# 검색 에이전트 도구: 세계관 정보 검색 (쿠키, 왕국, RAG)
SEARCH_AGENT_TOOLS = [
    get_cookie_info,
    list_cookies,
    get_cookie_skill,
    get_kingdom_info,
    list_kingdoms,
    search_worldview,
    search_worldview_lightrag,
]

# 분석 에이전트 도구: 유저 분석, ML 예측, 통계
ANALYSIS_AGENT_TOOLS = [
    analyze_user,
    get_user_segment,
    detect_user_anomaly,
    get_segment_statistics,
    get_anomaly_statistics,
    get_user_activity_report,
    get_event_statistics,
    get_dashboard_summary,
    # ML 예측
    predict_user_churn,
    predict_cookie_win_rate,
    get_cookie_win_rate,
    optimize_investment,
    # 분석
    get_churn_prediction,
    get_cohort_analysis,
    get_trend_analysis,
    get_revenue_prediction,
]

# 번역 에이전트 도구: 번역, 품질 평가, 텍스트 분류
TRANSLATION_AGENT_TOOLS = [
    translate_text,
    check_translation_quality,
    get_worldview_terms,
    get_translation_statistics,
    classify_text,
]

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
    search_worldview_lightrag,
    # 대시보드
    get_dashboard_summary,
    # ML 모델 예측
    predict_user_churn,
    predict_cookie_win_rate,
    get_cookie_win_rate,
    optimize_investment,
    # 분석 도구
    get_churn_prediction,
    get_cohort_analysis,
    get_trend_analysis,
    get_revenue_prediction,
]
