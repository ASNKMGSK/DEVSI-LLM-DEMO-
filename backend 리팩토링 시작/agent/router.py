"""
agent/router.py - LLM Router (의도 분류 전용)
============================================================
GPT가 제안한 "LLM Router" 패턴 구현

질문을 먼저 분류한 뒤, 해당 카테고리의 도구만 Executor에 노출합니다.

분류 우선순위:
1. 키워드 기반 분류 (가장 빠름, 비용 없음)
2. Semantic Router (임베딩 유사도, LLM 호출 없음)
3. LLM Router (gpt-4o-mini, fallback)

References:
- https://www.anthropic.com/research/building-effective-agents
- https://github.com/aurelio-labs/semantic-router
"""
import re
from typing import Literal, Optional, Tuple
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from core.utils import safe_str
import state as st


# ============================================================
# 카테고리 정의
# ============================================================
class IntentCategory(str, Enum):
    """질문 의도 카테고리"""
    ANALYSIS = "analysis"       # 매출, 이탈, DAU, 코호트, 트렌드
    WORLDVIEW = "worldview"     # 세계관, 스토리, 용어, 설정
    COOKIE = "cookie"           # 쿠키 정보, 스킬, 승률
    USER = "user"               # 유저 분석, 세그먼트, 이상 탐지
    TRANSLATE = "translate"     # 번역, 품질 검사
    DASHBOARD = "dashboard"     # 대시보드, 전체 현황
    GENERAL = "general"         # 일반 대화, 인사


# 카테고리별 도구 매핑
CATEGORY_TOOLS = {
    IntentCategory.ANALYSIS: [
        "get_churn_prediction",
        "get_revenue_prediction",
        "get_trend_analysis",
        "get_cohort_analysis",
    ],
    IntentCategory.WORLDVIEW: [
        "search_worldview",
        "search_worldview_lightrag",
    ],
    IntentCategory.COOKIE: [
        "get_cookie_info",
        "list_cookies",
        "get_cookie_skill",
        "get_cookie_win_rate",
        "predict_cookie_win_rate",
        "optimize_investment",
        "search_worldview",           # 쿠키 정보도 RAG에서 검색 (DB에 없는 쿠키 지원)
        "search_worldview_lightrag",  # LightRAG 검색
    ],
    IntentCategory.USER: [
        "analyze_user",
        "predict_user_churn",
        "get_user_segment",
        "detect_user_anomaly",
        "get_segment_statistics",
    ],
    IntentCategory.TRANSLATE: [
        "translate_text",
        "check_translation_quality",
        "get_worldview_terms",
        "get_translation_statistics",
    ],
    IntentCategory.DASHBOARD: [
        "get_dashboard_summary",
        "get_segment_statistics",
        "get_translation_statistics",
        "get_event_statistics",
    ],
    IntentCategory.GENERAL: [],  # 도구 없이 대화만
}


# ============================================================
# 키워드 기반 빠른 분류 (LLM 호출 없이)
# ============================================================
_ANALYSIS_KEYWORDS = [
    "매출", "수익", "revenue", "arpu", "arppu", "과금", "성장률",
    "이탈", "churn", "고위험", "중위험", "저위험", "이탈률", "이탈 요인",
    "코호트", "cohort", "리텐션", "retention", "잔존", "week1", "week4",
    "트렌드", "trend", "kpi", "dau", "mau", "wau", "지표", "변화율",
]

_WORLDVIEW_KEYWORDS = [
    # 핵심 세계관 키워드 (RAG 검색 필수)
    "세계관", "스토리", "역사", "배경", "설정", "용어", "뜻", "의미",
    "시대적", "시대 배경", "기원", "탄생", "창조", "전설", "신화",
    # 주요 엔티티
    "소울잼", "에인션트", "다크카카오", "홀리베리", "순수바닐라", "마녀",
    "왕국 역사", "쿠키런 세계", "마법", "저주", "오븐", "마녀의",
    # 질문 패턴 (세계관 지식 요청)
    "뭐야", "무엇", "어떤", "알려줘", "설명", "어떻게 됐", "왜 그런",
]

_COOKIE_KEYWORDS = [
    "쿠키 정보", "쿠키 스킬", "쿠키 능력", "쿠키 스탯",
    "승률", "pvp", "win rate", "전투",
    "투자", "육성", "레벨업", "추천",
    "등급", "에픽", "레전더리", "커먼", "레어",
]

_USER_KEYWORDS = [
    "유저 분석", "유저 정보", "사용자 분석", "플레이어",
    "세그먼트", "군집", "분류",
    "이상 탐지", "어뷰징", "봇", "비정상",
    "U0", "U1", "U2", "U3", "U4", "U5", "U6", "U7", "U8", "U9",  # 유저 ID 패턴
]

_TRANSLATE_KEYWORDS = [
    "번역", "translate", "영어로", "일본어로", "중국어로",
    "번역 품질", "품질 검사", "용어집",
]

_DASHBOARD_KEYWORDS = [
    "대시보드", "dashboard", "전체 현황", "요약", "통계",
    "현황", "유저 활동", "활동 현황",
]

_GENERAL_KEYWORDS = [
    "안녕", "하이", "헬로", "hi", "hello",
    "고마워", "감사", "thanks",
    "뭐해", "누구", "자기소개",
]


def _keyword_classify(text: str) -> Optional[IntentCategory]:
    """
    키워드 기반 빠른 분류 (LLM 호출 없이)

    Returns:
        IntentCategory or None (불확실한 경우)
    """
    t = text.lower()

    # 우선순위: 분석 > 유저 > 쿠키 > 세계관 > 번역 > 대시보드 > 일반

    # 1. 분석 키워드 (가장 높은 우선순위)
    if any(kw in t for kw in _ANALYSIS_KEYWORDS):
        return IntentCategory.ANALYSIS

    # 2. 유저 분석 키워드
    if any(kw in t for kw in _USER_KEYWORDS):
        # 유저 ID 패턴 체크 (U0001 등)
        if re.search(r'U\d{4,6}', text, re.IGNORECASE):
            return IntentCategory.USER
        if any(kw in t for kw in ["유저", "사용자", "플레이어", "세그먼트", "이상"]):
            return IntentCategory.USER

    # 3. 쿠키 관련 (승률, 정보, 스킬)
    if any(kw in t for kw in _COOKIE_KEYWORDS):
        return IntentCategory.COOKIE

    # 4. 번역 관련
    if any(kw in t for kw in _TRANSLATE_KEYWORDS):
        return IntentCategory.TRANSLATE

    # 5. 대시보드 관련
    if any(kw in t for kw in _DASHBOARD_KEYWORDS):
        return IntentCategory.DASHBOARD

    # 6. 세계관 관련 (스토리, 용어)
    if any(kw in t for kw in _WORLDVIEW_KEYWORDS):
        return IntentCategory.WORLDVIEW

    # 7. 일반 대화
    if any(kw in t for kw in _GENERAL_KEYWORDS):
        return IntentCategory.GENERAL

    # 불확실한 경우 None 반환 → LLM Router 사용
    return None


# ============================================================
# LLM Router (분류 전용)
# ============================================================
ROUTER_SYSTEM_PROMPT = """당신은 질문 분류 전문가입니다.
사용자 질문을 분석하여 **정확히 하나의 카테고리**를 반환하세요.

## 카테고리 정의

| 카테고리 | 설명 | 예시 질문 |
|----------|------|----------|
| analysis | 매출, 이탈, DAU, 코호트, 트렌드, KPI | "매출 성장률", "이탈 현황", "DAU 알려줘" |
| worldview | 세계관, 스토리, 용어, 설정, 캐릭터 배경 | "소울잼이 뭐야?", "다크카카오 역사" |
| cookie | 쿠키 정보, 스킬, 승률, 투자 추천 | "용감한 쿠키 정보", "CK001 승률" |
| user | 특정 유저 분석, 세그먼트, 이상 탐지 | "U0001 분석", "세그먼트 통계" |
| translate | 번역, 품질 검사, 용어집 | "영어로 번역해줘", "번역 품질" |
| dashboard | 대시보드, 전체 현황, 요약 통계 | "대시보드 보여줘", "전체 현황" |
| general | 일반 대화, 인사, 도움말 | "안녕", "뭐해?" |

## 규칙

1. **반드시 하나의 카테고리만** 반환
2. 복합 질문은 **핵심 의도** 기준으로 분류
3. 숫자/통계 관련 → analysis 우선
4. 세계관 지식 → worldview
5. 특정 쿠키 → cookie
6. 특정 유저 (U0001 등) → user

## 출력 형식

카테고리명만 반환하세요. 예: analysis"""


async def route_intent_llm(
    text: str,
    api_key: str,
    model: str = "gpt-4o-mini",  # 빠르고 저렴한 모델 사용
) -> IntentCategory:
    """
    LLM을 사용한 의도 분류 (키워드 분류 실패 시)

    Args:
        text: 사용자 질문
        api_key: OpenAI API 키
        model: 사용할 모델 (기본: gpt-4o-mini)

    Returns:
        IntentCategory
    """
    try:
        llm = ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            temperature=0,  # 결정론적 분류
            max_tokens=20,  # 카테고리명만 반환
        )

        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=f"질문: {text}"),
        ]

        response = llm.invoke(messages)
        category_str = safe_str(response.content).strip().lower()

        # 카테고리 매핑
        category_map = {
            "analysis": IntentCategory.ANALYSIS,
            "worldview": IntentCategory.WORLDVIEW,
            "cookie": IntentCategory.COOKIE,
            "user": IntentCategory.USER,
            "translate": IntentCategory.TRANSLATE,
            "dashboard": IntentCategory.DASHBOARD,
            "general": IntentCategory.GENERAL,
        }

        category = category_map.get(category_str, IntentCategory.GENERAL)

        st.logger.info(
            "ROUTER_LLM_CLASSIFY query=%s result=%s",
            text[:50], category.value,
        )

        return category

    except Exception as e:
        st.logger.warning("ROUTER_LLM_FAIL err=%s, fallback=general", safe_str(e))
        return IntentCategory.GENERAL


def route_intent_sync(
    text: str,
    api_key: str,
    model: str = "gpt-4o-mini",
) -> IntentCategory:
    """
    동기 버전의 의도 분류

    1. 키워드 기반 빠른 분류 시도
    2. 실패 시 LLM Router 사용
    """
    # 1단계: 키워드 기반 빠른 분류
    category = _keyword_classify(text)

    if category is not None:
        st.logger.info(
            "ROUTER_KEYWORD_CLASSIFY query=%s result=%s",
            text[:50], category.value,
        )
        return category

    # 2단계: LLM Router (키워드 분류 실패 시)
    # 동기 환경에서는 asyncio.run 사용
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 이미 이벤트 루프가 실행 중이면 새 스레드에서 실행
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    route_intent_llm(text, api_key, model)
                )
                return future.result(timeout=10)
        else:
            return loop.run_until_complete(route_intent_llm(text, api_key, model))
    except Exception:
        return asyncio.run(route_intent_llm(text, api_key, model))


def get_tools_for_category(category: IntentCategory) -> list[str]:
    """카테고리에 해당하는 도구 이름 목록 반환"""
    return CATEGORY_TOOLS.get(category, [])


# ============================================================
# 편의 함수
# ============================================================
def classify_and_get_tools(
    text: str,
    api_key: str,
    use_llm_fallback: bool = True,
    **kwargs,  # 하위 호환성 (use_semantic_router 등 무시)
) -> tuple[IntentCategory, list[str]]:
    """
    질문 분류 및 도구 목록 반환 (원스톱)

    분류 순서:
    1. 키워드 기반 분류 (가장 빠름, 비용 없음)
    2. LLM Router (gpt-4o-mini, fallback)

    Args:
        text: 사용자 질문
        api_key: OpenAI API 키
        use_llm_fallback: 키워드 분류 실패 시 LLM 사용 여부

    Returns:
        (카테고리, 도구 이름 목록)
    """
    # 1단계: 키워드 기반 분류 (가장 빠름)
    category = _keyword_classify(text)

    if category is not None:
        st.logger.info(
            "ROUTER_KEYWORD query=%s category=%s",
            text[:40], category.value,
        )
        tools = get_tools_for_category(category)
        return category, tools

    # 2단계: LLM Router (키워드 분류 실패 시 fallback)
    if use_llm_fallback and api_key:
        category = route_intent_sync(text, api_key)
        st.logger.info(
            "ROUTER_LLM query=%s category=%s",
            text[:40], category.value,
        )
    else:
        # 기본값: GENERAL
        category = IntentCategory.GENERAL
        st.logger.info(
            "ROUTER_DEFAULT query=%s category=general",
            text[:40],
        )

    tools = get_tools_for_category(category)
    return category, tools
