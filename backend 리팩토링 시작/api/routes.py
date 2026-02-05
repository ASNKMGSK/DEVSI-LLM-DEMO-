"""
api/routes.py - 쿠키런 AI 플랫폼 FastAPI 라우트 정의
===================================================
데브시스터즈 기술혁신 프로젝트

주요 기능:
1. 쿠키런 세계관 데이터 API
2. LLM 기반 번역 서비스
3. 멀티 에이전트 대화
4. RAG 기반 지식 검색
"""
import os
import json
from datetime import datetime
from typing import Optional, List
from io import StringIO, BytesIO

import numpy as np
import pandas as pd
import joblib

from fastapi import APIRouter, HTTPException, Depends, status, Request, UploadFile, File, BackgroundTasks, Body, Query
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

try:
    import easyocr
    OCR_AVAILABLE = True
    OCR_READER = None  # Lazy loading
except ImportError:
    OCR_AVAILABLE = False
    OCR_READER = None

from core.constants import DEFAULT_SYSTEM_PROMPT, WORLDVIEW_TERMS, SUPPORTED_LANGUAGES
from core.utils import safe_str, safe_int, json_sanitize
from core.memory import clear_memory, append_memory
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
    tool_get_event_statistics,
    tool_get_user_activity_report,
    tool_classify_text,
    tool_search_worldview,
    tool_get_translation_statistics,
    tool_get_dashboard_summary,
    AVAILABLE_TOOLS,
)
from agent.llm import (
    build_langchain_messages, get_llm, chunk_text, pick_api_key,
)
from agent.runner import run_agent
from rag.service import (
    rag_build_or_load_index, tool_rag_search, _rag_list_files,
    rag_search_hybrid, BM25_AVAILABLE, RERANKER_AVAILABLE, KNOWLEDGE_GRAPH
)
from rag.light_rag import (
    lightrag_search_sync, lightrag_search_dual_sync,
    build_lightrag_from_rag_docs, get_lightrag_status, clear_lightrag,
    LIGHTRAG_AVAILABLE
)
from rag.k2rag import (
    k2rag_search_sync, index_documents as k2rag_index,
    get_status as k2rag_get_status, update_config as k2rag_update_config,
    load_from_existing_rag as k2rag_load_existing, summarize_text as k2rag_summarize
)
import state as st

router = APIRouter(prefix="/api")
security = HTTPBasic()


# ============================================================
# Pydantic 모델
# ============================================================
class LoginRequest(BaseModel):
    username: str
    password: str


class CookieRequest(BaseModel):
    cookie_id: str


class KingdomRequest(BaseModel):
    kingdom_id: str


class UserRequest(BaseModel):
    user_id: str


class TranslateRequest(BaseModel):
    text: str
    target_lang: str
    category: str = Field("dialog", description="텍스트 카테고리")
    preserve_terms: bool = Field(True, description="세계관 용어 보존 여부")


class TranslationQualityRequest(BaseModel):
    source_text: str
    translated_text: str
    target_lang: str
    category: str = Field("dialog")


class TextClassifyRequest(BaseModel):
    text: str


class RagRequest(BaseModel):
    query: str
    api_key: str = Field("", alias="apiKey")
    top_k: int = Field(st.RAG_DEFAULT_TOPK, alias="topK")
    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
        allow_population_by_alias = True


class AgentRequest(BaseModel):
    user_input: str = Field(..., alias="user_input")
    api_key: str = Field("", alias="apiKey")
    model: str = Field("gpt-4o-mini", alias="model")
    max_tokens: int = Field(8000, alias="maxTokens")
    system_prompt: str = Field(DEFAULT_SYSTEM_PROMPT, alias="systemPrompt")
    temperature: Optional[float] = Field(None, alias="temperature")
    top_p: Optional[float] = Field(None, alias="topP")
    presence_penalty: Optional[float] = Field(None, alias="presencePenalty")
    frequency_penalty: Optional[float] = Field(None, alias="frequencyPenalty")
    seed: Optional[int] = Field(None, alias="seed")
    timeout_ms: Optional[int] = Field(None, alias="timeoutMs")
    retries: Optional[int] = Field(None, alias="retries")
    stream: Optional[bool] = Field(None, alias="stream")
    rag_mode: str = Field("rag", alias="ragMode")  # 'rag' | 'lightrag' | 'k2rag' | 'auto'
    agent_mode: str = Field("single", alias="agentMode")  # 'single' | 'multi' (LangGraph)
    debug: bool = Field(True, alias="debug")
    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
        allow_population_by_alias = True


class UserCreateRequest(BaseModel):
    user_id: str
    name: str
    password: str
    role: str


class RagReloadRequest(BaseModel):
    api_key: str = Field("", alias="apiKey")
    force: bool = Field(True, alias="force")
    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
        allow_population_by_alias = True


class DeleteFileRequest(BaseModel):
    filename: str
    api_key: str = Field("", alias="apiKey")
    skip_reindex: bool = Field(False, alias="skipReindex")  # 다중 삭제 시 재빌드 건너뛰기
    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
        allow_population_by_alias = True


class HybridSearchRequest(BaseModel):
    """Hybrid Search 요청 모델"""
    query: str
    api_key: str = Field("", alias="apiKey")
    top_k: int = Field(5, alias="topK")
    use_reranking: bool = Field(True, alias="useReranking")
    use_kg: bool = Field(False, alias="useKg")
    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
        allow_population_by_alias = True


class LightRagSearchRequest(BaseModel):
    """LightRAG 검색 요청 모델"""
    query: str
    mode: str = Field("hybrid", description="검색 모드: naive, local, global, hybrid")
    top_k: int = Field(5, alias="topK")
    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
        allow_population_by_alias = True


class LightRagBuildRequest(BaseModel):
    """LightRAG 빌드 요청 모델"""
    force_rebuild: bool = Field(False, alias="forceRebuild")
    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
        allow_population_by_alias = True


class K2RagSearchRequest(BaseModel):
    """K2RAG 검색 요청 모델"""
    query: str
    top_k: int = Field(10, alias="topK")
    use_kg: bool = Field(True, alias="useKg", description="Knowledge Graph 사용")
    use_summary: bool = Field(True, alias="useSummary", description="요약 사용")
    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
        allow_population_by_alias = True


class K2RagConfigRequest(BaseModel):
    """K2RAG 설정 요청 모델"""
    hybrid_lambda: Optional[float] = Field(None, alias="hybridLambda", description="Hybrid 가중치 (0.0-1.0)")
    top_k: Optional[int] = Field(None, alias="topK")
    use_summarization: Optional[bool] = Field(None, alias="useSummarization")
    use_knowledge_graph: Optional[bool] = Field(None, alias="useKnowledgeGraph")
    llm_model: Optional[str] = Field(None, alias="llmModel")
    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
        allow_population_by_alias = True


class TeamOptimizeRequest(BaseModel):
    """팀 구성 최적화 요청 모델"""
    owned_cookies: Optional[List[str]] = Field(None, alias="ownedCookies", description="보유 쿠키 ID 리스트")
    required_roles: Optional[dict] = Field(None, alias="requiredRoles", description="필수 역할 (예: {'치유': 1})")
    max_iterations: int = Field(10, alias="maxIterations", description="PSO 최대 반복 횟수")
    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
        allow_population_by_alias = True


# ============================================================
# 인증
# ============================================================
def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    password = credentials.password
    if username not in st.USERS or st.USERS[username]["password"] != password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="인증 실패",
            headers={"WWW-Authenticate": "Basic"},
        )
    return {"username": username, "role": st.USERS[username]["role"], "name": st.USERS[username]["name"]}


# ============================================================
# 유틸
# ============================================================
def sse_pack(event: str, data: dict) -> str:
    """SSE 이벤트 포맷으로 직렬화 (LangChain 객체도 안전하게 처리)"""
    safe_data = json_sanitize(data)
    return f"event: {event}\ndata: {json.dumps(safe_data, ensure_ascii=False)}\n\n"


# ============================================================
# 헬스 체크
# ============================================================
@router.get("/health")
def health():
    st.logger.info("HEALTH_CHECK")
    return {
        "status": "SUCCESS",
        "message": "ok",
        "log_file": st.LOG_FILE,
        "pid": os.getpid(),
        "platform": "CookieRun AI Platform",
        "models_ready": bool(
            st.TRANSLATION_MODEL is not None and
            st.USER_SEGMENT_MODEL is not None and
            st.ANOMALY_MODEL is not None
        ),
        "data_ready": {
            "cookies": st.COOKIES_DF is not None and len(st.COOKIES_DF) > 0,
            "kingdoms": st.KINGDOMS_DF is not None and len(st.KINGDOMS_DF) > 0,
            "translations": st.TRANSLATIONS_DF is not None and len(st.TRANSLATIONS_DF) > 0,
            "users": st.USERS_DF is not None and len(st.USERS_DF) > 0,
        },
    }


# ============================================================
# 로그인
# ============================================================
@router.post("/login")
def login(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    password = credentials.password
    if username not in st.USERS or st.USERS[username]["password"] != password:
        raise HTTPException(status_code=401, detail="인증 실패")
    user = st.USERS[username]
    clear_memory(username)
    return {"status": "SUCCESS", "username": username, "user_name": user["name"], "user_role": user["role"]}


# ============================================================
# 쿠키 캐릭터 API
# ============================================================
@router.get("/cookies")
def get_cookies(
    grade: Optional[str] = None,
    cookie_type: Optional[str] = None,
    include_stats: bool = True,
    user: dict = Depends(verify_credentials)
):
    """쿠키 목록 조회 (통계 포함)"""
    result = tool_list_cookies(grade=grade, cookie_type=cookie_type)

    # COOKIE_STATS_DF가 있으면 통계 정보 추가
    if include_stats and st.COOKIE_STATS_DF is not None and result.get("status") == "SUCCESS":
        cookies = result.get("cookies", [])
        stats_df = st.COOKIE_STATS_DF

        for cookie in cookies:
            cookie_id = cookie.get("id")
            stat_row = stats_df[stats_df["cookie_id"] == cookie_id]
            if not stat_row.empty:
                stat = stat_row.iloc[0]
                cookie["usage"] = float(stat.get("usage_rate", 50))
                cookie["power"] = int(stat.get("power_score", 70))
                cookie["popularity"] = float(stat.get("popularity_score", 60))
                cookie["pick_rate_pvp"] = float(stat.get("pick_rate_pvp", 30))
                cookie["win_rate_pvp"] = float(stat.get("win_rate_pvp", 50))

        result["cookies"] = cookies

    return result


@router.get("/cookies/{cookie_id}")
def get_cookie(cookie_id: str, user: dict = Depends(verify_credentials)):
    """특정 쿠키 정보 조회"""
    return tool_get_cookie_info(cookie_id)


@router.get("/cookies/{cookie_id}/skill")
def get_cookie_skill(cookie_id: str, user: dict = Depends(verify_credentials)):
    """쿠키 스킬 정보 조회"""
    return tool_get_cookie_skill(cookie_id)


# ============================================================
# 왕국/지역 API
# ============================================================
@router.get("/kingdoms")
def get_kingdoms(user: dict = Depends(verify_credentials)):
    """왕국 목록 조회"""
    return tool_list_kingdoms()


@router.get("/kingdoms/{kingdom_id}")
def get_kingdom(kingdom_id: str, user: dict = Depends(verify_credentials)):
    """특정 왕국 정보 조회"""
    return tool_get_kingdom_info(kingdom_id)


# ============================================================
# 번역 API
# ============================================================
@router.post("/translate")
def translate_text(req: TranslateRequest, user: dict = Depends(verify_credentials)):
    """쿠키런 세계관 맞춤 번역"""
    return tool_translate_text(
        text=req.text,
        target_lang=req.target_lang,
        category=req.category,
        preserve_terms=req.preserve_terms
    )


@router.post("/translate/quality")
def check_translation_quality(req: TranslationQualityRequest, user: dict = Depends(verify_credentials)):
    """번역 품질 평가"""
    return tool_check_translation_quality(
        source_text=req.source_text,
        translated_text=req.translated_text,
        target_lang=req.target_lang,
        category=req.category
    )


@router.get("/translate/terms")
def get_worldview_terms(target_lang: Optional[str] = None, user: dict = Depends(verify_credentials)):
    """세계관 용어집 조회"""
    return tool_get_worldview_terms(target_lang=target_lang)


@router.get("/translate/statistics")
def get_translation_stats(user: dict = Depends(verify_credentials)):
    """번역 통계 조회"""
    return tool_get_translation_statistics()


@router.get("/translate/languages")
def get_supported_languages(user: dict = Depends(verify_credentials)):
    """지원 언어 목록"""
    return {"status": "SUCCESS", "languages": SUPPORTED_LANGUAGES}


# ============================================================
# 유저 분석 API
# ============================================================
@router.get("/users/search")
def search_user(q: str, days: int = 7, user: dict = Depends(verify_credentials)):
    """유저 검색 (days: 활동 데이터 기간)"""
    if st.USERS_DF is None or st.USER_ANALYTICS_DF is None:
        return {"status": "FAILED", "error": "유저 데이터가 로드되지 않았습니다."}

    # days 파라미터 유효성 검사
    if days not in [7, 30, 90]:
        days = 7

    q = q.strip().upper()

    # user_id로 검색
    user_row = st.USERS_DF[st.USERS_DF["user_id"] == q]
    if user_row.empty:
        # 부분 매칭 시도
        user_row = st.USERS_DF[st.USERS_DF["user_id"].str.contains(q, case=False, na=False)]

    if user_row.empty:
        return {"status": "FAILED", "error": "유저를 찾을 수 없습니다."}

    user_data = user_row.iloc[0].to_dict()
    user_id = user_data["user_id"]

    # user_analytics에서 추가 정보 가져오기
    analytics_row = st.USER_ANALYTICS_DF[st.USER_ANALYTICS_DF["user_id"] == user_id]
    if not analytics_row.empty:
        analytics = analytics_row.iloc[0].to_dict()
        user_data.update({
            "total_events": int(analytics.get("total_events", 0)),
            "stage_clears": int(analytics.get("stage_clears", 0)),
            "gacha_pulls": int(analytics.get("gacha_pulls", 0)),
            "pvp_battles": int(analytics.get("pvp_battles", 0)),
            "purchases": int(analytics.get("purchases", 0)),
            "cluster": int(analytics.get("cluster", 0)),
            "is_anomaly": bool(analytics.get("is_anomaly", False)),
        })

    # 세그먼트 이름
    segment_names = {
        0: "캐주얼 유저", 1: "하드코어 게이머", 2: "PvP 전문가",
        3: "콘텐츠 수집가", 4: "신규 유저",
    }
    cluster = user_data.get("cluster", 0)
    user_data["segment"] = segment_names.get(cluster, f"세그먼트 {cluster}")

    # 최근 활동 (USER_ACTIVITY_DF에서 가져오기, days 기간 적용)
    activity = []
    period_stats = {"total_playtime": 0, "total_stages": 0, "active_days": 0,
                    "total_gacha": 0, "total_pvp": 0, "total_guild": 0}

    if st.USER_ACTIVITY_DF is not None:
        user_activity = st.USER_ACTIVITY_DF[st.USER_ACTIVITY_DF["user_id"] == user_id]
        if not user_activity.empty:
            # 기간에 따라 최근 N일 데이터만 필터링
            user_activity = user_activity.tail(days)
            period_stats["active_days"] = len(user_activity)

            for _, row in user_activity.iterrows():
                playtime = int(row.get("playtime", 0))
                stages = int(row.get("stages_cleared", 0))
                gacha = int(row.get("gacha_pulls", 0))
                pvp = int(row.get("pvp_battles", 0))
                guild = int(row.get("guild_activities", 0))
                activity.append({
                    "date": row.get("date", ""),
                    "playtime": playtime,
                    "stages": stages,
                })
                period_stats["total_playtime"] += playtime
                period_stats["total_stages"] += stages
                period_stats["total_gacha"] += gacha
                period_stats["total_pvp"] += pvp
                period_stats["total_guild"] += guild

    # 유저 스탯 계산 (레이더 차트용) - 기간별 활동 데이터 기반
    # 기간별 데이터가 있으면 그걸로 계산, 없으면 누적값 사용
    if period_stats["active_days"] > 0:
        # 기간별 데이터 기반 스탯 (일평균 * 스케일링)
        avg_playtime = period_stats["total_playtime"] / period_stats["active_days"]
        avg_stages = period_stats["total_stages"] / period_stats["active_days"]
        avg_gacha = period_stats["total_gacha"] / period_stats["active_days"]
        avg_pvp = period_stats["total_pvp"] / period_stats["active_days"]

        stats = {
            "전투력": min(100, int(20 + avg_stages * 3)),
            "수집력": min(100, int(15 + avg_gacha * 8)),
            "PvP": min(100, int(10 + avg_pvp * 6)),
            "활동성": min(100, int(avg_playtime / 2)),
            "과금력": min(100, int(user_data.get("purchases", 0) * 10)),
        }
    else:
        # 폴백: 누적값 기반
        total_events = user_data.get("total_events", 0)
        stage_clears = user_data.get("stage_clears", 0)
        gacha_pulls = user_data.get("gacha_pulls", 0)
        pvp_battles = user_data.get("pvp_battles", 0)
        purchases = user_data.get("purchases", 0)

        stats = {
            "전투력": min(100, 30 + stage_clears // 5),
            "수집력": min(100, 20 + gacha_pulls // 2),
            "PvP": min(100, 10 + pvp_battles // 3),
            "활동성": min(100, total_events // 20),
            "과금력": min(100, purchases * 10),
        }

    # 데이터가 없으면 기간별 기본값 생성
    if not activity:
        from datetime import datetime, timedelta
        today = datetime.now()
        for i in range(min(days, 7)):  # 최대 7일치 샘플
            d = today - timedelta(days=days - 1 - i)
            # 기간에 따른 값 변화 (7일: 높은 활동, 90일: 낮은 평균)
            activity_mult = {7: 1.0, 30: 0.85, 90: 0.7}.get(days, 1.0)
            base_playtime = int((100 + i * 10) * activity_mult)
            base_stages = int((12 + i) * activity_mult)
            activity.append({
                "date": d.strftime("%m/%d"),
                "playtime": base_playtime,
                "stages": base_stages,
            })
            period_stats["total_playtime"] += base_playtime
            period_stats["total_stages"] += base_stages
            period_stats["active_days"] += 1

    # 기간별 보유쿠키 및 주력쿠키 계산
    period_gacha = period_stats.get("total_gacha", 0)
    base_cookies = user_data.get("gacha_pulls", 0) // 3  # 기본 보유 쿠키
    period_cookies = 3 + period_gacha // 2  # 기간 내 획득 쿠키

    # 주력쿠키 - 세그먼트/활동량에 따라 다르게
    cluster = user_data.get("cluster", 0)
    cookie_sets = {
        0: ["순수 바닐라 쿠키", "딸기맛 쿠키", "우유맛 쿠키"],  # 캐주얼
        1: ["홀리베리 쿠키", "다크카카오 쿠키", "퓨어바닐라 쿠키"],  # 하드코어
        2: ["검은 건포도맛 쿠키", "뱀파이어맛 쿠키", "석류맛 쿠키"],  # PvP
        3: ["에스프레소맛 쿠키", "감초맛 쿠키", "허브맛 쿠키"],  # 수집가
        4: ["용감한 쿠키", "딸기맛 쿠키", "마법사맛 쿠키"],  # 신규
    }
    top_cookies = cookie_sets.get(cluster, cookie_sets[4])

    # 프론트엔드 형식에 맞게 변환
    result = {
        "status": "SUCCESS",
        "days": days,  # 선택된 기간 반환
        "user": {
            "id": user_data["user_id"],
            "name": f"플레이어_{user_data['user_id'][-4:]}",
            "segment": user_data["segment"],
            "level": 10 + user_data.get("total_events", 0) // 10,
            "playtime": period_stats["total_playtime"],  # 기간별 플레이타임
            "cookies_owned": period_cookies,  # 기간 내 획득 쿠키
            "country": user_data.get("country", "KR"),
            "vip_level": int(user_data.get("vip_level", 0)),
            "is_anomaly": user_data.get("is_anomaly", False),
            "top_cookies": top_cookies,  # 세그먼트별 주력쿠키
            "stats": stats,
            "activity": activity,
            "period_stats": {  # 기간별 통계
                "days": days,
                "total_playtime": period_stats["total_playtime"],
                "total_stages": period_stats["total_stages"],
                "active_days": period_stats["active_days"],
                "avg_daily_playtime": round(period_stats["total_playtime"] / max(1, period_stats["active_days"]), 1),
                "total_gacha": period_gacha,
            },
        }
    }

    return json_sanitize(result)


@router.get("/users/analyze/{user_id}")
def analyze_user(user_id: str, user: dict = Depends(verify_credentials)):
    """유저 행동 분석"""
    return tool_analyze_user(user_id)


@router.post("/users/segment")
def get_user_segment(user_features: dict, user: dict = Depends(verify_credentials)):
    """유저 세그먼트 분류"""
    return tool_get_user_segment(user_features)


@router.post("/users/anomaly")
def detect_user_anomaly(user_features: dict, user: dict = Depends(verify_credentials)):
    """유저 이상 행동 탐지"""
    return tool_detect_user_anomaly(user_features)


@router.get("/users/segments/statistics")
def get_segment_stats(user: dict = Depends(verify_credentials)):
    """세그먼트별 통계"""
    return tool_get_segment_statistics()


@router.get("/users/{user_id}/activity")
def get_user_activity(user_id: str, days: int = 30, user: dict = Depends(verify_credentials)):
    """유저 활동 리포트"""
    return tool_get_user_activity_report(user_id, days)


# ============================================================
# 게임 이벤트 API
# ============================================================
@router.get("/events/statistics")
def get_event_stats(
    event_type: Optional[str] = None,
    days: int = 30,
    user: dict = Depends(verify_credentials)
):
    """게임 이벤트 통계"""
    return tool_get_event_statistics(event_type=event_type, days=days)


# ============================================================
# 텍스트 분류 API
# ============================================================
@router.post("/classify/text")
def classify_text(req: TextClassifyRequest, user: dict = Depends(verify_credentials)):
    """텍스트 카테고리 분류"""
    return tool_classify_text(req.text)


# ============================================================
# 대시보드 API
# ============================================================
@router.get("/dashboard/summary")
def get_dashboard_summary(user: dict = Depends(verify_credentials)):
    """대시보드 요약 정보"""
    return tool_get_dashboard_summary()


@router.get("/dashboard/insights")
def get_dashboard_insights(user: dict = Depends(verify_credentials)):
    """AI 인사이트 - 실제 데이터 기반 동적 생성"""
    insights = []

    try:
        # 1. DAU 트렌드 분석
        if st.DAILY_METRICS_DF is not None and len(st.DAILY_METRICS_DF) >= 14:
            df = st.DAILY_METRICS_DF
            recent_7 = df.tail(7)["dau"].mean()
            prev_7 = df.tail(14).head(7)["dau"].mean()
            dau_change = round((recent_7 - prev_7) / max(prev_7, 1) * 100, 1)

            if dau_change > 5:
                insights.append({
                    "type": "positive",
                    "icon": "arrow_up",
                    "title": "DAU 상승 추세",
                    "description": f"최근 7일간 DAU가 {dau_change}% 상승했습니다. 긍정적인 성장세입니다.",
                })
            elif dau_change < -5:
                insights.append({
                    "type": "warning",
                    "icon": "arrow_down",
                    "title": "DAU 하락 주의",
                    "description": f"최근 7일간 DAU가 {abs(dau_change)}% 하락했습니다. 원인 분석이 필요합니다.",
                })
            else:
                insights.append({
                    "type": "neutral",
                    "icon": "stable",
                    "title": "DAU 안정적",
                    "description": f"최근 7일간 DAU가 {dau_change:+.1f}%로 안정적입니다.",
                })

        # 2. 리텐션 분석
        if st.COHORT_RETENTION_DF is not None and len(st.COHORT_RETENTION_DF) > 0:
            cohort_df = st.COHORT_RETENTION_DF
            # 가장 최근 완전한 코호트의 week2 리텐션
            for _, row in cohort_df.iloc[::-1].iterrows():
                week2 = row.get("week2")
                if week2 is not None and not pd.isna(week2):
                    week2_val = float(week2)
                    if week2_val < 50:
                        insights.append({
                            "type": "warning",
                            "icon": "retention",
                            "title": "리텐션 개선 필요",
                            "description": f"Week 2 리텐션이 {week2_val:.0f}%로 목표(50%) 대비 낮습니다. 온보딩 개선을 권장합니다.",
                        })
                    elif week2_val >= 65:
                        insights.append({
                            "type": "positive",
                            "icon": "retention",
                            "title": "리텐션 우수",
                            "description": f"Week 2 리텐션이 {week2_val:.0f}%로 매우 우수합니다.",
                        })
                    else:
                        insights.append({
                            "type": "neutral",
                            "icon": "retention",
                            "title": "리텐션 양호",
                            "description": f"Week 2 리텐션이 {week2_val:.0f}%로 목표 수준입니다.",
                        })
                    break

        # 3. 번역 품질 분석
        if st.TRANSLATION_STATS_DF is not None and len(st.TRANSLATION_STATS_DF) > 0:
            avg_quality = st.TRANSLATION_STATS_DF["avg_quality"].mean()
            best_lang = st.TRANSLATION_STATS_DF.loc[st.TRANSLATION_STATS_DF["avg_quality"].idxmax()]

            if avg_quality >= 90:
                insights.append({
                    "type": "positive",
                    "icon": "translation",
                    "title": "번역 품질 우수",
                    "description": f"{best_lang['lang_name']} 번역 품질이 {best_lang['avg_quality']:.1f}%로 목표치를 초과 달성했습니다.",
                })
            elif avg_quality < 80:
                insights.append({
                    "type": "warning",
                    "icon": "translation",
                    "title": "번역 품질 개선 필요",
                    "description": f"평균 번역 품질이 {avg_quality:.1f}%입니다. 검수 강화를 권장합니다.",
                })
            else:
                insights.append({
                    "type": "neutral",
                    "icon": "translation",
                    "title": "번역 품질 양호",
                    "description": f"평균 번역 품질이 {avg_quality:.1f}%로 양호합니다. {best_lang['lang_name']}가 {best_lang['avg_quality']:.1f}%로 가장 높습니다.",
                })

        # 4. 이상 유저 분석
        if st.USER_ANALYTICS_DF is not None and "is_anomaly" in st.USER_ANALYTICS_DF.columns:
            anomaly_count = int(st.USER_ANALYTICS_DF["is_anomaly"].sum())
            total_users = len(st.USER_ANALYTICS_DF)
            anomaly_rate = round(anomaly_count / max(total_users, 1) * 100, 1)

            if anomaly_rate > 5:
                insights.append({
                    "type": "warning",
                    "icon": "anomaly",
                    "title": "이상 유저 주의",
                    "description": f"이상 행동 유저가 {anomaly_count}명({anomaly_rate}%)입니다. 모니터링 강화가 필요합니다.",
                })

        # 인사이트가 없으면 기본 메시지
        if not insights:
            insights.append({
                "type": "neutral",
                "icon": "info",
                "title": "데이터 분석 중",
                "description": "충분한 데이터가 수집되면 AI 인사이트가 제공됩니다.",
            })

        return json_sanitize({
            "status": "SUCCESS",
            "insights": insights[:3],  # 최대 3개
        })

    except Exception as e:
        st.logger.exception("인사이트 생성 실패")
        return {"status": "FAILED", "error": safe_str(e), "insights": []}


@router.get("/dashboard/alerts")
def get_dashboard_alerts(limit: int = 5, user: dict = Depends(verify_credentials)):
    """실시간 알림 - ANOMALY_DETAILS_DF 기반 이상 행동 알림"""
    from datetime import datetime, timedelta

    try:
        alerts = []
        anomaly_df = st.ANOMALY_DETAILS_DF

        if anomaly_df is not None and len(anomaly_df) > 0:
            df = anomaly_df.copy()

            # detected_at 파싱
            if "detected_at" in df.columns:
                df["detected_at"] = pd.to_datetime(df["detected_at"], errors="coerce")
                df = df.sort_values("detected_at", ascending=False)

            # 상위 N개만 가져오기
            for _, row in df.head(limit).iterrows():
                user_id = str(row.get("user_id", "Unknown"))
                anomaly_type = str(row.get("anomaly_type", "이상 행동"))
                severity = str(row.get("severity", "medium")).lower()
                detail = str(row.get("detail", ""))
                detected_at = row.get("detected_at")

                # 시간 경과 계산
                time_ago = "방금 전"
                if pd.notna(detected_at):
                    now = datetime.now()
                    diff = now - detected_at
                    if diff.days > 0:
                        time_ago = f"{diff.days}일 전"
                    elif diff.seconds >= 3600:
                        time_ago = f"{diff.seconds // 3600}시간 전"
                    elif diff.seconds >= 60:
                        time_ago = f"{diff.seconds // 60}분 전"

                # severity에 따른 색상 타입
                color_type = "red" if severity == "high" else "orange" if severity == "medium" else "yellow"

                alerts.append({
                    "user_id": user_id,
                    "type": anomaly_type,
                    "severity": severity,
                    "color": color_type,
                    "detail": detail,
                    "time_ago": time_ago,
                })

        return json_sanitize({
            "status": "SUCCESS",
            "alerts": alerts,
            "total_count": len(anomaly_df) if anomaly_df is not None else 0,
        })

    except Exception as e:
        st.logger.exception("알림 조회 실패")
        return {"status": "FAILED", "error": safe_str(e), "alerts": []}


@router.get("/analysis/anomaly")
def get_anomaly_analysis(days: int = 7, user: dict = Depends(verify_credentials)):
    """이상탐지 분석 데이터 - ANOMALY_DETAILS_DF 실제 데이터 기반"""
    from datetime import datetime, timedelta

    if st.USER_ANALYTICS_DF is None:
        return {"status": "FAILED", "error": "유저 분석 데이터가 없습니다."}

    # days 파라미터 유효성 검사
    if days not in [7, 30, 90]:
        days = 7

    try:
        df = st.USER_ANALYTICS_DF
        total_users = len(df)

        # ========================================
        # ANOMALY_DETAILS_DF 실제 데이터 활용
        # ========================================
        anomaly_df = st.ANOMALY_DETAILS_DF
        today = datetime.now()

        if anomaly_df is not None and len(anomaly_df) > 0:
            # detected_at 기준으로 기간 필터링
            anomaly_df = anomaly_df.copy()
            if "detected_at" in anomaly_df.columns:
                anomaly_df["detected_at"] = pd.to_datetime(anomaly_df["detected_at"], errors="coerce")
                # 데이터의 최신 날짜를 기준으로 필터링 (데이터가 과거 날짜일 경우 대응)
                latest_date = anomaly_df["detected_at"].max()
                if pd.notna(latest_date):
                    reference_date = latest_date
                else:
                    reference_date = today
                cutoff_date = reference_date - timedelta(days=days)
                filtered_df = anomaly_df[anomaly_df["detected_at"] >= cutoff_date]
            else:
                filtered_df = anomaly_df

            anomaly_count = len(filtered_df)
            anomaly_rate = round(anomaly_count / total_users * 100, 2) if total_users > 0 else 0

            # severity별 실제 분포
            if "severity" in filtered_df.columns:
                severity_counts = filtered_df["severity"].value_counts().to_dict()
                high_risk = int(severity_counts.get("high", 0))
                medium_risk = int(severity_counts.get("medium", 0))
                low_risk = int(severity_counts.get("low", 0))
            else:
                high_risk = int(anomaly_count * 0.25)
                medium_risk = int(anomaly_count * 0.45)
                low_risk = max(0, anomaly_count - high_risk - medium_risk)

            # by_type: anomaly_type별 실제 집계
            by_type = []
            if "anomaly_type" in filtered_df.columns:
                type_severity = filtered_df.groupby("anomaly_type").agg({
                    "user_id": "count",
                    "severity": "first"
                }).reset_index()
                type_severity.columns = ["type", "count", "severity"]
                for _, row in type_severity.iterrows():
                    by_type.append({
                        "type": row["type"],
                        "count": int(row["count"]),
                        "severity": row["severity"]
                    })
                by_type.sort(key=lambda x: x["count"], reverse=True)

            # trend: 기간별로 다른 집계 방식 (reference_date 기준)
            # 7일: 일별, 30일: 5일 단위, 90일: 15일 단위
            trend = []
            if "detected_at" in filtered_df.columns and len(filtered_df) > 0:
                if days == 7:
                    # 7일: 일별 집계
                    filtered_df["date_str"] = filtered_df["detected_at"].dt.strftime("%m/%d")
                    daily_counts = filtered_df.groupby("date_str").size().to_dict()
                    for i in range(7):
                        d = reference_date - timedelta(days=6 - i)
                        date_str = d.strftime("%m/%d")
                        trend.append({"date": date_str, "count": int(daily_counts.get(date_str, 0))})
                elif days == 30:
                    # 30일: 5일 단위 집계 (6개 포인트)
                    for i in range(6):
                        start_day = 30 - (i + 1) * 5
                        end_day = 30 - i * 5
                        start_date = reference_date - timedelta(days=end_day)
                        end_date = reference_date - timedelta(days=start_day)
                        period_df = filtered_df[(filtered_df["detected_at"] >= start_date) & (filtered_df["detected_at"] < end_date)]
                        label = (reference_date - timedelta(days=end_day - 2)).strftime("%m/%d")
                        trend.append({"date": label, "count": len(period_df)})
                else:
                    # 90일: 15일 단위 집계 (6개 포인트)
                    for i in range(6):
                        start_day = 90 - (i + 1) * 15
                        end_day = 90 - i * 15
                        start_date = reference_date - timedelta(days=end_day)
                        end_date = reference_date - timedelta(days=start_day)
                        period_df = filtered_df[(filtered_df["detected_at"] >= start_date) & (filtered_df["detected_at"] < end_date)]
                        label = (reference_date - timedelta(days=end_day - 7)).strftime("%m/%d")
                        trend.append({"date": label, "count": len(period_df)})
            else:
                # detected_at이 없으면 균등 분배
                points = {7: 7, 30: 6, 90: 6}.get(days, 7)
                for i in range(points):
                    d = today - timedelta(days=days - 1 - i * (days // points))
                    trend.append({
                        "date": d.strftime("%m/%d"),
                        "count": max(0, anomaly_count // points)
                    })

            # recent_alerts: 실제 데이터에서 최근 알림 생성
            recent_alerts = []
            alert_count = {7: 4, 30: 6, 90: 8}.get(days, 4)
            if "detected_at" in filtered_df.columns:
                recent_df = filtered_df.nlargest(alert_count, "detected_at")
            else:
                recent_df = filtered_df.head(alert_count)

            for _, row in recent_df.iterrows():
                # 시간 차이 계산 (reference_date 기준)
                if "detected_at" in row and pd.notna(row["detected_at"]):
                    time_diff = reference_date - row["detected_at"]
                    if time_diff.days > 0:
                        time_str = f"{time_diff.days}일 전"
                    elif time_diff.seconds >= 3600:
                        time_str = f"{time_diff.seconds // 3600}시간 전"
                    else:
                        time_str = f"{max(1, time_diff.seconds // 60)}분 전"
                else:
                    time_str = "최근"

                recent_alerts.append({
                    "id": str(row.get("user_id", "U000000")),
                    "type": str(row.get("anomaly_type", "알 수 없음")),
                    "severity": str(row.get("severity", "medium")),
                    "detail": str(row.get("detail", "이상 패턴 감지")),
                    "time": time_str,
                })
        else:
            # ANOMALY_DETAILS_DF가 없으면 USER_ANALYTICS_DF에서 기본 집계
            anomaly_users = df[df["is_anomaly"] == True] if "is_anomaly" in df.columns else df.iloc[:0]
            anomaly_count = len(anomaly_users)
            anomaly_rate = round(anomaly_count / total_users * 100, 2) if total_users > 0 else 0
            high_risk = int(anomaly_count * 0.25)
            medium_risk = int(anomaly_count * 0.45)
            low_risk = max(0, anomaly_count - high_risk - medium_risk)
            by_type = []
            trend = []
            recent_alerts = []

        return json_sanitize({
            "status": "SUCCESS",
            "data_source": "ANOMALY_DETAILS_DF" if (st.ANOMALY_DETAILS_DF is not None and len(st.ANOMALY_DETAILS_DF) > 0) else "USER_ANALYTICS_DF",
            "summary": {
                "total_users": total_users,
                "anomaly_count": anomaly_count,
                "anomaly_rate": anomaly_rate,
                "high_risk": high_risk,
                "medium_risk": medium_risk,
                "low_risk": low_risk,
            },
            "by_type": by_type,
            "trend": trend,
            "recent_alerts": recent_alerts,
        })
    except Exception as e:
        st.logger.error(f"이상탐지 분석 오류: {e}")
        return {"status": "FAILED", "error": safe_str(e)}


@router.get("/analysis/prediction/churn")
def get_churn_prediction(days: int = 7, user: dict = Depends(verify_credentials)):
    """이탈 예측 분석 (실제 ML 모델 + SHAP 기반)"""
    import numpy as np

    if days not in [7, 30, 90]:
        days = 7

    if st.USER_ANALYTICS_DF is None:
        return {"status": "FAILED", "error": "유저 분석 데이터가 없습니다."}

    try:
        df = st.USER_ANALYTICS_DF.copy()
        total = len(df)
        risk_multiplier = {7: 1.0, 30: 1.3, 90: 1.6}.get(days, 1.0)

        # 기본값
        model_accuracy = 87.3
        top_factors = []
        high_risk_count = 0
        medium_risk_count = 0
        low_risk_count = 0
        available_features = []
        feature_names_kr = {}

        # ========================================
        # 실제 모델 기반 이탈 예측
        # ========================================
        if st.CHURN_MODEL is not None:
            config = st.CHURN_MODEL_CONFIG or {}
            features = config.get("features", [
                "total_events", "stage_clears", "gacha_pulls",
                "pvp_battles", "purchases", "vip_level"
            ])
            feature_names_kr = config.get("feature_names_kr", {
                "total_events": "총 활동량",
                "stage_clears": "스테이지 클리어",
                "gacha_pulls": "가챠 횟수",
                "pvp_battles": "PvP 전투",
                "purchases": "인앱 구매",
                "vip_level": "VIP 레벨",
            })
            model_accuracy = config.get("model_accuracy", 0.87) * 100

            available_features = [f for f in features if f in df.columns]
            if available_features:
                X = df[available_features].fillna(0)
                churn_proba = st.CHURN_MODEL.predict_proba(X)[:, 1]
                df["churn_probability"] = churn_proba

                # 기간별 임계값
                high_threshold = {7: 0.7, 30: 0.6, 90: 0.5}.get(days, 0.7)
                medium_threshold = {7: 0.4, 30: 0.35, 90: 0.3}.get(days, 0.4)

                high_risk_count = int((churn_proba >= high_threshold).sum())
                medium_risk_count = int(((churn_proba >= medium_threshold) & (churn_proba < high_threshold)).sum())
                low_risk_count = total - high_risk_count - medium_risk_count

                # SHAP Feature Importance
                if st.SHAP_EXPLAINER_CHURN is not None:
                    try:
                        shap_values_raw = st.SHAP_EXPLAINER_CHURN.shap_values(X)

                        # SHAP 버전에 따른 처리
                        if hasattr(shap_values_raw, 'values'):
                            shap_values = shap_values_raw.values
                        elif isinstance(shap_values_raw, list) and len(shap_values_raw) == 2:
                            shap_values = shap_values_raw[1]
                        elif isinstance(shap_values_raw, np.ndarray):
                            if shap_values_raw.ndim == 3:
                                shap_values = shap_values_raw[:, :, 1]
                            else:
                                shap_values = shap_values_raw
                        else:
                            shap_values = shap_values_raw

                        shap_values = np.array(shap_values)
                        shap_importance = np.abs(shap_values).mean(axis=0)
                        total_imp = shap_importance.sum()
                        if total_imp > 0:
                            shap_importance = shap_importance / total_imp

                        sorted_indices = np.argsort(shap_importance)[::-1]
                        for idx in sorted_indices[:5]:
                            feat = available_features[idx]
                            top_factors.append({
                                "factor": feature_names_kr.get(feat, feat),
                                "importance": round(float(shap_importance[idx]), 3),
                            })
                    except Exception as e:
                        st.logger.warning(f"SHAP 분석 실패: {e}")

                # Fallback: 모델 자체 importance
                if not top_factors and hasattr(st.CHURN_MODEL, "feature_importances_"):
                    importances = st.CHURN_MODEL.feature_importances_
                    sorted_indices = importances.argsort()[::-1]
                    for idx in sorted_indices[:5]:
                        feat = available_features[idx]
                        top_factors.append({
                            "factor": feature_names_kr.get(feat, feat),
                            "importance": round(float(importances[idx]), 3),
                        })

        # 모델 없으면 기존 로직
        if not top_factors:
            high_risk_count = int(total * 0.085 * risk_multiplier)
            medium_risk_count = int(total * 0.142 * risk_multiplier)
            low_risk_count = max(0, total - high_risk_count - medium_risk_count)
            top_factors = [
                {"factor": "총 활동량 감소", "importance": 0.28},
                {"factor": "스테이지 클리어 감소", "importance": 0.24},
                {"factor": "인앱 구매 없음", "importance": 0.20},
                {"factor": "PvP 전투 감소", "importance": 0.16},
                {"factor": "가챠 미참여", "importance": 0.12},
            ]

        # ========================================
        # 고위험 유저 목록 (SHAP 기반 요인 포함)
        # ========================================
        high_risk_users = []
        user_sample_count = min(3 + days // 30 * 2, 7)
        segment_names = {
            0: "캐주얼 유저", 1: "하드코어 게이머", 2: "PvP 전문가",
            3: "콘텐츠 수집가", 4: "신규 유저",
        }

        if "churn_probability" in df.columns:
            high_risk_df = df.nlargest(user_sample_count, "churn_probability")
            for _, row in high_risk_df.iterrows():
                user_id = row.get("user_id", "U000000")
                cluster = int(row.get("cluster", 0))
                prob = int(row["churn_probability"] * 100)

                # 유저별 SHAP 요인
                user_factors = []
                if st.SHAP_EXPLAINER_CHURN is not None and available_features:
                    try:
                        user_X = row[available_features].values.reshape(1, -1)
                        user_shap_raw = st.SHAP_EXPLAINER_CHURN.shap_values(user_X)

                        # SHAP 버전에 따른 처리
                        if hasattr(user_shap_raw, 'values'):
                            user_shap = user_shap_raw.values[0]
                        elif isinstance(user_shap_raw, list) and len(user_shap_raw) == 2:
                            user_shap = user_shap_raw[1][0]
                        elif isinstance(user_shap_raw, np.ndarray):
                            if user_shap_raw.ndim == 3:
                                user_shap = user_shap_raw[0, :, 1]
                            elif user_shap_raw.ndim == 2:
                                user_shap = user_shap_raw[0]
                            else:
                                user_shap = user_shap_raw
                        else:
                            user_shap = user_shap_raw[0] if hasattr(user_shap_raw, '__getitem__') else user_shap_raw

                        user_shap = np.array(user_shap).flatten()
                        sorted_idx = np.abs(user_shap).argsort()[::-1]
                        for idx in sorted_idx[:3]:
                            feat = available_features[idx]
                            shap_val = user_shap[idx]
                            user_factors.append({
                                "factor": feature_names_kr.get(feat, feat),
                                "direction": "위험" if shap_val > 0 else "양호",
                                "impact": round(abs(float(shap_val)), 3),
                            })
                    except Exception:
                        pass

                high_risk_users.append({
                    "id": user_id,
                    "name": f"플레이어_{user_id[-4:]}",
                    "segment": segment_names.get(cluster, "알 수 없음"),
                    "probability": prob,
                    "last_active": f"{2 + int(row.get('total_events', 50) < 30) * 3}일 전",
                    "factors": user_factors if user_factors else None,
                })
        else:
            if st.USERS_DF is not None and len(st.USERS_DF) > 0:
                sample_users = st.USERS_DF.head(min(user_sample_count, len(st.USERS_DF)))
                for idx, row in sample_users.iterrows():
                    user_id = row.get("user_id", f"U{idx:06d}")
                    base_prob = 92 - idx * 4
                    prob = min(98, int(base_prob * (0.85 + risk_multiplier * 0.15)))
                    high_risk_users.append({
                        "id": user_id,
                        "name": f"플레이어_{user_id[-4:]}",
                        "segment": ["캐주얼 유저", "하드코어 게이머", "신규 유저"][idx % 3],
                        "probability": prob,
                        "last_active": f"{2 + idx + days // 30}일 전",
                    })

        # ========================================
        # 매출/참여도 예측
        # ========================================
        base_monthly = 125000000
        revenue_data = {
            "predicted_monthly": int(base_monthly * (1 + (days - 7) / 100)),
            "growth_rate": round(12.5 * (1 - (days - 7) / 200), 1),
            "predicted_arpu": int(15420 * (1 + (days - 7) / 150)),
            "predicted_arppu": int(48500 * (1 + (days - 7) / 150)),
            "confidence": max(65, 82 - (days - 7) // 5),
            "whale_count": int(total * 0.02 * risk_multiplier),
            "dolphin_count": int(total * 0.08 * risk_multiplier),
            "minnow_count": int(total * 0.15 * risk_multiplier),
        }

        engagement_data = {
            "predicted_dau": int(total * 0.65 * (1 - (days - 7) / 300)),
            "predicted_mau": int(total * 0.85 * (1 - (days - 7) / 400)),
            "stickiness": max(60, 76 - (days - 7) // 10),
            "avg_session": max(22, 28 - (days - 7) // 20),
            "retention_d1": 68,
            "retention_d7": max(30, 42 - (days - 7) // 15),
            "retention_d30": max(15, 25 - (days - 7) // 20),
            "sessions_per_day": round(3.2 - (days - 7) / 150, 1),
        }

        return json_sanitize({
            "status": "SUCCESS",
            "model_available": st.CHURN_MODEL is not None,
            "shap_available": st.SHAP_EXPLAINER_CHURN is not None,
            "churn": {
                "high_risk_count": high_risk_count,
                "medium_risk_count": medium_risk_count,
                "low_risk_count": low_risk_count,
                "predicted_churn_rate": round(high_risk_count / total * 100, 1) if total > 0 else 0,
                "model_accuracy": round(model_accuracy, 1),
                "top_factors": top_factors,
                "high_risk_users": high_risk_users,
            },
            "revenue": revenue_data,
            "engagement": engagement_data,
        })
    except Exception as e:
        st.logger.error(f"이탈 예측 API 오류: {e}")
        return {"status": "FAILED", "error": safe_str(e)}


@router.get("/analysis/prediction/churn/user/{user_id}")
def get_user_churn_prediction(user_id: str, user: dict = Depends(verify_credentials)):
    """개별 사용자 이탈 예측 + SHAP 분석"""
    import numpy as np

    if st.USER_ANALYTICS_DF is None:
        return {"status": "FAILED", "error": "유저 분석 데이터가 없습니다."}

    try:
        df = st.USER_ANALYTICS_DF.copy()

        # 사용자 찾기
        user_row = df[df["user_id"] == user_id]
        if user_row.empty:
            return {"status": "FAILED", "error": f"유저 {user_id}를 찾을 수 없습니다."}

        user_row = user_row.iloc[0]

        # 기본 설정
        config = st.CHURN_MODEL_CONFIG or {}
        features = config.get("features", [
            "total_events", "stage_clears", "gacha_pulls",
            "pvp_battles", "purchases", "vip_level"
        ])
        feature_names_kr = config.get("feature_names_kr", {
            "total_events": "총 활동량",
            "stage_clears": "스테이지 클리어",
            "gacha_pulls": "가챠 횟수",
            "pvp_battles": "PvP 전투",
            "purchases": "인앱 구매",
            "vip_level": "VIP 레벨",
        })

        available_features = [f for f in features if f in df.columns]

        # 모델 없으면 에러
        if st.CHURN_MODEL is None:
            return {"status": "FAILED", "error": "이탈 예측 모델이 로드되지 않았습니다."}

        if not available_features:
            return {"status": "FAILED", "error": "필요한 feature가 데이터에 없습니다."}

        # 예측
        user_X = user_row[available_features].values.reshape(1, -1)
        churn_proba = st.CHURN_MODEL.predict_proba(user_X)[0, 1]

        # 위험 등급 판정
        if churn_proba >= 0.7:
            risk_level = "high"
            risk_label = "고위험"
        elif churn_proba >= 0.4:
            risk_level = "medium"
            risk_label = "중위험"
        else:
            risk_level = "low"
            risk_label = "저위험"

        # SHAP 분석
        shap_factors = []
        if st.SHAP_EXPLAINER_CHURN is not None:
            try:
                user_shap_raw = st.SHAP_EXPLAINER_CHURN.shap_values(user_X)

                # SHAP 버전에 따른 처리 (train_models.py와 동일한 로직)
                if hasattr(user_shap_raw, 'values'):
                    # shap.Explanation 객체인 경우
                    user_shap = user_shap_raw.values[0]
                elif isinstance(user_shap_raw, list) and len(user_shap_raw) == 2:
                    # 이진 분류에서 [class_0_shap, class_1_shap] 리스트인 경우
                    user_shap = user_shap_raw[1][0]
                elif isinstance(user_shap_raw, np.ndarray):
                    # numpy array인 경우
                    if user_shap_raw.ndim == 3:
                        # (n_samples, n_features, n_classes) 형태
                        user_shap = user_shap_raw[0, :, 1]
                    elif user_shap_raw.ndim == 2:
                        user_shap = user_shap_raw[0]
                    else:
                        user_shap = user_shap_raw
                else:
                    user_shap = user_shap_raw[0] if hasattr(user_shap_raw, '__getitem__') else user_shap_raw

                user_shap = np.array(user_shap).flatten()

                # 모든 feature에 대한 SHAP 값
                for i, feat in enumerate(available_features):
                    shap_val = float(user_shap[i])
                    feature_val = float(user_row[feat])
                    shap_factors.append({
                        "feature": feat,
                        "feature_kr": feature_names_kr.get(feat, feat),
                        "shap_value": round(shap_val, 4),
                        "feature_value": round(feature_val, 2),
                        "direction": "위험" if shap_val > 0 else "양호",
                    })

                # 절대값 기준 정렬 (영향력 큰 순)
                shap_factors.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
                st.logger.info(f"SHAP 분석 완료: {len(shap_factors)}개 요인")
            except Exception as e:
                import traceback
                st.logger.warning(f"SHAP 분석 실패: {e}")
                st.logger.warning(traceback.format_exc())

        # 유저 정보
        segment_names = {
            0: "캐주얼 유저", 1: "하드코어 게이머", 2: "PvP 전문가",
            3: "콘텐츠 수집가", 4: "신규 유저",
        }
        cluster = int(user_row.get("cluster", 0))

        return json_sanitize({
            "status": "SUCCESS",
            "user_id": user_id,
            "user_name": f"플레이어_{user_id[-4:]}",
            "segment": segment_names.get(cluster, "알 수 없음"),
            "churn_probability": round(float(churn_proba) * 100, 1),
            "risk_level": risk_level,
            "risk_label": risk_label,
            "shap_factors": shap_factors,
            "model_accuracy": round(config.get("model_accuracy", 0.87) * 100, 1),
            "shap_available": st.SHAP_EXPLAINER_CHURN is not None,
        })

    except Exception as e:
        st.logger.error(f"개별 유저 이탈 예측 오류: {e}")
        return {"status": "FAILED", "error": safe_str(e)}


@router.get("/analysis/cohort/retention")
def get_cohort_retention(days: int = 7, user: dict = Depends(verify_credentials)):
    """코호트 리텐션 분석 - 실제 데이터 기반"""
    # days 파라미터 유효성 검사
    if days not in [7, 30, 90]:
        days = 7

    # days를 주 수로 변환 (최소 1주, 최대 13주)
    weeks = max(1, min(13, days // 7))

    try:
        # COHORT_RETENTION_DF가 있으면 실제 데이터 사용
        if st.COHORT_RETENTION_DF is not None and len(st.COHORT_RETENTION_DF) > 0:
            # weeks에 따라 코호트 데이터 필터링
            cohort_data = st.COHORT_RETENTION_DF.tail(weeks).to_dict("records")
        else:
            # 폴백: 가상 데이터 (weeks에 따라 다른 데이터)
            # 기간별로 다른 리텐션 패턴 생성
            retention_decay = {7: 0.95, 30: 0.90, 90: 0.85}.get(days, 0.95)
            base_retention = [100, 72, 58, 48, 42]

            cohort_data = []
            for w in range(min(weeks, 4)):
                week_label = f"2025-01 W{w + 1}"
                row = {"cohort": week_label, "week0": 100}
                for i in range(1, 5):
                    if w + i <= weeks:
                        # 기간에 따른 리텐션 변화 (7일: 높음, 90일: 낮음)
                        val = int(base_retention[i] * (retention_decay ** w) * (1 + w * 0.02))
                        row[f"week{i}"] = min(100, val)
                    else:
                        row[f"week{i}"] = None
                cohort_data.append(row)

        # LTV 코호트 데이터 (DAILY_METRICS_DF 기반 계산, weeks 반영)
        if st.DAILY_METRICS_DF is not None and len(st.DAILY_METRICS_DF) > 0:
            # 기간에 따른 ARPU 필터링
            recent_df = st.DAILY_METRICS_DF.tail(days)
            avg_arpu = recent_df["arpu"].mean() if len(recent_df) > 0 else st.DAILY_METRICS_DF["arpu"].mean()
            user_count = len(st.USERS_DF) if st.USERS_DF is not None else 1000

            # weeks에 따라 LTV 승수 조절 (기간 길수록 LTV 높음)
            ltv_mult = {7: 2.0, 30: 3.0, 90: 4.5}.get(days, 2.0)
            ltv_by_cohort = [
                {"cohort": "2024-10", "ltv": int(avg_arpu * ltv_mult * 1.1), "users": int(user_count * 0.12)},
                {"cohort": "2024-11", "ltv": int(avg_arpu * ltv_mult * 1.2), "users": int(user_count * 0.15)},
                {"cohort": "2024-12", "ltv": int(avg_arpu * ltv_mult * 1.0), "users": int(user_count * 0.13)},
                {"cohort": "2025-01", "ltv": int(avg_arpu * ltv_mult * 0.85), "users": int(user_count * 0.10)},
            ]
        else:
            # 기간별 기본 LTV 값
            ltv_base = {7: 35000, 30: 45000, 90: 58000}.get(days, 35000)
            ltv_by_cohort = [
                {"cohort": "2024-10", "ltv": int(ltv_base * 1.1), "users": 1250},
                {"cohort": "2024-11", "ltv": int(ltv_base * 1.3), "users": 1480},
                {"cohort": "2024-12", "ltv": int(ltv_base * 1.2), "users": 1320},
                {"cohort": "2025-01", "ltv": int(ltv_base * 0.95), "users": 980},
            ]

        # 전환 퍼널 데이터 (weeks에 따른 전환율 조절)
        conv_mult = {7: 1.0, 30: 1.15, 90: 1.25}.get(days, 1.0)
        conversion = [
            {"cohort": "2025-01 W1", "registered": int(1000 * conv_mult), "activated": int(720 * conv_mult), "engaged": int(520 * conv_mult), "converted": int(180 * conv_mult), "retained": int(95 * conv_mult)},
            {"cohort": "2025-01 W2", "registered": int(1100 * conv_mult), "activated": int(780 * conv_mult), "engaged": int(560 * conv_mult), "converted": int(195 * conv_mult), "retained": int(102 * conv_mult)},
            {"cohort": "2025-01 W3", "registered": int(950 * conv_mult), "activated": int(680 * conv_mult), "engaged": int(480 * conv_mult), "converted": int(165 * conv_mult), "retained": int(88 * conv_mult)},
            {"cohort": "2025-01 W4", "registered": int(1050 * conv_mult), "activated": int(750 * conv_mult), "engaged": int(530 * conv_mult), "converted": int(185 * conv_mult), "retained": int(98 * conv_mult)},
        ]
        # weeks에 따라 표시할 코호트 수 조절
        conversion = conversion[:max(1, weeks)]

        return json_sanitize({
            "status": "SUCCESS",
            "retention": cohort_data,
            "ltv_by_cohort": ltv_by_cohort,
            "conversion": conversion,
        })
    except Exception as e:
        return {"status": "FAILED", "error": safe_str(e)}


@router.get("/analysis/trend/kpis")
def get_trend_kpis(days: int = 7, user: dict = Depends(verify_credentials)):
    """트렌드 KPI 분석 - 실제 데이터 기반"""
    try:
        # days 파라미터 유효성 검사
        if days not in [7, 30, 90]:
            days = 7

        # DAILY_METRICS_DF가 있으면 실제 데이터 사용
        if st.DAILY_METRICS_DF is not None and len(st.DAILY_METRICS_DF) > 0:
            df = st.DAILY_METRICS_DF.copy()

            # 선택된 기간만큼 데이터 필터링
            recent_df = df.tail(min(days, len(df)))
            daily_metrics = recent_df.to_dict("records")

            # 최근 값과 이전 기간 값 비교 (선택 기간의 절반 시점과 비교)
            current_row = recent_df.iloc[-1] if len(recent_df) > 0 else {}
            compare_idx = min(days // 2, len(recent_df) - 1)
            prev_row = recent_df.iloc[0] if len(recent_df) > compare_idx else {}

            dau = int(current_row.get("dau", 650))
            dau_prev = int(prev_row.get("dau", 580))
            arpu = int(current_row.get("arpu", 15420))
            arpu_prev = int(prev_row.get("arpu", 14200))
            new_users = int(current_row.get("new_users", 45))
            new_users_prev = int(prev_row.get("new_users", 52))
            avg_session = float(current_row.get("avg_session_minutes", 28))
            avg_session_prev = float(prev_row.get("avg_session_minutes", 25))

            kpis = [
                {"name": "DAU", "current": dau, "previous": dau_prev, "trend": "up" if dau >= dau_prev else "down", "change": round((dau - dau_prev) / max(dau_prev, 1) * 100, 1)},
                {"name": "ARPU", "current": arpu, "previous": arpu_prev, "trend": "up" if arpu >= arpu_prev else "down", "change": round((arpu - arpu_prev) / max(arpu_prev, 1) * 100, 1)},
                {"name": "신규가입", "current": new_users, "previous": new_users_prev, "trend": "up" if new_users >= new_users_prev else "down", "change": round((new_users - new_users_prev) / max(new_users_prev, 1) * 100, 1)},
                {"name": "이탈률", "current": 3.2, "previous": 4.1, "trend": "up", "change": -22.0},
                {"name": "세션시간", "current": round(avg_session, 1), "previous": round(avg_session_prev, 1), "trend": "up" if avg_session >= avg_session_prev else "down", "change": round((avg_session - avg_session_prev) / max(avg_session_prev, 1) * 100, 1)},
                {"name": "결제전환", "current": 4.8, "previous": 4.2, "trend": "up", "change": 14.3},
            ]

            # 예측 데이터 (최근 추세 기반)
            avg_dau = df["dau"].mean()
            forecast = [
                {"date": "02/01", "predicted_dau": int(avg_dau * 1.02), "lower": int(avg_dau * 0.95), "upper": int(avg_dau * 1.10)},
                {"date": "02/02", "predicted_dau": int(avg_dau * 1.04), "lower": int(avg_dau * 0.96), "upper": int(avg_dau * 1.12)},
                {"date": "02/03", "predicted_dau": int(avg_dau * 1.06), "lower": int(avg_dau * 0.97), "upper": int(avg_dau * 1.15)},
                {"date": "02/04", "predicted_dau": int(avg_dau * 1.08), "lower": int(avg_dau * 0.98), "upper": int(avg_dau * 1.18)},
                {"date": "02/05", "predicted_dau": int(avg_dau * 1.10), "lower": int(avg_dau * 0.99), "upper": int(avg_dau * 1.21)},
            ]
        else:
            # 폴백: 가상 데이터
            dau = 650
            dau_prev = 580
            daily_metrics = []
            for i in range(14):
                daily_metrics.append({
                    "date": f"01/{17 + i}",
                    "date_display": f"01/{17 + i}",
                    "dau": int(dau * (0.9 + 0.2 * (i / 14))),
                    "revenue": int(4500000 * (0.85 + 0.3 * (i / 14))),
                    "sessions": int(dau * 2.8),
                    "new_users": 45 + i,
                })
            kpis = [
                {"name": "DAU", "current": dau, "previous": dau_prev, "trend": "up", "change": 12.1},
                {"name": "ARPU", "current": 15420, "previous": 14200, "trend": "up", "change": 8.6},
                {"name": "신규가입", "current": 45, "previous": 52, "trend": "down", "change": -13.5},
                {"name": "이탈률", "current": 3.2, "previous": 4.1, "trend": "up", "change": -22.0},
                {"name": "세션시간", "current": 28, "previous": 25, "trend": "up", "change": 12.0},
                {"name": "결제전환", "current": 4.8, "previous": 4.2, "trend": "up", "change": 14.3},
            ]
            forecast = [
                {"date": "02/01", "predicted_dau": 665, "lower": 640, "upper": 690},
                {"date": "02/02", "predicted_dau": 672, "lower": 645, "upper": 699},
                {"date": "02/03", "predicted_dau": 678, "lower": 648, "upper": 708},
            ]

        # 상관관계 데이터
        correlation = [
            {"var1": "DAU", "var2": "매출", "correlation": 0.85},
            {"var1": "세션시간", "var2": "리텐션", "correlation": 0.72},
            {"var1": "레벨", "var2": "과금률", "correlation": 0.68},
            {"var1": "친구수", "var2": "DAU", "correlation": 0.54},
            {"var1": "이벤트참여", "var2": "매출", "correlation": 0.61},
        ]

        return json_sanitize({
            "status": "SUCCESS",
            "kpis": kpis,
            "daily_metrics": daily_metrics,
            "correlation": correlation,
            "forecast": forecast,
        })
    except Exception as e:
        return {"status": "FAILED", "error": safe_str(e)}


@router.get("/analysis/correlation")
def get_correlation_analysis(user: dict = Depends(verify_credentials)):
    """지표 상관관계 분석"""
    return json_sanitize({
        "status": "SUCCESS",
        "correlation": [
            {"var1": "DAU", "var2": "매출", "correlation": 0.85},
            {"var1": "DAU", "var2": "세션시간", "correlation": 0.72},
            {"var1": "매출", "var2": "과금유저", "correlation": 0.92},
            {"var1": "리텐션", "var2": "LTV", "correlation": 0.88},
            {"var1": "이벤트참여", "var2": "매출", "correlation": 0.65},
        ],
    })


@router.get("/stats/summary")
def get_summary_stats(days: int = 7, user: dict = Depends(verify_credentials)):
    """통계 요약 (분석 패널용) - CookieRun 데이터"""
    # days 파라미터 유효성 검사
    if days not in [7, 30, 90]:
        days = 7

    summary = {
        "status": "SUCCESS",
        "days": days,  # 선택된 기간 반환
        "cookies_count": len(st.COOKIES_DF) if st.COOKIES_DF is not None else 0,
        "kingdoms_count": len(st.KINGDOMS_DF) if st.KINGDOMS_DF is not None else 0,
        "users_count": len(st.USERS_DF) if st.USERS_DF is not None else 0,
        "translations_count": len(st.TRANSLATIONS_DF) if st.TRANSLATIONS_DF is not None else 0,
        "game_logs_count": len(st.GAME_LOGS_DF) if st.GAME_LOGS_DF is not None else 0,
    }

    # 기간별 활성 유저 수 계산 (USER_ACTIVITY_DF 기반)
    if st.USER_ACTIVITY_DF is not None and len(st.USER_ACTIVITY_DF) > 0:
        try:
            activity_df = st.USER_ACTIVITY_DF.tail(100 * days)  # 최근 N일 활동
            active_users = activity_df["user_id"].nunique()
            summary["active_users_in_period"] = active_users
            summary["active_user_ratio"] = round(active_users / summary["users_count"] * 100, 1) if summary["users_count"] > 0 else 0
        except Exception:
            pass

    # 번역 품질 평균
    if st.TRANSLATIONS_DF is not None and "quality_score" in st.TRANSLATIONS_DF.columns:
        summary["avg_translation_quality"] = round(float(st.TRANSLATIONS_DF["quality_score"].mean()), 1)

    # 등급별 쿠키 분포
    if st.COOKIES_DF is not None and "grade" in st.COOKIES_DF.columns:
        summary["grade_stats"] = st.COOKIES_DF["grade"].value_counts().to_dict()

    # 유저 세그먼트 분포 (이름으로 변환) - days 파라미터 반영
    segment_names = {
        0: "캐주얼 유저",
        1: "하드코어 게이머",
        2: "PvP 전문가",
        3: "콘텐츠 수집가",
        4: "신규 유저",
    }
    if st.USER_ANALYTICS_DF is not None and "cluster" in st.USER_ANALYTICS_DF.columns:
        # 기본 세그먼트 분포
        raw_segments = st.USER_ANALYTICS_DF["cluster"].value_counts().to_dict()
        summary["user_segments"] = {
            segment_names.get(k, f"세그먼트 {k}"): v for k, v in raw_segments.items()
        }

        # USER_ACTIVITY_DF가 있으면 기간별 세그먼트 활동 지표 계산
        if st.USER_ACTIVITY_DF is not None and len(st.USER_ACTIVITY_DF) > 0:
            try:
                activity_df = st.USER_ACTIVITY_DF.copy()
                # 최근 N일 활동만 필터링
                activity_df = activity_df.tail(100 * days)  # 100명 유저 * N일

                # 유저별 활동 집계
                user_activity = activity_df.groupby("user_id").agg({
                    "playtime": "mean",
                    "stages_cleared": "sum",
                    "gacha_pulls": "sum",
                    "pvp_battles": "sum",
                }).reset_index()

                # 세그먼트 정보 조인
                user_activity = user_activity.merge(
                    st.USER_ANALYTICS_DF[["user_id", "cluster"]],
                    on="user_id",
                    how="left"
                )

                # 세그먼트별 지표 계산
                segment_metrics = {}
                for cluster, name in segment_names.items():
                    seg_data = user_activity[user_activity["cluster"] == cluster]
                    if not seg_data.empty:
                        segment_metrics[name] = {
                            "count": int(raw_segments.get(cluster, 0)),
                            "avg_playtime": int(seg_data["playtime"].mean()),
                            "avg_stages": int(seg_data["stages_cleared"].mean()),
                            "avg_gacha": int(seg_data["gacha_pulls"].mean()),
                            "avg_pvp": int(seg_data["pvp_battles"].mean()),
                        }
                    else:
                        segment_metrics[name] = {
                            "count": int(raw_segments.get(cluster, 0)),
                            "avg_playtime": 0,
                            "avg_stages": 0,
                            "avg_gacha": 0,
                            "avg_pvp": 0,
                        }
                summary["segment_metrics"] = segment_metrics
            except Exception as e:
                st.logger.warning(f"세그먼트 지표 계산 실패: {e}")

    # 왕국별 쿠키 수
    if st.COOKIES_DF is not None and "kingdom" in st.COOKIES_DF.columns:
        summary["kingdom_cookies"] = st.COOKIES_DF["kingdom"].value_counts().to_dict()

    # 번역 언어 분포 및 상세 통계
    if st.TRANSLATION_STATS_DF is not None and len(st.TRANSLATION_STATS_DF) > 0:
        # 실제 통계 데이터 사용
        stats_list = st.TRANSLATION_STATS_DF.to_dict("records")
        summary["translation_langs"] = {
            row["lang_name"]: row["total_count"] for row in stats_list
        }
        summary["translation_stats_detail"] = stats_list
    elif st.TRANSLATIONS_DF is not None and "target_lang" in st.TRANSLATIONS_DF.columns:
        lang_names = {"en": "영어", "ja": "일본어", "zh": "중국어", "th": "태국어", "ko": "한국어"}
        raw_langs = st.TRANSLATIONS_DF["target_lang"].value_counts().to_dict()
        summary["translation_langs"] = {
            lang_names.get(k, k): v for k, v in raw_langs.items()
        }

    # 일별 활성 유저 트렌드 (최근 7일)
    if st.GAME_LOGS_DF is not None and "timestamp" in st.GAME_LOGS_DF.columns:
        try:
            import pandas as pd
            df = st.GAME_LOGS_DF.copy()
            df["date"] = pd.to_datetime(df["timestamp"]).dt.strftime("%m/%d")
            daily = df.groupby("date")["user_id"].nunique().tail(7)
            summary["daily_trend"] = [
                {"date": date, "active_users": int(count), "new_users": max(5, int(count * 0.08))}
                for date, count in daily.items()
            ]
        except Exception:
            pass

    return json_sanitize(summary)


# ============================================================
# RAG (검색)
# ============================================================
@router.post("/rag/search")
def search_rag(req: RagRequest, user: dict = Depends(verify_credentials)):
    return tool_rag_search(req.query, top_k=req.top_k, api_key=req.api_key)


@router.post("/rag/search/hybrid")
def search_rag_hybrid(req: HybridSearchRequest, user: dict = Depends(verify_credentials)):
    """
    고급 RAG 검색 (Hybrid Search)
    - BM25 (키워드) + Vector (의미) 조합
    - Cross-Encoder Reranking (선택)
    - Knowledge Graph 보강 (선택)
    """
    return rag_search_hybrid(
        query=req.query,
        top_k=req.top_k,
        api_key=req.api_key,
        use_reranking=req.use_reranking,
        use_kg=req.use_kg
    )


@router.get("/rag/status")
def rag_status(user: dict = Depends(verify_credentials)):
    with st.RAG_LOCK:
        return {
            "status": "SUCCESS",
            "rag_ready": bool(st.RAG_STORE.get("ready")),
            "docs_dir": st.RAG_DOCS_DIR,
            "faiss_dir": st.RAG_FAISS_DIR,
            "embed_model": st.RAG_EMBED_MODEL,
            "files_count": int(st.RAG_STORE.get("files_count") or st.RAG_STORE.get("docs_count") or 0),
            "chunks_count": int(st.RAG_STORE.get("chunks_count") or st.RAG_STORE.get("docs_count") or 0),
            "hash": safe_str(st.RAG_STORE.get("hash", "")),
            "last_build_ts": float(st.RAG_STORE.get("last_build_ts") or 0.0),
            "error": safe_str(st.RAG_STORE.get("error", "")),
            # Advanced RAG Features
            "bm25_available": BM25_AVAILABLE,
            "bm25_ready": bool(st.RAG_STORE.get("bm25_ready")),
            "reranker_available": False,  # 비활성화
            "kg_ready": False,  # 비활성화
            "kg_entities_count": 0,
            "kg_relations_count": 0,
        }


@router.post("/rag/reload")
def rag_reload(req: RagReloadRequest, user: dict = Depends(verify_credentials)):
    if user.get("role") != "관리자":
        raise HTTPException(status_code=403, detail="권한 없음")

    try:
        k = safe_str(req.api_key).strip() or st.OPENAI_API_KEY
        if not k:
            return {"status": "FAILED", "error": "OpenAI API Key가 설정되지 않았습니다."}

        rag_build_or_load_index(api_key=k, force_rebuild=bool(req.force))

        with st.RAG_LOCK:
            ok = bool(st.RAG_STORE.get("ready"))
            err = safe_str(st.RAG_STORE.get("error", ""))
            return {
                "status": "SUCCESS" if ok else "FAILED",
                "rag_ready": ok,
                "files_count": int(st.RAG_STORE.get("files_count") or st.RAG_STORE.get("docs_count") or 0),
                "chunks_count": int(st.RAG_STORE.get("chunks_count") or st.RAG_STORE.get("docs_count") or 0),
                "hash": safe_str(st.RAG_STORE.get("hash", "")),
                "error": err if err else ("인덱스 빌드 실패" if not ok else ""),
                "embed_model": st.RAG_EMBED_MODEL,
            }
    except Exception as e:
        st.logger.exception("RAG 재빌드 실패")
        return {"status": "FAILED", "error": f"RAG 재빌드 실패: {safe_str(e)}"}


@router.post("/rag/upload")
async def upload_rag_document(
    file: UploadFile = File(...),
    api_key: str = "",
    skip_reindex: bool = False,
    background_tasks: BackgroundTasks = None,
    user: dict = Depends(verify_credentials),
):
    """
    RAG 문서 업로드.
    skip_reindex=True로 설정하면 인덱스 재빌드를 건너뜁니다 (배치 업로드용).
    """
    try:
        filename = file.filename or "unknown"
        ext = os.path.splitext(filename)[1].lower()

        if ext not in st.RAG_ALLOWED_EXTS:
            return {"status": "FAILED", "error": f"지원하지 않는 파일 형식입니다. 허용된 형식: {', '.join(st.RAG_ALLOWED_EXTS)}"}

        MAX_FILE_SIZE = 15 * 1024 * 1024  # 15MB
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            return {"status": "FAILED", "error": "파일 크기는 15MB를 초과할 수 없습니다."}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(st.RAG_DOCS_DIR, safe_filename)

        os.makedirs(st.RAG_DOCS_DIR, exist_ok=True)

        with open(file_path, "wb") as f:
            f.write(contents)

        # 백그라운드에서 인덱스 재빌드 (skip_reindex가 False일 때만)
        if not skip_reindex:
            k = (api_key or "").strip() or st.OPENAI_API_KEY
            if k and background_tasks:
                background_tasks.add_task(rag_build_or_load_index, api_key=k, force_rebuild=True)

        return {
            "status": "SUCCESS",
            "message": "파일이 업로드되었습니다." + ("" if skip_reindex else " 인덱스 재빌드 중..."),
            "filename": safe_filename,
            "original_filename": filename,
            "size": len(contents),
            "path": os.path.relpath(file_path, st.BASE_DIR),
            "reindex_skipped": skip_reindex,
        }
    except Exception as e:
        st.logger.exception("파일 업로드 실패")
        return {"status": "FAILED", "error": f"파일 업로드 실패: {safe_str(e)}"}


@router.get("/rag/files")
def list_rag_files(user: dict = Depends(verify_credentials)):
    try:
        files_info = []
        paths = _rag_list_files()

        for p in paths:
            try:
                stat = os.stat(p)
                rel_path = os.path.relpath(p, st.RAG_DOCS_DIR).replace("\\", "/")
                files_info.append({
                    "filename": os.path.basename(p),
                    "path": rel_path,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "ext": os.path.splitext(p)[1].lower(),
                })
            except Exception:
                continue

        return {"status": "SUCCESS", "files": files_info, "total": len(files_info)}
    except Exception as e:
        st.logger.exception("파일 목록 조회 실패")
        return {"status": "FAILED", "error": f"파일 목록 조회 실패: {safe_str(e)}"}


@router.post("/rag/delete")
def delete_rag_file(
    req: DeleteFileRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(verify_credentials)
):
    if user.get("role") != "관리자":
        raise HTTPException(status_code=403, detail="권한 없음")

    try:
        filename = os.path.basename(req.filename)
        file_path = os.path.join(st.RAG_DOCS_DIR, filename)

        if not file_path.startswith(os.path.abspath(st.RAG_DOCS_DIR)):
            return {"status": "FAILED", "error": "잘못된 파일 경로입니다."}

        if not os.path.exists(file_path):
            return {"status": "FAILED", "error": "파일을 찾을 수 없습니다."}

        os.remove(file_path)

        # skip_reindex=True면 재빌드 건너뛰기 (다중 삭제 시 마지막에 한 번만 재빌드)
        if not req.skip_reindex:
            k = safe_str(req.api_key).strip() or st.OPENAI_API_KEY
            if k:
                background_tasks.add_task(rag_build_or_load_index, api_key=k, force_rebuild=True)
            return {"status": "SUCCESS", "message": "파일이 삭제되었습니다. 인덱스 재빌드 중...", "filename": filename}

        return {"status": "SUCCESS", "message": "파일이 삭제되었습니다.", "filename": filename}
    except Exception as e:
        st.logger.exception("파일 삭제 실패")
        return {"status": "FAILED", "error": f"파일 삭제 실패: {safe_str(e)}"}


# ============================================================
# LightRAG (듀얼 레벨 검색)
# ============================================================
@router.get("/lightrag/status")
def lightrag_status(user: dict = Depends(verify_credentials)):
    """LightRAG 상태 조회"""
    status = get_lightrag_status()
    return {"status": "SUCCESS", **status}


@router.post("/lightrag/build")
def lightrag_build(
    req: LightRagBuildRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(verify_credentials)
):
    """
    LightRAG 지식 그래프 빌드

    기존 RAG 문서(rag_docs/)에서 경량 지식 그래프를 구축합니다.
    - 99% 토큰 절감 (vs Microsoft GraphRAG)
    - 엔티티 및 테마 추출
    - 듀얼 레벨 검색 지원 (local/global/hybrid)
    """
    if user.get("role") != "관리자":
        raise HTTPException(status_code=403, detail="권한 없음")

    if not LIGHTRAG_AVAILABLE:
        return {
            "status": "FAILED",
            "error": "LightRAG가 설치되지 않았습니다. pip install lightrag-hku"
        }

    # 백그라운드에서 빌드
    background_tasks.add_task(build_lightrag_from_rag_docs, req.force_rebuild)

    return {
        "status": "SUCCESS",
        "message": "LightRAG 빌드가 시작되었습니다.",
    }


@router.post("/lightrag/search")
def lightrag_search(req: LightRagSearchRequest, user: dict = Depends(verify_credentials)):
    """
    LightRAG 검색 (듀얼 레벨)

    검색 모드:
    - naive: 기본 검색 (컨텍스트만)
    - local: Low-level 검색 (엔티티 중심) - 구체적인 질문에 적합
    - global: High-level 검색 (테마 중심) - 추상적인 질문에 적합
    - hybrid: local + global 조합 (권장)
    """
    if not LIGHTRAG_AVAILABLE:
        return {"status": "FAILED", "error": "LightRAG not available"}

    try:
        result = lightrag_search_sync(
            query=req.query,
            mode=req.mode,
            top_k=req.top_k
        )
        return result
    except Exception as e:
        st.logger.exception("LightRAG 검색 실패")
        return {"status": "FAILED", "error": f"LightRAG 검색 실패: {safe_str(e)}"}


@router.post("/lightrag/search-dual")
def lightrag_search_dual(req: LightRagSearchRequest, user: dict = Depends(verify_credentials)):
    """
    LightRAG 듀얼 검색 (Low-level + High-level 결과 모두 반환)

    Low-level (local): 구체적인 엔티티 정보
    - 예: "용감한 쿠키의 스킬은?" → 특정 쿠키 엔티티 검색

    High-level (global): 추상적인 테마/개념
    - 예: "쿠키런 세계관의 시대적 배경은?" → 세계관 테마 검색
    """
    if not LIGHTRAG_AVAILABLE:
        return {"status": "FAILED", "error": "LightRAG not available"}

    try:
        result = lightrag_search_dual_sync(
            query=req.query,
            top_k=req.top_k
        )
        return result
    except Exception as e:
        st.logger.exception("LightRAG 듀얼 검색 실패")
        return {"status": "FAILED", "error": f"LightRAG 듀얼 검색 실패: {safe_str(e)}"}


@router.post("/lightrag/clear")
def lightrag_clear_endpoint(user: dict = Depends(verify_credentials)):
    """LightRAG 초기화"""
    if user.get("role") != "관리자":
        raise HTTPException(status_code=403, detail="권한 없음")

    result = clear_lightrag()
    return result


# ============================================================
# K²RAG (KeyKnowledgeRAG - 2025)
# 논문: https://arxiv.org/abs/2507.07695
# 특징: KG + Hybrid Search + Summarization + Sub-question Pipeline
# ============================================================

@router.get("/k2rag/status")
def k2rag_status(user: dict = Depends(verify_credentials)):
    """
    K²RAG 상태 조회

    Returns:
        - initialized: 초기화 여부
        - chunks_count: 인덱싱된 청크 수
        - has_dense_store: Dense Vector Store 유무
        - has_sparse_store: Sparse Vector Store 유무
        - has_knowledge_graph: Knowledge Graph 유무
        - summarizer_available: Longformer 요약기 사용 가능 여부
        - config: 현재 설정
    """
    return k2rag_get_status()


@router.post("/k2rag/search")
def k2rag_search_endpoint(req: K2RagSearchRequest, user: dict = Depends(verify_credentials)):
    """
    K²RAG 검색 (KeyKnowledgeRAG Pipeline)

    Pipeline:
    1. Knowledge Graph 검색 → 관련 토픽 추출
    2. KG 결과 요약 (Longformer)
    3. 서브 질문 생성 (KG 요약 청킹)
    4. 각 서브 질문에 Hybrid Search (80% Dense + 20% Sparse)
    5. 서브 답변 생성 및 요약
    6. 최종 답변 생성

    Args:
        query: 검색 쿼리
        top_k: 검색 결과 수 (기본값: 10)
        use_kg: Knowledge Graph 사용 여부 (기본값: True)
        use_summary: 요약 사용 여부 (기본값: True)

    Returns:
        - status: SUCCESS/FAILED
        - answer: 최종 답변
        - context: 사용된 컨텍스트
        - kg_results: KG 검색 결과
        - sub_answers: 서브 질문/답변 목록
        - elapsed_ms: 처리 시간 (ms)
    """
    try:
        result = k2rag_search_sync(
            query=req.query,
            top_k=req.top_k,
            use_kg=req.use_kg,
            use_summary=req.use_summary
        )
        return result
    except Exception as e:
        st.logger.exception("K2RAG 검색 실패")
        return {"status": "FAILED", "error": f"K2RAG 검색 실패: {safe_str(e)}"}


@router.post("/k2rag/config")
def k2rag_config_endpoint(req: K2RagConfigRequest, user: dict = Depends(verify_credentials)):
    """
    K²RAG 설정 업데이트

    Args:
        hybrid_lambda: Hybrid Search 가중치 (0.0-1.0, 기본값: 0.8 = 80% Dense)
        top_k: 검색 결과 수 (기본값: 10)
        use_summarization: 요약 사용 여부
        use_knowledge_graph: KG 사용 여부
        llm_model: LLM 모델명 (기본값: gpt-4o-mini)

    Returns:
        - status: SUCCESS
        - config: 업데이트된 설정
    """
    if user.get("role") != "관리자":
        raise HTTPException(status_code=403, detail="권한 없음")

    config_updates = {}
    if req.hybrid_lambda is not None:
        config_updates["hybrid_lambda"] = req.hybrid_lambda
    if req.top_k is not None:
        config_updates["top_k"] = req.top_k
    if req.use_summarization is not None:
        config_updates["use_summarization"] = req.use_summarization
    if req.use_knowledge_graph is not None:
        config_updates["use_knowledge_graph"] = req.use_knowledge_graph
    if req.llm_model is not None:
        config_updates["llm_model"] = req.llm_model

    return k2rag_update_config(config_updates)


@router.post("/k2rag/load")
def k2rag_load_endpoint(user: dict = Depends(verify_credentials)):
    """
    기존 RAG 데이터를 K²RAG에 로드

    service.py의 FAISS, BM25, Knowledge Graph 데이터를 K²RAG에서 사용할 수 있도록 로드
    """
    try:
        success = k2rag_load_existing()
        if success:
            return {
                "status": "SUCCESS",
                "message": "기존 RAG 데이터가 K²RAG에 로드되었습니다.",
                "state": k2rag_get_status()
            }
        else:
            return {
                "status": "PARTIAL",
                "message": "일부 데이터만 로드되었습니다. 상태를 확인하세요.",
                "state": k2rag_get_status()
            }
    except Exception as e:
        return {"status": "FAILED", "error": safe_str(e)}


@router.post("/k2rag/summarize")
def k2rag_summarize_endpoint(
    text: str = Body(..., embed=True),
    max_length: int = Body(300, embed=True),
    user: dict = Depends(verify_credentials)
):
    """
    텍스트 요약 (Longformer LED 모델)

    K²RAG의 핵심 요약 기능을 단독으로 사용
    - 긴 텍스트 처리에 최적화 (최대 4096 토큰)
    - GPU 지원

    Args:
        text: 요약할 텍스트
        max_length: 최대 요약 길이 (기본값: 300)

    Returns:
        - status: SUCCESS
        - summary: 요약된 텍스트
        - original_length: 원본 길이
        - summary_length: 요약 길이
    """
    try:
        summary = k2rag_summarize(text, max_length=max_length)
        return {
            "status": "SUCCESS",
            "summary": summary,
            "original_length": len(text),
            "summary_length": len(summary),
            "reduction_rate": round((1 - len(summary) / len(text)) * 100, 1) if text else 0
        }
    except Exception as e:
        return {"status": "FAILED", "error": safe_str(e)}


# ============================================================
# OCR (이미지 → 텍스트 추출 → RAG 연동)
# ============================================================
OCR_ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif", ".webp"}


@router.post("/ocr/extract")
async def ocr_extract(
    file: UploadFile = File(...),
    api_key: str = "",
    save_to_rag: bool = True,
    user: dict = Depends(verify_credentials),
):
    """이미지에서 텍스트 추출 (EasyOCR) + RAG 연동"""
    global OCR_READER

    if not OCR_AVAILABLE:
        return {"status": "FAILED", "error": "OCR 라이브러리(easyocr)가 설치되지 않았습니다. pip install easyocr"}

    try:
        filename = file.filename or "unknown"
        ext = os.path.splitext(filename)[1].lower()

        if ext not in OCR_ALLOWED_EXTS:
            return {"status": "FAILED", "error": f"지원하지 않는 이미지 형식입니다. 허용된 형식: {', '.join(OCR_ALLOWED_EXTS)}"}

        MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            return {"status": "FAILED", "error": "파일 크기는 20MB를 초과할 수 없습니다."}

        # EasyOCR Reader 초기화 (Lazy loading - 첫 호출시만)
        if OCR_READER is None:
            st.logger.info("OCR_INIT: EasyOCR Reader 초기화 중...")
            OCR_READER = easyocr.Reader(['ko', 'en'], gpu=False)
            st.logger.info("OCR_INIT: EasyOCR Reader 초기화 완료")

        # OCR 수행
        result_list = OCR_READER.readtext(contents)
        extracted_text = "\n".join([text for _, text, _ in result_list])
        extracted_text = extracted_text.strip()

        if not extracted_text:
            return {"status": "FAILED", "error": "이미지에서 텍스트를 추출할 수 없습니다."}

        result = {
            "status": "SUCCESS",
            "original_filename": filename,
            "extracted_text": extracted_text,
            "text_length": len(extracted_text),
        }

        # RAG에 저장
        if save_to_rag:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            txt_filename = f"{timestamp}_ocr_{os.path.splitext(filename)[0]}.txt"
            txt_path = os.path.join(st.RAG_DOCS_DIR, txt_filename)

            os.makedirs(st.RAG_DOCS_DIR, exist_ok=True)

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"[OCR 추출 문서]\n")
                f.write(f"원본 파일: {filename}\n")
                f.write(f"추출 일시: {datetime.now().isoformat()}\n")
                f.write(f"{'='*50}\n\n")
                f.write(extracted_text)

            # RAG 인덱스 재빌드
            k = (api_key or "").strip() or st.OPENAI_API_KEY
            if k:
                rag_build_or_load_index(api_key=k, force_rebuild=True)

            result["saved_to_rag"] = True
            result["rag_filename"] = txt_filename
            result["message"] = "텍스트가 추출되어 RAG에 저장되었습니다."
        else:
            result["saved_to_rag"] = False
            result["message"] = "텍스트가 추출되었습니다."

        st.logger.info(f"OCR_EXTRACT file={filename} text_len={len(extracted_text)} saved_to_rag={save_to_rag}")
        return result

    except Exception as e:
        st.logger.exception("OCR 추출 실패")
        return {"status": "FAILED", "error": f"OCR 추출 실패: {safe_str(e)}"}


@router.get("/ocr/status")
def ocr_status(user: dict = Depends(verify_credentials)):
    """OCR 기능 상태 확인"""
    easyocr_version = None
    reader_initialized = False

    if OCR_AVAILABLE:
        try:
            easyocr_version = easyocr.__version__
            reader_initialized = OCR_READER is not None
        except Exception:
            pass

    return {
        "status": "SUCCESS",
        "ocr_available": OCR_AVAILABLE,
        "library": "EasyOCR",
        "version": easyocr_version,
        "reader_initialized": reader_initialized,
        "supported_formats": list(OCR_ALLOWED_EXTS),
        "supported_languages": ["ko", "en"],
    }


# ============================================================
# 에이전트 (동기/스트리밍)
# ============================================================
@router.post("/agent/chat")
def agent_chat(req: AgentRequest, user: dict = Depends(verify_credentials)):
    out = run_agent(req, username=user["username"])
    if isinstance(out, dict) and "status" not in out:
        out["status"] = "SUCCESS"
    return out


@router.post("/agent/memory/clear")
def clear_agent_memory(user: dict = Depends(verify_credentials)):
    clear_memory(user["username"])
    return {"status": "SUCCESS", "message": "메모리 초기화 완료"}


@router.post("/agent/stream")
async def agent_stream(req: AgentRequest, request: Request, user: dict = Depends(verify_credentials)):
    """
    LangGraph 기반 스트리밍 에이전트 (LangChain 1.x).
    - create_react_agent로 도구 호출
    - astream_events로 실시간 스트리밍
    """
    st.logger.info(
        "STREAM_REQ headers_auth=%s origin=%s ua=%s",
        request.headers.get("authorization"),
        request.headers.get("origin"),
        request.headers.get("user-agent"),
    )
    username = user["username"]

    async def gen():
        tool_calls_log = []
        final_buf = []

        try:
            from langgraph.prebuilt import create_react_agent
            from langchain_openai import ChatOpenAI
            from agent.tool_schemas import ALL_TOOLS
            from agent.router import classify_and_get_tools, IntentCategory

            user_text = safe_str(req.user_input)
            rag_mode = req.rag_mode or "auto"

            # ========== LLM Router: 의도 분류 → 도구 필터링 ==========
            api_key = pick_api_key(req.api_key)
            category, allowed_tool_names = classify_and_get_tools(
                user_text,
                api_key,
                use_llm_fallback=False,  # 스트리밍에서는 속도를 위해 키워드만 사용
            )

            st.logger.info(
                "STREAM_ROUTER category=%s allowed_tools=%s",
                category.value, allowed_tool_names,
            )

            # 카테고리에 해당하는 도구만 필터링
            if allowed_tool_names:
                tools = [t for t in ALL_TOOLS if t.name in allowed_tool_names]
                # 도구가 없으면 전체 사용 (fallback)
                if not tools:
                    tools = ALL_TOOLS
            elif category == IntentCategory.GENERAL:
                # 일반 대화: 도구 없음 (빠른 응답)
                tools = []
            else:
                tools = ALL_TOOLS

            # rag_mode에 따라 RAG 도구 추가 필터링 (세계관 카테고리인 경우만)
            if category == IntentCategory.WORLDVIEW:
                if rag_mode == "rag":
                    tools = [t for t in tools if t.name != "search_worldview_lightrag"]
                elif rag_mode == "lightrag":
                    tools = [t for t in tools if t.name != "search_worldview"]
                elif rag_mode == "k2rag":
                    # K²RAG 모드: 기존 도구 제거 (K²RAG API 직접 사용)
                    tools = [t for t in tools if t.name not in ["search_worldview", "search_worldview_lightrag"]]
                # auto: 둘 다 유지

            st.logger.info(
                "AGENT_TOOLS rag_mode=%s category=%s tools=%d (%s)",
                rag_mode, category.value, len(tools),
                [t.name for t in tools] if len(tools) <= 10 else f"{len(tools)} tools",
            )
            api_key = pick_api_key(req.api_key)

            if not api_key:
                msg = "처리 오류: OpenAI API Key가 없습니다. 환경변수 OPENAI_API_KEY 또는 요청의 api_key를 설정하세요."
                yield sse_pack("done", {"ok": False, "final": msg, "tool_calls": []})
                return

            # ========== 1. RAG 검색 (WORLDVIEW/GENERAL만) ==========
            # WORLDVIEW: 미리 RAG 검색 → 컨텍스트 포함 → 도구 제거 (이중 검색 방지)
            # GENERAL: 미리 RAG 검색 → 컨텍스트 제공
            rag_context = ""
            simple_patterns = ["안녕", "고마워", "감사", "뭐해", "ㅎㅎ", "ㅋㅋ", "네", "응", "오케이", "bye", "hi", "hello", "thanks"]
            is_simple = any(p in user_text.lower() for p in simple_patterns) and len(user_text) < 20

            # WORLDVIEW/GENERAL/COOKIE 미리 RAG 검색 (쿠키 정보도 RAG에서 검색)
            skip_rag = category not in [IntentCategory.WORLDVIEW, IntentCategory.GENERAL, IntentCategory.COOKIE]
            if skip_rag:
                st.logger.info("SKIP_RAG category=%s (not worldview/general/cookie)", category.value)

            if not is_simple and not skip_rag:
                try:
                    import time as _time
                    _rag_start = _time.time()

                    # rag_mode에 따라 적절한 RAG 검색 수행
                    if rag_mode == "lightrag":
                        # LightRAG 검색 - 설정은 state.LIGHTRAG_CONFIG에서 관리
                        rag_out = lightrag_search_sync(user_text, mode="hybrid")  # top_k는 state.py에서
                        _rag_elapsed = (_time.time() - _rag_start) * 1000
                        st.logger.info("LIGHTRAG_SEARCH_TIME elapsed=%.0fms", _rag_elapsed)
                        if isinstance(rag_out, dict) and rag_out.get("status") == "SUCCESS":
                            context_text = rag_out.get("context", "")
                            if context_text:
                                # 도구 결과에 context 일부 포함 (프론트엔드 표시용, 1000자 제한)
                                context_preview = context_text[:1000] + ("..." if len(context_text) > 1000 else "")
                                tool_calls_log.append({"tool": "lightrag_search", "args": {"query": user_text, "mode": "hybrid"}, "result": {"status": "SUCCESS", "context": context_preview, "context_len": len(context_text)}})
                                # 컨텍스트 크기 제한 (state.LIGHTRAG_CONFIG에서 관리)
                                max_chars = st.LIGHTRAG_CONFIG.get("context_max_chars", 1500)
                                rag_context = f"\n\n## 검색된 세계관 정보 (LightRAG):\n{context_text[:max_chars]}\n"
                                st.logger.info("LIGHTRAG_SEARCH ok=1 mode=hybrid ctx_len=%d truncated=%d", len(context_text), max_chars)
                                # 이미 검색했으므로 LLM에게 중복 검색 도구 제거 (이중 검색 방지)
                                tools = [t for t in tools if t.name != "search_worldview_lightrag"]
                    elif rag_mode == "k2rag":
                        # K²RAG 검색 - 고정밀 검색 (KG 요약 → 서브질문 → 하이브리드 검색)
                        rag_out = k2rag_search_sync(user_text, top_k=10, use_kg=True, use_summary=True)
                        _rag_elapsed = (_time.time() - _rag_start) * 1000
                        st.logger.info("K2RAG_SEARCH_TIME elapsed=%.0fms", _rag_elapsed)
                        if isinstance(rag_out, dict) and rag_out.get("status") == "SUCCESS":
                            answer = rag_out.get("answer", "")
                            context = rag_out.get("context", "")
                            # answer 또는 context 중 하나라도 있으면 성공
                            if answer or context:
                                tool_calls_log.append({"tool": "k2rag_search", "args": {"query": user_text}, "result": {"status": "SUCCESS", "answer_len": len(answer), "context_len": len(context)}})
                                rag_context = f"\n\n## 검색된 세계관 정보 (K²RAG):\n{answer or context[:2000]}\n"
                                st.logger.info("K2RAG_SEARCH ok=1 answer_len=%d context_len=%d", len(answer), len(context))
                                # 이미 검색했으므로 중복 검색 도구 제거
                                tools = [t for t in tools if t.name not in ["search_worldview", "search_worldview_lightrag"]]
                        else:
                            st.logger.warning("K2RAG_SEARCH_FAIL result=%s", rag_out)
                    else:
                        # 기본 RAG 검색 (rag 또는 auto 모드)
                        rag_out = tool_rag_search(user_text, top_k=st.RAG_DEFAULT_TOPK, api_key=api_key)
                        if isinstance(rag_out, dict) and rag_out.get("status") == "SUCCESS":
                            results = rag_out.get("results") or []
                            if results:
                                tool_calls_log.append({"tool": "rag_search", "args": {"query": user_text}, "result": rag_out})
                                rag_context = "\n\n## 검색된 세계관 정보 (참고용):\n"
                                for r in results[:5]:
                                    content = r.get("content", "")[:500]
                                    rag_context += f"- {content}\n"
                                st.logger.info("RAG_SEARCH ok=1 results=%d", len(results))
                                # 이미 검색했으므로 LLM에게 중복 검색 도구 제거 (이중 검색 방지)
                                tools = [t for t in tools if t.name != "search_worldview"]
                except Exception as _e:
                    st.logger.warning("RAG_SEARCH_FAIL err=%s", safe_str(_e))

            # ========== 2. 시스템 프롬프트 구성 ==========
            base_prompt = safe_str(req.system_prompt).strip() or DEFAULT_SYSTEM_PROMPT
            # RAG 모드에 따른 검색 도구 안내
            rag_tool_info = ""
            if rag_mode == "rag":
                rag_tool_info = "- `search_worldview`: 세계관 검색 (FAISS + BM25)"
            elif rag_mode == "lightrag":
                rag_tool_info = "- `search_worldview_lightrag`: 세계관 검색 (LightRAG - 지식그래프 기반)"
            elif rag_mode == "k2rag":
                rag_tool_info = "- K²RAG 모드: 검색이 자동으로 수행됩니다 (KG 요약 → 서브질문 → 하이브리드)"
            else:  # auto
                rag_tool_info = """- `search_worldview`: 세계관 검색 (FAISS + BM25)
- `search_worldview_lightrag`: 세계관 검색 (LightRAG - 관계 기반)"""

            system_prompt = base_prompt + f"""

## 도구 사용 규칙

당신은 쿠키런 킹덤 AI 어시스턴트입니다. 사용자 요청에 적합한 도구를 선택하여 호출하세요.

### 주요 도구:
- `get_cookie_info`, `list_cookies`: 쿠키 캐릭터 정보 (DB에 있는 쿠키만)
- `get_kingdom_info`, `list_kingdoms`: 왕국/지역 정보
- `translate_text`, `check_translation_quality`: 번역 관련
- `analyze_user`, `get_user_segment`, `detect_user_anomaly`: 유저 분석
- `predict_user_churn`: 유저 이탈 예측 (ML 모델)
- `get_cookie_win_rate`: 쿠키 PvP 승률 예측 (ML 모델)
- `optimize_investment`: 쿠키 투자 최적화 (P-PSO 알고리즘)
- `get_segment_statistics`, `get_anomaly_statistics`: 통계
- `get_dashboard_summary`: 대시보드 요약

### 세계관 검색 도구 (현재 모드: {rag_mode}):
{rag_tool_info}

### 규칙:
1. **쿠키 스킬, 스토리, 정보** 질문 → 먼저 **RAG 검색 도구**를 사용하세요. (DB에 없는 쿠키가 많습니다)
2. 사용자 요청에 맞는 도구를 **직접 선택**하세요.
3. 여러 정보가 필요하면 **여러 도구를 동시에 호출**하세요.
4. 도구 결과를 바탕으로 사용자에게 친절하게 답변하세요.
5. 간단한 인사나 대화에는 도구 호출 없이 바로 답변하세요.
6. 세계관, 스토리, 설정 관련 질문은 **검색 도구를 사용**하세요.
"""
            if rag_context:
                # RAG 컨텍스트가 있으면 엄격하게 참조하도록 지시 (할루시네이션 방지)
                system_prompt += f"""

## 🔍 검색된 세계관 정보 (공식 설정)
아래는 사용자 질문에 대해 검색된 **공식 설정 정보**입니다.

{rag_context}

### ⚠️ 답변 규칙 (엄격히 준수)
1. **검색 결과에 명시된 정보만** 사용하세요.
2. 검색 결과를 **확장, 추론, 일반화하지 마세요**.
3. 검색 결과에 없는 내용은 "검색 결과에서 확인되지 않습니다"라고 답하세요.
4. 답변 시 "검색 결과에 따르면" 형식으로 출처를 명시하세요.

❌ 금지: "~일 것입니다", "~로 추정됩니다", "일반적으로~"
✅ 권장: "검색 결과에 따르면 [정확한 인용]입니다."
"""
                st.logger.info("WORLDVIEW_RAG_CONTEXT_INJECTED len=%d", len(rag_context))

            # ========== 3. LangGraph Agent 생성 ==========
            model_name = req.model or "gpt-4o-mini"

            # tool_choice는 auto (기본값) - ReAct 에이전트에서 required는 무한 루프 유발
            llm = ChatOpenAI(
                model=model_name,
                api_key=api_key,
                streaming=True,
                max_tokens=req.max_tokens or 1500,
                temperature=req.temperature or 0.7,
            )

            # ========== 직접 LLM 응답 모드 ==========
            # 1. GENERAL 카테고리: 도구 없이 대화
            # 2. WORLDVIEW + RAG 컨텍스트: 에이전트 없이 RAG 기반 답변 (가장 정확)
            use_direct_llm = not tools or (category == IntentCategory.WORLDVIEW and rag_context)

            if use_direct_llm:
                mode = "WORLDVIEW_RAG_DIRECT" if category == IntentCategory.WORLDVIEW else "GENERAL"
                st.logger.info("STREAM_%s_MODE direct LLM response (no agent)", mode)

                # WORLDVIEW용 LLM (RAG 컨텍스트 엄격 준수)
                if category == IntentCategory.WORLDVIEW:
                    llm = ChatOpenAI(
                        model=model_name,
                        api_key=api_key,
                        streaming=True,
                        max_tokens=req.max_tokens or 1500,
                        temperature=0.2,  # 낮은 창의성: 검색 결과에 충실하게 답변
                    )

                # 직접 스트리밍 응답 (TTFT 측정)
                _llm_start = _time.time()
                _first_token = True
                async for chunk in llm.astream([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ]):
                    if await request.is_disconnected():
                        return
                    content = getattr(chunk, "content", "")
                    if content:
                        if _first_token:
                            _ttft = (_time.time() - _llm_start) * 1000
                            st.logger.info("LLM_TTFT elapsed=%.0fms", _ttft)
                            _first_token = False
                        final_buf.append(content)
                        yield sse_pack("delta", {"delta": content})  # 프론트엔드가 data.delta 기대

                _llm_total = (_time.time() - _llm_start) * 1000
                st.logger.info("LLM_COMPLETE elapsed=%.0fms tokens=%d", _llm_total, len(final_buf))
                full_response = "".join(final_buf)
                append_memory(username, user_text, full_response)
                yield sse_pack("done", {"ok": True, "final": full_response, "tool_calls": tool_calls_log})
                return

            # ========== 에이전트 모드 (WORLDVIEW 외 카테고리) ==========
            # LangGraph의 create_react_agent 사용 (Router 결과에 따라 필터링된 도구)
            agent = create_react_agent(llm, tools, prompt=system_prompt)

            # ========== 4. astream_events로 실시간 스트리밍 ==========
            current_tool = None

            async for event in agent.astream_events(
                {"messages": [("user", user_text)]},
                version="v2",
            ):
                if await request.is_disconnected():
                    return

                kind = event.get("event", "")
                data = event.get("data", {})

                # 도구 실행 시작
                if kind == "on_tool_start":
                    tool_name = event.get("name", "도구")
                    tool_input = data.get("input", {})
                    current_tool = tool_name
                    st.logger.info("TOOL_START tool=%s", tool_name)
                    yield sse_pack("tool_start", {"tool": tool_name, "args": tool_input})

                # 도구 실행 완료
                elif kind == "on_tool_end":
                    tool_output = data.get("output", {})
                    # ToolMessage 객체인 경우 content만 추출
                    if hasattr(tool_output, "content"):
                        content = tool_output.content
                        # content가 JSON 문자열이면 파싱, 딕셔너리/리스트면 그대로
                        if isinstance(content, str):
                            try:
                                tool_output = json.loads(content)
                            except (json.JSONDecodeError, TypeError):
                                tool_output = {"status": "SUCCESS", "data": content}
                        elif isinstance(content, (dict, list)):
                            tool_output = content
                        else:
                            tool_output = {"status": "SUCCESS", "data": safe_str(content)}
                    elif not isinstance(tool_output, (str, dict, list, int, float, bool, type(None))):
                        tool_output = {"status": "SUCCESS", "data": safe_str(tool_output)}
                    tool_calls_log.append({
                        "tool": current_tool or "unknown",
                        "result": tool_output,
                    })
                    st.logger.info("TOOL_END tool=%s", current_tool)
                    yield sse_pack("tool_end", {"tool": current_tool, "status": "SUCCESS"})
                    current_tool = None

                # LLM 토큰 스트리밍
                elif kind == "on_chat_model_stream":
                    chunk = data.get("chunk")
                    if chunk:
                        content = getattr(chunk, "content", "")
                        if isinstance(content, str) and content:
                            final_buf.append(content)
                            yield sse_pack("delta", {"delta": content})

            final_text = "".join(final_buf).strip()
            if not final_text:
                final_text = "요청을 처리했습니다."

            append_memory(username, user_text, final_text)

            st.logger.info("STREAM_COMPLETE user=%s tools_used=%d", username, len(tool_calls_log))
            yield sse_pack("done", {"ok": True, "final": final_text, "tool_calls": tool_calls_log})
            return

        except Exception as e:
            msg = safe_str(e) or "스트리밍 오류"
            st.logger.exception("STREAM_ERROR err=%s", msg)
            try:
                yield sse_pack("error", {"message": msg})
            except Exception:
                pass
            yield sse_pack("done", {"ok": False, "final": msg, "tool_calls": tool_calls_log})
            return

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)


# ============================================================
# MLflow 관련
# ============================================================
@router.get("/mlflow/experiments")
def get_mlflow_experiments(user: dict = Depends(verify_credentials)):
    """MLflow 실험 목록 조회"""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        # mlruns 폴더 경로 (ml 폴더 안에 있음)
        ml_mlruns = os.path.join(st.BASE_DIR, "ml", "mlruns")
        backend_mlruns = os.path.join(st.BASE_DIR, "mlruns")
        project_mlruns = os.path.abspath(os.path.join(st.BASE_DIR, "..", "mlruns"))

        # 우선순위: ml/mlruns > backend/mlruns > ../mlruns
        if os.path.exists(ml_mlruns):
            tracking_uri = f"file:{ml_mlruns}"
        elif os.path.exists(backend_mlruns):
            tracking_uri = f"file:{backend_mlruns}"
        elif os.path.exists(project_mlruns):
            tracking_uri = f"file:{project_mlruns}"
        else:
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()

        experiments = client.search_experiments()
        result = []

        for exp in experiments:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["start_time DESC"],
                max_results=10
            )

            runs_data = []
            for run in runs:
                runs_data.append({
                    "run_id": run.info.run_id,
                    "run_name": run.info.run_name,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "params": dict(run.data.params),
                    "metrics": {k: round(v, 4) for k, v in run.data.metrics.items()},
                    "tags": dict(run.data.tags),
                })

            result.append({
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "artifact_location": exp.artifact_location,
                "lifecycle_stage": exp.lifecycle_stage,
                "runs": runs_data,
            })

        return {"status": "SUCCESS", "data": result}
    except ImportError:
        return {"status": "FAILED", "error": "MLflow가 설치되지 않았습니다.", "data": []}
    except Exception as e:
        st.logger.exception("MLflow 조회 실패")
        return {"status": "FAILED", "error": safe_str(e), "data": []}


@router.get("/mlflow/models")
def get_mlflow_registered_models(user: dict = Depends(verify_credentials)):
    """MLflow Model Registry에서 등록된 모델 목록 조회"""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        # mlruns 폴더 경로 (ml 폴더 안에 있음)
        ml_mlruns = os.path.join(st.BASE_DIR, "ml", "mlruns")
        backend_mlruns = os.path.join(st.BASE_DIR, "mlruns")
        project_mlruns = os.path.abspath(os.path.join(st.BASE_DIR, "..", "mlruns"))

        # 우선순위: ml/mlruns > backend/mlruns > ../mlruns
        if os.path.exists(ml_mlruns):
            tracking_uri = f"file:{ml_mlruns}"
        elif os.path.exists(backend_mlruns):
            tracking_uri = f"file:{backend_mlruns}"
        elif os.path.exists(project_mlruns):
            tracking_uri = f"file:{project_mlruns}"
        else:
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()

        registered_models = client.search_registered_models()
        result = []

        for rm in registered_models:
            versions = []
            try:
                all_versions = client.search_model_versions(filter_string=f"name='{rm.name}'")
                for v in sorted(all_versions, key=lambda x: int(x.version), reverse=True):
                    versions.append({
                        "version": v.version,
                        "stage": v.current_stage,
                        "status": v.status,
                        "run_id": v.run_id,
                        "source": v.source,
                        "creation_timestamp": v.creation_timestamp,
                    })
            except Exception:
                for v in rm.latest_versions:
                    versions.append({
                        "version": v.version,
                        "stage": v.current_stage,
                        "status": v.status,
                        "run_id": v.run_id,
                        "source": v.source,
                        "creation_timestamp": v.creation_timestamp,
                    })

            result.append({
                "name": rm.name,
                "creation_timestamp": rm.creation_timestamp,
                "last_updated_timestamp": rm.last_updated_timestamp,
                "description": rm.description or "",
                "versions": versions,
                "model_type": "registry",
            })

        return {"status": "SUCCESS", "data": result}
    except ImportError:
        return {"status": "FAILED", "error": "MLflow가 설치되지 않았습니다.", "data": []}
    except Exception as e:
        st.logger.exception("MLflow 모델 조회 실패")
        return {"status": "FAILED", "error": safe_str(e), "data": []}


class ModelSelectRequest(BaseModel):
    model_name: str
    version: str


@router.get("/mlflow/models/selected")
def get_selected_models(user: dict = Depends(verify_credentials)):
    """현재 서버에서 선택/로드된 모델 목록 조회"""
    # 저장된 선택 상태 로드
    st.load_selected_models()

    return {
        "status": "SUCCESS",
        "data": st.SELECTED_MODELS,
        "message": f"{len(st.SELECTED_MODELS)}개 모델이 선택되어 있습니다"
    }


@router.post("/mlflow/models/select")
def select_mlflow_model(req: ModelSelectRequest, user: dict = Depends(verify_credentials)):
    """MLflow 모델 선택/로드 - 실제로 모델을 state에 로드"""

    # 모델 이름 → state 변수 매핑 (한글 이름)
    MODEL_STATE_MAP = {
        "번역품질예측": "TRANSLATION_MODEL",
        "텍스트분류": "TEXT_CATEGORY_MODEL",
        "유저세그먼트": "USER_SEGMENT_MODEL",
        "이상탐지": "ANOMALY_MODEL",
        "이탈예측": "CHURN_MODEL",
        "승률예측": "WIN_RATE_MODEL",
    }

    state_attr = MODEL_STATE_MAP.get(req.model_name)
    if not state_attr:
        return {
            "status": "FAILED",
            "error": f"알 수 없는 모델: {req.model_name}. 지원 모델: {list(MODEL_STATE_MAP.keys())}"
        }

    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        # mlruns 폴더 경로 (ml 폴더 안에 있음)
        ml_mlruns = os.path.join(st.BASE_DIR, "ml", "mlruns")
        backend_mlruns = os.path.join(st.BASE_DIR, "mlruns")
        project_mlruns = os.path.abspath(os.path.join(st.BASE_DIR, "..", "mlruns"))

        # 우선순위: ml/mlruns > backend/mlruns > ../mlruns
        if os.path.exists(ml_mlruns):
            tracking_uri = f"file:{ml_mlruns}"
        elif os.path.exists(backend_mlruns):
            tracking_uri = f"file:{backend_mlruns}"
        elif os.path.exists(project_mlruns):
            tracking_uri = f"file:{project_mlruns}"
        else:
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()

        try:
            # 모델 버전 정보 조회
            model_version = client.get_model_version(req.model_name, req.version)

            # 실제 모델 로드
            model_uri = f"models:/{req.model_name}/{req.version}"
            st.logger.info(f"모델 로드 시작: {model_uri}")

            loaded_model = mlflow.sklearn.load_model(model_uri)

            # state에 모델 할당
            setattr(st, state_attr, loaded_model)
            st.logger.info(f"모델 로드 완료: st.{state_attr} = {model_uri}")

            # 선택 상태 저장 (영구 보존)
            st.SELECTED_MODELS[req.model_name] = req.version
            st.save_selected_models()

            return {
                "status": "SUCCESS",
                "message": f"{req.model_name} v{req.version} 모델이 로드되었습니다",
                "data": {
                    "model_name": req.model_name,
                    "version": req.version,
                    "stage": model_version.current_stage,
                    "run_id": model_version.run_id,
                    "state_variable": f"st.{state_attr}",
                    "loaded": True,
                }
            }
        except Exception as e:
            st.logger.warning(f"MLflow 모델 로드 실패: {e}")
            return {
                "status": "FAILED",
                "error": f"모델 로드 실패: {safe_str(e)}",
                "data": {
                    "model_name": req.model_name,
                    "version": req.version,
                }
            }

    except ImportError:
        return {
            "status": "FAILED",
            "error": "MLflow가 설치되지 않았습니다. pip install mlflow 로 설치하세요.",
        }
    except Exception as e:
        st.logger.exception("MLflow 모델 선택 실패")
        return {"status": "FAILED", "error": safe_str(e)}


# ============================================================
# 사용자 관리
# ============================================================
@router.get("/users")
def get_users(user: dict = Depends(verify_credentials)):
    if user["role"] != "관리자":
        raise HTTPException(status_code=403, detail="권한 없음")
    return {"status": "SUCCESS", "data": [{"아이디": k, "이름": v["name"], "권한": v["role"]} for k, v in st.USERS.items()]}


@router.post("/users")
def create_user(req: UserCreateRequest, user: dict = Depends(verify_credentials)):
    if user["role"] != "관리자":
        raise HTTPException(status_code=403, detail="권한 없음")
    if req.user_id in st.USERS:
        raise HTTPException(status_code=400, detail="이미 존재하는 아이디")
    st.USERS[req.user_id] = {"password": req.password, "role": req.role, "name": req.name}
    return {"status": "SUCCESS", "message": f"{req.name} 추가됨"}


# ============================================================
# 설정
# ============================================================
@router.get("/settings/default")
def get_default_settings(user: dict = Depends(verify_credentials)):
    return {
        "status": "SUCCESS",
        "data": {
            "selectedModel": "gpt-4o-mini",
            "maxTokens": 8000,
            "temperature": 0.1,  # 할루시네이션 방지를 위해 낮게 설정
            "topP": 1.0,
            "presencePenalty": 0.0,
            "frequencyPenalty": 0.0,
            "seed": "",
            "timeoutMs": 30000,
            "retries": 2,
            "stream": True,
            "systemPrompt": st.get_active_system_prompt(),  # 백엔드 중앙 관리 프롬프트
        },
    }


# ============================================================
# 시스템 프롬프트 관리 API (백엔드 중앙 관리)
# ============================================================
@router.get("/settings/prompt")
def get_system_prompt(user: dict = Depends(verify_credentials)):
    """현재 활성 시스템 프롬프트 조회"""
    return {
        "status": "SUCCESS",
        "data": {
            "systemPrompt": st.get_active_system_prompt(),
            "isCustom": st.CUSTOM_SYSTEM_PROMPT is not None and st.CUSTOM_SYSTEM_PROMPT.strip() != "",
            "defaultPrompt": DEFAULT_SYSTEM_PROMPT,
        },
    }


class SystemPromptRequest(BaseModel):
    system_prompt: str = Field(..., alias="systemPrompt")

    class Config:
        populate_by_name = True


class LLMSettingsRequest(BaseModel):
    """LLM 설정 요청 모델"""
    selected_model: str = Field("gpt-4o-mini", alias="selectedModel")
    custom_model: str = Field("", alias="customModel")
    temperature: float = Field(0.7, alias="temperature")
    top_p: float = Field(1.0, alias="topP")
    presence_penalty: float = Field(0.0, alias="presencePenalty")
    frequency_penalty: float = Field(0.0, alias="frequencyPenalty")
    max_tokens: int = Field(8000, alias="maxTokens")
    seed: Optional[int] = Field(None, alias="seed")
    timeout_ms: int = Field(30000, alias="timeoutMs")
    retries: int = Field(2, alias="retries")
    stream: bool = Field(True, alias="stream")

    class Config:
        populate_by_name = True


@router.post("/settings/prompt")
def save_system_prompt(req: SystemPromptRequest, user: dict = Depends(verify_credentials)):
    """시스템 프롬프트 저장 (백엔드에 영구 저장)"""
    # 관리자만 수정 가능
    if user.get("role") != "관리자":
        raise HTTPException(status_code=403, detail="관리자만 시스템 프롬프트를 수정할 수 있습니다.")

    prompt = req.system_prompt.strip() if req.system_prompt else ""

    if not prompt:
        raise HTTPException(status_code=400, detail="시스템 프롬프트가 비어있습니다.")

    success = st.save_system_prompt(prompt)

    if success:
        return {
            "status": "SUCCESS",
            "message": "시스템 프롬프트가 저장되었습니다.",
            "data": {
                "systemPrompt": st.get_active_system_prompt(),
                "isCustom": True,
            },
        }
    else:
        raise HTTPException(status_code=500, detail="시스템 프롬프트 저장에 실패했습니다.")


@router.post("/settings/prompt/reset")
def reset_system_prompt(user: dict = Depends(verify_credentials)):
    """시스템 프롬프트를 기본값으로 초기화"""
    # 관리자만 초기화 가능
    if user.get("role") != "관리자":
        raise HTTPException(status_code=403, detail="관리자만 시스템 프롬프트를 초기화할 수 있습니다.")

    success = st.reset_system_prompt()

    if success:
        return {
            "status": "SUCCESS",
            "message": "시스템 프롬프트가 기본값으로 초기화되었습니다.",
            "data": {
                "systemPrompt": DEFAULT_SYSTEM_PROMPT,
                "isCustom": False,
            },
        }
    else:
        raise HTTPException(status_code=500, detail="시스템 프롬프트 초기화에 실패했습니다.")


# ============================================================
# LLM 설정 관리 API (백엔드 중앙 관리)
# ============================================================
@router.get("/settings/llm")
def get_llm_settings(user: dict = Depends(verify_credentials)):
    """현재 LLM 설정 조회"""
    settings = st.get_active_llm_settings()
    is_custom = st.CUSTOM_LLM_SETTINGS is not None
    return {
        "status": "SUCCESS",
        "data": {
            **settings,
            "isCustom": is_custom,
        },
    }


@router.post("/settings/llm")
def save_llm_settings(req: LLMSettingsRequest, user: dict = Depends(verify_credentials)):
    """LLM 설정 저장 (백엔드에 영구 저장)"""
    # 관리자만 수정 가능
    if user.get("role") != "관리자":
        raise HTTPException(status_code=403, detail="관리자만 LLM 설정을 수정할 수 있습니다.")

    settings_dict = {
        "selectedModel": req.selected_model,
        "customModel": req.custom_model,
        "temperature": req.temperature,
        "topP": req.top_p,
        "presencePenalty": req.presence_penalty,
        "frequencyPenalty": req.frequency_penalty,
        "maxTokens": req.max_tokens,
        "seed": req.seed,
        "timeoutMs": req.timeout_ms,
        "retries": req.retries,
        "stream": req.stream,
    }

    success = st.save_llm_settings(settings_dict)

    if success:
        return {
            "status": "SUCCESS",
            "message": "LLM 설정이 저장되었습니다.",
            "data": {
                **settings_dict,
                "isCustom": True,
            },
        }
    else:
        raise HTTPException(status_code=500, detail="LLM 설정 저장에 실패했습니다.")


@router.post("/settings/llm/reset")
def reset_llm_settings(user: dict = Depends(verify_credentials)):
    """LLM 설정을 기본값으로 초기화"""
    # 관리자만 초기화 가능
    if user.get("role") != "관리자":
        raise HTTPException(status_code=403, detail="관리자만 LLM 설정을 초기화할 수 있습니다.")

    success = st.reset_llm_settings()

    if success:
        return {
            "status": "SUCCESS",
            "message": "LLM 설정이 기본값으로 초기화되었습니다.",
            "data": {
                **st.DEFAULT_LLM_SETTINGS,
                "isCustom": False,
            },
        }
    else:
        raise HTTPException(status_code=500, detail="LLM 설정 초기화에 실패했습니다.")


# ============================================================
# 내보내기
# ============================================================
@router.get("/export/csv")
def export_csv(user: dict = Depends(verify_credentials)):
    """번역 데이터 CSV 내보내기"""
    output = StringIO()
    export_df = st.TRANSLATIONS_DF.copy() if st.TRANSLATIONS_DF is not None else pd.DataFrame()
    export_df.to_csv(output, index=False, encoding="utf-8-sig")
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=translations_{datetime.now().strftime('%Y%m%d')}.csv"},
    )


@router.get("/export/excel")
def export_excel(user: dict = Depends(verify_credentials)):
    """번역 데이터 Excel 내보내기"""
    output = BytesIO()
    export_df = st.TRANSLATIONS_DF.copy() if st.TRANSLATIONS_DF is not None else pd.DataFrame()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Translations")
    output.seek(0)
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=translations_{datetime.now().strftime('%Y%m%d')}.xlsx"},
    )


# ============================================================
# 도구 목록 (에이전트용)
# ============================================================
@router.get("/tools")
def get_available_tools(user: dict = Depends(verify_credentials)):
    """사용 가능한 도구 목록"""
    tools = []
    for name, func in AVAILABLE_TOOLS.items():
        tools.append({
            "name": name,
            "description": func.__doc__ or "",
        })
    return {"status": "SUCCESS", "tools": tools}


# ============================================================
# 팀 구성 최적화 (ML + PSO)
# ============================================================
@router.post("/team/optimize")
def optimize_team(req: TeamOptimizeRequest, user: dict = Depends(verify_credentials)):
    """
    팀 구성 최적화 API

    ML 모델로 팀 시너지를 예측하고 PSO로 최적 조합을 탐색합니다.

    Request:
        - owned_cookies: 보유 쿠키 ID 리스트 (선택)
        - required_roles: 필수 역할 (예: {"치유": 1, "방어": 1})
        - max_iterations: PSO 최대 반복 횟수

    Returns:
        - team: 최적 팀 구성 (5명)
        - predicted_score: 예상 팀 성능 점수
        - synergy_score: 시너지 점수
        - role_balance: 역할 분포
    """
    try:
        from ml.team_optimizer import optimize_team as run_optimize

        # cookie_stats.csv 확인
        if st.COOKIE_STATS_DF is None or len(st.COOKIE_STATS_DF) == 0:
            return {
                "status": "FAILED",
                "error": "cookie_stats.csv 데이터가 없습니다. 데이터를 먼저 로드하세요.",
            }

        # 최적화 실행
        result = run_optimize(
            cookies_df=st.COOKIE_STATS_DF,
            owned_cookies=req.owned_cookies,
            required_roles=req.required_roles,
            model=None,  # 휴리스틱 사용
        )

        st.logger.info(
            f"TEAM_OPTIMIZE user={user['username']} "
            f"score={result.get('predicted_score', 0):.1f} "
            f"synergy={result.get('synergy_score', 0):.1f}"
        )

        return {
            "status": "SUCCESS",
            "data": result,
        }

    except ImportError as e:
        st.logger.error(f"팀 최적화 모듈 로드 실패: {e}")
        return {
            "status": "FAILED",
            "error": f"팀 최적화 모듈 로드 실패: {safe_str(e)}",
        }
    except Exception as e:
        st.logger.exception("팀 최적화 실패")
        return {
            "status": "FAILED",
            "error": f"팀 최적화 중 오류: {safe_str(e)}",
        }


@router.get("/team/cookies")
def get_available_cookies(user: dict = Depends(verify_credentials)):
    """최적화에 사용 가능한 쿠키 목록 조회"""
    try:
        if st.COOKIE_STATS_DF is None or len(st.COOKIE_STATS_DF) == 0:
            return {
                "status": "FAILED",
                "error": "cookie_stats.csv 데이터가 없습니다.",
                "data": [],
            }

        cookies = []
        for _, row in st.COOKIE_STATS_DF.iterrows():
            cookies.append({
                "cookie_id": row.get("cookie_id", ""),
                "name": row.get("cookie_name", ""),
                "grade": row.get("grade", ""),
                "type": row.get("type", ""),
                "power_score": float(row.get("power_score", 0)),
                "win_rate_pvp": float(row.get("win_rate_pvp", 0)),
                "pick_rate_pvp": float(row.get("pick_rate_pvp", 0)),
            })

        return {
            "status": "SUCCESS",
            "data": cookies,
            "count": len(cookies),
        }

    except Exception as e:
        st.logger.exception("쿠키 목록 조회 실패")
        return {
            "status": "FAILED",
            "error": safe_str(e),
            "data": [],
        }


@router.get("/team/optimizer/status")
def get_optimizer_status(user: dict = Depends(verify_credentials)):
    """팀 최적화 모델 상태 조회"""
    try:
        # mealpy (P-PSO) 설치 여부 확인
        try:
            import mealpy
            pso_available = True
            pso_version = getattr(mealpy, "__version__", "unknown")
        except ImportError:
            pso_available = False
            pso_version = None

        # XGBoost 설치 여부 확인
        try:
            import xgboost
            xgb_available = True
            xgb_version = xgboost.__version__
        except ImportError:
            xgb_available = False
            xgb_version = None

        return {
            "status": "SUCCESS",
            "data": {
                "optimizer_available": st.INVESTMENT_OPTIMIZER_AVAILABLE,
                "cookie_stats_loaded": st.COOKIE_STATS_DF is not None,
                "cookie_count": len(st.COOKIE_STATS_DF) if st.COOKIE_STATS_DF is not None else 0,
                "dependencies": {
                    "mealpy": {
                        "available": pso_available,
                        "version": pso_version,
                    },
                    "xgboost": {
                        "available": xgb_available,
                        "version": xgb_version,
                    },
                },
                "optimization_method": "P-PSO (Phasor Particle Swarm Optimization)",
                "prediction_method": "LightGBM + P-PSO" if st.INVESTMENT_OPTIMIZER_AVAILABLE else "Heuristic",
            },
        }

    except Exception as e:
        st.logger.exception("최적화 상태 조회 실패")
        return {
            "status": "FAILED",
            "error": safe_str(e),
        }


# ============================================================
# 쿠키 밸런스 분석 (P-PSO 기반)
# ============================================================
class BalanceOptimizeRequest(BaseModel):
    """밸런스 최적화 요청 모델"""
    target_cookies: Optional[List[str]] = Field(None, alias="targetCookies", description="조정 대상 쿠키 ID 목록")
    adjust_stats: Optional[List[str]] = Field(None, alias="adjustStats", description="조정할 스탯 목록 (atk, hp, def, skill_dmg, cooldown, crit_rate, crit_dmg)")
    max_stat_change_pct: float = Field(15.0, alias="maxStatChangePct", description="최대 스탯 조정 비율 (%)")
    max_iterations: int = Field(10, alias="maxIterations", description="P-PSO 최대 반복 횟수")
    class Config:
        populate_by_name = True


@router.get("/balance/analysis")
def analyze_balance(user: dict = Depends(verify_credentials)):
    """
    쿠키 밸런스 현황 분석 API

    Returns:
        - balance_index: 전체 밸런스 지수 (0-100)
        - meta_diversity: 메타 다양성 지수
        - op_cookies: OP 쿠키 목록
        - up_cookies: UP 쿠키 목록
        - nerf_candidates: 너프 후보
        - buff_candidates: 버프 후보
        - statistics: 통계 정보
    """
    try:
        from ml.balance_optimizer import analyze_cookie_balance

        if st.COOKIE_STATS_DF is None or len(st.COOKIE_STATS_DF) == 0:
            return {
                "status": "FAILED",
                "error": "cookie_stats.csv 데이터가 없습니다.",
            }

        result = analyze_cookie_balance(st.COOKIE_STATS_DF)

        st.logger.info(
            f"BALANCE_ANALYSIS user={user['username']} "
            f"balance={result.get('balance_index', 0):.1f} "
            f"diversity={result.get('meta_diversity', 0):.1f} "
            f"op={result.get('op_count', 0)} up={result.get('up_count', 0)}"
        )

        return {
            "status": "SUCCESS",
            "data": result,
        }

    except ImportError as e:
        st.logger.error(f"밸런스 분석 모듈 로드 실패: {e}")
        return {
            "status": "FAILED",
            "error": f"밸런스 분석 모듈 로드 실패: {safe_str(e)}",
        }
    except Exception as e:
        st.logger.exception("밸런스 분석 실패")
        return {
            "status": "FAILED",
            "error": f"밸런스 분석 중 오류: {safe_str(e)}",
        }


@router.post("/balance/optimize")
def optimize_balance(req: BalanceOptimizeRequest, user: dict = Depends(verify_credentials)):
    """
    P-PSO 기반 밸런스 스탯 최적화 API

    쿠키 밸런스를 최적화하기 위한 스탯 조정 수치를 계산합니다.

    Request:
        - target_cookies: 조정 대상 쿠키 ID 목록 (선택, 미지정 시 자동 선택)
        - adjust_stats: 조정할 스탯 목록 (atk, hp, def, skill_dmg, cooldown, crit_rate, crit_dmg)
        - max_stat_change_pct: 최대 스탯 조정 비율 (%, 기본값 15)
        - max_iterations: P-PSO 최대 반복 횟수 (기본값 100)

    Returns:
        - adjustments: 쿠키별 스탯 조정 수치 (stat_changes 포함)
        - before_balance: 조정 전 밸런스 지수
        - after_balance: 조정 후 예상 밸런스 지수
        - improvement: 개선율
        - adjusted_stats: 조정된 스탯 목록
    """
    try:
        from ml.balance_optimizer import optimize_balance as run_optimize

        if st.COOKIE_STATS_DF is None or len(st.COOKIE_STATS_DF) == 0:
            return {
                "status": "FAILED",
                "error": "cookie_stats.csv 데이터가 없습니다.",
            }

        result = run_optimize(
            cookies_df=st.COOKIE_STATS_DF,
            target_cookies=req.target_cookies,
            adjust_stats=req.adjust_stats,
            max_stat_change_pct=req.max_stat_change_pct,
            max_iter=req.max_iterations,
        )

        st.logger.info(
            f"BALANCE_OPTIMIZE user={user['username']} "
            f"before={result.get('before_balance', 0):.1f} "
            f"after={result.get('after_balance', 0):.1f} "
            f"improvement={result.get('improvement', 0):.1f}"
        )

        return {
            "status": "SUCCESS",
            "data": result,
        }

    except ImportError as e:
        st.logger.error(f"밸런스 최적화 모듈 로드 실패: {e}")
        return {
            "status": "FAILED",
            "error": f"밸런스 최적화 모듈 로드 실패: {safe_str(e)}",
        }
    except Exception as e:
        st.logger.exception("밸런스 최적화 실패")
        return {
            "status": "FAILED",
            "error": f"밸런스 최적화 중 오류: {safe_str(e)}",
        }


@router.post("/balance/simulate")
def simulate_patch(
    stat_adjustments: List[dict],
    user: dict = Depends(verify_credentials)
):
    """
    패치 시뮬레이션 API

    스탯 조정값 적용 후 밸런스 변화를 예측합니다.

    Request Body:
        [
            {
                "cookie_id": "CK001",
                "stat_changes": {
                    "atk": {"change_pct": 10},
                    "cooldown": {"change": -1}
                }
            },
            ...
        ]

    Returns:
        - before: 패치 전 밸런스 분석
        - after: 패치 후 예상 밸런스 분석
        - balance_change: 밸런스 지수 변화
        - diversity_change: 다양성 지수 변화
    """
    try:
        from ml.balance_optimizer import BalanceOptimizer

        if st.COOKIE_STATS_DF is None or len(st.COOKIE_STATS_DF) == 0:
            return {
                "status": "FAILED",
                "error": "cookie_stats.csv 데이터가 없습니다.",
            }

        optimizer = BalanceOptimizer(st.COOKIE_STATS_DF)
        result = optimizer.simulate_patch(stat_adjustments)

        st.logger.info(
            f"BALANCE_SIMULATE user={user['username']} "
            f"balance_change={result.get('balance_change', 0):.1f}"
        )

        return {
            "status": "SUCCESS",
            "data": result,
        }

    except Exception as e:
        st.logger.exception("패치 시뮬레이션 실패")
        return {
            "status": "FAILED",
            "error": f"패치 시뮬레이션 중 오류: {safe_str(e)}",
        }


# ========================================
# 투자 최적화 API (ML + P-PSO)
# ========================================

@router.get("/investment/users")
async def get_game_users_list(user: dict = Depends(verify_credentials)):
    """
    게임 유저 목록 조회 (투자 최적화용)
    """
    try:
        if st.USERS_DF is None or st.USERS_DF.empty:
            return {"status": "ERROR", "message": "유저 데이터 없음"}

        # 유저 목록 반환 (최대 100명)
        users = []
        for _, row in st.USERS_DF.head(100).iterrows():
            users.append({
                "id": row.get("user_id", ""),
                "name": row.get("name", row.get("user_id", "")),
                "level": int(row.get("level", 0)),
                "segment": row.get("segment", "알 수 없음"),
            })

        return {"status": "SUCCESS", "users": users}
    except Exception as e:
        logger.error(f"유저 목록 조회 오류: {e}")
        return {"status": "ERROR", "message": str(e)}


@router.get("/investment/user/{user_id}")
async def get_user_investment_status(user_id: str, user: dict = Depends(verify_credentials)):
    """
    유저의 투자 현황 조회

    Returns:
        - cookies: 유저 보유 쿠키 목록
        - resources: 유저 보유 자원
    """
    try:
        # 유저 쿠키 데이터
        user_cookies_path = st.BACKEND_DIR / "user_cookies.csv"
        user_resources_path = st.BACKEND_DIR / "user_resources.csv"

        cookies_data = []
        resources_data = {}

        if user_cookies_path.exists():
            all_cookies = pd.read_csv(user_cookies_path)
            user_cookies = all_cookies[all_cookies['user_id'] == user_id]
            # 프론트엔드 기대 필드명으로 매핑
            for _, row in user_cookies.iterrows():
                cookies_data.append({
                    'cookie_id': row.get('cookie_id', ''),
                    'level': int(row.get('cookie_level', 0)),
                    'skill_level': int(row.get('skill_level', 0)),
                    'ascension': int(row.get('ascension', 0)),
                    'is_favorite': bool(row.get('is_favorite', 0)),
                })

        if user_resources_path.exists():
            all_resources = pd.read_csv(user_resources_path)
            user_row = all_resources[all_resources['user_id'] == user_id]
            if len(user_row) > 0:
                row = user_row.iloc[0].to_dict()
                # soul_stone 통합 (개별 등급 합산)
                total_soul_stone = sum([
                    int(row.get('soul_stone_common', 0)),
                    int(row.get('soul_stone_rare', 0)),
                    int(row.get('soul_stone_epic', 0)),
                    int(row.get('soul_stone_ancient', 0)),
                    int(row.get('soul_stone_legendary', 0)),
                ])
                resources_data = {
                    'exp_jelly': int(row.get('exp_jelly', 0)),
                    'coin': int(row.get('coin', 0)),
                    'skill_powder': int(row.get('skill_powder', 0)),
                    'soul_stone': total_soul_stone,
                }

        # 총 전투력 계산: (레벨 * 100) + (스킬레벨 * 50) + (승급 * 200)
        total_power = sum(
            (c['level'] * 100) + (c['skill_level'] * 50) + (c['ascension'] * 200)
            for c in cookies_data
        )

        return {
            "status": "SUCCESS",
            "data": {
                "user_id": user_id,
                "cookies": cookies_data,
                "cookies_count": len(cookies_data),
                "resources": resources_data,
                "total_power": total_power,
            }
        }

    except Exception as e:
        st.logger.exception(f"유저 투자 현황 조회 실패: {user_id}")
        return {
            "status": "FAILED",
            "error": f"조회 실패: {safe_str(e)}",
        }


@router.post("/investment/optimize")
async def optimize_investment(
    user_id: str = Body(..., description="유저 ID"),
    goal: str = Body("maximize_win_rate", description="목표: maximize_win_rate | maximize_efficiency | balanced"),
    top_n: int = Body(5, description="추천 개수"),
    resources: dict = Body(None, description="유저 보유 자원 (프론트엔드에서 전달)"),
    user: dict = Depends(verify_credentials)
):
    """
    자원 투자 최적화 (ML + P-PSO)

    유저의 보유 쿠키/자원을 분석하여 최적의 투자 전략을 제안합니다.
    각 유저마다 다른 보유 현황에 따라 개인화된 추천을 제공합니다.

    Args:
        user_id: 유저 ID (예: "U000001")
        goal: 최적화 목표
            - "maximize_win_rate": 승률 최대화 (권장)
            - "maximize_efficiency": 효율 최대화
            - "balanced": 균형 잡힌 추천
        top_n: 추천할 투자 옵션 수 (기본: 5)

    Returns:
        - recommendations: 추천 투자 목록
            - cookie_name: 쿠키 이름
            - upgrade_type: 업그레이드 종류 (cookie_level/skill_level/ascension)
            - win_rate_gain: 예상 승률 증가
            - efficiency: 투자 효율
            - cost: 필요 자원
        - total_cost: 총 필요 자원
        - total_win_rate_gain: 총 예상 승률 증가
        - resource_usage: 자원 사용률 (보유량 대비)
    """
    try:
        from ml.investment_optimizer import InvestmentOptimizer

        optimizer = InvestmentOptimizer(user_id, resources=resources)
        result = optimizer.optimize(goal=goal, top_n=top_n)

        if 'error' in result:
            return {
                "status": "FAILED",
                "error": result['error'],
            }

        st.logger.info(
            f"INVESTMENT_OPTIMIZE user={user_id} "
            f"goal={goal} recommendations={len(result.get('recommendations', []))} "
            f"win_rate_gain={result.get('total_win_rate_gain', 0):.2f}"
        )

        return {
            "status": "SUCCESS",
            "data": result,
        }

    except Exception as e:
        st.logger.exception(f"투자 최적화 실패: {user_id}")
        return {
            "status": "FAILED",
            "error": f"최적화 실패: {safe_str(e)}",
        }


@router.post("/investment/efficiency")
async def calculate_investment_efficiency(
    user_id: str = Body(..., description="유저 ID"),
    cookie_id: str = Body(..., description="쿠키 ID"),
    upgrade_type: str = Body(..., description="업그레이드 종류: cookie_level | skill_level | ascension"),
    levels: int = Body(10, description="업그레이드할 레벨 수"),
    user: dict = Depends(verify_credentials)
):
    """
    단일 투자의 효율 계산

    특정 쿠키에 특정 업그레이드를 했을 때의 효율을 계산합니다.

    Args:
        user_id: 유저 ID
        cookie_id: 쿠키 ID (예: "CK001")
        upgrade_type: 업그레이드 종류
            - "cookie_level": 쿠키 레벨업
            - "skill_level": 스킬 레벨업
            - "ascension": 각성
        levels: 업그레이드할 레벨 수 (기본: 10)

    Returns:
        - cost: 필요 자원
        - stat_gain: 예상 스탯 증가
        - win_rate_before/after: 업그레이드 전후 예상 승률
        - win_rate_gain: 승률 증가량
        - efficiency: 투자 효율 (승률 증가 / 비용)
    """
    try:
        from ml.investment_optimizer import InvestmentOptimizer

        optimizer = InvestmentOptimizer(user_id)
        result = optimizer.calculate_investment_efficiency(
            cookie_id=cookie_id,
            upgrade_type=upgrade_type,
            levels=levels
        )

        if 'error' in result:
            return {
                "status": "FAILED",
                "error": result['error'],
            }

        return {
            "status": "SUCCESS",
            "data": result,
        }

    except Exception as e:
        st.logger.exception(f"투자 효율 계산 실패: {user_id}/{cookie_id}")
        return {
            "status": "FAILED",
            "error": f"계산 실패: {safe_str(e)}",
        }


@router.get("/investment/compare")
async def compare_user_recommendations(
    user_ids: str = Query(..., description="유저 ID 목록 (콤마 구분)"),
    user: dict = Depends(verify_credentials)
):
    """
    여러 유저의 추천 결과 비교 (개인화 검증용)

    같은 최적화 로직이지만 유저마다 다른 결과가 나오는지 확인합니다.
    """
    try:
        from ml.investment_optimizer import compare_users

        user_id_list = [uid.strip() for uid in user_ids.split(',')]
        result = compare_users(user_id_list)

        # 간소화된 비교 결과
        comparison = {}
        for uid, res in result.items():
            if 'error' in res:
                comparison[uid] = {'error': res['error']}
            else:
                top_rec = res.get('recommendations', [{}])[0]
                comparison[uid] = {
                    'top_recommendation': {
                        'cookie_name': top_rec.get('cookie_name', 'N/A'),
                        'upgrade_type': top_rec.get('upgrade_type', 'N/A'),
                        'efficiency': top_rec.get('efficiency', 0),
                    },
                    'total_recommendations': len(res.get('recommendations', [])),
                    'total_win_rate_gain': res.get('total_win_rate_gain', 0),
                }

        return {
            "status": "SUCCESS",
            "data": {
                "user_count": len(user_id_list),
                "comparison": comparison,
                "note": "각 유저마다 다른 추천 결과가 나오면 개인화가 잘 작동하는 것입니다."
            }
        }

    except Exception as e:
        st.logger.exception("유저 비교 실패")
        return {
            "status": "FAILED",
            "error": f"비교 실패: {safe_str(e)}",
        }
