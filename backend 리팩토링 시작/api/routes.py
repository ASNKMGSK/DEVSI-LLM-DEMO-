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

from fastapi import APIRouter, HTTPException, Depends, status, Request, UploadFile, File, BackgroundTasks
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
from rag.graph_rag import (
    build_graph_from_chunks, search_graph_rag, get_graph_rag_status,
    clear_graph_rag, NETWORKX_AVAILABLE, GRAPH_RAG_STORE
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
    model: str = Field("gpt-4o", alias="model")
    max_tokens: int = Field(5000, alias="maxTokens")
    system_prompt: str = Field(DEFAULT_SYSTEM_PROMPT, alias="systemPrompt")
    temperature: Optional[float] = Field(None, alias="temperature")
    top_p: Optional[float] = Field(None, alias="topP")
    presence_penalty: Optional[float] = Field(None, alias="presencePenalty")
    frequency_penalty: Optional[float] = Field(None, alias="frequencyPenalty")
    seed: Optional[int] = Field(None, alias="seed")
    timeout_ms: Optional[int] = Field(None, alias="timeoutMs")
    retries: Optional[int] = Field(None, alias="retries")
    stream: Optional[bool] = Field(None, alias="stream")
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


class GraphRagBuildRequest(BaseModel):
    """GraphRAG 빌드 요청 모델"""
    api_key: str = Field("", alias="apiKey")
    max_chunks: int = Field(20, alias="maxChunks")
    class Config:
        populate_by_name = True
        allow_population_by_field_name = True
        allow_population_by_alias = True


class GraphRagSearchRequest(BaseModel):
    """GraphRAG 검색 요청 모델"""
    query: str
    api_key: str = Field("", alias="apiKey")
    top_k: int = Field(5, alias="topK")
    include_neighbors: bool = Field(True, alias="includeNeighbors")
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
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


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
def search_user(q: str, user: dict = Depends(verify_credentials)):
    """유저 검색"""
    if st.USERS_DF is None or st.USER_ANALYTICS_DF is None:
        return {"status": "FAILED", "error": "유저 데이터가 로드되지 않았습니다."}

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

    # 유저 스탯 계산 (레이더 차트용)
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

    # 최근 활동 (USER_ACTIVITY_DF에서 가져오기)
    activity = []
    if st.USER_ACTIVITY_DF is not None:
        user_activity = st.USER_ACTIVITY_DF[st.USER_ACTIVITY_DF["user_id"] == user_id]
        if not user_activity.empty:
            for _, row in user_activity.iterrows():
                activity.append({
                    "date": row.get("date", ""),
                    "playtime": int(row.get("playtime", 0)),
                    "stages": int(row.get("stages_cleared", 0)),
                })

    # 데이터가 없으면 기본값
    if not activity:
        activity = [
            {"date": "01/25", "playtime": 120, "stages": 15},
            {"date": "01/26", "playtime": 90, "stages": 12},
            {"date": "01/27", "playtime": 150, "stages": 18},
            {"date": "01/28", "playtime": 80, "stages": 10},
            {"date": "01/29", "playtime": 110, "stages": 14},
            {"date": "01/30", "playtime": 95, "stages": 11},
            {"date": "01/31", "playtime": 130, "stages": 16},
        ]

    # 프론트엔드 형식에 맞게 변환
    result = {
        "status": "SUCCESS",
        "user": {
            "id": user_data["user_id"],
            "name": f"플레이어_{user_data['user_id'][-4:]}",
            "segment": user_data["segment"],
            "level": 10 + total_events // 10,
            "playtime": total_events * 2,
            "cookies_owned": 5 + gacha_pulls // 2,
            "country": user_data.get("country", "KR"),
            "vip_level": int(user_data.get("vip_level", 0)),
            "is_anomaly": user_data.get("is_anomaly", False),
            "top_cookies": ["용감한 쿠키", "딸기맛 쿠키", "마법사맛 쿠키"],
            "stats": stats,
            "activity": activity,
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


@router.get("/analysis/anomaly")
def get_anomaly_analysis(days: int = 7, user: dict = Depends(verify_credentials)):
    """이상탐지 분석 데이터"""
    if st.USER_ANALYTICS_DF is None:
        return {"status": "FAILED", "error": "유저 분석 데이터가 없습니다."}

    # days 파라미터 유효성 검사
    if days not in [7, 30, 90]:
        days = 7

    try:
        df = st.USER_ANALYTICS_DF
        total_users = len(df)
        anomaly_users = df[df["is_anomaly"] == True] if "is_anomaly" in df.columns else df.iloc[:0]
        anomaly_count = len(anomaly_users)
        anomaly_rate = round(anomaly_count / total_users * 100, 2) if total_users > 0 else 0

        # 위험도 분류 (임의 로직)
        high_risk = int(anomaly_count * 0.22)
        medium_risk = int(anomaly_count * 0.52)
        low_risk = anomaly_count - high_risk - medium_risk

        # 일별 이상 탐지 트렌드 (선택 기간 기반)
        import random
        from datetime import datetime, timedelta
        today = datetime.now()
        trend = []
        for i in range(days):
            d = today - timedelta(days=days - 1 - i)
            trend.append({
                "date": d.strftime("%m/%d"),
                "count": max(1, int(anomaly_count * 0.12 * (0.8 + random.random() * 0.4)))
            })

        # 최근 알림
        recent_alerts = [
            {"id": "U000123", "type": "비정상 결제", "severity": "high", "detail": "24시간 내 50회 결제 시도", "time": "10분 전"},
            {"id": "U000456", "type": "봇 의심", "severity": "high", "detail": "반복적인 패턴 감지", "time": "25분 전"},
            {"id": "U000789", "type": "계정 공유", "severity": "medium", "detail": "다중 IP 접속", "time": "1시간 전"},
            {"id": "U000321", "type": "비정상 플레이", "severity": "low", "detail": "48시간 연속 접속", "time": "2시간 전"},
        ]

        return json_sanitize({
            "status": "SUCCESS",
            "summary": {
                "total_users": total_users,
                "anomaly_count": anomaly_count,
                "anomaly_rate": anomaly_rate,
                "high_risk": high_risk,
                "medium_risk": medium_risk,
                "low_risk": low_risk,
            },
            "by_type": [
                {"type": "비정상 결제 패턴", "count": max(1, int(anomaly_count * 0.35)), "severity": "high"},
                {"type": "봇 의심 행동", "count": max(1, int(anomaly_count * 0.26)), "severity": "high"},
                {"type": "계정 공유 의심", "count": max(1, int(anomaly_count * 0.22)), "severity": "medium"},
                {"type": "비정상 플레이 시간", "count": max(1, int(anomaly_count * 0.17)), "severity": "low"},
            ],
            "trend": trend,
            "recent_alerts": recent_alerts,
        })
    except Exception as e:
        return {"status": "FAILED", "error": safe_str(e)}


@router.get("/analysis/prediction/churn")
def get_churn_prediction(days: int = 7, user: dict = Depends(verify_credentials)):
    """이탈 예측 분석 (이탈/매출/참여도 통합)"""
    # days 파라미터는 예측 기간 조절용
    if st.USER_ANALYTICS_DF is None:
        return {"status": "FAILED", "error": "유저 분석 데이터가 없습니다."}

    try:
        df = st.USER_ANALYTICS_DF
        total = len(df)

        # 이탈 위험 분류 (가상 로직 - 실제는 ML 모델 사용)
        high_risk = int(total * 0.085)
        medium_risk = int(total * 0.142)
        low_risk = total - high_risk - medium_risk

        # 고위험 유저 목록 생성 (샘플)
        high_risk_users = []
        if st.USERS_DF is not None and len(st.USERS_DF) > 0:
            sample_users = st.USERS_DF.head(min(5, len(st.USERS_DF)))
            for idx, row in sample_users.iterrows():
                user_id = row.get("user_id", f"U{idx:06d}")
                high_risk_users.append({
                    "id": user_id,
                    "name": f"플레이어_{user_id[-4:]}",
                    "segment": ["캐주얼 유저", "하드코어 게이머", "신규 유저"][idx % 3],
                    "probability": 85 - idx * 5,
                    "last_active": f"{3 + idx}일 전",
                })

        # 매출 예측 데이터
        revenue_data = {
            "predicted_monthly": 125000000,  # 1.25억
            "growth_rate": 12.5,
            "predicted_arpu": 15420,
            "predicted_arppu": 48500,
            "confidence": 82,
            "whale_count": int(total * 0.02),
            "dolphin_count": int(total * 0.08),
            "minnow_count": int(total * 0.15),
        }

        # 참여도 예측 데이터
        engagement_data = {
            "predicted_dau": int(total * 0.65),
            "predicted_mau": int(total * 0.85),
            "stickiness": 76,  # DAU/MAU %
            "avg_session": 28,  # 분
            "retention_d1": 68,
            "retention_d7": 42,
            "retention_d30": 25,
            "sessions_per_day": 3.2,
        }

        return json_sanitize({
            "status": "SUCCESS",
            "churn": {
                "high_risk_count": high_risk,
                "medium_risk_count": medium_risk,
                "low_risk_count": low_risk,
                "predicted_churn_rate": round(high_risk / total * 100, 1) if total > 0 else 0,
                "model_accuracy": 87.3,
                "top_factors": [
                    {"factor": "7일간 미접속", "importance": 0.35},
                    {"factor": "플레이타임 급감", "importance": 0.25},
                    {"factor": "최근 과금 없음", "importance": 0.20},
                    {"factor": "길드 활동 감소", "importance": 0.12},
                    {"factor": "스테이지 진행 정체", "importance": 0.08},
                ],
                "high_risk_users": high_risk_users,
            },
            "revenue": revenue_data,
            "engagement": engagement_data,
        })
    except Exception as e:
        return {"status": "FAILED", "error": safe_str(e)}


@router.get("/analysis/cohort/retention")
def get_cohort_retention(days: int = 7, user: dict = Depends(verify_credentials)):
    """코호트 리텐션 분석 - 실제 데이터 기반"""
    # days 파라미터는 데이터 범위 조절용 (코호트는 주간 기반이므로 주 수로 변환)
    weeks = max(1, days // 7)
    try:
        # COHORT_RETENTION_DF가 있으면 실제 데이터 사용
        if st.COHORT_RETENTION_DF is not None and len(st.COHORT_RETENTION_DF) > 0:
            cohort_data = st.COHORT_RETENTION_DF.to_dict("records")
        else:
            # 폴백: 가상 데이터
            cohort_data = [
                {"cohort": "2025-01 W1", "week0": 100, "week1": 72, "week2": 58, "week3": 48, "week4": 42},
                {"cohort": "2025-01 W2", "week0": 100, "week1": 75, "week2": 62, "week3": 51, "week4": 45},
                {"cohort": "2025-01 W3", "week0": 100, "week1": 68, "week2": 55, "week3": 46, "week4": None},
                {"cohort": "2025-01 W4", "week0": 100, "week1": 70, "week2": 56, "week3": None, "week4": None},
            ]

        # LTV 코호트 데이터 (DAILY_METRICS_DF 기반 계산)
        if st.DAILY_METRICS_DF is not None and len(st.DAILY_METRICS_DF) > 0:
            avg_arpu = st.DAILY_METRICS_DF["arpu"].mean()
            user_count = len(st.USERS_DF) if st.USERS_DF is not None else 1000
            ltv_by_cohort = [
                {"cohort": "2024-10", "ltv": int(avg_arpu * 3.2), "users": int(user_count * 0.12)},
                {"cohort": "2024-11", "ltv": int(avg_arpu * 3.5), "users": int(user_count * 0.15)},
                {"cohort": "2024-12", "ltv": int(avg_arpu * 3.0), "users": int(user_count * 0.13)},
                {"cohort": "2025-01", "ltv": int(avg_arpu * 2.5), "users": int(user_count * 0.10)},
            ]
        else:
            ltv_by_cohort = [
                {"cohort": "2024-10", "ltv": 45000, "users": 1250},
                {"cohort": "2024-11", "ltv": 52000, "users": 1480},
                {"cohort": "2024-12", "ltv": 48000, "users": 1320},
                {"cohort": "2025-01", "ltv": 38000, "users": 980},
            ]

        # 전환 퍼널 데이터
        conversion = [
            {"cohort": "2025-01 W1", "registered": 1000, "activated": 720, "engaged": 520, "converted": 180, "retained": 95},
            {"cohort": "2025-01 W2", "registered": 1100, "activated": 780, "engaged": 560, "converted": 195, "retained": 102},
            {"cohort": "2025-01 W3", "registered": 950, "activated": 680, "engaged": 480, "converted": 165, "retained": 88},
            {"cohort": "2025-01 W4", "registered": 1050, "activated": 750, "engaged": 530, "converted": 185, "retained": 98},
        ]

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
    # days 파라미터는 일부 통계에 사용 가능 (예: 최근 N일 활동 유저 수 등)
    summary = {
        "status": "SUCCESS",
        "cookies_count": len(st.COOKIES_DF) if st.COOKIES_DF is not None else 0,
        "kingdoms_count": len(st.KINGDOMS_DF) if st.KINGDOMS_DF is not None else 0,
        "users_count": len(st.USERS_DF) if st.USERS_DF is not None else 0,
        "translations_count": len(st.TRANSLATIONS_DF) if st.TRANSLATIONS_DF is not None else 0,
        "game_logs_count": len(st.GAME_LOGS_DF) if st.GAME_LOGS_DF is not None else 0,
    }

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
    graph_status = get_graph_rag_status()
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
            "reranker_available": RERANKER_AVAILABLE,
            "kg_ready": bool(st.RAG_STORE.get("kg_ready")),
            "kg_entities_count": len(KNOWLEDGE_GRAPH.get("entities", {})) if KNOWLEDGE_GRAPH else 0,
            "kg_relations_count": len(KNOWLEDGE_GRAPH.get("relations", [])) if KNOWLEDGE_GRAPH else 0,
            # GraphRAG (LLM 기반)
            "graphrag_available": NETWORKX_AVAILABLE,
            "graphrag_ready": graph_status.get("ready", False),
            "graphrag_entities": graph_status.get("entity_count", 0),
            "graphrag_relations": graph_status.get("relation_count", 0),
            "graphrag_communities": graph_status.get("community_count", 0),
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
    background_tasks: BackgroundTasks = None,
    user: dict = Depends(verify_credentials),
):
    try:
        filename = file.filename or "unknown"
        ext = os.path.splitext(filename)[1].lower()

        if ext not in st.RAG_ALLOWED_EXTS:
            return {"status": "FAILED", "error": f"지원하지 않는 파일 형식입니다. 허용된 형식: {', '.join(st.RAG_ALLOWED_EXTS)}"}

        MAX_FILE_SIZE = 10 * 1024 * 1024
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            return {"status": "FAILED", "error": "파일 크기는 10MB를 초과할 수 없습니다."}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(st.RAG_DOCS_DIR, safe_filename)

        os.makedirs(st.RAG_DOCS_DIR, exist_ok=True)

        with open(file_path, "wb") as f:
            f.write(contents)

        # 백그라운드에서 인덱스 재빌드 (즉시 응답 반환)
        k = (api_key or "").strip() or st.OPENAI_API_KEY
        if k and background_tasks:
            background_tasks.add_task(rag_build_or_load_index, api_key=k, force_rebuild=True)

        return {
            "status": "SUCCESS",
            "message": "파일이 업로드되었습니다. 인덱스 재빌드 중...",
            "filename": safe_filename,
            "original_filename": filename,
            "size": len(contents),
            "path": os.path.relpath(file_path, st.BASE_DIR),
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

        # 백그라운드에서 인덱스 재빌드 (즉시 응답 반환)
        k = safe_str(req.api_key).strip() or st.OPENAI_API_KEY
        if k:
            background_tasks.add_task(rag_build_or_load_index, api_key=k, force_rebuild=True)

        return {"status": "SUCCESS", "message": "파일이 삭제되었습니다. 인덱스 재빌드 중...", "filename": filename}
    except Exception as e:
        st.logger.exception("파일 삭제 실패")
        return {"status": "FAILED", "error": f"파일 삭제 실패: {safe_str(e)}"}


# ============================================================
# GraphRAG (LLM 기반 지식 그래프)
# ============================================================
@router.get("/graphrag/status")
def graphrag_status(user: dict = Depends(verify_credentials)):
    """GraphRAG 상태 조회"""
    status = get_graph_rag_status()
    return {"status": "SUCCESS", **status}


@router.post("/graphrag/build")
def graphrag_build(
    req: GraphRagBuildRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(verify_credentials)
):
    """GraphRAG 지식 그래프 빌드 (LLM 기반 엔티티/관계 추출)"""
    if user.get("role") != "관리자":
        raise HTTPException(status_code=403, detail="권한 없음")

    try:
        k = safe_str(req.api_key).strip() or st.OPENAI_API_KEY
        if not k:
            return {"status": "FAILED", "error": "OpenAI API Key가 필요합니다."}

        if not NETWORKX_AVAILABLE:
            return {"status": "FAILED", "error": "NetworkX가 설치되지 않았습니다. pip install networkx"}

        # RAG 청크 가져오기
        with st.RAG_LOCK:
            idx = st.RAG_STORE.get("index")

        if idx is None:
            return {"status": "FAILED", "error": "RAG 인덱스가 없습니다. 먼저 문서를 업로드하세요."}

        # 청크 추출
        try:
            docstore = idx.docstore
            chunks = list(docstore._dict.values())
        except Exception:
            return {"status": "FAILED", "error": "RAG 인덱스에서 청크를 가져올 수 없습니다."}

        if not chunks:
            return {"status": "FAILED", "error": "RAG에 문서가 없습니다."}

        # 백그라운드에서 GraphRAG 빌드
        background_tasks.add_task(build_graph_from_chunks, chunks, k, req.max_chunks)

        return {
            "status": "SUCCESS",
            "message": f"GraphRAG 빌드 시작 (최대 {req.max_chunks}개 청크 처리)",
            "chunks_available": len(chunks),
        }

    except Exception as e:
        st.logger.exception("GraphRAG 빌드 실패")
        return {"status": "FAILED", "error": f"GraphRAG 빌드 실패: {safe_str(e)}"}


@router.post("/graphrag/search")
def graphrag_search(req: GraphRagSearchRequest, user: dict = Depends(verify_credentials)):
    """GraphRAG 검색 - 지식 그래프 기반 검색"""
    try:
        k = safe_str(req.api_key).strip() or st.OPENAI_API_KEY
        result = search_graph_rag(
            query=req.query,
            api_key=k,
            top_k=req.top_k,
            include_neighbors=req.include_neighbors
        )
        return result
    except Exception as e:
        st.logger.exception("GraphRAG 검색 실패")
        return {"status": "FAILED", "error": f"GraphRAG 검색 실패: {safe_str(e)}"}


@router.post("/graphrag/clear")
def graphrag_clear(user: dict = Depends(verify_credentials)):
    """GraphRAG 초기화"""
    if user.get("role") != "관리자":
        raise HTTPException(status_code=403, detail="권한 없음")

    clear_graph_rag()
    return {"status": "SUCCESS", "message": "GraphRAG가 초기화되었습니다."}


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
    st.logger.info(
        "STREAM_REQ headers_auth=%s origin=%s ua=%s",
        request.headers.get("authorization"),
        request.headers.get("origin"),
        request.headers.get("user-agent"),
    )
    username = user["username"]

    async def gen():
        tool_results = {}
        tool_calls = []
        final_buf = []

        try:
            user_text = safe_str(req.user_input)
            user_text_lower = user_text.lower()

            api_key = pick_api_key(req.api_key)

            # 질문 의도에 따라 적절한 도구 선택 및 호출
            from agent.tools import AVAILABLE_TOOLS

            # 도구 키워드 매핑 (쿠키런 관련 키워드 확장)
            tool_keywords = {
                "get_translation_statistics": ["번역 품질", "번역 현황", "번역 통계", "translation quality"],
                "get_anomaly_statistics": ["이상 행동", "이상 유저", "이상 탐지", "anomaly", "비정상"],
                "get_segment_statistics": ["세그먼트 통계", "유저 세그먼트", "유저 통계", "segment"],
                "analyze_user": ["유저 분석", "사용자 분석", "user analysis"],
                "get_event_statistics": ["이벤트 통계", "게임 통계", "event statistics"],
                "get_dashboard_summary": ["대시보드", "요약", "전체 현황", "dashboard"],
                # 예측 분석 - 새로 추가
                "get_churn_prediction": [
                    "이탈 예측", "이탈 분석", "churn", "이탈률", "이탈 위험",
                    "고위험 유저", "이탈 요인", "이탈 고위험",
                ],
                "get_revenue_prediction": [
                    "매출 예측", "수익 예측", "revenue", "ARPU", "ARPPU",
                    "과금 유저", "whale", "고래", "매출 분석",
                ],
                "get_cohort_analysis": [
                    "코호트", "cohort", "리텐션", "retention", "LTV",
                    "주간 리텐션", "유저 잔존", "잔존율",
                ],
                "get_trend_analysis": [
                    "트렌드", "trend", "KPI", "지표 분석", "추세",
                    "DAU", "MAU", "상관관계", "성장률",
                ],
                # 쿠키 관련 - 확장된 키워드
                "list_cookies": [
                    "쿠키 목록", "쿠키 리스트", "전체 쿠키", "쿠키들",
                    "에인션트", "레전더리", "에픽", "레어", "커먼",  # 등급
                    "마법 타입", "돌격 타입", "방어 타입", "치유 타입", "사격 타입",  # 타입
                    "타입 쿠키", "등급 쿠키", "쿠키 보여", "쿠키 알려",
                ],
                "get_cookie_info": ["쿠키 정보", "캐릭터 정보", "쿠키 알려줘"],
                "get_cookie_skill": ["스킬 설명", "스킬 정보", "능력 설명", "기술 설명"],
                "list_kingdoms": ["왕국 목록", "왕국 리스트", "전체 왕국"],
                "get_kingdom_info": [
                    "왕국 정보", "왕국 알려", "왕국은", "어떤 곳",
                    "다크카카오 왕국", "홀리베리 왕국", "바닐라 왕국",
                ],
                "get_worldview_terms": ["용어집", "번역 용어", "glossary", "세계관 용어"],
                "check_translation_quality": ["품질 검토", "품질 평가", "quality check"],
            }

            # 쿠키 이름으로 ID 찾기 헬퍼
            def find_cookie_id_by_name(name_query: str) -> str:
                if st.COOKIES_DF is None:
                    return None
                for _, row in st.COOKIES_DF.iterrows():
                    if row["name"] in name_query or name_query in row["name"]:
                        return row["id"]
                return None

            # 1. RAG 검색 먼저 시도
            rag_has_relevant = False
            if api_key:
                try:
                    rag_out = tool_rag_search(user_text, top_k=st.RAG_DEFAULT_TOPK, api_key=api_key)
                    if isinstance(rag_out, dict) and rag_out.get("status") == "SUCCESS":
                        results = rag_out.get("results") or []
                        if results:
                            # RAG 결과가 쿠키런 관련인지 확인
                            cookierun_keywords = ["쿠키", "cookie", "왕국", "kingdom", "소울잼", "세계관"]
                            for r in results:
                                content = (r.get("content") or "").lower()
                                title = (r.get("title") or "").lower()
                                if any(kw in content or kw in title for kw in cookierun_keywords):
                                    rag_has_relevant = True
                                    break
                            tool_results["rag"] = rag_out
                            st.logger.info("RAG_IN_AGENT ok=1 results=%d relevant=%s", len(results), rag_has_relevant)
                except Exception as _e:
                    st.logger.warning("RAG_IN_AGENT_FAIL err=%s", safe_str(_e))

            # 2. RAG에 관련 결과 없으면 도구 fallback
            if not rag_has_relevant:
                for tool_name, keywords in tool_keywords.items():
                    if any(kw in user_text_lower for kw in keywords):
                        if tool_name in AVAILABLE_TOOLS:
                            try:
                                tool_func = AVAILABLE_TOOLS[tool_name]

                                # 특정 쿠키 정보/스킬 요청 처리
                                if tool_name in ["get_cookie_info", "get_cookie_skill"]:
                                    cookie_id = find_cookie_id_by_name(user_text)
                                    if cookie_id:
                                        tool_out = tool_func(cookie_id)
                                        if isinstance(tool_out, dict):
                                            tool_results[tool_name] = tool_out
                                            st.logger.info("TOOL_FALLBACK tool=%s cookie_id=%s ok=1", tool_name, cookie_id)
                                    else:
                                        tool_out = AVAILABLE_TOOLS["list_cookies"]()
                                        if isinstance(tool_out, dict):
                                            tool_results["list_cookies"] = tool_out
                                    continue

                                # 특정 왕국 정보 요청 처리
                                if tool_name == "get_kingdom_info":
                                    # 왕국 이름으로 ID 찾기
                                    kingdom_id = None
                                    if st.KINGDOMS_DF is not None:
                                        for _, row in st.KINGDOMS_DF.iterrows():
                                            if row["name"] in user_text:
                                                kingdom_id = row["id"]
                                                break
                                    if kingdom_id:
                                        tool_out = tool_func(kingdom_id)
                                        if isinstance(tool_out, dict):
                                            tool_results[tool_name] = tool_out
                                    else:
                                        tool_out = AVAILABLE_TOOLS["list_kingdoms"]()
                                        if isinstance(tool_out, dict):
                                            tool_results["list_kingdoms"] = tool_out
                                    continue

                                # 유저 분석은 ID 필요 - 스킵
                                if tool_name == "analyze_user":
                                    continue

                                tool_out = tool_func()
                                if isinstance(tool_out, dict):
                                    tool_results[tool_name] = tool_out
                                    st.logger.info("TOOL_FALLBACK tool=%s ok=1", tool_name)
                            except Exception as _e:
                                st.logger.warning("TOOL_FALLBACK_FAIL tool=%s err=%s", tool_name, safe_str(_e))

                # 도구 결과가 있으면 RAG 결과 제거 (관련 없으므로)
                if len(tool_results) > 1 and "rag" in tool_results:
                    del tool_results["rag"]

            tool_calls = [{"tool": k, "result": v} for k, v in tool_results.items()]

            if not api_key:
                msg = "처리 오류: OpenAI API Key가 없습니다. 환경변수 OPENAI_API_KEY 또는 요청의 api_key를 설정하세요."
                yield sse_pack("done", {"ok": False, "final": msg, "tool_calls": tool_calls})
                return

            messages = build_langchain_messages(req.system_prompt, username, user_text, tool_results)
            llm = get_llm(
                req.model, api_key, req.max_tokens or 1500, streaming=True,
                temperature=req.temperature, top_p=req.top_p,
                presence_penalty=req.presence_penalty, frequency_penalty=req.frequency_penalty,
                seed=req.seed, timeout_ms=req.timeout_ms, max_retries=req.retries,
            )

            for chunk_obj in llm.stream(messages):
                if await request.is_disconnected():
                    return
                delta = chunk_text(chunk_obj)
                if delta:
                    final_buf.append(delta)
                    yield sse_pack("delta", {"delta": delta})

            final_text = "".join(final_buf).strip()
            if not final_text:
                final_text = "죄송합니다. 요청을 처리할 수 없습니다."

            append_memory(username, user_text, final_text)

            yield sse_pack("done", {"ok": True, "final": final_text, "tool_calls": tool_calls})
            return

        except Exception as e:
            msg = safe_str(e) or "스트리밍 오류"
            try:
                yield sse_pack("error", {"message": msg})
            except Exception:
                pass
            yield sse_pack("done", {"ok": False, "final": msg, "tool_calls": tool_calls})
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

        # 노트북에서 생성한 mlruns 폴더 경로 (project 루트)
        project_mlruns = os.path.abspath(os.path.join(st.BASE_DIR, "..", "mlruns"))
        backend_mlruns = os.path.join(st.BASE_DIR, "mlruns")

        # 두 경로 중 존재하는 것 사용
        if os.path.exists(project_mlruns):
            tracking_uri = f"file:{project_mlruns}"
        elif os.path.exists(backend_mlruns):
            tracking_uri = f"file:{backend_mlruns}"
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

        project_mlruns = os.path.abspath(os.path.join(st.BASE_DIR, "..", "mlruns"))
        backend_mlruns = os.path.join(st.BASE_DIR, "mlruns")

        if os.path.exists(project_mlruns):
            tracking_uri = f"file:{project_mlruns}"
        elif os.path.exists(backend_mlruns):
            tracking_uri = f"file:{backend_mlruns}"
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


@router.post("/mlflow/models/select")
def select_mlflow_model(req: ModelSelectRequest, user: dict = Depends(verify_credentials)):
    """MLflow 모델 선택/로드"""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        project_mlruns = os.path.abspath(os.path.join(st.BASE_DIR, "..", "mlruns"))
        backend_mlruns = os.path.join(st.BASE_DIR, "mlruns")

        if os.path.exists(project_mlruns):
            tracking_uri = f"file:{project_mlruns}"
        elif os.path.exists(backend_mlruns):
            tracking_uri = f"file:{backend_mlruns}"
        else:
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()

        # 모델 버전 정보 조회
        try:
            model_version = client.get_model_version(req.model_name, req.version)

            # 모델 로드 시도 (선택적)
            model_uri = f"models:/{req.model_name}/{req.version}"
            st.logger.info(f"모델 선택: {model_uri}")

            return {
                "status": "SUCCESS",
                "message": f"{req.model_name} v{req.version} 모델이 선택되었습니다",
                "data": {
                    "model_name": req.model_name,
                    "version": req.version,
                    "stage": model_version.current_stage,
                    "run_id": model_version.run_id,
                }
            }
        except Exception as e:
            # 모델이 레지스트리에 없어도 데모 모드로 성공 반환
            st.logger.warning(f"모델 조회 실패 (데모 모드): {e}")
            return {
                "status": "SUCCESS",
                "message": f"{req.model_name} v{req.version} 모델이 선택되었습니다 (데모)",
                "data": {
                    "model_name": req.model_name,
                    "version": req.version,
                    "stage": "Demo",
                }
            }

    except ImportError:
        # MLflow 없어도 데모 모드로 성공
        return {
            "status": "SUCCESS",
            "message": f"{req.model_name} v{req.version} 모델이 선택되었습니다 (데모)",
            "data": {
                "model_name": req.model_name,
                "version": req.version,
            }
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
            "selectedModel": "gpt-4o",
            "maxTokens": 4000,
            "temperature": 0.7,
            "topP": 1.0,
            "presencePenalty": 0.0,
            "frequencyPenalty": 0.0,
            "seed": "",
            "timeoutMs": 30000,
            "retries": 2,
            "stream": True,
            "systemPrompt": DEFAULT_SYSTEM_PROMPT,
        },
    }


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
