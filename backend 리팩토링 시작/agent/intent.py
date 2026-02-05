"""
agent/intent.py - 쿠키런 AI 플랫폼 인텐트 감지 및 도구 라우팅
============================================================
데브시스터즈 기술혁신 프로젝트

사용자 입력을 분석하여 적절한 도구를 실행합니다.
"""
import time
from typing import Optional, Dict, Any, Tuple

from core.constants import RAG_DOCUMENTS, SUMMARY_TRIGGERS
from core.utils import safe_str
from agent.tools import (
    tool_get_cookie_info,
    tool_list_cookies,
    tool_get_kingdom_info,
    tool_list_kingdoms,
    tool_translate_text,
    tool_get_worldview_terms,
    tool_analyze_user,
    tool_get_segment_statistics,
    tool_get_event_statistics,
    tool_classify_text,
    tool_get_dashboard_summary,
)
from rag.service import tool_rag_search
import state as st


# ============================================================
# 요약 트리거 / 컨텍스트 재활용
# ============================================================
def _has_summary_trigger(user_text: str) -> bool:
    t = (user_text or "").lower()
    return any(k.lower() in t for k in SUMMARY_TRIGGERS)


def set_last_context(username: str, context_id: Optional[str], results: Dict[str, Any], user_text: str, mode: str) -> None:
    if not username:
        return
    if not isinstance(results, dict) or len(results) == 0:
        return
    with st.LAST_CONTEXT_LOCK:
        st.LAST_CONTEXT_STORE[username] = {
            "context_id": safe_str(context_id).strip() if context_id else "",
            "results": results,
            "user_text": safe_str(user_text),
            "ts": time.time(),
            "mode": safe_str(mode),
        }


def get_last_context(username: str) -> Optional[Dict[str, Any]]:
    if not username:
        return None
    with st.LAST_CONTEXT_LOCK:
        return st.LAST_CONTEXT_STORE.get(username)


def can_reuse_last_context(username: str, context_id: Optional[str], user_text: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    if not _has_summary_trigger(user_text):
        return (False, None)

    ctx = get_last_context(username)
    if not ctx or not isinstance(ctx.get("results"), dict) or len(ctx["results"]) == 0:
        return (False, None)

    ts = float(ctx.get("ts") or 0.0)
    if ts > 0 and (time.time() - ts) > st.LAST_CONTEXT_TTL_SEC:
        return (False, None)

    # 컨텍스트 ID가 일치하면 재사용
    req_id = safe_str(context_id).strip() if context_id else ""
    last_id = safe_str(ctx.get("context_id")).strip()

    if req_id and req_id == last_id:
        return (True, ctx)

    # 이전 결과가 있으면 재사용
    r = ctx.get("results") or {}
    context_keys = ("get_cookie_info", "analyze_user", "translate", "rag_search", "dashboard")
    if any(isinstance(r.get(k), dict) for k in context_keys):
        return (True, ctx)

    return (False, None)


# ============================================================
# 인텐트 감지
# ============================================================
def detect_intent(user_text: str) -> Dict[str, bool]:
    t = (user_text or "").strip().lower()

    # RAG 트리거
    rag_triggers = ["뜻", "용어", "설명", "정의", "개념", "meaning", "definition", "세계관", "스토리"]

    # 문서 키워드 검색
    has_doc_keyword = False
    for _, doc in RAG_DOCUMENTS.items():
        for kw in doc.get("keywords", []):
            kw2 = (kw or "").strip().lower()
            if kw2 and (kw2 in t):
                has_doc_keyword = True
                break
        if has_doc_keyword:
            break

    # 대시보드/현황
    status_triggers = ["현황", "대시보드", "dashboard", "통계", "요약", "summary"]
    force_full_status = any(x in t for x in status_triggers)

    # 분석/예측 관련 키워드 (RAG 불필요)
    analytics_keywords = [
        "이탈", "churn", "코호트", "cohort", "리텐션", "retention", "잔존",
        "트렌드", "trend", "kpi", "dau", "mau", "wau",
        "매출", "revenue", "arpu", "arppu", "과금", "성장률",
        "승률", "pvp", "예측", "predict",
        "투자", "육성", "추천",
    ]
    want_analytics = any(kw in t for kw in analytics_keywords)

    return {
        # 쿠키 관련
        "want_cookie_info": ("쿠키" in t) and (("정보" in t) or ("누구" in t) or ("알려" in t)),
        "want_cookie_list": ("쿠키" in t) and (("목록" in t) or ("리스트" in t) or ("전체" in t)),

        # 왕국 관련
        "want_kingdom_info": ("왕국" in t) and (("정보" in t) or ("어디" in t) or ("알려" in t)),
        "want_kingdom_list": ("왕국" in t) and (("목록" in t) or ("리스트" in t) or ("전체" in t)),

        # 번역 관련
        "want_translate": ("번역" in t) or ("translate" in t) or ("영어로" in t) or ("일본어로" in t) or ("중국어로" in t),
        "want_terms": ("용어집" in t) or ("용어" in t and "번역" in t),

        # 유저 분석
        "want_user_analysis": ("유저" in t or "사용자" in t or "플레이어" in t) and ("분석" in t or "정보" in t),
        "want_segment": ("세그먼트" in t) or ("군집" in t) or ("분류" in t and "유저" in t),
        "want_anomaly": ("이상" in t) or ("어뷰징" in t) or ("봇" in t),

        # 게임 이벤트
        "want_event_stats": ("이벤트" in t) or ("로그" in t) or ("활동" in t),

        # 텍스트 분류
        "want_classify": ("분류" in t and "텍스트" in t) or ("카테고리" in t),

        # RAG 검색
        "want_rag": any(x in t for x in rag_triggers) or has_doc_keyword,

        # 대시보드
        "want_dashboard": force_full_status,

        # 분석/예측 (RAG 불필요)
        "want_analytics": want_analytics,
    }


# ============================================================
# 쿠키 ID/이름 추출
# ============================================================
def extract_cookie_id(user_text: str) -> Optional[str]:
    """텍스트에서 쿠키 ID 또는 이름을 추출합니다."""
    if not user_text:
        return None

    # CK로 시작하는 ID 패턴
    import re
    pattern = r'CK\d{3}'
    match = re.search(pattern, user_text.upper())
    if match:
        return match.group()

    # 알려진 쿠키 이름 검색
    cookie_names = [
        "용감한 쿠키", "딸기맛 쿠키", "마법사맛 쿠키", "닌자맛 쿠키", "기사맛 쿠키",
        "순수 바닐라 쿠키", "다크카카오 쿠키", "홀리베리 쿠키", "화이트릴리 쿠키", "골든치즈 쿠키",
        "다크엔챈트리스 쿠키", "크림유니콘 쿠키", "슬라임 쿠키", "허브맛 쿠키", "마들렌 쿠키",
    ]
    for name in cookie_names:
        if name in user_text:
            return name

    return None


# ============================================================
# 유저 ID 추출
# ============================================================
def extract_user_id(user_text: str) -> Optional[str]:
    """텍스트에서 유저 ID를 추출합니다."""
    if not user_text:
        return None

    import re
    pattern = r'U\d{4}'
    match = re.search(pattern, user_text.upper())
    if match:
        return match.group()

    return None


# ============================================================
# 대상 언어 추출
# ============================================================
def extract_target_lang(user_text: str) -> Optional[str]:
    """텍스트에서 번역 대상 언어를 추출합니다."""
    if not user_text:
        return None

    lang_map = {
        "영어": "en", "english": "en", "영문": "en",
        "일본어": "ja", "japanese": "ja", "일문": "ja",
        "중국어": "zh", "chinese": "zh", "중문": "zh",
        "번체": "zh-TW", "번체중국어": "zh-TW",
        "태국어": "th", "thai": "th",
        "인도네시아어": "id", "indonesian": "id",
        "독일어": "de", "german": "de",
        "프랑스어": "fr", "french": "fr",
        "스페인어": "es", "spanish": "es",
        "포르투갈어": "pt", "portuguese": "pt",
    }

    t = user_text.lower()
    for keyword, lang_code in lang_map.items():
        if keyword in t:
            return lang_code

    return "en"  # 기본값: 영어


# ============================================================
# 결정적 도구 실행 파이프라인
# ============================================================
def run_deterministic_tools(user_text: str, context_id: Optional[str] = None) -> Dict[str, Any]:
    """사용자 입력에 따라 적절한 도구를 실행합니다."""
    intents = detect_intent(user_text)
    results: Dict[str, Any] = {}

    # RAG 검색 (세계관 정보 질문)
    if intents.get("want_rag"):
        results["rag_search"] = tool_rag_search(
            user_text,
            top_k=min(5, st.RAG_MAX_TOPK),
            api_key=""
        )

    # 대시보드 요약
    if intents.get("want_dashboard"):
        results["dashboard"] = tool_get_dashboard_summary()
        return results

    # 쿠키 정보
    if intents.get("want_cookie_info"):
        cookie_id = extract_cookie_id(user_text)
        if cookie_id:
            results["get_cookie_info"] = tool_get_cookie_info(cookie_id)
        else:
            # ID를 못 찾으면 목록 반환
            results["list_cookies"] = tool_list_cookies()
        return results

    if intents.get("want_cookie_list"):
        # 등급/타입 필터 추출
        grade = None
        cookie_type = None

        grade_keywords = ["커먼", "레어", "에픽", "레전더리", "에인션트"]
        for g in grade_keywords:
            if g in user_text:
                grade = g
                break

        type_keywords = ["돌격", "마법", "사격", "방어", "지원", "치유"]
        for t in type_keywords:
            if t in user_text:
                cookie_type = t
                break

        results["list_cookies"] = tool_list_cookies(grade=grade, cookie_type=cookie_type)
        return results

    # 왕국 정보
    if intents.get("want_kingdom_info") or intents.get("want_kingdom_list"):
        results["list_kingdoms"] = tool_list_kingdoms()
        return results

    # 번역 요청
    if intents.get("want_translate"):
        target_lang = extract_target_lang(user_text)
        # 번역할 텍스트는 LLM에서 처리 (여기서는 컨텍스트만 설정)
        results["translate_context"] = {
            "status": "SUCCESS",
            "action": "TRANSLATE",
            "target_lang": target_lang,
            "message": f"'{target_lang}' 언어로 번역을 준비합니다.",
        }
        return results

    if intents.get("want_terms"):
        target_lang = extract_target_lang(user_text)
        results["worldview_terms"] = tool_get_worldview_terms(target_lang=target_lang)
        return results

    # 유저 분석
    if intents.get("want_user_analysis"):
        user_id = extract_user_id(user_text)
        if user_id:
            results["analyze_user"] = tool_analyze_user(user_id)
        else:
            results["segment_statistics"] = tool_get_segment_statistics()
        return results

    if intents.get("want_segment"):
        results["segment_statistics"] = tool_get_segment_statistics()
        return results

    # 게임 이벤트 통계
    if intents.get("want_event_stats"):
        results["event_statistics"] = tool_get_event_statistics()
        return results

    # 텍스트 분류
    if intents.get("want_classify"):
        results["classify_context"] = {
            "status": "SUCCESS",
            "action": "CLASSIFY",
            "message": "텍스트 분류를 위해 분류할 텍스트를 입력해주세요.",
        }
        return results

    return results
