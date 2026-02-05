"""
쿠키런 AI 플랫폼 - 분석 도구 함수들
=====================================
데브시스터즈 기술혁신 프로젝트

주요 기능:
1. 쿠키런 세계관 맞춤형 번역 지원
2. 임베딩 기반 지식 검색
3. 데이터 분석 및 의사결정 지원
"""

from typing import Optional, List, Dict, Any
import json
import numpy as np
import pandas as pd

from core.constants import (
    FEATURE_COLS_TRANSLATION,
    FEATURE_COLS_USER_SEGMENT,
    FEATURE_LABELS,
    WORLDVIEW_TERMS,
    SUPPORTED_LANGUAGES,
    USER_SEGMENT_NAMES,
    TRANSLATION_QUALITY_GRADES,
    TRANSLATION_CATEGORIES,
    DEFAULT_TOPN,
)
import state as st


# ============================================================
# 유틸리티 함수
# ============================================================
def safe_str(val, default: str = "") -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return str(val)


def safe_int(val, default: int = 0) -> int:
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return int(float(val))
    except (ValueError, TypeError):
        return default


def safe_float(val, default: float = 0.0) -> float:
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


# ============================================================
# 1. 쿠키 캐릭터 정보 조회
# ============================================================
def tool_get_cookie_info(cookie_id: str) -> dict:
    """쿠키 캐릭터 정보를 조회합니다."""
    if st.COOKIES_DF is None:
        return {"status": "FAILED", "error": "쿠키 데이터가 로드되지 않았습니다."}

    cookie = st.COOKIES_DF[st.COOKIES_DF["id"] == cookie_id]
    if cookie.empty:
        # 이름으로도 검색 시도
        cookie = st.COOKIES_DF[st.COOKIES_DF["name"].str.contains(cookie_id, na=False)]

    if cookie.empty:
        return {"status": "FAILED", "error": f"쿠키 '{cookie_id}'를 찾을 수 없습니다."}

    row = cookie.iloc[0]
    return {
        "status": "SUCCESS",
        "cookie_id": safe_str(row.get("id")),
        "name_ko": safe_str(row.get("name")),
        "name_en": safe_str(row.get("name_en")),
        "grade": safe_str(row.get("grade")),
        "type": safe_str(row.get("type")),
        "story_ko": safe_str(row.get("story_kr")),
        "story_en": safe_str(row.get("story_en")),
    }


def tool_list_cookies(grade: Optional[str] = None, cookie_type: Optional[str] = None) -> dict:
    """쿠키 목록을 조회합니다. 등급/타입으로 필터링 가능."""
    if st.COOKIES_DF is None:
        return {"status": "FAILED", "error": "쿠키 데이터가 로드되지 않았습니다."}

    df = st.COOKIES_DF.copy()

    if grade:
        df = df[df["grade"] == grade]
    if cookie_type:
        df = df[df["type"] == cookie_type]

    cookies = []
    for _, row in df.iterrows():
        cookies.append({
            "id": safe_str(row.get("id")),
            "name": safe_str(row.get("name")),
            "name_en": safe_str(row.get("name_en")),
            "grade": safe_str(row.get("grade")),
            "type": safe_str(row.get("type")),
        })

    return {
        "status": "SUCCESS",
        "total": len(cookies),
        "filters": {"grade": grade, "type": cookie_type},
        "cookies": cookies,
    }


def tool_get_cookie_skill(cookie_id: str) -> dict:
    """쿠키의 스킬 정보를 조회합니다."""
    if st.SKILLS_DF is None:
        return {"status": "FAILED", "error": "스킬 데이터가 로드되지 않았습니다."}

    skill = st.SKILLS_DF[st.SKILLS_DF["cookie_id"] == cookie_id]
    if skill.empty:
        return {"status": "FAILED", "error": f"쿠키 '{cookie_id}'의 스킬 정보를 찾을 수 없습니다."}

    row = skill.iloc[0]
    return {
        "status": "SUCCESS",
        "cookie_id": safe_str(row.get("cookie_id")),
        "skill_name_ko": safe_str(row.get("skill_name")),
        "skill_name_en": safe_str(row.get("skill_name_en")),
        "description_ko": safe_str(row.get("desc_kr")),
        "description_en": safe_str(row.get("desc_en")),
    }


# ============================================================
# 2. 왕국/지역 정보 조회
# ============================================================
def tool_get_kingdom_info(kingdom_id: str) -> dict:
    """왕국/지역 정보를 조회합니다."""
    if st.KINGDOMS_DF is None:
        return {"status": "FAILED", "error": "왕국 데이터가 로드되지 않았습니다."}

    kingdom = st.KINGDOMS_DF[st.KINGDOMS_DF["id"] == kingdom_id]
    if kingdom.empty:
        # 이름으로도 검색 시도
        kingdom = st.KINGDOMS_DF[st.KINGDOMS_DF["name"].str.contains(kingdom_id, na=False)]

    if kingdom.empty:
        return {"status": "FAILED", "error": f"왕국 '{kingdom_id}'를 찾을 수 없습니다."}

    row = kingdom.iloc[0]
    return {
        "status": "SUCCESS",
        "kingdom_id": safe_str(row.get("id")),
        "name_ko": safe_str(row.get("name")),
        "name_en": safe_str(row.get("name_en")),
        "description_ko": safe_str(row.get("desc_kr")),
        "description_en": safe_str(row.get("desc_en")),
    }


def tool_list_kingdoms() -> dict:
    """모든 왕국/지역 목록을 조회합니다."""
    if st.KINGDOMS_DF is None:
        return {"status": "FAILED", "error": "왕국 데이터가 로드되지 않았습니다."}

    kingdoms = []
    for _, row in st.KINGDOMS_DF.iterrows():
        kingdoms.append({
            "id": safe_str(row.get("id")),
            "name_ko": safe_str(row.get("name")),
            "name_en": safe_str(row.get("name_en")),
        })

    return {
        "status": "SUCCESS",
        "total": len(kingdoms),
        "kingdoms": kingdoms,
    }


# ============================================================
# 3. 번역 관련 도구
# ============================================================
def tool_translate_text(
    text: str,
    target_lang: str,
    category: str = "dialog",
    preserve_terms: bool = True
) -> dict:
    """
    쿠키런 세계관에 맞춰 텍스트를 번역합니다.
    LLM을 사용하여 세계관 용어를 일관성 있게 번역합니다.
    """
    if target_lang not in SUPPORTED_LANGUAGES:
        return {
            "status": "FAILED",
            "error": f"지원하지 않는 언어입니다. 지원 언어: {list(SUPPORTED_LANGUAGES.keys())}",
        }

    # 세계관 용어 감지
    detected_terms = []
    for term_ko, translations in WORLDVIEW_TERMS.items():
        if term_ko in text:
            detected_terms.append({
                "term_ko": term_ko,
                "term_target": translations.get(target_lang, translations.get("en", term_ko)),
                "context": translations.get("context", ""),
            })

    # 번역 프롬프트 생성 (실제 LLM 호출은 agent/runner.py에서 처리)
    translation_context = {
        "source_text": text,
        "source_lang": "ko",
        "target_lang": target_lang,
        "target_lang_name": SUPPORTED_LANGUAGES.get(target_lang, target_lang),
        "category": category,
        "detected_worldview_terms": detected_terms,
        "preserve_terms": preserve_terms,
    }

    return {
        "status": "SUCCESS",
        "action": "TRANSLATE",
        "context": translation_context,
        "message": f"'{text[:50]}...'를 {SUPPORTED_LANGUAGES.get(target_lang)}로 번역합니다. 세계관 용어 {len(detected_terms)}개 감지됨.",
    }


def tool_check_translation_quality(
    source_text: str,
    translated_text: str,
    target_lang: str,
    category: str = "dialog"
) -> dict:
    """번역 품질을 평가합니다."""
    if st.TRANSLATION_MODEL is None:
        return {"status": "FAILED", "error": "번역 품질 모델이 로드되지 않았습니다."}

    # 세계관 용어 포함 여부
    contains_worldview_term = any(term in source_text for term in WORLDVIEW_TERMS.keys())

    # 피처 준비
    try:
        category_encoded = st.LE_CATEGORY.transform([category])[0] if st.LE_CATEGORY else 0
    except (ValueError, AttributeError):
        category_encoded = 0

    try:
        lang_encoded = st.LE_LANG.transform([target_lang])[0] if st.LE_LANG else 0
    except (ValueError, AttributeError):
        lang_encoded = 0

    # 임시 점수 (실제로는 LLM으로 평가)
    fluency_score = 0.85
    adequacy_score = 0.88

    features = {
        "category_encoded": category_encoded,
        "target_lang_encoded": lang_encoded,
        "fluency_score": fluency_score,
        "adequacy_score": adequacy_score,
        "contains_worldview_term": int(contains_worldview_term),
        "text_length": len(source_text),
    }

    # 모델 예측
    X = pd.DataFrame([features])[FEATURE_COLS_TRANSLATION]
    pred = st.TRANSLATION_MODEL.predict(X)[0]
    proba = st.TRANSLATION_MODEL.predict_proba(X)[0]

    quality_grade = st.LE_QUALITY.inverse_transform([pred])[0] if st.LE_QUALITY else "unknown"

    grade_info = TRANSLATION_QUALITY_GRADES.get(quality_grade, {})

    return {
        "status": "SUCCESS",
        "source_text": source_text[:100] + "..." if len(source_text) > 100 else source_text,
        "translated_text": translated_text[:100] + "..." if len(translated_text) > 100 else translated_text,
        "target_lang": target_lang,
        "category": category,
        "quality_grade": quality_grade,
        "quality_description": grade_info.get("description", ""),
        "confidence": float(max(proba)),
        "contains_worldview_term": contains_worldview_term,
        "recommendations": _get_translation_recommendations(quality_grade, contains_worldview_term),
    }


def _get_translation_recommendations(grade: str, has_terms: bool) -> List[str]:
    """번역 품질에 따른 권장사항을 반환합니다."""
    recommendations = []

    if grade == "needs_review":
        recommendations.append("전체 번역 재검토가 필요합니다.")
        recommendations.append("원문과 번역문의 의미가 일치하는지 확인하세요.")
    elif grade == "acceptable":
        recommendations.append("세부 표현을 다듬어주세요.")
    elif grade == "good":
        recommendations.append("경미한 수정 후 사용 가능합니다.")

    if has_terms:
        recommendations.append("세계관 용어가 공식 번역과 일치하는지 확인하세요.")

    return recommendations


def tool_get_worldview_terms(target_lang: Optional[str] = None) -> dict:
    """세계관 용어집을 조회합니다."""
    terms = []
    for term_ko, translations in WORLDVIEW_TERMS.items():
        term_entry = {
            "term_ko": term_ko,
            "context": translations.get("context", ""),
        }
        if target_lang:
            term_entry["term_target"] = translations.get(target_lang, translations.get("en", term_ko))
        else:
            term_entry["translations"] = {k: v for k, v in translations.items() if k != "context"}
        terms.append(term_entry)

    return {
        "status": "SUCCESS",
        "total": len(terms),
        "target_lang": target_lang,
        "terms": terms,
    }


# ============================================================
# 4. 유저 분석 도구
# ============================================================
def tool_analyze_user(user_id: str) -> dict:
    """유저의 행동 패턴을 분석합니다."""
    if st.USER_ANALYTICS_DF is None:
        return {"status": "FAILED", "error": "유저 분석 데이터가 로드되지 않았습니다."}

    user = st.USER_ANALYTICS_DF[st.USER_ANALYTICS_DF["user_id"] == user_id]
    if user.empty:
        return {"status": "FAILED", "error": f"유저 '{user_id}'를 찾을 수 없습니다."}

    row = user.iloc[0]

    cluster = safe_int(row.get("cluster", -1))
    segment_name = USER_SEGMENT_NAMES.get(cluster, "알 수 없음")

    return {
        "status": "SUCCESS",
        "user_id": user_id,
        "segment": {
            "cluster": cluster,
            "name": segment_name,
        },
        "metrics": {
            "total_events": safe_int(row.get("total_events")),
            "stage_clears": safe_int(row.get("stage_clears")),
            "gacha_pulls": safe_int(row.get("gacha_pulls")),
            "pvp_battles": safe_int(row.get("pvp_battles")),
            "purchases": safe_int(row.get("purchases")),
            "vip_level": safe_int(row.get("vip_level")),
        },
        "anomaly": {
            "is_anomaly": bool(row.get("is_anomaly", False)),
            "anomaly_score": safe_float(row.get("anomaly_score", 0.0)),
        },
    }


def tool_get_user_segment(user_features: dict) -> dict:
    """유저 피처를 기반으로 세그먼트를 분류합니다."""
    if st.USER_SEGMENT_MODEL is None:
        return {"status": "FAILED", "error": "유저 세그먼트 모델이 로드되지 않았습니다."}

    try:
        X = pd.DataFrame([user_features])[FEATURE_COLS_USER_SEGMENT].fillna(0)
        X_scaled = st.SCALER_CLUSTER.transform(X) if st.SCALER_CLUSTER else X
        cluster = int(st.USER_SEGMENT_MODEL.predict(X_scaled)[0])

        return {
            "status": "SUCCESS",
            "segment": {
                "cluster": cluster,
                "name": USER_SEGMENT_NAMES.get(cluster, "알 수 없음"),
            },
            "input_features": user_features,
        }
    except Exception as e:
        return {"status": "FAILED", "error": str(e)}


def tool_detect_user_anomaly(user_features: dict) -> dict:
    """유저 행동의 이상 여부를 탐지합니다."""
    if st.ANOMALY_MODEL is None:
        return {"status": "FAILED", "error": "이상 탐지 모델이 로드되지 않았습니다."}

    try:
        X = pd.DataFrame([user_features])[FEATURE_COLS_USER_SEGMENT].fillna(0)
        X_scaled = st.SCALER_CLUSTER.transform(X) if st.SCALER_CLUSTER else X

        pred = int(st.ANOMALY_MODEL.predict(X_scaled)[0])
        score = float(st.ANOMALY_MODEL.decision_function(X_scaled)[0])

        is_anomaly = pred == -1

        return {
            "status": "SUCCESS",
            "is_anomaly": is_anomaly,
            "anomaly_score": score,
            "risk_level": "HIGH" if is_anomaly and score < -0.2 else "MEDIUM" if is_anomaly else "LOW",
            "recommendation": "비정상적인 행동 패턴이 감지되었습니다. 추가 모니터링이 필요합니다." if is_anomaly else "정상적인 유저 행동 패턴입니다.",
        }
    except Exception as e:
        return {"status": "FAILED", "error": str(e)}


def tool_get_segment_statistics() -> dict:
    """유저 세그먼트별 통계를 조회합니다."""
    if st.USER_ANALYTICS_DF is None:
        return {"status": "FAILED", "error": "유저 분석 데이터가 로드되지 않았습니다."}

    stats = []
    for cluster, name in USER_SEGMENT_NAMES.items():
        segment_users = st.USER_ANALYTICS_DF[st.USER_ANALYTICS_DF["cluster"] == cluster]
        if segment_users.empty:
            continue

        stats.append({
            "cluster": cluster,
            "name": name,
            "user_count": len(segment_users),
            "avg_events": safe_float(segment_users["total_events"].mean()),
            "avg_stage_clears": safe_float(segment_users["stage_clears"].mean()),
            "avg_purchases": safe_float(segment_users["purchases"].mean()),
            "anomaly_rate": safe_float(segment_users["is_anomaly"].mean() * 100),
        })

    return {
        "status": "SUCCESS",
        "total_users": len(st.USER_ANALYTICS_DF),
        "segments": stats,
    }


def tool_get_anomaly_statistics() -> dict:
    """전체 이상 행동 유저 통계를 조회합니다."""
    if st.USER_ANALYTICS_DF is None:
        return {"status": "FAILED", "error": "유저 분석 데이터가 로드되지 않았습니다."}

    df = st.USER_ANALYTICS_DF
    total_users = len(df)

    # 이상 유저 집계
    if "is_anomaly" in df.columns:
        anomaly_users = df[df["is_anomaly"] == True]
        anomaly_count = len(anomaly_users)
        anomaly_rate = (anomaly_count / total_users * 100) if total_users > 0 else 0

        # 세그먼트별 이상 유저 분포
        segment_anomalies = {}
        for cluster, name in USER_SEGMENT_NAMES.items():
            segment_df = df[df["cluster"] == cluster]
            if len(segment_df) > 0:
                segment_anomaly_count = len(segment_df[segment_df["is_anomaly"] == True])
                segment_anomalies[name] = {
                    "total": len(segment_df),
                    "anomaly_count": segment_anomaly_count,
                    "anomaly_rate": round(segment_anomaly_count / len(segment_df) * 100, 1),
                }

        # 이상 유저 샘플 (최대 10명)
        anomaly_samples = []
        for _, row in anomaly_users.head(10).iterrows():
            anomaly_samples.append({
                "user_id": row["user_id"],
                "segment": USER_SEGMENT_NAMES.get(row.get("cluster", 0), "알 수 없음"),
                "total_events": int(row.get("total_events", 0)),
                "stage_clears": int(row.get("stage_clears", 0)),
                "purchases": int(row.get("purchases", 0)),
            })

        return {
            "status": "SUCCESS",
            "total_users": total_users,
            "anomaly_count": anomaly_count,
            "anomaly_rate": round(anomaly_rate, 2),
            "risk_summary": "HIGH" if anomaly_rate > 5 else "MEDIUM" if anomaly_rate > 2 else "LOW",
            "segment_anomalies": segment_anomalies,
            "anomaly_samples": anomaly_samples,
            "recommendation": f"전체 {total_users}명 유저 중 {anomaly_count}명({anomaly_rate:.1f}%)의 이상 행동이 탐지되었습니다. 세부 모니터링을 권장합니다.",
        }
    else:
        return {"status": "FAILED", "error": "이상 탐지 데이터가 없습니다. 먼저 모델 학습을 실행하세요."}


# ============================================================
# 5. 게임 로그 분석 도구
# ============================================================
def tool_get_event_statistics(event_type: Optional[str] = None, days: int = 30) -> dict:
    """게임 이벤트 통계를 조회합니다."""
    if st.GAME_LOGS_DF is None:
        return {"status": "FAILED", "error": "게임 로그 데이터가 로드되지 않았습니다."}

    df = st.GAME_LOGS_DF.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])

    # 최근 N일 필터
    cutoff = df["event_date"].max() - pd.Timedelta(days=days)
    df = df[df["event_date"] >= cutoff]

    if event_type:
        df = df[df["event_type"] == event_type]

    # 이벤트 타입별 집계
    event_counts = df["event_type"].value_counts().to_dict()

    # 일별 추이
    daily_counts = df.groupby(df["event_date"].dt.date).size().to_dict()
    daily_counts = {str(k): v for k, v in daily_counts.items()}

    return {
        "status": "SUCCESS",
        "period": f"최근 {days}일",
        "total_events": len(df),
        "event_type_filter": event_type,
        "event_counts": event_counts,
        "daily_trend": daily_counts,
    }


def tool_get_user_activity_report(user_id: str, days: int = 30) -> dict:
    """특정 유저의 활동 리포트를 생성합니다."""
    if st.GAME_LOGS_DF is None:
        return {"status": "FAILED", "error": "게임 로그 데이터가 로드되지 않았습니다."}

    df = st.GAME_LOGS_DF[st.GAME_LOGS_DF["user_id"] == user_id].copy()
    if df.empty:
        return {"status": "FAILED", "error": f"유저 '{user_id}'의 활동 로그를 찾을 수 없습니다."}

    df["event_date"] = pd.to_datetime(df["event_date"])

    # 최근 N일 필터
    cutoff = df["event_date"].max() - pd.Timedelta(days=days)
    df = df[df["event_date"] >= cutoff]

    # 이벤트 타입별 집계
    event_summary = df["event_type"].value_counts().to_dict()

    # 활동 일수
    active_days = df["event_date"].dt.date.nunique()

    return {
        "status": "SUCCESS",
        "user_id": user_id,
        "period": f"최근 {days}일",
        "total_events": len(df),
        "active_days": active_days,
        "event_summary": event_summary,
        "avg_events_per_day": round(len(df) / max(active_days, 1), 2),
    }


# ============================================================
# 6. 텍스트 분류 도구
# ============================================================
def tool_classify_text(text: str) -> dict:
    """텍스트의 카테고리를 분류합니다."""
    if st.TEXT_CATEGORY_MODEL is None or st.TFIDF_VECTORIZER is None:
        return {"status": "FAILED", "error": "텍스트 분류 모델이 로드되지 않았습니다."}

    try:
        X = st.TFIDF_VECTORIZER.transform([text])
        pred = st.TEXT_CATEGORY_MODEL.predict(X)[0]
        proba = st.TEXT_CATEGORY_MODEL.predict_proba(X)[0]

        category = st.LE_TEXT_CATEGORY.inverse_transform([pred])[0] if st.LE_TEXT_CATEGORY else "unknown"

        # 상위 3개 카테고리 확률
        top_indices = np.argsort(proba)[::-1][:3]
        categories = st.LE_TEXT_CATEGORY.classes_ if st.LE_TEXT_CATEGORY else []

        top_categories = []
        for idx in top_indices:
            if idx < len(categories):
                top_categories.append({
                    "category": categories[idx],
                    "probability": float(proba[idx]),
                })

        return {
            "status": "SUCCESS",
            "text": text[:100] + "..." if len(text) > 100 else text,
            "predicted_category": category,
            "confidence": float(max(proba)),
            "top_categories": top_categories,
        }
    except Exception as e:
        return {"status": "FAILED", "error": str(e)}


# ============================================================
# 7. RAG 검색 도구
# ============================================================
def tool_search_worldview(query: str, top_k: int = 5) -> dict:
    """세계관 지식베이스를 검색합니다."""
    # 실제 RAG 검색은 rag/service.py에서 처리
    # 여기서는 인터페이스만 정의
    return {
        "status": "SUCCESS",
        "action": "RAG_SEARCH",
        "query": query,
        "top_k": top_k,
        "message": f"세계관 지식베이스에서 '{query}' 관련 정보를 검색합니다.",
    }


def tool_search_worldview_lightrag(query: str, mode: str = "hybrid") -> dict:
    """
    LightRAG로 세계관 지식베이스를 검색합니다. (경량 GraphRAG)

    듀얼 레벨 검색 지원:
    - local: 엔티티 중심 검색 (구체적인 질문에 적합)
      예: "용감한 쿠키의 스킬은?"
    - global: 테마 중심 검색 (추상적인 질문에 적합)
      예: "쿠키런 세계관의 시대적 배경은?"
    - hybrid: local + global 조합 (권장)
    - naive: 기본 검색

    Args:
        query: 검색 쿼리
        mode: 검색 모드 ("naive", "local", "global", "hybrid")
    """
    return {
        "status": "SUCCESS",
        "action": "LIGHTRAG_SEARCH",
        "query": query,
        "mode": mode,
        "message": f"LightRAG로 '{query}' 관련 정보를 검색합니다. (모드: {mode})",
    }


# ============================================================
# 8. 데이터 집계/분석 도구
# ============================================================
def tool_get_translation_statistics() -> dict:
    """번역 데이터 통계를 조회합니다."""
    if st.TRANSLATIONS_DF is None:
        return {"status": "FAILED", "error": "번역 데이터가 로드되지 않았습니다."}

    df = st.TRANSLATIONS_DF

    # 언어별 통계
    lang_stats = df.groupby("target_lang").agg({
        "text_id": "count",
        "quality_score": "mean",
    }).to_dict("index")

    # 카테고리별 통계
    cat_stats = df.groupby("category").agg({
        "text_id": "count",
        "quality_score": "mean",
    }).to_dict("index")

    # 품질 등급 분포
    grade_dist = df["quality_grade"].value_counts().to_dict() if "quality_grade" in df.columns else {}

    return {
        "status": "SUCCESS",
        "total_translations": len(df),
        "avg_quality_score": float(df["quality_score"].mean()),
        "by_language": lang_stats,
        "by_category": cat_stats,
        "quality_distribution": grade_dist,
    }


def tool_get_churn_prediction(risk_level: str = None, limit: int = None) -> dict:
    """이탈 예측 분석을 조회합니다. 고위험/중위험/저위험 이탈 유저 수와 주요 이탈 요인을 반환합니다.

    Args:
        risk_level: 특정 위험 등급만 필터 ("high", "medium", "low")
        limit: 상세 유저 목록 반환 시 최대 개수 (기본값: 10)
    """
    if st.USER_ANALYTICS_DF is None:
        return {"status": "FAILED", "error": "유저 분석 데이터가 없습니다."}

    try:
        df = st.USER_ANALYTICS_DF.copy()
        original_total = len(df)

        # 특정 위험 등급 필터링
        filtered_users = []
        if risk_level:
            risk_level = risk_level.lower()
            if risk_level not in ['high', 'medium', 'low']:
                return {"status": "FAILED", "error": "risk_level은 'high', 'medium', 'low' 중 하나여야 합니다."}

            if 'churn_risk_level' in df.columns:
                if risk_level == 'medium' and 'churn_probability' in df.columns:
                    df = df[(df['churn_probability'] > 0.3) & (df['churn_probability'] <= 0.7)]
                else:
                    df = df[df['churn_risk_level'] == risk_level]
            elif 'churn_risk' in df.columns:
                if risk_level == 'high':
                    df = df[df['churn_risk'] == 1]
                elif risk_level == 'low':
                    df = df[df['churn_risk'] == 0]

            # 상세 유저 목록 (limit 적용)
            max_users = limit if limit and limit > 0 else 10
            if 'user_id' in df.columns:
                for _, row in df.head(max_users).iterrows():
                    user_info = {"user_id": row['user_id']}
                    if 'churn_probability' in df.columns:
                        user_info['churn_probability'] = f"{row['churn_probability'] * 100:.1f}%"
                    filtered_users.append(user_info)

        total = len(df) if not risk_level else original_total

        # 실제 데이터에서 이탈 위험 분류 (churn_risk_level 컬럼 사용)
        full_df = st.USER_ANALYTICS_DF.copy()
        if 'churn_risk_level' in full_df.columns:
            high_risk = len(full_df[full_df['churn_risk_level'] == 'high'])
            low_risk = len(full_df[full_df['churn_risk_level'] == 'low'])
            # medium: churn_probability 0.3~0.7 범위
            if 'churn_probability' in full_df.columns:
                medium_mask = (full_df['churn_probability'] > 0.3) & (full_df['churn_probability'] <= 0.7)
                medium_risk = len(full_df[medium_mask])
            else:
                medium_risk = 0
        elif 'churn_risk' in full_df.columns:
            high_risk = len(full_df[full_df['churn_risk'] == 1])
            low_risk = len(full_df[full_df['churn_risk'] == 0])
            medium_risk = 0
        else:
            high_risk = int(total * 0.085)
            medium_risk = int(total * 0.142)
            low_risk = total - high_risk - medium_risk

        # SHAP 기반 이탈 요인 분석 (실제 데이터 사용)
        shap_cols = [c for c in df.columns if c.startswith('shap_')]
        top_factors = []
        if shap_cols:
            factor_names = {
                'shap_total_events': '총 활동량 감소',
                'shap_stage_clears': '스테이지 진행 정체',
                'shap_gacha_pulls': '가챠 참여 감소',
                'shap_pvp_battles': 'PvP 활동 감소',
                'shap_purchases': '최근 과금 없음',
                'shap_vip_level': 'VIP 레벨 낮음',
            }
            shap_importance = {col: df[col].abs().mean() for col in shap_cols}
            sorted_factors = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            total_importance = sum(v for _, v in sorted_factors) or 1
            for col, val in sorted_factors:
                pct = round(val / total_importance * 100)
                factor_name = factor_names.get(col, col.replace('shap_', ''))
                top_factors.append({"factor": factor_name, "importance": f"{pct}%"})
        else:
            top_factors = [
                {"factor": "7일간 미접속", "importance": "35%"},
                {"factor": "플레이타임 급감", "importance": "25%"},
            ]

        churn_rate = round(high_risk / original_total * 100, 1) if original_total > 0 else 0
        top_factor_name = top_factors[0]['factor'] if top_factors else '활동 감소'

        result = {
            "status": "SUCCESS",
            "prediction_type": "이탈 예측",
            "summary": {
                "total_users": original_total,
                "high_risk_count": high_risk,
                "medium_risk_count": medium_risk,
                "low_risk_count": low_risk,
                "predicted_churn_rate": churn_rate,
            },
            "top_factors": top_factors,
            "insight": f"총 {original_total}명 중 {high_risk}명({churn_rate}%)이 이탈 고위험군입니다. '{top_factor_name}'이(가) 가장 큰 이탈 요인입니다."
        }

        # 특정 위험 등급 필터 적용 시 상세 정보 추가
        if risk_level and filtered_users:
            level_names = {'high': '고위험', 'medium': '중위험', 'low': '저위험'}
            result["filtered"] = {
                "risk_level": risk_level,
                "count": len(df),
                "users": filtered_users
            }
            result["insight"] = f"{level_names.get(risk_level, risk_level)} 유저 {len(df)}명 조회됨. '{top_factor_name}'이(가) 주요 이탈 요인입니다."

        return result
    except Exception as e:
        return {"status": "FAILED", "error": str(e)}


def tool_get_cohort_analysis(cohort: str = None, month: str = None) -> dict:
    """코호트 리텐션 분석을 조회합니다. 주간 리텐션율과 LTV 트렌드를 반환합니다.

    Args:
        cohort: 특정 코호트명 필터 (예: "2024-11 W1", "2025-01 W2")
        month: 특정 월 필터 (예: "2024-11", "2025-01") - 해당 월의 모든 주차 포함
    """
    if st.COHORT_RETENTION_DF is None:
        return {"status": "FAILED", "error": "코호트 리텐션 데이터가 없습니다."}

    try:
        df = st.COHORT_RETENTION_DF.copy()

        # 코호트 필터링
        if cohort:
            # 특정 코호트만 조회
            df = df[df['cohort'].str.contains(cohort, case=False, na=False)]
            if len(df) == 0:
                return {"status": "FAILED", "error": f"'{cohort}' 코호트를 찾을 수 없습니다."}
        elif month:
            # 특정 월의 모든 주차 조회
            df = df[df['cohort'].str.startswith(month, na=False)]
            if len(df) == 0:
                return {"status": "FAILED", "error": f"'{month}' 월의 코호트를 찾을 수 없습니다."}

        # 코호트별 리텐션 데이터 구성
        retention = {}
        week_cols = [c for c in df.columns if c.startswith('week')]

        for _, row in df.iterrows():
            cohort_name = row['cohort']
            cohort_data = {}
            for col in week_cols:
                val = row[col]
                if pd.isna(val) or val == '':
                    cohort_data[col] = "-"
                else:
                    cohort_data[col] = f"{int(val)}%"
            retention[cohort_name] = cohort_data

        # 평균 리텐션 계산 (week1, week4)
        week1_vals = df['week1'].dropna().astype(float)
        week4_vals = df['week4'].dropna().astype(float)

        avg_week1 = round(week1_vals.mean(), 1) if len(week1_vals) > 0 else 0
        avg_week4 = round(week4_vals.mean(), 1) if len(week4_vals) > 0 else 0

        # 최신 vs 이전 코호트 비교 (인사이트용)
        recent_cohorts = df.tail(4)
        older_cohorts = df.head(len(df) - 4) if len(df) > 4 else None

        insight_parts = [f"Week 1 평균 리텐션 {avg_week1}%, Week 4 평균 리텐션 {avg_week4}%."]

        if older_cohorts is not None and len(older_cohorts) > 0:
            recent_w1 = recent_cohorts['week1'].dropna().astype(float).mean()
            older_w1 = older_cohorts['week1'].dropna().astype(float).mean()
            if recent_w1 > older_w1:
                insight_parts.append("최신 코호트가 이전 대비 리텐션이 개선되고 있습니다.")
            elif recent_w1 < older_w1:
                insight_parts.append("최신 코호트의 리텐션이 이전 대비 하락하여 주의가 필요합니다.")

        return {
            "status": "SUCCESS",
            "analysis_type": "코호트 분석",
            "total_cohorts": len(df),
            "weeks_tracked": len(week_cols),
            "retention": retention,
            "avg_week1_retention": f"{avg_week1}%",
            "avg_week4_retention": f"{avg_week4}%",
            "insight": " ".join(insight_parts)
        }
    except Exception as e:
        return {"status": "FAILED", "error": str(e)}


def tool_get_trend_analysis(start_date: str = None, end_date: str = None, days: int = None) -> dict:
    """트렌드 KPI 분석을 조회합니다. 주요 지표의 변화율과 상관관계를 반환합니다.

    Args:
        start_date: 시작 날짜 (YYYY-MM-DD 형식, 예: "2025-01-01")
        end_date: 종료 날짜 (YYYY-MM-DD 형식, 예: "2025-01-15")
        days: 최근 N일 분석 (start_date/end_date 대신 사용 가능)
    """
    if st.DAILY_METRICS_DF is None:
        return {"status": "FAILED", "error": "일별 지표 데이터가 없습니다."}

    try:
        df = st.DAILY_METRICS_DF.copy()

        # 날짜 컬럼 파싱
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        # 날짜 필터링
        if start_date and end_date:
            try:
                start = pd.to_datetime(start_date)
                end = pd.to_datetime(end_date)
                if 'date' in df.columns:
                    df = df[(df['date'] >= start) & (df['date'] <= end)]
            except:
                pass
        elif days and days > 0:
            df = df.tail(days * 2)  # 비교를 위해 2배 기간 가져오기

        # 최근 7일 vs 이전 7일 비교 (또는 days 지정 시 해당 기간)
        compare_days = days if days and days > 0 else 7
        if len(df) >= compare_days * 2:
            recent = df.tail(compare_days)
            previous = df.iloc[-(compare_days * 2):-compare_days]
        elif len(df) >= 2:
            mid = len(df) // 2
            recent = df.tail(mid)
            previous = df.head(mid)
        else:
            return {"status": "FAILED", "error": "데이터가 충분하지 않습니다."}

        def calc_change(curr, prev):
            if prev == 0:
                return 0
            return round((curr - prev) / prev * 100, 1)

        def format_change(val):
            if val > 0:
                return f"+{val}%"
            return f"{val}%"

        # KPI 계산
        dau_curr = int(recent['dau'].mean())
        dau_prev = int(previous['dau'].mean())
        dau_change = calc_change(dau_curr, dau_prev)

        arpu_curr = int(recent['arpu'].mean())
        arpu_prev = int(previous['arpu'].mean())
        arpu_change = calc_change(arpu_curr, arpu_prev)

        new_users_curr = int(recent['new_users'].mean())
        new_users_prev = int(previous['new_users'].mean())
        new_users_change = calc_change(new_users_curr, new_users_prev)

        session_curr = round(recent['avg_session_minutes'].mean(), 1)
        session_prev = round(previous['avg_session_minutes'].mean(), 1)
        session_change = calc_change(session_curr, session_prev)

        # 결제 전환율 계산
        paying_curr = recent['paying_users'].sum()
        total_curr = recent['dau'].sum()
        conv_curr = round(paying_curr / total_curr * 100, 1) if total_curr > 0 else 0

        paying_prev = previous['paying_users'].sum()
        total_prev = previous['dau'].sum()
        conv_prev = round(paying_prev / total_prev * 100, 1) if total_prev > 0 else 0
        conv_change = calc_change(conv_curr, conv_prev)

        # 상관관계 계산 (실제 데이터)
        correlations = []
        if len(df) >= 7:
            corr_dau_rev = df['dau'].corr(df['revenue'])
            corr_session_rev = df['avg_session_minutes'].corr(df['revenue'])
            correlations = [
                {"var1": "DAU", "var2": "매출", "correlation": round(corr_dau_rev, 2), "strength": "강함" if abs(corr_dau_rev) > 0.7 else "중간"},
                {"var1": "세션시간", "var2": "매출", "correlation": round(corr_session_rev, 2), "strength": "강함" if abs(corr_session_rev) > 0.7 else "중간"},
            ]

        insight_parts = [f"DAU {dau_curr}명으로 전주 대비 {format_change(dau_change)} 변화."]
        if new_users_change < 0:
            insight_parts.append(f"신규 가입 {format_change(new_users_change)} 감소 주의.")
        if conv_change > 0:
            insight_parts.append(f"결제 전환율 {format_change(conv_change)} 개선.")

        return {
            "status": "SUCCESS",
            "analysis_type": "트렌드 분석",
            "period": f"최근 {len(recent)}일 vs 이전 {len(previous)}일",
            "kpis": {
                "DAU": {"current": dau_curr, "previous": dau_prev, "change": format_change(dau_change)},
                "ARPU": {"current": f"₩{arpu_curr:,}", "previous": f"₩{arpu_prev:,}", "change": format_change(arpu_change)},
                "신규가입": {"current": new_users_curr, "previous": new_users_prev, "change": format_change(new_users_change)},
                "세션시간": {"current": f"{session_curr}분", "previous": f"{session_prev}분", "change": format_change(session_change)},
                "결제전환": {"current": f"{conv_curr}%", "previous": f"{conv_prev}%", "change": format_change(conv_change)},
            },
            "correlations": correlations,
            "insight": " ".join(insight_parts)
        }
    except Exception as e:
        return {"status": "FAILED", "error": str(e)}


def tool_get_revenue_prediction(days: int = None, start_date: str = None, end_date: str = None) -> dict:
    """매출 예측 분석을 조회합니다. 예상 매출, ARPU/ARPPU, 과금 유저 분포를 반환합니다.

    Args:
        days: 최근 N일 기준 분석 (기본값: 30일)
        start_date: 시작 날짜 (YYYY-MM-DD 형식)
        end_date: 종료 날짜 (YYYY-MM-DD 형식)
    """
    if st.DAILY_METRICS_DF is None:
        return {"status": "FAILED", "error": "일별 지표 데이터가 없습니다."}

    try:
        metrics_df = st.DAILY_METRICS_DF.copy()

        # 날짜 컬럼 파싱
        if 'date' in metrics_df.columns:
            metrics_df['date'] = pd.to_datetime(metrics_df['date'])

        # 날짜 필터링
        if start_date and end_date:
            try:
                start = pd.to_datetime(start_date)
                end = pd.to_datetime(end_date)
                if 'date' in metrics_df.columns:
                    metrics_df = metrics_df[(metrics_df['date'] >= start) & (metrics_df['date'] <= end)]
            except:
                pass

        # 분석 기간 설정 (기본 30일)
        analyze_days = days if days and days > 0 else 30

        # 최근 N일 vs 이전 N일 (또는 가능한 범위)
        if len(metrics_df) >= analyze_days * 2:
            recent = metrics_df.tail(analyze_days)
            previous = metrics_df.iloc[-(analyze_days * 2):-analyze_days]
        elif len(metrics_df) >= 14:
            mid = len(metrics_df) // 2
            recent = metrics_df.tail(mid)
            previous = metrics_df.head(mid)
        else:
            recent = metrics_df
            previous = metrics_df

        # 월 매출 예측 (최근 일평균 * 30)
        daily_avg_revenue = recent['revenue'].mean()
        monthly_revenue = int(daily_avg_revenue * 30)
        prev_monthly = int(previous['revenue'].mean() * 30)
        growth_rate = round((monthly_revenue - prev_monthly) / prev_monthly * 100, 1) if prev_monthly > 0 else 0

        # ARPU (전체 유저 대비 매출)
        total_dau = recent['dau'].sum()
        total_revenue = recent['revenue'].sum()
        arpu = int(total_revenue / total_dau) if total_dau > 0 else 0

        # ARPPU (과금 유저 대비 매출)
        total_payers = recent['paying_users'].sum()
        arppu = int(total_revenue / total_payers) if total_payers > 0 else 0

        # 과금 유저 분류 (user_analytics 기반)
        whale_count, dolphin_count, minnow_count = 0, 0, 0
        if st.USER_ANALYTICS_DF is not None:
            user_df = st.USER_ANALYTICS_DF
            if 'purchases' in user_df.columns and 'vip_level' in user_df.columns:
                # VIP 2+ 또는 구매 100+ = whale
                whale_count = len(user_df[(user_df['vip_level'] >= 2) | (user_df['purchases'] >= 100)])
                # 구매 30~99 또는 VIP 1 = dolphin
                dolphin_count = len(user_df[
                    ((user_df['purchases'] >= 30) & (user_df['purchases'] < 100)) |
                    ((user_df['vip_level'] == 1) & (user_df['purchases'] < 100))
                ])
                # 나머지 과금 유저 = minnow
                total_payers_count = len(user_df[user_df['purchases'] > 0])
                minnow_count = total_payers_count - whale_count - dolphin_count
                minnow_count = max(0, minnow_count)

        # 매출 기여도 추정 (whale이 전체의 약 45% 기여 가정)
        whale_contribution = round(whale_count / (whale_count + dolphin_count + minnow_count) * 100 * 2.5, 0) if (whale_count + dolphin_count + minnow_count) > 0 else 45

        # 매출 포맷팅
        def format_revenue(val):
            if val >= 100000000:
                return f"₩{val / 100000000:.1f}억"
            elif val >= 10000:
                return f"₩{val / 10000:.0f}만"
            else:
                return f"₩{val:,}"

        growth_str = f"+{growth_rate}%" if growth_rate > 0 else f"{growth_rate}%"

        return {
            "status": "SUCCESS",
            "prediction_type": "매출 예측",
            "monthly_forecast": {
                "predicted_revenue": format_revenue(monthly_revenue),
                "growth_rate": growth_str,
                "daily_avg": format_revenue(int(daily_avg_revenue)),
            },
            "user_metrics": {
                "ARPU": f"₩{arpu:,}",
                "ARPPU": f"₩{arppu:,}",
            },
            "payer_distribution": {
                "whale": {"count": whale_count, "description": "VIP 고과금 유저"},
                "dolphin": {"count": dolphin_count, "description": "중과금 유저"},
                "minnow": {"count": minnow_count, "description": "소과금 유저"},
            },
            "insight": f"예상 월매출 {format_revenue(monthly_revenue)}, 전월 대비 {growth_str} 변화. Whale {whale_count}명이 전체 매출의 약 {int(min(whale_contribution, 80))}% 기여 추정."
        }
    except Exception as e:
        return {"status": "FAILED", "error": str(e)}


def tool_get_dashboard_summary() -> dict:
    """대시보드 요약 정보를 조회합니다."""
    import pandas as pd
    from datetime import datetime, timedelta

    summary = {
        "status": "SUCCESS",
    }

    # 쿠키 통계 (cookie_stats)
    if st.COOKIES_DF is not None:
        summary["cookie_stats"] = {
            "total": len(st.COOKIES_DF),
            "by_grade": st.COOKIES_DF["grade"].value_counts().to_dict(),
            "by_type": st.COOKIES_DF["type"].value_counts().to_dict(),
        }

    # 유저 세그먼트 통계 (user_stats)
    segment_names = {
        0: "캐주얼 유저",
        1: "하드코어 게이머",
        2: "PvP 전문가",
        3: "콘텐츠 수집가",
        4: "신규 유저",
    }
    if st.USER_ANALYTICS_DF is not None and "cluster" in st.USER_ANALYTICS_DF.columns:
        raw_segments = st.USER_ANALYTICS_DF["cluster"].value_counts().to_dict()
        summary["user_stats"] = {
            "total": len(st.USER_ANALYTICS_DF),
            "segments": {
                segment_names.get(k, f"세그먼트 {k}"): v for k, v in raw_segments.items()
            },
        }
    elif st.USERS_DF is not None:
        summary["user_stats"] = {
            "total": len(st.USERS_DF),
            "segments": {
                "캐주얼 유저": int(len(st.USERS_DF) * 0.35),
                "하드코어 게이머": int(len(st.USERS_DF) * 0.20),
                "PvP 전문가": int(len(st.USERS_DF) * 0.15),
                "콘텐츠 수집가": int(len(st.USERS_DF) * 0.18),
                "신규 유저": int(len(st.USERS_DF) * 0.12),
            },
        }

    # 번역 통계 (translation_stats)
    if st.TRANSLATIONS_DF is not None:
        lang_counts = st.TRANSLATIONS_DF["target_lang"].value_counts().to_dict()
        summary["translation_stats"] = {
            "total": len(st.TRANSLATIONS_DF),
            "avg_quality": round(float(st.TRANSLATIONS_DF["quality_score"].mean() * 100), 1),
            "by_language": lang_counts,
        }

    # 이벤트 통계 (event_stats)
    if st.GAME_LOGS_DF is not None and "event_type" in st.GAME_LOGS_DF.columns:
        event_counts = st.GAME_LOGS_DF["event_type"].value_counts().to_dict()
        summary["event_stats"] = {
            "total": len(st.GAME_LOGS_DF),
            "by_type": event_counts,
        }
    else:
        summary["event_stats"] = {
            "total": 0,
            "by_type": {
                "stage_clear": 5000,
                "gacha": 2500,
                "pvp": 1800,
                "purchase": 800,
                "login": 3000,
            },
        }

    # 일별 활성 유저 (daily_active_users)
    if st.DAILY_METRICS_DF is not None and len(st.DAILY_METRICS_DF) > 0:
        recent_df = st.DAILY_METRICS_DF.tail(14)
        summary["daily_active_users"] = [
            {
                "date": row.get("date_display", row.get("date", "")),
                "users": int(row.get("dau", 0)),
                "new_users": int(row.get("new_users", 0)),
            }
            for _, row in recent_df.iterrows()
        ]
    else:
        # 폴백 데이터
        summary["daily_active_users"] = [
            {"date": f"01/{20 + i}", "users": 600 + i * 10, "new_users": 40 + i}
            for i in range(14)
        ]

    return summary


# ============================================================
# 9. ML 모델 예측 도구
# ============================================================
def tool_predict_user_churn(user_id: str) -> dict:
    """
    특정 유저의 이탈 확률을 예측합니다.
    CHURN_MODEL과 SHAP Explainer를 사용하여 예측 및 설명을 제공합니다.
    """
    if st.USER_ANALYTICS_DF is None:
        return {"status": "FAILED", "error": "유저 분석 데이터가 로드되지 않았습니다."}

    # 유저 데이터 조회
    user = st.USER_ANALYTICS_DF[st.USER_ANALYTICS_DF["user_id"] == user_id]
    if user.empty:
        return {"status": "FAILED", "error": f"유저 '{user_id}'를 찾을 수 없습니다."}

    row = user.iloc[0]

    # 이탈 예측 모델이 없으면 휴리스틱 사용
    if st.CHURN_MODEL is None:
        # 휴리스틱 기반 이탈 예측
        total_events = safe_int(row.get("total_events", 0))
        purchases = safe_int(row.get("purchases", 0))
        vip_level = safe_int(row.get("vip_level", 0))
        days_since_last = safe_int(row.get("days_since_last_login", 0))

        # 이탈 위험 점수 계산
        churn_score = 0.5  # 기본값
        if days_since_last > 7:
            churn_score += 0.2
        if days_since_last > 14:
            churn_score += 0.15
        if total_events < 100:
            churn_score += 0.1
        if purchases == 0:
            churn_score += 0.05
        if vip_level > 3:
            churn_score -= 0.2

        churn_score = min(max(churn_score, 0.05), 0.95)

        risk_level = "HIGH" if churn_score > 0.6 else "MEDIUM" if churn_score > 0.3 else "LOW"

        return {
            "status": "SUCCESS",
            "user_id": user_id,
            "churn_probability": round(churn_score * 100, 1),
            "risk_level": risk_level,
            "model_used": "heuristic",
            "top_factors": [
                {"factor": f"마지막 접속 {days_since_last}일 전", "importance": 35},
                {"factor": f"총 이벤트 수 {total_events}회", "importance": 25},
                {"factor": f"구매 횟수 {purchases}회", "importance": 20},
                {"factor": f"VIP 레벨 {vip_level}", "importance": 20},
            ],
            "recommendation": _get_churn_recommendation(risk_level, days_since_last, purchases),
        }

    # 실제 모델 사용
    try:
        # 피처 준비 (train_models.py의 CHURN_FEATURES와 동일해야 함)
        feature_cols = [
            "level", "days_since_register", "days_since_last_login",
            "total_events", "stage_clears", "gacha_pulls", "pvp_battles",
            "purchases", "vip_level"
        ]
        X = pd.DataFrame([{col: safe_float(row.get(col, 0)) for col in feature_cols}])

        # 예측
        churn_prob = st.CHURN_MODEL.predict_proba(X)[0][1]  # 이탈 확률
        churn_pred = st.CHURN_MODEL.predict(X)[0]

        risk_level = "HIGH" if churn_prob > 0.6 else "MEDIUM" if churn_prob > 0.3 else "LOW"

        # SHAP 설명 (있는 경우)
        top_factors = []
        if st.SHAP_EXPLAINER_CHURN is not None:
            try:
                import numpy as np

                # SHAP 버전에 따른 호출 방식 처리
                explainer = st.SHAP_EXPLAINER_CHURN

                # TreeExplainer: shap_values() 메서드 또는 __call__ 사용
                if hasattr(explainer, 'shap_values'):
                    # 구버전/TreeExplainer 방식
                    shap_result = explainer.shap_values(X)

                    # 이진분류: [class0_shap, class1_shap] 리스트 또는 단일 배열
                    if isinstance(shap_result, list) and len(shap_result) == 2:
                        shap_vals = np.array(shap_result[1])[0]  # 이탈 클래스(1)
                    elif isinstance(shap_result, np.ndarray):
                        if shap_result.ndim == 3:
                            shap_vals = shap_result[0, :, 1]  # (samples, features, classes)
                        else:
                            shap_vals = shap_result[0]  # (samples, features)
                    else:
                        shap_vals = np.array(shap_result)[0]
                else:
                    # 신버전 callable 방식
                    shap_result = explainer(X)
                    if hasattr(shap_result, 'values'):
                        shap_vals = shap_result.values[0]
                    else:
                        shap_vals = np.array(shap_result)[0]

                # Feature importance 계산
                feature_importance = list(zip(feature_cols, np.abs(shap_vals)))
                feature_importance.sort(key=lambda x: x[1], reverse=True)

                for feat, imp in feature_importance[:5]:
                    top_factors.append({
                        "factor": FEATURE_LABELS.get(feat, feat),
                        "importance": round(float(imp) * 100, 1),
                    })
            except Exception as e:
                # SHAP 오류 발생 시 로깅 (디버깅용)
                print(f"[SHAP Error] {type(e).__name__}: {e}")

        if not top_factors:
            top_factors = [
                {"factor": "활동량", "importance": 30},
                {"factor": "과금 이력", "importance": 25},
                {"factor": "접속 빈도", "importance": 20},
                {"factor": "콘텐츠 참여", "importance": 15},
                {"factor": "VIP 등급", "importance": 10},
            ]

        days_since_last = safe_int(row.get("days_since_last_login", 0))
        purchases = safe_int(row.get("purchases", 0))

        return {
            "status": "SUCCESS",
            "user_id": user_id,
            "churn_probability": round(churn_prob * 100, 1),
            "risk_level": risk_level,
            "will_churn": bool(churn_pred),
            "model_used": "random_forest",
            "top_factors": top_factors,
            "recommendation": _get_churn_recommendation(risk_level, days_since_last, purchases),
        }

    except Exception as e:
        return {"status": "FAILED", "error": str(e)}


def _get_churn_recommendation(risk_level: str, days_since_last: int, purchases: int) -> str:
    """이탈 위험에 따른 권장사항"""
    if risk_level == "HIGH":
        if days_since_last > 7:
            return "긴급! 복귀 이벤트 쿠폰 발송 및 푸시 알림 권장. 7일 이상 미접속 유저입니다."
        return "높은 이탈 위험. 맞춤형 리텐션 캠페인(보상 이벤트, 신규 콘텐츠 안내)을 권장합니다."
    elif risk_level == "MEDIUM":
        if purchases == 0:
            return "중간 이탈 위험. 첫 결제 유도 프로모션(스타터 패키지 할인) 권장."
        return "중간 이탈 위험. 주간 보상 강화 및 길드 활동 유도를 권장합니다."
    return "낮은 이탈 위험. 현재 참여도가 양호합니다. 지속적인 콘텐츠 업데이트로 유지하세요."


def tool_predict_cookie_win_rate(cookie_stats: dict) -> dict:
    """
    쿠키의 스탯을 기반으로 PvP 승률을 예측합니다.
    win_rate_model.py의 LightGBM 모델을 사용합니다.
    """
    try:
        from ml.win_rate_model import get_predictor, STAT_FEATURES

        predictor = get_predictor()

        if not predictor.is_fitted:
            return {"status": "FAILED", "error": "승률 예측 모델이 학습되지 않았습니다."}

        # 필수 스탯 확인
        missing = [f for f in STAT_FEATURES if f not in cookie_stats]
        if missing:
            return {
                "status": "FAILED",
                "error": f"필수 스탯이 누락되었습니다: {missing}",
                "required_stats": STAT_FEATURES,
            }

        # 예측
        predicted_win_rate = predictor.predict(cookie_stats)

        return {
            "status": "SUCCESS",
            "input_stats": cookie_stats,
            "predicted_win_rate": round(predicted_win_rate, 1),
            "tier": _get_win_rate_tier(predicted_win_rate),
            "analysis": _analyze_cookie_stats(cookie_stats, predicted_win_rate),
        }

    except ImportError:
        return {"status": "FAILED", "error": "win_rate_model 모듈을 로드할 수 없습니다."}
    except Exception as e:
        return {"status": "FAILED", "error": str(e)}


def _get_win_rate_tier(win_rate: float) -> str:
    """승률에 따른 티어 반환"""
    if win_rate >= 60:
        return "S (최상위)"
    elif win_rate >= 55:
        return "A (상위)"
    elif win_rate >= 50:
        return "B (평균)"
    elif win_rate >= 45:
        return "C (하위)"
    return "D (최하위)"


def _analyze_cookie_stats(stats: dict, win_rate: float) -> str:
    """쿠키 스탯 분석"""
    analysis = []

    atk = stats.get("atk", 0)
    hp = stats.get("hp", 0)
    skill_dmg = stats.get("skill_dmg", 0)
    crit_rate = stats.get("crit_rate", 0)

    if atk > 25000:
        analysis.append("높은 공격력으로 딜러 역할에 적합")
    if hp > 80000:
        analysis.append("높은 체력으로 생존력 우수")
    if skill_dmg > 300:
        analysis.append("강력한 스킬 데미지 보유")
    if crit_rate > 20:
        analysis.append("높은 치명타 확률로 버스트 딜 가능")

    if not analysis:
        if win_rate >= 50:
            analysis.append("균형 잡힌 스탯 구성")
        else:
            analysis.append("스탯 강화가 필요합니다")

    return ". ".join(analysis) + "."


def tool_get_cookie_win_rate(cookie_id: str) -> dict:
    """
    특정 쿠키의 현재 스탯을 기반으로 승률을 예측합니다.
    cookie_stats.csv 데이터와 win_rate_model을 사용합니다.
    """
    if st.COOKIE_STATS_DF is None:
        return {"status": "FAILED", "error": "쿠키 통계 데이터가 로드되지 않았습니다."}

    # 쿠키 ID로 검색
    cookie = st.COOKIE_STATS_DF[st.COOKIE_STATS_DF["cookie_id"] == cookie_id]
    if cookie.empty:
        # 이름으로 검색 시도
        cookie = st.COOKIE_STATS_DF[st.COOKIE_STATS_DF["cookie_name"].str.contains(cookie_id, na=False)]

    if cookie.empty:
        return {"status": "FAILED", "error": f"쿠키 '{cookie_id}'를 찾을 수 없습니다."}

    row = cookie.iloc[0]

    # 실제 승률 (데이터에 있는 경우)
    actual_win_rate = safe_float(row.get("win_rate_pvp", 50))

    # 모델 예측
    try:
        from ml.win_rate_model import get_predictor, STAT_FEATURES

        predictor = get_predictor()

        if predictor.is_fitted:
            stats = {f: safe_float(row.get(f, 0)) for f in STAT_FEATURES}
            predicted_win_rate = predictor.predict(stats)
        else:
            predicted_win_rate = actual_win_rate
    except Exception:
        predicted_win_rate = actual_win_rate

    return {
        "status": "SUCCESS",
        "cookie_id": safe_str(row.get("cookie_id")),
        "cookie_name": safe_str(row.get("cookie_name")),
        "grade": safe_str(row.get("grade")),
        "type": safe_str(row.get("type")),
        "actual_win_rate": round(actual_win_rate, 1),
        "predicted_win_rate": round(predicted_win_rate, 1),
        "tier": _get_win_rate_tier(predicted_win_rate),
        "stats": {
            "atk": safe_int(row.get("atk")),
            "hp": safe_int(row.get("hp")),
            "def": safe_int(row.get("def")),
            "skill_dmg": safe_int(row.get("skill_dmg")),
            "cooldown": safe_float(row.get("cooldown")),
            "crit_rate": safe_int(row.get("crit_rate")),
            "crit_dmg": safe_int(row.get("crit_dmg")),
        },
        "usage_rate": safe_float(row.get("usage_rate")),
        "pick_rate_pvp": safe_float(row.get("pick_rate_pvp")),
    }


def tool_optimize_investment(user_id: str, goal: str = "maximize_win_rate", resources: Optional[dict] = None) -> dict:
    """
    유저의 자원을 분석하여 최적의 쿠키 투자 전략을 제안합니다.
    P-PSO 알고리즘을 사용하여 승률 증가를 최대화하는 투자 조합을 찾습니다.

    Args:
        user_id: 유저 ID (예: U000001)
        goal: 최적화 목표 ('maximize_win_rate', 'maximize_efficiency', 'balanced')
        resources: 보유 자원 (선택사항, 없으면 DB에서 조회)
            - exp_jelly: 경험치 젤리
            - coin: 코인
            - skill_powder: 스킬 파우더
            - soul_stone: 소울스톤

    Returns:
        투자 추천 목록, 예상 승률 증가, 필요 자원
    """
    if not st.INVESTMENT_OPTIMIZER_AVAILABLE:
        return {"status": "FAILED", "error": "투자 최적화 모듈이 활성화되지 않았습니다."}

    try:
        from ml.investment_optimizer import InvestmentOptimizer

        optimizer = InvestmentOptimizer(user_id, resources=resources)
        result = optimizer.optimize(goal=goal, max_iterations=200, top_n=10)

        if "error" in result:
            return {"status": "FAILED", "error": result["error"]}

        # 응답 포맷팅
        recommendations = []
        for rec in result.get("recommendations", []):
            recommendations.append({
                "cookie_id": rec.get("cookie_id"),
                "cookie_name": rec.get("cookie_name"),
                "grade": rec.get("grade"),
                "upgrade_type": rec.get("upgrade_type"),
                "from_level": rec.get("from_level"),
                "to_level": rec.get("to_level"),
                "win_rate_before": rec.get("win_rate_before"),
                "win_rate_after": rec.get("win_rate_after"),
                "win_rate_gain": rec.get("win_rate_gain"),
                "cost": rec.get("cost"),
                "efficiency": rec.get("efficiency"),
            })

        return {
            "status": "SUCCESS",
            "user_id": user_id,
            "goal": goal,
            "optimization_method": "P-PSO (Phasor Particle Swarm Optimization)",
            "recommendations": recommendations,
            "summary": {
                "total_recommendations": len(recommendations),
                "total_win_rate_gain": result.get("total_win_rate_gain", 0),
                "total_cost": result.get("total_cost", {}),
            },
            "resource_usage": result.get("resource_usage", {}),
            "insight": f"총 {len(recommendations)}개의 투자를 통해 예상 승률 +{result.get('total_win_rate_gain', 0)}% 증가가 가능합니다.",
        }

    except ImportError:
        return {"status": "FAILED", "error": "investment_optimizer 모듈을 로드할 수 없습니다."}
    except Exception as e:
        return {"status": "FAILED", "error": str(e)}


# ============================================================
# 도구 레지스트리
# ============================================================
AVAILABLE_TOOLS = {
    # 쿠키 정보
    "get_cookie_info": tool_get_cookie_info,
    "list_cookies": tool_list_cookies,
    "get_cookie_skill": tool_get_cookie_skill,

    # 왕국 정보
    "get_kingdom_info": tool_get_kingdom_info,
    "list_kingdoms": tool_list_kingdoms,

    # 번역
    "translate_text": tool_translate_text,
    "check_translation_quality": tool_check_translation_quality,
    "get_worldview_terms": tool_get_worldview_terms,

    # 유저 분석
    "analyze_user": tool_analyze_user,
    "get_user_segment": tool_get_user_segment,
    "detect_user_anomaly": tool_detect_user_anomaly,
    "get_segment_statistics": tool_get_segment_statistics,
    "get_anomaly_statistics": tool_get_anomaly_statistics,

    # 게임 로그
    "get_event_statistics": tool_get_event_statistics,
    "get_user_activity_report": tool_get_user_activity_report,

    # 텍스트 분류
    "classify_text": tool_classify_text,

    # RAG 검색
    "search_worldview": tool_search_worldview,
    "search_worldview_lightrag": tool_search_worldview_lightrag,

    # 통계/대시보드
    "get_translation_statistics": tool_get_translation_statistics,
    "get_dashboard_summary": tool_get_dashboard_summary,

    # 예측 분석
    "get_churn_prediction": tool_get_churn_prediction,
    "get_revenue_prediction": tool_get_revenue_prediction,
    "get_cohort_analysis": tool_get_cohort_analysis,
    "get_trend_analysis": tool_get_trend_analysis,

    # ML 모델 예측
    "predict_user_churn": tool_predict_user_churn,
    "predict_cookie_win_rate": tool_predict_cookie_win_rate,
    "get_cookie_win_rate": tool_get_cookie_win_rate,
    "optimize_investment": tool_optimize_investment,
}
