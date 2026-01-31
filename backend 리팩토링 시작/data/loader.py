"""
쿠키런 AI 플랫폼 - 데이터 로더
==============================
데브시스터즈 기술혁신 프로젝트

CSV 데이터 및 ML 모델 로딩
"""

import os
from pathlib import Path
from typing import Optional
import joblib
import pandas as pd

import state as st


def get_data_path(filename: str) -> Path:
    """데이터 파일 경로 반환"""
    return Path(st.BASE_DIR) / filename


def load_csv_safe(filepath: Path, encoding: str = "utf-8-sig") -> Optional[pd.DataFrame]:
    """안전한 CSV 로딩"""
    if not filepath.exists():
        st.logger.warning(f"CSV 파일 없음: {filepath}")
        return None
    try:
        df = pd.read_csv(filepath, encoding=encoding)
        st.logger.info(f"CSV 로드 완료: {filepath.name} ({len(df)} rows)")
        return df
    except Exception as e:
        st.logger.error(f"CSV 로드 실패: {filepath} - {e}")
        return None


def load_model_safe(filepath: Path):
    """안전한 모델 로딩"""
    if not filepath.exists():
        st.logger.warning(f"모델 파일 없음: {filepath}")
        return None
    try:
        model = joblib.load(filepath)
        st.logger.info(f"모델 로드 완료: {filepath.name}")
        return model
    except Exception as e:
        st.logger.error(f"모델 로드 실패: {filepath} - {e}")
        return None


def load_all_data():
    """모든 데이터 로드"""
    st.logger.info("=" * 50)
    st.logger.info("쿠키런 AI 플랫폼 데이터 로딩 시작")
    st.logger.info("=" * 50)

    # ========================================
    # CSV 데이터 로드
    # ========================================

    # 쿠키 캐릭터 데이터
    st.COOKIES_DF = load_csv_safe(get_data_path("cookies.csv"))

    # 왕국/지역 데이터
    st.KINGDOMS_DF = load_csv_safe(get_data_path("kingdoms.csv"))

    # 스킬 데이터
    st.SKILLS_DF = load_csv_safe(get_data_path("skills.csv"))

    # 번역 데이터
    st.TRANSLATIONS_DF = load_csv_safe(get_data_path("translations.csv"))

    # 유저 데이터
    st.USERS_DF = load_csv_safe(get_data_path("users.csv"))

    # 게임 로그 데이터
    st.GAME_LOGS_DF = load_csv_safe(get_data_path("game_logs.csv"))

    # 유저 분석 데이터
    st.USER_ANALYTICS_DF = load_csv_safe(get_data_path("user_analytics.csv"))

    # 세계관 텍스트 데이터
    st.WORLDVIEW_TEXTS_DF = load_csv_safe(get_data_path("worldview_texts.csv"))

    # 세계관 용어집
    st.WORLDVIEW_TERMS_DF = load_csv_safe(get_data_path("worldview_terms.csv"))

    # ========================================
    # 분석용 추가 데이터 로드
    # ========================================

    # 쿠키별 사용률/인기도 통계
    st.COOKIE_STATS_DF = load_csv_safe(get_data_path("cookie_stats.csv"))

    # 일별 게임 지표
    st.DAILY_METRICS_DF = load_csv_safe(get_data_path("daily_metrics.csv"))

    # 번역 언어별 통계
    st.TRANSLATION_STATS_DF = load_csv_safe(get_data_path("translation_stats.csv"))

    # 이상탐지 상세 데이터
    st.ANOMALY_DETAILS_DF = load_csv_safe(get_data_path("anomaly_details.csv"))

    # 코호트 리텐션 데이터
    st.COHORT_RETENTION_DF = load_csv_safe(get_data_path("cohort_retention.csv"))

    # 유저 일별 활동 데이터
    st.USER_ACTIVITY_DF = load_csv_safe(get_data_path("user_activity.csv"))

    # ========================================
    # ML 모델 로드
    # ========================================

    # 번역 품질 예측 모델
    st.TRANSLATION_MODEL = load_model_safe(get_data_path("model_translation_quality.pkl"))

    # 텍스트 카테고리 분류 모델
    st.TEXT_CATEGORY_MODEL = load_model_safe(get_data_path("model_text_category.pkl"))

    # 유저 세그먼트 모델
    st.USER_SEGMENT_MODEL = load_model_safe(get_data_path("model_user_segment.pkl"))

    # 이상 탐지 모델
    st.ANOMALY_MODEL = load_model_safe(get_data_path("model_anomaly.pkl"))

    # TF-IDF 벡터라이저
    st.TFIDF_VECTORIZER = load_model_safe(get_data_path("tfidf_vectorizer.pkl"))

    # 스케일러
    st.SCALER_CLUSTER = load_model_safe(get_data_path("scaler_cluster.pkl"))

    # ========================================
    # 라벨 인코더 로드
    # ========================================

    st.LE_CATEGORY = load_model_safe(get_data_path("le_category.pkl"))
    st.LE_LANG = load_model_safe(get_data_path("le_lang.pkl"))
    st.LE_QUALITY = load_model_safe(get_data_path("le_quality.pkl"))
    st.LE_TEXT_CATEGORY = load_model_safe(get_data_path("le_text_category.pkl"))

    # ========================================
    # 캐시 구성
    # ========================================
    build_caches()

    # ========================================
    # 시스템 상태 업데이트
    # ========================================
    st.SYSTEM_STATUS["data_loaded"] = True
    st.SYSTEM_STATUS["models_loaded"] = (
        st.TRANSLATION_MODEL is not None or
        st.TEXT_CATEGORY_MODEL is not None or
        st.USER_SEGMENT_MODEL is not None
    )

    st.logger.info("=" * 50)
    st.logger.info("데이터 로딩 완료")
    st.logger.info(f"  [기본 데이터]")
    st.logger.info(f"  - 쿠키: {len(st.COOKIES_DF) if st.COOKIES_DF is not None else 0}개")
    st.logger.info(f"  - 왕국: {len(st.KINGDOMS_DF) if st.KINGDOMS_DF is not None else 0}개")
    st.logger.info(f"  - 번역: {len(st.TRANSLATIONS_DF) if st.TRANSLATIONS_DF is not None else 0}개")
    st.logger.info(f"  - 유저: {len(st.USERS_DF) if st.USERS_DF is not None else 0}명")
    st.logger.info(f"  - 게임 로그: {len(st.GAME_LOGS_DF) if st.GAME_LOGS_DF is not None else 0}건")
    st.logger.info(f"  [분석용 데이터]")
    st.logger.info(f"  - 쿠키 통계: {len(st.COOKIE_STATS_DF) if st.COOKIE_STATS_DF is not None else 0}개")
    st.logger.info(f"  - 일별 지표: {len(st.DAILY_METRICS_DF) if st.DAILY_METRICS_DF is not None else 0}일")
    st.logger.info(f"  - 번역 통계: {len(st.TRANSLATION_STATS_DF) if st.TRANSLATION_STATS_DF is not None else 0}개")
    st.logger.info(f"  - 코호트: {len(st.COHORT_RETENTION_DF) if st.COHORT_RETENTION_DF is not None else 0}개")
    st.logger.info("=" * 50)


def build_caches():
    """캐시 데이터 구성"""
    # 쿠키별 스킬 매핑
    if st.SKILLS_DF is not None and st.COOKIES_DF is not None:
        for _, row in st.SKILLS_DF.iterrows():
            cookie_id = row.get("cookie_id")
            if cookie_id:
                st.COOKIE_SKILL_MAP[cookie_id] = {
                    "skill_name": row.get("skill_name"),
                    "skill_name_en": row.get("skill_name_en"),
                    "desc_kr": row.get("desc_kr"),
                    "desc_en": row.get("desc_en"),
                }
        st.logger.info(f"쿠키 스킬 캐시 구성: {len(st.COOKIE_SKILL_MAP)}개")


def get_data_summary() -> dict:
    """데이터 요약 정보 반환"""
    return {
        "cookies": {
            "count": len(st.COOKIES_DF) if st.COOKIES_DF is not None else 0,
            "loaded": st.COOKIES_DF is not None,
        },
        "kingdoms": {
            "count": len(st.KINGDOMS_DF) if st.KINGDOMS_DF is not None else 0,
            "loaded": st.KINGDOMS_DF is not None,
        },
        "skills": {
            "count": len(st.SKILLS_DF) if st.SKILLS_DF is not None else 0,
            "loaded": st.SKILLS_DF is not None,
        },
        "translations": {
            "count": len(st.TRANSLATIONS_DF) if st.TRANSLATIONS_DF is not None else 0,
            "loaded": st.TRANSLATIONS_DF is not None,
        },
        "users": {
            "count": len(st.USERS_DF) if st.USERS_DF is not None else 0,
            "loaded": st.USERS_DF is not None,
        },
        "game_logs": {
            "count": len(st.GAME_LOGS_DF) if st.GAME_LOGS_DF is not None else 0,
            "loaded": st.GAME_LOGS_DF is not None,
        },
        "user_analytics": {
            "count": len(st.USER_ANALYTICS_DF) if st.USER_ANALYTICS_DF is not None else 0,
            "loaded": st.USER_ANALYTICS_DF is not None,
        },
        "cookie_stats": {
            "count": len(st.COOKIE_STATS_DF) if st.COOKIE_STATS_DF is not None else 0,
            "loaded": st.COOKIE_STATS_DF is not None,
        },
        "daily_metrics": {
            "count": len(st.DAILY_METRICS_DF) if st.DAILY_METRICS_DF is not None else 0,
            "loaded": st.DAILY_METRICS_DF is not None,
        },
        "translation_stats": {
            "count": len(st.TRANSLATION_STATS_DF) if st.TRANSLATION_STATS_DF is not None else 0,
            "loaded": st.TRANSLATION_STATS_DF is not None,
        },
        "cohort_retention": {
            "count": len(st.COHORT_RETENTION_DF) if st.COHORT_RETENTION_DF is not None else 0,
            "loaded": st.COHORT_RETENTION_DF is not None,
        },
        "models": {
            "translation_quality": st.TRANSLATION_MODEL is not None,
            "text_category": st.TEXT_CATEGORY_MODEL is not None,
            "user_segment": st.USER_SEGMENT_MODEL is not None,
            "anomaly": st.ANOMALY_MODEL is not None,
        },
    }


# 기존 함수 호환성을 위한 alias
def init_data_models():
    """데이터 로드 및 모델 초기화 (startup 시 호출)"""
    load_all_data()
