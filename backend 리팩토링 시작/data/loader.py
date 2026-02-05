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

    # 게임 로그 데이터 (메모리 절약을 위해 3만건만 로드)
    game_logs_path = get_data_path("game_logs.csv")
    if game_logs_path.exists():
        try:
            st.GAME_LOGS_DF = pd.read_csv(game_logs_path, encoding="utf-8-sig", nrows=30000)
            st.logger.info(f"CSV 로드 완료: game_logs.csv ({len(st.GAME_LOGS_DF)} rows, 제한됨)")
        except Exception as e:
            st.logger.error(f"CSV 로드 실패: game_logs.csv - {e}")
            st.GAME_LOGS_DF = None
    else:
        st.logger.warning(f"CSV 파일 없음: {game_logs_path}")
        st.GAME_LOGS_DF = None

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

    # 이탈 예측 모델
    st.CHURN_MODEL = load_model_safe(get_data_path("model_churn.pkl"))

    # SHAP Explainer (이탈 예측용)
    st.SHAP_EXPLAINER_CHURN = load_model_safe(get_data_path("shap_explainer_churn.pkl"))

    # 이탈 예측 모델 설정 (JSON)
    churn_config_path = get_data_path("churn_model_config.json")
    if churn_config_path.exists():
        try:
            import json
            with open(churn_config_path, "r", encoding="utf-8") as f:
                st.CHURN_MODEL_CONFIG = json.load(f)
            st.logger.info(f"이탈 예측 모델 설정 로드 완료: {churn_config_path.name}")
        except Exception as e:
            st.logger.warning(f"이탈 예측 모델 설정 로드 실패: {e}")
            st.CHURN_MODEL_CONFIG = None

    # TF-IDF 벡터라이저
    st.TFIDF_VECTORIZER = load_model_safe(get_data_path("tfidf_vectorizer.pkl"))

    # 스케일러
    st.SCALER_CLUSTER = load_model_safe(get_data_path("scaler_cluster.pkl"))

    # ========================================
    # 투자 최적화 모듈 확인 (LightGBM + P-PSO)
    # ========================================
    # 투자 최적화는 별도 모델 파일 없이 win_rate_model + P-PSO 알고리즘 사용
    try:
        from ml.investment_optimizer import InvestmentOptimizer
        st.INVESTMENT_OPTIMIZER_AVAILABLE = True
        st.logger.info("투자 최적화 모듈 로드 완료")
    except ImportError as e:
        st.INVESTMENT_OPTIMIZER_AVAILABLE = False
        st.logger.warning(f"투자 최적화 모듈 로드 실패: {e}")

    # ========================================
    # 라벨 인코더 로드
    # ========================================

    st.LE_CATEGORY = load_model_safe(get_data_path("le_category.pkl"))
    st.LE_LANG = load_model_safe(get_data_path("le_lang.pkl"))
    st.LE_QUALITY = load_model_safe(get_data_path("le_quality.pkl"))
    st.LE_TEXT_CATEGORY = load_model_safe(get_data_path("le_text_category.pkl"))

    # ========================================
    # 승률 예측 모델 (밸런스 최적화용)
    # ========================================
    try:
        from ml.win_rate_model import get_predictor, train_and_save, MODEL_PATH
        predictor = get_predictor()

        # 모델이 없거나 학습되지 않았으면 자동 학습
        if not predictor.is_fitted and st.COOKIE_STATS_DF is not None:
            st.logger.info("승률 예측 모델 학습 시작...")
            result = train_and_save(st.COOKIE_STATS_DF)
            st.logger.info(f"승률 예측 모델 학습 완료: R2={result['cv_r2_mean']:.3f}")
        elif predictor.is_fitted:
            st.logger.info("승률 예측 모델 로드 완료")
        else:
            st.logger.warning("승률 예측 모델 학습 불가 (cookie_stats.csv 없음)")
    except Exception as e:
        st.logger.warning(f"승률 예측 모델 초기화 실패: {e}")

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
        st.USER_SEGMENT_MODEL is not None or
        st.INVESTMENT_OPTIMIZER_AVAILABLE
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
    st.logger.info(f"  [ML 모델]")
    st.logger.info(f"  - 번역 품질: {'✓' if st.TRANSLATION_MODEL else '✗'}")
    st.logger.info(f"  - 텍스트 분류: {'✓' if st.TEXT_CATEGORY_MODEL else '✗'}")
    st.logger.info(f"  - 유저 세그먼트: {'✓' if st.USER_SEGMENT_MODEL else '✗'}")
    st.logger.info(f"  - 이상 탐지: {'✓' if st.ANOMALY_MODEL else '✗'}")
    st.logger.info(f"  - 이탈 예측: {'✓' if st.CHURN_MODEL else '✗'}")
    st.logger.info(f"  - 투자 최적화: {'✓' if st.INVESTMENT_OPTIMIZER_AVAILABLE else '✗'}")
    st.logger.info("=" * 50)

    # ========================================
    # 저장된 모델 선택 상태 로드 및 MLflow 모델 로드
    # ========================================
    load_selected_mlflow_models()


def load_selected_mlflow_models():
    """
    서버 시작 시 저장된 모델 선택 상태를 읽어서 MLflow 모델을 로드
    관리자가 선택한 모델이 서버 재시작 후에도 유지됨

    - Windows(로컬): MLflow API 정상 사용
    - Linux(Docker): 절대 경로 문제로 artifacts/model.pkl 직접 로드
    """
    import platform
    import yaml

    # 저장된 선택 상태 로드
    selected = st.load_selected_models()

    if not selected:
        st.logger.info("저장된 모델 선택 상태 없음 - 기본 pkl 모델 사용")
        return

    st.logger.info(f"저장된 모델 선택 상태 로드: {selected}")

    # 환경 감지: Windows=로컬, Linux=Docker
    is_local = platform.system() == "Windows"
    st.logger.info(f"환경 감지: {'로컬(Windows)' if is_local else 'Docker(Linux)'}")

    # 모델 이름 → state 변수 매핑
    MODEL_STATE_MAP = {
        "번역품질예측": "TRANSLATION_MODEL",
        "텍스트분류": "TEXT_CATEGORY_MODEL",
        "유저세그먼트": "USER_SEGMENT_MODEL",
        "이상탐지": "ANOMALY_MODEL",
        "이탈예측": "CHURN_MODEL",
        "승률예측": "WIN_RATE_MODEL",
    }

    # MLflow 기본 경로
    ml_mlruns = os.path.join(st.BASE_DIR, "ml", "mlruns")
    if not os.path.exists(ml_mlruns):
        ml_mlruns = os.path.join(st.BASE_DIR, "mlruns")

    if not os.path.exists(ml_mlruns):
        st.logger.warning(f"MLflow 폴더 없음: {ml_mlruns}")
        return

    # 실험 ID (고정값 - 프로젝트에서 단일 실험 사용)
    experiment_id = "660890565547137650"

    # 각 선택된 모델 로드
    for model_name, version in selected.items():
        state_attr = MODEL_STATE_MAP.get(model_name)
        if not state_attr:
            st.logger.warning(f"알 수 없는 모델: {model_name}")
            continue

        loaded_model = None
        load_method = None

        # ========================================
        # 1차 시도: Windows(로컬)에서 MLflow API 사용
        # ========================================
        if is_local:
            try:
                import mlflow

                mlflow.set_tracking_uri(f"file:///{ml_mlruns}")
                model_uri = f"models:/{model_name}/{version}"

                loaded_model = mlflow.pyfunc.load_model(model_uri)
                # MLflow pyfunc 래퍼에서 실제 sklearn 모델 추출
                if hasattr(loaded_model, "_model_impl"):
                    loaded_model = loaded_model._model_impl.python_model
                    if hasattr(loaded_model, "model"):
                        loaded_model = loaded_model.model

                load_method = "MLflow API"
            except Exception as e:
                st.logger.debug(f"MLflow API 실패, fallback 시도: {e}")
                loaded_model = None

        # ========================================
        # 2차 시도 (또는 Docker): joblib 직접 로드
        # ========================================
        if loaded_model is None:
            try:
                # 1. version meta.yaml에서 model_id 조회
                version_meta_path = os.path.join(
                    ml_mlruns, "models", model_name, f"version-{version}", "meta.yaml"
                )

                if not os.path.exists(version_meta_path):
                    st.logger.warning(f"버전 메타 없음: {version_meta_path}")
                    continue

                with open(version_meta_path, "r", encoding="utf-8") as f:
                    version_meta = yaml.safe_load(f)

                model_id = version_meta.get("model_id")
                if not model_id:
                    st.logger.warning(f"model_id 없음: {model_name} v{version}")
                    continue

                # 2. artifacts/model.pkl 직접 로드
                model_pkl_path = os.path.join(
                    ml_mlruns, experiment_id, "models", model_id, "artifacts", "model.pkl"
                )

                if not os.path.exists(model_pkl_path):
                    st.logger.warning(f"모델 파일 없음: {model_pkl_path}")
                    continue

                loaded_model = joblib.load(model_pkl_path)
                load_method = "직접 로드"
            except Exception as e:
                st.logger.warning(f"모델 로드 실패: {model_name} v{version} - {e}")
                continue

        # ========================================
        # 로드 성공 시 state에 저장
        # ========================================
        if loaded_model is not None:
            setattr(st, state_attr, loaded_model)
            st.logger.info(f"[{load_method}] 모델 로드 완료: {model_name} v{version} → st.{state_attr}")


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
            "churn": st.CHURN_MODEL is not None,
            "investment_optimizer": st.INVESTMENT_OPTIMIZER_AVAILABLE,
        },
    }


# 기존 함수 호환성을 위한 alias
def init_data_models():
    """데이터 로드 및 모델 초기화 (startup 시 호출)"""
    # 중복 로딩 방지
    if st.SYSTEM_STATUS.get("data_loaded"):
        st.logger.info("데이터 이미 로드됨 - 스킵")
        return
    load_all_data()
