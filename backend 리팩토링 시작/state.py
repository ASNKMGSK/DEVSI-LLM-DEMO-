"""
쿠키런 AI 플랫폼 - 전역 상태 관리
================================
데브시스터즈 기술혁신 프로젝트

모든 공유 가변 상태를 한 곳에서 관리합니다.
"""
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from threading import Lock
from collections import deque

import pandas as pd

# ============================================================
# 경로
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = Path(BASE_DIR)  # Path 객체 (routes.py에서 / 연산자 사용용)
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "backend.log")

# ============================================================
# 로깅
# ============================================================
def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_FILE, encoding="utf-8", delay=True),
        ],
        force=True,
    )
    lg = logging.getLogger("cookierun-ai")
    lg.setLevel(logging.INFO)
    lg.propagate = True
    for uvn in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        ul = logging.getLogger(uvn)
        ul.setLevel(logging.INFO)
        ul.propagate = True
    # pdfminer 경고 숨기기 (PDF 폰트 메타데이터 관련 - 기능에 영향 없음)
    for pdflogger in ("pdfminer", "pdfminer.pdffont", "pdfminer.pdfinterp", "pdfminer.pdfpage"):
        logging.getLogger(pdflogger).setLevel(logging.ERROR)
    lg.info("LOGGER_READY log_file=%s", LOG_FILE)
    return lg

logger = setup_logging()

# ============================================================
# OpenAI 설정
# ============================================================
def _load_api_key() -> str:
    """API 키 로드 (우선순위: 환경변수 > 파일)"""
    # 1. 환경변수에서 먼저 확인
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key

    # 2. 파일에서 읽기 시도
    key_file = os.path.join(BASE_DIR, "openai_api_key.txt")
    if os.path.exists(key_file):
        try:
            with open(key_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # 주석이나 빈 줄 무시
                    if line and not line.startswith("#"):
                        return line
        except Exception:
            pass

    return ""

OPENAI_API_KEY: str = _load_api_key()

# ============================================================
# 사용자 DB (메모리)
# ============================================================
USERS: Dict[str, Dict[str, str]] = {
    "admin": {"password": "admin123", "role": "관리자", "name": "관리자"},
    "user": {"password": "user123", "role": "사용자", "name": "사용자"},
    "translator": {"password": "trans123", "role": "번역가", "name": "번역가"},
    "analyst": {"password": "analyst123", "role": "분석가", "name": "분석가"},
}

# ============================================================
# 쿠키런 데이터프레임
# ============================================================
# 쿠키 캐릭터 데이터
COOKIES_DF: Optional[pd.DataFrame] = None

# 왕국/지역 데이터
KINGDOMS_DF: Optional[pd.DataFrame] = None

# 스킬 데이터
SKILLS_DF: Optional[pd.DataFrame] = None

# 번역 데이터
TRANSLATIONS_DF: Optional[pd.DataFrame] = None

# 유저 데이터
USERS_DF: Optional[pd.DataFrame] = None

# 게임 로그 데이터
GAME_LOGS_DF: Optional[pd.DataFrame] = None

# 유저 분석 데이터
USER_ANALYTICS_DF: Optional[pd.DataFrame] = None

# 세계관 텍스트 데이터
WORLDVIEW_TEXTS_DF: Optional[pd.DataFrame] = None

# 세계관 용어집
WORLDVIEW_TERMS_DF: Optional[pd.DataFrame] = None

# ============================================================
# 분석용 추가 데이터프레임
# ============================================================
# 쿠키별 사용률/인기도 통계
COOKIE_STATS_DF: Optional[pd.DataFrame] = None

# 일별 게임 지표 (DAU, 매출, 세션 등)
DAILY_METRICS_DF: Optional[pd.DataFrame] = None

# 번역 언어별 통계
TRANSLATION_STATS_DF: Optional[pd.DataFrame] = None

# 이상탐지 상세 데이터
ANOMALY_DETAILS_DF: Optional[pd.DataFrame] = None

# 코호트 리텐션 데이터
COHORT_RETENTION_DF: Optional[pd.DataFrame] = None

# 유저 일별 활동 데이터
USER_ACTIVITY_DF: Optional[pd.DataFrame] = None

# ============================================================
# ML 모델
# ============================================================
# 선택된 모델 상태 저장 파일
SELECTED_MODELS_FILE = os.path.join(BASE_DIR, "selected_models.json")

# 현재 선택된 모델 정보 {모델이름: 버전}
SELECTED_MODELS: Dict[str, str] = {}

def save_selected_models() -> bool:
    """선택된 모델 상태를 JSON 파일에 저장"""
    import json
    try:
        with open(SELECTED_MODELS_FILE, "w", encoding="utf-8") as f:
            json.dump(SELECTED_MODELS, f, ensure_ascii=False, indent=2)
        logger.info(f"선택된 모델 상태 저장 완료: {SELECTED_MODELS}")
        return True
    except Exception as e:
        logger.error(f"선택된 모델 상태 저장 실패: {e}")
        return False

def load_selected_models() -> Dict[str, str]:
    """저장된 모델 선택 상태 로드"""
    import json
    global SELECTED_MODELS
    try:
        if os.path.exists(SELECTED_MODELS_FILE):
            with open(SELECTED_MODELS_FILE, "r", encoding="utf-8") as f:
                SELECTED_MODELS = json.load(f)
                logger.info(f"선택된 모델 상태 로드 완료: {SELECTED_MODELS}")
                return SELECTED_MODELS
    except Exception as e:
        logger.warning(f"선택된 모델 상태 로드 실패: {e}")
    return {}

# 번역 품질 예측 모델
TRANSLATION_MODEL: Optional[Any] = None

# 텍스트 카테고리 분류 모델
TEXT_CATEGORY_MODEL: Optional[Any] = None

# 유저 세그먼트 모델 (K-Means)
USER_SEGMENT_MODEL: Optional[Any] = None

# 이상 탐지 모델
ANOMALY_MODEL: Optional[Any] = None

# 이탈 예측 모델
CHURN_MODEL: Optional[Any] = None

# SHAP Explainer (이탈 예측용)
SHAP_EXPLAINER_CHURN: Optional[Any] = None

# 이탈 예측 모델 설정
CHURN_MODEL_CONFIG: Optional[Dict[str, Any]] = None

# TF-IDF 벡터라이저
TFIDF_VECTORIZER: Optional[Any] = None

# 스케일러
SCALER_CLUSTER: Optional[Any] = None

# 투자 최적화 모듈 사용 가능 여부 (win_rate_model + P-PSO)
INVESTMENT_OPTIMIZER_AVAILABLE: bool = False

# ============================================================
# 라벨 인코더
# ============================================================
LE_CATEGORY: Optional[Any] = None      # 텍스트 카테고리
LE_LANG: Optional[Any] = None          # 번역 대상 언어
LE_QUALITY: Optional[Any] = None       # 번역 품질 등급
LE_TEXT_CATEGORY: Optional[Any] = None # 세계관 텍스트 카테고리

# ============================================================
# 캐시
# ============================================================
# 쿠키별 스킬 매핑
COOKIE_SKILL_MAP: Dict[str, Dict[str, Any]] = {}

# 유저별 분석 데이터 캐시
USER_CACHE: Dict[str, Dict[str, Any]] = {}

# ============================================================
# 최근 컨텍스트 저장 (요약 재활용)
# ============================================================
LAST_CONTEXT_STORE: Dict[str, Dict[str, Any]] = {}
LAST_CONTEXT_LOCK = Lock()
LAST_CONTEXT_TTL_SEC = 600

# ============================================================
# RAG 설정/상태
# ============================================================
RAG_DOCS_DIR = os.path.join(BASE_DIR, "rag_docs")
RAG_FAISS_DIR = os.path.join(BASE_DIR, "rag_faiss")
RAG_STATE_FILE = os.path.join(RAG_FAISS_DIR, "rag_state.json")
RAG_EMBED_MODEL = "text-embedding-3-small"  # 비용 효율 (Reranker로 성능 보완)
RAG_ALLOWED_EXTS = {".txt", ".md", ".json", ".csv", ".log", ".pdf"}
RAG_MAX_DOC_CHARS = 200000
RAG_SNIPPET_CHARS = 1200
RAG_DEFAULT_TOPK = 5
RAG_MAX_TOPK = 20

RAG_LOCK = Lock()
RAG_STORE: Dict[str, Any] = {
    "ready": False,
    "hash": "",
    "docs_count": 0,
    "last_build_ts": 0.0,
    "error": "",
    "index": None,
}

# ============================================================
# LightRAG 설정 (중앙 관리)
# ============================================================
# top_k: 검색할 엔티티/관계 수 (클수록 정확하지만 느림, 토큰 증가)
# context_max_chars: 컨텍스트 최대 글자 수 (rate limit 방지)
# 참고: hybrid 모드는 local+global 합쳐서 top_k*2 정도 나옴
LIGHTRAG_CONFIG = {
    "top_k": 10,             # 단일 검색 - 10으로 증가 (검색 정확도 향상)
    "top_k_dual": 1,         # 듀얼 검색 (3개 모드 병렬 실행) - 거의 사용 안 함
    "context_max_chars": 6000,  # 프론트에 전달할 컨텍스트 최대 글자 수 (엔티티 검색 결과 포함)
}

# ============================================================
# 대화 메모리
# ============================================================
CONVERSATION_MEMORY: Dict[str, deque] = {}
MAX_MEMORY_TURNS = 10

def get_memory(session_id: str) -> deque:
    """세션별 대화 메모리 가져오기"""
    if session_id not in CONVERSATION_MEMORY:
        CONVERSATION_MEMORY[session_id] = deque(maxlen=MAX_MEMORY_TURNS * 2)
    return CONVERSATION_MEMORY[session_id]

def clear_memory(session_id: str) -> None:
    """세션 메모리 초기화"""
    if session_id in CONVERSATION_MEMORY:
        CONVERSATION_MEMORY[session_id].clear()

# ============================================================
# 멀티 에이전트 상태
# ============================================================
AGENT_TASKS: Dict[str, Dict[str, Any]] = {}
AGENT_LOCK = Lock()

# ============================================================
# 번역 작업 큐
# ============================================================
TRANSLATION_QUEUE: List[Dict[str, Any]] = []
TRANSLATION_LOCK = Lock()

# ============================================================
# API 설정
# ============================================================
API_RATE_LIMIT = {
    "requests_per_minute": 60,
    "tokens_per_minute": 100000,
    "current_requests": 0,
    "current_tokens": 0,
    "last_reset": 0.0,
}

# ============================================================
# 시스템 상태
# ============================================================
SYSTEM_STATUS = {
    "initialized": False,
    "data_loaded": False,
    "models_loaded": False,
    "rag_ready": False,
    "startup_time": 0.0,
    "last_error": "",
}

# ============================================================
# 시스템 프롬프트 설정 (백엔드 중앙 관리)
# ============================================================
SYSTEM_PROMPT_FILE = os.path.join(BASE_DIR, "system_prompt.json")

# 현재 활성 시스템 프롬프트 (사용자가 수정 가능)
CUSTOM_SYSTEM_PROMPT: Optional[str] = None

def save_system_prompt(prompt: str) -> bool:
    """시스템 프롬프트를 파일에 저장"""
    import json
    global CUSTOM_SYSTEM_PROMPT
    try:
        CUSTOM_SYSTEM_PROMPT = prompt
        with open(SYSTEM_PROMPT_FILE, "w", encoding="utf-8") as f:
            json.dump({"system_prompt": prompt}, f, ensure_ascii=False, indent=2)
        logger.info("시스템 프롬프트 저장 완료")
        return True
    except Exception as e:
        logger.error(f"시스템 프롬프트 저장 실패: {e}")
        return False

def load_system_prompt() -> Optional[str]:
    """저장된 시스템 프롬프트 로드"""
    import json
    global CUSTOM_SYSTEM_PROMPT
    try:
        if os.path.exists(SYSTEM_PROMPT_FILE):
            with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                CUSTOM_SYSTEM_PROMPT = data.get("system_prompt")
                if CUSTOM_SYSTEM_PROMPT:
                    logger.info("시스템 프롬프트 로드 완료")
                return CUSTOM_SYSTEM_PROMPT
    except Exception as e:
        logger.warning(f"시스템 프롬프트 로드 실패: {e}")
    return None

def get_active_system_prompt() -> str:
    """현재 활성 시스템 프롬프트 반환 (커스텀 > 기본값)"""
    from core.constants import DEFAULT_SYSTEM_PROMPT
    if CUSTOM_SYSTEM_PROMPT and CUSTOM_SYSTEM_PROMPT.strip():
        return CUSTOM_SYSTEM_PROMPT
    return DEFAULT_SYSTEM_PROMPT

def reset_system_prompt() -> bool:
    """시스템 프롬프트를 기본값으로 초기화"""
    global CUSTOM_SYSTEM_PROMPT
    try:
        CUSTOM_SYSTEM_PROMPT = None
        if os.path.exists(SYSTEM_PROMPT_FILE):
            os.remove(SYSTEM_PROMPT_FILE)
        logger.info("시스템 프롬프트 초기화 완료")
        return True
    except Exception as e:
        logger.error(f"시스템 프롬프트 초기화 실패: {e}")
        return False

# ============================================================
# LLM 설정 (백엔드 중앙 관리)
# ============================================================
LLM_SETTINGS_FILE = os.path.join(BASE_DIR, "llm_settings.json")

# 기본 LLM 설정값
DEFAULT_LLM_SETTINGS: Dict[str, Any] = {
    "selectedModel": "gpt-4o-mini",
    "customModel": "",
    "temperature": 0.3,
    "topP": 1.0,
    "presencePenalty": 0.0,
    "frequencyPenalty": 0.0,
    "maxTokens": 8000,
    "seed": None,
    "timeoutMs": 30000,
    "retries": 2,
    "stream": True,
}

# 현재 LLM 설정 (None이면 기본값 사용)
CUSTOM_LLM_SETTINGS: Optional[Dict[str, Any]] = None

def save_llm_settings(settings: Dict[str, Any]) -> bool:
    """LLM 설정을 파일에 저장"""
    import json
    global CUSTOM_LLM_SETTINGS
    try:
        # 기본값과 병합
        merged = {**DEFAULT_LLM_SETTINGS, **settings}
        CUSTOM_LLM_SETTINGS = merged
        with open(LLM_SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        logger.info(f"LLM 설정 저장 완료: model={merged.get('selectedModel')}")
        return True
    except Exception as e:
        logger.error(f"LLM 설정 저장 실패: {e}")
        return False

def load_llm_settings() -> Dict[str, Any]:
    """저장된 LLM 설정 로드"""
    import json
    global CUSTOM_LLM_SETTINGS
    try:
        if os.path.exists(LLM_SETTINGS_FILE):
            with open(LLM_SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                CUSTOM_LLM_SETTINGS = {**DEFAULT_LLM_SETTINGS, **data}
                logger.info(f"LLM 설정 로드 완료: model={CUSTOM_LLM_SETTINGS.get('selectedModel')}")
                return CUSTOM_LLM_SETTINGS
    except Exception as e:
        logger.warning(f"LLM 설정 로드 실패: {e}")
    return DEFAULT_LLM_SETTINGS.copy()

def get_active_llm_settings() -> Dict[str, Any]:
    """현재 활성 LLM 설정 반환 (커스텀 > 기본값)"""
    if CUSTOM_LLM_SETTINGS:
        return CUSTOM_LLM_SETTINGS.copy()
    return DEFAULT_LLM_SETTINGS.copy()

def reset_llm_settings() -> bool:
    """LLM 설정을 기본값으로 초기화"""
    global CUSTOM_LLM_SETTINGS
    try:
        CUSTOM_LLM_SETTINGS = None
        if os.path.exists(LLM_SETTINGS_FILE):
            os.remove(LLM_SETTINGS_FILE)
        logger.info("LLM 설정 초기화 완료")
        return True
    except Exception as e:
        logger.error(f"LLM 설정 초기화 실패: {e}")
        return False
