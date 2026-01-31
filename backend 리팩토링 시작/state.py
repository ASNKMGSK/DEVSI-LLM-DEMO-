"""
쿠키런 AI 플랫폼 - 전역 상태 관리
================================
데브시스터즈 기술혁신 프로젝트

모든 공유 가변 상태를 한 곳에서 관리합니다.
"""
import os
import logging
from typing import Dict, List, Any, Optional
from threading import Lock
from collections import deque

import pandas as pd

# ============================================================
# 경로
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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
# 번역 품질 예측 모델
TRANSLATION_MODEL: Optional[Any] = None

# 텍스트 카테고리 분류 모델
TEXT_CATEGORY_MODEL: Optional[Any] = None

# 유저 세그먼트 모델 (K-Means)
USER_SEGMENT_MODEL: Optional[Any] = None

# 이상 탐지 모델
ANOMALY_MODEL: Optional[Any] = None

# TF-IDF 벡터라이저
TFIDF_VECTORIZER: Optional[Any] = None

# 스케일러
SCALER_CLUSTER: Optional[Any] = None

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
RAG_EMBED_MODEL = "text-embedding-3-small"
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

# GraphRAG 상태
GRAPH_RAG_STORE: Dict[str, Any] = {
    "ready": False,
    "graph": None,
    "communities": {},
    "entity_embeddings": {},
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
