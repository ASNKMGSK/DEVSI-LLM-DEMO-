"""
rag/light_rag.py - LightRAG 구현 (경량 GraphRAG)

LightRAG: Simple and Fast Retrieval-Augmented Generation
- 99% 토큰 절감 (vs Microsoft GraphRAG)
- 듀얼 레벨 검색: Low-level (엔티티) + High-level (테마)
- 증분 업데이트 지원
- 단일 API 호출로 지식 그래프 구축

References:
- Paper: https://arxiv.org/abs/2410.05779
- GitHub: https://github.com/HKUDS/LightRAG
"""
import os
import re
import json
import hashlib
import asyncio
import threading
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

from core.utils import safe_str
import state as st

# ============================================================
# Dedicated Event Loop for LightRAG (이벤트 루프 충돌 방지)
# ============================================================
_LIGHTRAG_LOOP: Optional[asyncio.AbstractEventLoop] = None
_LIGHTRAG_THREAD: Optional[threading.Thread] = None
_LIGHTRAG_LOCK = threading.Lock()


def _start_lightrag_event_loop():
    """LightRAG 전용 이벤트 루프 시작 (별도 스레드에서 실행)"""
    global _LIGHTRAG_LOOP
    _LIGHTRAG_LOOP = asyncio.new_event_loop()
    asyncio.set_event_loop(_LIGHTRAG_LOOP)
    _LIGHTRAG_LOOP.run_forever()


def _get_lightrag_loop() -> asyncio.AbstractEventLoop:
    """LightRAG 전용 이벤트 루프 반환 (필요시 생성)"""
    global _LIGHTRAG_LOOP, _LIGHTRAG_THREAD

    with _LIGHTRAG_LOCK:
        if _LIGHTRAG_LOOP is None or not _LIGHTRAG_LOOP.is_running():
            _LIGHTRAG_THREAD = threading.Thread(target=_start_lightrag_event_loop, daemon=True)
            _LIGHTRAG_THREAD.start()
            # 루프가 시작될 때까지 대기
            import time
            for _ in range(50):  # 최대 5초 대기
                if _LIGHTRAG_LOOP is not None and _LIGHTRAG_LOOP.is_running():
                    break
                time.sleep(0.1)

    return _LIGHTRAG_LOOP


def run_in_lightrag_loop(coro) -> Any:
    """LightRAG 전용 이벤트 루프에서 코루틴 실행"""
    _loop_start = time.time()
    loop = _get_lightrag_loop()
    if loop is None:
        raise RuntimeError("LightRAG event loop not available")
    _loop_get_elapsed = (time.time() - _loop_start) * 1000

    future = asyncio.run_coroutine_threadsafe(coro, loop)
    result = future.result(timeout=300)  # 5분 타임아웃
    _total_elapsed = (time.time() - _loop_start) * 1000
    st.logger.debug("LIGHTRAG_LOOP loop_get=%.0fms total=%.0fms", _loop_get_elapsed, _total_elapsed)
    return result

# ============================================================
# LightRAG Import (Optional)
# ============================================================
LIGHTRAG_AVAILABLE = False
LightRAG = None
QueryParam = None
openai_embed = None
gpt_4o_mini_complete = None

try:
    from lightrag import LightRAG as _LightRAG
    from lightrag import QueryParam as _QueryParam
    from lightrag.llm.openai import openai_embed as _openai_embed
    from lightrag.llm.openai import gpt_4o_mini_complete as _gpt_4o_mini_complete
    LightRAG = _LightRAG
    QueryParam = _QueryParam
    openai_embed = _openai_embed
    gpt_4o_mini_complete = _gpt_4o_mini_complete
    LIGHTRAG_AVAILABLE = True
except ImportError:
    pass

# OpenAI for LLM
try:
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None
    AsyncOpenAI = None
    OPENAI_AVAILABLE = False

# ============================================================
# OpenAI 클라이언트 싱글톤 (연결 오버헤드 감소)
# ============================================================
_OPENAI_SYNC_CLIENT: Optional[OpenAI] = None
_OPENAI_ASYNC_CLIENT: Optional[AsyncOpenAI] = None
_OPENAI_CLIENT_LOCK = threading.Lock()


def _get_sync_client() -> Optional[OpenAI]:
    """동기 OpenAI 클라이언트 싱글톤"""
    global _OPENAI_SYNC_CLIENT
    if not OPENAI_AVAILABLE:
        return None

    with _OPENAI_CLIENT_LOCK:
        if _OPENAI_SYNC_CLIENT is None:
            api_key = getattr(st, "OPENAI_API_KEY", None) or os.environ.get("OPENAI_API_KEY")
            if api_key:
                _OPENAI_SYNC_CLIENT = OpenAI(api_key=api_key)
        return _OPENAI_SYNC_CLIENT


def _get_async_client() -> Optional[AsyncOpenAI]:
    """비동기 OpenAI 클라이언트 싱글톤"""
    global _OPENAI_ASYNC_CLIENT
    if not OPENAI_AVAILABLE:
        return None

    with _OPENAI_CLIENT_LOCK:
        if _OPENAI_ASYNC_CLIENT is None:
            api_key = getattr(st, "OPENAI_API_KEY", None) or os.environ.get("OPENAI_API_KEY")
            if api_key:
                _OPENAI_ASYNC_CLIENT = AsyncOpenAI(api_key=api_key)
        return _OPENAI_ASYNC_CLIENT

# ============================================================
# Configuration
# ============================================================
LIGHTRAG_DIR = st.BACKEND_DIR / "lightrag_data"
LIGHTRAG_STATE_FILE = LIGHTRAG_DIR / "lightrag_state.json"

# ============================================================
# Custom LLM Functions for LightRAG
# ============================================================
async def openai_complete_async(
    prompt: str,
    system_prompt: str = None,
    history_messages: List[Dict] = None,
    **kwargs
) -> str:
    """Async OpenAI completion for LightRAG (싱글톤 클라이언트 사용)"""
    client = _get_async_client()
    if client is None:
        raise ValueError("OPENAI_API_KEY not set or OpenAI not available")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    response = await client.chat.completions.create(
        model=kwargs.get("model", "gpt-4o-mini"),
        messages=messages,
        temperature=kwargs.get("temperature", 0.1),
        max_tokens=kwargs.get("max_tokens", 2000),
    )

    return response.choices[0].message.content


def openai_complete_sync(
    prompt: str,
    system_prompt: str = None,
    history_messages: List[Dict] = None,
    **kwargs
) -> str:
    """Sync OpenAI completion for LightRAG (싱글톤 클라이언트 사용)"""
    client = _get_sync_client()
    if client is None:
        raise ValueError("OPENAI_API_KEY not set or OpenAI not available")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=kwargs.get("model", "gpt-4o-mini"),
        messages=messages,
        temperature=kwargs.get("temperature", 0.1),
        max_tokens=kwargs.get("max_tokens", 2000),
    )

    return response.choices[0].message.content


async def openai_embedding_async(texts: List[str], **kwargs) -> List[List[float]]:
    """Async OpenAI embedding for LightRAG (싱글톤 클라이언트 사용)"""
    client = _get_async_client()
    if client is None:
        raise ValueError("OPENAI_API_KEY not set or OpenAI not available")

    response = await client.embeddings.create(
        model=kwargs.get("model", "text-embedding-3-small"),
        input=texts,
    )

    return [item.embedding for item in response.data]


# ============================================================
# LightRAG Store (Global State)
# ============================================================
LIGHTRAG_STORE: Dict[str, Any] = {
    "instance": None,        # LightRAG 인스턴스
    "ready": False,
    "doc_hash": "",
    "docs_count": 0,
    "entities_count": 0,
    "relations_count": 0,
    "last_build_ts": 0,
    "error": "",
}

# ============================================================
# LightRAG 검색 결과 캐싱 (속도 최적화)
# - 쿼리 정규화: 공백/대소문자 통일 → 캐시 히트율 향상
# - TTL: 5분 후 만료 → 데이터 신선도 유지
# ============================================================
_LIGHTRAG_SEARCH_CACHE: Dict[str, Tuple[Dict[str, Any], float]] = {}  # {key: (result, timestamp)}
_LIGHTRAG_CACHE_MAX_SIZE = 100  # 최대 캐시 항목 수
_LIGHTRAG_CACHE_TTL = 300  # 캐시 TTL (초) - 5분


def _normalize_query(query: str) -> str:
    """쿼리 정규화 (캐시 히트율 향상)"""
    q = query.strip().lower()
    # 연속 공백 → 단일 공백
    q = re.sub(r'\s+', ' ', q)
    # 물음표, 느낌표 제거 (동일한 질문 취급)
    q = re.sub(r'[?!？！]+$', '', q)
    return q


def _get_cache_key(query: str, mode: str, top_k: int) -> str:
    """캐시 키 생성 (정규화된 쿼리 사용)"""
    normalized = _normalize_query(query)
    return f"{normalized}|{mode}|{top_k}"


def _get_cached_result(query: str, mode: str, top_k: int) -> Optional[Dict[str, Any]]:
    """캐시에서 검색 결과 조회 (TTL 확인)"""
    key = _get_cache_key(query, mode, top_k)
    cached = _LIGHTRAG_SEARCH_CACHE.get(key)

    if cached is None:
        return None

    result, timestamp = cached

    # TTL 확인
    if time.time() - timestamp > _LIGHTRAG_CACHE_TTL:
        # 만료된 캐시 삭제
        del _LIGHTRAG_SEARCH_CACHE[key]
        st.logger.debug("LIGHTRAG_CACHE_EXPIRED key=%s", key[:50])
        return None

    return result


def _cache_result(query: str, mode: str, top_k: int, result: Dict[str, Any]) -> None:
    """검색 결과를 캐시에 저장 (타임스탬프 포함)"""
    global _LIGHTRAG_SEARCH_CACHE

    # 캐시 크기 제한 + 만료 항목 정리
    if len(_LIGHTRAG_SEARCH_CACHE) >= _LIGHTRAG_CACHE_MAX_SIZE:
        _cleanup_expired_cache()

    # 여전히 가득 차면 가장 오래된 항목 삭제
    if len(_LIGHTRAG_SEARCH_CACHE) >= _LIGHTRAG_CACHE_MAX_SIZE:
        oldest_key = min(_LIGHTRAG_SEARCH_CACHE, key=lambda k: _LIGHTRAG_SEARCH_CACHE[k][1])
        del _LIGHTRAG_SEARCH_CACHE[oldest_key]

    key = _get_cache_key(query, mode, top_k)
    _LIGHTRAG_SEARCH_CACHE[key] = (result, time.time())


def _cleanup_expired_cache() -> int:
    """만료된 캐시 항목 정리"""
    global _LIGHTRAG_SEARCH_CACHE
    now = time.time()
    expired_keys = [k for k, (_, ts) in _LIGHTRAG_SEARCH_CACHE.items() if now - ts > _LIGHTRAG_CACHE_TTL]
    for k in expired_keys:
        del _LIGHTRAG_SEARCH_CACHE[k]
    return len(expired_keys)


def clear_lightrag_cache() -> int:
    """LightRAG 검색 캐시 초기화 (반환: 삭제된 항목 수)"""
    global _LIGHTRAG_SEARCH_CACHE
    count = len(_LIGHTRAG_SEARCH_CACHE)
    _LIGHTRAG_SEARCH_CACHE = {}
    st.logger.info("LIGHTRAG_CACHE_CLEARED count=%d", count)
    return count


# ============================================================
# 엔티티 목록 로드 및 쿼리 매칭 (검색 정확도 향상)
# ============================================================
_ENTITY_LIST: Optional[List[str]] = None  # 캐시된 엔티티 목록
_ENTITY_LIST_LOCK = threading.Lock()


def _load_entity_list() -> List[str]:
    """LightRAG 엔티티 목록 로드 (kv_store_entity_chunks.json에서)"""
    global _ENTITY_LIST

    with _ENTITY_LIST_LOCK:
        if _ENTITY_LIST is not None:
            return _ENTITY_LIST

        entity_file = LIGHTRAG_DIR / "kv_store_entity_chunks.json"
        if not entity_file.exists():
            st.logger.warning("ENTITY_FILE_NOT_FOUND path=%s", entity_file)
            _ENTITY_LIST = []
            return _ENTITY_LIST

        try:
            with open(entity_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 키가 엔티티 이름
            _ENTITY_LIST = list(data.keys())
            st.logger.info("ENTITY_LIST_LOADED count=%d", len(_ENTITY_LIST))
            return _ENTITY_LIST
        except Exception as e:
            st.logger.warning("ENTITY_LIST_LOAD_FAIL err=%s", safe_str(e))
            _ENTITY_LIST = []
            return _ENTITY_LIST


def _extract_entities_from_query(query: str, min_len: int = 2) -> List[str]:
    """
    쿼리에서 LightRAG에 존재하는 엔티티 추출

    Args:
        query: 사용자 질문
        min_len: 최소 엔티티 길이 (너무 짧은 것 필터링)

    Returns:
        매칭된 엔티티 목록 (긴 것 우선)
    """
    entities = _load_entity_list()
    if not entities:
        return []

    matched = []
    query_lower = query.lower()

    for entity in entities:
        if len(entity) < min_len:
            continue
        # 엔티티가 쿼리에 포함되어 있는지 확인
        if entity.lower() in query_lower:
            matched.append(entity)

    # 긴 엔티티 우선 (더 구체적인 것 우선)
    matched.sort(key=lambda x: len(x), reverse=True)

    # 상위 3개만 반환 (너무 많으면 검색 부담)
    return matched[:3]


def reload_entity_list() -> int:
    """엔티티 목록 강제 리로드 (반환: 로드된 엔티티 수)"""
    global _ENTITY_LIST
    with _ENTITY_LIST_LOCK:
        _ENTITY_LIST = None
    entities = _load_entity_list()
    return len(entities)


# ============================================================
# LightRAG Instance Management
# ============================================================
async def get_lightrag_instance_async(force_new: bool = False) -> Optional[Any]:
    """LightRAG 인스턴스 반환 (Lazy initialization) - Async 버전"""
    global LIGHTRAG_STORE

    if not LIGHTRAG_AVAILABLE:
        st.logger.warning("LIGHTRAG_NOT_AVAILABLE - pip install lightrag-hku")
        return None

    if not force_new and LIGHTRAG_STORE.get("instance") is not None:
        return LIGHTRAG_STORE["instance"]

    try:
        # 디렉토리 생성
        LIGHTRAG_DIR.mkdir(parents=True, exist_ok=True)

        # OPENAI_API_KEY 환경변수 설정 (lightrag 내장 함수용)
        api_key = getattr(st, "OPENAI_API_KEY", None) or os.environ.get("OPENAI_API_KEY")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        # LightRAG 인스턴스 생성 (라이브러리 내장 OpenAI 함수 사용)
        rag = LightRAG(
            working_dir=str(LIGHTRAG_DIR),
            llm_model_func=gpt_4o_mini_complete,
            embedding_func=openai_embed,
        )

        # 스토리지 초기화 (필수!)
        await rag.initialize_storages()

        LIGHTRAG_STORE["instance"] = rag
        st.logger.info("LIGHTRAG_INSTANCE_CREATED dir=%s", str(LIGHTRAG_DIR))
        return rag

    except Exception as e:
        st.logger.exception("LIGHTRAG_INIT_FAIL err=%s", safe_str(e))
        LIGHTRAG_STORE["error"] = safe_str(e)
        return None


def get_lightrag_instance(force_new: bool = False) -> Optional[Any]:
    """LightRAG 인스턴스 반환 (동기 버전 - 기존 인스턴스만 반환)"""
    global LIGHTRAG_STORE

    if not LIGHTRAG_AVAILABLE:
        return None

    return LIGHTRAG_STORE.get("instance")


# ============================================================
# Note: EmbeddingFunc is imported from lightrag.base
# ============================================================


# ============================================================
# Document Indexing
# ============================================================
async def lightrag_index_documents(
    documents: List[str],
    doc_names: List[str] = None,
    force_rebuild: bool = False
) -> Dict[str, Any]:
    """
    문서들을 LightRAG에 인덱싱

    Args:
        documents: 문서 텍스트 리스트
        doc_names: 문서 이름 리스트 (선택)
        force_rebuild: 강제 재빌드 여부

    Returns:
        인덱싱 결과
    """
    global LIGHTRAG_STORE

    if not LIGHTRAG_AVAILABLE:
        return {"status": "FAILED", "error": "LightRAG not available"}

    if not documents:
        return {"status": "FAILED", "error": "No documents provided"}

    try:
        # 해시 계산 (변경 감지)
        all_text = "".join(documents)
        new_hash = hashlib.sha256(all_text.encode()).hexdigest()[:16]

        # 기존 인덱스 확인
        if not force_rebuild and LIGHTRAG_STORE.get("doc_hash") == new_hash:
            st.logger.info("LIGHTRAG_INDEX_CACHED hash=%s", new_hash)
            return {
                "status": "SUCCESS",
                "message": "Index already up-to-date",
                "cached": True,
                "doc_hash": new_hash,
            }

        st.logger.info("LIGHTRAG_INDEX_START docs=%d hash=%s", len(documents), new_hash)

        # LightRAG 인스턴스 (async 초기화 포함)
        rag = await get_lightrag_instance_async(force_new=force_rebuild)
        if rag is None:
            return {"status": "FAILED", "error": "Failed to create LightRAG instance"}

        # 문서 인덱싱
        indexed_count = 0
        errors = []

        for i, doc in enumerate(documents):
            try:
                if len(doc.strip()) < 50:  # 너무 짧은 문서 스킵
                    continue

                # 문서명 추가 (컨텍스트용)
                doc_name = doc_names[i] if doc_names and i < len(doc_names) else f"doc_{i}"
                doc_with_context = f"[문서: {doc_name}]\n\n{doc}"

                # LightRAG에 삽입
                await rag.ainsert(doc_with_context)
                indexed_count += 1

                if (i + 1) % 5 == 0:
                    st.logger.info("LIGHTRAG_INDEX_PROGRESS %d/%d", i + 1, len(documents))

            except Exception as e:
                errors.append(f"Doc {i}: {safe_str(e)}")
                st.logger.warning("LIGHTRAG_INDEX_DOC_FAIL doc=%d err=%s", i, safe_str(e))

        # 상태 업데이트
        import time
        LIGHTRAG_STORE.update({
            "instance": rag,
            "ready": True,
            "doc_hash": new_hash,
            "docs_count": indexed_count,
            "last_build_ts": time.time(),
            "error": "",
        })

        # 상태 파일 저장
        _save_lightrag_state()

        st.logger.info("LIGHTRAG_INDEX_DONE indexed=%d errors=%d", indexed_count, len(errors))

        return {
            "status": "SUCCESS",
            "indexed_count": indexed_count,
            "doc_hash": new_hash,
            "errors": errors[:5] if errors else [],  # 최대 5개 에러만
        }

    except Exception as e:
        st.logger.exception("LIGHTRAG_INDEX_FAIL err=%s", safe_str(e))
        LIGHTRAG_STORE["error"] = safe_str(e)
        return {"status": "FAILED", "error": safe_str(e)}


def lightrag_index_documents_sync(
    documents: List[str],
    doc_names: List[str] = None,
    force_rebuild: bool = False
) -> Dict[str, Any]:
    """동기 버전의 문서 인덱싱 (전용 이벤트 루프 사용)"""
    try:
        return run_in_lightrag_loop(lightrag_index_documents(documents, doc_names, force_rebuild))
    except Exception as e:
        st.logger.warning("LIGHTRAG_INDEX_SYNC_FAIL err=%s", safe_str(e))
        return {"status": "FAILED", "error": safe_str(e)}


# ============================================================
# 단순 쿼리 판별 (RAG 스킵 대상)
# ============================================================
_SIMPLE_QUERY_PATTERNS = [
    r'^(안녕|하이|헬로|hi|hello)',  # 인사
    r'^(고마워|감사|thanks|thank you)',  # 감사
    r'^(ㅋ+|ㅎ+|ㅠ+|ㅜ+)$',  # 이모티콘
    r'^(네|응|예|ok|okay|yes|no|아니)$',  # 단답
    r'^.{1,3}$',  # 3글자 이하
]


def _is_simple_query(query: str) -> bool:
    """단순 쿼리 판별 - LightRAG 검색 스킵 대상"""
    q = query.strip().lower()

    # 너무 짧은 쿼리
    if len(q) <= 3:
        return True

    # 패턴 매칭
    for pattern in _SIMPLE_QUERY_PATTERNS:
        if re.match(pattern, q, re.IGNORECASE):
            return True

    return False


# ============================================================
# Dual-Level Retrieval
# ============================================================
async def lightrag_search(
    query: str,
    mode: str = "hybrid",
    top_k: int = None  # None이면 st.LIGHTRAG_CONFIG["top_k"] 사용
) -> Dict[str, Any]:
    """
    LightRAG 검색 (듀얼 레벨)

    Args:
        query: 검색 쿼리
        mode: 검색 모드
            - "naive": 기본 검색 (컨텍스트만)
            - "local": Low-level 검색 (엔티티 중심)
            - "global": High-level 검색 (테마/주제 중심)
            - "hybrid": local + global 조합 (권장)
        top_k: 반환할 결과 수 (None이면 state.LIGHTRAG_CONFIG 사용)

    Returns:
        검색 결과
    """
    global LIGHTRAG_STORE

    # 기본값은 state.py 중앙 설정에서
    if top_k is None:
        top_k = st.LIGHTRAG_CONFIG.get("top_k", 5)

    # ★ 단순 쿼리 스킵 (인사, 단답 등)
    if _is_simple_query(query):
        st.logger.info("LIGHTRAG_SIMPLE_QUERY_SKIP query=%s", query[:30])
        return {
            "status": "SUCCESS",
            "query": query,
            "mode": mode,
            "context": "",
            "context_len": 0,
            "skipped": True,
            "skip_reason": "simple_query",
            "source": "lightrag",
        }

    # ★ 캐시 확인 (속도 최적화)
    cached = _get_cached_result(query, mode, top_k)
    if cached:
        st.logger.info("LIGHTRAG_CACHE_HIT query=%s mode=%s", query[:30], mode)
        return {**cached, "cached": True}

    if not LIGHTRAG_AVAILABLE:
        return {"status": "FAILED", "error": "LightRAG not available"}

    if not LIGHTRAG_STORE.get("ready"):
        return {"status": "FAILED", "error": "LightRAG index not ready"}

    # 인스턴스가 없으면 자동으로 로드 (재시작 후 복원)
    rag = LIGHTRAG_STORE.get("instance")
    if rag is None:
        st.logger.info("LIGHTRAG_INSTANCE_RESTORE attempting to restore instance...")
        rag = await get_lightrag_instance_async(force_new=False)
        if rag is None:
            return {"status": "FAILED", "error": "LightRAG instance not found"}

    try:
        st.logger.info("LIGHTRAG_SEARCH query=%s mode=%s top_k=%d", query[:50], mode, top_k)

        # 검색 파라미터 (속도 최적화)
        # only_need_context=True: LLM 응답 생성 건너뜀 (10초 → 2초)
        _aquery_start = time.time()

        # 단일 모드 검색 (hybrid/naive/local/global)
        param = QueryParam(mode=mode, top_k=top_k, only_need_context=True)
        context = await rag.aquery(query, param=param)

        _aquery_elapsed = (time.time() - _aquery_start) * 1000
        st.logger.info("LIGHTRAG_AQUERY_TIME elapsed=%.0fms", _aquery_elapsed)

        # 컨텍스트 크기 제한 (불필요한 데이터 전송 방지)
        max_chars = st.LIGHTRAG_CONFIG.get("context_max_chars", 6000)
        original_len = len(context) if context else 0
        if context and len(context) > max_chars:
            context = context[:max_chars] + "\n...(truncated)"
            st.logger.info("LIGHTRAG_CONTEXT_TRUNCATED original=%d truncated=%d", original_len, max_chars)

        result = {
            "status": "SUCCESS",
            "query": query,
            "mode": mode,
            "context": context,  # 검색된 컨텍스트 (truncated)
            "context_len": original_len,  # 원본 길이 (디버깅용)
            "truncated": original_len > max_chars,
            "source": "lightrag",
        }

        # ★ 성공한 결과 캐싱
        _cache_result(query, mode, top_k, result)

        return result

    except Exception as e:
        st.logger.exception("LIGHTRAG_SEARCH_FAIL err=%s", safe_str(e))
        return {"status": "FAILED", "error": safe_str(e)}


def lightrag_search_sync(
    query: str,
    mode: str = "hybrid",
    top_k: int = None  # None이면 st.LIGHTRAG_CONFIG["top_k"] 사용
) -> Dict[str, Any]:
    """동기 버전의 검색 (전용 이벤트 루프 사용)"""
    try:
        return run_in_lightrag_loop(lightrag_search(query, mode, top_k))
    except Exception as e:
        st.logger.warning("LIGHTRAG_SEARCH_SYNC_FAIL err=%s", safe_str(e))
        return {"status": "FAILED", "error": safe_str(e)}


# ============================================================
# Multi-Mode Search (Low-level + High-level)
# ============================================================
async def lightrag_search_dual(
    query: str,
    top_k: int = None  # None이면 st.LIGHTRAG_CONFIG["top_k_dual"] 사용
) -> Dict[str, Any]:
    """
    듀얼 레벨 검색 (Low-level + High-level 결과 모두 반환)

    Low-level (local): 구체적인 엔티티 정보
    - "용감한 쿠키의 스킬은?" → 특정 쿠키 엔티티 검색

    High-level (global): 추상적인 테마/개념
    - "쿠키런 세계관의 시대적 배경은?" → 세계관 테마 검색

    Returns:
        {
            "local_result": 엔티티 중심 결과,
            "global_result": 테마 중심 결과,
            "hybrid_result": 통합 결과
        }
    """
    global LIGHTRAG_STORE

    # 기본값은 state.py 중앙 설정에서
    if top_k is None:
        top_k = st.LIGHTRAG_CONFIG.get("top_k_dual", 4)

    if not LIGHTRAG_AVAILABLE:
        return {"status": "FAILED", "error": "LightRAG not available"}

    if not LIGHTRAG_STORE.get("ready"):
        return {"status": "FAILED", "error": "LightRAG index not ready"}

    rag = LIGHTRAG_STORE.get("instance")
    if rag is None:
        return {"status": "FAILED", "error": "LightRAG instance not found"}

    try:
        # 공통 파라미터 (속도 최적화)
        # only_need_context=True: 컨텍스트만 반환 (LLM 응답 생성 건너뜀)
        common_params = {
            "top_k": top_k,
            "only_need_context": True,  # 컨텍스트만 반환 (속도 향상)
            "enable_rerank": False,
        }

        # 병렬로 세 가지 모드 검색
        local_task = rag.aquery(query, param=QueryParam(mode="local", **common_params))
        global_task = rag.aquery(query, param=QueryParam(mode="global", **common_params))
        hybrid_task = rag.aquery(query, param=QueryParam(mode="hybrid", **common_params))

        local_result, global_result, hybrid_result = await asyncio.gather(
            local_task, global_task, hybrid_task,
            return_exceptions=True
        )

        return {
            "status": "SUCCESS",
            "query": query,
            "local_result": local_result if not isinstance(local_result, Exception) else str(local_result),
            "global_result": global_result if not isinstance(global_result, Exception) else str(global_result),
            "hybrid_result": hybrid_result if not isinstance(hybrid_result, Exception) else str(hybrid_result),
            "recommended": "hybrid",
        }

    except Exception as e:
        st.logger.exception("LIGHTRAG_DUAL_SEARCH_FAIL err=%s", safe_str(e))
        return {"status": "FAILED", "error": safe_str(e)}


def lightrag_search_dual_sync(query: str, top_k: int = None) -> Dict[str, Any]:
    """동기 버전의 듀얼 레벨 검색 (전용 이벤트 루프 사용)"""
    try:
        return run_in_lightrag_loop(lightrag_search_dual(query, top_k))
    except Exception as e:
        st.logger.warning("LIGHTRAG_DUAL_SEARCH_SYNC_FAIL err=%s", safe_str(e))
        return {"status": "FAILED", "error": safe_str(e)}


# ============================================================
# RAG Docs Integration
# ============================================================
def build_lightrag_from_rag_docs(force_rebuild: bool = False) -> Dict[str, Any]:
    """
    기존 RAG 문서들(rag_docs/)에서 LightRAG 인덱스 구축
    """
    from pathlib import Path

    rag_docs_dir = Path(st.RAG_DOCS_DIR)
    if not rag_docs_dir.exists():
        return {"status": "FAILED", "error": f"RAG docs directory not found: {rag_docs_dir}"}

    try:
        documents = []
        doc_names = []

        # PDF 파일 처리
        for pdf_file in rag_docs_dir.glob("*.pdf"):
            try:
                text = _extract_text_from_pdf(str(pdf_file))
                if text and len(text) > 100:
                    documents.append(text)
                    doc_names.append(pdf_file.name)
                    st.logger.info("LIGHTRAG_LOAD_PDF file=%s chars=%d", pdf_file.name, len(text))
            except Exception as e:
                st.logger.warning("LIGHTRAG_PDF_FAIL file=%s err=%s", pdf_file.name, safe_str(e))

        # TXT/MD 파일 처리
        for ext in ["*.txt", "*.md"]:
            for text_file in rag_docs_dir.glob(ext):
                try:
                    text = text_file.read_text(encoding="utf-8")
                    if text and len(text) > 100:
                        documents.append(text)
                        doc_names.append(text_file.name)
                except Exception as e:
                    st.logger.warning("LIGHTRAG_TEXT_FAIL file=%s err=%s", text_file.name, safe_str(e))

        if not documents:
            return {"status": "FAILED", "error": "No documents found in RAG docs directory"}

        st.logger.info("LIGHTRAG_DOCS_LOADED count=%d", len(documents))

        # 인덱싱
        return lightrag_index_documents_sync(documents, doc_names, force_rebuild)

    except Exception as e:
        st.logger.exception("LIGHTRAG_BUILD_FAIL err=%s", safe_str(e))
        return {"status": "FAILED", "error": safe_str(e)}


def _extract_text_from_pdf(pdf_path: str) -> str:
    """PDF에서 텍스트 추출 (service.py의 함수 재사용)"""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n".join(text_parts)
    except ImportError:
        # Fallback to pypdf
        try:
            from pypdf import PdfReader
            reader = PdfReader(pdf_path)
            text_parts = []
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
            return "\n".join(text_parts)
        except Exception:
            return ""
    except Exception:
        return ""


# ============================================================
# State Management
# ============================================================
def _save_lightrag_state():
    """LightRAG 상태 저장"""
    try:
        LIGHTRAG_DIR.mkdir(parents=True, exist_ok=True)
        state = {
            "ready": LIGHTRAG_STORE.get("ready", False),
            "doc_hash": LIGHTRAG_STORE.get("doc_hash", ""),
            "docs_count": LIGHTRAG_STORE.get("docs_count", 0),
            "last_build_ts": LIGHTRAG_STORE.get("last_build_ts", 0),
        }
        with open(LIGHTRAG_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.logger.warning("LIGHTRAG_STATE_SAVE_FAIL err=%s", safe_str(e))


def _load_lightrag_state():
    """LightRAG 상태 로드"""
    global LIGHTRAG_STORE
    try:
        if LIGHTRAG_STATE_FILE.exists():
            with open(LIGHTRAG_STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
                LIGHTRAG_STORE.update(state)
                st.logger.info("LIGHTRAG_STATE_LOADED hash=%s", state.get("doc_hash", ""))
    except Exception as e:
        st.logger.warning("LIGHTRAG_STATE_LOAD_FAIL err=%s", safe_str(e))


# ============================================================
# Status & Utils
# ============================================================
def get_lightrag_status() -> Dict[str, Any]:
    """LightRAG 상태 반환"""
    global LIGHTRAG_STORE

    return {
        "available": LIGHTRAG_AVAILABLE,
        "openai_available": OPENAI_AVAILABLE,
        "ready": LIGHTRAG_STORE.get("ready", False),
        "doc_hash": LIGHTRAG_STORE.get("doc_hash", ""),
        "docs_count": LIGHTRAG_STORE.get("docs_count", 0),
        "last_build_ts": LIGHTRAG_STORE.get("last_build_ts", 0),
        "error": LIGHTRAG_STORE.get("error", ""),
        "working_dir": str(LIGHTRAG_DIR),
    }


def clear_lightrag():
    """LightRAG 초기화"""
    global LIGHTRAG_STORE

    import shutil

    try:
        # 디렉토리 삭제
        if LIGHTRAG_DIR.exists():
            shutil.rmtree(LIGHTRAG_DIR)

        # 상태 초기화
        LIGHTRAG_STORE = {
            "instance": None,
            "ready": False,
            "doc_hash": "",
            "docs_count": 0,
            "entities_count": 0,
            "relations_count": 0,
            "last_build_ts": 0,
            "error": "",
        }

        st.logger.info("LIGHTRAG_CLEARED")
        return {"status": "SUCCESS", "message": "LightRAG cleared"}

    except Exception as e:
        st.logger.warning("LIGHTRAG_CLEAR_FAIL err=%s", safe_str(e))
        return {"status": "FAILED", "error": safe_str(e)}


# ============================================================
# Initialize on import
# ============================================================
_load_lightrag_state()
