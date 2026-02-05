"""
main.py - 애플리케이션 진입점
FastAPI 앱 생성, 미들웨어, startup 이벤트, 라우터 등록
"""
import os

# OpenMP 충돌 방지 (EasyOCR + numpy/sklearn 등)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import traceback

import numpy as np

# numpy 호환성 패치
if not hasattr(np, "NaN"):
    np.NaN = np.nan  # noqa: N816

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import state as st
from api.routes import router as api_router
from data.loader import init_data_models
from rag.service import rag_build_or_load_index
from rag.light_rag import (
    LIGHTRAG_AVAILABLE,
    LIGHTRAG_STORE,
    run_in_lightrag_loop,
    get_lightrag_instance_async,
    lightrag_search,
)

# ============================================================
# 앱 생성
# ============================================================
app = FastAPI(title="LLM & AI AGENT 핀테크 플랫폼", version="2.0.0")

# ============================================================
# CORS
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 요청/응답 로깅 미들웨어
# ============================================================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    try:
        st.logger.info("REQ %s %s", request.method, request.url.path)
        resp = await call_next(request)
        st.logger.info("RES %s %s %s", request.method, request.url.path, resp.status_code)
        return resp
    except Exception:
        st.logger.exception("UNHANDLED %s %s", request.method, request.url.path)
        raise

# ============================================================
# 전역 예외 핸들러
# ============================================================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    st.logger.exception("EXCEPTION %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={
            "status": "FAILED",
            "error": str(exc),
            "trace": traceback.format_exc(),
            "log_file": st.LOG_FILE,
        },
    )

# ============================================================
# 라우터 등록
# ============================================================
app.include_router(api_router)

# ============================================================
# Startup 이벤트
# ============================================================
@app.on_event("startup")
def on_startup():
    st.logger.info("APP_STARTUP")
    st.logger.info("BASE_DIR=%s", st.BASE_DIR)
    st.logger.info("LOG_FILE=%s", st.LOG_FILE)
    st.logger.info("PID=%s", os.getpid())
    try:
        # 시스템 프롬프트 및 LLM 설정 로드 (백엔드 중앙 관리)
        st.load_system_prompt()
        st.load_llm_settings()

        init_data_models()
        _k = st.OPENAI_API_KEY
        if _k:
            rag_build_or_load_index(api_key=_k, force_rebuild=False)
        else:
            st.logger.info("RAG_SKIP_STARTUP no_env_api_key docs_dir=%s", st.RAG_DOCS_DIR)

        # ============================================================
        # LightRAG 인스턴스 초기화 + 워밍업 (콜드스타트 방지)
        # ============================================================
        if LIGHTRAG_AVAILABLE and LIGHTRAG_STORE.get("ready"):
            st.logger.info("LIGHTRAG_STARTUP_INIT starting...")
            try:
                # 인스턴스 초기화 (그래프/벡터DB 로드)
                rag_instance = run_in_lightrag_loop(get_lightrag_instance_async(force_new=False))
                if rag_instance:
                    st.logger.info("LIGHTRAG_STARTUP_INIT instance loaded")
                    # 워밍업 스킵 (rate limit 방지)
                    # warmup_result = run_in_lightrag_loop(lightrag_search("워밍업", mode="local", top_k=1))
                    st.logger.info("LIGHTRAG_STARTUP_WARMUP skipped (rate limit prevention)")
                else:
                    st.logger.warning("LIGHTRAG_STARTUP_INIT instance is None")
            except Exception as e:
                st.logger.warning("LIGHTRAG_STARTUP_INIT failed: %s", e)
        else:
            st.logger.info("LIGHTRAG_STARTUP_SKIP available=%s ready=%s",
                          LIGHTRAG_AVAILABLE, LIGHTRAG_STORE.get("ready", False))

        # Semantic Router 제거됨 - 키워드 분류만 사용
    except Exception as e:
        st.logger.exception("BOOTSTRAP_FAIL: %s", e)
        raise

# ============================================================
# 직접 실행
# ============================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_config=None,
        access_log=True,
    )
