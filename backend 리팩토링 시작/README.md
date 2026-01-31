# 데브시스터즈 쿠키런 AI 플랫폼 - Backend

FastAPI 기반 멀티 에이전트 게임 데이터 분석 및 번역 플랫폼 (데브시스터즈 기술혁신 프로젝트)

## 목차
- [프로젝트 구조](#프로젝트-구조)
- [주요 기능](#주요-기능)
- [멀티 에이전트 시스템](#멀티-에이전트-시스템)
- [세계관 번역 서비스](#세계관-번역-서비스)
- [게임 데이터 분석](#게임-데이터-분석)
- [Advanced RAG](#advanced-rag-검색-고도화)
- [ML 모델](#ml-모델)
- [MLOps](#mlops-mlflow)
- [시작하기](#시작하기)
- [사용 예시](#사용-예시)
- [FAQ](#faq)

## 주요 기능

- **멀티 에이전트**: 코디네이터, 번역, 분석, 검색 에이전트 조합
- **세계관 번역**: 쿠키런 세계관 용어 일관성을 유지하는 LLM 기반 번역
- **게임 데이터 분석**: 유저 세그먼트, 이상 탐지, 이벤트 통계, 이탈 예측
- **세계관 검색**: 쿠키 캐릭터, 왕국, 스킬 정보 조회
- **Advanced RAG**: Hybrid Search (BM25 + Vector), Reranking, GraphRAG
- **MLOps**: MLflow 실험 추적, 모델 레지스트리, 버전 관리
- **OCR**: 이미지 텍스트 추출 및 RAG 연동

## 프로젝트 구조

```
backend/
├── main.py                 # FastAPI 앱 진입점 (CORS, 미들웨어, startup)
├── state.py                # 전역 상태 관리 (설정, 로깅, 캐시, 모델 참조)
│
├── core/                   # 핵심 유틸리티
│   ├── __init__.py
│   ├── constants.py        # 세계관 데이터, ML Feature columns, 시스템 프롬프트
│   ├── memory.py           # 대화 메모리 관리 (get/append/clear)
│   ├── parsers.py          # 텍스트 파싱 (ID 추출, 쿠키/왕국 검색)
│   └── utils.py            # 유틸리티 (safe_*, json_sanitize, normalize_model_name)
│
├── ml/                     # ML 헬퍼
│   ├── __init__.py
│   ├── helpers.py          # to_numeric_df, build_feature_df, topk_importance
│   ├── mlflow_tracker.py   # MLflow 실험 추적 유틸리티
│   └── train_models.py     # 모델 학습 스크립트
│
├── data/                   # 데이터 로딩
│   ├── __init__.py
│   └── loader.py           # CSV/모델 로드, 캐시 구성, init_data_models()
│
├── translation/            # 번역 서비스 ⭐ NEW
│   ├── __init__.py
│   └── service.py          # 쿠키런 세계관 맞춤 번역기 (LLM 기반)
│
├── rag/                    # RAG 서비스
│   ├── __init__.py
│   ├── service.py          # FAISS 인덱싱, 검색, 파일 관리
│   └── graph_rag.py        # GraphRAG (LLM 기반 엔티티/관계 추출)
│
├── agent/                  # AI 에이전트
│   ├── __init__.py
│   ├── tools.py            # 도구 함수 (번역, 분석, 세계관 검색 등)
│   ├── tool_schemas.py     # LangChain @tool 정의 (Tool Calling용)
│   ├── multi_agent.py      # 멀티 에이전트 시스템 ⭐ NEW
│   ├── intent.py           # 인텐트 감지, 결정적 도구 라우팅
│   ├── llm.py              # LangChain LLM 호출, 메시지 빌더
│   └── runner.py           # Tool Calling 에이전트 실행기
│
├── api/                    # API 라우트
│   ├── __init__.py
│   └── routes.py           # FastAPI 엔드포인트 (APIRouter)
│
├── # 게임 데이터 파일 ──────────────────────
├── cookies.csv             # 쿠키 캐릭터 마스터 데이터
├── kingdoms.csv            # 왕국 정보
├── skills.csv              # 쿠키 스킬 데이터
├── users.csv               # 유저 데이터
├── user_analytics.csv      # 유저 분석 데이터
├── game_logs.csv           # 게임 이벤트 로그
├── translations.csv        # 번역 데이터
├── worldview_terms.csv     # 세계관 고유 용어
├── worldview_texts.csv     # 세계관 텍스트
│
├── # ML 모델 파일 ──────────────────────────
├── model_translation_quality.pkl  # 번역 품질 예측 모델
├── model_user_segment.pkl         # 유저 세그먼트 클러스터링 모델
├── model_anomaly.pkl              # 이상 탐지 모델 (Isolation Forest)
├── model_text_category.pkl        # 텍스트 카테고리 분류 모델
├── scaler_cluster.pkl             # 유저 세그먼트용 스케일러
├── tfidf_vectorizer.pkl           # TF-IDF 벡터라이저
├── le_*.pkl                       # 라벨 인코더 (category, lang, quality 등)
│
├── rag_docs/               # RAG 문서 저장소
├── rag_faiss/              # FAISS 벡터 인덱스
└── logs/                   # 애플리케이션 로그
```

## 모듈 설명

### main.py
- FastAPI 앱 생성 및 설정
- CORS 미들웨어
- 요청/응답 로깅 미들웨어
- 전역 예외 핸들러
- Startup 이벤트 (데이터/모델/RAG 초기화)

### state.py
- 경로 설정 (BASE_DIR, LOG_DIR)
- 로깅 설정
- OpenAI API 키 (환경변수 또는 openai_api_key.txt)
- 사용자 DB (메모리)
- DataFrame 참조 (COOKIES_DF, KINGDOMS_DF, USERS_DF, GAME_LOGS_DF 등)
- 캐시 (COOKIE_SKILL_MAP, USER_CACHE)
- ML 모델 참조 (TRANSLATION_MODEL, USER_SEGMENT_MODEL, ANOMALY_MODEL 등)
- 라벨 인코더 (LE_CATEGORY, LE_LANG, LE_QUALITY 등)
- RAG 설정/상태 (RAG_STORE, GRAPH_RAG_STORE, locks)
- 대화 메모리 (CONVERSATION_MEMORY)
- 컨텍스트 재사용 (LAST_CONTEXT_STORE)

### core/constants.py
- `FEATURE_COLS_REG` - 매출 예측 피처
- `FEATURE_COLS_ANOMALY` - 이상 탐지 피처
- `FEATURE_COLS_CLF` - 성장 분류 피처
- `FEATURE_LABELS` - 피처 한글 라벨
- `ML_MODEL_INFO` - 모델 메타데이터
- `RAG_DOCUMENTS` - 용어 사전
- `DEFAULT_SYSTEM_PROMPT` - LLM 시스템 프롬프트
- 설정값 (MAX_MEMORY_TURNS, DEFAULT_TOPN, SUMMARY_TRIGGERS 등)

### core/memory.py
- `get_user_memory()` - 사용자별 메모리 deque 반환
- `memory_messages()` - 대화 히스토리 리스트 반환
- `append_memory()` - 대화 내용 추가
- `clear_memory()` - 메모리 초기화

### core/parsers.py
- `extract_top_k_from_text()` - "상위 10개" -> 10
- `parse_month_range_from_text()` - 월 범위 파싱
- `extract_merchant_id()` - "M0001" 추출
- `extract_customer_id()` - "C00001" 추출
- `extract_industry_from_text()` - 업종명 추출
- `filter_metrics_by_month_range()` - DataFrame 월 필터링

### core/utils.py
- `safe_str()`, `safe_float()`, `safe_int()` - 안전한 타입 변환
- `json_sanitize()` - JSON 직렬화용 객체 변환
- `format_openai_error()` - OpenAI 에러 포맷팅
- `normalize_model_name()` - 모델명 정규화

### ml/helpers.py
- `to_numeric_df()` - DataFrame을 numeric으로 변환
- `build_feature_df()` - Series를 feature DataFrame으로 변환
- `normalize_importance()` - Importance 정규화
- `topk_importance()` - 상위 k개 중요 피처 반환

### ml/mlflow_tracker.py
MLflow 실험 추적 유틸리티:
- `init_mlflow()` - MLflow 초기화
- `MLflowExperiment` - 컨텍스트 매니저
- `log_params()`, `log_metrics()` - 파라미터/메트릭 로깅
- `log_model_sklearn()` - 모델 로깅 및 레지스트리 등록

### ml/train_models.py
모델 학습 스크립트 (MLflow 추적):
- `train_revenue_model()` - 매출 예측 모델
- `train_anomaly_model()` - 이상 탐지 모델
- `train_growth_model()` - 성장 분류 모델

### data/loader.py
- `load_dataframes()` - CSV 로드, lag/rolling 피처 생성
- `load_models_bundle()` - ML 모델 로드
- `init_data_models()` - 전체 초기화 (startup 시 호출)
- `_ensure_popular_merchants()` - 인기 가맹점 캐시

### rag/service.py
- `rag_build_or_load_index()` - FAISS 인덱스 구축/로드 + BM25 + Knowledge Graph
- `rag_search_local()` - 로컬 문서 검색 (Vector)
- `rag_search_hybrid()` - **Hybrid Search (BM25 + Vector + Reranking)**
- `rag_search_glossary()` - 용어 사전 검색
- `tool_rag_search()` - 통합 RAG 검색
- 파일 관리 (업로드, 삭제, 상태 확인)
- 한글 경로 우회 (`_safe_faiss_save`, `_safe_faiss_load`)

**Advanced RAG Features:**
- `_build_bm25_index()` - BM25 키워드 인덱스 구축
- `_bm25_search()` - BM25 키워드 검색
- `_rerank_results()` - Cross-Encoder 재정렬
- `_reciprocal_rank_fusion()` - BM25 + Vector 점수 융합
- `build_knowledge_graph()` - Knowledge Graph 구축
- `search_knowledge_graph()` - Knowledge Graph 검색

### agent/tools.py
도구 함수:

**세계관 검색**
- `tool_get_cookie_info()` - 쿠키 캐릭터 정보 조회
- `tool_list_cookies()` - 쿠키 목록 (등급/타입 필터)
- `tool_get_cookie_skill()` - 쿠키 스킬 정보
- `tool_get_kingdom_info()` - 왕국 정보 조회
- `tool_list_kingdoms()` - 왕국 목록

**번역**
- `tool_translate_text()` - 세계관 맞춤 번역
- `tool_check_translation_quality()` - 번역 품질 평가
- `tool_get_worldview_terms()` - 세계관 용어집 조회
- `tool_classify_text()` - 텍스트 카테고리 분류

**유저 분석**
- `tool_analyze_user()` - 유저 종합 분석
- `tool_get_user_segment()` - 유저 세그먼트 예측
- `tool_detect_user_anomaly()` - 유저 이상 탐지
- `tool_get_segment_statistics()` - 세그먼트 통계
- `tool_get_anomaly_statistics()` - 이상 탐지 통계
- `tool_get_event_statistics()` - 게임 이벤트 통계
- `tool_get_user_activity_report()` - 유저 활동 리포트

**분석 대시보드**
- `tool_get_dashboard_summary()` - 전체 대시보드 요약
- `tool_get_churn_prediction()` - 이탈 예측
- `tool_get_cohort_analysis()` - 코호트 분석
- `tool_get_trend_analysis()` - 트렌드 분석
- `tool_get_revenue_prediction()` - 수익 예측

### agent/multi_agent.py
멀티 에이전트 시스템:
- `AgentType` - 에이전트 타입 (COORDINATOR, TRANSLATOR, ANALYST, SEARCHER)
- `Agent` - 개별 에이전트 클래스
- `MultiAgentSystem` - 에이전트 조율 시스템
- `route_request()` - 요청을 적절한 에이전트로 라우팅
- `execute_pipeline()` - 순차적 에이전트 파이프라인 실행

### agent/tool_schemas.py
LangChain Tool Calling을 위한 도구 정의:
- `get_cookie_info` - 쿠키 정보 조회
- `list_cookies` - 쿠키 목록
- `get_cookie_skill` - 쿠키 스킬 조회
- `get_kingdom_info` - 왕국 정보 조회
- `translate_text` - 세계관 번역
- `check_translation_quality` - 번역 품질 검사
- `analyze_user` - 유저 분석
- `get_user_segment` - 유저 세그먼트
- `detect_user_anomaly` - 이상 탐지
- `get_dashboard_summary` - 대시보드 요약
- `search_worldview` - 세계관 정보 검색

### agent/intent.py
스트리밍 엔드포인트용 결정적 도구 실행:
- `detect_intent()` - 사용자 입력 인텐트 감지
- `run_deterministic_tools()` - 인텐트 기반 도구 자동 실행
- `set_last_context()` / `get_last_context()` - 컨텍스트 저장/조회
- `can_reuse_last_context()` - 요약 모드 판단

### agent/llm.py
- `build_langchain_messages()` - LangChain 메시지 빌더
- `get_llm()` - ChatOpenAI 인스턴스 생성
- `invoke_with_retry()` - 재시도 로직
- `pick_api_key()` - API 키 선택
- `chunk_text()` - 텍스트 청킹

### agent/runner.py
Tool Calling 방식의 에이전트 실행기:
- `run_agent()` - LLM이 직접 도구를 선택/호출
  1. LLM에 도구 바인딩 (`bind_tools`)
  2. 시스템 프롬프트에 도구 선택 가이드 포함
  3. LLM이 필요한 도구 자동 선택 및 호출
  4. 도구 결과를 바탕으로 최종 응답 생성
  5. 메모리 저장

### translation/service.py
쿠키런 세계관 맞춤 번역 서비스:
- `CookieRunTranslator` - 번역기 클래스
- `translate()` - 텍스트 번역 (세계관 용어 자동 감지)
- `batch_translate()` - 배치 번역
- `get_term_glossary()` - 세계관 용어집 조회
- `_detect_worldview_terms()` - 세계관 용어 감지
- `_evaluate_quality()` - 번역 품질 자동 평가

### api/routes.py
모든 FastAPI 엔드포인트:

**인증**
- `POST /api/login` - 로그인 (메모리 초기화 포함)
- `GET /api/users` - 사용자 목록 (관리자)
- `POST /api/users` - 사용자 생성 (관리자)

**쿠키 캐릭터** ⭐ NEW
- `GET /api/cookies` - 쿠키 목록 (등급/타입 필터)
- `GET /api/cookies/{cookie_id}` - 쿠키 정보
- `GET /api/cookies/{cookie_id}/skill` - 쿠키 스킬 정보

**왕국** ⭐ NEW
- `GET /api/kingdoms` - 왕국 목록
- `GET /api/kingdoms/{kingdom_id}` - 왕국 정보

**번역** ⭐ NEW
- `POST /api/translate` - 세계관 맞춤 번역
- `POST /api/translate/quality` - 번역 품질 검사
- `GET /api/translate/terms` - 세계관 용어집
- `GET /api/translate/statistics` - 번역 통계
- `GET /api/translate/languages` - 지원 언어 목록

**유저 분석** ⭐ NEW
- `GET /api/users/search` - 유저 검색
- `GET /api/users/analyze/{user_id}` - 유저 종합 분석
- `POST /api/users/segment` - 유저 세그먼트 예측
- `POST /api/users/anomaly` - 유저 이상 탐지
- `GET /api/users/segments/statistics` - 세그먼트 통계
- `GET /api/users/{user_id}/activity` - 유저 활동 리포트

**게임 이벤트** ⭐ NEW
- `GET /api/events/statistics` - 이벤트 통계
- `POST /api/classify/text` - 텍스트 카테고리 분류

**대시보드/분석** ⭐ NEW
- `GET /api/dashboard/summary` - 전체 대시보드 요약
- `GET /api/analysis/anomaly` - 이상 탐지 분석
- `GET /api/analysis/prediction/churn` - 이탈 예측
- `GET /api/analysis/cohort/retention` - 코호트 리텐션
- `GET /api/analysis/trend/kpis` - KPI 트렌드
- `GET /api/analysis/correlation` - 상관관계 분석
- `GET /api/stats/summary` - 통계 요약

**RAG**
- `POST /api/rag/upload` - 문서 업로드
- `GET /api/rag/files` - 파일 목록
- `POST /api/rag/delete` - 파일 삭제 (관리자)
- `GET /api/rag/status` - RAG 상태 (Advanced Features 포함)
- `POST /api/rag/reload` - 인덱스 재빌드 (관리자)
- `POST /api/rag/search` - RAG 검색 (기본)
- `POST /api/rag/search/hybrid` - **Hybrid Search (BM25 + Vector + Reranking + KG)**

**OCR**
- `POST /api/ocr/extract` - 이미지에서 텍스트 추출 → RAG 저장 (EasyOCR)
- `GET /api/ocr/status` - OCR 시스템 상태

**GraphRAG**
- `POST /api/graphrag/build` - GraphRAG 지식 그래프 빌드 (LLM 기반)
- `POST /api/graphrag/search` - 그래프 기반 검색
- `GET /api/graphrag/status` - GraphRAG 상태 조회
- `POST /api/graphrag/clear` - GraphRAG 초기화

**에이전트**
- `POST /api/agent/chat` - 일반 요청
- `POST /api/agent/stream` - 스트리밍 응답
- `POST /api/agent/memory/clear` - 메모리 초기화

**Export**
- `GET /api/export/csv` - CSV 다운로드
- `GET /api/export/excel` - Excel 다운로드

**MLflow**
- `GET /api/mlflow/experiments` - MLflow 실험 목록 및 run 정보
- `GET /api/mlflow/models` - Model Registry 모델 목록 (모든 버전 포함)
- `POST /api/mlflow/models/select` - 모델 버전 선택 및 로드

**시스템**
- `GET /api/health` - 헬스체크
- `GET /api/ml/models` - ML 모델 정보

## 멀티 에이전트 시스템

쿠키런 게임 데이터 분석 및 번역을 위한 4개의 전문 에이전트로 구성:

| 에이전트 | 역할 | 주요 도구 |
|----------|------|----------|
| **코디네이터** | 복잡한 요청을 여러 에이전트로 분배 및 조율 | get_dashboard_summary, search_worldview |
| **번역 에이전트** | 쿠키런 세계관에 맞춘 번역 | translate_text, check_translation_quality, get_worldview_terms |
| **분석 에이전트** | 유저 행동 및 게임 데이터 분석 | analyze_user, get_user_segment, detect_user_anomaly, get_event_statistics |
| **검색 에이전트** | 쿠키런 세계관 정보 검색 | get_cookie_info, list_cookies, get_kingdom_info, search_worldview |

### 요청 라우팅
```python
# 요청 키워드에 따라 자동으로 적절한 에이전트 선택
- "번역", "translate" → 번역 에이전트
- "분석", "통계", "유저" → 분석 에이전트
- "쿠키", "왕국", "세계관" → 검색 에이전트
- 기타 복합 요청 → 코디네이터
```

### 파이프라인 실행
순차적으로 여러 에이전트를 호출하고 결과를 컨텍스트로 전달:
```python
pipeline = [
    {"agent": "searcher", "description": "쿠키 정보 검색"},
    {"agent": "translator", "description": "검색 결과 번역"},
]
result = system.execute_pipeline(pipeline)
```

## 세계관 번역 서비스

쿠키런 세계관의 고유 용어와 캐릭터 특성을 반영하는 LLM 기반 번역 시스템:

### 지원 언어
| 코드 | 언어 |
|------|------|
| ko | 한국어 |
| en | English |
| ja | 日本語 |
| zh | 中文 (简体) |
| zh-TW | 繁體中文 |
| th | ไทย |
| id | Bahasa Indonesia |
| de | Deutsch |
| fr | Français |
| es | Español |
| pt | Português |

### 세계관 용어 자동 감지
번역 시 쿠키런 고유 용어를 자동으로 감지하고 일관된 번역 적용:
- `젤리` → Jelly / ゼリー / 果冻
- `오븐` → Oven / オーブン / 烤箱
- `소울잼` → Soul Jam / ソウルジャム / 灵魂宝石
- `쿠키 왕국` → Cookie Kingdom / クッキー王国 / 饼干王国

### 번역 품질 평가
| 등급 | 점수 | 설명 |
|------|------|------|
| excellent | 0.9+ | 즉시 사용 가능 |
| good | 0.8-0.9 | 경미한 수정 필요 |
| acceptable | 0.7-0.8 | 검토 권장 |
| needs_review | 0.7 미만 | 재번역 필요 |

### 번역 API 사용법
```bash
POST /api/translate
{
  "text": "용감한 쿠키가 젤리를 모아 쿠키 왕국을 지킵니다.",
  "target_lang": "en",
  "category": "story",
  "character_name": "용감한 쿠키"
}

# 응답
{
  "translated_text": "Brave Cookie collects Jelly to protect the Cookie Kingdom.",
  "detected_terms": [...],
  "quality": {"score": 0.92, "grade": "excellent"}
}
```

## 게임 데이터 분석

### 유저 세그먼트
KMeans 클러스터링 기반 유저 분류:
| 세그먼트 | 설명 |
|----------|------|
| Whale | 고과금 유저 |
| Regular | 일반 과금 유저 |
| F2P Active | 무과금 활성 유저 |
| F2P Casual | 무과금 캐주얼 유저 |
| Churned | 이탈 위험 유저 |

### 이상 탐지
Isolation Forest 기반 비정상 유저 패턴 감지:
- 비정상적인 이벤트 빈도
- 이상 결제 패턴
- 봇/핵 의심 행동

### 이벤트 통계
게임 내 이벤트 데이터 집계:
- `stage_clear`: 스테이지 클리어
- `gacha_pull`: 가챠 뽑기
- `pvp_battle`: PvP 대전
- `purchase`: 인앱 구매

## Advanced RAG (검색 고도화)

### Hybrid Search (BM25 + Vector)
BM25 키워드 검색과 FAISS 벡터 검색을 결합하여 검색 품질 향상

| 검색 방식 | 특징 | 용도 |
|-----------|------|------|
| BM25 | 키워드 기반 | 정확한 단어 매칭 |
| Vector (FAISS) | 의미 기반 | 유사한 의미 검색 |
| Hybrid (RRF) | BM25 + Vector 융합 | 최적의 검색 결과 |

**Reciprocal Rank Fusion (RRF):**
```
RRF_score = Σ 1/(k + rank)
```

### Cross-Encoder Reranking
검색 결과를 Cross-Encoder 모델로 재정렬하여 관련성 향상

- 모델: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Query-Document 쌍의 관련성 점수 직접 계산
- 초기 검색 결과의 top-k를 재정렬

### Knowledge Graph (Simple)
문서에서 엔티티와 관계를 추출하여 Knowledge Graph 구축 (정규식 기반)

| 기능 | 설명 |
|------|------|
| Entity Extraction | 고유명사, 기술 용어 추출 (정규식) |
| Relation Extraction | 엔티티 간 관계 추출 (패턴) |
| Graph Search | 쿼리 관련 엔티티/관계 검색 |

### GraphRAG (LLM 기반) ⭐ NEW
Microsoft GraphRAG 아키텍처 기반 - **LLM으로 엔티티/관계 추출** + **NetworkX 그래프**

**작동 방식:**
1. RAG 문서 청크에서 LLM으로 엔티티/관계 추출 (GPT-4o-mini)
2. NetworkX로 지식 그래프 구축 (노드: 엔티티, 엣지: 관계)
3. Louvain 알고리즘으로 커뮤니티 탐지 (유사 개념 클러스터링)
4. 쿼리 시 관련 엔티티 + 이웃 노드 탐색으로 검색

**사용 시나리오:**
- 복잡한 도메인 지식 연결 (예: 금융 용어 간 관계)
- 엔티티 중심 검색 (예: "이 회사와 관련된 모든 기술")
- 문서 간 숨겨진 연결 발견

**GraphRAG API 사용법:**
```bash
# 1. 상태 확인
GET /api/graphrag/status
# → graphrag_ready: false이면 빌드 필요

# 2. 빌드 (관리자만, LLM 비용 발생)
POST /api/graphrag/build
{ "maxChunks": 20 }  # 처리할 청크 수 (비용 조절)

# 3. 검색
POST /api/graphrag/search
{
  "query": "금융 규제",
  "topK": 5,
  "includeNeighbors": true  # 이웃 노드 포함 여부
}
# → 관련 엔티티 + 관계 + 커뮤니티 정보 반환

# 4. 초기화
POST /api/graphrag/clear
```

**GraphRAG vs Simple KG:**
| 항목 | Simple KG | GraphRAG |
|------|-----------|----------|
| 추출 방식 | 정규식 | LLM (GPT-4) |
| 정확도 | 낮음 | 높음 |
| 비용 | 무료 | API 비용 발생 |
| 커뮤니티 탐지 | ❌ | ✅ |

### Hybrid Search API

```bash
POST /api/rag/search/hybrid
{
  "query": "가맹점 매출 분석",
  "topK": 5,
  "useReranking": true,
  "useKg": false
}
```

응답:
```json
{
  "status": "SUCCESS",
  "search_method": "hybrid",
  "reranked": true,
  "bm25_available": true,
  "reranker_available": true,
  "kg_available": true,
  "results": [
    {
      "title": "...",
      "content": "...",
      "bm25_score": 0.85,
      "vector_score": 0.72,
      "fusion_score": 0.031,
      "rerank_score": 0.89
    }
  ],
  "kg_entities": [...]
}
```

### 의존성

```bash
pip install rank-bm25          # BM25 키워드 검색
pip install sentence-transformers  # Cross-Encoder Reranking
pip install networkx           # GraphRAG 그래프 라이브러리
```

## ML 모델

| 모델 | 타입 | 목표 | 주요 지표 |
|------|------|------|----------|
| model_translation_quality.pkl | Random Forest Classifier | 번역 품질 등급 예측 | Accuracy: 85%, F1: 0.82 |
| model_user_segment.pkl | KMeans Clustering | 유저 세그먼트 분류 | Silhouette: 0.45 |
| model_anomaly.pkl | Isolation Forest | 비정상 유저 패턴 탐지 | 이상 비율: 5% |
| model_text_category.pkl | SVM | 텍스트 카테고리 분류 | Accuracy: 88% |

## MLOps (MLflow)

### 모델 학습 및 실험 추적

```bash
# 모델 학습 (MLflow 추적)
python -m ml.train_models

# MLflow UI 실행
mlflow ui --port 5000
```

http://localhost:5000 에서 실험 결과 확인 가능

### MLflow 기능
- **실험 추적**: 파라미터, 메트릭 자동 기록
- **모델 레지스트리**: 모델 버전 관리 (v1, v2, v3... 선택 가능)
- **아티팩트 저장**: 모델 파일 및 관련 자료
- **API 연동**: 프론트엔드에서 실험/모델 조회 및 버전 선택

### 모델 타입
| 타입 | 설명 | 예시 |
|------|------|------|
| Registry | `mlflow.sklearn.log_model()` 사용, Model Registry 등록 | revenue, anomaly, growth |
| Artifact | `mlflow.log_artifact()` 사용, run 아티팩트로 저장 | recommendation (SAR) |

### 환경 변수
| 변수 | 기본값 | 설명 |
|------|--------|------|
| MLFLOW_TRACKING_URI | file:./mlruns | MLflow 저장 경로 |
| MLFLOW_EXPERIMENT_NAME | cookierun-ai-platform | 실험 이름 |

## 시작하기

### 의존성 설치

```bash
pip install fastapi uvicorn pandas numpy scikit-learn joblib
pip install langchain langchain-openai langchain-community langchain-text-splitters
pip install faiss-cpu pypdf
pip install openai  # LLM API
pip install mlflow  # MLOps
pip install easyocr  # OCR
pip install rank-bm25  # Hybrid Search (BM25)
pip install sentence-transformers  # Cross-Encoder Reranking
pip install networkx  # GraphRAG (그래프 라이브러리)
```

### 서버 실행

```bash
cd backend
python main.py
# 또는
uvicorn main:app --reload --port 8000
```

### API 문서

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 인증

HTTP Basic Authentication

| 계정 | 비밀번호 | 권한 |
|------|---------|------|
| admin | admin123 | 관리자 |
| user | user123 | 사용자 |
| test | test | 사용자 |

## 설정값 (state.py)

| 설정 | 값 | 설명 |
|------|-----|------|
| RAG_MAX_DOC_CHARS | 200,000 | 문서 최대 문자수 |
| RAG_SNIPPET_CHARS | 1,200 | 검색 결과 스니펫 길이 |
| RAG_DEFAULT_TOPK | 3 | 기본 검색 결과 수 |
| RAG_MAX_TOPK | 10 | 최대 검색 결과 수 |
| MAX_MEMORY_TURNS | 5 | 대화 히스토리 턴 수 |
| LAST_CONTEXT_TTL_SEC | 600 | 컨텍스트 재사용 TTL (10분) |
| DEFAULT_TOPN | 10 | 기본 랭킹 수 |
| MAX_TOPN | 50 | 최대 랭킹 수 |

## 사용 예시

### 세계관 번역
```bash
# 텍스트 번역 (세계관 용어 자동 반영)
curl -X POST http://localhost:8000/api/translate \
  -u admin:admin123 \
  -H "Content-Type: application/json" \
  -d '{
    "text": "용감한 쿠키가 소울잼을 찾아 쿠키 왕국을 모험합니다.",
    "target_lang": "en",
    "category": "story"
  }'

# 번역 품질 검사
curl -X POST http://localhost:8000/api/translate/quality \
  -u admin:admin123 \
  -H "Content-Type: application/json" \
  -d '{"source_text": "원문", "translated_text": "번역문", "target_lang": "en"}'
```

### 쿠키 캐릭터 조회
```bash
# 쿠키 목록 조회 (등급/타입 필터)
curl "http://localhost:8000/api/cookies?grade=에인션트" -u admin:admin123

# 쿠키 상세 정보
curl http://localhost:8000/api/cookies/cookie_001 -u admin:admin123

# 쿠키 스킬 정보
curl http://localhost:8000/api/cookies/cookie_001/skill -u admin:admin123
```

### 유저 분석
```bash
# 유저 종합 분석
curl http://localhost:8000/api/users/analyze/user_001 -u admin:admin123

# 유저 세그먼트 예측
curl -X POST http://localhost:8000/api/users/segment \
  -u admin:admin123 \
  -H "Content-Type: application/json" \
  -d '{"total_events": 500, "stage_clears": 100, "gacha_pulls": 50}'

# 이상 탐지
curl -X POST http://localhost:8000/api/users/anomaly \
  -u admin:admin123 \
  -H "Content-Type: application/json" \
  -d '{"total_events": 10000, "stage_clears": 5000}'
```

### 대시보드 및 분석
```bash
# 전체 대시보드 요약
curl http://localhost:8000/api/dashboard/summary -u admin:admin123

# 이상 탐지 분석
curl http://localhost:8000/api/analysis/anomaly -u admin:admin123

# 이탈 예측
curl http://localhost:8000/api/analysis/prediction/churn -u admin:admin123
```

### AI 에이전트 스트리밍
```bash
curl -X POST http://localhost:8000/api/agent/stream \
  -u admin:admin123 \
  -H "Content-Type: application/json" \
  -d '{"message": "용감한 쿠키에 대해 알려줘"}' \
  --no-buffer
```

### RAG 검색
```bash
# Hybrid Search
curl -X POST http://localhost:8000/api/rag/search/hybrid \
  -u admin:admin123 \
  -H "Content-Type: application/json" \
  -d '{"query": "쿠키런 세계관", "topK": 5, "useReranking": true}'
```

## FAQ

**Q: 번역 시 세계관 용어가 제대로 반영되지 않습니다.**
A: `worldview_terms.csv` 파일에 새로운 용어를 추가하고 서버를 재시작하세요. 또는 `core/constants.py`의 `WORLDVIEW_TERMS`를 직접 수정할 수 있습니다.

**Q: 유저 세그먼트가 정확하지 않습니다.**
A: `model_user_segment.pkl` 모델을 재학습해보세요. `ml/train_models.py`에서 클러스터 수 등의 하이퍼파라미터를 조정할 수 있습니다.

**Q: 멀티 에이전트가 잘못된 에이전트로 라우팅됩니다.**
A: `agent/multi_agent.py`의 `route_request()` 함수에서 키워드 매핑을 수정하거나, 코디네이터 에이전트를 통해 복합 요청을 처리하세요.

**Q: BM25/Simple KG가 "대기중"/"비활성"으로 표시됩니다.**
A: RAG 패널에서 "인덱스 재빌드" 버튼을 클릭하세요.

**Q: GraphRAG 빌드 시 비용은 얼마나 드나요?**
A: 청크 20개 기준 약 $0.05-0.10 (GPT-4o-mini 사용). maxChunks 파라미터로 조절 가능합니다.

**Q: 새로운 언어 지원을 추가하려면?**
A: `core/constants.py`의 `SUPPORTED_LANGUAGES`와 `WORLDVIEW_TERMS`에 새 언어 코드와 번역을 추가하세요.

**Q: OpenAI API 키는 어디에 설정하나요?**
A: 환경변수 `OPENAI_API_KEY` 또는 `state.py`의 `OPENAI_API_KEY`에 설정하세요.

## 로깅

- 로그 파일: `logs/backend.log`
- 로그 레벨: INFO
- 주요 이벤트: `APP_STARTUP`, `AGENT_START`, `RAG_READY`, `DATA_MODELS_READY`

---

**버전**: 4.0.0 (멀티 에이전트 + 세계관 번역 + 게임 데이터 분석)
**최종 업데이트**: 2026-01-31
