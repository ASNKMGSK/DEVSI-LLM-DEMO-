# 쿠키런 AI 플랫폼

**데브시스터즈 기술혁신 프로젝트 포트폴리오**

FastAPI + Next.js 기반 쿠키런 세계관 AI 서비스 플랫폼

---

## 프로젝트 소개

쿠키런 세계관의 글로벌 서비스를 지원하는 AI 기반 플랫폼입니다. LLM과 머신러닝을 활용하여 번역 품질 향상, 데이터 분석, 의사결정 지원 등 게임 개발 및 운영 전반의 생산성을 높입니다.

### 핵심 가치

```
"기술혁신을 통해 쿠키런이 전 세계 유저에게 사랑받을 수 있도록"
```

---

## 주요 기능

| 기능 | 설명 | 기술 |
|------|------|------|
| **세계관 맞춤 번역** | LLM 기반 번역 + 세계관 용어 일관성 유지 | GPT-4, 용어집 관리 |
| **멀티 에이전트** | 복잡한 분석 요청을 여러 전문 에이전트가 협업 | LangChain, Tool Calling |
| **지식 검색 (RAG)** | 세계관 정보 임베딩 기반 검색 | FAISS, Hybrid Search, GraphRAG |
| **유저 분석** | 세그먼트 분류, 이상 탐지, 이탈 예측 | K-Means, Isolation Forest |
| **번역 품질 평가** | ML 기반 자동 품질 등급 분류 | Random Forest |
| **텍스트 분류** | 게임 텍스트 카테고리 자동 분류 | TF-IDF, 분류 모델 |
| **실시간 대시보드** | DAU/ARPU/리텐션 등 핵심 KPI 모니터링 | 실시간 API 연동 |
| **AI 인사이트** | 데이터 기반 동적 분석 및 추천 | 자동 트렌드 분석 |
| **OCR 연동** | 이미지에서 텍스트 추출 후 RAG 자동 등록 | EasyOCR |

---

## 기술혁신 프로젝트 업무 매핑

공고의 예시 업무와 구현 기능 대응:

| 공고 예시 업무 | 구현 기능 | 상태 |
|--------------|----------|------|
| LLM 기반 쿠키런 세계관 맞춤형 번역 지원 시스템 | `translation/service.py` - 세계관 용어집 기반 LLM 번역 | ✅ |
| 데이터 분석 및 의사결정을 돕는 멀티 에이전트 기반 서비스 | `agent/multi_agent.py` - 4종 전문 에이전트 | ✅ |
| 임베딩 기반 지식 검색 서비스 | `rag/service.py` - FAISS 벡터 검색, Hybrid Search, GraphRAG | ✅ |

---

## 프로젝트 구조

```
쿠키런 AI 플랫폼/
├── backend 리팩토링 시작/     # FastAPI 백엔드
│   ├── main.py               # 앱 진입점 (CORS, 라우터 등록)
│   ├── state.py              # 전역 상태 관리 (DataFrame, 모델 로딩)
│   │
│   ├── api/
│   │   └── routes.py         # 전체 API 엔드포인트 (70+ 라우트)
│   │
│   ├── agent/                # AI 에이전트
│   │   ├── tools.py          # 17개 분석 도구
│   │   ├── multi_agent.py    # 멀티 에이전트 시스템
│   │   ├── runner.py         # 에이전트 실행기
│   │   ├── llm.py            # LLM 통합 (OpenAI)
│   │   ├── intent.py         # 의도 분류
│   │   └── tool_schemas.py   # 도구 스키마
│   │
│   ├── translation/          # 번역 서비스
│   │   └── service.py        # LLM 기반 번역기
│   │
│   ├── rag/                  # 지식 검색
│   │   ├── service.py        # RAG 서비스 (FAISS, BM25, Hybrid)
│   │   └── graph_rag.py      # GraphRAG (지식 그래프)
│   │
│   ├── ml/                   # ML 모델
│   │   ├── train_models.py   # 데이터 생성 & 모델 학습
│   │   ├── helpers.py        # ML 유틸리티
│   │   └── mlruns/           # MLflow 실험 기록
│   │
│   ├── core/                 # 공통 모듈
│   │   ├── constants.py      # 세계관 설정, 용어집
│   │   ├── utils.py          # 유틸리티 함수
│   │   ├── memory.py         # 대화 메모리 관리
│   │   └── parsers.py        # 파싱 도구
│   │
│   ├── data/                 # 데이터 로더
│   │   └── loader.py
│   │
│   ├── rag_docs/             # RAG 문서 저장소
│   ├── rag_faiss/            # FAISS 인덱스 저장소
│   │
│   └── *.csv                 # 데이터 파일들
│
├── nextjs/                   # Next.js 프론트엔드
│   ├── pages/
│   │   ├── _app.js           # 앱 진입점
│   │   ├── index.js          # 메인 페이지
│   │   └── api/agent/
│   │       └── stream.js     # SSE 프록시
│   │
│   ├── components/
│   │   ├── Layout.js         # 레이아웃
│   │   ├── Sidebar.js        # 쿠키런 테마 사이드바
│   │   ├── Topbar.js         # 상단바
│   │   ├── KpiCard.js        # KPI 카드 컴포넌트
│   │   ├── Tabs.js           # 탭 컴포넌트
│   │   ├── SectionHeader.js  # 섹션 헤더
│   │   ├── ToastProvider.js  # 토스트 알림
│   │   ├── EmptyState.js     # 빈 상태
│   │   ├── Skeleton.js       # 로딩 스켈레톤
│   │   │
│   │   └── panels/           # 기능별 패널
│   │       ├── DashboardPanel.js   # 대시보드 (KPI, AI 인사이트)
│   │       ├── AnalysisPanel.js    # 분석 (유저, 코호트, 이상탐지)
│   │       ├── ModelsPanel.js      # ML 모델 (MLflow 연동)
│   │       ├── AgentPanel.js       # AI 에이전트 채팅
│   │       ├── RagPanel.js         # RAG 문서 관리
│   │       ├── UsersPanel.js       # 유저 검색/분석
│   │       ├── SettingsPanel.js    # 설정
│   │       └── LogsPanel.js        # 로그 뷰어
│   │
│   ├── lib/
│   │   └── api.js            # API 호출 유틸리티
│   │
│   └── tailwind.config.js    # 데브시스터즈 브랜드 컬러
│
└── README.md
```

---

## 데이터 & 모델

### 쿠키런 세계관 데이터

| 데이터 | 파일 | 내용 |
|--------|------|------|
| 쿠키 캐릭터 | `cookies.csv` | 15종 쿠키 (ID, 이름, 등급, 타입, 왕국, 스토리) |
| 쿠키 통계 | `cookie_stats.csv` | 사용률, 파워점수, 인기도, PvP 픽률/승률 |
| 왕국/지역 | `kingdoms.csv` | 5개 왕국 정보 |
| 스킬 | `skills.csv` | 쿠키별 스킬 설명 |
| 번역 데이터 | `translations.csv` | 630개 번역쌍 (한→영/일/중) |
| 번역 통계 | `translation_stats.csv` | 언어별 번역 품질, 검수 현황 |
| 유저 데이터 | `users.csv` | 1,000명 유저 (ID, 가입일, 국가, VIP등급) |
| 유저 분석 | `user_analytics.csv` | 세그먼트, 활동 통계, 이상 탐지 결과 |
| 유저 활동 | `user_activity.csv` | 일별 플레이타임, 스테이지 클리어 |
| 게임 로그 | `game_logs.csv` | 50,000건 이벤트 로그 |
| 일간 지표 | `daily_metrics.csv` | DAU, MAU, ARPU, 매출, 세션 통계 |
| 코호트 리텐션 | `cohort_retention.csv` | 주간 코호트별 리텐션 데이터 |
| 이상 유저 | `anomaly_details.csv` | 이상 탐지 상세 정보 |
| 세계관 용어 | `worldview_terms.csv` | 고유 용어 다국어 사전 |
| 세계관 텍스트 | `worldview_texts.csv` | 세계관 설정 텍스트 |

### ML 모델

| 모델 | 파일 | 타입 | 용도 |
|------|------|------|------|
| 번역 품질 분류 | `model_translation_quality.pkl` | Random Forest | 번역 검수 우선순위 |
| 텍스트 카테고리 | `model_text_category.pkl` | TF-IDF + RF | 번역 스타일 가이드 |
| 유저 세그먼트 | `model_user_segment.pkl` | K-Means | 5개 세그먼트 분류 |
| 이상 탐지 | `model_anomaly.pkl` | Isolation Forest | 어뷰징/봇 탐지 |

### 보조 파일

| 파일 | 용도 |
|------|------|
| `tfidf_vectorizer.pkl` | 텍스트 벡터화 |
| `scaler_cluster.pkl` | 세그먼트 스케일러 |
| `le_*.pkl` | 라벨 인코더 (category, lang, quality 등) |

---

## API 엔드포인트

### 헬스 & 인증

```
GET  /api/health             # 서버 상태 확인
POST /api/login              # 로그인 (Basic Auth)
```

### 쿠키 캐릭터

```
GET  /api/cookies                    # 쿠키 목록 (등급/타입 필터, 통계 포함)
GET  /api/cookies/{cookie_id}        # 특정 쿠키 상세
GET  /api/cookies/{cookie_id}/skill  # 쿠키 스킬 정보
```

### 왕국

```
GET  /api/kingdoms               # 왕국 목록
GET  /api/kingdoms/{kingdom_id}  # 특정 왕국 정보
```

### 번역

```
POST /api/translate              # 세계관 맞춤 번역
POST /api/translate/quality      # 번역 품질 평가
GET  /api/translate/terms        # 세계관 용어집
GET  /api/translate/statistics   # 번역 통계
GET  /api/translate/languages    # 지원 언어 목록
```

### 유저 분석

```
GET  /api/users/search                   # 유저 검색 (레이더 차트 데이터 포함)
GET  /api/users/analyze/{user_id}        # 유저 행동 분석
POST /api/users/segment                  # 유저 세그먼트 분류
POST /api/users/anomaly                  # 이상 행동 탐지
GET  /api/users/segments/statistics      # 세그먼트별 통계
GET  /api/users/{user_id}/activity       # 유저 활동 리포트
```

### 대시보드

```
GET  /api/dashboard/summary      # 대시보드 요약 (KPI 카드용)
GET  /api/dashboard/insights     # AI 인사이트 (실시간 데이터 기반)
```

### 분석

```
GET  /api/analysis/anomaly              # 이상탐지 분석
GET  /api/analysis/prediction/churn     # 이탈/매출/참여도 예측
GET  /api/analysis/cohort/retention     # 코호트 리텐션 분석
GET  /api/analysis/trend/kpis           # 트렌드 KPI 분석
GET  /api/analysis/correlation          # 지표 상관관계
GET  /api/stats/summary                 # 통계 요약 (분석 패널용)
```

### 게임 이벤트

```
GET  /api/events/statistics      # 게임 이벤트 통계
```

### 텍스트 분류

```
POST /api/classify/text          # 텍스트 카테고리 분류
```

### RAG (지식 검색)

```
POST /api/rag/search             # 벡터 검색
POST /api/rag/search/hybrid      # 하이브리드 검색 (BM25 + Vector + Reranking)
GET  /api/rag/status             # RAG 상태 조회
POST /api/rag/reload             # 인덱스 재빌드
POST /api/rag/upload             # 문서 업로드
GET  /api/rag/files              # 업로드된 파일 목록
POST /api/rag/delete             # 파일 삭제
```

### GraphRAG (지식 그래프)

```
GET  /api/graphrag/status        # GraphRAG 상태
POST /api/graphrag/build         # 지식 그래프 빌드
POST /api/graphrag/search        # 그래프 기반 검색
POST /api/graphrag/clear         # 그래프 초기화
```

### OCR

```
POST /api/ocr/extract            # 이미지 텍스트 추출 → RAG 저장
GET  /api/ocr/status             # OCR 기능 상태
```

### AI 에이전트

```
POST /api/agent/chat             # 에이전트 대화 (동기)
POST /api/agent/stream           # 스트리밍 응답 (SSE)
POST /api/agent/memory/clear     # 대화 메모리 초기화
GET  /api/tools                  # 사용 가능한 도구 목록
```

### MLflow

```
GET  /api/mlflow/experiments     # MLflow 실험 목록
GET  /api/mlflow/models          # 등록된 모델 목록
```

### 사용자 관리

```
GET  /api/users                  # 사용자 목록 (관리자)
POST /api/users                  # 사용자 추가 (관리자)
```

### 설정 & 내보내기

```
GET  /api/settings/default       # 기본 설정
GET  /api/export/csv             # 번역 데이터 CSV 내보내기
GET  /api/export/excel           # 번역 데이터 Excel 내보내기
```

---

## 번역 시스템

### 세계관 용어 일관성

```python
# 세계관 고유 용어 자동 감지 및 번역 가이드 제공
WORLDVIEW_TERMS = {
    "젤리": {"en": "Jelly", "ja": "ゼリー", "zh": "果冻"},
    "소울잼": {"en": "Soul Jam", "ja": "ソウルジャム", "zh": "灵魂宝石"},
    "쿠키 왕국": {"en": "Cookie Kingdom", "ja": "クッキー王国", "zh": "饼干王国"},
    ...
}
```

### 번역 품질 등급

| 등급 | 점수 | 조치 |
|------|------|------|
| Excellent | 0.9+ | 출시 가능 |
| Good | 0.8+ | 경미한 수정 |
| Acceptable | 0.7+ | 검토 필요 |
| Needs Review | 0.7 미만 | 재번역 필요 |

---

## 멀티 에이전트 시스템

### 에이전트 구성

```
┌─────────────────────────────────────────────────────┐
│                   코디네이터                          │
│              (요청 분석 & 라우팅)                      │
└───────────┬───────────┬───────────┬────────────────┘
            │           │           │
    ┌───────▼───┐ ┌─────▼─────┐ ┌───▼───────┐
    │ 번역 에이전트│ │ 분석 에이전트│ │ 검색 에이전트│
    │  (번역/품질) │ │ (유저/통계) │ │ (세계관/RAG)│
    └───────────┘ └───────────┘ └───────────┘
```

### 자동 라우팅 예시

```
"이 대사 영어로 번역해줘" → 번역 에이전트
"유저 세그먼트 분석해줘" → 분석 에이전트
"용감한 쿠키가 누구야?" → 검색 에이전트 (RAG + 도구)
"DAU 트렌드 분석해줘" → 분석 에이전트 (트렌드 도구)
```

---

## 기술 스택

### Backend
- **FastAPI** - 비동기 웹 프레임워크
- **LangChain** - LLM 통합, Tool Calling
- **OpenAI GPT-4** - 번역, 분석, 대화
- **FAISS** - 벡터 검색
- **BM25** - 키워드 검색 (Hybrid Search)
- **Cross-Encoder** - 리랭킹
- **NetworkX** - 지식 그래프 (GraphRAG)
- **EasyOCR** - 이미지 텍스트 추출
- **scikit-learn** - ML 모델
- **MLflow** - 실험 추적 및 모델 레지스트리
- **pandas** - 데이터 처리

### Frontend
- **Next.js 14** - React 프레임워크
- **Tailwind CSS** - 스타일링 (데브시스터즈 브랜드 컬러)
- **Framer Motion** - 애니메이션
- **Plotly.js** - 데이터 시각화
- **Lucide React** - 아이콘

---

## 실행 방법

### 1. 데이터 및 모델 생성

```bash
cd "backend 리팩토링 시작"
pip install -r requirements.txt
python ml/train_models.py
```

### 2. 백엔드 실행

```bash
cd "backend 리팩토링 시작"
uvicorn main:app --reload --port 8000
```

### 3. 프론트엔드 실행

```bash
cd nextjs
npm install
npm run dev
```

### 접속

- **프론트엔드**: http://localhost:3000
- **백엔드 API**: http://localhost:8000
- **API 문서**: http://localhost:8000/docs

---

## 인증

| 계정 | 비밀번호 | 권한 |
|------|---------|------|
| admin | admin123 | 관리자 |
| user | user123 | 사용자 |
| translator | trans123 | 번역가 |
| analyst | analyst123 | 분석가 |

---

## 프론트엔드 패널 구성

| 패널 | 설명 | 주요 기능 |
|------|------|----------|
| **대시보드** | 핵심 KPI 모니터링 | DAU/ARPU/리텐션 카드, AI 인사이트, 쿠키 통계 |
| **분석** | 심층 데이터 분석 | 유저 세그먼트, 코호트 리텐션, 이상탐지, 예측 분석 |
| **ML 모델** | MLflow 연동 | 실험 목록, 모델 메트릭, 버전 관리 |
| **AI 에이전트** | 대화형 분석 | 스트리밍 채팅, 도구 호출 결과 표시 |
| **RAG** | 문서 관리 | 파일 업로드/삭제, GraphRAG, Hybrid Search |
| **유저** | 유저 검색 | 레이더 차트, 활동 그래프, 세그먼트 정보 |
| **설정** | 앱 설정 | 모델 선택, 파라미터 조정 |
| **로그** | 시스템 로그 | 실시간 로그 뷰어 |

---

## 포트폴리오 하이라이트

### 1. LLM 기반 번역 시스템
- 세계관 용어 자동 감지 및 일관된 번역
- 캐릭터 말투/성격 반영
- ML 기반 품질 자동 평가

### 2. 멀티 에이전트 아키텍처
- 요청 의도 파악 및 자동 라우팅
- 전문 에이전트 협업
- 17개 분석 도구 통합

### 3. End-to-End ML 파이프라인
- 데이터 생성 → 피처 엔지니어링 → 모델 학습 → 서빙
- MLflow 기반 실험 추적
- 모델 버전 관리

### 4. 고급 RAG 시스템
- FAISS 벡터 인덱싱
- Hybrid Search (BM25 + Vector + Cross-Encoder Reranking)
- GraphRAG (LLM 기반 지식 그래프)
- OCR → RAG 자동 연동

### 5. 실시간 대시보드
- 실제 데이터 기반 KPI 모니터링
- AI 인사이트 자동 생성
- 코호트 리텐션/이탈 예측 시각화

---

## 지원자격 대응

| 자격요건 | 구현 내용 |
|---------|----------|
| LLM 활용 서비스 개발 경험 | GPT-4 기반 번역/에이전트 시스템, RAG |
| ML 프로젝트 설계/구현/배포 | 4종 ML 모델 + MLflow 연동 + 실시간 서빙 |
| Python 경험 | FastAPI, scikit-learn, pandas, LangChain |
| 여러 직군과 협업 | 번역가/기획자/분석가 역할별 기능 및 권한 |

---

## 향후 개선 방향

- [ ] Fine-tuned 번역 모델 도입
- [ ] 실시간 번역 품질 피드백 루프
- [ ] A/B 테스트 기반 번역 최적화
- [ ] 대규모 데이터셋 확장
- [ ] Docker 컨테이너화
- [ ] 모델 자동 재학습 파이프라인

---

**버전**: 4.1.0
**업데이트**: 2026-01-31
**Author**: ML Engineer Portfolio for DEVSISTERS
