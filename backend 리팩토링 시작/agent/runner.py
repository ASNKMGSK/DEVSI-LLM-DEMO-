"""
agent/runner.py - 에이전트 실행 (Tool Calling 방식)
LLM이 직접 도구를 선택하고 호출합니다.
Rule-based 전처리로 필수 도구를 강제 호출합니다.

멀티 에이전트 모드 지원:
- agent_mode="single": 기존 단일 에이전트 + 다중 도구 (기본값)
- agent_mode="multi": LangGraph 기반 멀티 에이전트 시스템
"""
import json
import re
from datetime import datetime
from typing import Dict, Any, List, Set

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from core.constants import DEFAULT_SYSTEM_PROMPT
from core.utils import safe_str, format_openai_error, normalize_model_name, json_sanitize
from core.memory import append_memory, memory_messages
from agent.tool_schemas import ALL_TOOLS
from agent.llm import get_llm, pick_api_key
from agent.intent import detect_intent
from agent.router import classify_and_get_tools, IntentCategory
import state as st

# 멀티 에이전트 지원 확인
try:
    from agent.multi_agent import run_multi_agent, LANGGRAPH_AVAILABLE
except ImportError:
    LANGGRAPH_AVAILABLE = False
    run_multi_agent = None


# RAG 도구 이름 (분석 질문에서 제외)
RAG_TOOL_NAMES = {"search_worldview", "search_worldview_lightrag"}


MAX_TOOL_ITERATIONS = 10  # 무한 루프 방지

# 키워드-도구 매핑 (Rule-based 전처리용)
KEYWORD_TOOL_MAPPING = {
    "detect_user_anomaly": ["이상 탐지", "이상 유저", "비정상 유저", "어뷰징", "이상탐지", "이상징후"],
    "get_segment_statistics": ["세그먼트 통계", "유저 세그먼트", "유저 분포", "세그먼트 분석", "유저 현황"],
    "get_translation_statistics": ["번역 통계", "번역 현황", "번역 품질"],
    "get_dashboard_summary": ["대시보드", "전체 현황", "요약 통계", "유저 활동", "활동 현황", "전체 유저"],
    # ML 모델 예측 도구
    "predict_user_churn": ["이탈 예측", "이탈 확률", "이탈 위험", "이탈률", "churn", "유저 이탈"],
    "get_cookie_win_rate": ["승률 예측", "쿠키 승률", "pvp 승률", "승률 분석", "win rate"],
    "optimize_investment": ["투자 추천", "투자 최적화", "자원 투자", "육성 추천", "어디에 투자", "레벨업 추천"],
    # 분석 도구
    "get_churn_prediction": ["이탈 분석", "이탈 현황", "이탈 통계", "고위험 유저", "이탈 요인"],
    "get_cohort_analysis": ["코호트 분석", "리텐션 분석", "코호트 리텐션", "주간 리텐션", "잔존율"],
    "get_trend_analysis": ["트렌드 분석", "KPI 분석", "지표 분석", "DAU 분석", "상관관계"],
    "get_revenue_prediction": ["매출 예측", "매출 분석", "수익 분석", "ARPU", "ARPPU", "과금 유저"],
}


def extract_user_id(text: str) -> str | None:
    """텍스트에서 유저 ID 추출 (U0001 ~ U000001 형식, 4~6자리 지원)"""
    match = re.search(r'U\d{4,6}', text, re.IGNORECASE)
    return match.group(0).upper() if match else None


def extract_cookie_id(text: str) -> str | None:
    """텍스트에서 쿠키 ID 추출 (CK001 형식)"""
    match = re.search(r'CK\d{3}', text, re.IGNORECASE)
    return match.group(0).upper() if match else None


def extract_days(text: str) -> int | None:
    """텍스트에서 일수 추출 (최근 N일, N일간 등)"""
    patterns = [
        r'최근\s*(\d+)\s*일',
        r'(\d+)\s*일\s*(?:간|동안|기준)',
        r'지난\s*(\d+)\s*일',
        r'(\d+)days?',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def extract_date_range(text: str) -> tuple[str | None, str | None]:
    """텍스트에서 날짜 범위 추출 (YYYY-MM-DD 형식)"""
    # YYYY-MM-DD 패턴
    date_pattern = r'(\d{4}-\d{2}-\d{2})'
    dates = re.findall(date_pattern, text)
    if len(dates) >= 2:
        return dates[0], dates[1]
    elif len(dates) == 1:
        return dates[0], None
    return None, None


def extract_month(text: str) -> str | None:
    """텍스트에서 월 추출 (YYYY-MM 또는 N월 형식)"""
    # YYYY-MM 패턴
    match = re.search(r'(\d{4}-\d{2})', text)
    if match:
        return match.group(1)

    # N월 패턴 (현재 연도 기준)
    match = re.search(r'(\d{1,2})월', text)
    if match:
        month = int(match.group(1))
        if 1 <= month <= 12:
            # 연도 추출 시도
            year_match = re.search(r'(\d{4})년', text)
            if year_match:
                year = int(year_match.group(1))
            else:
                year = datetime.now().year
            return f"{year}-{month:02d}"
    return None


def extract_risk_level(text: str) -> str | None:
    """텍스트에서 위험 등급 추출 (high/medium/low)"""
    text_lower = text.lower()
    if '고위험' in text or 'high' in text_lower:
        return 'high'
    elif '중위험' in text or 'medium' in text_lower:
        return 'medium'
    elif '저위험' in text or 'low' in text_lower:
        return 'low'
    return None


def extract_cohort(text: str) -> str | None:
    """텍스트에서 코호트명 추출 (YYYY-MM WN 형식)"""
    match = re.search(r'(\d{4}-\d{2}\s*W\d)', text, re.IGNORECASE)
    if match:
        return match.group(1).upper().replace(' ', ' ')
    return None


def detect_required_tools(text: str) -> Set[str]:
    """텍스트에서 필수 도구 감지"""
    required = set()
    text_lower = text.lower()

    for tool_name, keywords in KEYWORD_TOOL_MAPPING.items():
        for keyword in keywords:
            if keyword in text_lower or keyword in text:
                required.add(tool_name)
                break

    return required


def execute_tool_by_name(tool_name: str, args: dict) -> dict:
    """도구 이름으로 실행"""
    for t in ALL_TOOLS:
        if t.name == tool_name:
            try:
                return t.invoke(args)
            except Exception as e:
                return {"status": "FAILED", "error": safe_str(e)}
    return {"status": "FAILED", "error": f"도구 '{tool_name}'을 찾을 수 없습니다."}


def run_agent(req, username: str) -> dict:
    """
    에이전트 실행 (모드 선택 가능).

    Args:
        req: 요청 객체 (user_input, model, agent_mode 등)
        username: 사용자 이름

    agent_mode:
        - "single" (기본값): 단일 에이전트 + 다중 도구
        - "multi": LangGraph 기반 멀티 에이전트 (Coordinator → 전문 에이전트)
    """
    # 멀티 에이전트 모드 체크
    agent_mode = getattr(req, "agent_mode", "single")

    if agent_mode == "multi":
        if not LANGGRAPH_AVAILABLE or run_multi_agent is None:
            return {
                "status": "FAILED",
                "response": "멀티 에이전트 모드를 사용하려면 langgraph를 설치하세요: pip install langgraph",
                "tool_calls": [],
                "log_file": st.LOG_FILE,
            }
        return run_multi_agent(req, username)

    # 기존 단일 에이전트 모드
    user_text = safe_str(req.user_input)

    st.logger.info(
        "AGENT_START user=%s model=%s input_len=%s mode=tool_calling",
        username, normalize_model_name(req.model), len(user_text),
    )

    try:
        # ========== LLM Router 패턴: 의도 분류 → 도구 필터링 ==========
        api_key = pick_api_key(req.api_key)
        category, allowed_tool_names = classify_and_get_tools(
            user_text,
            api_key,
            use_llm_fallback=True,  # 키워드 분류 실패 시 LLM 사용
        )

        st.logger.info(
            "ROUTER_RESULT user=%s category=%s tools=%s",
            username, category.value, allowed_tool_names,
        )

        # 카테고리에 해당하는 도구만 필터링
        if allowed_tool_names:
            filtered_tools = [t for t in ALL_TOOLS if t.name in allowed_tool_names]

            # 도구가 없으면 전체 도구 사용 (fallback)
            if not filtered_tools:
                st.logger.warning(
                    "ROUTER_NO_TOOLS category=%s, fallback to ALL_TOOLS",
                    category.value,
                )
                filtered_tools = ALL_TOOLS
            else:
                st.logger.info(
                    "TOOL_FILTER category=%s, tools=%d→%d",
                    category.value, len(ALL_TOOLS), len(filtered_tools),
                )
        else:
            # GENERAL 카테고리: 도구 없이 대화만
            if category == IntentCategory.GENERAL:
                filtered_tools = []
                st.logger.info("ROUTER_GENERAL_MODE no tools bound")
            else:
                filtered_tools = ALL_TOOLS

        # API 키 확인
        if not api_key:
            msg = "처리 오류: OpenAI API Key가 없습니다."
            append_memory(username, user_text, msg)
            return {"status": "FAILED", "response": msg, "tool_calls": [], "log_file": st.LOG_FILE}

        # LLM 생성 및 도구 바인딩
        # 사용자 설정 temperature 사용 (기본값: 0.3)
        user_temperature = req.temperature if req.temperature is not None else 0.3
        llm = get_llm(
            req.model, api_key, req.max_tokens, streaming=False,
            temperature=user_temperature, top_p=req.top_p,
            presence_penalty=req.presence_penalty, frequency_penalty=req.frequency_penalty,
            seed=req.seed, timeout_ms=req.timeout_ms, max_retries=req.retries,
        )

        # 도구가 있으면 바인딩, 없으면 일반 LLM 사용
        # tool_choice는 auto (기본값) - required는 ReAct 에이전트에서 무한 루프 유발
        if filtered_tools:
            llm_with_tools = llm.bind_tools(filtered_tools)
        else:
            llm_with_tools = llm  # GENERAL 카테고리: 도구 없이 대화

        # 시스템 프롬프트 구성 (할루시네이션 방지 규칙을 맨 앞에 배치)
        base_prompt = safe_str(req.system_prompt).strip() or DEFAULT_SYSTEM_PROMPT

        # 시스템 프롬프트 구성
        system_prompt = base_prompt + """

---

## 필수 키워드-도구 매핑 (반드시 준수!)

다음 키워드가 포함되면 **반드시** 해당 도구를 호출하세요:

| 키워드 | 필수 도구 |
|--------|----------|
| "이상 탐지", "이상 유저", "비정상 유저" | `detect_user_anomaly` |
| "세그먼트 통계", "유저 분포" | `get_segment_statistics` |
| "번역 통계", "번역 품질" | `get_translation_statistics` |
| "대시보드", "전체 현황" | `get_dashboard_summary` |
| "이탈 예측", "이탈 확률", "이탈 위험" | `predict_user_churn` |
| "승률 예측", "쿠키 승률", "PvP 승률" | `get_cookie_win_rate` |
| "투자 추천", "투자 최적화", "육성 추천" | `optimize_investment` |
| "코호트 분석", "리텐션 분석", "잔존율" | `get_cohort_analysis` |
| "트렌드 분석", "KPI 분석", "DAU" | `get_trend_analysis` |
| "매출 예측", "ARPU", "과금 유저" | `get_revenue_prediction` |
| "이탈 현황", "이탈 통계", "고위험 유저" | `get_churn_prediction` |

## 병렬 도구 호출 규칙 (매우 중요!)

사용자 요청에 **여러 키워드가 있으면 해당 도구를 모두 동시에 호출**하세요.

### 예시:
| 사용자 질문 | 호출할 도구들 |
|------------|-------------|
| "유저 세그먼트 통계랑 이상 유저 보여줘" | `get_segment_statistics` + `get_anomaly_statistics` (동시 호출) |
| "번역 통계랑 대시보드 요약 보여줘" | `get_translation_statistics` + `get_dashboard_summary` (동시 호출) |
| "U0001 이탈 예측하고 투자 추천해줘" | `predict_user_churn` + `optimize_investment` (동시 호출) |

### 핵심:
- 요청에 포함된 **모든 키워드에 해당하는 도구를 빠짐없이 호출**
- "~하고", "~와", "~그리고" 등 여러 요청은 **병렬 호출**

## 도구 선택 규칙 (반드시 준수)

### 핵심 규칙:
- 쿠키 정보 요청 → `get_cookie_info`, `list_cookies`, `get_cookie_skill`
- 왕국 정보 요청 → `get_kingdom_info`, `list_kingdoms`
- 번역 관련 요청 → `translate_text`, `check_translation_quality`, `get_worldview_terms`
- 유저 분석 요청 → `analyze_user`, `get_user_segment`, `detect_user_anomaly`
- 세계관 지식 검색 → `search_worldview` (RAG 검색)
- **이탈 예측 요청** → `predict_user_churn` (ML 모델 사용)
- **승률 예측 요청** → `get_cookie_win_rate`, `predict_cookie_win_rate` (ML 모델 사용)
- **투자 최적화 요청** → `optimize_investment` (P-PSO 알고리즘 사용)
- **코호트/리텐션 분석** → `get_cohort_analysis` (주간 리텐션율)
- **트렌드/KPI 분석** → `get_trend_analysis` (DAU, ARPU 등)
- **매출 예측** → `get_revenue_prediction` (ARPU, 과금 유저 분포)
- **전체 이탈 현황** → `get_churn_prediction` (고위험/중위험/저위험 통계)

### 예시:
| 사용자 질문 | 올바른 도구 |
|------------|-----------|
| "용감한 쿠키 정보 알려줘" | get_cookie_info(cookie_id="용감한 쿠키") |
| "에픽 등급 쿠키 목록" | list_cookies(grade="에픽") |
| "U0001 유저 분석해줘" | analyze_user(user_id="U0001") |
| "소울잼이 뭐야?" | search_worldview(query="소울잼 정의") |
| "세그먼트별 유저 통계" | get_segment_statistics() |
| "U0001 이탈 확률 예측해줘" | predict_user_churn(user_id="U0001") |
| "용감한 쿠키 승률 어때?" | get_cookie_win_rate(cookie_id="용감한 쿠키") |
| "U000001 투자 추천해줘" | optimize_investment(user_id="U000001") |
| "코호트 리텐션 분석" | get_cohort_analysis() |
| "트렌드 분석 보여줘" | get_trend_analysis() |
| "매출 예측해줘" | get_revenue_prediction() |
| "이탈 현황 분석" | get_churn_prediction() |

## RAG 검색 결과 해석 규칙 (매우 중요!)

`search_worldview` 도구로 검색한 결과를 해석할 때 **반드시 다음 규칙을 따르세요**:

### 핵심 원칙:
1. **RAG 결과 전체를 꼼꼼히 읽으세요** - 원하는 정확한 섹션명이 없어도 관련 정보가 다른 섹션에 있을 수 있습니다
2. **유사한 표현을 찾으세요** - "승급시 대사" → "필드 대사", "레벨업 시 대사", "성 대사" 등 비슷한 의미의 섹션을 확인
3. **숫자 패턴을 주목하세요** - "(1성)", "(2성)", "(3성)" 등의 표시가 있으면 해당 정보입니다
4. **"모르겠다"고 답하기 전에** - RAG 결과에 관련 정보가 정말 없는지 다시 확인하세요

### 예시 (클로티드 크림쿠키 승급 대사):
- 사용자 질문: "승급 시 대사 알려줘"
- RAG 결과에 "승급시 대사" 섹션이 비어있더라도
- "필드 대사" 섹션에 "(1성)", "(2성)" 등의 대사가 있다면 **그것이 답입니다!**
- 예: "승리를 위해서라면 수단을 가리지 않아야 할 때도 가끔 존재합니다."(1성)

### 금지 사항:
- RAG 결과에 정보가 있는데 "모르겠습니다", "기억하지 않습니다"라고 답변하지 마세요
- 섹션명이 정확히 일치하지 않는다고 관련 정보를 무시하지 마세요
- **섹션 번호를 데이터로 착각하지 마세요** - "5. 토핑 프리셋"의 "5"는 섹션 번호이지 토핑 개수가 아닙니다!
- **⚠️ RAG 결과에 없는 정보를 절대 지어내지 마세요!** - 검색 결과에 토핑 목록이 없는데 "크림 터틀, 마법 파우더, 딸기잼 젤리..." 등을 만들어내면 안 됩니다. 이것은 할루시네이션입니다!

### 숫자 질문 답변 규칙:
- "몇 개야?", "몇 명이야?" 등 개수를 묻는 질문에는 **RAG 결과에서 실제로 나열된 항목을 세어서** 답하세요
- 예: "라즈베리, 초코칩, 아몬드, 땅콩, 카라멜, 애플젤리"가 나열되면 → **"6개야"**
- 섹션 번호(1. 2. 3...)를 개수로 착각하지 마세요
- **RAG 결과에 목록이 없으면** → "RAG 문서에서 해당 목록을 찾을 수 없습니다"라고 솔직히 답하세요

## 대화 맥락 유지 규칙 (매우 중요!)

이전 대화에서 **특정 쿠키를 언급했다면**, 후속 질문도 그 쿠키에 대한 것으로 가정하세요.

### 예시:
| 이전 대화 | 현재 질문 | 올바른 해석 |
|----------|----------|------------|
| "클로티드 크림쿠키 승급 대사 알려줘" | "전투 팀 편성 시 대사는?" | **클로티드 크림쿠키**의 전투 팀 편성 시 대사 |
| "용감한 쿠키 정보 알려줘" | "스킬은 뭐야?" | **용감한 쿠키**의 스킬 |
| "커스터드 3세맛 쿠키 대사" | "승리 대사도 알려줘" | **커스터드 3세맛 쿠키**의 승리 대사 |

### 핵심:
- 이전 대화에서 쿠키 이름이 나왔으면 **맥락 유지**
- 새로운 쿠키 이름이 언급될 때까지 **이전 쿠키 기준**으로 답변
- RAG 검색 시에도 이전 쿠키 이름을 쿼리에 **포함**

## RAG 섹션 매칭 규칙 (정확한 대사 분류)

쿠키 대사 요청 시 **정확한 섹션**에서만 정보를 추출하세요:

| 요청 키워드 | 정확한 섹션명 | 주의 |
|------------|-------------|------|
| "전투 팀 편성", "팀 편성" | "전투 팀 편성 시 대사" 또는 "전투 편성 시 대사" | "승리 시 대사"와 혼동 금지 |
| "승리", "이겼을 때" | "전투 승리 시 대사" 또는 "승리 시 대사" | "편성 시 대사"와 혼동 금지 |
| "패배", "졌을 때" | "전투 패배 시 대사" | |
| "승급", "초월" | "승급시 대사", "초월 승급시 대사" | |
| "해금", "획득" | "해금 시 대사" | |
"""

        # 메시지 구성 (이전 대화 기록 포함)
        messages: List = [
            SystemMessage(content=system_prompt),
        ]

        # 이전 대화 기록 추가 (맥락 유지)
        prev_messages = memory_messages(username)
        for msg in prev_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

        # 현재 사용자 입력 추가
        messages.append(HumanMessage(content=user_text))

        tool_calls_log: List[Dict[str, Any]] = []
        iteration = 0

        # ========== Rule-based 전처리: 필수 도구 강제 호출 ==========
        required_tools = detect_required_tools(user_text)
        user_id = extract_user_id(user_text)

        if required_tools:
            st.logger.info(
                "RULE_BASED_PREPROCESS user=%s required_tools=%s user_id=%s",
                username, required_tools, user_id,
            )

            # 필수 도구 강제 실행
            forced_results = []
            cookie_id = extract_cookie_id(user_text)

            # 분석 도구용 파라미터 추출
            days = extract_days(user_text)
            start_date, end_date = extract_date_range(user_text)
            month = extract_month(user_text)
            risk_level = extract_risk_level(user_text)
            cohort = extract_cohort(user_text)

            for tool_name in required_tools:
                # 도구별 인자 설정
                if tool_name == "detect_user_anomaly" and user_id:
                    args = {"user_id": user_id}
                elif tool_name == "predict_user_churn" and user_id:
                    args = {"user_id": user_id}
                elif tool_name == "optimize_investment" and user_id:
                    args = {"user_id": user_id}
                elif tool_name == "get_cookie_win_rate" and cookie_id:
                    args = {"cookie_id": cookie_id}
                elif tool_name == "get_churn_prediction":
                    args = {}
                    if risk_level:
                        args["risk_level"] = risk_level
                elif tool_name == "get_cohort_analysis":
                    args = {}
                    if cohort:
                        args["cohort"] = cohort
                    elif month:
                        args["month"] = month
                elif tool_name == "get_trend_analysis":
                    args = {}
                    if days:
                        args["days"] = days
                    elif start_date:
                        args["start_date"] = start_date
                        if end_date:
                            args["end_date"] = end_date
                elif tool_name == "get_revenue_prediction":
                    args = {}
                    if days:
                        args["days"] = days
                    elif start_date:
                        args["start_date"] = start_date
                        if end_date:
                            args["end_date"] = end_date
                elif tool_name in ["get_segment_statistics", "get_translation_statistics", "get_dashboard_summary"]:
                    args = {}
                else:
                    args = {}

                # 도구 실행
                result = execute_tool_by_name(tool_name, args)

                st.logger.info(
                    "FORCED_TOOL_CALL user=%s tool=%s args=%s status=%s",
                    username, tool_name, args, result.get("status", "UNKNOWN"),
                )

                tool_calls_log.append({
                    "tool": tool_name,
                    "args": args,
                    "result": result,
                    "forced": True,  # 강제 호출 표시
                })

                forced_results.append({
                    "tool": tool_name,
                    "result": result,
                })

            # 강제 호출 결과를 메시지에 추가
            forced_context = "다음은 사용자 요청에 따라 자동으로 실행된 도구 결과입니다:\n\n"
            for fr in forced_results:
                try:
                    result_str = json.dumps(json_sanitize(fr["result"]), ensure_ascii=False, indent=2)
                except Exception:
                    result_str = safe_str(fr["result"])
                forced_context += f"### {fr['tool']} 결과:\n```json\n{result_str}\n```\n\n"

            forced_context += "위 결과를 종합하여 사용자에게 친절하고 전문적인 답변을 작성하세요."

            # 메시지에 도구 결과 컨텍스트 추가
            messages.append(HumanMessage(content=forced_context))

        # ========== Tool Calling 루프 ==========
        while iteration < MAX_TOOL_ITERATIONS:
            iteration += 1

            # LLM 호출
            response = llm_with_tools.invoke(messages)

            # 도구 호출이 없으면 최종 응답
            if not response.tool_calls:
                final_text = safe_str(response.content).strip()
                if not final_text:
                    final_text = "요청을 처리했습니다."

                append_memory(username, user_text, final_text)

                st.logger.info(
                    "AGENT_COMPLETE user=%s iterations=%s tools_used=%s",
                    username, iteration, len(tool_calls_log),
                )

                return {
                    "status": "SUCCESS",
                    "response": final_text,
                    "tool_calls": tool_calls_log,
                    "log_file": st.LOG_FILE,
                    "mode": "tool_calling",
                    "iterations": iteration,
                }

            # 도구 실행
            messages.append(response)  # AI 메시지 추가

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]

                st.logger.info(
                    "TOOL_CALL user=%s tool=%s args=%s",
                    username, tool_name, json.dumps(tool_args, ensure_ascii=False),
                )

                # 도구 찾기 및 실행
                tool_result = {"status": "FAILED", "error": f"도구 '{tool_name}'을 찾을 수 없습니다."}
                for t in ALL_TOOLS:
                    if t.name == tool_name:
                        try:
                            tool_result = t.invoke(tool_args)
                        except Exception as e:
                            tool_result = {"status": "FAILED", "error": safe_str(e)}
                            st.logger.exception("TOOL_EXEC_FAIL tool=%s err=%s", tool_name, e)
                        break

                # 결과 로깅
                tool_calls_log.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "result": tool_result,
                })

                # ToolMessage 추가
                try:
                    result_str = json.dumps(json_sanitize(tool_result), ensure_ascii=False)
                except Exception:
                    result_str = safe_str(tool_result)

                messages.append(ToolMessage(
                    content=result_str,
                    tool_call_id=tool_id,
                ))

        # 최대 반복 도달
        st.logger.warning("AGENT_MAX_ITERATIONS user=%s", username)
        final_text = "요청 처리 중 최대 반복 횟수에 도달했습니다."
        append_memory(username, user_text, final_text)

        return {
            "status": "SUCCESS",
            "response": final_text,
            "tool_calls": tool_calls_log,
            "log_file": st.LOG_FILE,
            "mode": "tool_calling",
            "iterations": iteration,
            "max_iterations_reached": True,
        }

    except Exception as e:
        err = format_openai_error(e)
        st.logger.exception("AGENT_FAIL err=%s", err)

        msg = f"처리 오류: {err.get('type', 'Unknown')} - {err.get('message', str(e))}"
        append_memory(username, user_text, msg)

        if req.debug:
            return {
                "status": "FAILED",
                "response": msg,
                "tool_calls": [],
                "debug_error": err,
                "log_file": st.LOG_FILE,
            }

        return {
            "status": "FAILED",
            "response": "처리 오류가 발생했습니다.",
            "tool_calls": [],
            "log_file": st.LOG_FILE,
        }
