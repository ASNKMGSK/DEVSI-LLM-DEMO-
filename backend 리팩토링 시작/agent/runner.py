"""
agent/runner.py - 에이전트 실행 (Tool Calling 방식)
LLM이 직접 도구를 선택하고 호출합니다.
Rule-based 전처리로 필수 도구를 강제 호출합니다.
"""
import json
import re
from typing import Dict, Any, List, Set

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from core.constants import DEFAULT_SYSTEM_PROMPT
from core.utils import safe_str, format_openai_error, normalize_model_name, json_sanitize
from core.memory import append_memory
from agent.tool_schemas import ALL_TOOLS
from agent.llm import get_llm, pick_api_key
import state as st


MAX_TOOL_ITERATIONS = 10  # 무한 루프 방지

# 키워드-도구 매핑 (Rule-based 전처리용)
KEYWORD_TOOL_MAPPING = {
    "predict_revenue": ["매출 예측", "예측해줘", "예측해", "다음 달 매출", "매출예측", "예상 매출"],
    "detect_anomaly": ["이상 탐지", "이상 징후", "이상 거래", "비정상", "이상탐지", "이상징후", "이상거래", "이상 있"],
    "recommend_similar_merchants": ["유사 가맹점", "비슷한 가맹점", "유사가맹점", "비슷한가맹점"],
    "get_merchant_metrics": ["지표", "현황", "메트릭", "상태 보여", "정보 보여"],
    "checklist": ["체크리스트", "점검", "진단"],
}


def extract_merchant_id(text: str) -> str | None:
    """텍스트에서 가맹점 ID 추출 (M0001 형식)"""
    match = re.search(r'M\d{4}', text, re.IGNORECASE)
    return match.group(0).upper() if match else None


def extract_customer_id(text: str) -> str | None:
    """텍스트에서 고객 ID 추출 (C00001 형식)"""
    match = re.search(r'C\d{5}', text, re.IGNORECASE)
    return match.group(0).upper() if match else None


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
    Tool Calling 방식의 에이전트 실행.
    LLM이 필요한 도구를 직접 선택하고 호출합니다.
    """
    user_text = safe_str(req.user_input)

    st.logger.info(
        "AGENT_START user=%s model=%s input_len=%s mode=tool_calling",
        username, normalize_model_name(req.model), len(user_text),
    )

    api_key = pick_api_key(req.api_key)
    if not api_key:
        msg = "처리 오류: OpenAI API Key가 없습니다."
        append_memory(username, user_text, msg)
        return {"status": "FAILED", "response": msg, "tool_calls": [], "log_file": st.LOG_FILE}

    try:
        # LLM 생성 및 도구 바인딩
        llm = get_llm(
            req.model, api_key, req.max_tokens, streaming=False,
            temperature=req.temperature, top_p=req.top_p,
            presence_penalty=req.presence_penalty, frequency_penalty=req.frequency_penalty,
            seed=req.seed, timeout_ms=req.timeout_ms, max_retries=req.retries,
        )
        llm_with_tools = llm.bind_tools(ALL_TOOLS)

        # 시스템 프롬프트 구성
        system_prompt = safe_str(req.system_prompt).strip() or DEFAULT_SYSTEM_PROMPT
        system_prompt += """

## 필수 키워드-도구 매핑 (반드시 준수!)

다음 키워드가 포함되면 **반드시** 해당 도구를 호출하세요:

| 키워드 | 필수 도구 |
|--------|----------|
| "매출 예측", "예측해줘", "다음 달 매출" | `predict_revenue` |
| "이상 탐지", "이상 징후", "이상 거래", "비정상" | `detect_anomaly` |
| "유사 가맹점", "비슷한 가맹점", "추천" | `recommend_similar_merchants` |
| "지표", "현황", "상태", "메트릭" | `get_merchant_metrics` |
| "체크리스트", "점검", "진단" | `checklist` |

## 병렬 도구 호출 규칙 (매우 중요!)

사용자 요청에 **여러 키워드가 있으면 해당 도구를 모두 동시에 호출**하세요.

### 예시:
| 사용자 질문 | 호출할 도구들 |
|------------|-------------|
| "M0001 매출 예측하고 이상 탐지해줘" | `predict_revenue` + `detect_anomaly` (동시 호출) |
| "M0001 매출 예측, 이상 징후, 유사 가맹점 5개" | `predict_revenue` + `detect_anomaly` + `recommend_similar_merchants` (동시 호출) |
| "M0001 지표 보여주고 이상 있는지 확인해" | `get_merchant_metrics` + `detect_anomaly` (동시 호출) |

### 핵심:
- 요청에 포함된 **모든 키워드에 해당하는 도구를 빠짐없이 호출**
- "~하고", "~와", "~그리고" 등 여러 요청은 **병렬 호출**
- RAG 검색만 하고 끝내지 말 것! 가맹점 분석 요청에는 ML 도구 필수!

## 도구 선택 규칙 (반드시 준수)

### 핵심 규칙:
- "Top N", "상위 N개", "순위", "1위~N위" 같은 **개별 가맹점 리스트 요청** → 반드시 `rank_merchants` 사용
- "업종별 비교", "지역별 통계", "평균 비교" 같은 **집계 통계 요청** → `rank_by_dimension` 사용
- `search_documents`는 **용어/개념 정의**만! 가맹점 데이터 조회에 절대 사용 금지

### 예시:
| 사용자 질문 | 올바른 도구 |
|------------|-----------|
| "음식점 업종 Top 10 가맹점" | rank_merchants(industry="음식점", top_n=10) |
| "서울 지역 매출 상위 5개" | rank_merchants(region="서울", top_n=5) |
| "카페 업종 성장률 Top 5" | rank_merchants(industry="카페", metric="revenue_growth_rate", top_n=5) |
| "업종별 평균 매출 비교" | rank_by_dimension(dimension="industry") |
| "LTV란 무엇인가?" | search_documents(query="LTV 정의") |

### 주의:
- 가맹점 순위/리스트 질문에 search_documents나 rank_by_dimension을 사용하면 안됩니다!
- "Top N", "상위", "순위"가 포함된 질문은 항상 rank_merchants입니다.
"""

        # 메시지 구성
        messages: List = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_text),
        ]

        tool_calls_log: List[Dict[str, Any]] = []
        iteration = 0

        # ========== Rule-based 전처리: 필수 도구 강제 호출 ==========
        required_tools = detect_required_tools(user_text)
        merchant_id = extract_merchant_id(user_text)

        if required_tools and merchant_id:
            st.logger.info(
                "RULE_BASED_PREPROCESS user=%s required_tools=%s merchant_id=%s",
                username, required_tools, merchant_id,
            )

            # 필수 도구 강제 실행
            forced_results = []
            for tool_name in required_tools:
                # 도구별 인자 설정
                if tool_name in ["predict_revenue", "detect_anomaly", "get_merchant_metrics",
                                 "classify_growth", "checklist"]:
                    args = {"merchant_id": merchant_id}
                elif tool_name == "recommend_similar_merchants":
                    # top_k 추출 시도
                    top_k_match = re.search(r'(\d+)\s*개', user_text)
                    top_k = int(top_k_match.group(1)) if top_k_match else 5
                    args = {"merchant_id": merchant_id, "top_k": top_k}
                else:
                    args = {"merchant_id": merchant_id}

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
