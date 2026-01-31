"""
멀티 에이전트 서비스
===================
데브시스터즈 기술혁신 프로젝트

데이터 분석 및 의사결정을 돕는 멀티 에이전트 기반 서비스:
- 번역 에이전트: 세계관 맞춤형 번역
- 분석 에이전트: 유저/게임 데이터 분석
- 검색 에이전트: 세계관 지식 검색
- 코디네이터: 에이전트 조율 및 결과 통합
"""

import os
import json
import asyncio
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from openai import OpenAI

from core.constants import (
    MULTI_AGENT_SYSTEM_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    TRANSLATION_SYSTEM_PROMPT,
)
from agent.tools import AVAILABLE_TOOLS
import state as st


class AgentType(str, Enum):
    COORDINATOR = "coordinator"
    TRANSLATOR = "translator"
    ANALYST = "analyst"
    SEARCHER = "searcher"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentTask:
    """에이전트 작업"""
    task_id: str
    agent_type: AgentType
    description: str
    input_data: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


@dataclass
class AgentConfig:
    """에이전트 설정"""
    agent_type: AgentType
    name: str
    description: str
    system_prompt: str
    tools: List[str]
    model: str = "gpt-4o-mini"


# 에이전트 설정
AGENT_CONFIGS = {
    AgentType.COORDINATOR: AgentConfig(
        agent_type=AgentType.COORDINATOR,
        name="코디네이터",
        description="여러 에이전트를 조율하여 복잡한 요청을 처리합니다.",
        system_prompt=MULTI_AGENT_SYSTEM_PROMPT,
        tools=["get_dashboard_summary", "search_worldview"],
        model="gpt-4o",
    ),
    AgentType.TRANSLATOR: AgentConfig(
        agent_type=AgentType.TRANSLATOR,
        name="번역 에이전트",
        description="쿠키런 세계관에 맞춰 텍스트를 번역합니다.",
        system_prompt=TRANSLATION_SYSTEM_PROMPT,
        tools=["translate_text", "check_translation_quality", "get_worldview_terms"],
    ),
    AgentType.ANALYST: AgentConfig(
        agent_type=AgentType.ANALYST,
        name="분석 에이전트",
        description="유저 및 게임 데이터를 분석합니다.",
        system_prompt="""당신은 게임 데이터 분석 전문가입니다.
유저 행동 패턴, 게임 이벤트 통계, 이상 탐지 등의 분석을 수행합니다.
분석 결과는 명확하고 구조화된 형태로 제공하며, 비즈니스 인사이트를 도출합니다.""",
        tools=[
            "analyze_user", "get_user_segment", "detect_user_anomaly",
            "get_segment_statistics", "get_anomaly_statistics", "get_event_statistics",
            "get_user_activity_report", "get_translation_statistics",
        ],
    ),
    AgentType.SEARCHER: AgentConfig(
        agent_type=AgentType.SEARCHER,
        name="검색 에이전트",
        description="쿠키런 세계관 정보를 검색합니다.",
        system_prompt="""당신은 쿠키런 세계관 전문가입니다.
쿠키 캐릭터, 왕국, 스킬, 스토리 등에 대한 정보를 검색하고 제공합니다.
정확한 정보를 바탕으로 질문에 답변합니다.""",
        tools=[
            "get_cookie_info", "list_cookies", "get_cookie_skill",
            "get_kingdom_info", "list_kingdoms", "search_worldview",
            "classify_text",
        ],
    ),
}


class Agent:
    """개별 에이전트"""

    def __init__(self, config: AgentConfig, client: OpenAI):
        self.config = config
        self.client = client
        self.tools = {name: AVAILABLE_TOOLS[name] for name in config.tools if name in AVAILABLE_TOOLS}

    def execute(self, task: AgentTask) -> Dict[str, Any]:
        """작업 실행"""
        try:
            task.status = TaskStatus.RUNNING

            # 도구 호출이 필요한지 판단
            tool_results = []
            for tool_name, tool_func in self.tools.items():
                if self._should_use_tool(task.description, tool_name):
                    try:
                        result = tool_func(**task.input_data) if task.input_data else tool_func()
                        tool_results.append({
                            "tool": tool_name,
                            "result": result,
                        })
                    except Exception as e:
                        tool_results.append({
                            "tool": tool_name,
                            "error": str(e),
                        })

            # LLM으로 결과 종합
            messages = [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": self._build_prompt(task, tool_results)},
            ]

            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=0.3,
                max_tokens=2000,
            )

            result_text = response.choices[0].message.content

            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = {
                "response": result_text,
                "tool_results": tool_results,
                "agent": self.config.name,
            }

            return task.result

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            return {"error": str(e), "agent": self.config.name}

    def _should_use_tool(self, description: str, tool_name: str) -> bool:
        """도구 사용 여부 판단"""
        tool_keywords = {
            "get_cookie_info": ["쿠키", "캐릭터", "정보"],
            "list_cookies": ["목록", "리스트", "전체"],
            "get_cookie_skill": ["스킬", "능력"],
            "get_kingdom_info": ["왕국", "지역"],
            "list_kingdoms": ["왕국 목록", "지역 목록"],
            "translate_text": ["번역", "translate"],
            "check_translation_quality": ["품질", "검토", "평가"],
            "get_worldview_terms": ["용어", "용어집", "glossary"],
            "analyze_user": ["유저 분석", "사용자 분석"],
            "get_user_segment": ["세그먼트", "유형"],
            "detect_user_anomaly": ["이상 유저", "이상 탐지 실행"],
            "get_anomaly_statistics": ["이상", "탐지", "anomaly", "비정상", "이상 행동"],
            "get_segment_statistics": ["세그먼트 통계", "유저 통계"],
            "get_event_statistics": ["이벤트 통계", "게임 통계"],
            "get_user_activity_report": ["활동 리포트", "활동 보고서"],
            "get_translation_statistics": ["번역 통계"],
            "search_worldview": ["검색", "찾아", "알려줘"],
            "classify_text": ["분류", "카테고리"],
            "get_dashboard_summary": ["대시보드", "요약", "전체"],
        }

        keywords = tool_keywords.get(tool_name, [])
        description_lower = description.lower()
        return any(kw in description_lower for kw in keywords)

    def _build_prompt(self, task: AgentTask, tool_results: List[Dict]) -> str:
        """프롬프트 생성"""
        prompt = f"**요청**: {task.description}\n\n"

        if task.input_data:
            prompt += f"**입력 데이터**: {json.dumps(task.input_data, ensure_ascii=False)}\n\n"

        if tool_results:
            prompt += "**도구 실행 결과**:\n"
            for tr in tool_results:
                prompt += f"- {tr['tool']}: {json.dumps(tr.get('result', tr.get('error')), ensure_ascii=False, indent=2)}\n"

        prompt += "\n위 정보를 바탕으로 요청에 대해 답변해주세요."
        return prompt


class MultiAgentSystem:
    """멀티 에이전트 시스템"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or st.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.agents = {}

        if self.client:
            for agent_type, config in AGENT_CONFIGS.items():
                self.agents[agent_type] = Agent(config, self.client)

        self.tasks: Dict[str, AgentTask] = {}
        self._task_counter = 0

    def _generate_task_id(self) -> str:
        """작업 ID 생성"""
        self._task_counter += 1
        return f"task_{self._task_counter}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    def create_task(
        self,
        agent_type: AgentType,
        description: str,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> AgentTask:
        """작업 생성"""
        task = AgentTask(
            task_id=self._generate_task_id(),
            agent_type=agent_type,
            description=description,
            input_data=input_data or {},
        )
        self.tasks[task.task_id] = task
        return task

    def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """작업 실행"""
        if not self.client:
            return {"error": "API 키가 설정되지 않았습니다."}

        agent = self.agents.get(task.agent_type)
        if not agent:
            return {"error": f"에이전트 타입 '{task.agent_type}'을 찾을 수 없습니다."}

        return agent.execute(task)

    def route_request(self, user_request: str, input_data: Optional[Dict] = None) -> Dict[str, Any]:
        """요청을 적절한 에이전트로 라우팅"""
        request_lower = user_request.lower()

        # 에이전트 타입 결정
        if any(kw in request_lower for kw in ["번역", "translate", "영어로", "일본어로", "중국어로"]):
            agent_type = AgentType.TRANSLATOR
        elif any(kw in request_lower for kw in ["분석", "통계", "세그먼트", "이상 탐지", "유저"]):
            agent_type = AgentType.ANALYST
        elif any(kw in request_lower for kw in ["쿠키", "왕국", "스킬", "세계관", "캐릭터"]):
            agent_type = AgentType.SEARCHER
        else:
            agent_type = AgentType.COORDINATOR

        # 작업 생성 및 실행
        task = self.create_task(agent_type, user_request, input_data)
        result = self.execute_task(task)

        return {
            "task_id": task.task_id,
            "agent_type": agent_type.value,
            "agent_name": AGENT_CONFIGS[agent_type].name,
            "status": task.status.value,
            "result": result,
        }

    def execute_pipeline(
        self,
        steps: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """파이프라인 실행 (순차적 에이전트 호출)"""
        results = []
        context = {}

        for i, step in enumerate(steps):
            agent_type = AgentType(step.get("agent", "coordinator"))
            description = step.get("description", "")
            input_data = step.get("input", {})

            # 이전 결과를 컨텍스트로 추가
            if context:
                input_data["previous_context"] = context

            task = self.create_task(agent_type, description, input_data)
            result = self.execute_task(task)

            results.append({
                "step": i + 1,
                "agent": agent_type.value,
                "result": result,
            })

            # 컨텍스트 업데이트
            if result.get("response"):
                context = {
                    "step": i + 1,
                    "response": result["response"],
                }

        return {
            "pipeline_completed": True,
            "total_steps": len(steps),
            "results": results,
        }


# 싱글톤 인스턴스
_multi_agent_system: Optional[MultiAgentSystem] = None


def get_multi_agent_system() -> MultiAgentSystem:
    """멀티 에이전트 시스템 인스턴스 가져오기"""
    global _multi_agent_system
    if _multi_agent_system is None:
        _multi_agent_system = MultiAgentSystem()
    return _multi_agent_system


def process_request(user_request: str, input_data: Optional[Dict] = None) -> Dict[str, Any]:
    """요청 처리 (간편 함수)"""
    system = get_multi_agent_system()
    return system.route_request(user_request, input_data)
