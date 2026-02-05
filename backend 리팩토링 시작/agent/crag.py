"""
agent/crag.py - Corrective RAG (CRAG) 패턴 구현
============================================================
검색 품질을 자동으로 평가하고 교정하는 CRAG 패턴

핵심 컴포넌트:
1. RetrievalGrader - 검색 결과 관련성 평가 (gpt-4o-mini)
2. QueryRewriter - 검색 실패 시 쿼리 재작성
3. CRAGWorkflow - Correct/Incorrect/Ambiguous 처리

References:
- https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag/
- https://arxiv.org/abs/2401.15884 (CRAG Paper)
"""
import time
from typing import Optional, Literal, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from core.utils import safe_str
import state as st


class RetrievalDecision(str, Enum):
    """검색 결과 평가 결과"""
    CORRECT = "correct"        # 관련성 높음 → 바로 사용
    INCORRECT = "incorrect"    # 관련성 없음 → 쿼리 재작성 후 재검색
    AMBIGUOUS = "ambiguous"    # 애매함 → 검색 + 웹검색 병합 (현재는 재검색만)


@dataclass
class GradeResult:
    """검색 결과 평가 결과"""
    decision: RetrievalDecision
    score: float  # 0.0 ~ 1.0
    reason: str
    latency_ms: float


@dataclass
class RewriteResult:
    """쿼리 재작성 결과"""
    original_query: str
    rewritten_query: str
    latency_ms: float


@dataclass
class CRAGResult:
    """CRAG 워크플로우 결과"""
    final_context: str
    decision: RetrievalDecision
    iterations: int
    total_latency_ms: float
    grader_results: List[GradeResult]
    rewrite_results: List[RewriteResult]
    search_results: List[Dict[str, Any]]


# ============================================================
# Retrieval Grader - 검색 결과 관련성 평가
# ============================================================
GRADER_SYSTEM_PROMPT = """당신은 검색 결과 관련성 평가 전문가입니다.
사용자 질문과 검색된 문서가 관련이 있는지 평가하세요.

## 평가 기준

### 관련성 높음 (yes)
- 문서가 질문에 대한 직접적인 답변을 포함
- 문서에 질문의 핵심 엔티티/개념이 언급됨
- 질문에 대한 부분적이라도 유용한 정보 포함

### 관련성 낮음 (no)
- 문서가 질문과 완전히 다른 주제
- 질문의 핵심 엔티티가 전혀 언급되지 않음
- 문서 내용이 질문 해결에 도움이 되지 않음

## 응답 형식
반드시 "yes" 또는 "no"만 응답하세요. 다른 텍스트 없이 단어 하나만.
"""


class RetrievalGrader:
    """검색 결과 관련성 평가기 (gpt-4o-mini 사용)"""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self._llm: Optional[ChatOpenAI] = None

    def _get_llm(self) -> ChatOpenAI:
        """LLM 싱글톤"""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=self.model,
                openai_api_key=self.api_key,
                temperature=0,  # 결정론적
                max_tokens=10,  # yes/no만
            )
        return self._llm

    def grade(
        self,
        query: str,
        document: str,
        threshold: float = 0.5,
    ) -> GradeResult:
        """
        검색 결과 관련성 평가

        Args:
            query: 사용자 질문
            document: 검색된 문서 내용
            threshold: 관련성 임계값 (사용 안 함, 향후 확장용)

        Returns:
            GradeResult
        """
        start_time = time.time()

        try:
            llm = self._get_llm()

            # 문서가 너무 길면 앞부분만 사용
            doc_preview = document[:2000] if len(document) > 2000 else document

            messages = [
                SystemMessage(content=GRADER_SYSTEM_PROMPT),
                HumanMessage(content=f"""질문: {query}

검색된 문서:
{doc_preview}

이 문서가 질문과 관련이 있습니까? (yes/no)"""),
            ]

            response = llm.invoke(messages)
            answer = safe_str(response.content).strip().lower()

            latency_ms = (time.time() - start_time) * 1000

            # yes/no 파싱
            is_relevant = answer.startswith("yes")
            score = 1.0 if is_relevant else 0.0

            decision = RetrievalDecision.CORRECT if is_relevant else RetrievalDecision.INCORRECT

            st.logger.info(
                "CRAG_GRADER query=%s decision=%s score=%.2f latency=%.0fms",
                query[:30], decision.value, score, latency_ms,
            )

            return GradeResult(
                decision=decision,
                score=score,
                reason=f"LLM grader response: {answer}",
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            st.logger.warning("CRAG_GRADER_FAIL err=%s", safe_str(e))

            # 실패 시 AMBIGUOUS로 처리 (안전하게)
            return GradeResult(
                decision=RetrievalDecision.AMBIGUOUS,
                score=0.5,
                reason=f"Grader error: {safe_str(e)}",
                latency_ms=latency_ms,
            )

    def grade_documents(
        self,
        query: str,
        documents: List[str],
        min_relevant_ratio: float = 0.3,
    ) -> Tuple[RetrievalDecision, List[GradeResult]]:
        """
        여러 문서 평가 후 종합 결정

        Args:
            query: 사용자 질문
            documents: 검색된 문서 목록
            min_relevant_ratio: 최소 관련 문서 비율 (기본 30%)

        Returns:
            (종합 결정, 개별 평가 결과 목록)
        """
        if not documents:
            return RetrievalDecision.INCORRECT, []

        results = []
        relevant_count = 0

        for doc in documents:
            result = self.grade(query, doc)
            results.append(result)
            if result.decision == RetrievalDecision.CORRECT:
                relevant_count += 1

        relevant_ratio = relevant_count / len(documents)

        # 종합 결정
        if relevant_ratio >= 0.5:
            decision = RetrievalDecision.CORRECT
        elif relevant_ratio >= min_relevant_ratio:
            decision = RetrievalDecision.AMBIGUOUS
        else:
            decision = RetrievalDecision.INCORRECT

        st.logger.info(
            "CRAG_GRADE_DOCS query=%s relevant=%d/%d ratio=%.2f decision=%s",
            query[:30], relevant_count, len(documents), relevant_ratio, decision.value,
        )

        return decision, results


# ============================================================
# Query Rewriter - 쿼리 재작성
# ============================================================
REWRITER_SYSTEM_PROMPT = """당신은 검색 쿼리 최적화 전문가입니다.
사용자 질문을 더 효과적인 검색 쿼리로 재작성하세요.

## 재작성 규칙

1. **핵심 키워드 추출**: 불필요한 조사/어미 제거
2. **동의어 추가**: 핵심 개념의 동의어 포함
3. **구체화**: 모호한 표현을 구체적으로 변환
4. **쿠키런 맥락 유지**: 게임 용어 보존

## 예시

| 원본 | 재작성 |
|------|--------|
| "쿠키런 킹덤 세계관 시대적 배경이 뭐야?" | "쿠키런 킹덤 세계관 역사 시대 배경 설정" |
| "소울잼이 뭐야?" | "소울잼 정의 의미 용어 설명" |
| "다크카카오 쿠키 스토리" | "다크카카오 쿠키 배경 스토리 역사 서사" |

## 응답 형식
재작성된 쿼리만 반환하세요. 설명 없이 쿼리 텍스트만.
"""


class QueryRewriter:
    """쿼리 재작성기 (gpt-4o-mini 사용)"""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self._llm: Optional[ChatOpenAI] = None

    def _get_llm(self) -> ChatOpenAI:
        """LLM 싱글톤"""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=self.model,
                openai_api_key=self.api_key,
                temperature=0.3,  # 약간의 창의성
                max_tokens=100,
            )
        return self._llm

    def rewrite(self, query: str) -> RewriteResult:
        """
        쿼리 재작성

        Args:
            query: 원본 쿼리

        Returns:
            RewriteResult
        """
        start_time = time.time()

        try:
            llm = self._get_llm()

            messages = [
                SystemMessage(content=REWRITER_SYSTEM_PROMPT),
                HumanMessage(content=f"원본 쿼리: {query}\n\n재작성된 쿼리:"),
            ]

            response = llm.invoke(messages)
            rewritten = safe_str(response.content).strip()

            latency_ms = (time.time() - start_time) * 1000

            st.logger.info(
                "CRAG_REWRITER original=%s rewritten=%s latency=%.0fms",
                query[:30], rewritten[:30], latency_ms,
            )

            return RewriteResult(
                original_query=query,
                rewritten_query=rewritten,
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            st.logger.warning("CRAG_REWRITER_FAIL err=%s", safe_str(e))

            # 실패 시 원본 반환
            return RewriteResult(
                original_query=query,
                rewritten_query=query,
                latency_ms=latency_ms,
            )


# ============================================================
# CRAG Workflow - Correct/Incorrect/Ambiguous 처리
# ============================================================
class CRAGWorkflow:
    """
    Corrective RAG 워크플로우

    1. 검색 실행
    2. Retrieval Grader로 관련성 평가
    3. Correct → 바로 사용
    4. Incorrect → Query Rewriter → 재검색 (최대 2회)
    5. Ambiguous → 검색 결과 + 재검색 결과 병합
    """

    def __init__(
        self,
        api_key: str,
        search_func,  # (query: str, top_k: int) -> dict
        max_retries: int = 2,
        grader_model: str = "gpt-4o-mini",
        rewriter_model: str = "gpt-4o-mini",
    ):
        self.api_key = api_key
        self.search_func = search_func
        self.max_retries = max_retries
        self.grader = RetrievalGrader(api_key, grader_model)
        self.rewriter = QueryRewriter(api_key, rewriter_model)

    def _extract_documents(self, search_result: Dict[str, Any]) -> List[str]:
        """검색 결과에서 문서 텍스트 추출"""
        documents = []

        # RAG service 형식
        if "snippets" in search_result:
            for snippet in search_result.get("snippets", []):
                if isinstance(snippet, dict):
                    text = snippet.get("text", "")
                elif isinstance(snippet, str):
                    text = snippet
                else:
                    text = str(snippet)
                if text:
                    documents.append(text)

        # LightRAG 형식
        if "result" in search_result:
            result = search_result.get("result", "")
            if result:
                documents.append(result)

        # entities가 있으면 추가
        if "entities" in search_result:
            for entity in search_result.get("entities", []):
                if isinstance(entity, dict):
                    desc = entity.get("description", "")
                    if desc:
                        documents.append(desc)

        return documents

    def _format_context(self, search_results: List[Dict[str, Any]]) -> str:
        """검색 결과를 컨텍스트 문자열로 포맷"""
        context_parts = []

        for i, result in enumerate(search_results):
            if result.get("status") != "SUCCESS":
                continue

            # 스니펫 추출
            snippets = result.get("snippets", [])
            if snippets:
                for snippet in snippets:
                    if isinstance(snippet, dict):
                        text = snippet.get("text", "")
                        source = snippet.get("source", "")
                        if text:
                            context_parts.append(f"[출처: {source}]\n{text}")
                    elif isinstance(snippet, str):
                        context_parts.append(snippet)

            # LightRAG 결과
            if "result" in result:
                context_parts.append(result["result"])

        return "\n\n---\n\n".join(context_parts)

    def run(
        self,
        query: str,
        top_k: int = 5,
        grade_all_docs: bool = False,  # True면 모든 문서 개별 평가, False면 전체 결과만 평가
    ) -> CRAGResult:
        """
        CRAG 워크플로우 실행

        Args:
            query: 사용자 질문
            top_k: 검색 결과 개수
            grade_all_docs: 모든 문서 개별 평가 여부

        Returns:
            CRAGResult
        """
        start_time = time.time()

        all_search_results = []
        all_grader_results = []
        all_rewrite_results = []
        current_query = query
        iteration = 0
        final_decision = RetrievalDecision.INCORRECT

        while iteration < self.max_retries:
            iteration += 1

            # 1. 검색 실행
            st.logger.info(
                "CRAG_SEARCH iteration=%d query=%s",
                iteration, current_query[:40],
            )

            search_result = self.search_func(current_query, top_k)
            all_search_results.append(search_result)

            if search_result.get("status") != "SUCCESS":
                st.logger.warning(
                    "CRAG_SEARCH_FAIL iteration=%d err=%s",
                    iteration, search_result.get("error"),
                )
                continue

            # 2. 문서 추출
            documents = self._extract_documents(search_result)

            if not documents:
                st.logger.warning("CRAG_NO_DOCS iteration=%d", iteration)
                # 쿼리 재작성 후 재시도
                if iteration < self.max_retries:
                    rewrite_result = self.rewriter.rewrite(current_query)
                    all_rewrite_results.append(rewrite_result)
                    current_query = rewrite_result.rewritten_query
                continue

            # 3. 관련성 평가
            if grade_all_docs:
                # 모든 문서 개별 평가
                decision, grade_results = self.grader.grade_documents(query, documents)
                all_grader_results.extend(grade_results)
            else:
                # 첫 번째 문서만 대표로 평가 (비용 절감)
                combined_doc = "\n\n".join(documents[:3])  # 상위 3개 결합
                grade_result = self.grader.grade(query, combined_doc)
                all_grader_results.append(grade_result)
                decision = grade_result.decision

            final_decision = decision

            # 4. 결정에 따른 처리
            if decision == RetrievalDecision.CORRECT:
                # 관련성 높음 → 종료
                st.logger.info("CRAG_CORRECT iteration=%d", iteration)
                break

            elif decision == RetrievalDecision.INCORRECT:
                # 관련성 없음 → 쿼리 재작성 후 재검색
                if iteration < self.max_retries:
                    rewrite_result = self.rewriter.rewrite(current_query)
                    all_rewrite_results.append(rewrite_result)
                    current_query = rewrite_result.rewritten_query
                    st.logger.info(
                        "CRAG_INCORRECT_REWRITE iteration=%d new_query=%s",
                        iteration, current_query[:40],
                    )

            elif decision == RetrievalDecision.AMBIGUOUS:
                # 애매함 → 쿼리 재작성 후 추가 검색 (결과 병합)
                if iteration < self.max_retries:
                    rewrite_result = self.rewriter.rewrite(current_query)
                    all_rewrite_results.append(rewrite_result)
                    current_query = rewrite_result.rewritten_query
                    st.logger.info(
                        "CRAG_AMBIGUOUS_REWRITE iteration=%d new_query=%s",
                        iteration, current_query[:40],
                    )

        # 5. 최종 컨텍스트 생성
        final_context = self._format_context(all_search_results)

        total_latency_ms = (time.time() - start_time) * 1000

        st.logger.info(
            "CRAG_COMPLETE decision=%s iterations=%d latency=%.0fms",
            final_decision.value, iteration, total_latency_ms,
        )

        return CRAGResult(
            final_context=final_context,
            decision=final_decision,
            iterations=iteration,
            total_latency_ms=total_latency_ms,
            grader_results=all_grader_results,
            rewrite_results=all_rewrite_results,
            search_results=all_search_results,
        )


# ============================================================
# 편의 함수
# ============================================================
def run_crag_search(
    query: str,
    api_key: str,
    search_func,
    top_k: int = 5,
    max_retries: int = 2,
) -> CRAGResult:
    """
    CRAG 검색 실행 (원스톱 함수)

    Args:
        query: 사용자 질문
        api_key: OpenAI API 키
        search_func: 검색 함수 (query, top_k) -> dict
        top_k: 검색 결과 개수
        max_retries: 최대 재시도 횟수

    Returns:
        CRAGResult
    """
    workflow = CRAGWorkflow(
        api_key=api_key,
        search_func=search_func,
        max_retries=max_retries,
    )

    return workflow.run(query, top_k)


def get_crag_context_for_worldview(
    query: str,
    api_key: str,
    use_lightrag: bool = True,
    top_k: int = 5,
) -> Tuple[str, Dict[str, Any]]:
    """
    세계관 질문에 대한 CRAG 컨텍스트 생성

    Args:
        query: 사용자 질문
        api_key: OpenAI API 키
        use_lightrag: LightRAG 사용 여부
        top_k: 검색 결과 개수

    Returns:
        (컨텍스트 문자열, 메타데이터)
    """
    from rag.service import tool_rag_search
    from rag.light_rag import lightrag_search_sync, get_lightrag_status

    # 검색 함수 선택
    if use_lightrag:
        status = get_lightrag_status()
        if status.get("ready"):
            def search_func(q: str, k: int) -> dict:
                return lightrag_search_sync(q, mode="hybrid")
        else:
            def search_func(q: str, k: int) -> dict:
                return tool_rag_search(q, top_k=k, api_key=api_key)
    else:
        def search_func(q: str, k: int) -> dict:
            return tool_rag_search(q, top_k=k, api_key=api_key)

    # CRAG 실행
    result = run_crag_search(
        query=query,
        api_key=api_key,
        search_func=search_func,
        top_k=top_k,
        max_retries=2,
    )

    metadata = {
        "decision": result.decision.value,
        "iterations": result.iterations,
        "total_latency_ms": result.total_latency_ms,
        "grader_count": len(result.grader_results),
        "rewrite_count": len(result.rewrite_results),
    }

    return result.final_context, metadata
