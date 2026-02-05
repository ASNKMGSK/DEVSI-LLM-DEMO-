"""
rag/k2rag.py - K²RAG (KeyKnowledgeRAG) 구현
논문: https://arxiv.org/abs/2507.07695 (July 2025)

핵심 특징:
1. Knowledge Graph + Hybrid Search (Dense 80% + Sparse 20%)
2. Corpus Summarization: 학습 시간 93% 감소
3. Sub-question Generation: KG 결과에서 서브 질문 생성
4. 경량 모델 사용: Quantized LLM + Longformer Summarizer

Pipeline:
[Query] → [KG Search] → [Summarize KG] → [Sub-questions]
       → [Hybrid Retrieval per sub-Q] → [Sub-answers]
       → [Summarize] → [Final Answer]
"""

import os
import re
import json
import time
import hashlib
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

import state as st
from core.utils import safe_str

# ============================================================
# Optional Dependencies
# ============================================================

# Summarization Model (Longformer LED)
SUMMARIZER = None
SUMMARIZER_TOKENIZER = None
SUMMARIZER_AVAILABLE = False
SUMMARIZER_MODEL_NAME = "pszemraj/led-base-book-summary"

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    SUMMARIZER_AVAILABLE = True
except ImportError:
    pass

# BM25 for Sparse Search
BM25Okapi = None
BM25_AVAILABLE = False
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    pass

# FAISS for Dense Search
FAISS = None
try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    pass

# OpenAI Embeddings
OpenAIEmbeddings = None
try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    pass

# OpenAI Client for LLM
OPENAI_CLIENT = None


# ============================================================
# Configuration
# ============================================================
@dataclass
class K2RAGConfig:
    """K²RAG 설정"""
    # Chunking for Vector Stores
    vector_chunk_size: int = 256
    vector_chunk_overlap: int = 20

    # Chunking for Knowledge Graph
    kg_chunk_size: int = 300
    kg_chunk_overlap: int = 100

    # Chunking for Sub-questions
    subq_chunk_size: int = 128
    subq_chunk_overlap: int = 10

    # Hybrid Search Weights (λ = 0.8 means 80% dense, 20% sparse)
    hybrid_lambda: float = 0.8

    # Top-K for retrieval
    top_k: int = 10

    # Summarization
    summary_max_length: int = 300
    summary_min_length: int = 50

    # LLM Model
    llm_model: str = "gpt-4o-mini"  # 경량 모델 사용

    # Enable/Disable components
    use_summarization: bool = True
    use_knowledge_graph: bool = True
    use_hybrid_search: bool = True


# Global config
K2RAG_CONFIG = K2RAGConfig()

# ============================================================
# State
# ============================================================
K2RAG_STATE = {
    "initialized": False,
    "summarized_corpus": {},  # doc_id -> summary
    "dense_store": None,      # FAISS store
    "sparse_store": None,     # BM25 store
    "chunks": [],             # All indexed chunks
    "chunk_texts": [],        # Texts for BM25
    "knowledge_graph": {},    # entity -> relations
}


# ============================================================
# Initialization
# ============================================================
def _get_openai_client():
    """Lazy OpenAI client initialization"""
    global OPENAI_CLIENT
    if OPENAI_CLIENT is not None:
        return OPENAI_CLIENT

    try:
        from openai import OpenAI
        api_key = getattr(st, "OPENAI_API_KEY", None)
        if not api_key:
            st.logger.warning("K2RAG_OPENAI_KEY_NOT_SET")
            return None
        OPENAI_CLIENT = OpenAI(api_key=api_key)
        st.logger.info("K2RAG_OPENAI_CLIENT_INITIALIZED")
        return OPENAI_CLIENT
    except Exception as e:
        st.logger.warning("K2RAG_OPENAI_INIT_FAIL err=%s", safe_str(e))
        return None


def _load_summarizer():
    """Longformer Summarizer 로드 (LED-base-book-summary)"""
    global SUMMARIZER, SUMMARIZER_TOKENIZER, SUMMARIZER_AVAILABLE

    if not SUMMARIZER_AVAILABLE:
        st.logger.warning("K2RAG_SUMMARIZER_NOT_AVAILABLE (pip install transformers)")
        return False

    if SUMMARIZER is not None:
        return True

    try:
        st.logger.info("K2RAG_LOADING_SUMMARIZER model=%s", SUMMARIZER_MODEL_NAME)
        start = time.time()

        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch

        SUMMARIZER_TOKENIZER = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL_NAME)
        SUMMARIZER = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL_NAME)

        # GPU 사용 가능하면 이동
        if torch.cuda.is_available():
            SUMMARIZER = SUMMARIZER.to("cuda")
            st.logger.info("K2RAG_SUMMARIZER_GPU_ENABLED")

        elapsed = time.time() - start
        st.logger.info("K2RAG_SUMMARIZER_LOADED elapsed=%.1fs", elapsed)
        return True
    except Exception as e:
        st.logger.error("K2RAG_SUMMARIZER_LOAD_FAIL err=%s", safe_str(e))
        SUMMARIZER_AVAILABLE = False
        return False


# ============================================================
# Text Processing Utilities
# ============================================================
def _sha1(text: str) -> str:
    """텍스트 해시 생성"""
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]


def _clean_text(text: str) -> str:
    """텍스트 정제"""
    if not text:
        return ""
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """텍스트를 청크로 분할 (토큰 기반 아님, 문자 기반)"""
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += (chunk_size - overlap)

    return chunks


# ============================================================
# Summarization Module
# ============================================================
def summarize_text(text: str, max_length: int = 300, min_length: int = 50) -> str:
    """
    Longformer LED 모델로 텍스트 요약
    - 긴 텍스트 처리에 최적화
    - GPU 지원
    """
    if not text or len(text.strip()) < min_length:
        return text

    # Summarizer 사용 불가시 원본 반환
    if not _load_summarizer():
        # Fallback: 앞부분 추출
        return text[:max_length] + "..." if len(text) > max_length else text

    try:
        import torch

        # 입력 토큰화
        inputs = SUMMARIZER_TOKENIZER(
            text,
            return_tensors="pt",
            max_length=4096,  # LED max input
            truncation=True
        )

        # GPU로 이동
        if torch.cuda.is_available() and SUMMARIZER.device.type == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # 요약 생성
        with torch.no_grad():
            summary_ids = SUMMARIZER.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        summary = SUMMARIZER_TOKENIZER.decode(summary_ids[0], skip_special_tokens=True)
        return summary.strip()

    except Exception as e:
        st.logger.warning("K2RAG_SUMMARIZE_FAIL err=%s", safe_str(e))
        return text[:max_length] + "..." if len(text) > max_length else text


def summarize_corpus(documents: List[Dict[str, str]]) -> Dict[str, str]:
    """
    전체 코퍼스 요약 (학습 전처리)
    - 학습 시간 93% 감소 효과

    Args:
        documents: [{"id": "doc1", "content": "..."}]

    Returns:
        {doc_id: summary}
    """
    st.logger.info("K2RAG_CORPUS_SUMMARIZATION_START count=%d", len(documents))
    start = time.time()

    summaries = {}
    for i, doc in enumerate(documents):
        doc_id = doc.get("id", f"doc_{i}")
        content = doc.get("content", "")

        if not content:
            continue

        summary = summarize_text(content, max_length=K2RAG_CONFIG.summary_max_length)
        summaries[doc_id] = summary

        if (i + 1) % 50 == 0:
            st.logger.info("K2RAG_SUMMARIZING progress=%d/%d", i + 1, len(documents))

    elapsed = time.time() - start
    st.logger.info("K2RAG_CORPUS_SUMMARIZATION_DONE count=%d elapsed=%.1fs", len(summaries), elapsed)

    K2RAG_STATE["summarized_corpus"] = summaries
    return summaries


# ============================================================
# Hybrid Search Module
# ============================================================
def _compute_hybrid_score(
    dense_score: float,
    sparse_score: float,
    lambda_weight: float = 0.8
) -> float:
    """
    Hybrid Score 계산: λ * dense + (1-λ) * sparse
    논문: λ = 0.8 (80% dense, 20% sparse)
    """
    return lambda_weight * dense_score + (1 - lambda_weight) * sparse_score


def _normalize_scores(scores: List[float]) -> List[float]:
    """점수 정규화 [0, 1]"""
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def hybrid_search(
    query: str,
    top_k: int = 10,
    lambda_weight: float = 0.8
) -> List[Tuple[str, float]]:
    """
    Hybrid Search: Dense (FAISS) + Sparse (BM25)

    Args:
        query: 검색 쿼리
        top_k: 반환할 결과 수
        lambda_weight: Dense 비중 (0.8 = 80%)

    Returns:
        [(chunk_text, score), ...]
    """
    dense_store = K2RAG_STATE.get("dense_store")
    sparse_store = K2RAG_STATE.get("sparse_store")
    chunks = K2RAG_STATE.get("chunks", [])
    chunk_texts = K2RAG_STATE.get("chunk_texts", [])

    if not chunks:
        st.logger.warning("K2RAG_HYBRID_SEARCH_NO_CHUNKS")
        return []

    results = {}  # chunk_idx -> {"dense": score, "sparse": score, "text": text}

    # 1. Dense Search (FAISS)
    if dense_store is not None:
        try:
            dense_results = dense_store.similarity_search_with_score(query, k=min(top_k * 2, len(chunks)))
            for doc, score in dense_results:
                # FAISS는 거리 반환, 유사도로 변환
                similarity = 1 / (1 + score)
                text = doc.page_content
                idx = chunk_texts.index(text) if text in chunk_texts else -1
                if idx >= 0:
                    if idx not in results:
                        results[idx] = {"text": text, "dense": 0, "sparse": 0}
                    results[idx]["dense"] = similarity
        except Exception as e:
            st.logger.warning("K2RAG_DENSE_SEARCH_FAIL err=%s", safe_str(e))

    # 2. Sparse Search (BM25)
    if sparse_store is not None and chunk_texts:
        try:
            # 한국어 토큰화 (간단한 공백 분리)
            query_tokens = query.split()
            bm25_scores = sparse_store.get_scores(query_tokens)

            # 상위 결과만 추가
            top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k * 2]

            for idx in top_indices:
                if idx not in results:
                    results[idx] = {"text": chunk_texts[idx], "dense": 0, "sparse": 0}
                results[idx]["sparse"] = bm25_scores[idx]
        except Exception as e:
            st.logger.warning("K2RAG_SPARSE_SEARCH_FAIL err=%s", safe_str(e))

    # 3. 점수 정규화 및 Hybrid Score 계산
    if not results:
        return []

    # 각 점수 정규화
    indices = list(results.keys())
    dense_scores = [results[i]["dense"] for i in indices]
    sparse_scores = [results[i]["sparse"] for i in indices]

    norm_dense = _normalize_scores(dense_scores)
    norm_sparse = _normalize_scores(sparse_scores)

    # Hybrid Score
    final_results = []
    for i, idx in enumerate(indices):
        hybrid_score = _compute_hybrid_score(norm_dense[i], norm_sparse[i], lambda_weight)
        final_results.append((results[idx]["text"], hybrid_score))

    # 정렬 및 Top-K 반환
    final_results.sort(key=lambda x: x[1], reverse=True)
    return final_results[:top_k]


# ============================================================
# Sub-question Generation
# ============================================================
def generate_subquestion(chunk: str) -> str:
    """
    청크에서 서브 질문 생성
    논문: Question Generation System Prompt 사용
    """
    client = _get_openai_client()
    if not client:
        return f"What is the main topic about: {chunk[:100]}?"

    prompt = f"""Instruction: Your task is to create a small question out of the below information.
Information: {chunk}
Answer:"""

    try:
        response = client.chat.completions.create(
            model=K2RAG_CONFIG.llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.logger.warning("K2RAG_SUBQUESTION_GEN_FAIL err=%s", safe_str(e))
        return f"What is the main information about: {chunk[:50]}?"


def generate_answer(question: str, context: str) -> str:
    """
    컨텍스트 기반 답변 생성
    논문: Answer Generation System Prompt 사용
    """
    client = _get_openai_client()
    if not client:
        return context[:500]

    prompt = f"""Additional Information: {context}

Instruction: You are a smart LLM who gives an answer to the question in as little words as possible using the additional information provided above.

Question: {question}
Short Answer:"""

    try:
        response = client.chat.completions.create(
            model=K2RAG_CONFIG.llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.logger.warning("K2RAG_ANSWER_GEN_FAIL err=%s", safe_str(e))
        return context[:500]


# ============================================================
# Knowledge Graph Module (Simple)
# ============================================================
def query_knowledge_graph(query: str) -> str:
    """
    Knowledge Graph에서 관련 정보 검색
    (기존 LightRAG/service.py의 KG 활용 가능)
    """
    kg = K2RAG_STATE.get("knowledge_graph", {})
    if not kg:
        # Fallback: service.py의 KNOWLEDGE_GRAPH 사용
        try:
            from rag.service import KNOWLEDGE_GRAPH
            kg = KNOWLEDGE_GRAPH
        except:
            pass

    if not kg:
        return ""

    # 간단한 키워드 매칭으로 관련 엔티티 찾기
    query_words = set(query.lower().split())
    results = []

    for entity, relations in kg.items():
        entity_lower = entity.lower()
        if any(w in entity_lower for w in query_words):
            for rel in relations:
                results.append(f"{entity} - {rel.get('relation', '')} - {rel.get('target', '')}")

    return "\n".join(results[:20])  # 최대 20개


# ============================================================
# K²RAG Main Pipeline
# ============================================================
async def k2rag_search(
    query: str,
    top_k: int = 10,
    use_kg: bool = True,
    use_summary: bool = True
) -> Dict[str, Any]:
    """
    K²RAG 전체 파이프라인 실행

    Pipeline:
    1. [A] Knowledge Graph 검색
    2. [B] KG 결과 요약
    3. [C] KG 요약에서 서브 질문 생성
    4. [D] 각 서브 질문에 대해 Hybrid Search → 서브 답변 생성 → 요약
    5. [E] 최종 답변 생성

    Args:
        query: 사용자 질문
        top_k: 검색 결과 수
        use_kg: Knowledge Graph 사용 여부
        use_summary: 요약 사용 여부

    Returns:
        {
            "status": "SUCCESS",
            "answer": "...",
            "context": "...",
            "kg_results": "...",
            "sub_answers": [...],
            "elapsed_ms": ...
        }
    """
    start_time = time.time()
    st.logger.info("K2RAG_SEARCH_START query=%s top_k=%d", query[:50], top_k)

    # 초기화 체크 - 데이터가 없으면 자동 로드
    if not K2RAG_STATE.get("initialized") or not K2RAG_STATE.get("chunk_texts"):
        st.logger.info("K2RAG_AUTO_LOAD - 데이터가 없어서 기존 RAG에서 로드 시도")
        load_from_existing_rag()

        # 로드 실패 시 에러 반환
        if not K2RAG_STATE.get("chunk_texts"):
            return {
                "status": "FAILED",
                "error": "K²RAG 데이터가 없습니다. 먼저 /api/rag/reload를 호출하세요.",
                "query": query,
                "answer": "",
                "context": "",
                "kg_results": "",
                "sub_answers": [],
                "elapsed_ms": 0
            }

    result = {
        "status": "SUCCESS",
        "query": query,
        "answer": "",
        "context": "",
        "kg_results": "",
        "sub_answers": [],
        "elapsed_ms": 0
    }

    try:
        # ============================================
        # Step A: Knowledge Graph 검색
        # ============================================
        kg_results = ""
        if use_kg:
            kg_results = query_knowledge_graph(query)
            result["kg_results"] = kg_results
            st.logger.info("K2RAG_KG_RESULTS len=%d", len(kg_results))

        # ============================================
        # Step B: KG 결과 요약
        # ============================================
        kg_summary = ""
        if kg_results and use_summary:
            kg_summary = summarize_text(kg_results, max_length=300)
            st.logger.info("K2RAG_KG_SUMMARY len=%d", len(kg_summary))
        else:
            kg_summary = kg_results

        # ============================================
        # Step C: 서브 질문 생성 (KG 요약 청킹)
        # ============================================
        sub_questions = []
        if kg_summary:
            kg_chunks = _chunk_text(
                kg_summary,
                K2RAG_CONFIG.subq_chunk_size,
                K2RAG_CONFIG.subq_chunk_overlap
            )

            for chunk in kg_chunks[:5]:  # 최대 5개 서브 질문
                sub_q = generate_subquestion(chunk)
                sub_questions.append(sub_q)

            st.logger.info("K2RAG_SUB_QUESTIONS count=%d", len(sub_questions))

        # KG 결과가 없으면 원본 쿼리로 직접 검색
        if not sub_questions:
            sub_questions = [query]

        # ============================================
        # Step D: 각 서브 질문에 대해 Hybrid Search + 답변
        # ============================================
        sub_answers = []
        all_contexts = []

        for sub_q in sub_questions:
            # Hybrid Search
            search_results = hybrid_search(sub_q, top_k=K2RAG_CONFIG.top_k, lambda_weight=K2RAG_CONFIG.hybrid_lambda)

            if search_results:
                # 컨텍스트 구성
                sub_context = "\n".join([r[0] for r in search_results[:5]])
                all_contexts.append(sub_context)

                # 서브 답변 생성
                sub_answer = generate_answer(sub_q, sub_context)

                # 서브 답변 요약
                if use_summary and len(sub_answer) > 200:
                    sub_answer = summarize_text(sub_answer, max_length=150)

                sub_answers.append({
                    "question": sub_q,
                    "answer": sub_answer
                })

        result["sub_answers"] = sub_answers
        st.logger.info("K2RAG_SUB_ANSWERS count=%d", len(sub_answers))

        # ============================================
        # Step E: 최종 답변 생성
        # ============================================
        # 서브 답변들 합치기
        sub_answer_context = "\n".join([sa["answer"] for sa in sub_answers])

        # 요약
        if use_summary and len(sub_answer_context) > 500:
            sub_answer_context = summarize_text(sub_answer_context, max_length=400)

        # 전체 컨텍스트 = KG 요약 + 서브 답변 요약
        full_context = f"{kg_summary}\n\n{sub_answer_context}".strip()
        result["context"] = full_context

        # 최종 답변 생성
        final_answer = generate_answer(query, full_context)
        result["answer"] = final_answer

        elapsed_ms = (time.time() - start_time) * 1000
        result["elapsed_ms"] = round(elapsed_ms, 1)

        st.logger.info("K2RAG_SEARCH_DONE elapsed=%.0fms answer_len=%d", elapsed_ms, len(final_answer))

    except Exception as e:
        st.logger.error("K2RAG_SEARCH_FAIL err=%s", safe_str(e))
        result["status"] = "FAILED"
        result["error"] = safe_str(e)

    return result


def k2rag_search_sync(query: str, top_k: int = 10, use_kg: bool = True, use_summary: bool = True) -> Dict[str, Any]:
    """동기 버전의 K²RAG 검색"""
    return asyncio.run(k2rag_search(query, top_k, use_kg, use_summary))


# ============================================================
# Indexing Module
# ============================================================
def index_documents(
    documents: List[Dict[str, str]],
    use_summarization: bool = True
) -> Dict[str, Any]:
    """
    문서 인덱싱 (K²RAG 방식)

    1. (Optional) 코퍼스 요약 - 학습 시간 93% 감소
    2. 청킹
    3. Dense Vector Store (FAISS) 생성
    4. Sparse Vector Store (BM25) 생성

    Args:
        documents: [{"id": "doc1", "content": "...", "metadata": {...}}]
        use_summarization: 요약 사용 여부

    Returns:
        {"status": "SUCCESS", "chunks": count, "elapsed_s": ...}
    """
    st.logger.info("K2RAG_INDEX_START docs=%d summarize=%s", len(documents), use_summarization)
    start_time = time.time()

    try:
        # 1. 코퍼스 요약 (선택적)
        if use_summarization:
            summaries = summarize_corpus(documents)
            # 요약된 문서 사용
            processed_docs = [
                {"id": d.get("id", f"doc_{i}"), "content": summaries.get(d.get("id", f"doc_{i}"), d.get("content", ""))}
                for i, d in enumerate(documents)
            ]
        else:
            processed_docs = documents

        # 2. 청킹
        all_chunks = []
        chunk_texts = []

        for doc in processed_docs:
            content = _clean_text(doc.get("content", ""))
            if not content:
                continue

            chunks = _chunk_text(content, K2RAG_CONFIG.vector_chunk_size, K2RAG_CONFIG.vector_chunk_overlap)
            for chunk in chunks:
                all_chunks.append({
                    "text": chunk,
                    "doc_id": doc.get("id", ""),
                    "metadata": doc.get("metadata", {})
                })
                chunk_texts.append(chunk)

        K2RAG_STATE["chunks"] = all_chunks
        K2RAG_STATE["chunk_texts"] = chunk_texts

        st.logger.info("K2RAG_CHUNKING_DONE chunks=%d", len(all_chunks))

        # 3. Dense Vector Store (FAISS)
        if FAISS and OpenAIEmbeddings and chunk_texts:
            try:
                api_key = getattr(st, "OPENAI_API_KEY", None)
                if api_key:
                    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

                    from langchain_core.documents import Document
                    docs = [Document(page_content=c["text"], metadata=c.get("metadata", {})) for c in all_chunks]

                    dense_store = FAISS.from_documents(docs, embeddings)
                    K2RAG_STATE["dense_store"] = dense_store
                    st.logger.info("K2RAG_DENSE_STORE_CREATED")
            except Exception as e:
                st.logger.warning("K2RAG_DENSE_STORE_FAIL err=%s", safe_str(e))

        # 4. Sparse Vector Store (BM25)
        if BM25_AVAILABLE and chunk_texts:
            try:
                # 한국어 토큰화 (간단한 공백 분리)
                tokenized = [text.split() for text in chunk_texts]
                sparse_store = BM25Okapi(tokenized)
                K2RAG_STATE["sparse_store"] = sparse_store
                st.logger.info("K2RAG_SPARSE_STORE_CREATED")
            except Exception as e:
                st.logger.warning("K2RAG_SPARSE_STORE_FAIL err=%s", safe_str(e))

        K2RAG_STATE["initialized"] = True
        elapsed = time.time() - start_time

        st.logger.info("K2RAG_INDEX_DONE chunks=%d elapsed=%.1fs", len(all_chunks), elapsed)

        return {
            "status": "SUCCESS",
            "chunks": len(all_chunks),
            "documents": len(documents),
            "elapsed_s": round(elapsed, 1)
        }

    except Exception as e:
        st.logger.error("K2RAG_INDEX_FAIL err=%s", safe_str(e))
        return {
            "status": "FAILED",
            "error": safe_str(e)
        }


# ============================================================
# Integration with existing RAG
# ============================================================
def load_from_existing_rag():
    """
    기존 service.py의 RAG 데이터 로드
    - FAISS: st.RAG_STORE["index"]
    - BM25: service.BM25_INDEX
    - KG: service.KNOWLEDGE_GRAPH
    - Chunks: service.BM25_CORPUS
    """
    try:
        from rag import service

        loaded_count = 0

        # FAISS store: st.RAG_STORE에서 로드
        with st.RAG_LOCK:
            faiss_idx = st.RAG_STORE.get("index")
            if faiss_idx is not None:
                K2RAG_STATE["dense_store"] = faiss_idx
                st.logger.info("K2RAG_LOADED_DENSE_STORE from st.RAG_STORE")
                loaded_count += 1

        # BM25 store 공유
        if hasattr(service, 'BM25_INDEX') and service.BM25_INDEX is not None:
            K2RAG_STATE["sparse_store"] = service.BM25_INDEX
            st.logger.info("K2RAG_LOADED_SPARSE_STORE from service.BM25_INDEX")
            loaded_count += 1

        # Knowledge Graph 공유
        if hasattr(service, 'KNOWLEDGE_GRAPH') and service.KNOWLEDGE_GRAPH:
            K2RAG_STATE["knowledge_graph"] = service.KNOWLEDGE_GRAPH
            st.logger.info("K2RAG_LOADED_KG count=%d", len(service.KNOWLEDGE_GRAPH))
            loaded_count += 1

        # Chunks: BM25_CORPUS에서 로드
        if hasattr(service, 'BM25_CORPUS') and service.BM25_CORPUS:
            K2RAG_STATE["chunk_texts"] = service.BM25_CORPUS
            K2RAG_STATE["chunks"] = [{"text": t} for t in service.BM25_CORPUS]
            st.logger.info("K2RAG_LOADED_CHUNKS count=%d", len(service.BM25_CORPUS))
            loaded_count += 1

        # BM25_DOC_MAP이 있으면 더 상세한 정보 사용
        if hasattr(service, 'BM25_DOC_MAP') and service.BM25_DOC_MAP:
            K2RAG_STATE["chunks"] = service.BM25_DOC_MAP
            K2RAG_STATE["chunk_texts"] = [d.get("content", "") for d in service.BM25_DOC_MAP]

        if loaded_count > 0:
            K2RAG_STATE["initialized"] = True
            st.logger.info("K2RAG_LOAD_FROM_RAG_SUCCESS loaded=%d/4", loaded_count)
            return True
        else:
            st.logger.warning("K2RAG_LOAD_FROM_RAG_EMPTY - RAG 데이터가 없습니다. 먼저 /api/rag/reload 호출 필요")
            return False

    except Exception as e:
        st.logger.warning("K2RAG_LOAD_FROM_RAG_FAIL err=%s", safe_str(e))
        return False


# ============================================================
# API Functions
# ============================================================
def get_status() -> Dict[str, Any]:
    """K²RAG 상태 반환"""
    return {
        "initialized": K2RAG_STATE.get("initialized", False),
        "chunks_count": len(K2RAG_STATE.get("chunks", [])),
        "has_dense_store": K2RAG_STATE.get("dense_store") is not None,
        "has_sparse_store": K2RAG_STATE.get("sparse_store") is not None,
        "has_knowledge_graph": bool(K2RAG_STATE.get("knowledge_graph")),
        "summarizer_available": SUMMARIZER_AVAILABLE,
        "config": {
            "hybrid_lambda": K2RAG_CONFIG.hybrid_lambda,
            "top_k": K2RAG_CONFIG.top_k,
            "llm_model": K2RAG_CONFIG.llm_model
        }
    }


def update_config(config_updates: Dict[str, Any]) -> Dict[str, Any]:
    """K²RAG 설정 업데이트"""
    global K2RAG_CONFIG

    for key, value in config_updates.items():
        if hasattr(K2RAG_CONFIG, key):
            setattr(K2RAG_CONFIG, key, value)
            st.logger.info("K2RAG_CONFIG_UPDATED %s=%s", key, value)

    return {
        "status": "SUCCESS",
        "config": {
            "hybrid_lambda": K2RAG_CONFIG.hybrid_lambda,
            "top_k": K2RAG_CONFIG.top_k,
            "llm_model": K2RAG_CONFIG.llm_model,
            "use_summarization": K2RAG_CONFIG.use_summarization,
            "use_knowledge_graph": K2RAG_CONFIG.use_knowledge_graph
        }
    }


# ============================================================
# Initialization on import
# ============================================================
def _init_k2rag():
    """모듈 로드시 초기화"""
    st.logger.info("K2RAG_MODULE_INIT")

    # 기존 RAG 데이터 로드 시도
    load_from_existing_rag()


_init_k2rag()
