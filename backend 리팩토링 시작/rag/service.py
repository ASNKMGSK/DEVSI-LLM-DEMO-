"""
rag/service.py - RAG (Retrieval-Augmented Generation) 서비스
임베딩 인덱스 구축, 검색, 파일 관리

Advanced Features:
- Hybrid Search: BM25 + Vector (FAISS) 조합
- Reranking: Cross-Encoder 기반 재정렬
- Knowledge Graph: 간단한 Entity-Relation 추출
"""
import os
import re
import json
import time
import hashlib
import tempfile
import shutil
import logging
from typing import List, Any, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# pdfminer 경고 숨기기 (PDF 폰트 메타데이터 관련 - 기능에 영향 없음)
# pdfplumber import 전에 설정해야 함
for _pdflogger in ("pdfminer", "pdfminer.pdffont", "pdfminer.pdfinterp", "pdfminer.pdfpage"):
    logging.getLogger(_pdflogger).setLevel(logging.ERROR)

from core.utils import safe_str
import state as st

# ============================================================
# 선택적 import (없으면 RAG 비활성화)
# ============================================================
FAISS = None
OpenAIEmbeddings = None
Document = None
RecursiveCharacterTextSplitter = None

try:
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    pass

# SAR import (현재 파일에서는 사용하지 않지만 기존 코드 유지)
SARSingleNode = None
SAR_AVAILABLE = False
try:
    from recommenders.models.sar.sar_singlenode import SARSingleNode
    SAR_AVAILABLE = True
except Exception:
    pass

# ============================================================
# Hybrid Search: BM25 (Optional)
# ============================================================
BM25Okapi = None
BM25_AVAILABLE = False
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    pass

# ============================================================
# Reranking: Cross-Encoder (Optional)
# ============================================================
CrossEncoder = None
RERANKER_AVAILABLE = False
RERANKER_MODEL = None
try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    pass

# ============================================================
# Contextual Retrieval (Anthropic 2024)
# - 각 청크에 LLM으로 문서 맥락 요약 추가
# - 검색 정확도 대폭 향상
# ============================================================
OPENAI_CLIENT = None
CONTEXTUAL_CACHE: Dict[str, str] = {}  # chunk_hash -> contextual_prefix
CONTEXTUAL_CACHE_FILE = None  # 런타임에 설정됨
CONTEXTUAL_RETRIEVAL_ENABLED = True  # 활성화 여부
CONTEXTUAL_MODEL = "gpt-4o-mini"  # 저렴한 모델 사용

# 병렬 처리 설정
CONTEXTUAL_MAX_WORKERS = 5   # 동시 API 호출 수 (Rate limit으로 10→5 감소)
CONTEXTUAL_MAX_RETRIES = 3   # 429 에러 시 재시도 횟수
CONTEXTUAL_CACHE_LOCK = threading.Lock()  # 캐시 동시 접근 방지


def _get_openai_client():
    """Lazy initialization of OpenAI client"""
    global OPENAI_CLIENT
    if OPENAI_CLIENT is not None:
        return OPENAI_CLIENT

    try:
        from openai import OpenAI
        # state.py에서 관리하는 API 키 사용
        api_key = getattr(st, "OPENAI_API_KEY", None)
        if not api_key:
            st.logger.warning("OPENAI_API_KEY_NOT_SET in state.py")
            return None
        OPENAI_CLIENT = OpenAI(api_key=api_key)
        st.logger.info("OPENAI_CLIENT_INITIALIZED for Contextual Retrieval")
        return OPENAI_CLIENT
    except ImportError as e:
        st.logger.warning("OPENAI_IMPORT_FAIL err=%s (pip install openai)", safe_str(e))
        return None
    except Exception as e:
        st.logger.warning("OPENAI_CLIENT_INIT_FAIL err=%s", safe_str(e))
        return None

# ============================================================
# Parent-Child Chunking (검색 정확도 향상)
# ============================================================
# Child: 작은 청크 (300자) - 검색용 (정밀 매칭)
# Parent: 큰 청크 (1500자) - 반환용 (충분한 문맥)
PARENT_CHUNKS_STORE: Dict[str, Any] = {}  # parent_id -> Document
CHILD_TO_PARENT_MAP: Dict[str, str] = {}  # child_content_hash -> parent_id

# ============================================================
# Knowledge Graph (Simple Entity-Relation Extraction)
# ============================================================
KNOWLEDGE_GRAPH: Dict[str, List[Dict]] = {}  # entity -> relations


# ============================================================
# 내부 유틸
# ============================================================
def _sha1_text(s: str) -> str:
    try:
        return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()
    except Exception:
        return ""


def _clean_text_for_rag(txt: str) -> str:
    if not txt:
        return ""
    # 제어문자 제거 (개행/탭은 살림)
    txt = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", " ", txt)
    # 공백 정리
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()


# ============================================================
# 나무위키 PDF 전용 노이즈 제거 및 구조화 (RAG 품질 향상)
# ============================================================
def _clean_namuwiki_pdf_noise(txt: str) -> str:
    """
    나무위키 PDF에서 RAG 품질을 떨어뜨리는 노이즈 제거 + 구조화

    처리 대상:
    1. 각주 번호: [1], [14], [15] 등 제거
    2. 광고 문구: "포토샵 포함...", "구매하기" 등 제거
    3. UI 문구: "[ 펼치기 · 접기 ]", "분류:", "최근 수정 시각:" 등 제거
    4. 페이지 마커: <IMAGE FOR PAGE...>, <PARSED TEXT...> 제거
    5. 섹션 제목 정규화: "15.왕국 레벨" → "15. 왕국 레벨"
    6. 불릿 항목 표준화: "왕국 활동 - 설명" → "- 왕국 활동: 설명"
    """
    if not txt:
        return ""

    # 1. 각주 번호 제거: [1], [14], [15] 등
    txt = re.sub(r'\[\d+\]', '', txt)

    # 2. 광고/프로모션 문구 제거
    ad_patterns = [
        r'포토샵\s*포함.*?다운로드\s*받으세요\s*\.?',
        r'구매하기',
        r'어도비\s*\d+\s*캘린더.*?\.?',
        r'Adobe\s*\d+.*?download.*?\.?',
        r'Creative\s*Cloud.*?완성해보세요\.?',
        r'타이포그래피도.*?더보기',
        r'굿즈\s*디자인까지',
    ]
    for pattern in ad_patterns:
        txt = re.sub(pattern, '', txt, flags=re.IGNORECASE | re.DOTALL)

    # 3. 나무위키 UI 문구 제거
    ui_patterns = [
        r'\[\s*펼치기\s*[·•]\s*접기\s*\]',
        r'\[\s*확률표\s*펼치기\s*[·•]\s*접기\s*\]',
        r'최근\s*수정\s*시각:\s*[\d\-:\s]+',
        r'분류:[가-힣A-Za-z0-9\s/:]+',
        r'상위\s*문서:\s*[가-힣A-Za-z0-9:\s]+',
        r'자세한\s*내용은\s*[가-힣A-Za-z0-9:\s/]+문서를?\s*참고하십시오\.?',
        r'문서의?\s*r\d+\s*판',
        r'^목차\s*$',
    ]
    for pattern in ui_patterns:
        txt = re.sub(pattern, '', txt, flags=re.IGNORECASE | re.MULTILINE)

    # 4. 페이지 마커 제거
    txt = re.sub(r'<IMAGE\s+FOR\s+PAGE[^>]*>', '', txt)
    txt = re.sub(r'<PARSED\s+TEXT[^>]*>', '', txt)

    # 5. ★ 섹션 제목 정규화: "15.왕국" → "15. 왕국"
    # 숫자+점+한글(공백없이) → 숫자+점+공백+한글
    txt = re.sub(r'(\d+)\.([가-힣])', r'\1. \2', txt)

    # 6. ★ 불릿 항목 표준화 (줄 시작에서 "항목명 - 설명" 패턴)
    # "왕국 활동 - 물품 생산..." → "- 왕국 활동: 물품 생산..."
    def convert_bullet_line(match):
        item_name = match.group(1).strip()
        description = match.group(2).strip()
        return f"- {item_name}: {description}"

    txt = re.sub(
        r'^([가-힣]+(?:\s[가-힣]+)?(?:\s보상)?)\s*-\s*(.+)$',
        convert_bullet_line,
        txt,
        flags=re.MULTILINE
    )

    # 7. 연속 공백/줄바꿈 정리
    txt = re.sub(r'[ \t]+', ' ', txt)
    txt = re.sub(r'\n{3,}', '\n\n', txt)
    txt = re.sub(r'^\s+', '', txt, flags=re.MULTILINE)

    return txt.strip()


def _extract_location_relations_from_text(txt: str) -> List[Dict]:
    """
    나무위키 PDF 텍스트에서 대륙-지역 위치 관계 추출

    패턴:
    - "크리스피 대륙(Crispia)" 섹션 아래 나열된 왕국/지역들
    - "X 대륙에 위치" 패턴
    - 불릿 목록 구조

    Returns:
        [{"source": "골드치즈 왕국", "target": "크리스피 대륙", "type": "located_in"}, ...]
    """
    relations = []

    # 패턴 1: "X 대륙" 섹션 아래 나열된 지역들
    # 예: "크리스피 대륙(Crispia)[14]\n... 바닐라 왕국, 홀리베리 왕국, 다크카카오 왕국, 골드치즈 왕국..."
    continent_patterns = [
        (r'크리스피\s*대륙', '크리스피 대륙'),
        (r'홀그레인\s*대륙', '홀그레인 대륙'),
        (r'비스트이스트\s*대륙', '비스트이스트 대륙'),
        (r'슈가버그\s*랜드', '슈가버그 랜드'),
        (r'캔디스틱\s*제도', '캔디스틱 제도'),
    ]

    # 지역/왕국 이름 패턴
    location_patterns = [
        r'([가-힣]+\s?왕국)',           # 바닐라 왕국, 골드치즈 왕국
        r'([가-힣]+\s?공화국)',          # 크림 공화국
        r'([가-힣]+\s?제도)',            # 트로피컬 소다 제도
        r'([가-힣]+\s?산맥)',            # 자이언트 아이싱 산맥
        r'([가-힣]+\s?호수)',            # 밀키웨이 호수
        r'([가-힣]+\s?협곡)',            # 용의 협곡
        r'([가-힣]+\s?숲)',              # 비밀의 무화과 숲
        r'([가-힣]+\s?섬)',              # 두리안 섬
        r'([가-힣의]+\s?도시)',          # 마법사들의 도시
    ]

    # 대륙별 섹션에서 지역 추출
    for continent_pattern, continent_name in continent_patterns:
        # 해당 대륙 섹션 찾기
        match = re.search(continent_pattern, txt)
        if not match:
            continue

        # 섹션 시작점부터 다음 대륙/섹션까지의 텍스트
        start_pos = match.end()
        # 다음 대륙 또는 큰 섹션 제목까지
        next_section = re.search(r'\n(?:홀그레인|크리스피|비스트이스트|슈가버그|캔디스틱|비스트\s*이스트|\d+\.\s+[가-힣])', txt[start_pos:])
        end_pos = start_pos + next_section.start() if next_section else len(txt)

        section_text = txt[start_pos:end_pos]

        # 해당 섹션에서 지역 추출
        for loc_pattern in location_patterns:
            locations = re.findall(loc_pattern, section_text)
            for loc in locations:
                loc = loc.strip()
                if loc and len(loc) >= 3:  # 너무 짧은 것 제외
                    relations.append({
                        "source": loc,
                        "target": continent_name,
                        "type": "located_in"
                    })
                    # 역방향도 추가
                    relations.append({
                        "source": continent_name,
                        "target": loc,
                        "type": "contains"
                    })

    # 패턴 2: 명시적 "X에 위치" 패턴
    # "골드치즈 왕국이 이 대륙에 위치"
    explicit_patterns = [
        (r'([가-힣]+\s?왕국)[이가]\s*(?:이\s*)?대륙에\s*위치', None),  # 대륙명은 컨텍스트에서
        (r'([가-힣]+\s?왕국)이\s*([가-힣]+\s*대륙)에\s*위치', None),
        (r'([가-힣]+\s?왕국),\s*([가-힣]+\s?왕국),.*?이\s*대륙에\s*위치', None),
    ]

    # "고대의 네 왕국이(바닐라 왕국, 홀리베리 왕국, 다크카카오 왕국, 골드치즈 왕국)이 이 대륙에 위치"
    kingdom_list_match = re.search(
        r'네\s*왕국[이가]?\s*\(([^)]+)\)[이가]?\s*(?:이\s*)?대륙에\s*위치',
        txt
    )
    if kingdom_list_match:
        kingdoms_str = kingdom_list_match.group(1)
        kingdoms = re.findall(r'([가-힣]+\s?왕국)', kingdoms_str)
        for kingdom in kingdoms:
            relations.append({
                "source": kingdom.strip(),
                "target": "크리스피 대륙",  # 컨텍스트상 크리스피 대륙
                "type": "located_in"
            })

    # 중복 제거
    seen = set()
    unique_relations = []
    for rel in relations:
        key = (rel["source"], rel["target"], rel["type"])
        if key not in seen:
            seen.add(key)
            unique_relations.append(rel)

    return unique_relations


def _split_bullet_list_to_chunks(txt: str, section_title: str = "") -> List[Dict]:
    """
    불릿/목록 항목을 개별 chunk로 분리

    나무위키 PDF에서 "대륙 아래 왕국 목록" 같은 구조를
    1항목=1chunk로 분리하여 "A는 어느 대륙?" 질문 정확도 향상

    Args:
        txt: 섹션 텍스트
        section_title: 상위 섹션 제목 (메타데이터용)

    Returns:
        [{"content": "골드치즈 왕국(Gold Cheese Kingdom)",
          "metadata": {"type": "list_item", "parent_section": "크리스피 대륙", ...}}, ...]
    """
    chunks = []

    # 불릿/목록 패턴 감지
    # 나무위키 스타일: "왕국이름(English Name)" 형태가 연속으로 나옴
    list_item_patterns = [
        # "골드치즈 왕국(Gold Cheese Kingdom)" 형태
        r'([가-힣]+(?:\s[가-힣]+)?\s?(?:왕국|산맥|호수|협곡|숲|섬|도시|제도|마을|대륙))\s*\([A-Za-z\s]+\)',
        # "바닐라 왕국" 형태 (영문 없이)
        r'^([가-힣]+\s?(?:왕국|산맥|호수|협곡|숲|섬|도시|제도|마을))\s*$',
    ]

    lines = txt.split('\n')

    for line in lines:
        line = line.strip()
        if not line or len(line) < 3:
            continue

        # 목록 항목인지 확인
        for pattern in list_item_patterns:
            match = re.match(pattern, line, re.MULTILINE)
            if match:
                item_name = match.group(1).strip()
                chunks.append({
                    "content": line,
                    "metadata": {
                        "type": "list_item",
                        "item_name": item_name,
                        "parent_section": section_title,
                        "chunk_type": "bullet"
                    }
                })
                break

    return chunks


def _is_garbage_text(txt: str) -> bool:
    if not txt:
        return True
    t = txt.strip()
    if len(t) < 50:
        return True

    # 문자 다양도 너무 낮으면(깨진 PDF/목차/반복) 제거
    uniq = len(set(t))
    if uniq / max(1, len(t)) < 0.02:
        return True

    # 한글/영문/숫자 비율이 너무 낮으면(깨진 텍스트) 제거
    # 나무위키 PDF는 표/이모지/특수문자가 많아서 0.15로 완화
    meaningful = re.findall(r"[가-힣A-Za-z0-9]", t)
    if len(meaningful) / max(1, len(t)) < 0.15:
        return True

    return False


# ============================================================
# Contextual Retrieval: 캐싱 및 맥락 생성
# ============================================================
def _load_contextual_cache() -> Dict[str, str]:
    """캐시 파일에서 Contextual Prefix 로드"""
    global CONTEXTUAL_CACHE, CONTEXTUAL_CACHE_FILE

    if CONTEXTUAL_CACHE_FILE is None:
        CONTEXTUAL_CACHE_FILE = os.path.join(st.RAG_FAISS_DIR, "contextual_cache.json")

    try:
        if os.path.exists(CONTEXTUAL_CACHE_FILE):
            with open(CONTEXTUAL_CACHE_FILE, "r", encoding="utf-8") as f:
                CONTEXTUAL_CACHE = json.load(f)
                st.logger.info("CONTEXTUAL_CACHE_LOADED count=%d", len(CONTEXTUAL_CACHE))
    except Exception as e:
        st.logger.warning("CONTEXTUAL_CACHE_LOAD_FAIL err=%s", safe_str(e))
        CONTEXTUAL_CACHE = {}

    return CONTEXTUAL_CACHE


def _save_contextual_cache() -> None:
    """캐시를 파일로 저장"""
    global CONTEXTUAL_CACHE, CONTEXTUAL_CACHE_FILE

    if CONTEXTUAL_CACHE_FILE is None:
        CONTEXTUAL_CACHE_FILE = os.path.join(st.RAG_FAISS_DIR, "contextual_cache.json")

    try:
        os.makedirs(os.path.dirname(CONTEXTUAL_CACHE_FILE), exist_ok=True)
        with open(CONTEXTUAL_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(CONTEXTUAL_CACHE, f, ensure_ascii=False, indent=2)
        st.logger.info("CONTEXTUAL_CACHE_SAVED count=%d", len(CONTEXTUAL_CACHE))
    except Exception as e:
        st.logger.warning("CONTEXTUAL_CACHE_SAVE_FAIL err=%s", safe_str(e))


def _generate_contextual_prefix(
    doc_content: str,
    chunk_content: str,
    source: str,
    section_title: str = ""
) -> str:
    """
    Contextual Retrieval: LLM으로 청크의 문서 내 맥락 생성 (Anthropic 2024 기법)

    각 청크에 "이 청크가 전체 문서에서 어떤 위치/맥락인지" 설명 추가
    → BM25 + Vector 검색 정확도 대폭 향상

    Args:
        doc_content: 전체 문서 내용 (앞부분만 사용)
        chunk_content: 청크 내용
        source: 문서 출처
        section_title: 섹션 제목 (있으면)

    Returns:
        맥락 설명 문자열 (예: "이 청크는 클로티드 크림 쿠키의 평가 섹션으로...")
    """
    global CONTEXTUAL_CACHE

    if not CONTEXTUAL_RETRIEVAL_ENABLED:
        return ""

    # Lazy initialization
    client = _get_openai_client()
    if client is None:
        return ""

    # 캐시 키 생성 (청크 내용 기반)
    cache_key = _sha1_text(chunk_content)[:32]

    # 캐시에서 먼저 확인 (thread-safe)
    with CONTEXTUAL_CACHE_LOCK:
        if cache_key in CONTEXTUAL_CACHE:
            return CONTEXTUAL_CACHE[cache_key]

    # 문서 앞부분만 사용 (토큰 절약)
    doc_preview = doc_content[:6000] if len(doc_content) > 6000 else doc_content

    # 프롬프트 구성 (Anthropic 권장 형식 기반)
    prompt = f"""<document>
{doc_preview}
</document>

위 문서에서 추출한 청크입니다:

<chunk>
{chunk_content[:1000]}
</chunk>

{f'섹션: {section_title}' if section_title else ''}
문서 출처: {source}

이 청크가 전체 문서에서 어떤 맥락에 있는지 간결하게 설명해주세요.
검색 정확도 향상을 위해 핵심 키워드(인물명, 주제, 카테고리)를 포함해주세요.
1-2문장으로 답변하세요. 예: "이 청크는 [인물/주제]의 [섹션명]에서 [핵심 내용]을 설명합니다."
"""

    # Retry with exponential backoff (Rate limit 대응)
    for attempt in range(CONTEXTUAL_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=CONTEXTUAL_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0
            )

            contextual_prefix = response.choices[0].message.content.strip()

            # 캐시에 저장 (thread-safe)
            with CONTEXTUAL_CACHE_LOCK:
                CONTEXTUAL_CACHE[cache_key] = contextual_prefix

            return contextual_prefix

        except Exception as e:
            err_str = safe_str(e)
            # Rate limit (429) 또는 서버 에러 (5xx) 시 재시도
            if "429" in err_str or "500" in err_str or "502" in err_str or "503" in err_str:
                wait_time = (2 ** attempt) + 0.5  # 0.5, 2.5, 4.5초
                st.logger.warning("CONTEXTUAL_RATE_LIMIT attempt=%d/%d wait=%.1fs err=%s",
                                 attempt + 1, CONTEXTUAL_MAX_RETRIES, wait_time, err_str[:50])
                time.sleep(wait_time)
            else:
                st.logger.warning("CONTEXTUAL_PREFIX_FAIL err=%s", err_str)
                return ""

    st.logger.warning("CONTEXTUAL_PREFIX_MAX_RETRIES_EXCEEDED source=%s", source[:30])
    return ""


def _extract_key_nouns(text: str, top_k: int = 15) -> List[str]:
    """
    텍스트에서 핵심 명사 추출 (빈도 기반)

    BM25 키워드 검색 정확도 향상을 위해:
    - 한글 명사 (2~6자) 추출
    - 복합 명사 (공백 포함) 추출
    - 게임 용어, 재료명 등 중요 키워드 우선

    Args:
        text: 분석할 텍스트
        top_k: 추출할 최대 키워드 수

    Returns:
        핵심 명사 리스트 (빈도순)
    """
    if not text:
        return []

    # 1. 한글 명사 추출 (2~6자)
    nouns = re.findall(r'[가-힣]{2,6}', text)

    # 2. 복합 명사 추출 ("바다 희귀재료", "오로라 재료" 등)
    compound_nouns = re.findall(r'[가-힣]{2,6}\s+[가-힣]{2,6}', text)

    # 3. 게임 특화 패턴 ("~재료", "~아이템", "~쿠키" 등)
    game_terms = re.findall(r'[가-힣]{1,4}(?:재료|아이템|쿠키|스킬|토핑|왕국|대사)', text)

    # 불용어 (일반적인 조사, 접미사 등)
    stopwords = {
        '이다', '하다', '있다', '없다', '되다', '않다', '것이', '수가', '등의',
        '으로', '에서', '까지', '부터', '에게', '한다', '된다', '이며', '이고',
        '그리고', '하지만', '그러나', '따라서', '그래서', '때문에', '위해서',
        '경우', '통해', '대한', '관한', '대해', '관해', '사용', '기능', '설명',
    }

    # 빈도 계산
    from collections import Counter

    all_terms = nouns + compound_nouns + game_terms
    counter = Counter(all_terms)

    # 불용어 제거 및 정렬
    result = []
    seen = set()
    for term, count in counter.most_common(top_k * 3):
        term_clean = term.strip()
        if term_clean in stopwords:
            continue
        if term_clean in seen:
            continue
        if len(term_clean) < 2:
            continue
        seen.add(term_clean)
        result.append(term_clean)
        if len(result) >= top_k:
            break

    return result


# ============================================================
# ★ 구조 기반 청킹: 불릿/목록 → 1줄=1chunk
# ============================================================

# 불릿 패턴 정의 (섹션 제목과 구분)
BULLET_PATTERNS = [
    r'^[-•*○●◦▪▸►→]\s+',      # 일반 불릿: - • * ○ ● 등
    r'^[가-힣][.)]\s+',         # 한글 목록: 가. 나) 등
    r'^[a-zA-Z][.)]\s+',        # 알파벳 목록: a. b) 등
]
BULLET_REGEX = re.compile('|'.join(BULLET_PATTERNS))

# 섹션 제목 패턴 (불릿에서 제외)
SECTION_TITLE_PATTERN = re.compile(r'^\d+\.(?:\d+\.)*\s+.{2,50}$')


def _is_bullet_line(line: str) -> bool:
    """줄이 불릿/목록 항목인지 확인 (섹션 제목 제외)"""
    stripped = line.strip()

    # 섹션 제목 패턴이면 불릿 아님 (예: "15. 왕국 레벨")
    if SECTION_TITLE_PATTERN.match(stripped):
        return False

    return bool(BULLET_REGEX.match(stripped))


def _extract_bullet_blocks(text: str) -> List[Dict[str, Any]]:
    """
    텍스트에서 불릿/목록 블록과 일반 텍스트 블록을 분리

    ★ 핵심 개선: 불릿 제목 + 후속 설명 문단을 함께 묶음
    - "● 소생 마법" + 그 아래 설명 → 하나의 item으로
    - 다음 불릿이 나올 때까지 설명을 계속 병합

    Returns:
        [{"type": "bullet", "header": "베이킹 마법", "items": [
            {"title": "소생 마법", "description": "죽은 쿠키를 살리는..."},
            {"title": "봉인 마법", "description": "실패한 디저트를..."}
         ]},
         {"type": "prose", "content": "일반 서술형 텍스트..."}]
    """
    lines = text.split('\n')
    blocks: List[Dict[str, Any]] = []

    current_prose: List[str] = []
    current_header = ""
    current_bullet_items: List[Dict[str, str]] = []  # [{title, description}, ...]
    current_item_title = ""
    current_item_desc: List[str] = []
    in_bullet_block = False
    empty_line_count = 0  # 연속 빈 줄 카운트

    def save_current_item():
        """현재 불릿 항목 저장"""
        nonlocal current_item_title, current_item_desc
        if current_item_title:
            desc = '\n'.join(current_item_desc).strip()
            current_bullet_items.append({
                "title": current_item_title,
                "description": desc
            })
            current_item_title = ""
            current_item_desc = []

    def save_bullet_block():
        """현재 불릿 블록 저장"""
        nonlocal current_bullet_items, current_header, in_bullet_block
        save_current_item()
        if current_bullet_items:
            blocks.append({
                "type": "bullet",
                "header": current_header,
                "items": current_bullet_items
            })
        current_bullet_items = []
        current_header = ""
        in_bullet_block = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        if not stripped:
            empty_line_count += 1
            # 빈 줄 2개 연속 = 블록 종료
            if empty_line_count >= 2 and in_bullet_block:
                save_bullet_block()
            elif in_bullet_block and current_item_title:
                # 빈 줄 1개는 설명에 포함 (문단 구분)
                current_item_desc.append("")
            elif current_prose:
                current_prose.append(line)
            continue

        empty_line_count = 0  # 빈 줄 카운트 리셋

        if _is_bullet_line(stripped):
            # 새 불릿 항목 발견
            if not in_bullet_block:
                # 불릿 블록 시작: 이전 일반 텍스트 처리
                if current_prose:
                    # 마지막 비어있지 않은 줄을 헤더로 추출
                    for j in range(len(current_prose) - 1, -1, -1):
                        if current_prose[j].strip():
                            current_header = current_prose[j].strip()
                            current_prose = current_prose[:j]
                            break

                    # 나머지 일반 텍스트 저장
                    prose_text = '\n'.join(current_prose).strip()
                    if prose_text:
                        blocks.append({"type": "prose", "content": prose_text})
                    current_prose = []

                in_bullet_block = True
            else:
                # 이미 불릿 블록 안: 이전 항목 저장
                save_current_item()

            # 새 항목 시작
            current_item_title = BULLET_REGEX.sub('', stripped).strip()

        else:
            # 일반 텍스트
            if in_bullet_block and current_item_title:
                # ★ 핵심: 불릿 항목의 설명으로 병합
                current_item_desc.append(stripped)
            elif in_bullet_block and not current_item_title:
                # 불릿 블록이지만 아직 항목 없음 = 블록 종료
                save_bullet_block()
                current_prose.append(line)
            else:
                current_prose.append(line)

    # 마지막 블록 처리
    if in_bullet_block:
        save_bullet_block()
    elif current_prose:
        prose_text = '\n'.join(current_prose).strip()
        if prose_text:
            blocks.append({"type": "prose", "content": prose_text})

    return blocks


def _create_bullet_chunks(
    blocks: List[Dict[str, Any]],
    section_title: str,
    source: str,
    base_metadata: Dict[str, Any]
) -> List[Any]:
    """
    불릿 블록을 개별 청크로 변환

    ★ 개선: 불릿 제목 + 설명을 함께 1 청크로
    - {"title": "소생 마법", "description": "죽은 쿠키를..."} → 1 chunk

    Returns:
        List of Document objects
    """
    chunks: List[Any] = []

    for block in blocks:
        if block["type"] == "bullet":
            header = block.get("header", "")
            items = block.get("items", [])

            for item in items:
                # ★ 새 형식: item = {"title": ..., "description": ...}
                if isinstance(item, dict):
                    item_title = item.get("title", "")
                    item_desc = item.get("description", "")
                else:
                    # 이전 형식 호환 (문자열)
                    item_title = item
                    item_desc = ""

                # 청크 내용 구성: [섹션] + 헤더 + 제목 + 설명
                content_parts = []

                if section_title:
                    content_parts.append(f"[섹션: {section_title}]")
                if header:
                    content_parts.append(f"{header}:")

                # 제목 + 설명
                if item_desc:
                    content_parts.append(f"{item_title}: {item_desc}")
                else:
                    content_parts.append(item_title)

                chunk_content = '\n'.join(content_parts)

                # Document 생성
                chunk_meta = {
                    **base_metadata,
                    "source": source,
                    "section_title": section_title,
                    "chunk_type": "bullet_item",
                    "bullet_header": header,
                }
                chunks.append(Document(page_content=chunk_content, metadata=chunk_meta))

        else:  # prose
            # 일반 텍스트는 그대로 Document로 (나중에 splitter로 처리)
            prose_content = block.get("content", "")
            if prose_content and len(prose_content) >= 50:
                chunk_meta = {
                    **base_metadata,
                    "source": source,
                    "section_title": section_title,
                    "chunk_type": "prose",
                }
                chunks.append(Document(page_content=prose_content, metadata=chunk_meta))

    return chunks


def _split_by_sections(text: str, source: str = "") -> List[Tuple[str, str]]:
    """
    나무위키 스타일 문서를 섹션 단위로 분리 (핵심 정확도 향상 기법)

    "22. 광고를 통한 보상 수여" 같은 섹션이 다른 섹션과 합쳐지는 문제 해결

    Args:
        text: 원본 문서 텍스트
        source: 문서 출처

    Returns:
        List of (section_title, section_content) tuples
    """
    if not text:
        return [("", text)]

    # 섹션 패턴: "1. ", "22. ", "2.1. " 등으로 시작하는 줄
    # 나무위키 특성상 숫자+점+공백으로 섹션 구분
    section_pattern = re.compile(r'^(\d+\.(?:\d+\.)*\s*.+)$', re.MULTILINE)

    lines = text.split('\n')
    sections: List[Tuple[str, str]] = []
    current_title = ""
    current_content: List[str] = []

    for line in lines:
        # 섹션 제목인지 확인
        match = section_pattern.match(line.strip())
        if match and len(line.strip()) < 100:  # 섹션 제목은 보통 짧음
            # 이전 섹션 저장
            if current_content:
                content = '\n'.join(current_content).strip()
                if content:
                    sections.append((current_title, content))

            # 새 섹션 시작
            current_title = line.strip()
            current_content = [line]  # 제목도 내용에 포함
        else:
            current_content.append(line)

    # 마지막 섹션 저장
    if current_content:
        content = '\n'.join(current_content).strip()
        if content:
            sections.append((current_title, content))

    # 섹션이 없으면 원본 반환
    if not sections:
        return [("", text)]

    st.logger.info("SECTIONS_SPLIT source=%s sections=%d", source, len(sections))
    return sections


def _create_parent_child_chunks(
    docs: List[Any],
    parent_size: int = 3000,
    parent_overlap: int = 500,
    child_size: int = 500,
    child_overlap: int = 100,
    enable_contextual: bool = True  # ★ Contextual Retrieval 활성화 여부
) -> Tuple[List[Any], Dict[str, Any], Dict[str, str]]:
    """
    Parent-Child Chunking 구현 (섹션 경계 기반)

    ★ 핵심 개선: 먼저 문서를 섹션 단위로 분리한 후 청킹
    - "22. 광고를 통한 보상 수여" 같은 섹션이 항상 Parent 시작에 위치
    - 섹션 제목이 컨텍스트에 항상 포함됨

    1. 문서 → 섹션 분리 (숫자 기반)
    2. 섹션 → Parent 청크 (큰 단위, 전체 섹션 포함)
    3. Parent → Child 청크 (중간 단위, 섹션 제목+내용 함께)
    4. Child 청크에 parent_id 메타데이터 추가

    Args:
        docs: 원본 Document 리스트
        parent_size: Parent 청크 크기 (기본 3000자 - 희귀재료 같은 큰 섹션 포함)
        parent_overlap: Parent 청크 오버랩 (기본 500자)
        child_size: Child 청크 크기 (기본 500자 - 섹션 제목+내용 함께)
        child_overlap: Child 청크 오버랩 (기본 100자)

    Returns:
        child_chunks: Child 청크 리스트 (검색용, FAISS/BM25에 저장)
        parent_store: {parent_id: Document} (반환용)
        child_to_parent: {child_hash: parent_id} (매핑용)
    """
    if RecursiveCharacterTextSplitter is None:
        # Fallback: 원본 문서 그대로 반환
        parent_store = {}
        child_to_parent = {}
        for i, doc in enumerate(docs):
            pid = f"p_{i}"
            parent_store[pid] = doc
            child_to_parent[_sha1_text(safe_str(getattr(doc, "page_content", "")))[:16]] = pid
        return docs, parent_store, child_to_parent

    # ★ 1단계: 문서를 섹션 단위로 먼저 분리
    # "22. 광고를 통한 보상 수여" 같은 섹션이 별도 Document가 됨
    section_docs: List[Any] = []
    bullet_child_chunks: List[Any] = []  # ★ 불릿 항목 청크 (1항목=1청크)

    for doc in docs:
        content = safe_str(getattr(doc, "page_content", ""))
        metadata = getattr(doc, "metadata", {})
        source = metadata.get("source", "")

        sections = _split_by_sections(content, source)
        for section_title, section_content in sections:
            if not section_content or len(section_content.strip()) < 50:
                continue

            # ★ 구조 기반 청킹: 불릿/목록 블록 추출
            blocks = _extract_bullet_blocks(section_content)

            # 불릿 항목이 있는지 확인
            has_bullets = any(b["type"] == "bullet" for b in blocks)

            if has_bullets:
                # ★ 불릿 항목: 1항목 = 1청크로 직접 생성
                bullet_chunks = _create_bullet_chunks(blocks, section_title, source, metadata)
                bullet_child_chunks.extend(bullet_chunks)

                # 일반 텍스트(prose)만 section_docs에 추가 (Parent-Child 청킹용)
                for block in blocks:
                    if block["type"] == "prose":
                        prose_content = block.get("content", "")
                        if prose_content and len(prose_content) >= 100:
                            new_meta = {**metadata, "section_title": section_title}
                            section_docs.append(Document(page_content=prose_content, metadata=new_meta))
            else:
                # 불릿 없음: 기존 방식으로 처리
                new_meta = {**metadata, "section_title": section_title}
                section_docs.append(Document(page_content=section_content, metadata=new_meta))

    # 섹션 분리된 문서가 없으면 원본 사용
    if not section_docs:
        section_docs = docs

    st.logger.info("SECTION_DOCS_CREATED original=%d sections=%d bullet_items=%d",
                   len(docs), len(section_docs), len(bullet_child_chunks))

    # 한국어/나무위키 스타일 구분자
    separators = [
        "\n## ",       # 마크다운 헤더
        "\n### ",
        "\n#### ",
        "\n\n",        # 빈 줄 (문단 구분)
        "\n",          # 일반 줄바꿈
        ". ",          # 문장 구분
        " ",           # 단어 구분
    ]

    try:
        # 2단계: Parent 청크 생성 (문맥 보존용) - section_docs 사용
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size,
            chunk_overlap=parent_overlap,
            separators=separators,
            length_function=len,
        )
        parent_chunks = parent_splitter.split_documents(section_docs)  # ★ section_docs 사용

        # 2단계: Child 청크 생성 (검색용)
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size,
            chunk_overlap=child_overlap,
            separators=separators,
            length_function=len,
        )

        parent_store: Dict[str, Any] = {}
        child_to_parent: Dict[str, str] = {}
        all_child_chunks: List[Any] = []

        # ★ Contextual Retrieval: enable_contextual=True일 때만 활성화
        contextual_generated_count = 0
        contextual_client = None

        if enable_contextual and CONTEXTUAL_RETRIEVAL_ENABLED:
            _load_contextual_cache()
            contextual_client = _get_openai_client()
            st.logger.info("CONTEXTUAL_RETRIEVAL_STATUS enabled=%s client=%s workers=%d",
                          True, "OK" if contextual_client is not None else "None",
                          CONTEXTUAL_MAX_WORKERS)
        else:
            st.logger.info("CONTEXTUAL_RETRIEVAL_SKIPPED enable_contextual=%s global_enabled=%s",
                          enable_contextual, CONTEXTUAL_RETRIEVAL_ENABLED)

        # ============================================================
        # ★ 1단계: 모든 Child 정보 수집 (병렬 처리 준비)
        # ============================================================
        child_infos: List[Dict[str, Any]] = []  # 모든 child 정보 저장

        for i, parent in enumerate(parent_chunks):
            parent_id = f"p_{i}"
            parent_content = safe_str(getattr(parent, "page_content", ""))
            parent_metadata = getattr(parent, "metadata", {})
            source = parent_metadata.get("source", "")

            # Parent 저장 (문서 제목 포함해서 문맥 강화)
            # 핵심: Parent 내용 앞에 문서 출처를 추가하여 검색 시 문맥 제공
            contextual_content = f"[문서: {source}]\n\n{parent_content}"
            parent_store[parent_id] = Document(
                page_content=contextual_content,
                metadata={**parent_metadata, "parent_id": parent_id}
            )

            # ★ 개선된 컨텍스트 추출: Parent 첫 부분을 Child에 포함
            # 섹션 분리 단계에서 이미 섹션 제목을 메타데이터에 저장함

            # 1. 섹션 제목: 메타데이터에서 먼저 가져오기 (섹션 분리 단계에서 저장됨)
            section_title = parent_metadata.get("section_title", "")

            # 메타데이터에 없으면 기존 방식으로 추출 (fallback)
            if not section_title:
                for line in parent_content.strip().split('\n')[:15]:  # 첫 15줄 검사
                    line = line.strip()
                    if not line or len(line) > 100:
                        continue
                    # 다양한 섹션 패턴:
                    # - "14. 희귀재료" 또는 "14.희귀재료"
                    # - "2.1. 상세" 또는 "2.1.상세"
                    # - "## 섹션명"
                    if re.match(r'^\d+\.\d*\.?\s*\S', line):
                        section_title = line
                        break
                    elif line.startswith('## ') or line.startswith('### '):
                        section_title = line.lstrip('#').strip()
                        break

            # 2. Parent의 첫 500자를 컨텍스트로 추출 (바다 희귀재료 등 하위 섹션 포함)
            parent_first = parent_content.strip()[:500].replace('\n', ' ').strip()
            # 컨텍스트 길이 확대: 100자 → 300자 (더 많은 하위 섹션 포함)
            parent_context = parent_first[:300] if len(parent_first) > 300 else parent_first

            # 3. ★ 서브섹션 키워드 추출 (정확도 향상 핵심 기법)
            # "바다 희귀재료", "오로라 재료" 같은 하위 섹션 제목 추출
            subsection_keywords = []
            for line in parent_content.strip().split('\n')[:30]:  # 첫 30줄 검사
                line = line.strip()
                if not line or len(line) > 50:
                    continue
                # 서브섹션 패턴: "바다 희귀재료", "오로라 재료", "일반 재료" 등
                # - 한글 2~10자 + 공백 + 한글 (복합 명사)
                # - 또는 단독 명사 (재료, 아이템 등)
                if re.match(r'^[가-힣]{2,10}\s+[가-힣]{2,10}$', line):
                    subsection_keywords.append(line)
                # "19.1. 바다 희귀재료" 형태도 처리
                elif re.match(r'^\d+\.\d+\.?\s*(.+)$', line):
                    match = re.match(r'^\d+\.\d+\.?\s*(.+)$', line)
                    if match:
                        subsection_keywords.append(match.group(1).strip())

            # 4. ★ 핵심 명사 추출 (BM25 키워드 매칭 강화)
            # Parent 내 주요 키워드 자동 추출 (빈도 기반)
            key_nouns = _extract_key_nouns(parent_content)
            keywords_tag = f"[키워드: {', '.join(key_nouns[:10])}]" if key_nouns else ""
            subsections_tag = f"[하위섹션: {', '.join(subsection_keywords[:5])}]" if subsection_keywords else ""

            # Parent를 Child로 분할
            try:
                temp_doc = Document(page_content=parent_content, metadata=parent_metadata)
                children = child_splitter.split_documents([temp_doc])

                for child in children:
                    child_content = safe_str(getattr(child, "page_content", ""))
                    if not child_content or len(child_content) < 30:
                        continue

                    # ★ Child 정보 저장 (병렬 처리용)
                    child_infos.append({
                        "parent_id": parent_id,
                        "parent_content": parent_content,
                        "parent_metadata": parent_metadata,
                        "child_content": child_content,
                        "source": source,
                        "section_title": section_title,
                        "subsections_tag": subsections_tag,
                        "keywords_tag": keywords_tag,
                        "parent_context": parent_context,
                    })

            except Exception as e:
                st.logger.warning("PARENT_CHILD_SPLIT_FAIL parent=%s err=%s", parent_id, safe_str(e))
                # Fallback: Parent 자체를 Child로 사용
                child_hash = _sha1_text(parent_content)[:16]
                child_to_parent[child_hash] = parent_id
                all_child_chunks.append(parent)

        # ============================================================
        # ★ 2단계: 병렬로 Contextual Prefix 생성 (10개 동시)
        # ============================================================
        # enable_contextual=True이고 클라이언트가 있을 때만 실행
        contextual_prefixes: Dict[int, str] = {}  # index -> prefix

        if enable_contextual and contextual_client is not None:
            st.logger.info("CONTEXTUAL_PARALLEL_START total_children=%d workers=%d",
                          len(child_infos), CONTEXTUAL_MAX_WORKERS)

            def process_child(idx: int) -> Tuple[int, str]:
                """Worker function for parallel processing"""
                info = child_infos[idx]
                prefix = _generate_contextual_prefix(
                    doc_content=info["parent_content"],
                    chunk_content=info["child_content"],
                    source=info["source"],
                    section_title=info["section_title"]
                )
                return idx, prefix

            # ThreadPoolExecutor로 병렬 처리
            with ThreadPoolExecutor(max_workers=CONTEXTUAL_MAX_WORKERS) as executor:
                futures = {executor.submit(process_child, i): i for i in range(len(child_infos))}

                completed = 0
                for future in as_completed(futures):
                    idx, prefix = future.result()
                    contextual_prefixes[idx] = prefix
                    if prefix:
                        contextual_generated_count += 1

                    completed += 1
                    # 진행 상황 로깅 (100개마다)
                    if completed % 100 == 0 or completed == len(child_infos):
                        st.logger.info("CONTEXTUAL_PROGRESS %d/%d (%.1f%%)",
                                      completed, len(child_infos),
                                      100.0 * completed / max(1, len(child_infos)))
        else:
            st.logger.info("CONTEXTUAL_PARALLEL_SKIPPED total_children=%d", len(child_infos))

        # ============================================================
        # ★ 3단계: Document 조합
        # ============================================================
        for idx, info in enumerate(child_infos):
            section_title = info["section_title"]
            subsections_tag = info["subsections_tag"]
            keywords_tag = info["keywords_tag"]
            parent_context = info["parent_context"]
            source = info["source"]
            child_content = info["child_content"]
            parent_id = info["parent_id"]
            parent_metadata = info["parent_metadata"]

            # 태그 생성
            tags = []

            # ★ 섹션 제목 부스팅: 3번 반복하여 BM25 가중치 대폭 증가
            if section_title:
                pure_title = re.sub(r'^\d+\.[\d.]*\s*', '', section_title).strip()
                tags.append(f"[섹션: {section_title}]")
                tags.append(f"[섹션제목: {section_title}]")
                tags.append(f"[제목: {pure_title}]")
                if pure_title and pure_title != section_title:
                    tags.append(f"[주제: {pure_title}]")

            if subsections_tag:
                tags.append(subsections_tag)
            if keywords_tag:
                tags.append(keywords_tag)
            tags.append(f"[컨텍스트: {parent_context}]")
            tags.append(f"[문서: {source}]")

            # Contextual Prefix 추가
            contextual_prefix = contextual_prefixes.get(idx, "")
            if contextual_prefix:
                tags.insert(0, f"[맥락: {contextual_prefix}]")

            contextual_child_content = " ".join(tags) + " " + child_content

            # Child 메타데이터에 parent_id 추가
            child_meta = {**parent_metadata, "parent_id": parent_id}
            new_child = Document(page_content=contextual_child_content, metadata=child_meta)
            all_child_chunks.append(new_child)

            # Child-Parent 매핑 저장
            child_hash = _sha1_text(contextual_child_content)[:16]
            child_to_parent[child_hash] = parent_id

        # ★ Contextual Retrieval: 캐시 저장
        if contextual_generated_count > 0:
            _save_contextual_cache()

        # ★ 불릿 청크 추가 (1항목=1청크로 만든 것들)
        # Parent가 없는 독립 청크이므로 별도 parent_id 할당
        for i, bullet_chunk in enumerate(bullet_child_chunks):
            bullet_parent_id = f"bullet_{i}"
            # 불릿 청크는 자기 자신이 Parent (짧으므로 전체가 컨텍스트)
            parent_store[bullet_parent_id] = bullet_chunk
            child_hash = _sha1_text(safe_str(getattr(bullet_chunk, "page_content", "")))[:16]
            child_to_parent[child_hash] = bullet_parent_id
            # 메타데이터에 parent_id 추가
            bullet_chunk.metadata["parent_id"] = bullet_parent_id
            all_child_chunks.append(bullet_chunk)

        st.logger.info("PARENT_CHILD_CHUNKS_CREATED parents=%d children=%d bullet=%d contextual=%d",
                       len(parent_store), len(all_child_chunks), len(bullet_child_chunks), contextual_generated_count)

        return all_child_chunks, parent_store, child_to_parent

    except Exception as e:
        st.logger.warning("PARENT_CHILD_CHUNK_FAIL err=%s", safe_str(e))
        # Fallback: 기존 방식
        parent_store = {}
        child_to_parent = {}
        for i, doc in enumerate(docs):
            pid = f"p_{i}"
            parent_store[pid] = doc
            child_to_parent[_sha1_text(safe_str(getattr(doc, "page_content", "")))[:16]] = pid
        return docs, parent_store, child_to_parent


def _rag_list_files() -> List[str]:
    files: List[str] = []
    try:
        os.makedirs(st.RAG_DOCS_DIR, exist_ok=True)
        for root, _, names in os.walk(st.RAG_DOCS_DIR):
            for n in names:
                ext = os.path.splitext(n)[1].lower()
                if ext in st.RAG_ALLOWED_EXTS:
                    files.append(os.path.join(root, n))
    except Exception:
        return []
    return sorted(list(set(files)))


def _rag_files_fingerprint(paths: List[str]) -> str:
    parts: List[str] = []
    for p in paths:
        try:
            s = os.stat(p)
            parts.append(f"{os.path.relpath(p, st.RAG_DOCS_DIR)}|{s.st_size}|{int(s.st_mtime)}")
        except Exception:
            parts.append(f"{os.path.relpath(p, st.RAG_DOCS_DIR)}|ERR")
    return _sha1_text("\n".join(parts))


def _reorder_namuwiki_dialogue_labels(text: str) -> str:
    """
    나무위키 PDF 후처리: 라벨-내용 순서 재정렬

    나무위키 PDF 구조 (시각적):
        ┌─────────────────────────┐
        │ "승리는 잠시만 즐기겠습니다." │  ← 내용이 먼저
        │ "여러분들의 승리나 다름없습니다." │
        ├─────────────────────────┤
        │      승리 시 대사         │  ← 라벨이 아래에
        └─────────────────────────┘

    추출된 텍스트 순서:
        "승리는 잠시만 즐기겠습니다."
        "여러분들의 승리나 다름없습니다."
        승리 시 대사              ← 라벨이 내용 뒤에 옴
        "실수가 있었습니다."       ← 이것은 패배 대사인데 위치상 승리 뒤에 있음
        "패배를 인정합니다."
        패배 시 대사

    이 함수는 라벨과 그 직전의 인용구(대사)를 찾아서 재정렬:
        승리 시 대사
        "승리는 잠시만 즐기겠습니다."
        "여러분들의 승리나 다름없습니다."
        패배 시 대사
        "실수가 있었습니다."
        "패배를 인정합니다."
    """
    if not text:
        return text

    # 대사 라벨 패턴 (나무위키에서 사용하는 형식)
    dialogue_labels = [
        "승리 시 대사", "패배 시 대사", "해금 시 대사", "해금시 대사",
        "전투 팀 편성 시 대사", "전투 편성 시 대사", "팀 편성 시 대사",
        "스킬 시전 시 대사", "스킬 사용 시 대사",
        "승급시 대사", "승급 시 대사", "초월 승급시 대사",
        "필드 대사", "캐릭터창 대사", "레벨업 시 대사",
        "만우절 이벤트 대사", "이벤트 대사",
    ]

    lines = text.split('\n')
    result_lines = []
    pending_quotes = []  # 라벨을 만나기 전 수집한 인용구들

    for line in lines:
        stripped = line.strip()

        # 이 줄이 대사 라벨인지 확인
        is_label = False
        for label in dialogue_labels:
            if label in stripped and len(stripped) < 50:  # 라벨은 보통 짧음
                is_label = True
                break

        if is_label:
            # 라벨을 만남 → 라벨을 먼저 출력하고, 그 뒤에 수집된 인용구들
            result_lines.append(stripped)
            result_lines.extend(pending_quotes)
            pending_quotes = []
        elif stripped.startswith('"') or stripped.startswith('"') or stripped.startswith("'"):
            # 인용구(대사)는 일단 보류
            pending_quotes.append(stripped)
        else:
            # 일반 텍스트: 보류 중인 인용구가 있으면 먼저 출력 (라벨 없이 끝난 경우)
            if pending_quotes:
                result_lines.extend(pending_quotes)
                pending_quotes = []
            result_lines.append(stripped)

    # 마지막에 남은 인용구 처리
    if pending_quotes:
        result_lines.extend(pending_quotes)

    return '\n'.join(result_lines)


def _extract_text_from_pdf(path: str) -> str:
    """
    PDF에서 텍스트 추출 (pymupdf 우선 사용)

    pymupdf(fitz)는 더 나은 텍스트 추출을 제공:
    - 레이아웃 보존 옵션
    - 더 정확한 텍스트 순서
    - 복잡한 PDF 구조 처리
    """
    raw_text = ""

    # 1차 시도: pymupdf (fitz) - 가장 강력한 PDF 파서
    try:
        import fitz  # pymupdf

        text_parts = []
        with fitz.open(path) as doc:
            for page in doc:
                # 텍스트 추출 (레이아웃 보존)
                page_text = page.get_text("text")
                if page_text:
                    text_parts.append(page_text)

        raw_text = "\n".join(text_parts).strip()
        if raw_text:
            st.logger.info("PDF_EXTRACTED_PYMUPDF path=%s chars=%d",
                          os.path.basename(path), len(raw_text))

    except ImportError:
        st.logger.debug("pymupdf not available, falling back to pypdf")
    except Exception as e:
        st.logger.warning("PYMUPDF_FAIL path=%s err=%s", path, safe_str(e))

    # 2차 시도: pypdf/PyPDF2 (fallback)
    if not raw_text:
        try:
            try:
                from pypdf import PdfReader
            except ImportError:
                try:
                    from PyPDF2 import PdfReader
                except ImportError:
                    return ""
            reader = PdfReader(path)
            text_parts = []
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
            raw_text = "\n".join(text_parts).strip()
            if raw_text:
                st.logger.info("PDF_EXTRACTED_PYPDF path=%s chars=%d",
                              os.path.basename(path), len(raw_text))
        except Exception as e:
            st.logger.warning("PYPDF_FAIL path=%s err=%s", path, safe_str(e))
            return ""

    if not raw_text:
        return ""

    # 나무위키 PDF 후처리: 라벨-내용 순서 재정렬
    result = _reorder_namuwiki_dialogue_labels(raw_text)
    st.logger.info("PDF_REORDERED path=%s before=%d after=%d",
                  os.path.basename(path), len(raw_text), len(result))

    # 나무위키 PDF 노이즈 제거 (광고, 각주, UI 문구 등)
    result = _clean_namuwiki_pdf_noise(result)
    st.logger.info("PDF_NOISE_CLEANED path=%s chars=%d",
                  os.path.basename(path), len(result))

    return result


def _rag_read_file(path: str) -> str:
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            txt = _extract_text_from_pdf(path)
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()

        txt = (txt or "").strip()
        if len(txt) > st.RAG_MAX_DOC_CHARS:
            txt = txt[:st.RAG_MAX_DOC_CHARS]

        txt = _clean_text_for_rag(txt)
        if _is_garbage_text(txt):
            st.logger.warning("RAG_SKIP_GARBAGE path=%s len=%d", os.path.basename(path), len(txt or ""))
            return ""
        return txt
    except Exception:
        return ""


def _make_embeddings(api_key: str):
    if OpenAIEmbeddings is None:
        return None
    k = (api_key or "").strip()
    try:
        return OpenAIEmbeddings(model=st.RAG_EMBED_MODEL, openai_api_key=k)
    except TypeError:
        try:
            return OpenAIEmbeddings(model=st.RAG_EMBED_MODEL, api_key=k)
        except TypeError:
            try:
                return OpenAIEmbeddings(model=st.RAG_EMBED_MODEL)
            except Exception:
                return None
    except Exception:
        return None


def _rag_load_state_file() -> dict:
    try:
        if not os.path.exists(st.RAG_STATE_FILE):
            return {}
        with open(st.RAG_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _rag_save_state_file(payload: dict) -> None:
    try:
        os.makedirs(st.RAG_FAISS_DIR, exist_ok=True)
        with open(st.RAG_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ============================================================
# BM25 인덱스 관리
# ============================================================
BM25_INDEX: Optional[Any] = None
BM25_CORPUS: List[str] = []
BM25_DOC_MAP: List[Dict] = []  # BM25 corpus index -> document metadata


def _tokenize_korean(text: str) -> List[str]:
    """한국어/영어 토큰화 (간단한 whitespace + 한글 음절 분리)"""
    if not text:
        return []
    # 공백 기준 토큰화
    tokens = text.lower().split()
    # 추가: 한글 음절 단위로도 분리 (형태소 분석기 없이 단순화)
    result = []
    for tok in tokens:
        result.append(tok)
        # 한글이면 2글자 이상일 때 ngram 추가
        if re.search(r'[가-힣]', tok) and len(tok) >= 2:
            for i in range(len(tok) - 1):
                result.append(tok[i:i+2])
    return result


def _build_bm25_index(chunks: List[Any]) -> bool:
    """BM25 인덱스 빌드 (Parent-Child 지원)"""
    global BM25_INDEX, BM25_CORPUS, BM25_DOC_MAP

    if not BM25_AVAILABLE or BM25Okapi is None:
        return False

    try:
        BM25_CORPUS = []
        BM25_DOC_MAP = []

        for chunk in chunks:
            try:
                content = safe_str(getattr(chunk, "page_content", ""))
                metadata = getattr(chunk, "metadata", {})
                if content:
                    BM25_CORPUS.append(content)
                    BM25_DOC_MAP.append({
                        "content": content,
                        "source": metadata.get("source", ""),
                        "parent_id": metadata.get("parent_id", ""),  # Parent-Child 지원
                    })
            except Exception:
                continue

        if not BM25_CORPUS:
            return False

        # BM25 인덱스 생성
        tokenized_corpus = [_tokenize_korean(doc) for doc in BM25_CORPUS]
        BM25_INDEX = BM25Okapi(tokenized_corpus)
        st.logger.info("BM25_INDEX_BUILT docs=%d", len(BM25_CORPUS))
        return True
    except Exception as e:
        st.logger.warning("BM25_BUILD_FAIL err=%s", safe_str(e))
        return False


def _get_parent_content(parent_id: str, fallback_content: str = "") -> str:
    """
    Parent ID로 Parent 청크의 content를 조회

    Args:
        parent_id: Parent 청크 ID
        fallback_content: Parent가 없으면 반환할 기본 content

    Returns:
        Parent 청크의 content (또는 fallback)
    """
    global PARENT_CHUNKS_STORE

    if not parent_id or not PARENT_CHUNKS_STORE:
        return fallback_content

    parent = PARENT_CHUNKS_STORE.get(parent_id)
    if parent is None:
        return fallback_content

    try:
        parent_content = safe_str(getattr(parent, "page_content", ""))
        return parent_content if parent_content else fallback_content
    except Exception:
        return fallback_content


def _bm25_search(query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
    """BM25 검색 (키워드 기반) - Parent-Child 지원"""
    global BM25_INDEX, BM25_DOC_MAP

    if BM25_INDEX is None or not BM25_DOC_MAP:
        return []

    try:
        tokenized_query = _tokenize_korean(query)
        scores = BM25_INDEX.get_scores(tokenized_query)

        # 상위 top_k 결과
        scored_docs = list(zip(range(len(scores)), scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        results = []
        seen_parents = set()  # 같은 Parent 중복 방지

        for idx, score in scored_docs:
            if score <= 0:
                continue
            if len(results) >= top_k:
                break

            doc = BM25_DOC_MAP[idx]
            child_content = doc.get("content", "")
            source = doc.get("source", "")
            parent_id = doc.get("parent_id", "")

            # Parent-Child: Parent content 반환 (중복 방지)
            if parent_id and parent_id not in seen_parents:
                parent_content = _get_parent_content(parent_id, child_content)
                seen_parents.add(parent_id)
                content = parent_content
            elif not parent_id:
                content = child_content
            else:
                continue  # 이미 본 Parent, 스킵

            results.append((
                {
                    "content": content,
                    "source": source,
                    "parent_id": parent_id,
                    "matched_child": child_content[:100],  # 검색에 매칭된 Child
                },
                score
            ))

        return results
    except Exception as e:
        st.logger.warning("BM25_SEARCH_FAIL err=%s", safe_str(e))
        return []


# ============================================================
# Cross-Encoder Reranking (BGE Reranker - 한국어/다국어 지원)
# ============================================================
# BGE Reranker는 임베딩 업그레이드보다 성능 향상 효과가 더 큼 (20-35%)
# 참고: https://github.com/FlagOpen/FlagEmbedding

def _get_reranker():
    """Reranker 모델 로드 (Lazy Loading) - BGE Reranker v2 M3 (다국어)"""
    global RERANKER_MODEL

    if not RERANKER_AVAILABLE or CrossEncoder is None:
        return None

    if RERANKER_MODEL is not None:
        return RERANKER_MODEL

    # 모델 우선순위: BGE 다국어 > BGE large > ms-marco (영어)
    models_to_try = [
        ('BAAI/bge-reranker-v2-m3', 1024),      # 다국어 최고 성능
        ('BAAI/bge-reranker-large', 512),       # 영어 기반, 다국어 가능
        ('cross-encoder/ms-marco-MiniLM-L-6-v2', 512),  # Fallback
    ]

    for model_name, max_len in models_to_try:
        try:
            RERANKER_MODEL = CrossEncoder(model_name, max_length=max_len)
            st.logger.info("RERANKER_LOADED model=%s max_length=%d", model_name, max_len)
            return RERANKER_MODEL
        except Exception as e:
            st.logger.warning("RERANKER_TRY_FAIL model=%s err=%s", model_name, safe_str(e))
            continue

    st.logger.warning("RERANKER_ALL_MODELS_FAILED")
    return None


def _rerank_results(query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Cross-Encoder로 결과 재정렬 (BGE Reranker)

    검색 파이프라인:
    1. BM25 + Vector에서 넉넉하게 가져옴 (retrieval_k = top_k * 4)
    2. Reranker로 정밀 재정렬
    3. 최종 top_k개 반환

    이 방식이 임베딩 모델 업그레이드보다 비용 대비 효과가 좋음
    """
    reranker = _get_reranker()
    if reranker is None or not results:
        return results[:top_k]

    try:
        # query-document 쌍 생성 (더 긴 컨텍스트 사용)
        pairs = [(query, r.get("content", "")[:800]) for r in results]

        # Cross-encoder 점수 계산 (BGE Reranker는 높을수록 관련성 높음)
        scores = reranker.predict(pairs)

        # 점수로 정렬
        scored_results = list(zip(results, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        reranked = []
        for r, score in scored_results[:top_k]:
            r["rerank_score"] = round(float(score), 4)
            reranked.append(r)

        st.logger.info("RERANK_DONE query=%s input=%d output=%d top_score=%.3f",
                       query[:30], len(results), len(reranked),
                       reranked[0]["rerank_score"] if reranked else 0)

        return reranked
    except Exception as e:
        st.logger.warning("RERANK_FAIL err=%s", safe_str(e))
        return results[:top_k]


# ============================================================
# Hybrid Search (BM25 + Vector Fusion)
# ============================================================
def _reciprocal_rank_fusion(
    bm25_results: List[Tuple[Dict, float]],
    vector_results: List[Tuple[Dict, float]],
    k: int = 60,
    bm25_weight: float = 1.5,    # BM25 가중치 (키워드 매칭 강화)
    vector_weight: float = 1.0   # Vector 가중치
) -> List[Dict]:
    """
    Reciprocal Rank Fusion (RRF) - BM25와 Vector 결과 병합
    RRF score = sum(weight / (k + rank))

    BM25 가중치를 높여서 정확한 키워드 매칭 결과를 우선시합니다.
    예: "스킬"이라는 정확한 단어가 있는 문서가 더 높은 점수를 받음
    """
    fusion_scores: Dict[str, Dict] = {}

    # BM25 결과 처리 (가중치 적용)
    for rank, (doc, score) in enumerate(bm25_results):
        key = doc.get("content", "")[:100]  # 내용 해시로 중복 방지
        if key not in fusion_scores:
            fusion_scores[key] = {
                "doc": doc,
                "bm25_score": score,
                "vector_score": 0.0,
                "rrf_score": 0.0,
            }
        fusion_scores[key]["bm25_score"] = score
        fusion_scores[key]["rrf_score"] += bm25_weight / (k + rank + 1)

    # Vector 결과 처리 (score = distance, 낮을수록 좋음)
    for rank, (doc, dist) in enumerate(vector_results):
        key = doc.get("content", "")[:100]
        if key not in fusion_scores:
            fusion_scores[key] = {
                "doc": doc,
                "bm25_score": 0.0,
                "vector_score": 1.0 / (1.0 + dist),  # distance -> similarity
                "rrf_score": 0.0,
            }
        fusion_scores[key]["vector_score"] = 1.0 / (1.0 + dist)
        fusion_scores[key]["rrf_score"] += vector_weight / (k + rank + 1)

    # RRF 점수로 정렬
    sorted_results = sorted(
        fusion_scores.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )

    return [
        {
            **item["doc"],
            "bm25_score": round(item["bm25_score"], 4),
            "vector_score": round(item["vector_score"], 4),
            "fusion_score": round(item["rrf_score"], 4),
        }
        for item in sorted_results
    ]


# ============================================================
# Knowledge Graph (Simple Entity-Relation Extraction)
# ============================================================
def _extract_entities_simple(text: str) -> List[str]:
    """쿠키런 킹덤 개체명 추출 (정규식 기반)"""
    entities = []

    # 쿠키 이름 패턴: "XXX맛 쿠키", "XXX 쿠키", "XXX쿠키"
    cookie_patterns = [
        r'[가-힣]{2,10}맛\s?쿠키',           # 용감한맛 쿠키, 딸기맛 쿠키
        r'[가-힣]{2,10}\s쿠키',              # 클로티드 크림 쿠키
        r'(?:순수\s?바닐라|다크\s?카카오|홀리베리|바다요정|서리여왕|화이트릴리|골든치즈|어둠마녀|마녀오븐)\s?쿠키',  # 특수 쿠키
    ]
    for pattern in cookie_patterns:
        matches = re.findall(pattern, text)
        entities.extend(matches)

    # 왕국/지역 이름
    kingdom_patterns = [
        r'[가-힣]{2,8}\s?왕국',              # 바닐라 왕국, 다크카카오 왕국
        r'(?:바닐라|다크카카오|홀리베리|쿠키|크림|카카오)\s?왕국',
        r'[가-힣]{2,8}\s?공화국',            # 크림 공화국
        r'[가-힣]{2,6}\s?마을',              # 쿠키 마을
    ]
    for pattern in kingdom_patterns:
        matches = re.findall(pattern, text)
        entities.extend(matches)

    # 등급
    grades = re.findall(r'(?:에인션트|레전더리|슈퍼에픽|에픽|슈퍼레어|레어|커먼)\s?등급?', text)
    entities.extend(grades)

    # 스킬 이름 (따옴표나 괄호 안의 텍스트)
    skill_names = re.findall(r'[「"\'](.*?)[」"\']', text)
    entities.extend([s for s in skill_names if 2 <= len(s) <= 20])

    # 주요 키워드
    keywords = re.findall(r'(?:소울잼|소울스톤|토핑|보물|길드|아레나|킹덤|오븐|마녀)', text)
    entities.extend(keywords)

    # 영문 고유명사 (대문자로 시작)
    english_entities = re.findall(r'[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*', text)
    entities.extend([e for e in english_entities if len(e) > 3])

    return list(set(entities))


def _extract_relations_simple(text: str, entities: List[str]) -> List[Dict]:
    """쿠키런 킹덤 관계 추출 (패턴 기반)"""
    relations = []

    # 쿠키런 관계 패턴
    relation_patterns = [
        # 소속 관계: "XXX 왕국의 쿠키", "XXX 왕국 소속"
        (r'([가-힣]+\s?왕국)(?:의|에\s?속한|소속)\s*([가-힣]+\s?쿠키)', 'belongs_to'),
        # 스킬 관계: "스킬은 XXX", "스킬: XXX"
        (r'([가-힣]+\s?쿠키).*?스킬[은는:]\s*[「"\'"]?([가-힣a-zA-Z\s]{2,20})', 'has_skill'),
        # 등급 관계: "에인션트 등급 쿠키"
        (r'(에인션트|레전더리|에픽|슈퍼레어|레어|커먼)\s*등급?\s*([가-힣]+\s?쿠키)', 'has_grade'),
        # 관계/동료: "XXX와 함께", "XXX의 동료"
        (r'([가-힣]+\s?쿠키)(?:와|과)\s+([가-힣]+\s?쿠키)', 'ally_of'),
        # 적대 관계: "XXX를 상대로", "XXX와 대립"
        (r'([가-힣]+\s?쿠키).*?(?:적|대립|싸움|전투).*?([가-힣]+\s?쿠키)', 'enemy_of'),
        # 타입: "돌격형", "마법형"
        (r'([가-힣]+\s?쿠키).*?(돌격|방어|마법|사격|치유|지원|폭발|복수)(?:형|타입)', 'has_type'),
    ]

    for pattern, rel_type in relation_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if len(match) >= 2:
                relations.append({
                    "source": match[0].strip(),
                    "target": match[1].strip(),
                    "type": rel_type,
                })

    return relations


def build_knowledge_graph(chunks: List[Any]) -> Dict:
    """청크에서 Knowledge Graph 구축"""
    global KNOWLEDGE_GRAPH

    KNOWLEDGE_GRAPH = {}
    entity_docs: Dict[str, List[str]] = {}  # entity -> document sources
    all_relations = []

    # 전체 텍스트 수집 (위치 관계 추출용)
    full_text_parts = []

    for chunk in chunks:
        try:
            content = safe_str(getattr(chunk, "page_content", ""))
            source = getattr(chunk, "metadata", {}).get("source", "unknown")

            full_text_parts.append(content)

            # 개체 추출
            entities = _extract_entities_simple(content)
            for entity in entities:
                if entity not in entity_docs:
                    entity_docs[entity] = []
                if source not in entity_docs[entity]:
                    entity_docs[entity].append(source)

            # 관계 추출
            relations = _extract_relations_simple(content, entities)
            all_relations.extend(relations)
        except Exception:
            continue

    # ★ 전체 텍스트에서 대륙-지역 위치 관계 추출 (located_in, contains)
    full_text = "\n".join(full_text_parts)
    location_relations = _extract_location_relations_from_text(full_text)
    all_relations.extend(location_relations)

    # 중복 관계 제거
    seen_relations = set()
    unique_relations = []
    for rel in all_relations:
        key = (rel.get("source", ""), rel.get("target", ""), rel.get("type", ""))
        if key not in seen_relations:
            seen_relations.add(key)
            unique_relations.append(rel)

    # Knowledge Graph 구조화
    KNOWLEDGE_GRAPH = {
        "entities": entity_docs,
        "relations": unique_relations,
        "stats": {
            "entity_count": len(entity_docs),
            "relation_count": len(unique_relations),
            "location_relations": len(location_relations),
        }
    }

    st.logger.info("KNOWLEDGE_GRAPH_BUILT entities=%d relations=%d (location=%d)",
                   len(entity_docs), len(unique_relations), len(location_relations))
    return KNOWLEDGE_GRAPH


def search_knowledge_graph(query: str, top_k: int = 5) -> List[Dict]:
    """
    Knowledge Graph에서 관련 엔티티 검색

    개선: "A는 어느 대륙?" 같은 위치 질문에서 located_in 관계 우선 반환
    """
    global KNOWLEDGE_GRAPH

    if not KNOWLEDGE_GRAPH or "entities" not in KNOWLEDGE_GRAPH:
        return []

    results = []
    query_lower = query.lower()

    # ★ 위치 질문 패턴 감지: "어느 대륙", "어디에 위치", "위치가" 등
    is_location_query = any(kw in query_lower for kw in [
        '어느 대륙', '어디에', '위치', '어디', '대륙', '소속', '어느 나라'
    ])

    # ★ 위치 질문이면 located_in 관계에서 직접 검색
    if is_location_query:
        relations = KNOWLEDGE_GRAPH.get("relations", [])
        for rel in relations:
            rel_type = rel.get("type", "")
            source = rel.get("source", "")
            target = rel.get("target", "")

            # located_in 또는 contains 관계만
            if rel_type not in ("located_in", "contains"):
                continue

            # 쿼리에 source나 target이 포함되면 관련 관계
            source_lower = source.lower()
            target_lower = target.lower()

            if source_lower in query_lower or any(w in source_lower for w in query_lower.split() if len(w) >= 2):
                # "골드치즈 왕국은 어느 대륙?" → located_in 관계 찾음
                results.append({
                    "entity": source,
                    "sources": [],
                    "relations": [rel],
                    "score": 20,  # 위치 관계는 높은 점수
                    "location_answer": target if rel_type == "located_in" else source,
                })

    for entity, sources in KNOWLEDGE_GRAPH.get("entities", {}).items():
        entity_lower = entity.lower()
        score = 0

        # 정확히 일치
        if entity_lower in query_lower or query_lower in entity_lower:
            score = 10
        # 부분 일치
        elif any(word in entity_lower for word in query_lower.split() if len(word) >= 2):
            score = 5

        if score > 0:
            # 관련 관계 찾기
            related_relations = [
                r for r in KNOWLEDGE_GRAPH.get("relations", [])
                if r.get("source") == entity or r.get("target") == entity
            ]

            results.append({
                "entity": entity,
                "sources": sources,
                "relations": related_relations[:3],
                "score": score,
            })

    # 점수로 정렬
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# ============================================================
# FAISS 한글 경로 대응 (Windows C++ I/O 우회)
# ============================================================
def _safe_faiss_save(idx, target_dir: str) -> None:
    """FAISS 인덱스를 저장. 한글 경로면 임시 디렉토리 경유."""
    os.makedirs(target_dir, exist_ok=True)
    try:
        idx.save_local(target_dir)
        return
    except Exception:
        pass
    with tempfile.TemporaryDirectory() as tmp:
        idx.save_local(tmp)
        for fname in os.listdir(tmp):
            shutil.copy2(os.path.join(tmp, fname), os.path.join(target_dir, fname))


def _safe_faiss_load(target_dir: str, emb):
    """FAISS 인덱스를 로드. 한글 경로면 임시 디렉토리로 복사 후 로드."""
    try:
        try:
            return FAISS.load_local(target_dir, emb, allow_dangerous_deserialization=True)
        except TypeError:
            return FAISS.load_local(target_dir, emb)
    except Exception:
        pass
    with tempfile.TemporaryDirectory() as tmp:
        for fname in os.listdir(target_dir):
            src = os.path.join(target_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(tmp, fname))
        try:
            return FAISS.load_local(tmp, emb, allow_dangerous_deserialization=True)
        except TypeError:
            return FAISS.load_local(tmp, emb)


# ============================================================
# 인덱스 빌드/로드
# ============================================================
def rag_build_or_load_index(api_key: str, force_rebuild: bool = False) -> None:
    global PARENT_CHUNKS_STORE, CHILD_TO_PARENT_MAP

    with st.RAG_LOCK:
        st.RAG_STORE["error"] = ""

    if (FAISS is None) or (OpenAIEmbeddings is None) or (Document is None):
        with st.RAG_LOCK:
            st.RAG_STORE.update({
                "ready": False, "index": None,
                "error": "RAG 비활성화: langchain_community/FAISS 또는 OpenAIEmbeddings 또는 Document import 실패",
            })
        return

    k = (api_key or "").strip()
    if not k:
        with st.RAG_LOCK:
            st.RAG_STORE.update({
                "ready": False, "index": None,
                "error": "RAG 비활성화: OpenAI API Key가 없습니다.(환경변수 OPENAI_API_KEY 또는 요청 apiKey 필요)",
            })
        return

    paths = _rag_list_files()
    fp = _rag_files_fingerprint(paths)

    # 기존 인덱스 로드 (파일 해시 동일)
    if (not force_rebuild) and os.path.exists(st.RAG_FAISS_DIR):
        saved = _rag_load_state_file()
        if isinstance(saved, dict) and saved.get("hash") == fp:
            try:
                emb = _make_embeddings(k)
                if emb is None:
                    raise RuntimeError("embeddings_init_failed")
                idx = _safe_faiss_load(st.RAG_FAISS_DIR, emb)

                # ★ BM25 인덱스 빌드 (서버 재시작 시에도 동작하도록)
                # FAISS는 파일에서 로드하지만, BM25는 메모리 전용이라 다시 빌드 필요
                bm25_built = False

                # 문서 다시 읽어서 Parent-Child 청킹 + BM25 빌드
                load_docs: List[Any] = []
                for p in paths:
                    txt = _rag_read_file(p)
                    if not txt:
                        continue
                    rel = os.path.relpath(p, st.RAG_DOCS_DIR).replace("\\", "/")
                    try:
                        load_docs.append(Document(page_content=txt, metadata={"source": rel}))
                    except Exception:
                        continue

                if load_docs:
                    # Parent-Child 청킹 (검색 시 Parent content 반환용)
                    # ★ 서버 재시작 시에는 Contextual Retrieval 건너뛰기 (빠른 로드)
                    child_chunks, parent_store, child_to_parent = _create_parent_child_chunks(
                        load_docs,
                        parent_size=3000,
                        parent_overlap=500,
                        child_size=500,
                        child_overlap=100,
                        enable_contextual=False  # 기존 인덱스 로드 시 건너뛰기
                    )
                    PARENT_CHUNKS_STORE = parent_store
                    CHILD_TO_PARENT_MAP = child_to_parent

                    # BM25 인덱스 빌드
                    bm25_built = _build_bm25_index(child_chunks)

                    # Knowledge Graph 빌드 (K²RAG용)
                    kg_result = build_knowledge_graph(child_chunks)
                    kg_built = bool(kg_result and kg_result.get("entities"))
                    st.logger.info("BM25_KG_REBUILT_ON_LOAD docs=%d bm25=%s kg=%s", len(load_docs), bm25_built, kg_built)

                with st.RAG_LOCK:
                    st.RAG_STORE.update({
                        "ready": True, "hash": fp,
                        "files_count": int(saved.get("files_count") or saved.get("docs_count") or 0),
                        "chunks_count": int(saved.get("chunks_count") or saved.get("docs_count") or 0),
                        "last_build_ts": float(saved.get("last_build_ts") or time.time()),
                        "error": "", "index": idx,
                        "bm25_ready": bm25_built,
                        "kg_ready": kg_built,
                    })
                st.logger.info("RAG_READY(load) files=%s chunks=%s bm25=%s hash=%s",
                              st.RAG_STORE.get("files_count"), st.RAG_STORE.get("chunks_count"), bm25_built, safe_str(fp)[:10])
                return
            except Exception as e:
                st.logger.warning("RAG_LOAD_FAIL err=%s", safe_str(e))

    # 새로 빌드
    docs: List[Any] = []
    for p in paths:
        txt = _rag_read_file(p)
        if not txt:
            continue
        rel = os.path.relpath(p, st.RAG_DOCS_DIR).replace("\\", "/")
        try:
            docs.append(Document(page_content=txt, metadata={"source": rel}))
        except Exception:
            continue

    if not docs:
        with st.RAG_LOCK:
            st.RAG_STORE.update({
                "ready": False, "index": None, "hash": fp,
                "files_count": 0, "chunks_count": 0,
                "last_build_ts": time.time(),
                "error": "rag_docs 폴더에 인덱싱할 문서가 없습니다.",
            })
        _rag_save_state_file({
            "hash": fp, "files_count": 0, "chunks_count": 0,
            "last_build_ts": float(st.RAG_STORE.get("last_build_ts") or time.time()),
            "error": safe_str(st.RAG_STORE.get("error", "")),
            "embed_model": st.RAG_EMBED_MODEL,
        })
        st.logger.info("RAG_EMPTY docs_dir=%s", st.RAG_DOCS_DIR)
        return

    # Parent-Child Chunking (검색 정확도 대폭 향상)
    # Child: 중간 청크 (500자) - 검색용 (섹션 제목+내용 함께 포함)
    # Parent: 큰 청크 (3000자) - 반환용 (전체 섹션 문맥 제공)
    files_count = len(docs)  # 원본 문서 수
    child_chunks, parent_store, child_to_parent = _create_parent_child_chunks(
        docs,
        parent_size=3000,   # Parent: 전체 섹션 포함 (희귀재료 등)
        parent_overlap=500,
        child_size=500,     # Child: 섹션 제목+내용 함께 검색
        child_overlap=100,
        enable_contextual=True  # ★ 사용자 재빌드 시에만 Contextual Retrieval 실행
    )

    # 전역 변수에 저장 (검색 시 Parent 반환에 사용)
    PARENT_CHUNKS_STORE = parent_store
    CHILD_TO_PARENT_MAP = child_to_parent

    chunks = child_chunks  # FAISS/BM25에는 Child 청크 저장
    chunks_count = len(chunks)
    parent_count = len(parent_store)
    st.logger.info("PARENT_CHILD_READY children=%d parents=%d files=%d",
                   chunks_count, parent_count, files_count)

    try:
        emb = _make_embeddings(k)
        if emb is None:
            raise RuntimeError("embeddings_init_failed")

        idx = FAISS.from_documents(chunks, emb)
        _safe_faiss_save(idx, st.RAG_FAISS_DIR)

        # BM25 인덱스 빌드 (Hybrid Search용)
        bm25_built = _build_bm25_index(chunks)

        # Knowledge Graph 빌드 (K²RAG용)
        kg_result = build_knowledge_graph(chunks)
        kg_built = bool(kg_result and kg_result.get("entities"))

        with st.RAG_LOCK:
            st.RAG_STORE.update({
                "ready": True, "hash": fp,
                "files_count": files_count,
                "chunks_count": chunks_count,
                "last_build_ts": time.time(),
                "error": "", "index": idx,
                "bm25_ready": bm25_built,
                "kg_ready": kg_built,
            })

        _rag_save_state_file({
            "hash": fp, "files_count": files_count, "chunks_count": chunks_count,
            "last_build_ts": float(st.RAG_STORE.get("last_build_ts") or time.time()),
            "error": "", "embed_model": st.RAG_EMBED_MODEL,
            "bm25_ready": bm25_built, "kg_ready": kg_built,
        })
        st.logger.info("RAG_READY(build) files=%s chunks=%s bm25=%s hash=%s",
                       files_count, chunks_count, bm25_built, safe_str(fp)[:10])
    except Exception as e:
        with st.RAG_LOCK:
            st.RAG_STORE.update({
                "ready": False, "index": None, "hash": fp,
                "files_count": 0, "chunks_count": 0,
                "error": f"RAG 인덱싱 실패: {safe_str(e)}",
            })
        st.logger.exception("RAG_BUILD_FAIL err=%s", safe_str(e))


# ============================================================
# FAISS 벡터 검색 (기본)
# ============================================================
def rag_search_local(query: str, top_k: int = st.RAG_DEFAULT_TOPK, api_key: str = "") -> List[dict]:
    original_q = safe_str(query).strip()
    if not original_q:
        return []

    # 쿼리 확장 (동의어/관련어 추가)
    q = _expand_query(original_q)

    k = max(1, min(int(top_k), st.RAG_MAX_TOPK))

    with st.RAG_LOCK:
        ready = bool(st.RAG_STORE.get("ready"))
        idx = st.RAG_STORE.get("index")
        err = safe_str(st.RAG_STORE.get("error", ""))

    if (not ready) or (idx is None):
        rag_build_or_load_index(api_key=api_key, force_rebuild=False)
        with st.RAG_LOCK:
            ready = bool(st.RAG_STORE.get("ready"))
            idx = st.RAG_STORE.get("index")
            err = safe_str(st.RAG_STORE.get("error", ""))
        if (not ready) or (idx is None):
            return [{"title": "RAG_ERROR", "source": "", "score": 0.0, "content": err}] if err else []

    try:
        # Parent-Child: 더 많은 후보에서 중복 제거 후 top_k 반환
        pairs = idx.similarity_search_with_score(q, k=k * 3)

        max_dist = float(getattr(st, "RAG_MAX_DISTANCE", 1.6))

        out: List[dict] = []
        seen_parents = set()  # 같은 Parent 중복 방지

        for doc, score in pairs:
            if len(out) >= k:
                break
            if score is None:
                continue
            try:
                dist = float(score)
            except Exception:
                continue
            if dist > max_dist:
                continue

            try:
                metadata = getattr(doc, "metadata", {})
                src = safe_str(metadata.get("source", ""))
                parent_id = metadata.get("parent_id", "")
            except Exception:
                src = ""
                parent_id = ""

            try:
                child_txt = safe_str(getattr(doc, "page_content", ""))
            except Exception:
                child_txt = ""

            # Parent-Child: Parent content 반환 (중복 방지)
            if parent_id and parent_id not in seen_parents:
                txt = _get_parent_content(parent_id, child_txt)
                seen_parents.add(parent_id)
            elif not parent_id:
                txt = child_txt
            else:
                continue  # 이미 본 Parent, 스킵

            if not txt:
                continue

            out.append({
                "title": src or "doc",
                "source": src,
                "score": round(dist, 6),
                "content": txt[:st.RAG_SNIPPET_CHARS],
            })
        return out
    except Exception as e:
        return [{"title": "RAG_ERROR", "source": "", "score": 0.0, "content": f"RAG 검색 실패: {safe_str(e)}"}]


# ============================================================
# Query Rewriting (쿼리 확장/동의어 매핑)
# ============================================================
# 사용자 질문을 PDF 문서 표현에 맞게 확장
QUERY_EXPANSION_MAP: Dict[str, List[str]] = {
    # 세계관/배경 관련
    "시대적 배경": ["모티브", "근대", "세계대전 이전", "고대"],
    "배경": ["모티브", "설정", "톤"],
    "형태": ["모습", "모양", "형상", "닮", "거대한 쿠키"],
    "닮았": ["형태", "모습", "모양"],
    "세계 이름": ["어썸브레드", "대륙", "세계"],

    # 캐릭터/쿠키 관련
    "출신": ["국가", "왕국", "소속", "고향"],
    "등급": ["커먼", "레어", "에픽", "슈퍼에픽", "레전더리", "에인션트", "드래곤", "비스트"],
    "전투 배치": ["전방", "중앙", "후방", "포지션"],

    # 마법 관련
    "마법": ["베이킹 마법", "다크문마법", "백마법", "마녀"],
    "베이킹 마법": ["마녀", "오븐", "굽다"],

    # 장소 관련
    "신전": ["찬란한 영웅들의 신전", "레벨 보정", "배치"],
    "세계지도": ["지도", "대륙", "바다", "해양", "어썸브레드"],

    # 설화/스토리 관련
    "설화": ["창조", "탄생", "기원", "전설"],
    "빛의 신": ["창조", "탄생", "설화", "기원"],

    # 존재/생태 관련
    "비스트": ["비스트이스트", "비스트 쿠키", "봉인", "타락"],
    "생태": ["생물", "디저트", "크리쳐"],
}

# 쿼리에 포함되면 특정 표현 추가
QUERY_INJECTION_RULES: List[Tuple[str, str]] = [
    # (쿼리에 포함된 키워드, 추가할 표현)
    ("거대한 쿠키", "지도 속 땅 전체 모습"),
    ("세계지도 형태", "거대한 쿠키 형태"),
    ("시대적 배경", "고대 근대 모티브"),
    ("클로티드 크림", "크렘 공화국 집정관"),
    ("비스트이스트", "비스트 쿠키 봉인 대륙"),
]


# ============================================================
# RAG-Fusion: Multi-Query Generation & Fusion (2025)
# ============================================================
RAG_FUSION_ENABLED = True  # RAG-Fusion 활성화 여부
RAG_FUSION_NUM_QUERIES = 4  # 생성할 쿼리 변형 수 (원본 포함)

# ============================================================
# RAG-Fusion 최적화: 캐싱 & 단순 쿼리 스킵
# ============================================================
from functools import lru_cache

# 쿼리 캐시 (최대 200개)
_FUSION_QUERY_CACHE: Dict[str, List[str]] = {}
_FUSION_CACHE_MAX_SIZE = 200


def _is_simple_query(query: str) -> bool:
    """
    단순 쿼리 판별 - Fusion 스킵 대상

    단순 쿼리 기준:
    - 10자 이하 (단일 키워드)
    - 공백 없음 (단일 단어)
    - 특정 패턴 (이름만 검색)
    """
    q = query.strip()

    # 10자 이하 단순 키워드
    if len(q) <= 10:
        return True

    # 공백 없는 단일 단어
    if ' ' not in q:
        return True

    # 쿠키/왕국 이름만 검색하는 패턴
    simple_patterns = [
        r'^[가-힣A-Za-z]+ ?쿠키$',  # "용감한 쿠키", "브레이브쿠키"
        r'^[가-힣A-Za-z]+ ?왕국$',  # "바닐라 왕국"
        r'^[가-힣A-Za-z]+[은는이가] ?뭐',  # "소울잼이 뭐야"
    ]

    for pattern in simple_patterns:
        if re.match(pattern, q):
            return True

    return False


def _get_cached_fusion_queries(query: str) -> Optional[List[str]]:
    """캐시에서 Fusion 쿼리 조회"""
    return _FUSION_QUERY_CACHE.get(query)


def _cache_fusion_queries(query: str, queries: List[str]) -> None:
    """Fusion 쿼리를 캐시에 저장"""
    global _FUSION_QUERY_CACHE

    # 캐시 크기 제한 (FIFO)
    if len(_FUSION_QUERY_CACHE) >= _FUSION_CACHE_MAX_SIZE:
        oldest = next(iter(_FUSION_QUERY_CACHE))
        del _FUSION_QUERY_CACHE[oldest]

    _FUSION_QUERY_CACHE[query] = queries


def _generate_fusion_queries(original_query: str, api_key: str = "", num_queries: int = 4) -> List[str]:
    """
    RAG-Fusion: LLM으로 다양한 쿼리 변형 생성

    원본 쿼리를 여러 관점에서 재작성하여 검색 recall 향상
    - 동의어/유사 표현 사용
    - 질문 형식 변경
    - 구체적/추상적 표현

    최적화:
    - 단순 쿼리 → Fusion 스킵 (LLM 호출 절약)
    - 캐싱 → 동일 쿼리 재사용

    Returns:
        [원본 쿼리, 변형1, 변형2, 변형3, ...]
    """
    queries = [original_query]  # 원본은 항상 포함

    if not original_query or num_queries <= 1:
        return queries

    # ★ 최적화 1: 단순 쿼리는 Fusion 스킵
    if _is_simple_query(original_query):
        st.logger.info("RAG_FUSION_SKIP: Simple query '%s'", original_query[:30])
        return queries

    # ★ 최적화 2: 캐시 확인
    cached = _get_cached_fusion_queries(original_query)
    if cached:
        st.logger.info("RAG_FUSION_CACHE_HIT: '%s' → %d queries", original_query[:30], len(cached))
        return cached

    effective_key = safe_str(api_key).strip() or st.OPENAI_API_KEY
    if not effective_key:
        st.logger.warning("RAG_FUSION: No API key, skipping query generation")
        return queries

    try:
        import openai
        client = openai.OpenAI(api_key=effective_key)

        prompt = f"""당신은 검색 쿼리 전문가입니다. 주어진 쿼리를 다양한 관점에서 {num_queries - 1}개의 변형 쿼리로 재작성하세요.

원본 쿼리: "{original_query}"

규칙:
1. 원본 의도를 유지하되 다른 표현/키워드 사용
2. 동의어, 유사어, 관련 용어 활용
3. 질문 형식과 서술 형식 혼합
4. 쿠키런 세계관 용어가 있으면 관련 용어도 포함
5. 각 쿼리는 한 줄로, 번호 없이

예시:
원본: "바닐라 왕국 위치"
변형1: 바닐라 왕국 어디에 있나
변형2: 바닐라 왕국 지리적 위치 대륙
변형3: 바닐라 왕국 소재지 지도

{num_queries - 1}개의 변형 쿼리만 출력하세요 (번호/설명 없이):"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",  # 빠르고 저렴한 모델
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7,  # 다양성 확보
        )

        generated = response.choices[0].message.content.strip()
        new_queries = [q.strip() for q in generated.split("\n") if q.strip()]

        # 원본과 중복 제거, 최대 개수 제한
        for q in new_queries[:num_queries - 1]:
            if q and q != original_query and q not in queries:
                queries.append(q)

        st.logger.info("RAG_FUSION: Generated %d queries from '%s'", len(queries), original_query[:30])
        for i, q in enumerate(queries):
            st.logger.debug("  [%d] %s", i, q[:50])

        # ★ 최적화 3: 캐시에 저장
        if len(queries) > 1:
            _cache_fusion_queries(original_query, queries)

    except Exception as e:
        st.logger.warning("RAG_FUSION: Query generation failed: %s", safe_str(e)[:100])

    return queries


def _multi_query_rrf(
    all_results: List[List[Tuple[Dict, float]]],
    k: int = 60
) -> List[Tuple[Dict, float]]:
    """
    Multi-Query RRF: 여러 쿼리 결과를 Reciprocal Rank Fusion으로 병합

    Args:
        all_results: [쿼리1 결과, 쿼리2 결과, ...] 각각 [(doc, score), ...]
        k: RRF 상수 (기본 60)

    Returns:
        병합된 결과 [(doc, rrf_score), ...]
    """
    fusion_scores: Dict[str, Dict] = {}

    for query_idx, results in enumerate(all_results):
        for rank, (doc, score) in enumerate(results):
            # content 기반 key (중복 방지)
            key = doc.get("content", "")[:150]
            if not key:
                continue

            if key not in fusion_scores:
                fusion_scores[key] = {
                    "doc": doc,
                    "rrf_score": 0.0,
                    "query_hits": 0,
                    "best_rank": rank,
                }

            # RRF 점수 누적
            fusion_scores[key]["rrf_score"] += 1.0 / (k + rank + 1)
            fusion_scores[key]["query_hits"] += 1
            fusion_scores[key]["best_rank"] = min(fusion_scores[key]["best_rank"], rank)

    # 여러 쿼리에서 발견된 문서에 보너스
    for key, data in fusion_scores.items():
        if data["query_hits"] > 1:
            # 2개 이상 쿼리에서 발견 → 20% 보너스
            data["rrf_score"] *= (1.0 + 0.1 * data["query_hits"])

    # RRF 점수 순 정렬
    sorted_results = sorted(
        fusion_scores.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )

    return [(item["doc"], item["rrf_score"]) for item in sorted_results]


def _expand_query(query: str) -> str:
    """
    쿼리를 PDF 문서 표현에 맞게 확장
    - 동의어/관련어 추가
    - 문서에 실제 있는 표현으로 매핑
    """
    if not query:
        return query

    expanded_terms = set()
    query_lower = query.lower()

    # 1. 동의어 확장
    for keyword, expansions in QUERY_EXPANSION_MAP.items():
        if keyword in query or keyword in query_lower:
            expanded_terms.update(expansions)

    # 2. 특정 표현 주입
    for trigger, injection in QUERY_INJECTION_RULES:
        if trigger in query or trigger in query_lower:
            expanded_terms.add(injection)

    # 3. 원본 쿼리 + 확장 표현 결합
    if expanded_terms:
        # 원본 쿼리에 이미 있는 표현 제거
        new_terms = [t for t in expanded_terms if t not in query]
        if new_terms:
            expanded_query = f"{query} {' '.join(new_terms[:5])}"  # 최대 5개 확장어
            st.logger.info("QUERY_EXPANSION original='%s' expanded='%s'", query[:50], expanded_query[:80])
            return expanded_query

    return query


# ============================================================
# Hybrid Search (BM25 + Vector + Reranking + RAG-Fusion)
# ============================================================
def _search_single_query(
    q: str,
    retrieval_k: int,
    effective_key: str
) -> Tuple[List[Tuple[Dict, float]], List[Tuple[Dict, float]]]:
    """
    단일 쿼리로 Vector + BM25 검색 실행 (RAG-Fusion 내부 헬퍼)

    Returns:
        (vector_results, bm25_results)
    """
    vector_results = []
    bm25_results = []

    # Vector Search (FAISS)
    with st.RAG_LOCK:
        ready = bool(st.RAG_STORE.get("ready"))
        idx = st.RAG_STORE.get("index")

    if ready and idx is not None:
        try:
            pairs = idx.similarity_search_with_score(q, k=retrieval_k)
            seen_parents = set()

            for doc, dist in pairs:
                try:
                    child_content = safe_str(getattr(doc, "page_content", ""))
                    metadata = getattr(doc, "metadata", {})
                    source = safe_str(metadata.get("source", ""))
                    parent_id = metadata.get("parent_id", "")

                    if parent_id and parent_id not in seen_parents:
                        parent_content = _get_parent_content(parent_id, child_content)
                        seen_parents.add(parent_id)
                        content = parent_content
                    elif not parent_id:
                        content = child_content
                    else:
                        continue

                    if content:
                        vector_results.append((
                            {
                                "content": content,
                                "source": source,
                                "title": source or "doc",
                                "parent_id": parent_id,
                                "matched_child": child_content[:100],
                            },
                            float(dist)
                        ))
                except Exception:
                    continue
        except Exception as e:
            st.logger.warning("SEARCH_VECTOR_FAIL q=%s err=%s", q[:20], safe_str(e)[:50])

    # BM25 Search
    with st.RAG_LOCK:
        bm25_ready = bool(st.RAG_STORE.get("bm25_ready"))

    if bm25_ready and BM25_INDEX is not None:
        bm25_results = _bm25_search(q, top_k=retrieval_k)

    return vector_results, bm25_results


def rag_search_hybrid(
    query: str,
    top_k: int = st.RAG_DEFAULT_TOPK,
    api_key: str = "",
    use_reranking: bool = True,
    use_kg: bool = True,  # ★ KG 기본 활성화
    use_fusion: bool = True  # ★ RAG-Fusion 옵션 추가
) -> dict:
    """
    고급 RAG 검색 (2025 Best Practice):
    - KG 검색: 관계 질의(어느 대륙, 어디 위치)에서 정확한 답 제공
    - RAG-Fusion: 다중 쿼리 생성 및 RRF 병합 (recall 향상)
    - Hybrid Search: BM25 (키워드) + Vector (의미) 조합
    - Reranking: BGE Cross-Encoder로 결과 재정렬 (20-35% 정확도 향상)

    검색 전략:
    1. ★ KG 검색: located_in 등 관계 먼저 조회 (병렬)
    2. RAG-Fusion: 쿼리 → 4개 변형 생성 (LLM)
    3. 각 쿼리로 BM25 + Vector 검색
    4. KG 결과를 텍스트 결과와 병합 (KG 매칭 문서에 가중치)
    5. Reranking: Cross-Encoder로 정밀 재정렬
    6. 최종: top_k개 반환
    """
    original_query = safe_str(query).strip()
    if not original_query:
        return {"status": "FAILED", "error": "Empty query", "results": []}

    k = max(1, min(int(top_k), st.RAG_MAX_TOPK))
    effective_key = safe_str(api_key).strip() or st.OPENAI_API_KEY

    # RAG 인덱스 준비 확인
    with st.RAG_LOCK:
        ready = bool(st.RAG_STORE.get("ready"))
        idx = st.RAG_STORE.get("index")
        kg_ready = bool(st.RAG_STORE.get("kg_ready"))

    if (not ready) or (idx is None):
        rag_build_or_load_index(api_key=effective_key, force_rebuild=False)

    # ★ KG 검색 먼저 수행 (병렬로 텍스트 검색과 동시에)
    kg_entities = []
    kg_location_answer = None  # "A는 어느 대륙?" 직접 답변용
    kg_matched_entities = set()  # KG에서 매칭된 엔티티들 (텍스트 보너스용)

    if use_kg and kg_ready and KNOWLEDGE_GRAPH:
        kg_entities = search_knowledge_graph(original_query, top_k=5)

        # KG에서 location_answer가 있으면 (위치 질문에 대한 직접 답)
        for kg_ent in kg_entities:
            if kg_ent.get("location_answer"):
                kg_location_answer = kg_ent.get("location_answer")
                st.logger.info("KG_LOCATION_ANSWER query=%s answer=%s",
                              original_query[:30], kg_location_answer)

            # 매칭된 엔티티 수집 (텍스트 검색 결과 보너스용)
            kg_matched_entities.add(kg_ent.get("entity", "").lower())
            for rel in kg_ent.get("relations", []):
                kg_matched_entities.add(rel.get("source", "").lower())
                kg_matched_entities.add(rel.get("target", "").lower())

        st.logger.info("KG_SEARCH query=%s entities=%d location=%s",
                      original_query[:30], len(kg_entities),
                      kg_location_answer or "none")

    # Reranker 사용 시 더 많은 후보 확보
    retrieval_k = k * 4 if use_reranking and RERANKER_AVAILABLE else k * 2

    # ★ RAG-Fusion: 다중 쿼리 생성 및 검색
    if use_fusion and RAG_FUSION_ENABLED:
        # 1. 쿼리 변형 생성 (원본 + LLM 생성)
        fusion_queries = _generate_fusion_queries(
            original_query,
            api_key=effective_key,
            num_queries=RAG_FUSION_NUM_QUERIES
        )

        # 2. 각 쿼리에 동의어 확장 적용 후 검색 (★ 병렬 실행)
        def _search_and_fuse(fq: str) -> List[Tuple[Dict, float]]:
            """단일 쿼리 검색 + RRF (병렬 처리용 헬퍼)"""
            expanded_q = _expand_query(fq)
            vec_res, bm25_res = _search_single_query(expanded_q, retrieval_k, effective_key)

            # 단일 쿼리 내 Vector + BM25 RRF
            if bm25_res and vec_res:
                single_fused = _reciprocal_rank_fusion(bm25_res, vec_res)
                return [(r, r.get("fusion_score", 0.5)) for r in single_fused]
            elif vec_res:
                return [(doc, 1.0 / (1.0 + dist)) for doc, dist in vec_res]
            elif bm25_res:
                return [(doc, score / 100.0) for doc, score in bm25_res]
            return []

        # ★ ThreadPoolExecutor로 병렬 검색 (4개 쿼리 동시 실행)
        all_combined_results: List[List[Tuple[Dict, float]]] = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_search_and_fuse, fq) for fq in fusion_queries]
            for future in as_completed(futures):
                try:
                    combined = future.result()
                    if combined:
                        all_combined_results.append(combined)
                except Exception as e:
                    st.logger.warning("RAG_FUSION_PARALLEL_FAIL: %s", safe_str(e)[:50])

        # 3. Multi-Query RRF 병합
        if all_combined_results:
            multi_fused = _multi_query_rrf(all_combined_results)
            fused_results = [
                {**doc, "fusion_score": round(score, 4), "multi_query_hits": doc.get("query_hits", 1)}
                for doc, score in multi_fused
            ]
            search_method = f"rag_fusion_{len(fusion_queries)}q"
            st.logger.info("RAG_FUSION: %d queries → %d results", len(fusion_queries), len(fused_results))
        else:
            return {"status": "FAILED", "error": "No search results from RAG-Fusion", "results": []}

    else:
        # ★ 기존 단일 쿼리 검색 (RAG-Fusion 비활성화)
        q = _expand_query(original_query)
        vector_results, bm25_results = _search_single_query(q, retrieval_k, effective_key)

        # Reciprocal Rank Fusion
        if bm25_results and vector_results:
            fused_results = _reciprocal_rank_fusion(bm25_results, vector_results)
            search_method = "hybrid"
        elif vector_results:
            fused_results = [
                {**doc, "vector_score": round(1.0 / (1.0 + dist), 4), "fusion_score": round(1.0 / (1.0 + dist), 4)}
                for doc, dist in vector_results
            ]
            search_method = "vector"
        elif bm25_results:
            fused_results = [
                {**doc, "bm25_score": round(score, 4), "fusion_score": round(score / 100.0, 4)}
                for doc, score in bm25_results
            ]
            search_method = "bm25"
        else:
            return {"status": "FAILED", "error": "No search results", "results": []}

    # 검색 결과가 없으면 실패
    if not fused_results:
        return {"status": "FAILED", "error": "No search results", "results": []}

    # ★ 4. 쿼리-섹션 제목 매칭 보너스 (핵심 정확도 향상!)
    # "평가"라는 쿼리가 "[섹션: 2.2.2. 평가]" 태그에 매칭되면 점수 대폭 상향
    query_keywords = set(re.findall(r'[가-힣]{2,}', original_query))  # 쿼리에서 한글 키워드 추출
    if query_keywords:
        for r in fused_results:
            content = r.get("content", "") + r.get("matched_child", "")
            # 섹션 제목 추출 (태그에서)
            section_matches = re.findall(r'\[섹션[^:]*:\s*([^\]]+)\]', content)
            title_matches = re.findall(r'\[제목:\s*([^\]]+)\]', content)
            topic_matches = re.findall(r'\[주제:\s*([^\]]+)\]', content)
            all_titles = section_matches + title_matches + topic_matches

            # 쿼리 키워드가 섹션 제목에 있으면 보너스
            section_match_bonus = 0.0
            for title in all_titles:
                for kw in query_keywords:
                    if kw in title:
                        section_match_bonus = max(section_match_bonus, 0.5)  # 50% 보너스
                        # 정확히 일치하면 더 큰 보너스
                        pure_title = re.sub(r'^\d+\.[\d.]*\s*', '', title).strip()
                        if kw == pure_title:
                            section_match_bonus = max(section_match_bonus, 1.0)  # 100% 보너스

            if section_match_bonus > 0:
                original_score = r.get("fusion_score", 0.0)
                r["fusion_score"] = round(original_score + section_match_bonus, 4)
                r["section_match_bonus"] = round(section_match_bonus, 4)
                st.logger.debug("SECTION_MATCH_BONUS query=%s title=%s bonus=%.2f",
                               original_query[:20], all_titles[:2], section_match_bonus)

        # 보너스 적용 후 재정렬
        fused_results.sort(key=lambda x: x.get("fusion_score", 0), reverse=True)

    # ★ KG 매칭 보너스: KG에서 찾은 엔티티가 텍스트에 포함되면 점수 상향
    if kg_matched_entities and fused_results:
        for r in fused_results:
            content_lower = r.get("content", "").lower()
            kg_match_bonus = 0.0

            for kg_entity in kg_matched_entities:
                if kg_entity and len(kg_entity) >= 2 and kg_entity in content_lower:
                    kg_match_bonus = max(kg_match_bonus, 0.3)  # 30% 보너스

                    # KG location_answer가 텍스트에 있으면 더 큰 보너스
                    if kg_location_answer and kg_location_answer.lower() in content_lower:
                        kg_match_bonus = max(kg_match_bonus, 0.8)  # 80% 보너스

            if kg_match_bonus > 0:
                original_score = r.get("fusion_score", 0.0)
                r["fusion_score"] = round(original_score + kg_match_bonus, 4)
                r["kg_match_bonus"] = round(kg_match_bonus, 4)
                st.logger.debug("KG_MATCH_BONUS content=%s bonus=%.2f",
                               content_lower[:50], kg_match_bonus)

        # KG 보너스 적용 후 재정렬
        fused_results.sort(key=lambda x: x.get("fusion_score", 0), reverse=True)

    # 4. Reranking (BGE Cross-Encoder) - 핵심 성능 향상!
    reranked = False
    if use_reranking and RERANKER_AVAILABLE and len(fused_results) > 1:
        fused_results = _rerank_results(original_query, fused_results, top_k=k)
        reranked = True

    # top_k 제한 및 content 자르기
    final_results = []
    for r in fused_results[:k]:
        r["content"] = r.get("content", "")[:st.RAG_SNIPPET_CHARS]
        final_results.append(r)

    # ★ KG 검색은 이미 위에서 수행됨 (kg_entities, kg_location_answer)

    return {
        "status": "SUCCESS",
        "query": original_query,
        "search_method": search_method,
        "fusion_enabled": use_fusion and RAG_FUSION_ENABLED,
        "top_k": k,
        "reranked": reranked,
        "bm25_available": BM25_AVAILABLE and BM25_INDEX is not None,
        "reranker_available": RERANKER_AVAILABLE,
        "kg_available": bool(KNOWLEDGE_GRAPH),
        "kg_used": use_kg and kg_ready,  # ★ KG 사용 여부
        "kg_location_answer": kg_location_answer,  # ★ 위치 질문 직접 답 (있으면)
        "results": final_results,
        "kg_entities": kg_entities,
    }


# ============================================================
# 글로서리(사전) 검색
# ============================================================
def rag_search_glossary(query: str, top_k: int = 3) -> List[dict]:
    from core.constants import RAG_DOCUMENTS
    query_lower = (query or "").lower()
    scores = []

    for _, doc in RAG_DOCUMENTS.items():
        score = 0
        for kw in doc.get("keywords", []):
            kw_lower = (kw or "").lower().strip()
            if kw_lower and kw_lower in query_lower:
                score += 2
        title_lower = (doc.get("title") or "").lower().strip()
        if title_lower and title_lower in query_lower:
            score += 3  # 제목 매칭은 더 크게
        scores.append((score, doc))

    scores.sort(key=lambda x: x[0], reverse=True)
    return [
        {"title": doc["title"], "content": doc["content"], "source": "glossary", "score": float(sc)}
        for sc, doc in scores[:top_k]
        if sc > 0
    ]


# ============================================================
# 통합 RAG 검색 (Hybrid Search + 글로서리)
# ============================================================
def tool_rag_search(query: str, top_k: int = st.RAG_DEFAULT_TOPK, api_key: str = "") -> dict:
    """
    통합 RAG 검색 - Hybrid Search (BM25 + Vector) 사용

    변경 이력:
    - 기존: rag_search_local (FAISS 벡터 검색만 사용)
    - 현재: rag_search_hybrid (BM25 키워드 + FAISS 벡터 + Reranking)

    BM25 키워드 검색을 추가하여 "희귀재료" 같은 정확한 키워드 매칭 개선
    """
    effective_key = safe_str(api_key).strip() or st.OPENAI_API_KEY

    # ★ 안전한 top_k 변환 (API 키가 잘못 들어와도 노출 방지)
    try:
        k = int(max(1, min(int(top_k), st.RAG_MAX_TOPK)))
    except (ValueError, TypeError):
        # top_k가 숫자가 아닌 경우 기본값 사용 (에러 메시지에 값 노출 금지)
        k = st.RAG_DEFAULT_TOPK

    gloss = rag_search_glossary(query, top_k=k)

    # ★ 핵심 변경: rag_search_local → rag_search_hybrid
    # BM25 키워드 검색을 추가하여 정확한 키워드 매칭 개선
    hybrid_result = rag_search_hybrid(query, top_k=k, api_key=effective_key, use_reranking=False)
    hybrid_results = hybrid_result.get("results", []) if hybrid_result.get("status") == "SUCCESS" else []

    merged: List[dict] = []
    seen = set()

    # 1) glossary 먼저
    if isinstance(gloss, list):
        for r in gloss:
            key = safe_str(r.get("title") or "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                merged.append({
                    "title": r.get("title"),
                    "source": "glossary",
                    "score": 0.0,
                    "priority": 1000.0 + float(r.get("score") or 0.0),
                    "content": safe_str(r.get("content") or "")[:st.RAG_SNIPPET_CHARS],
                })

    # 2) Hybrid 검색 결과 (BM25 + Vector + Reranking)
    remain = max(0, k - len(merged))
    if remain > 0 and isinstance(hybrid_results, list):
        for r in hybrid_results:
            key = safe_str(r.get("source") or r.get("title") or "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                # Hybrid 결과는 fusion_score 또는 rerank_score 사용
                fusion_score = float(r.get("rerank_score") or r.get("fusion_score") or 0.0)
                merged.append({
                    "title": r.get("title"),
                    "source": r.get("source"),
                    "score": round(fusion_score, 6),
                    "priority": 100.0 * fusion_score,  # fusion_score가 높을수록 좋음
                    "content": safe_str(r.get("content") or "")[:st.RAG_SNIPPET_CHARS],
                    # ★ 디버깅용: 어떤 Child가 매칭되었는지 표시
                    "matched_child": safe_str(r.get("matched_child") or "")[:200],
                })
                remain -= 1
                if remain <= 0:
                    break

    # 3) priority로 정렬
    merged.sort(key=lambda x: float(x.get("priority") or 0.0), reverse=True)
    for m in merged:
        m.pop("priority", None)

    with st.RAG_LOCK:
        rag_ready = bool(st.RAG_STORE.get("ready"))
        rag_err = safe_str(st.RAG_STORE.get("error", ""))
        rag_docs = int(st.RAG_STORE.get("docs_count") or 0)

    # guardrail 제거됨 (할루시네이션 방지 규칙 비활성화)
    guardrail = ""

    return {
        "status": "SUCCESS" if merged else "FAILED",
        "query": safe_str(query),
        "top_k": k,
        "rag_ready": rag_ready,
        "rag_docs_count": rag_docs,
        "rag_error": rag_err,
        "search_method": hybrid_result.get("search_method", "unknown"),
        "reranked": hybrid_result.get("reranked", False),
        "results": merged,
        "guardrail": guardrail,
    }
