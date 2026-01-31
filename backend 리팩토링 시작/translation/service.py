"""
쿠키런 세계관 맞춤형 번역 서비스
================================
데브시스터즈 기술혁신 프로젝트

LLM 기반 번역 시스템:
- 세계관 용어 일관성 유지
- 캐릭터 말투/성격 반영
- 번역 품질 자동 평가
"""

import os
import json
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

from openai import OpenAI

from core.constants import (
    WORLDVIEW_TERMS,
    SUPPORTED_LANGUAGES,
    TRANSLATION_SYSTEM_PROMPT,
    TRANSLATION_CATEGORIES,
    TRANSLATION_QUALITY_GRADES,
)
import state as st


class TranslationCategory(str, Enum):
    UI = "UI"
    STORY = "story"
    SKILL = "skill"
    DIALOG = "dialog"
    ITEM = "item"
    QUEST = "quest"
    ACHIEVEMENT = "achievement"
    NOTICE = "notice"
    EVENT = "event"
    TUTORIAL = "tutorial"


@dataclass
class TranslationRequest:
    """번역 요청"""
    source_text: str
    source_lang: str = "ko"
    target_lang: str = "en"
    category: str = "dialog"
    context: Optional[str] = None
    character_name: Optional[str] = None
    preserve_terms: bool = True


@dataclass
class TranslationResult:
    """번역 결과"""
    source_text: str
    translated_text: str
    target_lang: str
    category: str
    detected_terms: List[Dict[str, str]]
    quality_score: float
    quality_grade: str
    suggestions: List[str]
    metadata: Dict[str, Any]


class CookieRunTranslator:
    """쿠키런 세계관 맞춤형 번역기"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or st.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.model = "gpt-4o-mini"

    def _detect_worldview_terms(self, text: str) -> List[Dict[str, str]]:
        """세계관 용어 감지"""
        detected = []
        for term_ko, translations in WORLDVIEW_TERMS.items():
            if term_ko in text:
                detected.append({
                    "term_ko": term_ko,
                    "translations": {k: v for k, v in translations.items() if k != "context"},
                    "context": translations.get("context", ""),
                })
        return detected

    def _build_translation_prompt(self, request: TranslationRequest, detected_terms: List[Dict]) -> str:
        """번역 프롬프트 생성"""
        target_lang_name = SUPPORTED_LANGUAGES.get(request.target_lang, request.target_lang)

        prompt = f"""다음 쿠키런 게임 텍스트를 {target_lang_name}로 번역해주세요.

**원문**: {request.source_text}
**텍스트 유형**: {request.category}
"""

        if request.character_name:
            prompt += f"**발화 캐릭터**: {request.character_name}\n"

        if request.context:
            prompt += f"**추가 맥락**: {request.context}\n"

        if detected_terms and request.preserve_terms:
            prompt += "\n**세계관 용어 번역 가이드**:\n"
            for term in detected_terms:
                target_translation = term["translations"].get(
                    request.target_lang,
                    term["translations"].get("en", term["term_ko"])
                )
                prompt += f"- {term['term_ko']} → {target_translation} ({term['context']})\n"

        prompt += """
**번역 원칙**:
1. 세계관 용어는 가이드를 따라 일관되게 번역
2. 캐릭터의 말투와 성격을 자연스럽게 반영
3. 게임 맥락에 맞는 자연스러운 표현 사용
4. 원문의 뉘앙스와 감정을 살려 번역

번역 결과만 출력하세요 (설명 없이)."""

        return prompt

    def translate(self, request: TranslationRequest) -> TranslationResult:
        """텍스트 번역"""
        if not self.client:
            return TranslationResult(
                source_text=request.source_text,
                translated_text="[API 키가 설정되지 않았습니다]",
                target_lang=request.target_lang,
                category=request.category,
                detected_terms=[],
                quality_score=0.0,
                quality_grade="needs_review",
                suggestions=["OpenAI API 키를 설정해주세요."],
                metadata={"error": "no_api_key"},
            )

        # 세계관 용어 감지
        detected_terms = self._detect_worldview_terms(request.source_text)

        # 번역 프롬프트 생성
        prompt = self._build_translation_prompt(request, detected_terms)

        try:
            # LLM 호출
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=2000,
            )

            translated_text = response.choices[0].message.content.strip()

            # 품질 평가
            quality_result = self._evaluate_quality(
                request.source_text,
                translated_text,
                request.target_lang,
                request.category,
                detected_terms,
            )

            return TranslationResult(
                source_text=request.source_text,
                translated_text=translated_text,
                target_lang=request.target_lang,
                category=request.category,
                detected_terms=detected_terms,
                quality_score=quality_result["score"],
                quality_grade=quality_result["grade"],
                suggestions=quality_result["suggestions"],
                metadata={
                    "model": self.model,
                    "tokens_used": response.usage.total_tokens if response.usage else 0,
                },
            )

        except Exception as e:
            return TranslationResult(
                source_text=request.source_text,
                translated_text=f"[번역 오류: {str(e)}]",
                target_lang=request.target_lang,
                category=request.category,
                detected_terms=detected_terms,
                quality_score=0.0,
                quality_grade="needs_review",
                suggestions=[f"오류가 발생했습니다: {str(e)}"],
                metadata={"error": str(e)},
            )

    def _evaluate_quality(
        self,
        source_text: str,
        translated_text: str,
        target_lang: str,
        category: str,
        detected_terms: List[Dict],
    ) -> Dict[str, Any]:
        """번역 품질 평가"""
        suggestions = []
        score = 0.85  # 기본 점수

        # 길이 비율 체크
        len_ratio = len(translated_text) / max(len(source_text), 1)
        if len_ratio < 0.3 or len_ratio > 3.0:
            score -= 0.1
            suggestions.append("번역 길이가 원문과 크게 다릅니다. 내용이 누락되거나 추가되지 않았는지 확인하세요.")

        # 세계관 용어 체크
        for term in detected_terms:
            expected_trans = term["translations"].get(target_lang, term["translations"].get("en"))
            if expected_trans and expected_trans not in translated_text:
                score -= 0.05
                suggestions.append(f"세계관 용어 '{term['term_ko']}'가 '{expected_trans}'로 번역되었는지 확인하세요.")

        # 빈 번역 체크
        if not translated_text.strip() or translated_text.startswith("["):
            score = 0.0
            suggestions.append("번역 결과가 비어있거나 오류가 있습니다.")

        # 등급 결정
        if score >= 0.9:
            grade = "excellent"
        elif score >= 0.8:
            grade = "good"
        elif score >= 0.7:
            grade = "acceptable"
        else:
            grade = "needs_review"

        if not suggestions and grade in ["excellent", "good"]:
            suggestions.append("번역 품질이 양호합니다.")

        return {
            "score": max(0.0, min(1.0, score)),
            "grade": grade,
            "suggestions": suggestions,
        }

    def batch_translate(
        self,
        texts: List[str],
        target_lang: str,
        category: str = "dialog",
    ) -> List[TranslationResult]:
        """배치 번역"""
        results = []
        for text in texts:
            request = TranslationRequest(
                source_text=text,
                target_lang=target_lang,
                category=category,
            )
            result = self.translate(request)
            results.append(result)
        return results

    def get_term_glossary(self, target_lang: Optional[str] = None) -> Dict[str, Any]:
        """세계관 용어집 조회"""
        glossary = []
        for term_ko, translations in WORLDVIEW_TERMS.items():
            entry = {
                "term_ko": term_ko,
                "context": translations.get("context", ""),
            }
            if target_lang:
                entry["term_target"] = translations.get(target_lang, translations.get("en", term_ko))
            else:
                entry["translations"] = {k: v for k, v in translations.items() if k != "context"}
            glossary.append(entry)

        return {
            "total": len(glossary),
            "target_lang": target_lang,
            "terms": glossary,
        }


# 싱글톤 인스턴스
_translator: Optional[CookieRunTranslator] = None


def get_translator() -> CookieRunTranslator:
    """번역기 인스턴스 가져오기"""
    global _translator
    if _translator is None:
        _translator = CookieRunTranslator()
    return _translator


def translate_text(
    text: str,
    target_lang: str,
    category: str = "dialog",
    context: Optional[str] = None,
    character_name: Optional[str] = None,
) -> Dict[str, Any]:
    """간편 번역 함수"""
    translator = get_translator()
    request = TranslationRequest(
        source_text=text,
        target_lang=target_lang,
        category=category,
        context=context,
        character_name=character_name,
    )
    result = translator.translate(request)

    return {
        "status": "SUCCESS" if result.quality_score > 0 else "FAILED",
        "source_text": result.source_text,
        "translated_text": result.translated_text,
        "target_lang": result.target_lang,
        "category": result.category,
        "detected_terms": result.detected_terms,
        "quality": {
            "score": result.quality_score,
            "grade": result.quality_grade,
            "description": TRANSLATION_QUALITY_GRADES.get(result.quality_grade, {}).get("description", ""),
        },
        "suggestions": result.suggestions,
        "metadata": result.metadata,
    }
