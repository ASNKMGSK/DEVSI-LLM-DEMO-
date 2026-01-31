"""
쿠키런 AI 플랫폼 - 상수 및 설정
==============================
데브시스터즈 기술혁신 프로젝트
"""

# ============================================
# 쿠키런 세계관 데이터
# ============================================

# 쿠키 등급
COOKIE_GRADES = ["커먼", "레어", "슈퍼레어", "에픽", "레전더리", "에인션트"]

# 쿠키 타입
COOKIE_TYPES = ["돌격", "마법", "사격", "방어", "지원", "폭발", "치유", "복수"]

# 지원 언어
SUPPORTED_LANGUAGES = {
    "ko": "한국어",
    "en": "English",
    "ja": "日本語",
    "zh": "中文",
    "zh-TW": "繁體中文",
    "th": "ไทย",
    "id": "Bahasa Indonesia",
    "de": "Deutsch",
    "fr": "Français",
    "es": "Español",
    "pt": "Português",
}

# 세계관 고유 용어 (번역 시 일관성 유지 필요)
WORLDVIEW_TERMS = {
    "젤리": {"en": "Jelly", "ja": "ゼリー", "zh": "果冻", "context": "게임 내 화폐/아이템"},
    "오븐": {"en": "Oven", "ja": "オーブン", "zh": "烤箱", "context": "쿠키가 태어나는 곳"},
    "소울잼": {"en": "Soul Jam", "ja": "ソウルジャム", "zh": "灵魂宝石", "context": "고대의 신비한 보석"},
    "마녀": {"en": "Witch", "ja": "魔女", "zh": "魔女", "context": "쿠키들의 적"},
    "다크엔챈트리스 쿠키": {"en": "Dark Enchantress Cookie", "ja": "ダークエンチャントレスクッキー", "zh": "黑魔女饼干", "context": "메인 빌런"},
    "트로피컬 소다 섬": {"en": "Tropical Soda Islands", "ja": "トロピカルソーダ諸島", "zh": "热带苏打岛", "context": "지역명"},
    "어둠의 군단": {"en": "Darkness Legion", "ja": "闇の軍団", "zh": "黑暗军团", "context": "적 세력"},
    "쿠키 왕국": {"en": "Cookie Kingdom", "ja": "クッキー王国", "zh": "饼干王国", "context": "메인 왕국"},
    "홀리베리 왕국": {"en": "Hollyberry Kingdom", "ja": "ホーリーベリー王国", "zh": "圣树莓王国", "context": "고대 왕국"},
    "다크카카오 왕국": {"en": "Dark Cacao Kingdom", "ja": "ダークカカオ王国", "zh": "黑可可王国", "context": "고대 왕국"},
}

# ============================================
# ML Feature Columns
# ============================================

# 번역 품질 예측 피처
FEATURE_COLS_TRANSLATION = [
    "category_encoded",
    "target_lang_encoded",
    "fluency_score",
    "adequacy_score",
    "contains_worldview_term",
    "text_length",
]

# 유저 세그먼트 클러스터링 피처
FEATURE_COLS_USER_SEGMENT = [
    "total_events",
    "stage_clears",
    "gacha_pulls",
    "pvp_battles",
    "purchases",
    "vip_level",
]

# 피처 라벨 (한글)
FEATURE_LABELS = {
    "category_encoded": "텍스트 카테고리",
    "target_lang_encoded": "번역 대상 언어",
    "fluency_score": "유창성 점수",
    "adequacy_score": "적절성 점수",
    "contains_worldview_term": "세계관 용어 포함",
    "text_length": "텍스트 길이",
    "total_events": "총 이벤트 수",
    "stage_clears": "스테이지 클리어 수",
    "gacha_pulls": "가챠 뽑기 수",
    "pvp_battles": "PvP 전투 수",
    "purchases": "구매 수",
    "vip_level": "VIP 레벨",
}

# ============================================
# ML Model Metadata
# ============================================

ML_MODEL_INFO = {
    "model_translation_quality.pkl": {
        "name": "번역 품질 예측 모델",
        "type": "Random Forest Classifier",
        "target": "번역 품질 등급 예측 (excellent/good/acceptable/needs_review)",
        "features": ["카테고리", "대상 언어", "유창성", "적절성", "세계관 용어 포함", "텍스트 길이"],
        "metrics": {
            "Accuracy": 0.85,
            "F1_macro": 0.82,
        },
        "description": "LLM 번역 결과물의 품질을 자동으로 평가하여 검수 우선순위 결정에 활용",
    },
    "model_text_category.pkl": {
        "name": "텍스트 카테고리 분류 모델",
        "type": "TF-IDF + Random Forest Classifier",
        "target": "텍스트 유형 분류 (UI/story/skill/dialog/item 등)",
        "features": ["TF-IDF 벡터 (500차원)"],
        "metrics": {
            "Accuracy": 0.78,
            "F1_macro": 0.75,
        },
        "description": "입력 텍스트의 카테고리를 자동 분류하여 번역 스타일 가이드 적용",
    },
    "model_user_segment.pkl": {
        "name": "유저 세그먼트 모델",
        "type": "K-Means Clustering",
        "target": "유저 유형 분류 (5개 세그먼트)",
        "features": ["이벤트 수", "스테이지", "가챠", "PvP", "구매", "VIP"],
        "metrics": {
            "Silhouette_Score": 0.42,
            "N_Clusters": 5,
        },
        "description": "유저 행동 패턴 기반 세그먼트 분류로 맞춤형 서비스 제공",
    },
    "model_anomaly.pkl": {
        "name": "이상 탐지 모델",
        "type": "Isolation Forest",
        "target": "비정상 유저 행동 탐지",
        "features": ["이벤트 수", "스테이지", "가챠", "PvP", "구매", "VIP"],
        "metrics": {
            "Contamination": 0.05,
            "N_Estimators": 150,
        },
        "description": "비정상적인 유저 행동 패턴을 탐지하여 어뷰징/봇 모니터링",
    },
}

# 유저 세그먼트 이름
USER_SEGMENT_NAMES = {
    0: "캐주얼 플레이어",
    1: "하드코어 게이머",
    2: "PvP 전문가",
    3: "콘텐츠 수집가",
    4: "신규 유저",
}

# ============================================
# RAG Documents (세계관 기본 지식)
# ============================================

RAG_DOCUMENTS = {
    "쿠키런_소개": {
        "title": "쿠키런이란?",
        "content": "쿠키런은 데브시스터즈에서 개발한 런닝 액션 게임입니다. 오븐에서 탈출한 쿠키들이 주인공이며, 독특한 세계관과 다양한 캐릭터가 특징입니다.",
        "keywords": ["쿠키런", "데브시스터즈", "게임", "소개"],
    },
    "오븐": {
        "title": "오븐",
        "content": "오븐은 쿠키들이 태어나는 곳입니다. 마녀가 쿠키들을 굽는 장소이며, 쿠키들은 이곳에서 탈출하여 자유를 찾습니다.",
        "keywords": ["오븐", "탄생", "마녀", "탈출"],
    },
    "소울잼": {
        "title": "소울잼",
        "content": "소울잼은 고대의 신비한 보석으로, 강력한 힘을 가지고 있습니다. 에인션트 쿠키들이 소유하고 있으며, 세계의 균형에 중요한 역할을 합니다.",
        "keywords": ["소울잼", "보석", "에인션트", "힘"],
    },
    "에인션트_쿠키": {
        "title": "에인션트 쿠키",
        "content": "에인션트 쿠키는 고대의 영웅들로, 각자 소울잼을 보유하고 있습니다. 순수 바닐라, 다크카카오, 홀리베리, 골든치즈, 화이트릴리 쿠키가 있습니다.",
        "keywords": ["에인션트", "영웅", "고대", "소울잼"],
    },
    "번역_가이드": {
        "title": "번역 가이드라인",
        "content": "쿠키런 세계관의 고유 용어는 일관된 번역을 유지해야 합니다. 캐릭터명, 지역명, 아이템명은 공식 번역을 따르며, 세계관의 톤앤매너를 유지합니다.",
        "keywords": ["번역", "가이드", "용어", "일관성"],
    },
}

# ============================================
# Default System Prompts
# ============================================

DEFAULT_SYSTEM_PROMPT = """당신은 쿠키런 세계관 전문 AI 어시스턴트입니다.

**역할**:
1. 쿠키런 세계관에 대한 질문에 정확하게 답변합니다.
2. 번역 품질 평가 및 개선 제안을 제공합니다.
3. 게임 데이터 분석 결과를 해석하고 인사이트를 제공합니다.
4. 세계관 일관성을 유지하며 콘텐츠 제작을 지원합니다.

**응답 원칙**:
- 쿠키런 세계관의 톤앤매너를 유지합니다 (밝고 친근하지만 때로는 진지한)
- 고유 용어는 정확하게 사용합니다
- 데이터 기반 분석 시 수치를 명확히 제시합니다
- 불확실한 정보는 추측임을 명시합니다

**세계관 핵심 정보**:
- 쿠키들은 오븐에서 태어나며 마녀로부터 도망칩니다
- 소울잼은 고대의 강력한 힘을 가진 보석입니다
- 에인션트 쿠키들은 각 왕국의 수호자입니다
- 어둠의 세력(다크엔챈트리스)과의 대립이 주요 스토리입니다"""

TRANSLATION_SYSTEM_PROMPT = """당신은 쿠키런 게임의 전문 번역가입니다.

**번역 원칙**:
1. 세계관 고유 용어는 공식 번역을 따릅니다
2. 캐릭터의 말투와 성격을 반영합니다
3. 문화적 맥락을 고려한 자연스러운 번역을 합니다
4. 게임 UI/UX 맥락에 맞는 간결한 표현을 사용합니다

**세계관 용어 번역 가이드**:
- 젤리 → Jelly (ゼリー, 果冻)
- 오븐 → Oven (オーブン, 烤箱)
- 소울잼 → Soul Jam (ソウルジャム, 灵魂宝石)
- 쿠키 왕국 → Cookie Kingdom (クッキー王国, 饼干王国)

**주의사항**:
- 캐릭터 이름은 음역하지 않고 공식 명칭 사용
- 스킬명/아이템명은 게임 내 공식 번역 참조
- 스토리 대사는 감정을 살려 번역"""

MULTI_AGENT_SYSTEM_PROMPT = """당신은 데이터 분석 멀티 에이전트 시스템의 코디네이터입니다.

**역할**:
여러 전문 분석 도구를 조율하여 복잡한 분석 요청을 처리합니다.

**사용 가능한 도구**:
1. 번역 품질 분석: 번역 결과물의 품질을 평가합니다
2. 텍스트 분류: 텍스트의 카테고리를 분류합니다
3. 유저 세그먼트 분석: 유저 행동 패턴을 분석합니다
4. 이상 탐지: 비정상적인 패턴을 감지합니다
5. 세계관 검색: RAG를 통해 세계관 정보를 검색합니다

**분석 절차**:
1. 요청 분석: 필요한 도구와 데이터 파악
2. 도구 실행: 관련 분석 도구 순차적/병렬 실행
3. 결과 통합: 각 도구의 결과를 종합
4. 인사이트 도출: 비즈니스 관점의 인사이트 제공"""

# ============================================
# Memory Settings
# ============================================

MAX_MEMORY_TURNS = 10

# ============================================
# Ranking Settings
# ============================================

DEFAULT_TOPN = 10
MAX_TOPN = 50

# ============================================
# Summary Triggers
# ============================================

SUMMARY_TRIGGERS = [
    "요약", "정리", "요점", "핵심", "한줄", "한 줄", "간단히", "짧게",
    "요약해줘", "요약해 줘", "정리해줘", "정리해 줘",
    "summary", "summarize", "tl;dr", "tldr", "brief"
]

# ============================================
# Translation Settings
# ============================================

TRANSLATION_CATEGORIES = [
    "UI",           # 인터페이스 텍스트
    "story",        # 스토리/대사
    "skill",        # 스킬 설명
    "dialog",       # 캐릭터 대화
    "item",         # 아이템명/설명
    "quest",        # 퀘스트
    "achievement",  # 업적
    "notice",       # 공지사항
    "event",        # 이벤트
    "tutorial",     # 튜토리얼
]

TRANSLATION_QUALITY_GRADES = {
    "excellent": {"min_score": 0.9, "description": "출시 가능", "color": "#22c55e"},
    "good": {"min_score": 0.8, "description": "경미한 수정 필요", "color": "#3b82f6"},
    "acceptable": {"min_score": 0.7, "description": "검토 필요", "color": "#f59e0b"},
    "needs_review": {"min_score": 0.0, "description": "재번역 필요", "color": "#ef4444"},
}

# ============================================
# API Rate Limits
# ============================================

RATE_LIMIT_REQUESTS_PER_MINUTE = 60
RATE_LIMIT_TOKENS_PER_MINUTE = 100000

# ============================================
# File Upload Settings
# ============================================

MAX_UPLOAD_SIZE_MB = 10
ALLOWED_EXTENSIONS = [".txt", ".pdf", ".docx", ".csv", ".json", ".md"]
