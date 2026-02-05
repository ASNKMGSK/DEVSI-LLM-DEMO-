"""
쿠키런 세계관 AI 플랫폼 - 데이터 생성 및 모델 학습
===============================================
데브시스터즈 기술혁신 프로젝트 포트폴리오

주요 기능:
1. 쿠키런 세계관 맞춤형 번역 지원 시스템용 데이터/모델
2. 임베딩 기반 지식 검색 시스템용 데이터/모델
3. 데이터 분석 의사결정 지원용 분석 모델
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
import json
import re

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    silhouette_score,
)
from sklearn.cluster import KMeans
import joblib
import warnings

warnings.filterwarnings("ignore")

# ========================================
# MLflow 설정 (선택적)
# ========================================
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True

    MLFLOW_TRACKING_URI = "file:./mlruns"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    EXPERIMENT_NAME = "cookierun-ai-platform"
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"MLflow Experiment: {EXPERIMENT_NAME}")
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not available - skipping experiment tracking")

seed = 42
rng = np.random.default_rng(seed)

# ========================================
# 저장 경로
# ========================================
# Jupyter notebook에서도 동작하도록 __file__ 대체 처리
try:
    BACKEND_DIR = Path(__file__).parent.parent
except NameError:
    # Jupyter notebook이나 interactive 환경에서 실행 시
    BACKEND_DIR = Path.cwd()
    if BACKEND_DIR.name == "ml":
        BACKEND_DIR = BACKEND_DIR.parent
    elif "backend" in str(BACKEND_DIR).lower() or "리팩토링" in str(BACKEND_DIR):
        pass  # 이미 backend 폴더에 있음
    else:
        # 기본 경로 사용
        BACKEND_DIR = Path.cwd()

BACKEND_DIR.mkdir(parents=True, exist_ok=True)

# ========================================
# 1. 쿠키런 세계관 데이터 생성
# ========================================
print("=" * 60)
print("1. 쿠키런 세계관 데이터 생성")
print("=" * 60)

# 쿠키 등급
COOKIE_GRADES = ["커먼", "레어", "슈퍼레어", "에픽", "레전더리", "에인션트"]

# 쿠키 타입
COOKIE_TYPES = ["돌격", "마법", "사격", "방어", "지원", "폭발", "치유", "복수"]

# 쿠키 캐릭터 데이터 (실제 쿠키런 킹덤 기반)
COOKIES = [
    # 커먼
    {"id": "CK001", "name": "용감한 쿠키", "name_en": "GingerBrave", "grade": "커먼", "type": "돌격",
     "story_kr": "오븐에서 탈출한 최초의 쿠키! 두려움 없이 달려나가는 용감한 쿠키입니다.",
     "story_en": "The first cookie to escape from the oven! A brave cookie who runs without fear."},
    {"id": "CK002", "name": "딸기맛 쿠키", "name_en": "Strawberry Cookie", "grade": "커먼", "type": "방어",
     "story_kr": "달콤한 딸기향이 나는 수줍은 쿠키입니다. 부끄러움이 많지만 친구를 위해선 용감해져요.",
     "story_en": "A shy cookie with a sweet strawberry scent. Though timid, becomes brave for friends."},
    # 레어
    {"id": "CK003", "name": "마법사맛 쿠키", "name_en": "Wizard Cookie", "grade": "레어", "type": "마법",
     "story_kr": "신비로운 마법을 부리는 쿠키입니다. 아직 수련 중이라 가끔 마법이 엉뚱하게 나가기도 해요.",
     "story_en": "A cookie who wields mysterious magic. Still in training, so spells sometimes go awry."},
    {"id": "CK004", "name": "닌자맛 쿠키", "name_en": "Ninja Cookie", "grade": "레어", "type": "돌격",
     "story_kr": "그림자처럼 빠르고 조용한 쿠키입니다. 비밀스러운 닌자 마을에서 수련을 받았어요.",
     "story_en": "A cookie as fast and silent as a shadow. Trained in a secret ninja village."},
    {"id": "CK005", "name": "천사맛 쿠키", "name_en": "Angel Cookie", "grade": "레어", "type": "치유",
     "story_kr": "하늘에서 내려온 천사 쿠키입니다. 따뜻한 빛으로 아군을 회복시켜요.",
     "story_en": "An angel cookie from the sky. Heals allies with warm light."},
    # 슈퍼레어
    {"id": "CK006", "name": "웨어울프맛 쿠키", "name_en": "Werewolf Cookie", "grade": "슈퍼레어", "type": "돌격",
     "story_kr": "달이 뜨면 변신하는 쿠키입니다. 강력한 힘을 숨기고 있어요.",
     "story_en": "A cookie who transforms under the moon. Hides tremendous power."},
    {"id": "CK007", "name": "칠리맛 쿠키", "name_en": "Chili Pepper Cookie", "grade": "슈퍼레어", "type": "돌격",
     "story_kr": "매운맛을 좋아하는 활발한 쿠키입니다. 뜨거운 성격만큼 강한 공격력을 자랑해요.",
     "story_en": "A spicy cookie who loves heat. Has powerful attacks matching her hot personality."},
    # 에픽
    {"id": "CK008", "name": "뱀파이어맛 쿠키", "name_en": "Vampire Cookie", "grade": "에픽", "type": "복수",
     "story_kr": "밤의 귀족 쿠키입니다. 와인잼을 좋아하며, 햇빛을 피해 성에서 살고 있어요.",
     "story_en": "A noble cookie of the night. Loves wine jam and lives in a castle away from sunlight."},
    {"id": "CK009", "name": "허브맛 쿠키", "name_en": "Herb Cookie", "grade": "에픽", "type": "치유",
     "story_kr": "식물을 사랑하는 치유사 쿠키입니다. 허브 정원에서 다양한 약초를 기르고 있어요.",
     "story_en": "A healer cookie who loves plants. Grows various herbs in the herb garden."},
    {"id": "CK010", "name": "감초맛 쿠키", "name_en": "Licorice Cookie", "grade": "에픽", "type": "마법",
     "story_kr": "어둠의 마법을 연구하는 쿠키입니다. 감초 하인들을 소환해 싸울 수 있어요.",
     "story_en": "A cookie researching dark magic. Can summon licorice servants to fight."},
    {"id": "CK011", "name": "에스프레소맛 쿠키", "name_en": "Espresso Cookie", "grade": "에픽", "type": "마법",
     "story_kr": "커피 마법학의 천재 쿠키입니다. 차가운 성격이지만 실력은 인정받고 있어요.",
     "story_en": "A genius cookie of coffee magic. Cold personality but skills are recognized."},
    {"id": "CK012", "name": "검은건포도맛 쿠키", "name_en": "Black Raisin Cookie", "grade": "에픽", "type": "돌격",
     "story_kr": "까마귀와 함께하는 미스터리한 쿠키입니다. 빠른 속도로 적을 베어버려요.",
     "story_en": "A mysterious cookie with crows. Slashes enemies at high speed."},
    {"id": "CK013", "name": "호밀맛 쿠키", "name_en": "Rye Cookie", "grade": "에픽", "type": "사격",
     "story_kr": "서부 출신의 총잡이 쿠키입니다. 정확한 사격 솜씨로 악당들을 물리쳐요.",
     "story_en": "A gunslinger cookie from the West. Defeats villains with precise shooting."},
    {"id": "CK014", "name": "석류맛 쿠키", "name_en": "Pomegranate Cookie", "grade": "에픽", "type": "지원",
     "story_kr": "어둠의 마녀를 섬기는 신비한 쿠키입니다. 아군의 공격력을 높여줘요.",
     "story_en": "A mysterious cookie serving the Dark Witch. Boosts allies' attack power."},
    {"id": "CK015", "name": "민트초코 쿠키", "name_en": "Mint Choco Cookie", "grade": "에픽", "type": "지원",
     "story_kr": "바이올린을 연주하는 음악가 쿠키입니다. 아름다운 선율로 아군을 버프해요.",
     "story_en": "A musician cookie who plays violin. Buffs allies with beautiful melodies."},
    # 에인션트
    {"id": "CK016", "name": "순수 바닐라 쿠키", "name_en": "Pure Vanilla Cookie", "grade": "에인션트", "type": "치유",
     "story_kr": "고대 영웅 중 한 명인 에인션트 쿠키입니다. 평화를 사랑하며 모든 쿠키를 치유해요.",
     "story_en": "One of the Ancient Heroes. Loves peace and heals all cookies."},
    {"id": "CK017", "name": "다크카카오 쿠키", "name_en": "Dark Cacao Cookie", "grade": "에인션트", "type": "돌격",
     "story_kr": "다크카카오 왕국의 왕입니다. 무거운 검을 휘두르며 왕국을 지키고 있어요.",
     "story_en": "King of the Dark Cacao Kingdom. Wields a heavy sword to protect the kingdom."},
    {"id": "CK018", "name": "홀리베리 쿠키", "name_en": "Hollyberry Cookie", "grade": "에인션트", "type": "방어",
     "story_kr": "홀리베리 왕국의 여왕입니다. 강인하고 용감하며 모든 쿠키들의 존경을 받아요.",
     "story_en": "Queen of the Hollyberry Kingdom. Strong and brave, respected by all cookies."},
    # 레전더리
    {"id": "CK019", "name": "바다요정 쿠키", "name_en": "Sea Fairy Cookie", "grade": "레전더리", "type": "폭발",
     "story_kr": "깊은 바다에서 온 신비로운 쿠키입니다. 엄청난 마법력으로 적을 얼려버려요.",
     "story_en": "A mysterious cookie from the deep sea. Freezes enemies with tremendous magic."},
    {"id": "CK020", "name": "서리여왕 쿠키", "name_en": "Frost Queen Cookie", "grade": "레전더리", "type": "마법",
     "story_kr": "얼음 왕국을 다스리는 여왕 쿠키입니다. 차가운 눈보라로 모든 것을 얼려버려요.",
     "story_en": "Queen cookie ruling the Ice Kingdom. Freezes everything with cold blizzards."},
]

# 게임 내 지역/왕국
KINGDOMS = [
    {"id": "KD001", "name": "쿠키 왕국", "name_en": "Cookie Kingdom",
     "desc_kr": "용감한 쿠키와 친구들이 세운 왕국입니다.", "desc_en": "A kingdom founded by GingerBrave and friends."},
    {"id": "KD002", "name": "다크카카오 왕국", "name_en": "Dark Cacao Kingdom",
     "desc_kr": "눈 덮인 산맥에 위치한 강인한 왕국입니다.", "desc_en": "A strong kingdom in the snowy mountains."},
    {"id": "KD003", "name": "홀리베리 왕국", "name_en": "Hollyberry Kingdom",
     "desc_kr": "용맹한 전사들의 고향입니다.", "desc_en": "Home of brave warriors."},
    {"id": "KD004", "name": "바닐라 왕국", "name_en": "Vanilla Kingdom",
     "desc_kr": "한때 번영했던 고대 왕국입니다.", "desc_en": "An ancient kingdom that once prospered."},
    {"id": "KD005", "name": "크레페 공화국", "name_en": "Crème Republic",
     "desc_kr": "기술과 과학이 발전한 도시입니다.", "desc_en": "A city of advanced technology and science."},
    {"id": "KD006", "name": "트로피컬 소다 섬", "name_en": "Tropical Soda Islands",
     "desc_kr": "파도가 넘실대는 열대의 낙원입니다.", "desc_en": "A tropical paradise where waves splash."},
    {"id": "KD007", "name": "용의 계곡", "name_en": "Dragon's Valley",
     "desc_kr": "고대 용들이 잠든 신비로운 계곡입니다.", "desc_en": "A mysterious valley where ancient dragons sleep."},
    {"id": "KD008", "name": "어둠의 영지", "name_en": "Dark Realm",
     "desc_kr": "어둠의 마녀가 지배하는 위험한 땅입니다.", "desc_en": "A dangerous land ruled by the Dark Witch."},
]

# 스킬 설명 데이터
SKILLS = [
    {"cookie_id": "CK001", "skill_name": "용감한 돌진", "skill_name_en": "Brave Dash",
     "desc_kr": "전방으로 돌진하며 적에게 피해를 줍니다.", "desc_en": "Dashes forward and damages enemies."},
    {"cookie_id": "CK002", "skill_name": "딸기 방패", "skill_name_en": "Strawberry Shield",
     "desc_kr": "딸기 방패로 아군을 보호합니다.", "desc_en": "Protects allies with a strawberry shield."},
    {"cookie_id": "CK003", "skill_name": "번개 마법", "skill_name_en": "Lightning Magic",
     "desc_kr": "강력한 번개를 소환하여 적을 공격합니다.", "desc_en": "Summons powerful lightning to attack enemies."},
    {"cookie_id": "CK004", "skill_name": "그림자 베기", "skill_name_en": "Shadow Slash",
     "desc_kr": "그림자 속에서 나타나 적을 빠르게 베어버립니다.", "desc_en": "Emerges from shadows to slash enemies swiftly."},
    {"cookie_id": "CK005", "skill_name": "천사의 축복", "skill_name_en": "Angel's Blessing",
     "desc_kr": "하늘의 빛으로 아군을 축복하고 회복시킵니다.", "desc_en": "Blesses and heals allies with heavenly light."},
    {"cookie_id": "CK006", "skill_name": "늑대의 포효", "skill_name_en": "Wolf's Howl",
     "desc_kr": "포효와 함께 변신하여 강력한 공격을 퍼붓습니다.", "desc_en": "Transforms with a howl and unleashes powerful attacks."},
    {"cookie_id": "CK007", "skill_name": "칠리 폭풍", "skill_name_en": "Chili Storm",
     "desc_kr": "뜨거운 불꽃을 일으켜 적들을 태워버립니다.", "desc_en": "Creates hot flames to burn enemies."},
    {"cookie_id": "CK008", "skill_name": "피의 연회", "skill_name_en": "Blood Feast",
     "desc_kr": "적의 생명력을 흡수하여 자신을 회복합니다.", "desc_en": "Absorbs enemy life force to heal self."},
    {"cookie_id": "CK009", "skill_name": "생명의 허브", "skill_name_en": "Herb of Life",
     "desc_kr": "아군 전체의 HP를 회복시킵니다.", "desc_en": "Heals HP of all allies."},
    {"cookie_id": "CK010", "skill_name": "어둠의 소환", "skill_name_en": "Dark Summon",
     "desc_kr": "감초 하인들을 소환하여 적을 공격합니다.", "desc_en": "Summons licorice servants to attack enemies."},
    {"cookie_id": "CK011", "skill_name": "커피 토네이도", "skill_name_en": "Coffee Tornado",
     "desc_kr": "거대한 커피 소용돌이로 적들을 휩쓸어버립니다.", "desc_en": "Sweeps enemies with a massive coffee tornado."},
    {"cookie_id": "CK012", "skill_name": "까마귀 칼날", "skill_name_en": "Raven's Blade",
     "desc_kr": "까마귀와 함께 빠르게 돌진하여 적을 베어버립니다.", "desc_en": "Rushes with crows to slash enemies."},
    {"cookie_id": "CK013", "skill_name": "정의의 총탄", "skill_name_en": "Justice Bullets",
     "desc_kr": "정확한 사격으로 적들을 처치합니다.", "desc_en": "Defeats enemies with precise shooting."},
    {"cookie_id": "CK014", "skill_name": "석류의 저주", "skill_name_en": "Pomegranate's Curse",
     "desc_kr": "아군의 공격력을 올리고 적에게 저주를 겁니다.", "desc_en": "Boosts ally attack and curses enemies."},
    {"cookie_id": "CK015", "skill_name": "선율의 물결", "skill_name_en": "Wave of Melody",
     "desc_kr": "아름다운 선율로 아군을 강화시킵니다.", "desc_en": "Strengthens allies with beautiful melody."},
    {"cookie_id": "CK016", "skill_name": "순수한 치유", "skill_name_en": "Pure Healing",
     "desc_kr": "신성한 빛으로 아군을 치유하고 보호합니다.", "desc_en": "Heals and protects allies with holy light."},
    {"cookie_id": "CK017", "skill_name": "다크카카오의 심판", "skill_name_en": "Dark Cacao's Judgment",
     "desc_kr": "거대한 검으로 적에게 강력한 일격을 가합니다.", "desc_en": "Strikes enemies with a massive sword."},
    {"cookie_id": "CK018", "skill_name": "홀리베리의 축복", "skill_name_en": "Hollyberry's Blessing",
     "desc_kr": "강력한 방패로 아군을 보호하고 적을 밀쳐냅니다.", "desc_en": "Protects allies with a shield and pushes enemies."},
    {"cookie_id": "CK019", "skill_name": "해일 소환", "skill_name_en": "Summon Tidal Wave",
     "desc_kr": "거대한 해일을 소환하여 적을 휩쓸어버립니다.", "desc_en": "Summons a massive tidal wave to sweep enemies."},
    {"cookie_id": "CK020", "skill_name": "절대 영도", "skill_name_en": "Absolute Zero",
     "desc_kr": "모든 것을 얼려버리는 눈보라를 소환합니다.", "desc_en": "Summons a blizzard that freezes everything."},
]

# DataFrame 생성
cookies_df = pd.DataFrame(COOKIES)
kingdoms_df = pd.DataFrame(KINGDOMS)
skills_df = pd.DataFrame(SKILLS)

print(f"쿠키 캐릭터: {len(cookies_df)}개")
print(f"왕국/지역: {len(kingdoms_df)}개")
print(f"스킬 설명: {len(skills_df)}개")

# ========================================
# 2. 번역 데이터셋 생성 (한국어 -> 다국어)
# ========================================
print("\n" + "=" * 60)
print("2. 번역 데이터셋 생성")
print("=" * 60)

# 번역 품질 점수 (가상 데이터)
# 실제로는 번역가 평가 또는 자동 품질 측정 점수
TRANSLATION_PAIRS = []

# 일반 게임 텍스트
GAME_TEXTS = [
    {"text_id": "TXT001", "category": "UI", "ko": "전투 시작", "en": "Start Battle", "ja": "バトル開始", "zh": "开始战斗"},
    {"text_id": "TXT002", "category": "UI", "ko": "스테이지 클리어", "en": "Stage Clear", "ja": "ステージクリア", "zh": "关卡通关"},
    {"text_id": "TXT003", "category": "UI", "ko": "보물을 획득했습니다!", "en": "Treasure obtained!", "ja": "宝物を獲得しました！", "zh": "获得了宝物！"},
    {"text_id": "TXT004", "category": "story", "ko": "오븐에서 탈출해야 해!", "en": "I must escape from the oven!", "ja": "オーブンから脱出しなきゃ！", "zh": "必须从烤箱逃出去！"},
    {"text_id": "TXT005", "category": "story", "ko": "친구들, 같이 가자!", "en": "Friends, let's go together!", "ja": "みんな、一緒に行こう！", "zh": "朋友们，一起走吧！"},
    {"text_id": "TXT006", "category": "skill", "ko": "나의 힘을 받아라!", "en": "Feel my power!", "ja": "私の力を受けろ！", "zh": "接受我的力量吧！"},
    {"text_id": "TXT007", "category": "skill", "ko": "모두를 지켜줄게!", "en": "I'll protect everyone!", "ja": "みんなを守るよ！", "zh": "我会保护大家的！"},
    {"text_id": "TXT008", "category": "dialog", "ko": "이 맛은... 마녀의 저주!", "en": "This taste... the witch's curse!", "ja": "この味は…魔女の呪い！", "zh": "这个味道...是魔女的诅咒！"},
    {"text_id": "TXT009", "category": "dialog", "ko": "쿠키들의 왕국을 세우자!", "en": "Let's build a kingdom for cookies!", "ja": "クッキーたちの王国を作ろう！", "zh": "建立饼干们的王国吧！"},
    {"text_id": "TXT010", "category": "item", "ko": "체력 회복 젤리", "en": "HP Recovery Jelly", "ja": "体力回復ゼリー", "zh": "体力恢复果冻"},
]

# 세계관 고유 용어 (번역 시 주의 필요)
WORLDVIEW_TERMS = [
    {"term_ko": "젤리", "term_en": "Jelly", "context": "게임 내 화폐/아이템"},
    {"term_ko": "오븐", "term_en": "Oven", "context": "쿠키가 태어나는 곳"},
    {"term_ko": "소울잼", "term_en": "Soul Jam", "context": "고대의 신비한 보석"},
    {"term_ko": "마녀", "term_en": "Witch", "context": "쿠키들의 적"},
    {"term_ko": "다크엔챈트리스 쿠키", "term_en": "Dark Enchantress Cookie", "context": "메인 빌런"},
    {"term_ko": "트로피컬 소다 섬", "term_en": "Tropical Soda Islands", "context": "지역명"},
    {"term_ko": "어둠의 군단", "term_en": "Darkness Legion", "context": "적 세력"},
]

# 번역 품질 데이터 생성 (실제로는 번역가 평가 기반)
translation_data = []

for text in GAME_TEXTS:
    # 각 언어쌍에 대해 품질 점수 생성
    for target_lang, target_text in [("en", text["en"]), ("ja", text["ja"]), ("zh", text["zh"])]:
        quality_score = rng.uniform(0.7, 1.0)  # 가상 품질 점수
        fluency = rng.uniform(0.6, 1.0)
        adequacy = rng.uniform(0.7, 1.0)

        # 세계관 용어 포함 여부 체크
        contains_worldview_term = any(term["term_ko"] in text["ko"] for term in WORLDVIEW_TERMS)

        translation_data.append({
            "text_id": text["text_id"],
            "category": text["category"],
            "source_text": text["ko"],
            "target_text": target_text,
            "target_lang": target_lang,
            "quality_score": round(quality_score, 3),
            "fluency_score": round(fluency, 3),
            "adequacy_score": round(adequacy, 3),
            "contains_worldview_term": int(contains_worldview_term),
            "text_length": len(text["ko"]),
        })

# 추가 합성 데이터 생성 (더 많은 학습 데이터)
CATEGORIES = ["UI", "story", "skill", "dialog", "item", "quest", "achievement", "notice"]

for i in range(200):
    category = rng.choice(CATEGORIES)
    text_length = rng.integers(5, 100)
    contains_term = rng.choice([0, 1], p=[0.7, 0.3])

    # 품질 점수는 카테고리와 세계관 용어 포함 여부에 따라 다르게
    base_quality = 0.8 if category in ["UI", "item"] else 0.75
    if contains_term:
        base_quality -= 0.05  # 세계관 용어 있으면 번역 난이도 상승

    for target_lang in ["en", "ja", "zh"]:
        quality_score = np.clip(rng.normal(base_quality, 0.1), 0.4, 1.0)
        fluency = np.clip(rng.normal(0.8, 0.1), 0.5, 1.0)
        adequacy = np.clip(rng.normal(0.85, 0.08), 0.5, 1.0)

        translation_data.append({
            "text_id": f"SYN{i:04d}",
            "category": category,
            "source_text": f"[합성 텍스트 {i}]",
            "target_text": f"[Synthetic {i}]",
            "target_lang": target_lang,
            "quality_score": round(quality_score, 3),
            "fluency_score": round(fluency, 3),
            "adequacy_score": round(adequacy, 3),
            "contains_worldview_term": contains_term,
            "text_length": text_length,
        })

translation_df = pd.DataFrame(translation_data)
print(f"번역 데이터: {len(translation_df)}개")

# ========================================
# 3. 게임 로그 데이터 생성 (분석용) - 현실적 패턴 적용
# ========================================
print("\n" + "=" * 60)
print("3. 게임 로그 데이터 생성 (현실적 패턴)")
print("=" * 60)

n_users = 1000

# 유저 데이터 생성
users_df = pd.DataFrame({
    "user_id": [f"U{i:06d}" for i in range(1, n_users + 1)],
    "country": rng.choice(["KR", "US", "JP", "CN", "TW", "TH", "ID"], n_users,
                          p=[0.3, 0.2, 0.15, 0.15, 0.08, 0.07, 0.05]),
    "register_date": pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, n_users), unit="D"),
    "vip_level": rng.choice([0, 1, 2, 3, 4, 5], n_users, p=[0.5, 0.25, 0.12, 0.08, 0.03, 0.02]),
})

# 유저별 활동량 (파레토 분포 - 상위 20%가 80% 활동)
user_activity_weights = rng.pareto(1.5, n_users)
user_activity_weights = user_activity_weights / user_activity_weights.sum()

# 시간대별 활동 가중치 (저녁/밤에 더 활발)
HOUR_WEIGHTS = {
    0: 0.3, 1: 0.2, 2: 0.1, 3: 0.05, 4: 0.03, 5: 0.02,
    6: 0.05, 7: 0.1, 8: 0.3, 9: 0.5, 10: 0.6, 11: 0.7,
    12: 0.9, 13: 0.8, 14: 0.7, 15: 0.6, 16: 0.7, 17: 0.8,
    18: 1.0, 19: 1.2, 20: 1.5, 21: 1.8, 22: 1.5, 23: 0.8,
}

# 요일별 활동 가중치 (주말에 더 활발)
DAY_WEIGHTS = {0: 1.0, 1: 0.8, 2: 0.8, 3: 0.9, 4: 1.0, 5: 1.3, 6: 1.4}  # 월~일

# 쿠키별 인기도 (레전더리/에인션트가 더 많이 사용됨)
cookie_ids = [c["id"] for c in COOKIES]
cookie_grades = {c["id"]: c["grade"] for c in COOKIES}
grade_popularity = {"커먼": 0.5, "레어": 0.8, "슈퍼레어": 1.0, "에픽": 1.5, "에인션트": 2.0, "레전더리": 2.5}
cookie_weights = np.array([grade_popularity[cookie_grades[cid]] for cid in cookie_ids])
cookie_weights = cookie_weights / cookie_weights.sum()

# 게임 이벤트 로그
EVENT_TYPES = ["stage_clear", "gacha_pull", "cookie_upgrade", "pvp_battle", "guild_activity", "shop_purchase", "login"]

logs = []
base_date = pd.to_datetime("2024-06-01")
end_date = pd.to_datetime("2024-11-28")
n_days = (end_date - base_date).days

print("게임 로그 생성 중 (현실적 패턴 적용)...")

for day_offset in tqdm(range(n_days), desc="일별 로그 생성"):
    current_date = base_date + pd.Timedelta(days=day_offset)
    day_of_week = current_date.weekday()

    # 해당 일에 활동할 유저 수 (요일에 따라 다름)
    base_active_ratio = 0.3 * DAY_WEIGHTS[day_of_week]
    n_active_users = int(n_users * base_active_ratio)

    # 활동량 기반으로 유저 선택 (자주 하는 유저가 더 자주 선택됨)
    active_users = rng.choice(users_df["user_id"].values, n_active_users,
                               p=user_activity_weights, replace=True)
    active_users = list(set(active_users))  # 중복 제거

    for user in active_users:
        user_vip = users_df[users_df["user_id"] == user]["vip_level"].values[0]

        # 유저별 세션 수 (VIP일수록 더 많이 플레이)
        n_sessions = rng.integers(1, 3 + user_vip)

        for _ in range(n_sessions):
            # 세션 시작 시간 (시간대 가중치 적용)
            hour_probs = np.array([HOUR_WEIGHTS[h] for h in range(24)])
            hour_probs = hour_probs / hour_probs.sum()
            session_hour = rng.choice(range(24), p=hour_probs)
            session_minute = rng.integers(0, 60)

            session_time = current_date + pd.Timedelta(hours=int(session_hour), minutes=int(session_minute))

            # 로그인 이벤트
            logs.append({
                "log_id": f"LOG{len(logs):08d}",
                "user_id": user,
                "event_type": "login",
                "event_date": session_time,
                "detail": json.dumps({}, ensure_ascii=False),
            })

            # 세션 내 활동 수 (5~20개)
            n_events = rng.integers(5, 15 + user_vip * 3)

            for event_idx in range(n_events):
                event_time = session_time + pd.Timedelta(minutes=int(event_idx * rng.integers(1, 5)))
                event_type = rng.choice(EVENT_TYPES[:-1], p=[0.35, 0.12, 0.15, 0.15, 0.13, 0.10])

                # 이벤트별 상세 데이터
                if event_type == "stage_clear":
                    stage = int(rng.integers(1, 500))
                    stars = int(rng.choice([1, 2, 3], p=[0.15, 0.35, 0.5]))
                    used_cookie = rng.choice(cookie_ids, p=cookie_weights)
                    detail = {"stage": stage, "stars": stars, "cookie_id": used_cookie}
                elif event_type == "gacha_pull":
                    pull_type = rng.choice(["single", "ten_pull"], p=[0.25, 0.75])
                    # 에픽 이상 확률 실제 게임 반영
                    got_grade = rng.choice(["커먼", "레어", "슈퍼레어", "에픽", "레전더리"],
                                           p=[0.30, 0.35, 0.25, 0.09, 0.01])
                    detail = {"pull_type": pull_type, "result_grade": got_grade}
                elif event_type == "pvp_battle":
                    win = int(rng.choice([0, 1], p=[0.48, 0.52]))
                    opponent_tier = rng.choice(["브론즈", "실버", "골드", "다이아", "마스터"],
                                               p=[0.1, 0.25, 0.35, 0.2, 0.1])
                    used_cookie = rng.choice(cookie_ids, p=cookie_weights)
                    detail = {"result": "win" if win else "lose", "opponent_tier": opponent_tier, "cookie_id": used_cookie}
                elif event_type == "cookie_upgrade":
                    upgraded_cookie = rng.choice(cookie_ids, p=cookie_weights)
                    new_level = int(rng.integers(1, 70))
                    detail = {"cookie_id": upgraded_cookie, "new_level": new_level}
                elif event_type == "guild_activity":
                    activity = rng.choice(["guild_boss", "donation", "chat"], p=[0.4, 0.35, 0.25])
                    detail = {"activity_type": activity}
                elif event_type == "shop_purchase":
                    item_type = rng.choice(["crystal", "coin", "stamina", "package"], p=[0.3, 0.3, 0.25, 0.15])
                    amount = int(rng.choice([100, 500, 1000, 5000], p=[0.5, 0.3, 0.15, 0.05]))
                    detail = {"item_type": item_type, "amount": amount}
                else:
                    detail = {}

                logs.append({
                    "log_id": f"LOG{len(logs):08d}",
                    "user_id": user,
                    "event_type": event_type,
                    "event_date": event_time,
                    "detail": json.dumps(detail, ensure_ascii=False),
                })

logs_df = pd.DataFrame(logs)
print(f"유저 데이터: {len(users_df)}명")
print(f"게임 로그: {len(logs_df)}건")
print(f"기간: {logs_df['event_date'].min()} ~ {logs_df['event_date'].max()}")

# ========================================
# 4. 모델 1: 번역 품질 예측 모델
# ========================================
print("\n" + "=" * 60)
print("모델 1: 번역 품질 예측 (RandomForest Classifier)")
print("=" * 60)

# 품질 점수를 등급으로 변환
def quality_to_grade(score):
    if score >= 0.9:
        return "excellent"
    elif score >= 0.8:
        return "good"
    elif score >= 0.7:
        return "acceptable"
    else:
        return "needs_review"

translation_df["quality_grade"] = translation_df["quality_score"].apply(quality_to_grade)

# 피처 준비
le_category = LabelEncoder()
le_lang = LabelEncoder()
le_quality = LabelEncoder()

translation_df["category_encoded"] = le_category.fit_transform(translation_df["category"])
translation_df["target_lang_encoded"] = le_lang.fit_transform(translation_df["target_lang"])
translation_df["quality_encoded"] = le_quality.fit_transform(translation_df["quality_grade"])

feature_cols_translation = [
    "category_encoded",
    "target_lang_encoded",
    "fluency_score",
    "adequacy_score",
    "contains_worldview_term",
    "text_length",
]

X_trans = translation_df[feature_cols_translation].copy()
y_trans = translation_df["quality_encoded"].copy()

X_train_trans, X_test_trans, y_train_trans, y_test_trans = train_test_split(
    X_trans, y_trans, test_size=0.2, random_state=seed, stratify=y_trans
)

# 모델 학습 (MLflow 선택적)
params = {"n_estimators": 150, "max_depth": 10, "random_state": seed, "class_weight": "balanced"}
rf_translation = RandomForestClassifier(**params, n_jobs=-1)
rf_translation.fit(X_train_trans, y_train_trans)

y_pred_trans = rf_translation.predict(X_test_trans)
accuracy = accuracy_score(y_test_trans, y_pred_trans)
f1_macro = f1_score(y_test_trans, y_pred_trans, average="macro")

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 (macro): {f1_macro:.4f}")

if MLFLOW_AVAILABLE:
    with mlflow.start_run(run_name="translation_quality_model"):
        mlflow.set_tag("model_type", "classification")
        mlflow.set_tag("target", "translation_quality_grade")
        mlflow.set_tag("domain", "cookierun")
        mlflow.log_params(params)
        mlflow.log_param("n_features", len(feature_cols_translation))
        mlflow.log_param("train_samples", len(X_train_trans))
        mlflow.log_param("test_samples", len(X_test_trans))
        mlflow.log_metrics({"accuracy": accuracy, "f1_macro": f1_macro})
        mlflow.sklearn.log_model(rf_translation, "translation_quality_model",
                                 registered_model_name="번역품질예측")
        print(f"[MLflow] Run ID: {mlflow.active_run().info.run_id}")

print("\n[Classification Report]")
print(classification_report(y_test_trans, y_pred_trans, target_names=le_quality.classes_))

# ========================================
# 5. 모델 2: 텍스트 카테고리 분류
# ========================================
print("\n" + "=" * 60)
print("모델 2: 텍스트 카테고리 분류 (TF-IDF + RandomForest)")
print("=" * 60)

# 세계관 텍스트 데이터 준비
worldview_texts = []

# 쿠키 스토리
for cookie in COOKIES:
    worldview_texts.append({
        "text": cookie["story_kr"],
        "category": "character_story",
        "entity_id": cookie["id"],
    })

# 왕국 설명
for kingdom in KINGDOMS:
    worldview_texts.append({
        "text": kingdom["desc_kr"],
        "category": "location",
        "entity_id": kingdom["id"],
    })

# 스킬 설명
for skill in SKILLS:
    worldview_texts.append({
        "text": skill["desc_kr"],
        "category": "skill",
        "entity_id": skill["cookie_id"],
    })

# 게임 텍스트
for text in GAME_TEXTS:
    worldview_texts.append({
        "text": text["ko"],
        "category": text["category"],
        "entity_id": text["text_id"],
    })

worldview_df = pd.DataFrame(worldview_texts)

# TF-IDF 벡터화
tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(worldview_df["text"])

le_text_category = LabelEncoder()
y_text_cat = le_text_category.fit_transform(worldview_df["category"])

X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(
    X_tfidf, y_text_cat, test_size=0.2, random_state=seed
)

params = {"n_estimators": 100, "max_depth": 8, "random_state": seed}
rf_text = RandomForestClassifier(**params, n_jobs=-1)
rf_text.fit(X_train_text, y_train_text)

y_pred_text = rf_text.predict(X_test_text)
accuracy_text = accuracy_score(y_test_text, y_pred_text)
f1_text = f1_score(y_test_text, y_pred_text, average="macro")

print(f"Accuracy: {accuracy_text:.4f}")
print(f"F1 (macro): {f1_text:.4f}")

if MLFLOW_AVAILABLE:
    with mlflow.start_run(run_name="text_category_model"):
        mlflow.set_tag("model_type", "text_classification")
        mlflow.set_tag("vectorizer", "TF-IDF")
        mlflow.log_params(params)
        mlflow.log_metrics({"accuracy": accuracy_text, "f1_macro": f1_text})
        mlflow.sklearn.log_model(rf_text, "text_category_model",
                                 registered_model_name="텍스트분류")
        print(f"[MLflow] Run ID: {mlflow.active_run().info.run_id}")

# ========================================
# 6. 모델 3: 유저 세그먼트 클러스터링
# ========================================
print("\n" + "=" * 60)
print("모델 3: 유저 세그먼트 클러스터링 (K-Means)")
print("=" * 60)

# 유저별 집계 데이터
logs_df["event_date"] = pd.to_datetime(logs_df["event_date"])

user_agg = logs_df.groupby("user_id").agg(
    total_events=("log_id", "count"),
    stage_clears=("event_type", lambda x: (x == "stage_clear").sum()),
    gacha_pulls=("event_type", lambda x: (x == "gacha_pull").sum()),
    pvp_battles=("event_type", lambda x: (x == "pvp_battle").sum()),
    purchases=("event_type", lambda x: (x == "shop_purchase").sum()),
).reset_index()

user_agg = user_agg.merge(users_df[["user_id", "vip_level"]], on="user_id", how="left")

# 클러스터링 피처
cluster_features = ["total_events", "stage_clears", "gacha_pulls", "pvp_battles", "purchases", "vip_level"]
X_cluster = user_agg[cluster_features].fillna(0)

scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
cluster_labels = kmeans.fit_predict(X_cluster_scaled)

silhouette = silhouette_score(X_cluster_scaled, cluster_labels)
inertia = kmeans.inertia_

print(f"Silhouette Score: {silhouette:.4f}")
print(f"Inertia: {inertia:.2f}")

if MLFLOW_AVAILABLE:
    with mlflow.start_run(run_name="user_segmentation_model"):
        mlflow.set_tag("model_type", "clustering")
        mlflow.set_tag("algorithm", "KMeans")
        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_metrics({"silhouette_score": silhouette, "inertia": inertia})
        mlflow.sklearn.log_model(kmeans, "user_segmentation_model",
                                 registered_model_name="유저세그먼트")
        print(f"[MLflow] Run ID: {mlflow.active_run().info.run_id}")

user_agg["cluster"] = cluster_labels
print("\n[클러스터별 유저 수]")
print(user_agg["cluster"].value_counts().sort_index())

# 클러스터별 특성
SEGMENT_NAMES = {
    0: "캐주얼 플레이어",
    1: "하드코어 게이머",
    2: "PvP 전문가",
    3: "콘텐츠 수집가",
    4: "신규 유저",
}

# ========================================
# 7. 모델 4: 이상 탐지 (비정상 유저 행동)
# ========================================
print("\n" + "=" * 60)
print("모델 4: 이상 탐지 (Isolation Forest)")
print("=" * 60)

params = {"n_estimators": 150, "contamination": 0.05, "random_state": seed}
iso_forest = IsolationForest(**params)
anomaly_pred = iso_forest.fit_predict(X_cluster_scaled)
anomaly_scores = iso_forest.decision_function(X_cluster_scaled)

anomaly_count = int((anomaly_pred == -1).sum())
normal_count = int((anomaly_pred == 1).sum())

print(f"정상 유저: {normal_count}명 ({normal_count/len(anomaly_pred)*100:.1f}%)")
print(f"이상 유저: {anomaly_count}명 ({anomaly_count/len(anomaly_pred)*100:.1f}%)")

if MLFLOW_AVAILABLE:
    with mlflow.start_run(run_name="anomaly_detection_model"):
        mlflow.set_tag("model_type", "anomaly_detection")
        mlflow.log_params(params)
        mlflow.log_metrics({
            "anomaly_count": anomaly_count,
            "normal_count": normal_count,
            "anomaly_ratio": anomaly_count / len(anomaly_pred),
        })
        mlflow.sklearn.log_model(iso_forest, "anomaly_model",
                                 registered_model_name="이상탐지")
        print(f"[MLflow] Run ID: {mlflow.active_run().info.run_id}")

user_agg["is_anomaly"] = (anomaly_pred == -1).astype(int)
user_agg["anomaly_score"] = anomaly_scores

# ========================================
# 7.5. 모델 5: 이탈 예측 (Churn Prediction) + SHAP
# ========================================
print("\n" + "=" * 60)
print("모델 5: 이탈 예측 (RandomForest + SHAP)")
print("=" * 60)

# SHAP 라이브러리 체크
try:
    import shap
    SHAP_AVAILABLE = True
    print("SHAP 라이브러리 로드 완료")
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available - pip install shap 으로 설치 필요")

# 이탈 레이블 생성
# 기준: 최근 활동 데이터에서 마지막 7일 플레이타임이 이전 대비 70% 이상 감소하면 이탈 위험
print("\n이탈 레이블 생성 중...")

# user_activity_df가 아직 없으면 생성 (위에서 logs_df 기반으로 생성)
# 여기서는 user_agg 기반으로 이탈 피처 생성

# 이탈 예측 피처 정의
CHURN_FEATURES = [
    "total_events",       # 총 이벤트 수
    "stage_clears",       # 스테이지 클리어 수
    "gacha_pulls",        # 가챠 횟수
    "pvp_battles",        # PvP 전투 수
    "purchases",          # 구매 횟수
    "vip_level",          # VIP 레벨
]

CHURN_FEATURE_NAMES_KR = {
    "total_events": "총 활동량",
    "stage_clears": "스테이지 클리어",
    "gacha_pulls": "가챠 횟수",
    "pvp_battles": "PvP 전투",
    "purchases": "인앱 구매",
    "vip_level": "VIP 레벨",
}

# 이탈 레이블 생성 (휴리스틱 기반)
# 낮은 활동량 + 낮은 VIP = 이탈 위험 높음
def create_churn_label(row):
    """이탈 위험 레이블 생성 (0: 유지, 1: 이탈 위험)"""
    risk_score = 0

    # 낮은 총 이벤트 (하위 25%)
    if row["total_events"] < 30:
        risk_score += 2
    elif row["total_events"] < 45:
        risk_score += 1

    # 낮은 스테이지 클리어
    if row["stage_clears"] < 10:
        risk_score += 2
    elif row["stage_clears"] < 15:
        risk_score += 1

    # 낮은 가챠 (engagement 지표)
    if row["gacha_pulls"] < 3:
        risk_score += 1

    # 낮은 PvP (소셜 engagement)
    if row["pvp_battles"] < 5:
        risk_score += 1

    # 구매 없음
    if row["purchases"] == 0:
        risk_score += 1

    # VIP 0이면 무과금
    if row["vip_level"] == 0:
        risk_score += 1

    # 이상 유저는 이탈 위험 높음
    if row.get("is_anomaly", 0) == 1:
        risk_score += 2

    # 임계값 기반 이탈 판정 (risk_score >= 5면 이탈 위험)
    return 1 if risk_score >= 5 else 0

user_agg["churn_risk"] = user_agg.apply(create_churn_label, axis=1)

churn_count = user_agg["churn_risk"].sum()
retain_count = len(user_agg) - churn_count
print(f"유지 유저: {retain_count}명 ({retain_count/len(user_agg)*100:.1f}%)")
print(f"이탈 위험 유저: {churn_count}명 ({churn_count/len(user_agg)*100:.1f}%)")

# 이탈 예측 모델 학습
X_churn = user_agg[CHURN_FEATURES].fillna(0)
y_churn = user_agg["churn_risk"]

X_train_churn, X_test_churn, y_train_churn, y_test_churn = train_test_split(
    X_churn, y_churn, test_size=0.2, random_state=seed, stratify=y_churn
)

churn_params = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": seed,
}
rf_churn = RandomForestClassifier(**churn_params, n_jobs=-1)
rf_churn.fit(X_train_churn, y_train_churn)

y_pred_churn = rf_churn.predict(X_test_churn)
y_proba_churn = rf_churn.predict_proba(X_test_churn)[:, 1]

accuracy_churn = accuracy_score(y_test_churn, y_pred_churn)
f1_churn = f1_score(y_test_churn, y_pred_churn)

print(f"\nAccuracy: {accuracy_churn:.4f}")
print(f"F1 Score: {f1_churn:.4f}")

# Feature Importance (기본)
feature_importances = dict(zip(CHURN_FEATURES, rf_churn.feature_importances_))
print("\n[Feature Importance (기본)]")
for feat, imp in sorted(feature_importances.items(), key=lambda x: -x[1]):
    print(f"  {CHURN_FEATURE_NAMES_KR.get(feat, feat)}: {imp:.4f}")

# SHAP 분석
shap_explainer = None
shap_values_all = None

if SHAP_AVAILABLE:
    print("\n[SHAP 분석 시작]")
    try:
        # TreeExplainer 사용 (RandomForest에 최적화)
        shap_explainer = shap.TreeExplainer(rf_churn)

        # 전체 데이터에 대한 SHAP values 계산
        shap_values_raw = shap_explainer.shap_values(X_churn)

        # SHAP 버전에 따른 처리
        # 최신 SHAP: shap.Explanation 객체 또는 numpy array
        # 이전 SHAP: list of arrays (클래스별)
        if hasattr(shap_values_raw, 'values'):
            # shap.Explanation 객체인 경우
            shap_values_all = shap_values_raw.values
        elif isinstance(shap_values_raw, list) and len(shap_values_raw) == 2:
            # 이진 분류에서 [class_0_shap, class_1_shap] 리스트인 경우
            shap_values_all = shap_values_raw[1]  # 이탈 클래스(1)의 SHAP values
        elif isinstance(shap_values_raw, np.ndarray):
            # 이미 numpy array인 경우
            if shap_values_raw.ndim == 3:
                # (n_samples, n_features, n_classes) 형태
                shap_values_all = shap_values_raw[:, :, 1]
            else:
                shap_values_all = shap_values_raw
        else:
            shap_values_all = shap_values_raw

        # numpy array로 변환
        shap_values_all = np.array(shap_values_all)

        # Global Feature Importance (SHAP 기반)
        shap_importance = np.abs(shap_values_all).mean(axis=0)
        shap_feature_importance = dict(zip(CHURN_FEATURES, shap_importance))

        print("\n[Feature Importance (SHAP 기반)]")
        for feat, imp in sorted(shap_feature_importance.items(), key=lambda x: -x[1]):
            print(f"  {CHURN_FEATURE_NAMES_KR.get(feat, feat)}: {imp:.4f}")

        # SHAP values를 user_agg에 저장 (유저별 요인 분석용)
        for i, feat in enumerate(CHURN_FEATURES):
            user_agg[f"shap_{feat}"] = shap_values_all[:, i]

        print(f"\n유저별 SHAP values 저장 완료 ({len(CHURN_FEATURES)}개 피처)")

    except Exception as e:
        import traceback
        print(f"SHAP 분석 오류: {e}")
        traceback.print_exc()
        SHAP_AVAILABLE = False

# 이탈 확률 저장
user_agg["churn_probability"] = rf_churn.predict_proba(X_churn)[:, 1]

# 이탈 위험 등급 분류
def get_risk_level(prob):
    if prob >= 0.7:
        return "high"
    elif prob >= 0.4:
        return "medium"
    else:
        return "low"

user_agg["churn_risk_level"] = user_agg["churn_probability"].apply(get_risk_level)

# MLflow 로깅
if MLFLOW_AVAILABLE:
    with mlflow.start_run(run_name="churn_prediction_model"):
        mlflow.set_tag("model_type", "classification")
        mlflow.set_tag("target", "churn_prediction")
        mlflow.set_tag("shap_enabled", str(SHAP_AVAILABLE))
        mlflow.log_params(churn_params)
        mlflow.log_metrics({
            "accuracy": accuracy_churn,
            "f1_score": f1_churn,
            "churn_count": int(churn_count),
            "retain_count": int(retain_count),
            "churn_ratio": churn_count / len(user_agg),
        })
        mlflow.sklearn.log_model(rf_churn, "churn_model",
                                 registered_model_name="이탈예측")
        print(f"[MLflow] Run ID: {mlflow.active_run().info.run_id}")

# 모델 저장
joblib.dump(rf_churn, BACKEND_DIR / "model_churn.pkl")
print(f"\n모델 저장: model_churn.pkl")

# SHAP Explainer 저장 (있으면)
if SHAP_AVAILABLE and shap_explainer is not None:
    joblib.dump(shap_explainer, BACKEND_DIR / "shap_explainer_churn.pkl")
    print(f"SHAP Explainer 저장: shap_explainer_churn.pkl")

# Feature names 저장 (API에서 사용)
churn_config = {
    "features": CHURN_FEATURES,
    "feature_names_kr": CHURN_FEATURE_NAMES_KR,
    "feature_importances": {k: float(v) for k, v in feature_importances.items()},
    "shap_available": SHAP_AVAILABLE,
    "model_accuracy": float(accuracy_churn),
    "model_f1": float(f1_churn),
}
with open(BACKEND_DIR / "churn_model_config.json", "w", encoding="utf-8") as f:
    json.dump(churn_config, f, ensure_ascii=False, indent=2)
print(f"모델 설정 저장: churn_model_config.json")

print("\n[이탈 위험 등급별 분포]")
print(user_agg["churn_risk_level"].value_counts())

# ========================================
# 7.6. 모델 6: 쿠키 추천 (LightFM via recommenders)
# ========================================
print("\n" + "=" * 60)
print("모델 6: 쿠키 추천 (LightFM via recommenders)")
print("=" * 60)

# recommenders + LightFM 라이브러리 체크
LIGHTFM_AVAILABLE = False
RECOMMENDERS_UTILS_AVAILABLE = False
try:
    from lightfm import LightFM
    from lightfm.data import Dataset
    from lightfm.evaluation import precision_at_k, auc_score
    LIGHTFM_AVAILABLE = True
    print("LightFM 라이브러리 로드 완료")

    # recommenders 유틸리티 함수들 (선택적)
    try:
        from recommenders.models.lightfm.lightfm_utils import (
            similar_items,
            track_model_metrics,
        )
        RECOMMENDERS_UTILS_AVAILABLE = True
        print("recommenders.models.lightfm 유틸리티 로드 완료")
    except ImportError:
        print("recommenders 유틸리티 미사용 (기본 LightFM만 사용)")

except ImportError as e:
    print(f"LightFM 로드 실패: {e}")
    print("설치: pip install recommenders[lightfm]")

if LIGHTFM_AVAILABLE:
    print("\n유저-쿠키 상호작용 데이터 생성 중...")

    # 유저-쿠키 상호작용 데이터 생성
    # (어떤 유저가 어떤 쿠키를 얼마나 사용했는지)
    user_cookie_interactions = []

    cookie_ids = [c["id"] for c in COOKIES]
    user_ids = user_agg["user_id"].tolist()

    for user_id in user_ids:
        # 각 유저는 3~8개의 쿠키를 주로 사용
        n_cookies = rng.integers(3, 9)
        used_cookies = rng.choice(cookie_ids, size=n_cookies, replace=False)

        for cookie_id in used_cookies:
            # 사용 횟수 (1~100)
            usage_count = int(rng.integers(1, 101))
            user_cookie_interactions.append({
                "user_id": user_id,
                "cookie_id": cookie_id,
                "usage_count": usage_count,
            })

    interactions_df = pd.DataFrame(user_cookie_interactions)
    print(f"  - 상호작용 데이터: {len(interactions_df)}건")

    # 쿠키 특성 데이터 (등급, 타입)
    cookie_features_list = []
    for cookie in COOKIES:
        cookie_features_list.append((cookie["id"], [
            f"grade:{cookie['grade']}",
            f"type:{cookie['type']}",
        ]))

    # LightFM Dataset 구성
    print("\nLightFM Dataset 구성 중...")
    dataset = Dataset()

    # 유저, 아이템, 아이템 특성 등록
    all_features = []
    for grade in COOKIE_GRADES:
        all_features.append(f"grade:{grade}")
    for ctype in COOKIE_TYPES:
        all_features.append(f"type:{ctype}")

    dataset.fit(
        users=user_ids,
        items=cookie_ids,
        item_features=all_features,
    )

    # 상호작용 매트릭스 생성
    (interactions_matrix, weights_matrix) = dataset.build_interactions([
        (row["user_id"], row["cookie_id"], row["usage_count"])
        for _, row in interactions_df.iterrows()
    ])

    # 아이템 특성 매트릭스 생성
    item_features_matrix = dataset.build_item_features(cookie_features_list)

    print(f"  - 상호작용 매트릭스: {interactions_matrix.shape}")
    print(f"  - 아이템 특성 매트릭스: {item_features_matrix.shape}")

    # LightFM 모델 학습
    print("\nLightFM 모델 학습 중...")
    lightfm_params = {
        "no_components": 32,      # 임베딩 차원
        "learning_rate": 0.05,
        "loss": "warp",           # Weighted Approximate-Rank Pairwise
        "random_state": seed,
    }

    lightfm_model = LightFM(**lightfm_params)
    lightfm_model.fit(
        interactions_matrix,
        item_features=item_features_matrix,
        sample_weight=weights_matrix,
        epochs=30,
        num_threads=4,
        verbose=True,
    )

    # 평가
    print("\n[모델 평가]")
    train_precision = precision_at_k(
        lightfm_model, interactions_matrix,
        item_features=item_features_matrix, k=5
    ).mean()
    train_auc = auc_score(
        lightfm_model, interactions_matrix,
        item_features=item_features_matrix
    ).mean()

    print(f"  Precision@5: {train_precision:.4f}")
    print(f"  AUC: {train_auc:.4f}")

    # MLflow 로깅
    if MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name="cookie_recommend_model"):
            mlflow.set_tag("model_type", "recommendation")
            mlflow.set_tag("algorithm", "LightFM")
            mlflow.set_tag("loss", "warp")
            mlflow.log_params(lightfm_params)
            mlflow.log_metrics({
                "precision_at_5": float(train_precision),
                "auc": float(train_auc),
                "n_users": len(user_ids),
                "n_items": len(cookie_ids),
                "n_interactions": len(interactions_df),
            })
            # LightFM은 sklearn 모델이 아니라 joblib로 저장
            model_path = BACKEND_DIR / "model_cookie_recommend.pkl"
            joblib.dump(lightfm_model, model_path)
            mlflow.log_artifact(str(model_path))
            mlflow.log_artifact(str(BACKEND_DIR / "cookie_recommend_config.json"))
            print(f"[MLflow] Run ID: {mlflow.active_run().info.run_id}")

    # 모델 및 메타데이터 저장
    joblib.dump(lightfm_model, BACKEND_DIR / "model_cookie_recommend.pkl")
    joblib.dump(dataset, BACKEND_DIR / "cookie_recommend_dataset.pkl")

    # 설정 파일 저장
    recommend_config = {
        "algorithm": "LightFM",
        "loss": "warp",
        "no_components": lightfm_params["no_components"],
        "n_users": len(user_ids),
        "n_items": len(cookie_ids),
        "n_interactions": len(interactions_df),
        "precision_at_5": float(train_precision),
        "auc": float(train_auc),
        "cookie_ids": cookie_ids,
        "features": all_features,
    }
    with open(BACKEND_DIR / "cookie_recommend_config.json", "w", encoding="utf-8") as f:
        json.dump(recommend_config, f, ensure_ascii=False, indent=2)

    # 상호작용 데이터 저장 (API에서 사용)
    interactions_df.to_csv(BACKEND_DIR / "user_cookie_interactions.csv", index=False, encoding="utf-8-sig")

    print(f"\n모델 저장: model_cookie_recommend.pkl")
    print(f"데이터셋 저장: cookie_recommend_dataset.pkl")
    print(f"설정 저장: cookie_recommend_config.json")
    print(f"상호작용 데이터 저장: user_cookie_interactions.csv")

    # 추천 테스트
    print("\n[추천 테스트]")
    test_user_idx = 0
    test_user_id = user_ids[test_user_idx]

    # 모든 쿠키에 대한 점수 예측
    n_items = len(cookie_ids)
    scores = lightfm_model.predict(
        test_user_idx,
        np.arange(n_items),
        item_features=item_features_matrix,
    )

    # 상위 5개 추천
    top_items = np.argsort(-scores)[:5]
    print(f"  유저 {test_user_id}에게 추천:")
    for rank, item_idx in enumerate(top_items, 1):
        cookie_id = cookie_ids[item_idx]
        cookie_name = next(c["name"] for c in COOKIES if c["id"] == cookie_id)
        print(f"    {rank}. {cookie_name} (점수: {scores[item_idx]:.3f})")

else:
    print("LightFM 미설치로 쿠키 추천 모델 학습 스킵")
    print("설치: pip install lightfm")

# ========================================
# 8. 분석 패널용 추가 데이터 생성
# ========================================
print("\n" + "=" * 60)
print("8. 분석 패널용 추가 데이터 생성")
print("=" * 60)

# 8.1 쿠키별 사용률/인기도 통계
print("8.1 쿠키별 통계 생성...")
cookie_stats_data = []
for idx, cookie in enumerate(COOKIES):
    # 등급에 따른 기본 인기도
    grade_base = {"에인션트": 90, "레전더리": 85, "에픽": 70, "슈퍼레어": 55, "레어": 40, "커먼": 25}
    base_popularity = grade_base.get(cookie["grade"], 50)

    # 등급별 기본 스탯 배율
    grade_multiplier = {"에인션트": 1.8, "레전더리": 1.6, "에픽": 1.2, "슈퍼레어": 1.0, "레어": 0.8, "커먼": 0.6}
    mult = grade_multiplier.get(cookie["grade"], 1.0)

    # 타입별 스탯 보정
    type_bonus = {
        "돌격": {"atk": 1.3, "hp": 1.0, "def": 0.9, "skill_dmg": 1.1, "cooldown": 0.9, "crit_rate": 1.1, "crit_dmg": 1.1},
        "방어": {"atk": 0.7, "hp": 1.4, "def": 1.4, "skill_dmg": 0.8, "cooldown": 1.1, "crit_rate": 0.8, "crit_dmg": 0.9},
        "마법": {"atk": 1.1, "hp": 0.9, "def": 0.9, "skill_dmg": 1.4, "cooldown": 1.0, "crit_rate": 1.0, "crit_dmg": 1.0},
        "치유": {"atk": 0.7, "hp": 1.2, "def": 1.1, "skill_dmg": 0.9, "cooldown": 1.2, "crit_rate": 0.8, "crit_dmg": 0.8},
        "지원": {"atk": 0.8, "hp": 1.1, "def": 1.1, "skill_dmg": 0.9, "cooldown": 1.1, "crit_rate": 0.9, "crit_dmg": 0.9},
        "사격": {"atk": 1.2, "hp": 0.85, "def": 0.85, "skill_dmg": 1.2, "cooldown": 1.0, "crit_rate": 1.5, "crit_dmg": 1.3},
        "폭발": {"atk": 1.2, "hp": 0.9, "def": 0.9, "skill_dmg": 1.5, "cooldown": 1.1, "crit_rate": 1.2, "crit_dmg": 1.1},
        "복수": {"atk": 1.1, "hp": 1.0, "def": 1.0, "skill_dmg": 1.2, "cooldown": 1.0, "crit_rate": 1.2, "crit_dmg": 1.2},
    }
    tbonus = type_bonus.get(cookie["type"], {k: 1.0 for k in ["atk", "hp", "def", "skill_dmg", "cooldown", "crit_rate", "crit_dmg"]})

    cookie_stats_data.append({
        "cookie_id": cookie["id"],
        "cookie_name": cookie["name"],
        "grade": cookie["grade"],
        "type": cookie["type"],
        "usage_rate": round(np.clip(rng.normal(base_popularity, 10), 10, 99), 1),
        "power_score": int(np.clip(rng.normal(base_popularity + 5, 8), 50, 100)),
        "popularity_score": round(np.clip(rng.normal(base_popularity, 12), 15, 98), 1),
        "pick_rate_pvp": round(np.clip(rng.normal(base_popularity - 10, 15), 5, 90), 1),
        "win_rate_pvp": round(np.clip(rng.normal(50, 8), 35, 65), 1),
        "avg_stage_score": int(rng.integers(50000, 200000)),
        # 스탯 컬럼 추가 (밸런스 최적화용)
        "atk": int(np.clip(rng.normal(18000 * mult * tbonus["atk"], 2000), 8000, 40000)),
        "hp": int(np.clip(rng.normal(60000 * mult * tbonus["hp"], 8000), 40000, 120000)),
        "def": int(np.clip(rng.normal(350 * mult * tbonus["def"], 40), 250, 550)),
        "skill_dmg": int(np.clip(rng.normal(200 * mult * tbonus["skill_dmg"], 30), 100, 400)),
        "cooldown": round(np.clip(rng.normal(10 / tbonus["cooldown"], 1.5), 6, 16), 0),
        "crit_rate": int(np.clip(rng.normal(15 * tbonus["crit_rate"], 4), 5, 35)),
        "crit_dmg": int(np.clip(rng.normal(150 * tbonus["crit_dmg"], 15), 120, 200)),
    })

cookie_stats_df = pd.DataFrame(cookie_stats_data)
print(f"  - 쿠키 통계: {len(cookie_stats_df)}개")

# 8.2 일별 게임 지표 데이터 (최근 90일)
print("8.2 일별 게임 지표 생성...")
daily_metrics_data = []
base_date = pd.to_datetime("2024-11-01")  # 90일치 데이터를 위해 시작일 조정
base_dau = 650
base_revenue = 4500000

for day in range(90):
    current_date = base_date + pd.Timedelta(days=day)
    # 주말 효과 (토/일 DAU 증가)
    weekend_boost = 1.15 if current_date.dayofweek >= 5 else 1.0

    dau = int(base_dau * weekend_boost * rng.uniform(0.9, 1.1))
    revenue = int(base_revenue * weekend_boost * rng.uniform(0.85, 1.25))
    new_users = int(rng.integers(35, 65) * weekend_boost)

    daily_metrics_data.append({
        "date": current_date.strftime("%Y-%m-%d"),
        "date_display": current_date.strftime("%m/%d"),
        "dau": dau,
        "mau": int(dau * 1.4),  # 추정
        "new_users": new_users,
        "returning_users": int(dau * 0.15),
        "sessions": int(dau * rng.uniform(2.5, 3.5)),
        "avg_session_minutes": round(rng.uniform(22, 35), 1),
        "revenue": revenue,
        "arpu": round(revenue / dau, 0),
        "paying_users": int(dau * rng.uniform(0.03, 0.06)),
        "stage_clears": int(dau * rng.uniform(8, 15)),
        "gacha_pulls": int(dau * rng.uniform(0.8, 1.5)),
        "pvp_battles": int(dau * rng.uniform(0.5, 1.2)),
    })

daily_metrics_df = pd.DataFrame(daily_metrics_data)
print(f"  - 일별 지표: {len(daily_metrics_df)}일")

# 8.3 번역 언어별 통계
print("8.3 번역 언어별 통계 생성...")
translation_stats_data = []
for lang_code, lang_name in [("en", "영어"), ("ja", "일본어"), ("zh", "중국어"), ("th", "태국어"), ("id", "인도네시아어")]:
    lang_translations = translation_df[translation_df["target_lang"] == lang_code] if lang_code in translation_df["target_lang"].values else pd.DataFrame()
    count = len(lang_translations) if not lang_translations.empty else int(rng.integers(200, 400))
    avg_quality = float(lang_translations["quality_score"].mean() * 100) if not lang_translations.empty else rng.uniform(85, 95)

    translation_stats_data.append({
        "lang_code": lang_code,
        "lang_name": lang_name,
        "total_count": count,
        "avg_quality": round(avg_quality, 1),
        "pending_count": int(rng.integers(10, 50)),
        "reviewed_count": int(count * rng.uniform(0.7, 0.9)),
        "auto_translated": int(count * rng.uniform(0.3, 0.5)),
        "human_reviewed": int(count * rng.uniform(0.4, 0.6)),
    })

translation_stats_df = pd.DataFrame(translation_stats_data)
print(f"  - 번역 통계: {len(translation_stats_df)}개 언어")

# 8.4 이상탐지 상세 데이터 (90일치)
print("8.4 이상탐지 상세 데이터 생성...")
anomaly_users = user_agg[user_agg["is_anomaly"] == 1].copy() if "is_anomaly" in user_agg.columns else pd.DataFrame()
anomaly_detail_data = []

anomaly_types = [
    ("비정상 결제 패턴", "high"),
    ("봇 의심 행동", "high"),
    ("계정 공유 의심", "medium"),
    ("비정상 플레이 시간", "low"),
]

# 90일치 데이터 생성 (매일 1~4건씩)
base_date = pd.to_datetime("2025-01-31")
anomaly_idx = 0
user_ids = anomaly_users["user_id"].tolist() if not anomaly_users.empty else [f"U{str(i).zfill(6)}" for i in range(100, 200)]

for day in range(90):
    current_date = base_date - pd.Timedelta(days=day)
    # 매일 1~4건의 이상 탐지 (주말에 더 많이)
    daily_count = rng.integers(1, 5) if current_date.dayofweek < 5 else rng.integers(2, 6)

    for _ in range(daily_count):
        anomaly_type, severity = anomaly_types[anomaly_idx % len(anomaly_types)]
        user_id = user_ids[anomaly_idx % len(user_ids)]
        hour = int(rng.integers(0, 24))

        anomaly_detail_data.append({
            "user_id": user_id,
            "anomaly_type": anomaly_type,
            "severity": severity,
            "anomaly_score": round(float(rng.uniform(-0.1, 0.0)), 3),
            "detected_at": (current_date + pd.Timedelta(hours=hour)).isoformat(),
            "detail": f"비정상 패턴 {anomaly_idx+1}번째 감지",
        })
        anomaly_idx += 1

anomaly_detail_df = pd.DataFrame(anomaly_detail_data)
print(f"  - 이상탐지 상세: {len(anomaly_detail_df)}건 (90일치)")

# 8.5 코호트 리텐션 데이터 (12주치 - 90일 지원)
print("8.5 코호트 리텐션 데이터 생성...")
cohort_data = []
months = ["2024-11", "2024-12", "2025-01"]
for month_idx, month in enumerate(months):
    for week_num in range(1, 5):
        week_offset = month_idx * 4 + week_num - 1
        cohort_name = f"{month} W{week_num}"
        base_retention = 100
        week_retentions = {"cohort": cohort_name, "week0": 100}

        for w in range(1, 5):
            if week_offset + w <= 12:
                decay = rng.uniform(0.65, 0.85) ** w
                week_retentions[f"week{w}"] = int(base_retention * decay)
            else:
                week_retentions[f"week{w}"] = None

        cohort_data.append(week_retentions)

cohort_df = pd.DataFrame(cohort_data)
print(f"  - 코호트 데이터: {len(cohort_df)}개")

# 8.6 유저 일별 활동 데이터 (샘플 유저들 - 90일치)
print("8.6 유저 일별 활동 데이터 생성...")
user_activity_data = []
sample_user_ids = users_df["user_id"].head(100).tolist()

for user_id in sample_user_ids:
    for day in range(90):
        date_str = (pd.to_datetime("2024-11-01") + pd.Timedelta(days=day)).strftime("%m/%d")
        user_activity_data.append({
            "user_id": user_id,
            "date": date_str,
            "playtime": int(rng.integers(30, 240)),
            "stages_cleared": int(rng.integers(5, 40)),
            "gacha_pulls": int(rng.integers(0, 10)),
            "pvp_battles": int(rng.integers(0, 15)),
            "guild_activities": int(rng.integers(0, 5)),
        })

user_activity_df = pd.DataFrame(user_activity_data)
print(f"  - 유저 활동 데이터: {len(user_activity_df)}건")

# 8.7 유저별 쿠키 보유 데이터 (투자 최적화용)
print("8.7 유저별 쿠키 보유 데이터 생성...")

# 등급별 보유 확률 (VIP 0 기준)
GRADE_OWNERSHIP_PROB = {
    '커먼': 0.95, '레어': 0.85, '슈퍼레어': 0.70,
    '에픽': 0.50, '에인션트': 0.25, '레전더리': 0.10
}
# 등급별 평균 레벨 (레어한 쿠키일수록 키우기 어려움)
# level_cost.csv 구간: 1-10, 10-20, 20-30, 30-40, 40-50, 50-60, 60-70
GRADE_AVG_LEVEL = {
    '커먼': 38, '레어': 35, '슈퍼레어': 32,
    '에픽': 28, '에인션트': 25, '레전더리': 22
}
# 등급별 평균 각성 (0-5)
GRADE_AVG_ASCENSION = {
    '커먼': 2.5, '레어': 2, '슈퍼레어': 1.5,
    '에픽': 1.2, '에인션트': 0.8, '레전더리': 0.5
}

user_cookies_data = []
for _, user in users_df.iterrows():
    user_id = user['user_id']
    vip = user['vip_level']
    vip_bonus = 1 + vip * 0.15  # VIP 4면 1.6배

    for cookie in COOKIES:
        cookie_id = cookie['id']
        grade = cookie['grade']

        # 보유 확률
        base_prob = GRADE_OWNERSHIP_PROB.get(grade, 0.5)
        own_prob = min(1.0, base_prob * vip_bonus)

        if rng.random() > own_prob:
            continue  # 미보유

        # 레벨 결정
        base_level = GRADE_AVG_LEVEL.get(grade, 30)
        level = int(np.clip(rng.normal(base_level * vip_bonus, 10), 1, 70))

        # 스킬 레벨 (쿠키 레벨의 70~100%)
        skill_level = int(np.clip(level * rng.uniform(0.7, 1.0), 1, 70))

        # 각성 단계
        base_asc = GRADE_AVG_ASCENSION.get(grade, 1)
        ascension = int(np.clip(rng.normal(base_asc * vip_bonus, 1), 0, 5))

        # 즐겨찾기
        is_favorite = 1 if rng.random() < (level / 100) else 0

        user_cookies_data.append({
            'user_id': user_id,
            'cookie_id': cookie_id,
            'cookie_level': level,
            'skill_level': skill_level,
            'ascension': ascension,
            'is_favorite': is_favorite
        })

user_cookies_df = pd.DataFrame(user_cookies_data)
print(f"  - 유저 쿠키 데이터: {len(user_cookies_df)}건 ({len(users_df)}명 유저)")

# 8.8 유저별 자원 데이터 생성
print("8.8 유저별 자원 데이터 생성...")

user_resources_data = []
for _, user in users_df.iterrows():
    user_id = user['user_id']
    vip = user['vip_level']
    vip_mult = 1 + vip * 0.4  # VIP 보너스 (VIP 5면 3배)

    # 현실적인 자원 분포 (level_cost.csv 기준)
    # - 40-50 레벨업: 50,000 exp_jelly, 100,000 coin
    # - 50-60 레벨업: 120,000 exp_jelly, 240,000 coin
    # 중앙값 기준: exp(12) ≈ 163,000, exp(12.5) ≈ 270,000
    user_resources_data.append({
        'user_id': user_id,
        'exp_jelly': int(np.clip(rng.lognormal(12, 0.9) * vip_mult, 10000, 3000000)),
        'coin': int(np.clip(rng.lognormal(12.5, 0.9) * vip_mult, 20000, 5000000)),
        'skill_powder': int(np.clip(rng.lognormal(8, 0.9) * vip_mult, 200, 100000)),
        'soul_stone_common': int(np.clip(rng.lognormal(5.5, 1.2) * vip_mult, 10, 5000)),
        'soul_stone_rare': int(np.clip(rng.lognormal(5, 1.2) * vip_mult, 5, 3000)),
        'soul_stone_epic': int(np.clip(rng.lognormal(4, 1.2) * vip_mult, 0, 1000)),
        'soul_stone_ancient': int(np.clip(rng.lognormal(3, 1.2) * vip_mult, 0, 500)),
        'soul_stone_legendary': int(np.clip(rng.lognormal(2, 1.2) * vip_mult, 0, 200)),
    })

user_resources_df = pd.DataFrame(user_resources_data)
print(f"  - 유저 자원 데이터: {len(user_resources_df)}건")

# ========================================
# 9. 파일 저장
# ========================================
print("\n" + "=" * 60)
print("파일 저장")
print("=" * 60)

# CSV 저장 (기본 데이터)
cookies_df.to_csv(BACKEND_DIR / "cookies.csv", index=False, encoding="utf-8-sig")
kingdoms_df.to_csv(BACKEND_DIR / "kingdoms.csv", index=False, encoding="utf-8-sig")
skills_df.to_csv(BACKEND_DIR / "skills.csv", index=False, encoding="utf-8-sig")
translation_df.to_csv(BACKEND_DIR / "translations.csv", index=False, encoding="utf-8-sig")
users_df.to_csv(BACKEND_DIR / "users.csv", index=False, encoding="utf-8-sig")
logs_df.to_csv(BACKEND_DIR / "game_logs.csv", index=False, encoding="utf-8-sig")
user_agg.to_csv(BACKEND_DIR / "user_analytics.csv", index=False, encoding="utf-8-sig")
worldview_df.to_csv(BACKEND_DIR / "worldview_texts.csv", index=False, encoding="utf-8-sig")

# 세계관 용어집 저장
terms_df = pd.DataFrame(WORLDVIEW_TERMS)
terms_df.to_csv(BACKEND_DIR / "worldview_terms.csv", index=False, encoding="utf-8-sig")

# 분석용 추가 데이터 저장
cookie_stats_df.to_csv(BACKEND_DIR / "cookie_stats.csv", index=False, encoding="utf-8-sig")
daily_metrics_df.to_csv(BACKEND_DIR / "daily_metrics.csv", index=False, encoding="utf-8-sig")
translation_stats_df.to_csv(BACKEND_DIR / "translation_stats.csv", index=False, encoding="utf-8-sig")
anomaly_detail_df.to_csv(BACKEND_DIR / "anomaly_details.csv", index=False, encoding="utf-8-sig")
cohort_df.to_csv(BACKEND_DIR / "cohort_retention.csv", index=False, encoding="utf-8-sig")
user_activity_df.to_csv(BACKEND_DIR / "user_activity.csv", index=False, encoding="utf-8-sig")

# 투자 최적화용 데이터 저장
user_cookies_df.to_csv(BACKEND_DIR / "user_cookies.csv", index=False, encoding="utf-8-sig")
user_resources_df.to_csv(BACKEND_DIR / "user_resources.csv", index=False, encoding="utf-8-sig")

# 모델/전처리 저장
joblib.dump(rf_translation, BACKEND_DIR / "model_translation_quality.pkl")
joblib.dump(rf_text, BACKEND_DIR / "model_text_category.pkl")
joblib.dump(kmeans, BACKEND_DIR / "model_user_segment.pkl")
joblib.dump(iso_forest, BACKEND_DIR / "model_anomaly.pkl")
joblib.dump(tfidf, BACKEND_DIR / "tfidf_vectorizer.pkl")
joblib.dump(scaler_cluster, BACKEND_DIR / "scaler_cluster.pkl")
joblib.dump(le_category, BACKEND_DIR / "le_category.pkl")
joblib.dump(le_lang, BACKEND_DIR / "le_lang.pkl")
joblib.dump(le_quality, BACKEND_DIR / "le_quality.pkl")
joblib.dump(le_text_category, BACKEND_DIR / "le_text_category.pkl")

print("\n저장 완료:")
print(f"  [기본 데이터]")
print(f"  - cookies.csv ({len(cookies_df)} rows)")
print(f"  - kingdoms.csv ({len(kingdoms_df)} rows)")
print(f"  - skills.csv ({len(skills_df)} rows)")
print(f"  - translations.csv ({len(translation_df)} rows)")
print(f"  - users.csv ({len(users_df)} rows)")
print(f"  - game_logs.csv ({len(logs_df)} rows)")
print(f"  - user_analytics.csv ({len(user_agg)} rows)")
print(f"  - worldview_texts.csv ({len(worldview_df)} rows)")
print(f"  - worldview_terms.csv ({len(terms_df)} rows)")
print(f"  [분석용 추가 데이터]")
print(f"  - cookie_stats.csv ({len(cookie_stats_df)} rows)")
print(f"  - daily_metrics.csv ({len(daily_metrics_df)} rows)")
print(f"  - translation_stats.csv ({len(translation_stats_df)} rows)")
print(f"  - anomaly_details.csv ({len(anomaly_detail_df)} rows)")
print(f"  - cohort_retention.csv ({len(cohort_df)} rows)")
print(f"  - user_activity.csv ({len(user_activity_df)} rows)")
print(f"  [투자 최적화용 데이터]")
print(f"  - user_cookies.csv ({len(user_cookies_df)} rows)")
print(f"  - user_resources.csv ({len(user_resources_df)} rows)")
print(f"  [ML 모델]")
print(f"  - model_translation_quality.pkl")
print(f"  - model_text_category.pkl")
print(f"  - model_user_segment.pkl")
print(f"  - model_anomaly.pkl")

# ========================================
# 9. 예측/분석 함수
# ========================================
def predict_translation_quality(text_features: dict) -> dict:
    """번역 품질 예측"""
    X = pd.DataFrame([text_features])[feature_cols_translation].astype(float)
    pred = rf_translation.predict(X)[0]
    proba = rf_translation.predict_proba(X)[0]
    quality_grade = le_quality.inverse_transform([pred])[0]
    return {
        "quality_grade": quality_grade,
        "probabilities": {le_quality.classes_[i]: float(proba[i]) for i in range(len(proba))},
    }


def classify_text_category(text: str) -> dict:
    """텍스트 카테고리 분류"""
    X = tfidf.transform([text])
    pred = rf_text.predict(X)[0]
    proba = rf_text.predict_proba(X)[0]
    category = le_text_category.inverse_transform([pred])[0]
    return {
        "category": category,
        "probabilities": {le_text_category.classes_[i]: float(proba[i]) for i in range(len(proba))},
    }


def get_user_segment(user_features: dict) -> dict:
    """유저 세그먼트 분류"""
    X = pd.DataFrame([user_features])[cluster_features].fillna(0)
    X_scaled = scaler_cluster.transform(X)
    cluster = int(kmeans.predict(X_scaled)[0])
    return {
        "cluster": cluster,
        "segment_name": SEGMENT_NAMES.get(cluster, "Unknown"),
    }


def detect_anomaly(user_features: dict) -> dict:
    """이상 유저 탐지"""
    X = pd.DataFrame([user_features])[cluster_features].fillna(0)
    X_scaled = scaler_cluster.transform(X)
    pred = int(iso_forest.predict(X_scaled)[0])
    score = float(iso_forest.decision_function(X_scaled)[0])
    return {
        "is_anomaly": pred == -1,
        "anomaly_score": score,
    }


# ========================================
# 10. 테스트
# ========================================
print("\n" + "=" * 60)
print("예측/분석 함수 테스트")
print("=" * 60)

# 번역 품질 테스트
sample_trans = {
    "category_encoded": 0,
    "target_lang_encoded": 0,
    "fluency_score": 0.85,
    "adequacy_score": 0.90,
    "contains_worldview_term": 1,
    "text_length": 25,
}
print("\n[번역 품질 예측]")
print(predict_translation_quality(sample_trans))

# 텍스트 분류 테스트
sample_text = "용감한 쿠키가 오븐에서 탈출했습니다!"
print("\n[텍스트 카테고리 분류]")
print(f"입력: '{sample_text}'")
print(classify_text_category(sample_text))

# 유저 세그먼트 테스트
sample_user = {
    "total_events": 150,
    "stage_clears": 80,
    "gacha_pulls": 20,
    "pvp_battles": 30,
    "purchases": 5,
    "vip_level": 2,
}
print("\n[유저 세그먼트]")
print(get_user_segment(sample_user))

# 이상 탐지 테스트
print("\n[이상 탐지]")
print(detect_anomaly(sample_user))

# ========================================
# 11. 완료 요약
# ========================================
print("\n" + "=" * 60)
print("데이터 생성 및 모델 학습 완료!")
print("=" * 60)
print("\n[생성된 데이터 파일]")
print(f"  - 기본 데이터: cookies.csv, kingdoms.csv, skills.csv, ...")
print(f"  - 분석 데이터: cookie_stats.csv, daily_metrics.csv, ...")
print("\n[학습된 모델]")
print("  - 번역 품질 예측 모델")
print("  - 텍스트 카테고리 분류 모델")
print("  - 유저 세그먼트 모델")
print("  - 이상 탐지 모델")

if MLFLOW_AVAILABLE:
    print("\n[MLflow 실험 추적]")
    print(f"  Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"  Experiment: {EXPERIMENT_NAME}")
    print("  MLflow UI 실행: mlflow ui --port 5000")
else:
    print("\n[참고] MLflow가 설치되지 않아 실험 추적이 비활성화되었습니다.")
    print("  설치: pip install mlflow")

print("\n" + "=" * 60)
print("백엔드 서버 시작: cd backend\\ && python main.py")
print("=" * 60)
