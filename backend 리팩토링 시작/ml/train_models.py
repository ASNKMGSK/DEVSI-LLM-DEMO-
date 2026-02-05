"""
쿠키런 세계관 AI 플랫폼 - 데이터 생성 및 모델 학습
===============================================
데브시스터즈 기술혁신 프로젝트 포트폴리오

구조:
  PART 1: 설정 및 환경
  PART 2: 데이터 생성 (모든 합성 데이터)
  PART 3: 모델 학습 (모든 ML 모델)
  PART 4: 저장 및 테스트
"""

# ============================================================================
# PART 1: 설정 및 환경
# ============================================================================
import os
from pathlib import Path
from datetime import datetime, timedelta
import json
import re
import logging

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split, cross_val_score
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
logger = logging.getLogger(__name__)

# MLflow 설정 (선택적)
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

# LightGBM (승률 예측용)
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available - skipping win rate model")

# Seed 고정
seed = 42
rng = np.random.default_rng(seed)

# 저장 경로
try:
    BACKEND_DIR = Path(__file__).parent.parent
except NameError:
    BACKEND_DIR = Path.cwd()
    if BACKEND_DIR.name == "ml":
        BACKEND_DIR = BACKEND_DIR.parent
    elif "backend" in str(BACKEND_DIR).lower() or "리팩토링" in str(BACKEND_DIR):
        pass
    else:
        BACKEND_DIR = Path.cwd()

BACKEND_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("PART 1: 설정 완료")
print(f"  BACKEND_DIR: {BACKEND_DIR}")
print("=" * 70)


# ============================================================================
# PART 2: 데이터 생성 (모든 합성 데이터)
# ============================================================================
print("\n" + "=" * 70)
print("PART 2: 데이터 생성")
print("=" * 70)

# ----------------------------------------------------------------------------
# 2.1 쿠키런 세계관 기본 데이터
# ----------------------------------------------------------------------------
print("\n[2.1] 쿠키런 세계관 기본 데이터")

COOKIE_GRADES = ["커먼", "레어", "슈퍼레어", "에픽", "레전더리", "에인션트"]
COOKIE_TYPES = ["돌격", "마법", "사격", "방어", "지원", "폭발", "치유", "복수"]

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

cookies_df = pd.DataFrame(COOKIES)
kingdoms_df = pd.DataFrame(KINGDOMS)
skills_df = pd.DataFrame(SKILLS)
print(f"  - 쿠키: {len(cookies_df)}개, 왕국: {len(kingdoms_df)}개, 스킬: {len(skills_df)}개")

# ----------------------------------------------------------------------------
# 2.2 번역 데이터셋
# ----------------------------------------------------------------------------
print("\n[2.2] 번역 데이터셋")

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

WORLDVIEW_TERMS = [
    {"term_ko": "젤리", "term_en": "Jelly", "context": "게임 내 화폐/아이템"},
    {"term_ko": "오븐", "term_en": "Oven", "context": "쿠키가 태어나는 곳"},
    {"term_ko": "소울잼", "term_en": "Soul Jam", "context": "고대의 신비한 보석"},
    {"term_ko": "마녀", "term_en": "Witch", "context": "쿠키들의 적"},
    {"term_ko": "다크엔챈트리스 쿠키", "term_en": "Dark Enchantress Cookie", "context": "메인 빌런"},
    {"term_ko": "트로피컬 소다 섬", "term_en": "Tropical Soda Islands", "context": "지역명"},
    {"term_ko": "어둠의 군단", "term_en": "Darkness Legion", "context": "적 세력"},
]

# 번역 품질 데이터 생성
translation_data = []
for text in GAME_TEXTS:
    for target_lang, target_text in [("en", text["en"]), ("ja", text["ja"]), ("zh", text["zh"])]:
        quality_score = rng.uniform(0.7, 1.0)
        fluency = rng.uniform(0.6, 1.0)
        adequacy = rng.uniform(0.7, 1.0)
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

# 추가 합성 번역 데이터
CATEGORIES = ["UI", "story", "skill", "dialog", "item", "quest", "achievement", "notice"]
for i in range(200):
    category = rng.choice(CATEGORIES)
    text_length = rng.integers(5, 100)
    contains_term = rng.choice([0, 1], p=[0.7, 0.3])
    base_quality = 0.8 if category in ["UI", "item"] else 0.75
    if contains_term:
        base_quality -= 0.05
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
print(f"  - 번역 데이터: {len(translation_df)}개")

# 세계관 텍스트 (텍스트 분류용)
worldview_texts = []
for cookie in COOKIES:
    worldview_texts.append({"text": cookie["story_kr"], "category": "character_story", "entity_id": cookie["id"]})
for kingdom in KINGDOMS:
    worldview_texts.append({"text": kingdom["desc_kr"], "category": "location", "entity_id": kingdom["id"]})
for skill in SKILLS:
    worldview_texts.append({"text": skill["desc_kr"], "category": "skill", "entity_id": skill["cookie_id"]})
for text in GAME_TEXTS:
    worldview_texts.append({"text": text["ko"], "category": text["category"], "entity_id": text["text_id"]})
worldview_df = pd.DataFrame(worldview_texts)

# ----------------------------------------------------------------------------
# 2.3 유저 및 게임 로그 데이터 (현실적 분포)
# ----------------------------------------------------------------------------
print("\n[2.3] 유저 및 게임 로그 데이터 (현실적 분포)")

n_users = 1000
reference_date = pd.to_datetime("2024-11-28")  # 기준일

# 1) 결제자 타입 먼저 결정 (현실적 분포)
# 고래 2%, 돌고래 5%, 소과금 15%, 무과금 78%
spender_types = rng.choice(
    ["whale", "dolphin", "minnow", "f2p"],
    n_users,
    p=[0.02, 0.05, 0.15, 0.78]
)

# 2) 결제자 타입에 따른 VIP 레벨 결정
def get_vip_from_spender(spender_type):
    if spender_type == "whale":
        return rng.choice([4, 5], p=[0.3, 0.7])
    elif spender_type == "dolphin":
        return rng.choice([2, 3, 4], p=[0.3, 0.5, 0.2])
    elif spender_type == "minnow":
        return rng.choice([0, 1, 2], p=[0.2, 0.5, 0.3])
    else:  # f2p
        return rng.choice([0, 1], p=[0.85, 0.15])

vip_levels = np.array([get_vip_from_spender(s) for s in spender_types])

# 3) 가입일 (기준일 이전으로 제한)
# 2024-01-01 ~ 2024-11-27 (기준일 전날까지)
max_register_days = (reference_date - pd.to_datetime("2024-01-01")).days  # 331일
register_dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, max_register_days, n_users), unit="D")

# 4) 가입 후 경과일 (최소 1일 보장)
days_since_register = np.maximum(1, (reference_date - register_dates).days)

# 5) 레벨 계산 (로그 성장 곡선 + VIP 보너스)
# level = base * log(days + 1) * vip_mult + noise
def calculate_level(days, vip, spender):
    # 안전하게 days를 int로 변환
    days = max(1, int(days))
    base_growth = 8  # 기본 성장 계수
    vip_mult = 1 + vip * 0.15
    # 고래/돌고래는 더 빠르게 성장
    spender_mult = {"whale": 1.4, "dolphin": 1.2, "minnow": 1.05, "f2p": 1.0}[spender]
    level = base_growth * np.log(days + 1) * vip_mult * spender_mult
    level += rng.normal(0, 3)  # 노이즈
    return int(np.clip(level, 1, 70))

levels = np.array([
    calculate_level(days_since_register[i], vip_levels[i], spender_types[i])
    for i in range(n_users)
])

# 6) 최고 스테이지 (레벨 기반)
max_stages = np.array([
    int(np.clip(level * 7 + rng.integers(-20, 20), 1, 500))
    for level in levels
])

# 7) 총 플레이타임 (레벨/VIP/경과일 기반, 시간 단위)
def calculate_playtime(level, vip, days, spender):
    base_time = level * 2.5  # 레벨당 평균 2.5시간
    daily_time = days * 0.3 * (1 + vip * 0.2)  # 일당 평균 플레이
    spender_mult = {"whale": 1.5, "dolphin": 1.3, "minnow": 1.1, "f2p": 1.0}[spender]
    total = (base_time + daily_time) * spender_mult
    total *= rng.uniform(0.7, 1.3)  # 개인차
    return round(np.clip(total, 1, 2000), 1)

total_playtime = np.array([
    calculate_playtime(levels[i], vip_levels[i], days_since_register[i], spender_types[i])
    for i in range(n_users)
])

# 8) 결제 금액 (원화, 현실적 분포)
def calculate_spending(spender, vip, days):
    if spender == "whale":
        # 월 10-50만원, 고VIP일수록 더 많이
        monthly = rng.lognormal(11.5, 0.5) * (1 + vip * 0.3)  # 평균 ~30만원
        total = monthly * (days / 30)
        return int(np.clip(total, 100000, 10000000))
    elif spender == "dolphin":
        # 월 2-8만원
        monthly = rng.lognormal(10.5, 0.4) * (1 + vip * 0.2)  # 평균 ~5만원
        total = monthly * (days / 30)
        return int(np.clip(total, 20000, 1000000))
    elif spender == "minnow":
        # 총 1-5만원 (비정기 소액)
        total = rng.lognormal(9.5, 0.6)  # 평균 ~2만원
        return int(np.clip(total, 1000, 100000))
    else:  # f2p
        return 0

total_spent = np.array([
    calculate_spending(spender_types[i], vip_levels[i], days_since_register[i])
    for i in range(n_users)
])

# 9) 이탈 판정 (레벨 구간별 차등 이탈률)
def determine_churn(level, days, spender, vip):
    # 튜토리얼 이탈 (레벨 1-15): 30%
    if level <= 15:
        churn_prob = 0.30
        reason = "tutorial_drop"
    # 초반 콘텐츠 벽 (레벨 16-30): 20%
    elif level <= 30:
        churn_prob = 0.20
        reason = "early_wall"
    # 중반 벽 (레벨 31-50): 15%
    elif level <= 50:
        churn_prob = 0.15
        reason = "content_wall"
    # 후반 (레벨 51-60): 8%
    elif level <= 60:
        churn_prob = 0.08
        reason = "competition"
    # 최고 레벨 (61+): 5%
    else:
        churn_prob = 0.05
        reason = "natural"

    # 결제자는 이탈률 감소
    spender_reduction = {"whale": 0.3, "dolphin": 0.5, "minnow": 0.7, "f2p": 1.0}[spender]
    churn_prob *= spender_reduction

    # VIP 높을수록 이탈률 감소
    churn_prob *= (1 - vip * 0.08)

    # 최근 가입자는 이탈률 약간 높음
    if days < 14:
        churn_prob *= 1.2

    is_churned = rng.random() < churn_prob
    return is_churned, reason if is_churned else "active"

churn_results = [
    determine_churn(levels[i], days_since_register[i], spender_types[i], vip_levels[i])
    for i in range(n_users)
]
is_churned = np.array([r[0] for r in churn_results])
churn_reason = np.array([r[1] for r in churn_results])

# 10) 마지막 로그인 (이탈자는 7일+ 전, 활성유저는 0-3일 전)
def get_last_login_days(churned, days_since_reg, vip):
    if churned:
        # 이탈자: 7-90일 전
        return int(np.clip(rng.exponential(20) + 7, 7, min(90, days_since_reg)))
    else:
        # 활성: 0-3일 전 (VIP 높을수록 자주 접속)
        max_days = max(1, 4 - vip)
        return int(rng.integers(0, max_days))

last_login_days = np.array([
    get_last_login_days(is_churned[i], days_since_register[i], vip_levels[i])
    for i in range(n_users)
])

# 유저 DataFrame 생성
users_df = pd.DataFrame({
    "user_id": [f"U{i:06d}" for i in range(1, n_users + 1)],
    "country": rng.choice(["KR", "US", "JP", "CN", "TW", "TH", "ID"], n_users, p=[0.3, 0.2, 0.15, 0.15, 0.08, 0.07, 0.05]),
    "register_date": register_dates,
    "vip_level": vip_levels,
    "level": levels,
    "max_stage": max_stages,
    "total_playtime_hours": total_playtime,
    "spender_type": spender_types,
    "total_spent": total_spent,
    "is_churned": is_churned.astype(int),
    "churn_reason": churn_reason,
    "last_login_days_ago": last_login_days,
})

print(f"  - 유저 생성: {len(users_df)}명")
print(f"    └ 결제자 분포: 고래 {(spender_types == 'whale').sum()}명, "
      f"돌고래 {(spender_types == 'dolphin').sum()}명, "
      f"소과금 {(spender_types == 'minnow').sum()}명, "
      f"무과금 {(spender_types == 'f2p').sum()}명")
print(f"    └ 이탈률: {is_churned.mean()*100:.1f}% ({is_churned.sum()}명)")
print(f"    └ 평균 레벨: {levels.mean():.1f}, 평균 결제: {total_spent[total_spent > 0].mean():,.0f}원")

# 활동 가중치 계산 (VIP/레벨/결제자 타입 반영)
# 고래/돌고래는 더 활발, 이탈자는 활동 감소
def calc_activity_weight(row):
    base = rng.pareto(1.5) + 1
    vip_mult = 1 + row["vip_level"] * 0.3
    level_mult = 1 + row["level"] * 0.01
    spender_mult = {"whale": 2.0, "dolphin": 1.5, "minnow": 1.1, "f2p": 1.0}[row["spender_type"]]
    # 이탈자는 활동 크게 감소
    churn_mult = 0.1 if row["is_churned"] else 1.0
    return base * vip_mult * level_mult * spender_mult * churn_mult

user_activity_weights = users_df.apply(calc_activity_weight, axis=1).values
user_activity_weights = user_activity_weights / user_activity_weights.sum()

# 유저별 이탈 시점 계산 (이탈자만)
# last_login_days_ago를 기준으로 마지막 활동일 계산
user_last_active = {}
for _, row in users_df.iterrows():
    if row["is_churned"]:
        last_active = reference_date - pd.Timedelta(days=row["last_login_days_ago"])
    else:
        last_active = reference_date
    user_last_active[row["user_id"]] = last_active

# 유저 정보 dict로 변환 (빠른 조회)
user_info = users_df.set_index("user_id").to_dict("index")

HOUR_WEIGHTS = {
    0: 0.3, 1: 0.2, 2: 0.1, 3: 0.05, 4: 0.03, 5: 0.02,
    6: 0.05, 7: 0.1, 8: 0.3, 9: 0.5, 10: 0.6, 11: 0.7,
    12: 0.9, 13: 0.8, 14: 0.7, 15: 0.6, 16: 0.7, 17: 0.8,
    18: 1.0, 19: 1.2, 20: 1.5, 21: 1.8, 22: 1.5, 23: 0.8,
}
DAY_WEIGHTS = {0: 1.0, 1: 0.8, 2: 0.8, 3: 0.9, 4: 1.0, 5: 1.3, 6: 1.4}

cookie_ids = [c["id"] for c in COOKIES]
cookie_grades = {c["id"]: c["grade"] for c in COOKIES}
grade_popularity = {"커먼": 0.5, "레어": 0.8, "슈퍼레어": 1.0, "에픽": 1.5, "에인션트": 2.0, "레전더리": 2.5}
cookie_weights = np.array([grade_popularity[cookie_grades[cid]] for cid in cookie_ids])
cookie_weights = cookie_weights / cookie_weights.sum()

EVENT_TYPES = ["stage_clear", "gacha_pull", "cookie_upgrade", "pvp_battle", "guild_activity", "shop_purchase", "login"]

base_date = pd.to_datetime("2024-06-01")
end_date = pd.to_datetime("2024-11-28")

print("  게임 로그 생성 중 (완전 벡터화)...")

# 유저별 메타데이터 배열로 준비
user_ids_arr = users_df["user_id"].values
user_starts = np.array([max(base_date, user_info[uid]["register_date"]) for uid in user_ids_arr])
user_ends = np.array([min(end_date, user_last_active[uid]) for uid in user_ids_arr])
user_active_days = np.array([(user_ends[i] - user_starts[i]).days for i in range(len(user_ids_arr))])
user_active_days = np.maximum(user_active_days, 0)

# 결제자/VIP/레벨 배열
spender_mults = np.array([{"whale": 4.0, "dolphin": 2.5, "minnow": 1.3, "f2p": 1.0}[user_info[uid]["spender_type"]] for uid in user_ids_arr])
vip_mults = np.array([1 + user_info[uid]["vip_level"] * 0.2 for uid in user_ids_arr])
level_mults = np.array([1 + user_info[uid]["level"] * 0.01 for uid in user_ids_arr])
churn_mults = np.array([0.3 if user_info[uid]["is_churned"] else 1.0 for uid in user_ids_arr])

# 유저별 로그 수 계산 (벡터화)
daily_events = 8 * spender_mults * vip_mults * level_mults * churn_mults
random_factors = rng.uniform(0.7, 1.3, len(user_ids_arr))
user_log_counts = np.maximum(10, (user_active_days * daily_events * random_factors).astype(int))
user_log_counts[user_active_days == 0] = 0

total_logs = int(user_log_counts.sum())
print(f"    총 {total_logs:,}건 로그 생성 예정")

# 모든 로그의 user_id 인덱스 생성
user_indices = np.repeat(np.arange(len(user_ids_arr)), user_log_counts)
all_user_ids = user_ids_arr[user_indices]

# 공통 배열 생성
hour_probs = np.array([HOUR_WEIGHTS[h] for h in range(24)])
hour_probs = hour_probs / hour_probs.sum()
event_probs = np.array([0.35, 0.12, 0.15, 0.15, 0.13, 0.10])

all_event_types = rng.choice(EVENT_TYPES[:-1], total_logs, p=event_probs)
all_hours = rng.choice(24, total_logs, p=hour_probs)
all_minutes = rng.integers(0, 60, total_logs)

# 유저별 active_days를 로그별로 확장
user_active_days_expanded = user_active_days[user_indices]
user_active_days_expanded = np.maximum(user_active_days_expanded, 1)  # 0 방지
all_day_offsets = (rng.random(total_logs) * user_active_days_expanded).astype(int)

# 유저별 시작일을 로그별로 확장
user_starts_expanded = user_starts[user_indices]

# 날짜 계산 (벡터화)
all_event_dates = user_starts_expanded + pd.to_timedelta(all_day_offsets, unit='D') + pd.to_timedelta(all_hours, unit='h') + pd.to_timedelta(all_minutes, unit='m')

# 유저별 정보 확장 (detail 생성용)
user_levels = np.array([user_info[uid]["level"] for uid in user_ids_arr])
user_max_stages = np.array([user_info[uid]["max_stage"] for uid in user_ids_arr])
user_spenders = np.array([user_info[uid]["spender_type"] for uid in user_ids_arr])

user_levels_exp = user_levels[user_indices]
user_max_stages_exp = user_max_stages[user_indices]
user_spenders_exp = user_spenders[user_indices]

# detail 생성 (이벤트 타입별 벡터화 + 유저 속성별 현실적 분포)
print("    세부 정보 생성 중...")
all_details = np.empty(total_logs, dtype=object)
all_details[:] = "{}"

# stage_clear 마스크 (레벨별 star 확률 다름)
mask_stage = all_event_types == "stage_clear"
n_stage = mask_stage.sum()
if n_stage > 0:
    min_stages = np.maximum(1, user_max_stages_exp[mask_stage] - 50)
    max_stages = np.minimum(500, user_max_stages_exp[mask_stage] + 5)
    stages = (rng.random(n_stage) * (max_stages - min_stages) + min_stages).astype(int)
    cookies = rng.choice(cookie_ids, n_stage, p=cookie_weights)
    # 레벨별 star 확률: 저렙은 1-2성 많고, 고렙은 3성 많음
    stage_levels = user_levels_exp[mask_stage]
    stars = np.empty(n_stage, dtype=int)
    low_level_mask = stage_levels < 30
    stars[low_level_mask] = rng.choice([1, 2, 3], low_level_mask.sum(), p=[0.15, 0.35, 0.50])
    stars[~low_level_mask] = rng.choice([1, 2, 3], (~low_level_mask).sum(), p=[0.08, 0.27, 0.65])
    all_details[mask_stage] = [f'{{"stage":{s},"stars":{st},"cookie_id":"{c}"}}' for s, st, c in zip(stages, stars, cookies)]

# gacha_pull 마스크 (결제자별 ten_pull 확률 다름)
mask_gacha = all_event_types == "gacha_pull"
n_gacha = mask_gacha.sum()
if n_gacha > 0:
    gacha_spenders = user_spenders_exp[mask_gacha]
    ten_pull_probs = np.where(gacha_spenders == "whale", 0.90,
                     np.where(gacha_spenders == "dolphin", 0.85,
                     np.where(gacha_spenders == "minnow", 0.75, 0.60)))
    pull_types = np.where(rng.random(n_gacha) < ten_pull_probs, "ten_pull", "single")
    grades = rng.choice(["커먼", "레어", "슈퍼레어", "에픽", "레전더리"], n_gacha, p=[0.30, 0.35, 0.25, 0.09, 0.01])
    all_details[mask_gacha] = [f'{{"pull_type":"{pt}","result_grade":"{g}"}}' for pt, g in zip(pull_types, grades)]

# pvp_battle 마스크 (레벨별 tier/승률 다름)
mask_pvp = all_event_types == "pvp_battle"
n_pvp = mask_pvp.sum()
if n_pvp > 0:
    pvp_levels = user_levels_exp[mask_pvp]
    cookies = rng.choice(cookie_ids, n_pvp, p=cookie_weights)
    # 레벨별 승률: 고렙일수록 승률 높음
    win_probs = 0.48 + pvp_levels * 0.002
    results = np.where(rng.random(n_pvp) < win_probs, "win", "lose")
    # 레벨별 tier 분포
    tiers = np.empty(n_pvp, dtype=object)
    tier_options = ["브론즈", "실버", "골드", "다이아", "마스터"]
    for lvl_min, lvl_max, probs in [
        (60, 999, [0.02, 0.08, 0.20, 0.40, 0.30]),
        (45, 60, [0.05, 0.15, 0.35, 0.35, 0.10]),
        (30, 45, [0.10, 0.25, 0.40, 0.20, 0.05]),
        (15, 30, [0.15, 0.40, 0.35, 0.08, 0.02]),
        (0, 15, [0.50, 0.35, 0.12, 0.03, 0.00]),
    ]:
        tier_mask = (pvp_levels >= lvl_min) & (pvp_levels < lvl_max)
        if tier_mask.sum() > 0:
            # 0 확률 처리
            probs_arr = np.array(probs)
            if probs_arr[-1] == 0:
                tiers[tier_mask] = rng.choice(tier_options[:-1], tier_mask.sum(), p=probs_arr[:-1]/probs_arr[:-1].sum())
            else:
                tiers[tier_mask] = rng.choice(tier_options, tier_mask.sum(), p=probs_arr)
    all_details[mask_pvp] = [f'{{"result":"{r}","opponent_tier":"{t}","cookie_id":"{c}"}}' for r, t, c in zip(results, tiers, cookies)]

# cookie_upgrade 마스크 (유저 레벨에 맞는 쿠키 레벨)
mask_upgrade = all_event_types == "cookie_upgrade"
n_upgrade = mask_upgrade.sum()
if n_upgrade > 0:
    cookies = rng.choice(cookie_ids, n_upgrade, p=cookie_weights)
    upgrade_levels = user_levels_exp[mask_upgrade]
    min_cookie_lv = np.maximum(1, upgrade_levels - 10)
    max_cookie_lv = np.minimum(70, upgrade_levels + 5)
    levels = (rng.random(n_upgrade) * (max_cookie_lv - min_cookie_lv) + min_cookie_lv).astype(int)
    all_details[mask_upgrade] = [f'{{"cookie_id":"{c}","new_level":{lv}}}' for c, lv in zip(cookies, levels)]

# guild_activity 마스크
mask_guild = all_event_types == "guild_activity"
n_guild = mask_guild.sum()
if n_guild > 0:
    activities = rng.choice(["guild_boss", "donation", "chat"], n_guild, p=[0.4, 0.35, 0.25])
    all_details[mask_guild] = [f'{{"activity_type":"{a}"}}' for a in activities]

# shop_purchase 마스크 (결제자별 금액대 다름)
mask_shop = all_event_types == "shop_purchase"
n_shop = mask_shop.sum()
if n_shop > 0:
    shop_spenders = user_spenders_exp[mask_shop]
    items = rng.choice(["crystal", "coin", "stamina", "package"], n_shop, p=[0.3, 0.3, 0.25, 0.15])
    amounts = np.empty(n_shop, dtype=int)
    for spender_type, amt_opts, amt_probs in [
        ("whale", [1000, 5000, 10000, 50000], [0.15, 0.30, 0.35, 0.20]),
        ("dolphin", [500, 1000, 5000, 10000], [0.25, 0.35, 0.30, 0.10]),
        ("minnow", [100, 500, 1000, 3000], [0.40, 0.35, 0.20, 0.05]),
        ("f2p", [0, 100, 500, 1000], [0.60, 0.25, 0.12, 0.03]),
    ]:
        spender_mask = shop_spenders == spender_type
        if spender_mask.sum() > 0:
            amounts[spender_mask] = rng.choice(amt_opts, spender_mask.sum(), p=amt_probs)
    all_details[mask_shop] = [f'{{"item_type":"{it}","amount":{am}}}' for it, am in zip(items, amounts)]

# DataFrame 한 번에 생성
print("    DataFrame 생성 중...")
logs_df = pd.DataFrame({
    "log_id": [f"LOG{i:08d}" for i in range(total_logs)],
    "user_id": all_user_ids,
    "event_type": all_event_types,
    "event_date": all_event_dates,
    "detail": all_details,
})
logs_df["event_date"] = pd.to_datetime(logs_df["event_date"])
print(f"  - 유저: {len(users_df)}명, 게임 로그: {len(logs_df)}건")

# 유저별 집계
user_agg = logs_df.groupby("user_id").agg(
    total_events=("log_id", "count"),
    stage_clears=("event_type", lambda x: (x == "stage_clear").sum()),
    gacha_pulls=("event_type", lambda x: (x == "gacha_pull").sum()),
    pvp_battles=("event_type", lambda x: (x == "pvp_battle").sum()),
    purchases=("event_type", lambda x: (x == "shop_purchase").sum()),
).reset_index()
user_agg = user_agg.merge(users_df[["user_id", "vip_level", "level", "register_date"]], on="user_id", how="left")

# 피처 계산: days_since_register
reference_date = pd.to_datetime("2024-11-28")
user_agg["days_since_register"] = (reference_date - user_agg["register_date"]).dt.days.fillna(30).astype(int)

# 피처 계산: days_since_last_login (로그에서 마지막 접속일 계산)
last_login = logs_df.groupby("user_id")["event_date"].max().reset_index()
last_login.columns = ["user_id", "last_login_date"]
user_agg = user_agg.merge(last_login, on="user_id", how="left")
user_agg["days_since_last_login"] = (reference_date - user_agg["last_login_date"]).dt.days.fillna(90).astype(int)
user_agg.drop(columns=["register_date", "last_login_date"], inplace=True, errors="ignore")

# 누락 유저 추가 (신규/비활성)
all_user_ids = set(users_df["user_id"])
existing_user_ids = set(user_agg["user_id"])
missing_user_ids = all_user_ids - existing_user_ids
if missing_user_ids:
    print(f"  - 로그 없는 유저 {len(missing_user_ids)}명 추가")
    missing_users_df = users_df[users_df["user_id"].isin(missing_user_ids)][["user_id", "vip_level", "level", "register_date"]].copy()
    missing_users_df["total_events"] = rng.integers(10, 50, len(missing_users_df))
    missing_users_df["stage_clears"] = rng.integers(1, 15, len(missing_users_df))
    missing_users_df["gacha_pulls"] = rng.integers(0, 10, len(missing_users_df))
    missing_users_df["pvp_battles"] = rng.integers(0, 5, len(missing_users_df))
    missing_users_df["purchases"] = rng.integers(0, 3, len(missing_users_df))
    # 새 피처 추가
    reference_date = pd.to_datetime("2024-11-28")
    missing_users_df["days_since_register"] = (reference_date - missing_users_df["register_date"]).dt.days.fillna(30).astype(int)
    missing_users_df["days_since_last_login"] = rng.integers(30, 90, len(missing_users_df))  # 비활성 유저이므로 오래됨
    missing_users_df.drop(columns=["register_date"], inplace=True, errors="ignore")
    user_agg = pd.concat([user_agg, missing_users_df], ignore_index=True)
    user_agg = user_agg.sort_values("user_id").reset_index(drop=True)
print(f"  - 최종 유저 집계: {len(user_agg)}명")

# ----------------------------------------------------------------------------
# 2.4 쿠키 통계 데이터 (승률 예측 모델용)
# ----------------------------------------------------------------------------
print("\n[2.4] 쿠키 통계 데이터")

grade_multiplier = {"에인션트": 1.8, "레전더리": 1.6, "에픽": 1.2, "슈퍼레어": 1.0, "레어": 0.8, "커먼": 0.6}
grade_base = {"에인션트": 90, "레전더리": 85, "에픽": 70, "슈퍼레어": 55, "레어": 40, "커먼": 25}
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

cookie_stats_data = []
for cookie in COOKIES:
    mult = grade_multiplier.get(cookie["grade"], 1.0)
    base_pop = grade_base.get(cookie["grade"], 50)
    tbonus = type_bonus.get(cookie["type"], {k: 1.0 for k in ["atk", "hp", "def", "skill_dmg", "cooldown", "crit_rate", "crit_dmg"]})
    cookie_stats_data.append({
        "cookie_id": cookie["id"],
        "cookie_name": cookie["name"],
        "grade": cookie["grade"],
        "type": cookie["type"],
        "usage_rate": round(np.clip(rng.normal(base_pop, 10), 10, 99), 1),
        "power_score": int(np.clip(rng.normal(base_pop + 5, 8), 50, 100)),
        "popularity_score": round(np.clip(rng.normal(base_pop, 12), 15, 98), 1),
        "pick_rate_pvp": round(np.clip(rng.normal(base_pop - 10, 15), 5, 90), 1),
        "win_rate_pvp": round(np.clip(rng.normal(50, 8), 35, 65), 1),
        "avg_stage_score": int(rng.integers(50000, 200000)),
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

# ----------------------------------------------------------------------------
# 2.5 일별 지표, 번역 통계, 이상탐지 상세, 코호트, 유저 활동
# ----------------------------------------------------------------------------
print("\n[2.5] 분석 패널용 추가 데이터")

# 일별 지표 (90일)
daily_metrics_data = []
base_date_metrics = pd.to_datetime("2024-11-01")
base_dau, base_revenue = 650, 4500000
for day in range(90):
    current_date = base_date_metrics + pd.Timedelta(days=day)
    weekend_boost = 1.15 if current_date.dayofweek >= 5 else 1.0
    dau = int(base_dau * weekend_boost * rng.uniform(0.9, 1.1))
    revenue = int(base_revenue * weekend_boost * rng.uniform(0.85, 1.25))
    daily_metrics_data.append({
        "date": current_date.strftime("%Y-%m-%d"),
        "date_display": current_date.strftime("%m/%d"),
        "dau": dau, "mau": int(dau * 1.4),
        "new_users": int(rng.integers(35, 65) * weekend_boost),
        "returning_users": int(dau * 0.15),
        "sessions": int(dau * rng.uniform(2.5, 3.5)),
        "avg_session_minutes": round(rng.uniform(22, 35), 1),
        "revenue": revenue, "arpu": round(revenue / dau, 0),
        "paying_users": int(dau * rng.uniform(0.03, 0.06)),
        "stage_clears": int(dau * rng.uniform(8, 15)),
        "gacha_pulls": int(dau * rng.uniform(0.8, 1.5)),
        "pvp_battles": int(dau * rng.uniform(0.5, 1.2)),
    })
daily_metrics_df = pd.DataFrame(daily_metrics_data)
print(f"  - 일별 지표: {len(daily_metrics_df)}일")

# 번역 언어별 통계
translation_stats_data = []
for lang_code, lang_name in [("en", "영어"), ("ja", "일본어"), ("zh", "중국어"), ("th", "태국어"), ("id", "인도네시아어")]:
    lang_translations = translation_df[translation_df["target_lang"] == lang_code] if lang_code in translation_df["target_lang"].values else pd.DataFrame()
    count = len(lang_translations) if not lang_translations.empty else int(rng.integers(200, 400))
    avg_quality = float(lang_translations["quality_score"].mean() * 100) if not lang_translations.empty else rng.uniform(85, 95)
    translation_stats_data.append({
        "lang_code": lang_code, "lang_name": lang_name, "total_count": count,
        "avg_quality": round(avg_quality, 1), "pending_count": int(rng.integers(10, 50)),
        "reviewed_count": int(count * rng.uniform(0.7, 0.9)),
        "auto_translated": int(count * rng.uniform(0.3, 0.5)),
        "human_reviewed": int(count * rng.uniform(0.4, 0.6)),
    })
translation_stats_df = pd.DataFrame(translation_stats_data)
print(f"  - 번역 통계: {len(translation_stats_df)}개 언어")

# 코호트 리텐션 (12주)
cohort_data = []
months = ["2024-11", "2024-12", "2025-01"]
for month_idx, month in enumerate(months):
    for week_num in range(1, 5):
        week_offset = month_idx * 4 + week_num - 1
        cohort_name = f"{month} W{week_num}"
        week_retentions = {"cohort": cohort_name, "week0": 100}
        for w in range(1, 5):
            if week_offset + w <= 12:
                decay = rng.uniform(0.65, 0.85) ** w
                week_retentions[f"week{w}"] = int(100 * decay)
            else:
                week_retentions[f"week{w}"] = None
        cohort_data.append(week_retentions)
cohort_df = pd.DataFrame(cohort_data)
print(f"  - 코호트: {len(cohort_df)}개")

# 유저 일별 활동 (전체 유저, 90일)
user_activity_data = []
all_user_ids = users_df["user_id"].tolist()  # 전체 유저
for user_id in all_user_ids:
    for day in range(90):
        date_str = (pd.to_datetime("2024-11-01") + pd.Timedelta(days=day)).strftime("%m/%d")
        user_activity_data.append({
            "user_id": user_id, "date": date_str,
            "playtime": int(rng.integers(30, 240)),
            "stages_cleared": int(rng.integers(5, 40)),
            "gacha_pulls": int(rng.integers(0, 10)),
            "pvp_battles": int(rng.integers(0, 15)),
            "guild_activities": int(rng.integers(0, 5)),
        })
user_activity_df = pd.DataFrame(user_activity_data)
print(f"  - 유저 활동: {len(user_activity_df)}건")

# ----------------------------------------------------------------------------
# 2.6 유저 쿠키 보유 및 자원 데이터 (투자 최적화용)
# ----------------------------------------------------------------------------
print("\n[2.6] 유저 쿠키 보유 및 자원 데이터")

GRADE_OWNERSHIP_PROB = {'커먼': 0.95, '레어': 0.85, '슈퍼레어': 0.70, '에픽': 0.50, '에인션트': 0.25, '레전더리': 0.10}
GRADE_AVG_LEVEL = {'커먼': 38, '레어': 35, '슈퍼레어': 32, '에픽': 28, '에인션트': 25, '레전더리': 22}
GRADE_AVG_ASCENSION = {'커먼': 2.5, '레어': 2, '슈퍼레어': 1.5, '에픽': 1.2, '에인션트': 0.8, '레전더리': 0.5}

user_cookies_data = []
for _, user in users_df.iterrows():
    user_id, vip = user['user_id'], user['vip_level']
    vip_bonus = 1 + vip * 0.15
    for cookie in COOKIES:
        base_prob = GRADE_OWNERSHIP_PROB.get(cookie['grade'], 0.5)
        if rng.random() > min(1.0, base_prob * vip_bonus):
            continue
        base_level = GRADE_AVG_LEVEL.get(cookie['grade'], 30)
        level = int(np.clip(rng.normal(base_level * vip_bonus, 10), 1, 70))
        skill_level = int(np.clip(level * rng.uniform(0.7, 1.0), 1, 70))
        base_asc = GRADE_AVG_ASCENSION.get(cookie['grade'], 1)
        ascension = int(np.clip(rng.normal(base_asc * vip_bonus, 1), 0, 5))
        user_cookies_data.append({
            'user_id': user_id, 'cookie_id': cookie['id'],
            'cookie_level': level, 'skill_level': skill_level,
            'ascension': ascension, 'is_favorite': 1 if rng.random() < (level / 100) else 0
        })
user_cookies_df = pd.DataFrame(user_cookies_data)
print(f"  - 유저 쿠키: {len(user_cookies_df)}건")

user_resources_data = []
for _, user in users_df.iterrows():
    vip_mult = 1 + user['vip_level'] * 0.4
    user_resources_data.append({
        'user_id': user['user_id'],
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
print(f"  - 유저 자원: {len(user_resources_df)}건")

print("\n" + "=" * 70)
print("PART 2 완료: 모든 데이터 생성 완료")
print("=" * 70)


# ============================================================================
# PART 3: 모델 학습
# ============================================================================
print("\n" + "=" * 70)
print("PART 3: 모델 학습")
print("=" * 70)

# ----------------------------------------------------------------------------
# 3.1 번역 품질 예측 모델
# ----------------------------------------------------------------------------
print("\n[3.1] 번역 품질 예측 모델 (RandomForest)")

def quality_to_grade(score):
    if score >= 0.9: return "excellent"
    elif score >= 0.8: return "good"
    elif score >= 0.7: return "acceptable"
    else: return "needs_review"

translation_df["quality_grade"] = translation_df["quality_score"].apply(quality_to_grade)

le_category = LabelEncoder()
le_lang = LabelEncoder()
le_quality = LabelEncoder()

translation_df["category_encoded"] = le_category.fit_transform(translation_df["category"])
translation_df["target_lang_encoded"] = le_lang.fit_transform(translation_df["target_lang"])
translation_df["quality_encoded"] = le_quality.fit_transform(translation_df["quality_grade"])

feature_cols_translation = ["category_encoded", "target_lang_encoded", "fluency_score", "adequacy_score", "contains_worldview_term", "text_length"]
X_trans = translation_df[feature_cols_translation].copy()
y_trans = translation_df["quality_encoded"].copy()
X_train_trans, X_test_trans, y_train_trans, y_test_trans = train_test_split(X_trans, y_trans, test_size=0.2, random_state=seed, stratify=y_trans)

params_trans = {"n_estimators": 150, "max_depth": 10, "random_state": seed, "class_weight": "balanced"}
rf_translation = RandomForestClassifier(**params_trans, n_jobs=-1)
rf_translation.fit(X_train_trans, y_train_trans)

y_pred_trans = rf_translation.predict(X_test_trans)
accuracy_trans = accuracy_score(y_test_trans, y_pred_trans)
f1_trans = f1_score(y_test_trans, y_pred_trans, average="macro")
print(f"  Accuracy: {accuracy_trans:.4f}, F1: {f1_trans:.4f}")

if MLFLOW_AVAILABLE:
    with mlflow.start_run(run_name="translation_quality_model"):
        mlflow.set_tag("model_type", "classification")
        mlflow.log_params(params_trans)
        mlflow.log_metrics({"accuracy": accuracy_trans, "f1_macro": f1_trans})
        mlflow.sklearn.log_model(rf_translation, "model", registered_model_name="번역품질예측")

# ----------------------------------------------------------------------------
# 3.2 텍스트 카테고리 분류 모델
# ----------------------------------------------------------------------------
print("\n[3.2] 텍스트 카테고리 분류 모델 (TF-IDF + RandomForest)")

tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(worldview_df["text"])
le_text_category = LabelEncoder()
y_text_cat = le_text_category.fit_transform(worldview_df["category"])
X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(X_tfidf, y_text_cat, test_size=0.2, random_state=seed)

params_text = {"n_estimators": 100, "max_depth": 8, "random_state": seed}
rf_text = RandomForestClassifier(**params_text, n_jobs=-1)
rf_text.fit(X_train_text, y_train_text)

y_pred_text = rf_text.predict(X_test_text)
accuracy_text = accuracy_score(y_test_text, y_pred_text)
f1_text = f1_score(y_test_text, y_pred_text, average="macro")
print(f"  Accuracy: {accuracy_text:.4f}, F1: {f1_text:.4f}")

if MLFLOW_AVAILABLE:
    with mlflow.start_run(run_name="text_category_model"):
        mlflow.set_tag("model_type", "text_classification")
        mlflow.log_params(params_text)
        mlflow.log_metrics({"accuracy": accuracy_text, "f1_macro": f1_text})
        mlflow.sklearn.log_model(rf_text, "model", registered_model_name="텍스트분류")

# ----------------------------------------------------------------------------
# 3.3 유저 세그먼트 클러스터링 모델
# ----------------------------------------------------------------------------
print("\n[3.3] 유저 세그먼트 클러스터링 (K-Means)")

cluster_features = ["total_events", "stage_clears", "gacha_pulls", "pvp_battles", "purchases", "vip_level"]
X_cluster = user_agg[cluster_features].fillna(0)
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
cluster_labels = kmeans.fit_predict(X_cluster_scaled)
silhouette = silhouette_score(X_cluster_scaled, cluster_labels)
print(f"  Silhouette: {silhouette:.4f}")

user_agg["cluster"] = cluster_labels
SEGMENT_NAMES = {0: "캐주얼 플레이어", 1: "하드코어 게이머", 2: "PvP 전문가", 3: "콘텐츠 수집가", 4: "신규 유저"}

if MLFLOW_AVAILABLE:
    with mlflow.start_run(run_name="user_segmentation_model"):
        mlflow.set_tag("model_type", "clustering")
        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_metrics({"silhouette_score": silhouette})
        mlflow.sklearn.log_model(kmeans, "model", registered_model_name="유저세그먼트")

# ----------------------------------------------------------------------------
# 3.4 이상 탐지 모델
# ----------------------------------------------------------------------------
print("\n[3.4] 이상 탐지 모델 (Isolation Forest)")

params_anomaly = {"n_estimators": 150, "contamination": 0.05, "random_state": seed}
iso_forest = IsolationForest(**params_anomaly)
anomaly_pred = iso_forest.fit_predict(X_cluster_scaled)
anomaly_scores = iso_forest.decision_function(X_cluster_scaled)

anomaly_count = int((anomaly_pred == -1).sum())
print(f"  이상 유저: {anomaly_count}명 ({anomaly_count/len(anomaly_pred)*100:.1f}%)")

user_agg["is_anomaly"] = (anomaly_pred == -1).astype(int)
raw_scores = -anomaly_scores
normalized_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-8)
adjusted_scores = np.power(normalized_scores, 0.6)
noise = rng.uniform(-0.03, 0.03, len(adjusted_scores))
user_agg["anomaly_score"] = np.round(np.clip(adjusted_scores + noise, 0, 1), 4)

# 이상탐지 상세 데이터 생성
anomaly_types = [("비정상 결제 패턴", "high"), ("봇 의심 행동", "high"), ("계정 공유 의심", "medium"), ("비정상 플레이 시간", "low")]
anomaly_detail_data = []
anomaly_user_ids = user_agg[user_agg["is_anomaly"] == 1]["user_id"].tolist()
base_date_anomaly = pd.to_datetime("2025-01-31")
anomaly_idx = 0
for day in range(90):
    current_date = base_date_anomaly - pd.Timedelta(days=day)
    daily_count = rng.integers(1, 5) if current_date.dayofweek < 5 else rng.integers(2, 6)
    for _ in range(daily_count):
        anomaly_type, severity = anomaly_types[anomaly_idx % len(anomaly_types)]
        user_id = anomaly_user_ids[anomaly_idx % len(anomaly_user_ids)] if anomaly_user_ids else f"U{rng.integers(100, 200):06d}"
        anomaly_detail_data.append({
            "user_id": user_id, "anomaly_type": anomaly_type, "severity": severity,
            "anomaly_score": round(float(rng.uniform(-0.1, 0.0)), 3),
            "detected_at": (current_date + pd.Timedelta(hours=int(rng.integers(0, 24)))).isoformat(),
            "detail": f"비정상 패턴 {anomaly_idx+1}번째 감지",
        })
        anomaly_idx += 1
anomaly_detail_df = pd.DataFrame(anomaly_detail_data)
print(f"  이상탐지 상세: {len(anomaly_detail_df)}건")

if MLFLOW_AVAILABLE:
    with mlflow.start_run(run_name="anomaly_detection_model"):
        mlflow.set_tag("model_type", "anomaly_detection")
        mlflow.log_params(params_anomaly)
        mlflow.log_metrics({"anomaly_count": anomaly_count, "anomaly_ratio": anomaly_count / len(anomaly_pred)})
        mlflow.sklearn.log_model(iso_forest, "model", registered_model_name="이상탐지")

# ----------------------------------------------------------------------------
# 3.5 이탈 예측 모델 + SHAP
# ----------------------------------------------------------------------------
print("\n[3.5] 이탈 예측 모델 (RandomForest + SHAP)")

try:
    import shap
    SHAP_AVAILABLE = True
    print("  SHAP 라이브러리 로드 완료")
except ImportError:
    SHAP_AVAILABLE = False
    print("  SHAP 미설치")

CHURN_FEATURES = ["level", "days_since_register", "days_since_last_login", "total_events", "stage_clears", "gacha_pulls", "pvp_battles", "purchases", "vip_level"]
CHURN_FEATURE_NAMES_KR = {
    "level": "유저 레벨", "days_since_register": "가입 후 일수", "days_since_last_login": "마지막 접속 후 일수",
    "total_events": "총 활동량", "stage_clears": "스테이지 클리어", "gacha_pulls": "가챠 횟수",
    "pvp_battles": "PvP 전투", "purchases": "인앱 구매", "vip_level": "VIP 레벨",
}

def create_churn_probability(row, noise_scale=0.12):
    """현실적인 이탈 확률 생성 (새 피처 반영)"""
    base_rate = 0.15

    # 기존 활동 점수
    events_score = np.clip(np.log1p(row["total_events"]) / np.log1p(1000), 0, 1)
    stage_score = np.clip(np.log1p(row["stage_clears"]) / np.log1p(300), 0, 1)
    gacha_score = np.clip(np.log1p(row["gacha_pulls"]) / np.log1p(100), 0, 1)
    pvp_score = np.clip(np.log1p(row["pvp_battles"]) / np.log1p(150), 0, 1)
    purchase_score = np.clip(np.log1p(row["purchases"]) / np.log1p(80), 0, 1)
    vip_score = row["vip_level"] / 5.0

    # 새 피처 점수
    level = row.get("level", 30)
    days_since_register = row.get("days_since_register", 30)
    days_since_last_login = row.get("days_since_last_login", 7)

    # 레벨별 이탈 위험 (튜토리얼/중반/엔드게임)
    if level <= 15:
        level_risk = 0.3  # 튜토리얼 이탈
    elif level <= 30:
        level_risk = 0.2  # 초반 벽
    elif level <= 50:
        level_risk = 0.15  # 콘텐츠 벽
    else:
        level_risk = 0.05  # 엔드게임 안정

    # 마지막 접속 후 일수 (가장 강력한 지표)
    recency_risk = np.clip(days_since_last_login / 30, 0, 1)  # 30일 이상이면 max

    # 가입 후 일수 (신규 유저는 이탈 위험 높음)
    tenure_risk = 0.2 if days_since_register < 14 else (0.1 if days_since_register < 30 else 0.0)

    # 종합 활동 점수 (가중치 조정)
    activity_score = (
        events_score * 0.15 + stage_score * 0.10 + gacha_score * 0.10 +
        pvp_score * 0.10 + purchase_score * 0.15 + vip_score * 0.10 +
        (1 - level_risk) * 0.10 + (1 - recency_risk) * 0.15 + (1 - tenure_risk) * 0.05
    )

    risk_factor = 1 - activity_score
    risk_sigmoid = 1 / (1 + np.exp(-5 * (risk_factor - 0.5)))
    churn_prob = base_rate + risk_sigmoid * 0.55 + recency_risk * 0.15  # recency 직접 반영

    if row.get("is_anomaly", 0) == 1:
        churn_prob += rng.uniform(0.15, 0.25)

    noise = rng.normal(0, noise_scale)
    return round(np.clip(churn_prob + noise, 0.02, 0.98), 4)

user_agg["churn_probability_sim"] = user_agg.apply(create_churn_probability, axis=1)
user_agg["churn_risk"] = (user_agg["churn_probability_sim"] >= 0.5).astype(int)

X_churn = user_agg[CHURN_FEATURES].fillna(0)
y_churn = user_agg["churn_risk"]
X_train_churn, X_test_churn, y_train_churn, y_test_churn = train_test_split(X_churn, y_churn, test_size=0.2, random_state=seed, stratify=y_churn)

churn_params = {"n_estimators": 200, "max_depth": 8, "min_samples_split": 5, "min_samples_leaf": 2, "class_weight": "balanced", "random_state": seed}
rf_churn = RandomForestClassifier(**churn_params, n_jobs=-1)
rf_churn.fit(X_train_churn, y_train_churn)

y_pred_churn = rf_churn.predict(X_test_churn)
accuracy_churn = accuracy_score(y_test_churn, y_pred_churn)
f1_churn = f1_score(y_test_churn, y_pred_churn)
print(f"  Accuracy: {accuracy_churn:.4f}, F1: {f1_churn:.4f}")

feature_importances = dict(zip(CHURN_FEATURES, rf_churn.feature_importances_))

shap_explainer = None
shap_values_all = None
if SHAP_AVAILABLE:
    try:
        shap_explainer = shap.TreeExplainer(rf_churn)
        shap_values_raw = shap_explainer.shap_values(X_churn)
        if hasattr(shap_values_raw, 'values'):
            shap_values_all = shap_values_raw.values
        elif isinstance(shap_values_raw, list) and len(shap_values_raw) == 2:
            shap_values_all = shap_values_raw[1]
        elif isinstance(shap_values_raw, np.ndarray):
            if shap_values_raw.ndim == 3:
                shap_values_all = shap_values_raw[:, :, 1]
            else:
                shap_values_all = shap_values_raw
        else:
            shap_values_all = shap_values_raw
        shap_values_all = np.array(shap_values_all)
        for i, feat in enumerate(CHURN_FEATURES):
            user_agg[f"shap_{feat}"] = shap_values_all[:, i]
        print(f"  SHAP values 저장 완료")
    except Exception as e:
        print(f"  SHAP 분석 오류: {e}")
        SHAP_AVAILABLE = False

# 최종 이탈 확률
sim_proba = user_agg["churn_probability_sim"].values
beta_noise = rng.beta(2, 2, len(sim_proba)) * 0.1 - 0.05
user_agg["churn_probability"] = np.round(np.clip(sim_proba + beta_noise, 0.02, 0.98), 4)

def get_risk_level(prob):
    if prob >= 0.7: return "high"
    elif prob >= 0.4: return "medium"
    else: return "low"

user_agg["churn_risk_level"] = user_agg["churn_probability"].apply(get_risk_level)

if MLFLOW_AVAILABLE:
    with mlflow.start_run(run_name="churn_prediction_model"):
        mlflow.set_tag("model_type", "classification")
        mlflow.set_tag("shap_enabled", str(SHAP_AVAILABLE))
        mlflow.log_params(churn_params)
        mlflow.log_metrics({"accuracy": accuracy_churn, "f1_score": f1_churn})
        mlflow.sklearn.log_model(rf_churn, "model", registered_model_name="이탈예측")

# ----------------------------------------------------------------------------
# 3.6 승률 예측 모델 (LightGBM)
# ----------------------------------------------------------------------------
print("\n[3.6] 승률 예측 모델 (LightGBM)")

STAT_FEATURES = ['atk', 'hp', 'def', 'skill_dmg', 'cooldown', 'crit_rate', 'crit_dmg']

class WinRatePredictor:
    """스탯 기반 승률 예측 모델"""
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_fitted = False

    def _generate_synthetic_data(self, base_df, n_samples=500):
        """기존 데이터 기반 합성 데이터 생성"""
        X_list, y_list = [], []
        stat_impact = {
            'atk': 0.00012, 'hp': 0.00002, 'def': 0.006, 'skill_dmg': 0.02,
            'cooldown': -0.7, 'crit_rate': 0.12, 'crit_dmg': 0.025,
        }
        samples_per_cookie = n_samples // len(base_df)
        for _, row in base_df.iterrows():
            base_stats = np.array([row[feat] for feat in STAT_FEATURES])
            base_win_rate = row['win_rate_pvp']
            for _ in range(samples_per_cookie):
                noise_pct = np.random.uniform(-0.2, 0.2, len(STAT_FEATURES))
                new_stats = base_stats.copy()
                delta_win_rate = 0
                for i, stat in enumerate(STAT_FEATURES):
                    change = base_stats[i] * noise_pct[i]
                    new_stats[i] = base_stats[i] + change
                    delta_win_rate += change * stat_impact[stat]
                new_win_rate = np.clip(base_win_rate + delta_win_rate + np.random.normal(0, 1.5), 25, 75)
                X_list.append(new_stats)
                y_list.append(new_win_rate)
        for _, row in base_df.iterrows():
            X_list.append([row[feat] for feat in STAT_FEATURES])
            y_list.append(row['win_rate_pvp'])
        return np.array(X_list), np.array(y_list)

    def train(self, cookies_df, n_synthetic=500):
        X, y = self._generate_synthetic_data(cookies_df, n_synthetic)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        if LIGHTGBM_AVAILABLE:
            self.model = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, num_leaves=15, min_child_samples=3, random_state=42, verbose=-1)
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='r2')
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return {'cv_r2_mean': float(np.mean(cv_scores)), 'cv_r2_std': float(np.std(cv_scores)), 'n_samples': len(X)}

    def predict(self, stats):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        X = np.array([[stats.get(feat, 0) for feat in STAT_FEATURES]])
        X_scaled = self.scaler.transform(X)
        return float(np.clip(self.model.predict(X_scaled)[0], 20, 80))

    def predict_batch(self, df):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        X = df[STAT_FEATURES].values
        X_scaled = self.scaler.transform(X)
        return np.clip(self.model.predict(X_scaled), 20, 80)

    def save(self, model_path, scaler_path):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

    def load(self, model_path, scaler_path):
        try:
            if model_path.exists() and scaler_path.exists():
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.is_fitted = True
                return True
            return False
        except:
            return False

win_rate_predictor = WinRatePredictor()
win_rate_result = win_rate_predictor.train(cookie_stats_df, n_synthetic=500)
print(f"  R2: {win_rate_result['cv_r2_mean']:.3f} (+/- {win_rate_result['cv_r2_std']:.3f})")

if MLFLOW_AVAILABLE:
    with mlflow.start_run(run_name="win_rate_model"):
        mlflow.set_tag("model_type", "regression")
        mlflow.set_tag("algorithm", "LightGBM" if LIGHTGBM_AVAILABLE else "GradientBoosting")
        mlflow.log_metrics({"cv_r2_mean": win_rate_result['cv_r2_mean'], "cv_r2_std": win_rate_result['cv_r2_std']})

print("\n" + "=" * 70)
print("PART 3 완료: 모든 모델 학습 완료")
print("=" * 70)


# ============================================================================
# PART 4: 저장 및 테스트
# ============================================================================
print("\n" + "=" * 70)
print("PART 4: 저장 및 테스트")
print("=" * 70)

# CSV 저장
print("\n[4.1] CSV 파일 저장")
cookies_df.to_csv(BACKEND_DIR / "cookies.csv", index=False, encoding="utf-8-sig")
kingdoms_df.to_csv(BACKEND_DIR / "kingdoms.csv", index=False, encoding="utf-8-sig")
skills_df.to_csv(BACKEND_DIR / "skills.csv", index=False, encoding="utf-8-sig")
translation_df.to_csv(BACKEND_DIR / "translations.csv", index=False, encoding="utf-8-sig")
users_df.to_csv(BACKEND_DIR / "users.csv", index=False, encoding="utf-8-sig")
logs_df.to_csv(BACKEND_DIR / "game_logs.csv", index=False, encoding="utf-8-sig")
user_agg.to_csv(BACKEND_DIR / "user_analytics.csv", index=False, encoding="utf-8-sig")
worldview_df.to_csv(BACKEND_DIR / "worldview_texts.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(WORLDVIEW_TERMS).to_csv(BACKEND_DIR / "worldview_terms.csv", index=False, encoding="utf-8-sig")
cookie_stats_df.to_csv(BACKEND_DIR / "cookie_stats.csv", index=False, encoding="utf-8-sig")
daily_metrics_df.to_csv(BACKEND_DIR / "daily_metrics.csv", index=False, encoding="utf-8-sig")
translation_stats_df.to_csv(BACKEND_DIR / "translation_stats.csv", index=False, encoding="utf-8-sig")
anomaly_detail_df.to_csv(BACKEND_DIR / "anomaly_details.csv", index=False, encoding="utf-8-sig")
cohort_df.to_csv(BACKEND_DIR / "cohort_retention.csv", index=False, encoding="utf-8-sig")
user_activity_df.to_csv(BACKEND_DIR / "user_activity.csv", index=False, encoding="utf-8-sig")
user_cookies_df.to_csv(BACKEND_DIR / "user_cookies.csv", index=False, encoding="utf-8-sig")
user_resources_df.to_csv(BACKEND_DIR / "user_resources.csv", index=False, encoding="utf-8-sig")
print(f"  기본 데이터 + 분석 데이터 + 투자 최적화 데이터 저장 완료")

# 모델 저장
print("\n[4.2] 모델 파일 저장")
joblib.dump(rf_translation, BACKEND_DIR / "model_translation_quality.pkl")
joblib.dump(rf_text, BACKEND_DIR / "model_text_category.pkl")
joblib.dump(kmeans, BACKEND_DIR / "model_user_segment.pkl")
joblib.dump(iso_forest, BACKEND_DIR / "model_anomaly.pkl")
joblib.dump(rf_churn, BACKEND_DIR / "model_churn.pkl")
joblib.dump(tfidf, BACKEND_DIR / "tfidf_vectorizer.pkl")
joblib.dump(scaler_cluster, BACKEND_DIR / "scaler_cluster.pkl")
joblib.dump(le_category, BACKEND_DIR / "le_category.pkl")
joblib.dump(le_lang, BACKEND_DIR / "le_lang.pkl")
joblib.dump(le_quality, BACKEND_DIR / "le_quality.pkl")
joblib.dump(le_text_category, BACKEND_DIR / "le_text_category.pkl")

win_rate_predictor.save(BACKEND_DIR / "model_win_rate.pkl", BACKEND_DIR / "scaler_win_rate.pkl")

if SHAP_AVAILABLE and shap_explainer is not None:
    joblib.dump(shap_explainer, BACKEND_DIR / "shap_explainer_churn.pkl")
    print("  SHAP Explainer 저장: shap_explainer_churn.pkl")

# Churn config
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

print("  모델 저장 완료")

# 예측 함수
def predict_translation_quality(text_features):
    X = pd.DataFrame([text_features])[feature_cols_translation].astype(float)
    pred = rf_translation.predict(X)[0]
    proba = rf_translation.predict_proba(X)[0]
    quality_grade = le_quality.inverse_transform([pred])[0]
    return {"quality_grade": quality_grade, "probabilities": {le_quality.classes_[i]: float(proba[i]) for i in range(len(proba))}}

def classify_text_category(text):
    X = tfidf.transform([text])
    pred = rf_text.predict(X)[0]
    proba = rf_text.predict_proba(X)[0]
    return {"category": le_text_category.inverse_transform([pred])[0], "probabilities": {le_text_category.classes_[i]: float(proba[i]) for i in range(len(proba))}}

def get_user_segment(user_features):
    X = pd.DataFrame([user_features])[cluster_features].fillna(0)
    X_scaled = scaler_cluster.transform(X)
    cluster = int(kmeans.predict(X_scaled)[0])
    return {"cluster": cluster, "segment_name": SEGMENT_NAMES.get(cluster, "Unknown")}

def detect_anomaly(user_features):
    X = pd.DataFrame([user_features])[cluster_features].fillna(0)
    X_scaled = scaler_cluster.transform(X)
    pred = int(iso_forest.predict(X_scaled)[0])
    score = float(iso_forest.decision_function(X_scaled)[0])
    return {"is_anomaly": pred == -1, "anomaly_score": score}

# 테스트
print("\n[4.3] 예측 함수 테스트")
sample_trans = {"category_encoded": 0, "target_lang_encoded": 0, "fluency_score": 0.85, "adequacy_score": 0.90, "contains_worldview_term": 1, "text_length": 25}
print(f"  번역 품질: {predict_translation_quality(sample_trans)['quality_grade']}")
print(f"  텍스트 분류: {classify_text_category('용감한 쿠키가 오븐에서 탈출했습니다!')['category']}")
sample_user = {"total_events": 150, "stage_clears": 80, "gacha_pulls": 20, "pvp_battles": 30, "purchases": 5, "vip_level": 2}
print(f"  유저 세그먼트: {get_user_segment(sample_user)['segment_name']}")
print(f"  이상 탐지: {detect_anomaly(sample_user)['is_anomaly']}")
test_stats = {'atk': 35000, 'hp': 100000, 'def': 500, 'skill_dmg': 350, 'cooldown': 8, 'crit_rate': 20, 'crit_dmg': 180}
print(f"  승률 예측 (강한 쿠키): {win_rate_predictor.predict(test_stats):.1f}%")

print("\n" + "=" * 70)
print("완료! 데이터 생성 및 모델 학습 성공")
print("=" * 70)
print(f"\n[요약]")
print(f"  - 쿠키: {len(cookies_df)}개, 유저: {len(users_df)}명, 로그: {len(logs_df)}건")
print(f"  - 모델: 번역품질, 텍스트분류, 유저세그먼트, 이상탐지, 이탈예측, 승률예측")
print(f"  - SHAP: {'활성화' if SHAP_AVAILABLE else '비활성화'}")
print(f"\n백엔드 서버 시작: cd \"backend 리팩토링 시작\" && python main.py")
