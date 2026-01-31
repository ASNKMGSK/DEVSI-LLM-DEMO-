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
# MLflow 설정
# ========================================
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = "file:./mlruns"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

EXPERIMENT_NAME = "cookierun-ai-platform"
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    mlflow.create_experiment(EXPERIMENT_NAME)
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
print(f"MLflow Experiment: {EXPERIMENT_NAME}")

seed = 42
rng = np.random.default_rng(seed)

# ========================================
# 저장 경로
# ========================================
BACKEND_DIR = Path(__file__).parent.parent
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

# 쿠키 캐릭터 데이터 (실제 쿠키런 캐릭터 기반 + 가상 캐릭터)
COOKIES = [
    {"id": "CK001", "name": "용감한 쿠키", "name_en": "GingerBrave", "grade": "커먼", "type": "돌격",
     "story_kr": "오븐에서 탈출한 최초의 쿠키! 두려움 없이 달려나가는 용감한 쿠키입니다.",
     "story_en": "The first cookie to escape from the oven! A brave cookie who runs without fear."},
    {"id": "CK002", "name": "딸기맛 쿠키", "name_en": "Strawberry Cookie", "grade": "커먼", "type": "방어",
     "story_kr": "달콤한 딸기향이 나는 수줍은 쿠키입니다. 부끄러움이 많지만 친구를 위해선 용감해져요.",
     "story_en": "A shy cookie with a sweet strawberry scent. Though timid, becomes brave for friends."},
    {"id": "CK003", "name": "마법사맛 쿠키", "name_en": "Wizard Cookie", "grade": "레어", "type": "마법",
     "story_kr": "신비로운 마법을 부리는 쿠키입니다. 아직 수련 중이라 가끔 마법이 엉뚱하게 나가기도 해요.",
     "story_en": "A cookie who wields mysterious magic. Still in training, so spells sometimes go awry."},
    {"id": "CK004", "name": "닌자맛 쿠키", "name_en": "Ninja Cookie", "grade": "레어", "type": "돌격",
     "story_kr": "그림자처럼 빠르고 조용한 쿠키입니다. 비밀스러운 닌자 마을에서 수련을 받았어요.",
     "story_en": "A cookie as fast and silent as a shadow. Trained in a secret ninja village."},
    {"id": "CK005", "name": "뱀파이어맛 쿠키", "name_en": "Vampire Cookie", "grade": "에픽", "type": "복수",
     "story_kr": "밤의 귀족 쿠키입니다. 와인잼을 좋아하며, 햇빛을 피해 성에서 살고 있어요.",
     "story_en": "A noble cookie of the night. Loves wine jam and lives in a castle away from sunlight."},
    {"id": "CK006", "name": "허브맛 쿠키", "name_en": "Herb Cookie", "grade": "에픽", "type": "치유",
     "story_kr": "식물을 사랑하는 치유사 쿠키입니다. 허브 정원에서 다양한 약초를 기르고 있어요.",
     "story_en": "A healer cookie who loves plants. Grows various herbs in the herb garden."},
    {"id": "CK007", "name": "감초맛 쿠키", "name_en": "Licorice Cookie", "grade": "에픽", "type": "마법",
     "story_kr": "어둠의 마법을 연구하는 쿠키입니다. 감초 하인들을 소환해 싸울 수 있어요.",
     "story_en": "A cookie researching dark magic. Can summon licorice servants to fight."},
    {"id": "CK008", "name": "에스프레소맛 쿠키", "name_en": "Espresso Cookie", "grade": "에픽", "type": "마법",
     "story_kr": "커피 마법학의 천재 쿠키입니다. 차가운 성격이지만 실력은 인정받고 있어요.",
     "story_en": "A genius cookie of coffee magic. Cold personality but skills are recognized."},
    {"id": "CK009", "name": "순수 바닐라 쿠키", "name_en": "Pure Vanilla Cookie", "grade": "에인션트", "type": "치유",
     "story_kr": "고대 영웅 중 한 명인 에인션트 쿠키입니다. 평화를 사랑하며 모든 쿠키를 치유해요.",
     "story_en": "One of the Ancient Heroes. Loves peace and heals all cookies."},
    {"id": "CK010", "name": "다크카카오 쿠키", "name_en": "Dark Cacao Cookie", "grade": "에인션트", "type": "돌격",
     "story_kr": "다크카카오 왕국의 왕입니다. 무거운 검을 휘두르며 왕국을 지키고 있어요.",
     "story_en": "King of the Dark Cacao Kingdom. Wields a heavy sword to protect the kingdom."},
    {"id": "CK011", "name": "홀리베리 쿠키", "name_en": "Hollyberry Cookie", "grade": "에인션트", "type": "방어",
     "story_kr": "홀리베리 왕국의 여왕입니다. 강인하고 용감하며 모든 쿠키들의 존경을 받아요.",
     "story_en": "Queen of the Hollyberry Kingdom. Strong and brave, respected by all cookies."},
    {"id": "CK012", "name": "바다요정 쿠키", "name_en": "Sea Fairy Cookie", "grade": "레전더리", "type": "폭발",
     "story_kr": "깊은 바다에서 온 신비로운 쿠키입니다. 엄청난 마법력으로 적을 얼려버려요.",
     "story_en": "A mysterious cookie from the deep sea. Freezes enemies with tremendous magic."},
    {"id": "CK013", "name": "서리여왕 쿠키", "name_en": "Frost Queen Cookie", "grade": "레전더리", "type": "마법",
     "story_kr": "얼음 왕국을 다스리는 여왕 쿠키입니다. 차가운 눈보라로 모든 것을 얼려버려요.",
     "story_en": "Queen cookie ruling the Ice Kingdom. Freezes everything with cold blizzards."},
    {"id": "CK014", "name": "검은건포도맛 쿠키", "name_en": "Black Raisin Cookie", "grade": "에픽", "type": "돌격",
     "story_kr": "까마귀와 함께하는 미스터리한 쿠키입니다. 빠른 속도로 적을 베어버려요.",
     "story_en": "A mysterious cookie with crows. Slashes enemies at high speed."},
    {"id": "CK015", "name": "호밀맛 쿠키", "name_en": "Rye Cookie", "grade": "에픽", "type": "사격",
     "story_kr": "서부 출신의 총잡이 쿠키입니다. 정확한 사격 솜씨로 악당들을 물리쳐요.",
     "story_en": "A gunslinger cookie from the West. Defeats villains with precise shooting."},
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
]

# 스킬 설명 데이터
SKILLS = [
    {"cookie_id": "CK001", "skill_name": "용감한 돌진", "skill_name_en": "Brave Dash",
     "desc_kr": "전방으로 돌진하며 적에게 피해를 줍니다.", "desc_en": "Dashes forward and damages enemies."},
    {"cookie_id": "CK002", "skill_name": "딸기 방패", "skill_name_en": "Strawberry Shield",
     "desc_kr": "딸기 방패로 아군을 보호합니다.", "desc_en": "Protects allies with a strawberry shield."},
    {"cookie_id": "CK003", "skill_name": "번개 마법", "skill_name_en": "Lightning Magic",
     "desc_kr": "강력한 번개를 소환하여 적을 공격합니다.", "desc_en": "Summons powerful lightning to attack enemies."},
    {"cookie_id": "CK005", "skill_name": "피의 연회", "skill_name_en": "Blood Feast",
     "desc_kr": "적의 생명력을 흡수하여 자신을 회복합니다.", "desc_en": "Absorbs enemy life force to heal self."},
    {"cookie_id": "CK006", "skill_name": "생명의 허브", "skill_name_en": "Herb of Life",
     "desc_kr": "아군 전체의 HP를 회복시킵니다.", "desc_en": "Heals HP of all allies."},
    {"cookie_id": "CK009", "skill_name": "순수한 치유", "skill_name_en": "Pure Healing",
     "desc_kr": "신성한 빛으로 아군을 치유하고 보호합니다.", "desc_en": "Heals and protects allies with holy light."},
    {"cookie_id": "CK010", "skill_name": "다크카카오의 심판", "skill_name_en": "Dark Cacao's Judgment",
     "desc_kr": "거대한 검으로 적에게 강력한 일격을 가합니다.", "desc_en": "Strikes enemies with a massive sword."},
    {"cookie_id": "CK012", "skill_name": "해일 소환", "skill_name_en": "Summon Tidal Wave",
     "desc_kr": "거대한 해일을 소환하여 적을 휩쓸어버립니다.", "desc_en": "Summons a massive tidal wave to sweep enemies."},
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
# 3. 게임 로그 데이터 생성 (분석용)
# ========================================
print("\n" + "=" * 60)
print("3. 게임 로그 데이터 생성")
print("=" * 60)

n_users = 1000
n_logs = 50000

users_df = pd.DataFrame({
    "user_id": [f"U{i:06d}" for i in range(1, n_users + 1)],
    "country": rng.choice(["KR", "US", "JP", "CN", "TW", "TH", "ID"], n_users,
                          p=[0.3, 0.2, 0.15, 0.15, 0.08, 0.07, 0.05]),
    "register_date": pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, n_users), unit="D"),
    "vip_level": rng.choice([0, 1, 2, 3, 4, 5], n_users, p=[0.5, 0.25, 0.12, 0.08, 0.03, 0.02]),
})

# 게임 이벤트 로그
EVENT_TYPES = ["stage_clear", "gacha_pull", "cookie_upgrade", "pvp_battle", "guild_activity", "shop_purchase"]

logs = []
for _ in tqdm(range(n_logs), desc="게임 로그 생성"):
    user = rng.choice(users_df["user_id"].values)
    event_type = rng.choice(EVENT_TYPES, p=[0.35, 0.15, 0.15, 0.15, 0.10, 0.10])

    log_date = pd.to_datetime("2024-06-01") + pd.to_timedelta(rng.integers(0, 180), unit="D")

    # 이벤트별 상세 데이터
    if event_type == "stage_clear":
        stage = rng.integers(1, 500)
        stars = rng.choice([1, 2, 3], p=[0.2, 0.3, 0.5])
        detail = {"stage": stage, "stars": stars}
    elif event_type == "gacha_pull":
        pull_type = rng.choice(["single", "ten_pull"], p=[0.3, 0.7])
        got_epic = rng.choice([0, 1], p=[0.9, 0.1])
        detail = {"pull_type": pull_type, "got_epic": got_epic}
    elif event_type == "pvp_battle":
        win = rng.choice([0, 1], p=[0.45, 0.55])
        detail = {"result": "win" if win else "lose"}
    else:
        detail = {}

    logs.append({
        "log_id": f"LOG{len(logs):08d}",
        "user_id": user,
        "event_type": event_type,
        "event_date": log_date,
        "detail": json.dumps(detail, ensure_ascii=False),
    })

logs_df = pd.DataFrame(logs)
print(f"유저 데이터: {len(users_df)}명")
print(f"게임 로그: {len(logs_df)}건")

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

# MLflow 실험 추적
with mlflow.start_run(run_name="translation_quality_model"):
    mlflow.set_tag("model_type", "classification")
    mlflow.set_tag("target", "translation_quality_grade")
    mlflow.set_tag("domain", "cookierun")

    params = {"n_estimators": 150, "max_depth": 10, "random_state": seed, "class_weight": "balanced"}
    mlflow.log_params(params)
    mlflow.log_param("n_features", len(feature_cols_translation))
    mlflow.log_param("train_samples", len(X_train_trans))
    mlflow.log_param("test_samples", len(X_test_trans))

    rf_translation = RandomForestClassifier(**params, n_jobs=-1)
    rf_translation.fit(X_train_trans, y_train_trans)

    y_pred_trans = rf_translation.predict(X_test_trans)
    accuracy = accuracy_score(y_test_trans, y_pred_trans)
    f1_macro = f1_score(y_test_trans, y_pred_trans, average="macro")

    mlflow.log_metrics({"accuracy": accuracy, "f1_macro": f1_macro})
    mlflow.sklearn.log_model(rf_translation, "translation_quality_model",
                             registered_model_name="cookierun-translation-quality")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 (macro): {f1_macro:.4f}")
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

with mlflow.start_run(run_name="text_category_model"):
    mlflow.set_tag("model_type", "text_classification")
    mlflow.set_tag("vectorizer", "TF-IDF")

    params = {"n_estimators": 100, "max_depth": 8, "random_state": seed}
    mlflow.log_params(params)

    rf_text = RandomForestClassifier(**params, n_jobs=-1)
    rf_text.fit(X_train_text, y_train_text)

    y_pred_text = rf_text.predict(X_test_text)
    accuracy_text = accuracy_score(y_test_text, y_pred_text)
    f1_text = f1_score(y_test_text, y_pred_text, average="macro")

    mlflow.log_metrics({"accuracy": accuracy_text, "f1_macro": f1_text})
    mlflow.sklearn.log_model(rf_text, "text_category_model",
                             registered_model_name="cookierun-text-category")

    print(f"Accuracy: {accuracy_text:.4f}")
    print(f"F1 (macro): {f1_text:.4f}")
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

with mlflow.start_run(run_name="user_segmentation_model"):
    mlflow.set_tag("model_type", "clustering")
    mlflow.set_tag("algorithm", "KMeans")

    n_clusters = 5
    mlflow.log_param("n_clusters", n_clusters)

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_labels = kmeans.fit_predict(X_cluster_scaled)

    silhouette = silhouette_score(X_cluster_scaled, cluster_labels)
    inertia = kmeans.inertia_

    mlflow.log_metrics({"silhouette_score": silhouette, "inertia": inertia})
    mlflow.sklearn.log_model(kmeans, "user_segmentation_model",
                             registered_model_name="cookierun-user-segment")

    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Inertia: {inertia:.2f}")
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

with mlflow.start_run(run_name="anomaly_detection_model"):
    mlflow.set_tag("model_type", "anomaly_detection")

    params = {"n_estimators": 150, "contamination": 0.05, "random_state": seed}
    mlflow.log_params(params)

    iso_forest = IsolationForest(**params)
    anomaly_pred = iso_forest.fit_predict(X_cluster_scaled)
    anomaly_scores = iso_forest.decision_function(X_cluster_scaled)

    anomaly_count = int((anomaly_pred == -1).sum())
    normal_count = int((anomaly_pred == 1).sum())

    mlflow.log_metrics({
        "anomaly_count": anomaly_count,
        "normal_count": normal_count,
        "anomaly_ratio": anomaly_count / len(anomaly_pred),
    })
    mlflow.sklearn.log_model(iso_forest, "anomaly_model",
                             registered_model_name="cookierun-anomaly-detection")

    print(f"정상 유저: {normal_count}명 ({normal_count/len(anomaly_pred)*100:.1f}%)")
    print(f"이상 유저: {anomaly_count}명 ({anomaly_count/len(anomaly_pred)*100:.1f}%)")
    print(f"[MLflow] Run ID: {mlflow.active_run().info.run_id}")

user_agg["is_anomaly"] = (anomaly_pred == -1).astype(int)
user_agg["anomaly_score"] = anomaly_scores

# ========================================
# 8. 파일 저장
# ========================================
print("\n" + "=" * 60)
print("파일 저장")
print("=" * 60)

# CSV 저장
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
print(f"  - cookies.csv ({len(cookies_df)} rows)")
print(f"  - kingdoms.csv ({len(kingdoms_df)} rows)")
print(f"  - skills.csv ({len(skills_df)} rows)")
print(f"  - translations.csv ({len(translation_df)} rows)")
print(f"  - users.csv ({len(users_df)} rows)")
print(f"  - game_logs.csv ({len(logs_df)} rows)")
print(f"  - user_analytics.csv ({len(user_agg)} rows)")
print(f"  - worldview_texts.csv ({len(worldview_df)} rows)")
print(f"  - worldview_terms.csv ({len(terms_df)} rows)")
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
# 11. MLflow 요약
# ========================================
print("\n" + "=" * 60)
print("MLflow 실험 추적 완료")
print("=" * 60)
print(f"Tracking URI: {MLFLOW_TRACKING_URI}")
print(f"Experiment: {EXPERIMENT_NAME}")
print("\n등록된 모델:")
print("  - cookierun-translation-quality (번역 품질 예측)")
print("  - cookierun-text-category (텍스트 카테고리 분류)")
print("  - cookierun-user-segment (유저 세그먼트)")
print("  - cookierun-anomaly-detection (이상 탐지)")
print("\nMLflow UI 실행: mlflow ui --port 5000")
