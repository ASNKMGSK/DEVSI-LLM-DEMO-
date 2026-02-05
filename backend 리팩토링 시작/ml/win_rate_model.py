"""
ml/win_rate_model.py - 승률 예측 모델
=====================================
스탯 기반 승률 예측 모델 학습 및 추론

입력: atk, hp, def, skill_dmg, cooldown, crit_rate, crit_dmg
출력: win_rate_pvp (예측 승률)

[주피터 노트북에서 실행 시]
이 파일 전체를 복사해서 셀에 붙여넣고 실행하면 됩니다.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import logging
from typing import Dict, List, Optional, Tuple
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)

# ========================================
# MLflow 설정 (train_models.py와 동일)
# ========================================
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not available - skipping experiment tracking")

# 프로젝트 루트 (주피터/스크립트 호환)
try:
    # 스크립트 실행 시
    PROJECT_ROOT = Path(__file__).parent.parent
except NameError:
    # 주피터 노트북 실행 시
    # 이미 BACKEND_DIR이 정의되어 있으면 사용
    if 'BACKEND_DIR' in dir():
        PROJECT_ROOT = BACKEND_DIR
    else:
        # ml 폴더에서 실행 시 부모 폴더로
        _cwd = Path(".").resolve()
        if _cwd.name == "ml":
            PROJECT_ROOT = _cwd.parent
        else:
            PROJECT_ROOT = _cwd

# 스탯 컬럼
STAT_FEATURES = ['atk', 'hp', 'def', 'skill_dmg', 'cooldown', 'crit_rate', 'crit_dmg']

# 모델 저장 경로
MODEL_PATH = PROJECT_ROOT / "model_win_rate.pkl"
SCALER_PATH = PROJECT_ROOT / "scaler_win_rate.pkl"


class WinRatePredictor:
    """스탯 기반 승률 예측 모델"""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_fitted = False

    def _generate_synthetic_data(
        self,
        base_df: pd.DataFrame,
        n_samples: int = 500
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        기존 데이터 기반 합성 데이터 생성

        전략:
        - 기존 쿠키 스탯에 노이즈를 추가해서 augmentation
        - 스탯 변화에 따른 승률 변화 시뮬레이션
        """
        logger.info(f"Generating {n_samples} synthetic samples from {len(base_df)} base cookies")

        X_list = []
        y_list = []

        # 스탯별 영향도 (실제 게임 밸런싱 기반 추정)
        stat_impact = {
            'atk': 0.00012,       # ATK 1000 증가 → 승률 +0.12%
            'hp': 0.00002,        # HP 1000 증가 → 승률 +0.02%
            'def': 0.006,         # DEF 10 증가 → 승률 +0.06%
            'skill_dmg': 0.02,    # 스킬데미지 10 증가 → 승률 +0.2%
            'cooldown': -0.7,     # 쿨타임 1초 증가 → 승률 -0.7%
            'crit_rate': 0.12,    # 치명타율 1% 증가 → 승률 +0.12%
            'crit_dmg': 0.025,    # 치명타피해 10 증가 → 승률 +0.25%
        }

        samples_per_cookie = n_samples // len(base_df)

        for _, row in base_df.iterrows():
            base_stats = np.array([row[feat] for feat in STAT_FEATURES])
            base_win_rate = row['win_rate_pvp']

            for _ in range(samples_per_cookie):
                # 랜덤 스탯 변화 (-20% ~ +20%)
                noise_pct = np.random.uniform(-0.2, 0.2, len(STAT_FEATURES))

                new_stats = base_stats.copy()
                delta_win_rate = 0

                for i, stat in enumerate(STAT_FEATURES):
                    change = base_stats[i] * noise_pct[i]
                    new_stats[i] = base_stats[i] + change

                    # 승률 변화 계산
                    if stat == 'cooldown':
                        delta_win_rate += change * stat_impact['cooldown']
                    else:
                        delta_win_rate += change * stat_impact[stat]

                # 승률에 약간의 랜덤성 추가
                new_win_rate = base_win_rate + delta_win_rate + np.random.normal(0, 1.5)
                new_win_rate = np.clip(new_win_rate, 25, 75)  # 25~75% 범위 제한

                X_list.append(new_stats)
                y_list.append(new_win_rate)

        # 원본 데이터도 포함
        for _, row in base_df.iterrows():
            X_list.append([row[feat] for feat in STAT_FEATURES])
            y_list.append(row['win_rate_pvp'])

        return np.array(X_list), np.array(y_list)

    def train(self, cookies_df: pd.DataFrame, n_synthetic: int = 500) -> Dict:
        """
        모델 학습

        Args:
            cookies_df: cookie_stats.csv 데이터프레임
            n_synthetic: 생성할 합성 데이터 수

        Returns:
            학습 결과 (cv_score, feature_importance 등)
        """
        logger.info("Training win rate prediction model...")

        # 합성 데이터 생성
        X, y = self._generate_synthetic_data(cookies_df, n_synthetic)

        # 스케일링
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 모델 학습 (LightGBM)
        self.model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            num_leaves=15,
            min_child_samples=3,
            random_state=42,
            verbose=-1
        )

        # 교차 검증
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='r2')

        # 전체 데이터로 최종 학습
        self.model.fit(X_scaled, y)
        self.is_fitted = True

        # feature importance
        feature_importance = dict(zip(STAT_FEATURES, self.model.feature_importances_))

        result = {
            'cv_r2_mean': float(np.mean(cv_scores)),
            'cv_r2_std': float(np.std(cv_scores)),
            'n_samples': len(X),
            'feature_importance': feature_importance,
        }

        logger.info(f"Model trained: R2 = {result['cv_r2_mean']:.3f} (+/- {result['cv_r2_std']:.3f})")

        return result

    def predict(self, stats: Dict[str, float]) -> float:
        """
        단일 쿠키 승률 예측

        Args:
            stats: {'atk': 25000, 'hp': 80000, ...}

        Returns:
            예측 승률 (%)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() or load() first.")

        X = np.array([[stats.get(feat, 0) for feat in STAT_FEATURES]])
        X_scaled = self.scaler.transform(X)
        pred = self.model.predict(X_scaled)[0]

        return float(np.clip(pred, 20, 80))

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        여러 쿠키 승률 예측

        Args:
            df: 스탯 컬럼이 있는 데이터프레임

        Returns:
            예측 승률 배열
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() or load() first.")

        X = df[STAT_FEATURES].values
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)

        return np.clip(preds, 20, 80)

    def save(self, model_path: Path = MODEL_PATH, scaler_path: Path = SCALER_PATH):
        """모델 저장"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Model saved to {model_path}")

    def load(self, model_path: Path = MODEL_PATH, scaler_path: Path = SCALER_PATH) -> bool:
        """모델 로딩"""
        try:
            if model_path.exists() and scaler_path.exists():
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.is_fitted = True
                logger.info(f"Model loaded from {model_path}")
                return True
            else:
                logger.warning(f"Model file not found: {model_path}")
                return False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False


# 전역 인스턴스 (싱글톤 패턴)
_predictor_instance: Optional[WinRatePredictor] = None


def get_predictor() -> WinRatePredictor:
    """승률 예측 모델 인스턴스 반환"""
    global _predictor_instance

    if _predictor_instance is None:
        _predictor_instance = WinRatePredictor()
        # 저장된 모델 로딩 시도
        _predictor_instance.load()

    return _predictor_instance


def train_and_save(cookies_df: pd.DataFrame, register_mlflow: bool = True) -> Dict:
    """
    모델 학습 및 저장 (MLflow 등록 포함)

    Args:
        cookies_df: cookie_stats.csv 데이터프레임
        register_mlflow: MLflow에 등록 여부 (기본 True)

    Returns:
        학습 결과 dict
    """
    predictor = WinRatePredictor()
    result = predictor.train(cookies_df)
    predictor.save()

    # 전역 인스턴스 업데이트
    global _predictor_instance
    _predictor_instance = predictor

    # MLflow 등록
    if register_mlflow and MLFLOW_AVAILABLE:
        try:
            # MLflow 설정 (train_models.py와 동일한 경로)
            mlflow_tracking_uri = f"file:{PROJECT_ROOT / 'ml' / 'mlruns'}"
            mlflow.set_tracking_uri(mlflow_tracking_uri)

            experiment_name = "cookierun-ai-platform"
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name="win_rate_model"):
                # 태그
                mlflow.set_tag("model_type", "regression")
                mlflow.set_tag("target", "win_rate_pvp")
                mlflow.set_tag("algorithm", "LightGBM")
                mlflow.set_tag("domain", "cookierun")

                # 하이퍼파라미터
                mlflow.log_params({
                    "n_estimators": 100,
                    "max_depth": 4,
                    "learning_rate": 0.1,
                    "num_leaves": 15,
                    "min_child_samples": 3,
                    "n_features": len(STAT_FEATURES),
                    "n_samples": result['n_samples'],
                })

                # 메트릭
                mlflow.log_metrics({
                    "cv_r2_mean": result['cv_r2_mean'],
                    "cv_r2_std": result['cv_r2_std'],
                })

                # 모델 등록
                mlflow.sklearn.log_model(
                    predictor.model,
                    "win_rate_model",
                    registered_model_name="승률예측"
                )

                print(f"[MLflow] Run ID: {mlflow.active_run().info.run_id}")
                print(f"[MLflow] Model registered as 'cookierun-win-rate'")

        except Exception as e:
            logger.warning(f"MLflow registration failed: {e}")
            print(f"[Warning] MLflow 등록 실패: {e}")

    return result


# =============================================================================
# 아래부터 주피터에서 실행할 코드
# =============================================================================

if __name__ == "__main__":  # 직접 실행 시에만 학습 (주피터에서는 별도 실행)
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"MODEL_PATH: {MODEL_PATH}")

    # cookie_stats.csv 로딩
    df = pd.read_csv(PROJECT_ROOT / "cookie_stats.csv")
    print(f"\n쿠키 데이터 로딩 완료: {len(df)}개")
    print(df[['cookie_name', 'win_rate_pvp'] + STAT_FEATURES].head())

    # 학습
    print("\n" + "="*50)
    print("모델 학습 시작...")
    result = train_and_save(df)
    print(f"\n학습 결과:")
    print(f"  - R2 Score: {result['cv_r2_mean']:.3f} (+/- {result['cv_r2_std']:.3f})")
    print(f"  - 학습 샘플 수: {result['n_samples']}")
    print(f"\nFeature Importance:")
    for feat, imp in sorted(result['feature_importance'].items(), key=lambda x: -x[1]):
        print(f"  - {feat}: {imp:.3f}")

    # 예측 테스트
    print("\n" + "="*50)
    print("예측 테스트:")
    predictor = get_predictor()

    # 테스트 1: 가상의 강한 쿠키
    test_stats = {
        'atk': 35000, 'hp': 100000, 'def': 500,
        'skill_dmg': 350, 'cooldown': 8,
        'crit_rate': 20, 'crit_dmg': 180
    }
    pred = predictor.predict(test_stats)
    print(f"  강한 쿠키 (높은 스탯): {pred:.1f}%")

    # 테스트 2: 가상의 약한 쿠키
    test_stats2 = {
        'atk': 12000, 'hp': 40000, 'def': 200,
        'skill_dmg': 150, 'cooldown': 15,
        'crit_rate': 8, 'crit_dmg': 120
    }
    pred2 = predictor.predict(test_stats2)
    print(f"  약한 쿠키 (낮은 스탯): {pred2:.1f}%")

    # 테스트 3: 실제 쿠키들 예측
    print(f"\n실제 쿠키 승률 예측 vs 실제:")
    predictions = predictor.predict_batch(df)
    for i, row in df.iterrows():
        actual = row['win_rate_pvp']
        predicted = predictions[i]
        diff = predicted - actual
        print(f"  {row['cookie_name']}: 실제 {actual:.1f}% / 예측 {predicted:.1f}% (차이: {diff:+.1f}%)")
