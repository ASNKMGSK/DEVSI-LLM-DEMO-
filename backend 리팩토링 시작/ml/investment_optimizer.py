"""
ml/investment_optimizer.py - 자원 투자 최적화 (ML + P-PSO)
==========================================================
유저의 보유 쿠키/자원을 분석하여 최적의 투자 전략을 제안

핵심 기능:
1. 투자 효율 계산: 레벨업/스킬업/각성 시 예상 전투력/승률 증가
2. P-PSO 최적화: 자원 제약 하에서 최적 투자 조합 탐색
3. 개인화: 유저마다 다른 보유 현황 → 다른 추천 결과
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
import json

logger = logging.getLogger(__name__)

# 프로젝트 루트
try:
    PROJECT_ROOT = Path(__file__).parent.parent
except NameError:
    # 주피터 노트북 실행 시
    if 'BACKEND_DIR' in dir():
        PROJECT_ROOT = BACKEND_DIR
    else:
        _cwd = Path(".").resolve()
        if _cwd.name == "ml":
            PROJECT_ROOT = _cwd.parent
        else:
            PROJECT_ROOT = _cwd

# ========================================
# 스탯 증가 규칙 (쿠키런 킹덤 기반)
# ========================================

# 등급별 기본 스탯 배율
GRADE_MULTIPLIER = {
    '커먼': 1.0,
    '레어': 1.15,
    '슈퍼레어': 1.3,
    '에픽': 1.5,
    '에인션트': 1.8,
    '레전더리': 2.0
}

# 레벨업 시 스탯 증가율 (레벨당)
LEVEL_STAT_GAIN = {
    'atk': 0.012,      # 레벨당 ATK +1.2%
    'hp': 0.010,       # 레벨당 HP +1.0%
    'def': 0.008,      # 레벨당 DEF +0.8%
}

# 스킬 레벨업 시 스킬데미지 증가율
SKILL_LEVEL_GAIN = 0.015  # 스킬레벨당 스킬데미지 +1.5%

# 각성 시 스탯 증가율 (단계당)
ASCENSION_STAT_GAIN = {
    0: 0,       # 0성 → 1성
    1: 0.05,    # 1성 → 2성: 전체 스탯 +5%
    2: 0.05,    # 2성 → 3성: 전체 스탯 +5%
    3: 0.08,    # 3성 → 4성: 전체 스탯 +8%
    4: 0.10,    # 4성 → 5성: 전체 스탯 +10%
}


class InvestmentOptimizer:
    """자원 투자 최적화기"""

    def __init__(self, user_id: str, resources: dict = None):
        self.user_id = user_id
        self.user_cookies_df = None
        self.user_resources = None
        self.level_cost_df = None
        self.cookie_stats_df = None
        self.win_rate_predictor = None
        self._provided_resources = resources  # 프론트엔드에서 전달받은 자원

        self._load_data()
        self._load_win_rate_model()

        # 프론트엔드에서 전달받은 자원이 있으면 사용 (우선순위)
        if self._provided_resources:
            self.user_resources = {
                'exp_jelly': int(self._provided_resources.get('exp_jelly', 0)),
                'coin': int(self._provided_resources.get('coin', 0)),
                'skill_powder': int(self._provided_resources.get('skill_powder', 0)),
                'soul_stone': int(self._provided_resources.get('soul_stone', 0)),
            }
            logger.info(f"Using provided resources: {self.user_resources}")

    def _load_data(self):
        """데이터 로딩"""
        try:
            # 유저 쿠키 데이터
            user_cookies_path = PROJECT_ROOT / "user_cookies.csv"
            if user_cookies_path.exists():
                all_user_cookies = pd.read_csv(user_cookies_path)
                self.user_cookies_df = all_user_cookies[
                    all_user_cookies['user_id'] == self.user_id
                ].copy()
                logger.info(f"Loaded {len(self.user_cookies_df)} cookies for user {self.user_id}")
            else:
                logger.warning("user_cookies.csv not found")
                self.user_cookies_df = pd.DataFrame()

            # 유저 자원 데이터
            user_resources_path = PROJECT_ROOT / "user_resources.csv"
            if user_resources_path.exists():
                all_resources = pd.read_csv(user_resources_path)
                user_row = all_resources[all_resources['user_id'] == self.user_id]
                if len(user_row) > 0:
                    row = user_row.iloc[0].to_dict()
                    # soul_stone 통합 (개별 등급 합산)
                    total_soul_stone = sum([
                        int(row.get('soul_stone_common', 0)),
                        int(row.get('soul_stone_rare', 0)),
                        int(row.get('soul_stone_epic', 0)),
                        int(row.get('soul_stone_ancient', 0)),
                        int(row.get('soul_stone_legendary', 0)),
                    ])
                    self.user_resources = {
                        'exp_jelly': int(row.get('exp_jelly', 0)),
                        'coin': int(row.get('coin', 0)),
                        'skill_powder': int(row.get('skill_powder', 0)),
                        'soul_stone': total_soul_stone,
                    }
                else:
                    self.user_resources = {}
                logger.info(f"Loaded resources for user {self.user_id}: {self.user_resources}")
            else:
                logger.warning("user_resources.csv not found")
                self.user_resources = {}

            # 레벨업 비용 테이블
            level_cost_path = PROJECT_ROOT / "level_cost.csv"
            if level_cost_path.exists():
                self.level_cost_df = pd.read_csv(level_cost_path)
            else:
                logger.warning("level_cost.csv not found")
                self.level_cost_df = pd.DataFrame()

            # 쿠키 스탯 데이터
            cookie_stats_path = PROJECT_ROOT / "cookie_stats.csv"
            if cookie_stats_path.exists():
                self.cookie_stats_df = pd.read_csv(cookie_stats_path)
            else:
                logger.warning("cookie_stats.csv not found")
                self.cookie_stats_df = pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def _load_win_rate_model(self):
        """승률 예측 모델 로딩"""
        try:
            from ml.win_rate_model import get_predictor
            self.win_rate_predictor = get_predictor()
            if self.win_rate_predictor and self.win_rate_predictor.is_fitted:
                logger.info("Win rate predictor loaded successfully")
            else:
                logger.warning("Win rate predictor not fitted")
                self.win_rate_predictor = None
        except Exception as e:
            logger.warning(f"Failed to load win rate model: {e}")
            self.win_rate_predictor = None

    def get_cookie_current_stats(self, cookie_id: str) -> Optional[Dict]:
        """쿠키의 현재 스탯 조회"""
        if self.cookie_stats_df is None or len(self.cookie_stats_df) == 0:
            return None

        row = self.cookie_stats_df[self.cookie_stats_df['cookie_id'] == cookie_id]
        if len(row) == 0:
            return None

        return row.iloc[0].to_dict()

    def get_user_cookie_status(self, cookie_id: str) -> Optional[Dict]:
        """유저의 특정 쿠키 보유 상태 조회"""
        if self.user_cookies_df is None or len(self.user_cookies_df) == 0:
            return None

        row = self.user_cookies_df[self.user_cookies_df['cookie_id'] == cookie_id]
        if len(row) == 0:
            return None

        return row.iloc[0].to_dict()

    def get_upgrade_cost(self, upgrade_type: str, from_level: int, to_level: int) -> Dict[str, int]:
        """업그레이드 비용 계산"""
        if self.level_cost_df is None or len(self.level_cost_df) == 0:
            # 기본 비용 반환 (테이블 없을 때)
            return {'exp_jelly': 10000, 'coin': 20000, 'skill_powder': 500, 'soul_stone': 0}

        # 테이블 구간과 겹치는 모든 비용 합산
        # level_cost.csv: from_level ~ to_level 구간별 비용
        costs = self.level_cost_df[
            (self.level_cost_df['upgrade_type'] == upgrade_type) &
            (self.level_cost_df['from_level'] < to_level) &
            (self.level_cost_df['to_level'] > from_level)
        ]

        if len(costs) == 0:
            # 매칭 안 되면 기본 비용
            return {'exp_jelly': 10000, 'coin': 20000, 'skill_powder': 500, 'soul_stone': 0}

        return {
            'exp_jelly': int(costs['exp_jelly'].sum()),
            'coin': int(costs['coin'].sum()),
            'skill_powder': int(costs['skill_powder'].sum()),
            'soul_stone': int(costs['soul_stone'].sum()),
        }

    def calculate_stat_after_upgrade(
        self,
        current_stats: Dict,
        upgrade_type: str,
        levels: int,
        current_level: int
    ) -> Dict[str, float]:
        """업그레이드 후 예상 스탯 계산"""
        new_stats = current_stats.copy()

        if upgrade_type == 'cookie_level':
            # 레벨업: atk, hp, def 증가
            for stat, gain_rate in LEVEL_STAT_GAIN.items():
                if stat in new_stats:
                    # 복리 계산
                    multiplier = (1 + gain_rate) ** levels
                    new_stats[stat] = float(new_stats[stat]) * multiplier

        elif upgrade_type == 'skill_level':
            # 스킬업: skill_dmg 증가
            if 'skill_dmg' in new_stats:
                multiplier = (1 + SKILL_LEVEL_GAIN) ** levels
                new_stats['skill_dmg'] = float(new_stats['skill_dmg']) * multiplier

        elif upgrade_type == 'ascension':
            # 각성: 전체 스탯 증가
            total_gain = sum(
                ASCENSION_STAT_GAIN.get(i, 0)
                for i in range(current_level, current_level + levels)
            )
            for stat in ['atk', 'hp', 'def', 'skill_dmg', 'crit_rate', 'crit_dmg']:
                if stat in new_stats:
                    new_stats[stat] = float(new_stats[stat]) * (1 + total_gain)

        return new_stats

    def calculate_investment_efficiency(
        self,
        cookie_id: str,
        upgrade_type: str,
        levels: int = 10
    ) -> Dict[str, Any]:
        """
        단일 투자의 효율 계산

        Args:
            cookie_id: 쿠키 ID
            upgrade_type: 'cookie_level' | 'skill_level' | 'ascension'
            levels: 업그레이드할 레벨 수

        Returns:
            {
                'cookie_id': str,
                'upgrade_type': str,
                'levels': int,
                'cost': {...},
                'stat_gain': {...},
                'win_rate_before': float,
                'win_rate_after': float,
                'win_rate_gain': float,
                'efficiency': float  # win_rate_gain / total_cost
            }
        """
        current_stats = self.get_cookie_current_stats(cookie_id)
        user_status = self.get_user_cookie_status(cookie_id)

        if current_stats is None:
            return {'error': f'Cookie {cookie_id} not found in stats'}
        if user_status is None:
            return {'error': f'User does not own cookie {cookie_id}'}

        # 현재 레벨 (level_cost.csv 구간에 맞게 계산)
        if upgrade_type == 'cookie_level':
            current_level = user_status.get('cookie_level', 1)
            # 구간: 1-10, 10-20, 20-30, ... (10 단위)
            from_level = (current_level // 10) * 10  # 41 → 40
            to_level = min(from_level + 10, 70)      # 40 → 50
            levels = to_level - from_level
        elif upgrade_type == 'skill_level':
            current_level = user_status.get('skill_level', 1)
            from_level = (current_level // 10) * 10  # 41 → 40
            to_level = min(from_level + 10, 70)      # 40 → 50
            levels = to_level - from_level
        elif upgrade_type == 'ascension':
            current_level = user_status.get('ascension', 0)
            from_level = current_level
            to_level = min(current_level + levels, 5)
            levels = to_level - from_level
        else:
            return {'error': f'Invalid upgrade_type: {upgrade_type}'}

        if levels <= 0:
            return {'error': 'Already at max level'}

        # 비용 계산
        cost = self.get_upgrade_cost(upgrade_type, from_level, to_level)
        total_cost = (
            cost['exp_jelly'] * 0.001 +  # exp_jelly 가치
            cost['coin'] * 0.0001 +       # coin 가치
            cost['skill_powder'] * 0.1 +  # skill_powder 가치
            cost['soul_stone'] * 1.0      # soul_stone 가치 (기준)
        )

        # 스탯 변화 계산
        new_stats = self.calculate_stat_after_upgrade(
            current_stats, upgrade_type, levels, current_level
        )

        # 승률 예측
        if self.win_rate_predictor:
            win_rate_before = self.win_rate_predictor.predict({
                'atk': current_stats.get('atk', 0),
                'hp': current_stats.get('hp', 0),
                'def': current_stats.get('def', 0),
                'skill_dmg': current_stats.get('skill_dmg', 0),
                'cooldown': current_stats.get('cooldown', 10),
                'crit_rate': current_stats.get('crit_rate', 0),
                'crit_dmg': current_stats.get('crit_dmg', 100),
            })
            win_rate_after = self.win_rate_predictor.predict({
                'atk': new_stats.get('atk', 0),
                'hp': new_stats.get('hp', 0),
                'def': new_stats.get('def', 0),
                'skill_dmg': new_stats.get('skill_dmg', 0),
                'cooldown': new_stats.get('cooldown', 10),
                'crit_rate': new_stats.get('crit_rate', 0),
                'crit_dmg': new_stats.get('crit_dmg', 100),
            })
        else:
            # 휴리스틱
            win_rate_before = current_stats.get('win_rate_pvp', 50)
            # 스탯 증가에 따른 대략적 승률 증가
            stat_boost = (
                (new_stats.get('atk', 0) - current_stats.get('atk', 0)) * 0.0001 +
                (new_stats.get('hp', 0) - current_stats.get('hp', 0)) * 0.00002 +
                (new_stats.get('skill_dmg', 0) - current_stats.get('skill_dmg', 0)) * 0.01
            )
            win_rate_after = min(80, win_rate_before + stat_boost)

        win_rate_gain = win_rate_after - win_rate_before
        # 효율성: 승률 증가 / 비용 (합리적인 범위로 제한)
        efficiency = win_rate_gain / max(total_cost, 10.0)  # 최소 비용 10
        efficiency = min(efficiency, 1.0)  # 최대 효율 1.0 (100%)

        return {
            'cookie_id': cookie_id,
            'cookie_name': current_stats.get('cookie_name', cookie_id),
            'grade': current_stats.get('grade', ''),
            'upgrade_type': upgrade_type,
            'from_level': current_level,  # 실제 현재 레벨 표시
            'to_level': to_level,
            'levels': levels,
            'cost': cost,
            'total_cost_normalized': round(total_cost, 2),
            'stat_gain': {
                k: round(new_stats.get(k, 0) - current_stats.get(k, 0), 1)
                for k in ['atk', 'hp', 'def', 'skill_dmg']
            },
            'win_rate_before': round(win_rate_before, 1),
            'win_rate_after': round(win_rate_after, 1),
            'win_rate_gain': round(win_rate_gain, 2),
            'efficiency': round(efficiency, 4),
        }

    def optimize(
        self,
        goal: str = 'maximize_win_rate',
        max_iterations: int = 200,
        top_n: int = 10
    ) -> Dict[str, Any]:
        """
        P-PSO로 최적 투자 조합 탐색

        Args:
            goal: 'maximize_win_rate' | 'maximize_power' | 'balanced'
            max_iterations: P_PSO 반복 횟수 (기본 200)
            top_n: 상위 N개 추천 (최대 10개)

        Returns:
            {
                'user_id': str,
                'goal': str,
                'recommendations': [...],
                'total_cost': {...},
                'total_win_rate_gain': float,
                'resource_usage': {...}
            }
        """
        # 최대 10개로 제한
        top_n = min(top_n, 10)

        if self.user_cookies_df is None or len(self.user_cookies_df) == 0:
            return {'error': '해당 유저의 쿠키를 찾을 수 없습니다'}

        # 모든 가능한 투자 옵션의 효율 계산
        all_options = []

        for _, row in self.user_cookies_df.iterrows():
            cookie_id = row['cookie_id']

            for upgrade_type in ['cookie_level', 'skill_level', 'ascension']:
                # levels는 calculate_investment_efficiency 내부에서 자동 계산됨
                # (현재 레벨 기준으로 다음 구간까지)
                result = self.calculate_investment_efficiency(
                    cookie_id, upgrade_type, 10  # levels는 내부에서 재계산됨
                )

                if 'error' not in result and result['efficiency'] > 0:
                    all_options.append(result)

        if not all_options:
            return {'error': '유효한 업그레이드 옵션이 없습니다'}

        # 효율 기준 정렬
        all_options.sort(key=lambda x: x['efficiency'], reverse=True)

        # 1. 먼저 단일로 구매 가능한 옵션만 필터링
        affordable_options = self._filter_affordable_options(all_options)

        if not affordable_options:
            return {'error': '보유 자원으로 가능한 업그레이드가 없습니다. 자원을 더 모아주세요.'}

        logger.info(f"Affordable options: {len(affordable_options)} / {len(all_options)}")

        # 2. P-PSO로 최적 조합 탐색 (실패 시 greedy fallback)
        try:
            recommendations = self._run_pso_optimization(
                affordable_options, goal, max_iterations, top_n
            )
        except Exception as e:
            logger.warning(f"PSO optimization failed: {e}, using greedy fallback")
            # Greedy fallback: 자원 내에서 승률 증가 순으로 선택
            recommendations = self._filter_by_resources(affordable_options)
            if not recommendations:
                return {'error': '최적화 실패하였습니다'}

        # 승률 증가가 0% 이하인 추천 제외
        recommendations = [r for r in recommendations if r['win_rate_gain'] > 0]

        if not recommendations:
            return {'error': '승률 증가가 있는 유효한 업그레이드 옵션이 없습니다'}

        # 총 비용/효과 계산
        total_cost = {
            'exp_jelly': sum(r['cost']['exp_jelly'] for r in recommendations),
            'coin': sum(r['cost']['coin'] for r in recommendations),
            'skill_powder': sum(r['cost']['skill_powder'] for r in recommendations),
            'soul_stone': sum(r['cost']['soul_stone'] for r in recommendations),
        }
        total_win_rate_gain = sum(r['win_rate_gain'] for r in recommendations)

        # 자원 사용률
        resource_usage = {}
        if self.user_resources:
            for resource, cost in total_cost.items():
                available = self.user_resources.get(resource, 0)
                usage_pct = (cost / available * 100) if available > 0 else 0
                resource_usage[resource] = {
                    'available': available,
                    'cost': cost,
                    'usage_pct': round(usage_pct, 1),
                    'affordable': cost <= available
                }

        return {
            'user_id': self.user_id,
            'goal': goal,
            'recommendations': recommendations,
            'total_cost': total_cost,
            'total_win_rate_gain': round(total_win_rate_gain, 2),
            'resource_usage': resource_usage,
            'optimization_method': 'P-PSO (Phasor Particle Swarm Optimization)',
        }

    def _filter_affordable_options(self, options: List[Dict]) -> List[Dict]:
        """
        단일로 구매 가능한 옵션만 필터링 (PSO 전처리)

        각 옵션이 개별적으로 보유 자원 내에서 구매 가능한지 체크
        """
        if not self.user_resources:
            return options

        available = {
            'exp_jelly': self.user_resources.get('exp_jelly', 0),
            'coin': self.user_resources.get('coin', 0),
            'skill_powder': self.user_resources.get('skill_powder', 0),
            'soul_stone': self.user_resources.get('soul_stone', 0),
        }

        affordable = []
        for opt in options:
            cost = opt.get('cost', {})
            # 단일 옵션이 자원 내인지 체크
            can_afford = all(
                cost.get(res, 0) <= available.get(res, 0)
                for res in ['exp_jelly', 'coin', 'skill_powder', 'soul_stone']
            )
            if can_afford:
                affordable.append(opt)

        return affordable

    def _filter_by_resources(self, options: List[Dict]) -> List[Dict]:
        """
        자원 제약 내에서 유효한 옵션만 필터링 (최대 10개)

        전략:
        1. win_rate_gain 순으로 정렬 (승률 증가가 큰 것 우선)
        2. 자원 내에서 최대한 선택
        3. 같은 쿠키의 같은 업그레이드는 중복 방지
        """
        if not self.user_resources:
            return options[:10]

        available = {
            'exp_jelly': self.user_resources.get('exp_jelly', 0),
            'coin': self.user_resources.get('coin', 0),
            'skill_powder': self.user_resources.get('skill_powder', 0),
            'soul_stone': self.user_resources.get('soul_stone', 0),
        }

        # 승률 증가 순으로 정렬
        sorted_by_winrate = sorted(options, key=lambda x: x.get('win_rate_gain', 0), reverse=True)

        selected = []
        remaining = available.copy()
        used_cookies = set()  # 같은 쿠키+업그레이드 중복 방지

        for opt in sorted_by_winrate:
            cookie_id = opt.get('cookie_id')
            upgrade_type = opt.get('upgrade_type')
            key = f"{cookie_id}_{upgrade_type}"

            if key in used_cookies:
                continue

            cost = opt.get('cost', {})

            # 모든 자원이 충분한지 체크
            can_afford = all(
                cost.get(res, 0) <= remaining.get(res, 0)
                for res in ['exp_jelly', 'coin', 'skill_powder', 'soul_stone']
            )

            if can_afford:
                selected.append(opt)
                used_cookies.add(key)
                for res in remaining:
                    remaining[res] -= cost.get(res, 0)

                # 최대 10개 제한
                if len(selected) >= 10:
                    break

        # 자원 활용률 로깅
        total_used = {res: available[res] - remaining[res] for res in available}
        for res in available:
            if available[res] > 0:
                pct = total_used[res] / available[res] * 100
                logger.info(f"Resource {res}: used {total_used[res]:,} / {available[res]:,} ({pct:.1f}%)")

        logger.info(f"P-PSO selected {len(selected)} options within resource constraints (max 10)")
        return selected

    def _run_pso_optimization(
        self,
        options: List[Dict],
        goal: str,
        max_iterations: int,
        top_n: int
    ) -> List[Dict]:
        """P-PSO 최적화 실행 (최대 10개) - P_PSO만 사용"""
        # mealpy 3.0.3: P_PSO (Phasor PSO)
        from mealpy.swarm_based.PSO import P_PSO
        from mealpy.utils.space import BinaryVar
        logger.info(f"Using P_PSO with {max_iterations} iterations")

        n_options = len(options)
        # 최대 10개로 제한
        top_n = min(top_n, 10)

        if n_options == 0:
            raise ValueError("최적화할 옵션이 없습니다")

        # 자원 제약
        available_resources = {
            'exp_jelly': self.user_resources.get('exp_jelly', float('inf')) if self.user_resources else float('inf'),
            'coin': self.user_resources.get('coin', float('inf')) if self.user_resources else float('inf'),
            'skill_powder': self.user_resources.get('skill_powder', float('inf')) if self.user_resources else float('inf'),
            'soul_stone': self.user_resources.get('soul_stone', 0) if self.user_resources else 0,
        }

        def fitness_function(solution):
            """
            solution: [0,1,0,1,...] 각 옵션 선택 여부 (이진)
            """
            selected_indices = np.where(np.array(solution) > 0.5)[0]

            if len(selected_indices) == 0:
                return -1000  # 아무것도 선택 안하면 최악
            if len(selected_indices) > top_n:
                return -500   # 너무 많이 선택하면 페널티

            selected = [options[i] for i in selected_indices]

            # 자원 제약 체크
            total_cost = {
                'exp_jelly': sum(o['cost']['exp_jelly'] for o in selected),
                'coin': sum(o['cost']['coin'] for o in selected),
                'skill_powder': sum(o['cost']['skill_powder'] for o in selected),
                'soul_stone': sum(o['cost']['soul_stone'] for o in selected),
            }

            for resource, cost in total_cost.items():
                if cost > available_resources.get(resource, float('inf')):
                    return -100  # 자원 초과 페널티

            # 목표에 따른 점수
            if goal == 'maximize_win_rate':
                score = sum(o['win_rate_gain'] for o in selected)
            elif goal == 'maximize_efficiency':
                score = sum(o['efficiency'] for o in selected)
            else:  # balanced
                score = (
                    sum(o['win_rate_gain'] for o in selected) * 0.5 +
                    sum(o['efficiency'] for o in selected) * 10
                )

            return score

        # mealpy 3.x 방식: BinaryVar 사용
        bounds = BinaryVar(n_vars=n_options, name="selection")

        problem = {
            "obj_func": fitness_function,
            "bounds": bounds,
            "minmax": "max",
        }

        # P_PSO 실행 (iteration 증가, pop_size 증가)
        model = P_PSO(epoch=max_iterations, pop_size=50)
        best_agent = model.solve(problem)
        best_solution = best_agent.solution

        # 최적 해에서 선택된 옵션 추출
        selected_indices = np.where(np.array(best_solution) > 0.5)[0]

        if len(selected_indices) == 0:
            raise ValueError("P_PSO가 유효한 조합을 찾지 못했습니다")

        selected = [options[i] for i in selected_indices]

        # 승률 증가 순 정렬
        selected.sort(key=lambda x: x['win_rate_gain'], reverse=True)

        # 최종 자원 검증
        total_cost = {
            'exp_jelly': sum(o['cost']['exp_jelly'] for o in selected),
            'coin': sum(o['cost']['coin'] for o in selected),
            'skill_powder': sum(o['cost']['skill_powder'] for o in selected),
            'soul_stone': sum(o['cost']['soul_stone'] for o in selected),
        }

        # 자원 초과 검증 (초과 시 에러)
        for resource, cost in total_cost.items():
            avail = available_resources.get(resource, float('inf'))
            if cost > avail:
                raise ValueError(f"P_PSO 결과가 {resource} 자원을 초과합니다 (필요: {cost:,}, 보유: {avail:,})")

        logger.info(f"P_PSO 최적화 완료: {len(selected)}개 추천, 총 승률 증가 +{sum(o['win_rate_gain'] for o in selected):.2f}%")
        return selected[:top_n]  # 최대 10개


# ========================================
# 유틸리티 함수
# ========================================

def get_investment_recommendations(user_id: str, goal: str = 'maximize_win_rate') -> Dict:
    """유저에게 투자 추천 제공"""
    try:
        optimizer = InvestmentOptimizer(user_id)
        return optimizer.optimize(goal=goal)
    except Exception as e:
        logger.error(f"Optimization failed for user {user_id}: {e}")
        return {'error': str(e)}


def compare_users(user_ids: List[str]) -> Dict:
    """여러 유저의 추천 결과 비교 (개인화 검증용)"""
    results = {}
    for user_id in user_ids:
        results[user_id] = get_investment_recommendations(user_id)
    return results


# ========================================
# 테스트
# ========================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("투자 최적화 테스트")
    print("=" * 60)

    # 테스트 유저
    test_user = "U000001"

    print(f"\n유저 {test_user}에 대한 투자 최적화...")
    result = get_investment_recommendations(test_user, goal='maximize_win_rate')

    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"\n[추천 결과]")
        print(f"목표: {result['goal']}")
        print(f"최적화 방법: {result['optimization_method']}")
        print(f"총 승률 증가 예상: +{result['total_win_rate_gain']}%")

        print(f"\n[투자 추천 ({len(result['recommendations'])}개)]")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"  {i}. {rec['cookie_name']} ({rec['grade']})")
            print(f"     - 업그레이드: {rec['upgrade_type']} Lv{rec['from_level']}→{rec['to_level']}")
            print(f"     - 예상 승률: {rec['win_rate_before']}% → {rec['win_rate_after']}% (+{rec['win_rate_gain']}%)")
            print(f"     - 효율: {rec['efficiency']}")

        print(f"\n[필요 자원]")
        for resource, cost in result['total_cost'].items():
            if cost > 0:
                usage = result['resource_usage'].get(resource, {})
                affordable = "✓" if usage.get('affordable', True) else "✗"
                print(f"  {resource}: {cost:,} ({usage.get('usage_pct', 0):.1f}% 사용) {affordable}")

    # 개인화 검증: 다른 유저와 비교
    print("\n" + "=" * 60)
    print("개인화 검증 (유저별 추천 비교)")
    print("=" * 60)

    comparison = compare_users(["U000001", "U000002", "U000003"])
    for user_id, res in comparison.items():
        if 'error' in res:
            print(f"\n{user_id}: Error - {res['error']}")
        else:
            top_rec = res['recommendations'][0] if res['recommendations'] else None
            if top_rec:
                print(f"\n{user_id} 1순위 추천: {top_rec['cookie_name']} {top_rec['upgrade_type']}")
            else:
                print(f"\n{user_id}: 추천 없음")
