// components/panels/AnalysisPanel.js
// CookieRun AI Platform - 상세 분석 패널

import { useEffect, useMemo, useState } from 'react';
import toast from 'react-hot-toast';
import { SkeletonCard } from '@/components/Skeleton';
import {
  Users, Globe, Search, Calendar, Filter, TrendingUp,
  Crown, RefreshCw, ChevronDown, User, Gamepad2, Languages,
  AlertTriangle, Brain, Target, Activity, Zap, Shield,
  BarChart3, PieChartIcon, ArrowUpRight, ArrowDownRight,
  Clock, UserMinus, DollarSign, Repeat, Eye
} from 'lucide-react';
import SectionHeader from '@/components/SectionHeader';
import {
  PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, LineChart, Line, RadarChart,
  PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, AreaChart, Area,
  ComposedChart, Scatter
} from 'recharts';

// CookieRun 테마 색상
const COLORS = {
  primary: ['#FF8C42', '#FFD93D', '#4ADE80', '#60A5FA', '#F472B6', '#A78BFA'],
  grades: {
    에인션트: '#8B5CF6',
    레전더리: '#F59E0B',
    슈퍼에픽: '#EC4899',
    에픽: '#8B5CF6',
    레어: '#3B82F6',
    커먼: '#6B7280',
  }
};

// 분석 탭 정의 (확장)
const ANALYSIS_TABS = [
  { key: 'user', label: '유저 분석', icon: User },
  { key: 'segment', label: '세그먼트', icon: Users },
  { key: 'anomaly', label: '이상탐지', icon: AlertTriangle },
  { key: 'prediction', label: '예측 분석', icon: Brain },
  { key: 'cohort', label: '코호트', icon: Target },
  { key: 'trend', label: '트렌드', icon: TrendingUp },
  { key: 'cookie', label: '쿠키 분석', icon: Gamepad2 },
  { key: 'translation', label: '번역 분석', icon: Languages },
];

// 기간 옵션
const DATE_OPTIONS = [
  { value: '7d', label: '최근 7일' },
  { value: '30d', label: '최근 30일' },
  { value: '90d', label: '최근 90일' },
];

// 샘플 유저 데이터 (백엔드 형식: U000001 ~ U001000, 6자리)
const SAMPLE_USERS = [
  { id: 'U000001', name: '용감한플레이어', segment: '하드코어 게이머', level: 85, playtime: 1250, cookies_owned: 62 },
  { id: 'U000025', name: '쿠키마스터', segment: '하드코어 게이머', level: 92, playtime: 1580, cookies_owned: 71 },
  { id: 'U000100', name: '캐주얼유저123', segment: '캐주얼 유저', level: 35, playtime: 180, cookies_owned: 18 },
  { id: 'U000500', name: '신규가입자', segment: '신규 유저', level: 8, playtime: 25, cookies_owned: 5 },
  { id: 'U001000', name: '복귀한쿠키', segment: '복귀 유저', level: 55, playtime: 420, cookies_owned: 38 },
];

// 샘플 유저 상세 데이터
const SAMPLE_USER_DETAIL = {
  id: 'U000001',
  name: '용감한플레이어',
  segment: '하드코어 게이머',
  level: 85,
  playtime: 1250,
  cookies_owned: 62,
  top_cookies: ['순수 바닐라 쿠키', '다크카카오 쿠키', '홀리베리 쿠키'],
  activity: [
    { date: '01/25', playtime: 180, stages: 25 },
    { date: '01/26', playtime: 210, stages: 32 },
    { date: '01/27', playtime: 165, stages: 22 },
    { date: '01/28', playtime: 195, stages: 28 },
    { date: '01/29', playtime: 240, stages: 35 },
    { date: '01/30', playtime: 185, stages: 26 },
    { date: '01/31', playtime: 200, stages: 30 },
  ],
  stats: {
    전투력: 85,
    수집률: 73,
    활동성: 92,
    과금: 45,
    소셜: 68,
  }
};

// 샘플 세그먼트 데이터
const SAMPLE_SEGMENTS = {
  '하드코어 게이머': { count: 350, avg_level: 78, avg_playtime: 1200, avg_cookies: 58, retention: 92 },
  '캐주얼 유저': { count: 580, avg_level: 42, avg_playtime: 280, avg_cookies: 25, retention: 65 },
  '신규 유저': { count: 280, avg_level: 12, avg_playtime: 45, avg_cookies: 8, retention: 48 },
  '복귀 유저': { count: 190, avg_level: 52, avg_playtime: 380, avg_cookies: 32, retention: 72 },
  '휴면 유저': { count: 100, avg_level: 38, avg_playtime: 15, avg_cookies: 22, retention: 12 },
};

// 샘플 쿠키 분석 데이터
const SAMPLE_COOKIE_STATS = [
  { name: '순수 바닐라 쿠키', grade: '에인션트', usage: 89, power: 95, popularity: 92 },
  { name: '다크카카오 쿠키', grade: '에인션트', usage: 85, power: 93, popularity: 88 },
  { name: '홀리베리 쿠키', grade: '에인션트', usage: 82, power: 91, popularity: 85 },
  { name: '프로즌 퀸 쿠키', grade: '레전더리', usage: 78, power: 88, popularity: 80 },
  { name: '씨솔트 쿠키', grade: '에픽', usage: 75, power: 82, popularity: 78 },
  { name: '블랙펄 쿠키', grade: '레전더리', usage: 72, power: 86, popularity: 76 },
];

// 샘플 번역 분석 데이터
const SAMPLE_TRANSLATION_STATS = {
  languages: [
    { lang: '영어', count: 750, quality: 94.2, pending: 12 },
    { lang: '일본어', count: 620, quality: 92.8, pending: 25 },
    { lang: '중국어', count: 580, quality: 91.5, pending: 18 },
    { lang: '태국어', count: 320, quality: 89.3, pending: 35 },
    { lang: '인도네시아어', count: 280, quality: 88.7, pending: 42 },
  ],
  recent: [
    { text: '용감한 쿠키가 오븐에서 탈출했어요!', lang: '영어', quality: 96 },
    { text: '소울잼의 힘이 깨어납니다', lang: '일본어', quality: 94 },
    { text: '다크엔챈트리스 쿠키가 나타났다!', lang: '중국어', quality: 92 },
  ]
};

// 이상탐지 샘플 데이터
const SAMPLE_ANOMALY_DATA = {
  summary: {
    total_users: 1000,
    anomaly_count: 23,
    anomaly_rate: 2.3,
    high_risk: 5,
    medium_risk: 12,
    low_risk: 6,
  },
  by_type: [
    { type: '비정상 결제 패턴', count: 8, severity: 'high' },
    { type: '봇 의심 행동', count: 6, severity: 'high' },
    { type: '계정 공유 의심', count: 5, severity: 'medium' },
    { type: '비정상 플레이 시간', count: 4, severity: 'low' },
  ],
  recent_alerts: [
    { id: 'U000523', type: '비정상 결제', time: '10분 전', severity: 'high', detail: '24시간 내 15회 결제 시도' },
    { id: 'U000891', type: '봇 의심', time: '25분 전', severity: 'high', detail: '패턴화된 반복 행동 감지' },
    { id: 'U000234', type: '계정 공유', time: '1시간 전', severity: 'medium', detail: '다중 기기/IP 동시 접속' },
    { id: 'U000456', type: '비정상 플레이', time: '2시간 전', severity: 'low', detail: '48시간 연속 플레이' },
  ],
  trend: [
    { date: '01/25', count: 3 },
    { date: '01/26', count: 5 },
    { date: '01/27', count: 2 },
    { date: '01/28', count: 4 },
    { date: '01/29', count: 3 },
    { date: '01/30', count: 6 },
    { date: '01/31', count: 5 },
  ]
};

// 예측 분석 샘플 데이터
const SAMPLE_PREDICTION_DATA = {
  churn: {
    high_risk_count: 85,
    medium_risk_count: 142,
    low_risk_count: 773,
    predicted_churn_rate: 8.5,
    model_accuracy: 87.3,
    top_factors: [
      { factor: '7일간 미접속', importance: 0.35 },
      { factor: '플레이타임 급감', importance: 0.25 },
      { factor: '최근 과금 없음', importance: 0.20 },
      { factor: '길드 활동 감소', importance: 0.12 },
      { factor: '스테이지 진행 정체', importance: 0.08 },
    ],
    high_risk_users: [
      { id: 'U000342', name: '쿠키헌터', probability: 92, last_active: '7일 전', segment: '하드코어 게이머' },
      { id: 'U000567', name: '달콤한세상', probability: 88, last_active: '5일 전', segment: '캐주얼 유저' },
      { id: 'U000123', name: '별빛쿠키', probability: 85, last_active: '4일 전', segment: '하드코어 게이머' },
    ]
  },
  revenue: {
    predicted_monthly: 15420000,
    predicted_arpu: 15420,
    predicted_arppu: 45800,
    whale_count: 12,
    dolphin_count: 48,
    minnow_count: 285,
    growth_rate: 12.5,
    confidence: 82.1,
  },
  engagement: {
    predicted_dau: 650,
    predicted_mau: 920,
    stickiness: 70.6,
    avg_session: 28,
    sessions_per_day: 3.2,
  }
};

// 코호트 분석 샘플 데이터
const SAMPLE_COHORT_DATA = {
  retention: [
    { cohort: '2025-01 W1', week0: 100, week1: 72, week2: 58, week3: 48, week4: 42 },
    { cohort: '2025-01 W2', week0: 100, week1: 75, week2: 62, week3: 51, week4: 45 },
    { cohort: '2025-01 W3', week0: 100, week1: 68, week2: 55, week3: 46, week4: null },
    { cohort: '2025-01 W4', week0: 100, week1: 70, week2: 56, week3: null, week4: null },
  ],
  ltv_by_cohort: [
    { cohort: '2024-10', ltv: 42500, users: 180 },
    { cohort: '2024-11', ltv: 38200, users: 210 },
    { cohort: '2024-12', ltv: 35800, users: 195 },
    { cohort: '2025-01', ltv: 28500, users: 225 },
  ],
  conversion: [
    { cohort: '2024-12 W1', registered: 120, activated: 95, engaged: 68, converted: 22, retained: 18 },
    { cohort: '2024-12 W2', registered: 135, activated: 108, engaged: 75, converted: 28, retained: 24 },
    { cohort: '2024-12 W3', registered: 98, activated: 82, engaged: 55, converted: 18, retained: 15 },
    { cohort: '2025-01 W1', registered: 142, activated: 118, engaged: 82, converted: 32, retained: 28 },
  ]
};

// 트렌드 분석 샘플 데이터
const SAMPLE_TREND_DATA = {
  kpis: [
    { name: 'DAU', current: 650, previous: 580, trend: 'up', change: 12.1 },
    { name: 'ARPU', current: 15420, previous: 14200, trend: 'up', change: 8.6 },
    { name: '신규가입', current: 45, previous: 52, trend: 'down', change: -13.5 },
    { name: '이탈률', current: 3.2, previous: 4.1, trend: 'up', change: -22.0 },
    { name: '세션시간', current: 28, previous: 25, trend: 'up', change: 12.0 },
    { name: '결제전환', current: 4.8, previous: 4.2, trend: 'up', change: 14.3 },
  ],
  daily_metrics: [
    { date: '01/25', dau: 580, revenue: 8500000, sessions: 1850, new_users: 42 },
    { date: '01/26', dau: 612, revenue: 9200000, sessions: 1920, new_users: 48 },
    { date: '01/27', dau: 598, revenue: 8800000, sessions: 1880, new_users: 38 },
    { date: '01/28', dau: 625, revenue: 9500000, sessions: 1950, new_users: 45 },
    { date: '01/29', dau: 640, revenue: 10200000, sessions: 2010, new_users: 52 },
    { date: '01/30', dau: 658, revenue: 11800000, sessions: 2080, new_users: 55 },
    { date: '01/31', dau: 650, revenue: 11200000, sessions: 2050, new_users: 45 },
  ],
  correlation: [
    { var1: 'DAU', var2: '매출', correlation: 0.85 },
    { var1: 'DAU', var2: '세션시간', correlation: 0.72 },
    { var1: '매출', var2: '과금유저', correlation: 0.92 },
    { var1: '리텐션', var2: 'LTV', correlation: 0.88 },
    { var1: '이벤트참여', var2: '매출', correlation: 0.65 },
  ],
  forecast: [
    { date: '02/01', predicted_dau: 665, lower: 640, upper: 690, predicted_revenue: 11500000 },
    { date: '02/02', predicted_dau: 672, lower: 645, upper: 699, predicted_revenue: 11800000 },
    { date: '02/03', predicted_dau: 678, lower: 648, upper: 708, predicted_revenue: 12100000 },
    { date: '02/04', predicted_dau: 685, lower: 652, upper: 718, predicted_revenue: 12400000 },
    { date: '02/05', predicted_dau: 690, lower: 655, upper: 725, predicted_revenue: 12600000 },
  ]
};

// 커스텀 툴팁
const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-xl border-2 border-cookie-orange/20 bg-white/95 px-3 py-2 shadow-lg backdrop-blur">
      <p className="text-xs font-bold text-cookie-brown">{label}</p>
      {payload.map((entry, idx) => (
        <p key={idx} className="text-sm font-semibold" style={{ color: entry.color || entry.fill }}>
          {entry.name}: {typeof entry.value === 'number' ? entry.value.toLocaleString() : entry.value}
        </p>
      ))}
    </div>
  );
};

export default function AnalysisPanel({ auth, apiCall }) {
  const [activeTab, setActiveTab] = useState('user');
  const [dateRange, setDateRange] = useState('7d');
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedUser, setSelectedUser] = useState(null);
  const [selectedSegment, setSelectedSegment] = useState('전체');
  const [showDateDropdown, setShowDateDropdown] = useState(false);

  // API 데이터 상태 (초기값 null - API 실패 시 데이터 없음 표시)
  const [summaryData, setSummaryData] = useState(null);
  const [segmentsData, setSegmentsData] = useState(null);
  const [cookiesData, setCookiesData] = useState(null);
  const [translationData, setTranslationData] = useState(null);
  const [dataLoaded, setDataLoaded] = useState(false);

  // 빠른 선택용 샘플 유저 ID (UI용)
  const quickSelectUsers = ['U000001', 'U000025', 'U000100', 'U000500'];

  // 새로운 분석 데이터 상태
  const [anomalyData, setAnomalyData] = useState(null);
  const [predictionData, setPredictionData] = useState(null);
  const [cohortData, setCohortData] = useState(null);
  const [trendData, setTrendData] = useState(null);
  const [predictionTab, setPredictionTab] = useState('churn'); // churn, revenue, engagement
  const [cohortTab, setCohortTab] = useState('retention'); // retention, ltv, conversion

  // API 데이터 로드
  useEffect(() => {
    async function fetchData() {
      setLoading(true);

      // 기간을 일수로 변환
      const daysMap = { '7d': 7, '30d': 30, '90d': 90 };
      const days = daysMap[dateRange] || 7;

      try {
        // 통계 요약 API 호출
        const summaryRes = await apiCall({
          endpoint: `/api/stats/summary?days=${days}`,
          auth,
          timeoutMs: 10000,
        });

        if (summaryRes?.status === 'SUCCESS') {
          setSummaryData(summaryRes);

          // 세그먼트 데이터 변환 - API의 segment_metrics 사용
          if (summaryRes.user_segments) {
            const segments = {};
            const metrics = summaryRes.segment_metrics || {};
            Object.entries(summaryRes.user_segments).forEach(([name, count]) => {
              const m = metrics[name] || {};
              segments[name] = {
                count,
                avg_level: Math.floor(30 + (m.avg_stages || 0) / 10),
                avg_playtime: m.avg_playtime || Math.floor(100 + Math.random() * 200),
                avg_cookies: Math.floor(10 + (m.avg_gacha || 0) / 5),
                retention: Math.floor(40 + Math.min(50, (m.avg_pvp || 0) / 2)),
              };
            });
            if (Object.keys(segments).length > 0) {
              setSegmentsData(segments);
            }
          }

          // 쿠키 데이터가 있으면 변환
          if (summaryRes.grade_stats) {
            // 쿠키 API 별도 호출
            try {
              const cookiesRes = await apiCall({
                endpoint: '/api/cookies',
                auth,
                timeoutMs: 10000,
              });
              if (cookiesRes?.status === 'SUCCESS' && cookiesRes.cookies) {
                const transformed = cookiesRes.cookies.slice(0, 10).map(c => ({
                  name: c.name,
                  grade: c.grade,
                  // 백엔드에서 제공하는 실제 통계 사용, 없으면 폴백
                  usage: c.usage ?? Math.floor(50 + Math.random() * 40),
                  power: c.power ?? Math.floor(70 + Math.random() * 25),
                  popularity: c.popularity ?? Math.floor(60 + Math.random() * 35),
                }));
                if (transformed.length > 0) {
                  setCookiesData(transformed);
                }
              }
            } catch (e) {
              console.log('쿠키 API 실패');
            }
          }

          // 번역 데이터 변환 - 상세 통계가 있으면 사용
          if (summaryRes.translation_stats_detail) {
            // 백엔드에서 제공하는 상세 통계 사용
            const langs = summaryRes.translation_stats_detail.map(stat => ({
              lang: stat.lang_name,
              count: stat.total_count,
              quality: stat.avg_quality?.toFixed(1) ?? '90.0',
              pending: stat.pending_count ?? 0,
            }));
            if (langs.length > 0) {
              setTranslationData({ languages: langs, recent: [] });
            }
          } else if (summaryRes.translation_langs) {
            // 폴백: 기본 데이터에 랜덤 값 추가
            const langs = Object.entries(summaryRes.translation_langs).map(([lang, count]) => ({
              lang,
              count,
              quality: (85 + Math.random() * 10).toFixed(1),
              pending: Math.floor(Math.random() * 30),
            }));
            if (langs.length > 0) {
              setTranslationData({ languages: langs, recent: [] });
            }
          }
        }

        // 이상탐지 API 호출
        try {
          const anomalyRes = await apiCall({
            endpoint: `/api/analysis/anomaly?days=${days}`,
            auth,
            timeoutMs: 10000,
          });
          if (anomalyRes?.status === 'SUCCESS') {
            setAnomalyData({
              summary: anomalyRes.summary || {},
              by_type: anomalyRes.by_type || [],
              recent_alerts: anomalyRes.recent_alerts || [],
              trend: anomalyRes.trend || [],
            });
          }
        } catch (e) {
          console.log('이상탐지 API 실패');
        }

        // 예측 분석 API 호출
        try {
          const churnRes = await apiCall({
            endpoint: `/api/analysis/prediction/churn?days=${days}`,
            auth,
            timeoutMs: 10000,
          });
          if (churnRes?.status === 'SUCCESS' && churnRes.churn) {
            setPredictionData({
              churn: churnRes.churn,
              revenue: churnRes.revenue || {},
              engagement: churnRes.engagement || {},
            });
          }
        } catch (e) {
          console.log('예측 API 실패');
        }

        // 코호트 API 호출
        try {
          const cohortRes = await apiCall({
            endpoint: `/api/analysis/cohort/retention?days=${days}`,
            auth,
            timeoutMs: 10000,
          });
          if (cohortRes?.status === 'SUCCESS' && cohortRes.retention) {
            setCohortData({
              retention: cohortRes.retention,
              ltv_by_cohort: cohortRes.ltv_by_cohort || [],
              conversion: cohortRes.conversion || [],
            });
          }
        } catch (e) {
          console.log('코호트 API 실패');
        }

        // 트렌드 KPI API 호출
        try {
          const trendRes = await apiCall({
            endpoint: `/api/analysis/trend/kpis?days=${days}`,
            auth,
            timeoutMs: 10000,
          });
          if (trendRes?.status === 'SUCCESS' && trendRes.kpis) {
            setTrendData({
              kpis: trendRes.kpis,
              daily_metrics: trendRes.daily_metrics || [],
              correlation: trendRes.correlation || [],
              forecast: trendRes.forecast || [],
            });
          }
        } catch (e) {
          console.log('트렌드 API 실패');
        }

      } catch (e) {
        console.log('API 호출 실패');
      }
      setDataLoaded(true);
      setLoading(false);
    }

    if (auth) {
      fetchData();
    }
  }, [auth, apiCall, dateRange]);

  // 유저 검색
  const handleUserSearch = async () => {
    if (!searchQuery.trim()) {
      toast.error('유저 ID를 입력하세요');
      return;
    }
    setLoading(true);

    try {
      // API 호출 시도
      const res = await apiCall({
        endpoint: `/api/users/search?q=${encodeURIComponent(searchQuery)}`,
        auth,
        timeoutMs: 10000,
      });

      if (res?.status === 'SUCCESS' && res.user) {
        setSelectedUser({
          id: res.user.id,
          name: res.user.name || res.user.id,
          segment: res.user.segment || '알 수 없음',
          level: res.user.level || 0,
          playtime: res.user.playtime || 0,
          cookies_owned: res.user.cookies_owned || 0,
          top_cookies: res.user.top_cookies || [],
          stats: res.user.stats || {},
          activity: res.user.activity || [],
        });
        toast.success(`${res.user.name || res.user.id} 유저 정보를 불러왔습니다`);
      } else {
        toast.error('유저를 찾을 수 없습니다');
        setSelectedUser(null);
      }
    } catch (e) {
      console.log('유저 검색 API 실패');
      toast.error('유저 검색에 실패했습니다. 백엔드 연결을 확인하세요.');
      setSelectedUser(null);
    }
    setLoading(false);
  };

  // 유저 레이더 차트 데이터
  const userRadarData = useMemo(() => {
    if (!selectedUser?.stats) return [];
    return Object.entries(selectedUser.stats).map(([key, value]) => ({
      subject: key,
      value,
      fullMark: 100,
    }));
  }, [selectedUser]);

  // 세그먼트 비교 차트 데이터
  const segmentCompareData = useMemo(() => {
    if (!segmentsData) return [];
    return Object.entries(segmentsData).map(([name, data]) => ({
      name: name.replace(' ', '\n'),
      유저수: data.count,
      평균레벨: data.avg_level,
      리텐션: data.retention,
    }));
  }, [segmentsData]);

  // 쿠키 사용률 차트 데이터
  const cookieUsageData = useMemo(() => {
    if (!cookiesData) return [];
    return cookiesData.map(cookie => ({
      name: cookie.name.replace(' 쿠키', ''),
      사용률: cookie.usage,
      전투력: cookie.power,
      인기도: cookie.popularity,
      fill: COLORS.grades[cookie.grade] || COLORS.primary[0],
    }));
  }, [cookiesData]);

  return (
    <div>
      <SectionHeader
        title="상세 분석"
        subtitle="유저 · 세그먼트 · 쿠키 · 번역 데이터 심층 분석"
        right={
          <div className="flex items-center gap-2">
            {/* 데이터 소스 배지 */}
            {dataLoaded && (
              <span className={`rounded-full border-2 px-2 py-1 text-[10px] font-black ${
                summaryData
                  ? 'border-green-400/50 bg-green-50 text-green-700'
                  : 'border-red-400/50 bg-red-50 text-red-700'
              }`}>
                {summaryData ? 'LIVE' : 'NO DATA'}
              </span>
            )}
            {/* 기간 선택 */}
            <div className="relative">
              <button
                onClick={() => setShowDateDropdown(!showDateDropdown)}
                className="flex items-center gap-1.5 rounded-full border-2 border-cookie-orange/20 bg-white/80 px-3 py-1.5 text-xs font-bold text-cookie-brown hover:bg-cookie-beige transition"
              >
                <Calendar size={12} />
                {DATE_OPTIONS.find(d => d.value === dateRange)?.label}
                <ChevronDown size={12} />
              </button>
              {showDateDropdown && (
                <div className="absolute right-0 top-full mt-1 z-10 rounded-xl border-2 border-cookie-orange/20 bg-white shadow-lg overflow-hidden">
                  {DATE_OPTIONS.map(opt => (
                    <button
                      key={opt.value}
                      onClick={() => { setDateRange(opt.value); setShowDateDropdown(false); }}
                      className={`block w-full px-4 py-2 text-left text-xs font-semibold hover:bg-cookie-beige transition ${
                        dateRange === opt.value ? 'bg-cookie-yellow/30 text-cookie-brown' : 'text-cookie-brown/70'
                      }`}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        }
      />

      {/* 분석 유형 탭 */}
      <div className="flex gap-2 mb-6 overflow-x-auto pb-2">
        {ANALYSIS_TABS.map(tab => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-xl font-bold text-sm whitespace-nowrap transition-all ${
                activeTab === tab.key
                  ? 'bg-gradient-to-r from-cookie-yellow to-cookie-orange text-white shadow-md'
                  : 'bg-white/80 border-2 border-cookie-orange/20 text-cookie-brown hover:bg-cookie-beige'
              }`}
            >
              <Icon size={16} />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* 유저 분석 */}
      {activeTab === 'user' && (
        <div className="space-y-6">
          {/* 유저 검색 */}
          <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
            <div className="flex items-center gap-2 mb-4">
              <Search size={18} className="text-cookie-orange" />
              <span className="text-sm font-black text-cookie-brown">유저 검색</span>
            </div>
            <div className="flex gap-3">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleUserSearch()}
                placeholder="유저 ID 또는 닉네임 입력 (예: U000001)"
                className="flex-1 px-4 py-2.5 rounded-xl border-2 border-cookie-orange/20 bg-white text-sm text-cookie-brown placeholder:text-cookie-brown/40 outline-none focus:border-cookie-orange transition"
              />
              <button
                onClick={handleUserSearch}
                disabled={loading}
                className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-cookie-yellow to-cookie-orange text-white font-bold text-sm shadow-md hover:shadow-lg transition disabled:opacity-50"
              >
                {loading ? '검색 중...' : '검색'}
              </button>
            </div>
            {/* 빠른 선택 */}
            <div className="mt-3 flex flex-wrap gap-2">
              <span className="text-xs text-cookie-brown/60">빠른 선택:</span>
              {quickSelectUsers.map(userId => (
                <button
                  key={userId}
                  onClick={() => { setSearchQuery(userId); }}
                  className="px-2 py-1 rounded-lg bg-cookie-beige text-xs font-semibold text-cookie-brown hover:bg-cookie-yellow/30 transition"
                >
                  {userId}
                </button>
              ))}
            </div>
          </div>

          {/* 유저 상세 정보 */}
          {selectedUser && (
            <>
              {/* 기본 정보 */}
              <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <h3 className="text-lg font-black text-cookie-brown">{selectedUser.name}</h3>
                    <p className="text-sm text-cookie-brown/60">{selectedUser.id} · {selectedUser.segment}</p>
                  </div>
                  <span className="px-3 py-1 rounded-full bg-cookie-yellow/30 text-xs font-bold text-cookie-brown">
                    Lv.{selectedUser.level}
                  </span>
                </div>
                <div className="grid grid-cols-3 gap-4">
                  <div className="text-center p-3 rounded-2xl bg-cookie-beige/50">
                    <div className="text-2xl font-black text-cookie-brown">{selectedUser.playtime}</div>
                    <div className="text-xs text-cookie-brown/60">총 플레이 시간(분)</div>
                  </div>
                  <div className="text-center p-3 rounded-2xl bg-cookie-beige/50">
                    <div className="text-2xl font-black text-cookie-brown">{selectedUser.cookies_owned}</div>
                    <div className="text-xs text-cookie-brown/60">보유 쿠키</div>
                  </div>
                  <div className="text-center p-3 rounded-2xl bg-cookie-beige/50">
                    <div className="text-2xl font-black text-cookie-brown">{selectedUser.top_cookies?.length || 0}</div>
                    <div className="text-xs text-cookie-brown/60">주력 쿠키</div>
                  </div>
                </div>
              </div>

              {/* 차트 그리드 */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* 활동 트렌드 */}
                <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
                  <div className="mb-4 text-sm font-black text-cookie-brown">일별 활동 트렌드</div>
                  <ResponsiveContainer width="100%" height={250}>
                    <AreaChart data={selectedUser.activity}>
                      <defs>
                        <linearGradient id="colorPlaytime" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#FFD93D" stopOpacity={0.4}/>
                          <stop offset="95%" stopColor="#FFD93D" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#FFD93D40" />
                      <XAxis dataKey="date" tick={{ fill: '#5C4A3D', fontSize: 11 }} />
                      <YAxis tick={{ fill: '#5C4A3D', fontSize: 11 }} />
                      <Tooltip content={<CustomTooltip />} />
                      <Legend />
                      <Area type="monotone" dataKey="playtime" name="플레이시간(분)" stroke="#FFD93D" fill="url(#colorPlaytime)" />
                      <Line type="monotone" dataKey="stages" name="클리어 스테이지" stroke="#4ADE80" strokeWidth={2} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>

                {/* 유저 스탯 레이더 */}
                <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
                  <div className="mb-4 text-sm font-black text-cookie-brown">유저 특성 분석</div>
                  <ResponsiveContainer width="100%" height={250}>
                    <RadarChart data={userRadarData}>
                      <PolarGrid stroke="#FFD93D60" />
                      <PolarAngleAxis dataKey="subject" tick={{ fill: '#5C4A3D', fontSize: 11 }} />
                      <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: '#5C4A3D', fontSize: 10 }} />
                      <Radar name="스탯" dataKey="value" stroke="#FF8C42" fill="#FF8C42" fillOpacity={0.5} />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </>
          )}

          {!selectedUser && !loading && (
            <div className="text-center py-12 text-cookie-brown/50">
              <User size={48} className="mx-auto mb-3 opacity-30" />
              <p className="text-sm">유저 ID를 검색하여 상세 분석을 확인하세요</p>
            </div>
          )}
        </div>
      )}

      {/* 세그먼트 분석 */}
      {activeTab === 'segment' && (
        <div className="space-y-6">
          {!segmentsData ? (
            <div className="text-center py-16 rounded-3xl border-2 border-cookie-orange/20 bg-white/80">
              <Users size={48} className="mx-auto mb-3 text-cookie-brown/30" />
              <p className="text-sm font-semibold text-cookie-brown/50">세그먼트 데이터를 불러올 수 없습니다</p>
              <p className="text-xs text-cookie-brown/40 mt-1">백엔드 API 연결을 확인하세요</p>
            </div>
          ) : (
          <>
          {/* 세그먼트 비교 */}
          <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
            <div className="flex items-center gap-2 mb-4">
              <Users size={18} className="text-cookie-orange" />
              <span className="text-sm font-black text-cookie-brown">세그먼트 비교 분석</span>
            </div>
            <ResponsiveContainer width="100%" height={350}>
              <BarChart data={segmentCompareData} margin={{ top: 20, right: 30, left: 0, bottom: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#FFD93D40" />
                <XAxis
                  dataKey="name"
                  tick={{ fill: '#5C4A3D', fontSize: 10 }}
                  interval={0}
                  angle={-15}
                  textAnchor="end"
                />
                <YAxis tick={{ fill: '#5C4A3D', fontSize: 11 }} />
                <Tooltip content={<CustomTooltip />} />
                <Legend />
                <Bar dataKey="유저수" fill="#FF8C42" radius={[4, 4, 0, 0]} />
                <Bar dataKey="평균레벨" fill="#FFD93D" radius={[4, 4, 0, 0]} />
                <Bar dataKey="리텐션" fill="#4ADE80" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* 세그먼트 상세 테이블 */}
          <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
            <div className="mb-4 text-sm font-black text-cookie-brown">세그먼트별 상세 지표</div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b-2 border-cookie-orange/10">
                    <th className="text-left py-3 px-2 font-bold text-cookie-brown">세그먼트</th>
                    <th className="text-right py-3 px-2 font-bold text-cookie-brown">유저 수</th>
                    <th className="text-right py-3 px-2 font-bold text-cookie-brown">평균 레벨</th>
                    <th className="text-right py-3 px-2 font-bold text-cookie-brown">평균 플레이타임</th>
                    <th className="text-right py-3 px-2 font-bold text-cookie-brown">평균 쿠키 보유</th>
                    <th className="text-right py-3 px-2 font-bold text-cookie-brown">리텐션</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(segmentsData).map(([name, data]) => (
                    <tr key={name} className="border-b border-cookie-orange/5 hover:bg-cookie-beige/30 transition">
                      <td className="py-3 px-2 font-semibold text-cookie-brown">{name}</td>
                      <td className="py-3 px-2 text-right text-cookie-brown/80">{data.count.toLocaleString()}명</td>
                      <td className="py-3 px-2 text-right text-cookie-brown/80">Lv.{data.avg_level}</td>
                      <td className="py-3 px-2 text-right text-cookie-brown/80">{data.avg_playtime}분</td>
                      <td className="py-3 px-2 text-right text-cookie-brown/80">{data.avg_cookies}개</td>
                      <td className="py-3 px-2 text-right">
                        <span className={`px-2 py-0.5 rounded-full text-xs font-bold ${
                          data.retention >= 70 ? 'bg-green-100 text-green-700' :
                          data.retention >= 40 ? 'bg-yellow-100 text-yellow-700' :
                          'bg-red-100 text-red-700'
                        }`}>
                          {data.retention}%
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          </>
          )}
        </div>
      )}

      {/* 쿠키 분석 */}
      {activeTab === 'cookie' && (
        <div className="space-y-6">
          {!cookiesData ? (
            <div className="text-center py-16 rounded-3xl border-2 border-cookie-orange/20 bg-white/80">
              <Gamepad2 size={48} className="mx-auto mb-3 text-cookie-brown/30" />
              <p className="text-sm font-semibold text-cookie-brown/50">쿠키 데이터를 불러올 수 없습니다</p>
              <p className="text-xs text-cookie-brown/40 mt-1">백엔드 API 연결을 확인하세요</p>
            </div>
          ) : (
          <>
          {/* 쿠키 사용률 차트 */}
          <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
            <div className="flex items-center gap-2 mb-4">
              <span className="text-lg">🍪</span>
              <span className="text-sm font-black text-cookie-brown">인기 쿠키 분석</span>
            </div>
            <ResponsiveContainer width="100%" height={350}>
              <BarChart data={cookieUsageData} layout="vertical" margin={{ top: 10, right: 30, left: 100, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#FFD93D40" horizontal={true} vertical={false} />
                <XAxis type="number" domain={[0, 100]} tick={{ fill: '#5C4A3D', fontSize: 11 }} />
                <YAxis type="category" dataKey="name" tick={{ fill: '#5C4A3D', fontSize: 11 }} width={90} />
                <Tooltip content={<CustomTooltip />} />
                <Legend />
                <Bar dataKey="사용률" fill="#FF8C42" radius={[0, 4, 4, 0]} barSize={16} />
                <Bar dataKey="전투력" fill="#60A5FA" radius={[0, 4, 4, 0]} barSize={16} />
                <Bar dataKey="인기도" fill="#4ADE80" radius={[0, 4, 4, 0]} barSize={16} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* 쿠키 상세 리스트 */}
          <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
            <div className="mb-4 text-sm font-black text-cookie-brown">쿠키별 상세 통계</div>
            <div className="space-y-3">
              {cookiesData.map((cookie, idx) => (
                <div key={cookie.name} className="flex items-center gap-4 p-3 rounded-2xl bg-cookie-beige/30 hover:bg-cookie-beige/50 transition">
                  <span
                    className="w-8 h-8 rounded-full flex items-center justify-center text-white font-bold text-sm"
                    style={{ backgroundColor: COLORS.grades[cookie.grade] }}
                  >
                    {idx + 1}
                  </span>
                  <div className="flex-1">
                    <div className="font-bold text-cookie-brown">{cookie.name}</div>
                    <div className="text-xs text-cookie-brown/60">{cookie.grade}</div>
                  </div>
                  <div className="flex gap-4 text-sm">
                    <div className="text-center">
                      <div className="font-bold text-cookie-brown">{cookie.usage}%</div>
                      <div className="text-[10px] text-cookie-brown/50">사용률</div>
                    </div>
                    <div className="text-center">
                      <div className="font-bold text-cookie-brown">{cookie.power}</div>
                      <div className="text-[10px] text-cookie-brown/50">전투력</div>
                    </div>
                    <div className="text-center">
                      <div className="font-bold text-cookie-brown">{cookie.popularity}%</div>
                      <div className="text-[10px] text-cookie-brown/50">인기도</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
          </>
          )}
        </div>
      )}

      {/* 번역 분석 */}
      {activeTab === 'translation' && (
        <div className="space-y-6">
          {!translationData ? (
            <div className="text-center py-16 rounded-3xl border-2 border-cookie-orange/20 bg-white/80">
              <Languages size={48} className="mx-auto mb-3 text-cookie-brown/30" />
              <p className="text-sm font-semibold text-cookie-brown/50">번역 데이터를 불러올 수 없습니다</p>
              <p className="text-xs text-cookie-brown/40 mt-1">백엔드 API 연결을 확인하세요</p>
            </div>
          ) : (
          <>
          {/* 언어별 번역 현황 */}
          <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
            <div className="flex items-center gap-2 mb-4">
              <Globe size={18} className="text-cookie-orange" />
              <span className="text-sm font-black text-cookie-brown">언어별 번역 현황</span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b-2 border-cookie-orange/10">
                    <th className="text-left py-3 px-2 font-bold text-cookie-brown">언어</th>
                    <th className="text-right py-3 px-2 font-bold text-cookie-brown">번역 수</th>
                    <th className="text-right py-3 px-2 font-bold text-cookie-brown">평균 품질</th>
                    <th className="text-right py-3 px-2 font-bold text-cookie-brown">대기중</th>
                    <th className="text-left py-3 px-2 font-bold text-cookie-brown">품질 바</th>
                  </tr>
                </thead>
                <tbody>
                  {translationData.languages.map(lang => (
                    <tr key={lang.lang} className="border-b border-cookie-orange/5 hover:bg-cookie-beige/30 transition">
                      <td className="py-3 px-2 font-semibold text-cookie-brown">{lang.lang}</td>
                      <td className="py-3 px-2 text-right text-cookie-brown/80">{lang.count.toLocaleString()}</td>
                      <td className="py-3 px-2 text-right">
                        <span className={`font-bold ${parseFloat(lang.quality) >= 90 ? 'text-green-600' : 'text-yellow-600'}`}>
                          {lang.quality}%
                        </span>
                      </td>
                      <td className="py-3 px-2 text-right text-cookie-brown/80">{lang.pending}</td>
                      <td className="py-3 px-2 w-40">
                        <div className="h-2 bg-cookie-beige rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full bg-gradient-to-r from-cookie-yellow to-cookie-orange"
                            style={{ width: `${lang.quality}%` }}
                          />
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* 최근 번역 샘플 */}
          {translationData.recent && translationData.recent.length > 0 && (
          <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
            <div className="mb-4 text-sm font-black text-cookie-brown">최근 번역 샘플</div>
            <div className="space-y-3">
              {translationData.recent.map((item, idx) => (
                <div key={idx} className="p-4 rounded-2xl bg-cookie-beige/30">
                  <div className="flex items-center justify-between mb-2">
                    <span className="px-2 py-0.5 rounded-full bg-cookie-orange/20 text-xs font-bold text-cookie-brown">
                      {item.lang}
                    </span>
                    <span className={`text-sm font-bold ${item.quality >= 95 ? 'text-green-600' : 'text-yellow-600'}`}>
                      품질 {item.quality}%
                    </span>
                  </div>
                  <p className="text-sm text-cookie-brown">&ldquo;{item.text}&rdquo;</p>
                </div>
              ))}
            </div>
          </div>
          )}
          </>
          )}
        </div>
      )}

      {/* 이상탐지 분석 */}
      {activeTab === 'anomaly' && (
        <div className="space-y-6">
          {!anomalyData ? (
            <div className="text-center py-16 rounded-3xl border-2 border-cookie-orange/20 bg-white/80">
              <AlertTriangle size={48} className="mx-auto mb-3 text-cookie-brown/30" />
              <p className="text-sm font-semibold text-cookie-brown/50">이상탐지 데이터를 불러올 수 없습니다</p>
              <p className="text-xs text-cookie-brown/40 mt-1">백엔드 API 연결을 확인하세요</p>
            </div>
          ) : (
          <>
          {/* 이상탐지 요약 카드 */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="rounded-2xl border-2 border-red-200 bg-red-50 p-4">
              <div className="flex items-center gap-2 mb-2">
                <AlertTriangle size={18} className="text-red-500" />
                <span className="text-xs font-bold text-red-700">고위험</span>
              </div>
              <div className="text-2xl font-black text-red-600">{anomalyData.summary?.high_risk || 0}</div>
              <div className="text-xs text-red-600/70">즉시 조치 필요</div>
            </div>
            <div className="rounded-2xl border-2 border-orange-200 bg-orange-50 p-4">
              <div className="flex items-center gap-2 mb-2">
                <Shield size={18} className="text-orange-500" />
                <span className="text-xs font-bold text-orange-700">중위험</span>
              </div>
              <div className="text-2xl font-black text-orange-600">{anomalyData.summary?.medium_risk || 0}</div>
              <div className="text-xs text-orange-600/70">모니터링 필요</div>
            </div>
            <div className="rounded-2xl border-2 border-yellow-200 bg-yellow-50 p-4">
              <div className="flex items-center gap-2 mb-2">
                <Eye size={18} className="text-yellow-600" />
                <span className="text-xs font-bold text-yellow-700">저위험</span>
              </div>
              <div className="text-2xl font-black text-yellow-600">{anomalyData.summary?.low_risk || 0}</div>
              <div className="text-xs text-yellow-600/70">관찰 대상</div>
            </div>
            <div className="rounded-2xl border-2 border-cookie-orange/20 bg-white/80 p-4">
              <div className="flex items-center gap-2 mb-2">
                <Activity size={18} className="text-cookie-orange" />
                <span className="text-xs font-bold text-cookie-brown">탐지율</span>
              </div>
              <div className="text-2xl font-black text-cookie-brown">{anomalyData.summary?.anomaly_rate || 0}%</div>
              <div className="text-xs text-cookie-brown/60">{anomalyData.summary?.anomaly_count || 0}/{anomalyData.summary?.total_users || 0}</div>
            </div>
          </div>

          {/* 이상유형별 분포 & 트렌드 */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
              <div className="mb-4 text-sm font-black text-cookie-brown">이상 유형별 분포</div>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={anomalyData.by_type || []} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#FFD93D40" horizontal={false} />
                  <XAxis type="number" tick={{ fill: '#5C4A3D', fontSize: 11 }} />
                  <YAxis type="category" dataKey="type" tick={{ fill: '#5C4A3D', fontSize: 10 }} width={120} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="count" name="탐지 수" radius={[0, 4, 4, 0]}>
                    {(anomalyData.by_type || []).map((entry, idx) => (
                      <Cell key={idx} fill={entry.severity === 'high' ? '#EF4444' : entry.severity === 'medium' ? '#F97316' : '#EAB308'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
              <div className="mb-4 text-sm font-black text-cookie-brown">일별 이상 탐지 트렌드</div>
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={anomalyData.trend || []}>
                  <defs>
                    <linearGradient id="colorAnomaly" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#EF4444" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#EF4444" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#FFD93D40" />
                  <XAxis dataKey="date" tick={{ fill: '#5C4A3D', fontSize: 11 }} />
                  <YAxis tick={{ fill: '#5C4A3D', fontSize: 11 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Area type="monotone" dataKey="count" name="탐지 수" stroke="#EF4444" fill="url(#colorAnomaly)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* 최근 알림 */}
          <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
            <div className="flex items-center gap-2 mb-4">
              <Zap size={18} className="text-red-500" />
              <span className="text-sm font-black text-cookie-brown">실시간 이상 탐지 알림</span>
            </div>
            <div className="space-y-3">
              {(anomalyData.recent_alerts || []).map((alert, idx) => (
                <div key={idx} className={`flex items-center gap-4 p-4 rounded-2xl border-2 ${
                  alert.severity === 'high' ? 'border-red-200 bg-red-50' :
                  alert.severity === 'medium' ? 'border-orange-200 bg-orange-50' :
                  'border-yellow-200 bg-yellow-50'
                }`}>
                  <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
                    alert.severity === 'high' ? 'bg-red-500' :
                    alert.severity === 'medium' ? 'bg-orange-500' : 'bg-yellow-500'
                  }`}>
                    <AlertTriangle size={18} className="text-white" />
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-bold text-cookie-brown">{alert.id}</span>
                      <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                        alert.severity === 'high' ? 'bg-red-200 text-red-700' :
                        alert.severity === 'medium' ? 'bg-orange-200 text-orange-700' :
                        'bg-yellow-200 text-yellow-700'
                      }`}>{alert.type}</span>
                    </div>
                    <p className="text-sm text-cookie-brown/70">{alert.detail}</p>
                  </div>
                  <div className="text-xs text-cookie-brown/50">{alert.time}</div>
                </div>
              ))}
            </div>
          </div>
          </>
          )}
        </div>
      )}

      {/* 예측 분석 */}
      {activeTab === 'prediction' && (
        <div className="space-y-6">
          {!predictionData ? (
            <div className="text-center py-16 rounded-3xl border-2 border-cookie-orange/20 bg-white/80">
              <Brain size={48} className="mx-auto mb-3 text-cookie-brown/30" />
              <p className="text-sm font-semibold text-cookie-brown/50">예측 데이터를 불러올 수 없습니다</p>
              <p className="text-xs text-cookie-brown/40 mt-1">백엔드 API 연결을 확인하세요</p>
            </div>
          ) : (
          <>
          {/* 예측 유형 선택 */}
          <div className="flex gap-2">
            {[
              { key: 'churn', label: '이탈 예측', icon: UserMinus },
              { key: 'revenue', label: '매출 예측', icon: DollarSign },
              { key: 'engagement', label: '참여도 예측', icon: Activity },
            ].map(tab => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.key}
                  onClick={() => setPredictionTab(tab.key)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-bold transition-all ${
                    predictionTab === tab.key
                      ? 'bg-cookie-brown text-white'
                      : 'bg-white border-2 border-cookie-orange/20 text-cookie-brown hover:bg-cookie-beige'
                  }`}
                >
                  <Icon size={14} />
                  {tab.label}
                </button>
              );
            })}
          </div>

          {/* 이탈 예측 */}
          {predictionTab === 'churn' && (
            <>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="rounded-2xl border-2 border-red-200 bg-red-50 p-4">
                  <div className="text-xs font-bold text-red-700 mb-1">고위험 이탈</div>
                  <div className="text-2xl font-black text-red-600">{predictionData.churn.high_risk_count}</div>
                  <div className="text-xs text-red-600/70">유저</div>
                </div>
                <div className="rounded-2xl border-2 border-orange-200 bg-orange-50 p-4">
                  <div className="text-xs font-bold text-orange-700 mb-1">중위험 이탈</div>
                  <div className="text-2xl font-black text-orange-600">{predictionData.churn.medium_risk_count}</div>
                  <div className="text-xs text-orange-600/70">유저</div>
                </div>
                <div className="rounded-2xl border-2 border-green-200 bg-green-50 p-4">
                  <div className="text-xs font-bold text-green-700 mb-1">안전</div>
                  <div className="text-2xl font-black text-green-600">{predictionData.churn.low_risk_count}</div>
                  <div className="text-xs text-green-600/70">유저</div>
                </div>
                <div className="rounded-2xl border-2 border-cookie-orange/20 bg-white/80 p-4">
                  <div className="text-xs font-bold text-cookie-brown mb-1">모델 정확도</div>
                  <div className="text-2xl font-black text-cookie-brown">{predictionData.churn.model_accuracy}%</div>
                  <div className="text-xs text-cookie-brown/60">F1 Score</div>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* 이탈 요인 분석 */}
                <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
                  <div className="mb-4 text-sm font-black text-cookie-brown">이탈 예측 주요 요인</div>
                  <div className="space-y-3">
                    {predictionData.churn.top_factors.map((factor, idx) => (
                      <div key={idx} className="flex items-center gap-3">
                        <span className="w-6 h-6 rounded-full bg-cookie-orange text-white text-xs font-bold flex items-center justify-center">
                          {idx + 1}
                        </span>
                        <div className="flex-1">
                          <div className="flex justify-between mb-1">
                            <span className="text-sm font-semibold text-cookie-brown">{factor.factor}</span>
                            <span className="text-sm font-bold text-cookie-orange">{(factor.importance * 100).toFixed(0)}%</span>
                          </div>
                          <div className="h-2 bg-cookie-beige rounded-full overflow-hidden">
                            <div
                              className="h-full rounded-full bg-gradient-to-r from-cookie-yellow to-cookie-orange"
                              style={{ width: `${factor.importance * 100}%` }}
                            />
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* 고위험 유저 목록 */}
                <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
                  <div className="mb-4 text-sm font-black text-cookie-brown">이탈 고위험 유저</div>
                  <div className="space-y-3">
                    {(predictionData.churn?.high_risk_users || []).map((user, idx) => (
                      <div key={idx} className="flex items-center gap-4 p-3 rounded-2xl bg-red-50 border border-red-200">
                        <div className="w-10 h-10 rounded-full bg-red-500 text-white font-bold flex items-center justify-center text-sm">
                          {user.probability}%
                        </div>
                        <div className="flex-1">
                          <div className="font-bold text-cookie-brown">{user.name}</div>
                          <div className="text-xs text-cookie-brown/60">{user.id} · {user.segment}</div>
                        </div>
                        <div className="text-xs text-red-600 font-semibold">{user.last_active}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </>
          )}

          {/* 매출 예측 */}
          {predictionTab === 'revenue' && predictionData?.revenue && (
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="rounded-2xl border-2 border-green-200 bg-green-50 p-4">
                <div className="text-xs font-bold text-green-700 mb-1">예상 월매출</div>
                <div className="text-xl font-black text-green-600">₩{((predictionData.revenue.predicted_monthly || 0) / 10000).toFixed(0)}만</div>
                <div className="flex items-center gap-1 text-xs text-green-600">
                  <ArrowUpRight size={12} />+{predictionData.revenue.growth_rate || 0}%
                </div>
              </div>
              <div className="rounded-2xl border-2 border-blue-200 bg-blue-50 p-4">
                <div className="text-xs font-bold text-blue-700 mb-1">예상 ARPU</div>
                <div className="text-xl font-black text-blue-600">₩{(predictionData.revenue.predicted_arpu || 0).toLocaleString()}</div>
                <div className="text-xs text-blue-600/70">유저당 평균</div>
              </div>
              <div className="rounded-2xl border-2 border-purple-200 bg-purple-50 p-4">
                <div className="text-xs font-bold text-purple-700 mb-1">예상 ARPPU</div>
                <div className="text-xl font-black text-purple-600">₩{(predictionData.revenue.predicted_arppu || 0).toLocaleString()}</div>
                <div className="text-xs text-purple-600/70">과금유저 평균</div>
              </div>
              <div className="rounded-2xl border-2 border-cookie-orange/20 bg-white/80 p-4">
                <div className="text-xs font-bold text-cookie-brown mb-1">신뢰도</div>
                <div className="text-xl font-black text-cookie-brown">{predictionData.revenue.confidence || 0}%</div>
                <div className="text-xs text-cookie-brown/60">예측 정확도</div>
              </div>
              <div className="rounded-2xl border-2 border-pink-200 bg-pink-50 p-4 col-span-2 lg:col-span-1">
                <div className="text-xs font-bold text-pink-700 mb-1">Whale</div>
                <div className="text-xl font-black text-pink-600">{predictionData.revenue.whale_count || 0}명</div>
                <div className="text-xs text-pink-600/70">VIP 고과금 유저</div>
              </div>
              <div className="rounded-2xl border-2 border-cyan-200 bg-cyan-50 p-4 col-span-2 lg:col-span-1">
                <div className="text-xs font-bold text-cyan-700 mb-1">Dolphin</div>
                <div className="text-xl font-black text-cyan-600">{predictionData.revenue.dolphin_count || 0}명</div>
                <div className="text-xs text-cyan-600/70">중과금 유저</div>
              </div>
              <div className="rounded-2xl border-2 border-teal-200 bg-teal-50 p-4 col-span-2">
                <div className="text-xs font-bold text-teal-700 mb-1">Minnow</div>
                <div className="text-xl font-black text-teal-600">{predictionData.revenue.minnow_count || 0}명</div>
                <div className="text-xs text-teal-600/70">소과금 유저</div>
              </div>
            </div>
          )}

          {/* 참여도 예측 */}
          {predictionTab === 'engagement' && predictionData?.engagement && (
            <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
              <div className="rounded-2xl border-2 border-blue-200 bg-blue-50 p-4">
                <div className="text-xs font-bold text-blue-700 mb-1">예상 DAU</div>
                <div className="text-2xl font-black text-blue-600">{predictionData.engagement.predicted_dau || 0}</div>
                <div className="text-xs text-blue-600/70">일일 활성 유저</div>
              </div>
              <div className="rounded-2xl border-2 border-indigo-200 bg-indigo-50 p-4">
                <div className="text-xs font-bold text-indigo-700 mb-1">예상 MAU</div>
                <div className="text-2xl font-black text-indigo-600">{predictionData.engagement.predicted_mau || 0}</div>
                <div className="text-xs text-indigo-600/70">월간 활성 유저</div>
              </div>
              <div className="rounded-2xl border-2 border-violet-200 bg-violet-50 p-4">
                <div className="text-xs font-bold text-violet-700 mb-1">Stickiness</div>
                <div className="text-2xl font-black text-violet-600">{predictionData.engagement.stickiness || 0}%</div>
                <div className="text-xs text-violet-600/70">DAU/MAU</div>
              </div>
              <div className="rounded-2xl border-2 border-cookie-orange/20 bg-white/80 p-4">
                <div className="text-xs font-bold text-cookie-brown mb-1">평균 세션</div>
                <div className="text-2xl font-black text-cookie-brown">{predictionData.engagement.avg_session || 0}분</div>
                <div className="text-xs text-cookie-brown/60">세션당 플레이 시간</div>
              </div>
              <div className="rounded-2xl border-2 border-cookie-orange/20 bg-white/80 p-4 col-span-2">
                <div className="text-xs font-bold text-cookie-brown mb-1">일일 세션 수</div>
                <div className="text-2xl font-black text-cookie-brown">{predictionData.engagement.sessions_per_day || 0}</div>
                <div className="text-xs text-cookie-brown/60">유저당 평균 접속 횟수</div>
              </div>
            </div>
          )}
          </>
          )}
        </div>
      )}

      {/* 코호트 분석 */}
      {activeTab === 'cohort' && (
        <div className="space-y-6">
          {!cohortData ? (
            <div className="text-center py-16 rounded-3xl border-2 border-cookie-orange/20 bg-white/80">
              <Target size={48} className="mx-auto mb-3 text-cookie-brown/30" />
              <p className="text-sm font-semibold text-cookie-brown/50">코호트 데이터를 불러올 수 없습니다</p>
              <p className="text-xs text-cookie-brown/40 mt-1">백엔드 API 연결을 확인하세요</p>
            </div>
          ) : (
          <>
          {/* 코호트 유형 선택 */}
          <div className="flex gap-2">
            {[
              { key: 'retention', label: '리텐션', icon: Repeat },
              { key: 'ltv', label: 'LTV', icon: DollarSign },
              { key: 'conversion', label: '전환 퍼널', icon: Target },
            ].map(tab => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.key}
                  onClick={() => setCohortTab(tab.key)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-bold transition-all ${
                    cohortTab === tab.key
                      ? 'bg-cookie-brown text-white'
                      : 'bg-white border-2 border-cookie-orange/20 text-cookie-brown hover:bg-cookie-beige'
                  }`}
                >
                  <Icon size={14} />
                  {tab.label}
                </button>
              );
            })}
          </div>

          {/* 리텐션 히트맵 */}
          {cohortTab === 'retention' && (
            <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
              <div className="mb-4 text-sm font-black text-cookie-brown">주간 리텐션 코호트</div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b-2 border-cookie-orange/10">
                      <th className="text-left py-3 px-3 font-bold text-cookie-brown">코호트</th>
                      <th className="text-center py-3 px-3 font-bold text-cookie-brown">Week 0</th>
                      <th className="text-center py-3 px-3 font-bold text-cookie-brown">Week 1</th>
                      <th className="text-center py-3 px-3 font-bold text-cookie-brown">Week 2</th>
                      <th className="text-center py-3 px-3 font-bold text-cookie-brown">Week 3</th>
                      <th className="text-center py-3 px-3 font-bold text-cookie-brown">Week 4</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(cohortData.retention || []).map((row, idx) => (
                      <tr key={idx} className="border-b border-cookie-orange/5">
                        <td className="py-3 px-3 font-semibold text-cookie-brown">{row.cohort}</td>
                        {['week0', 'week1', 'week2', 'week3', 'week4'].map((week) => (
                          <td key={week} className="py-3 px-3 text-center">
                            {row[week] !== null ? (
                              <span
                                className="inline-block px-3 py-1 rounded-lg text-xs font-bold"
                                style={{
                                  backgroundColor: `rgba(255, 140, 66, ${row[week] / 100})`,
                                  color: row[week] > 50 ? 'white' : '#5C4A3D'
                                }}
                              >
                                {row[week]}%
                              </span>
                            ) : (
                              <span className="text-cookie-brown/30">-</span>
                            )}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* LTV 코호트 */}
          {cohortTab === 'ltv' && (
            <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
              <div className="mb-4 text-sm font-black text-cookie-brown">월별 코호트 LTV</div>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={cohortData.ltv_by_cohort || []}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#FFD93D40" />
                  <XAxis dataKey="cohort" tick={{ fill: '#5C4A3D', fontSize: 11 }} />
                  <YAxis tick={{ fill: '#5C4A3D', fontSize: 11 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <Bar dataKey="ltv" name="LTV (원)" fill="#FF8C42" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="users" name="유저 수" fill="#4ADE80" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* 전환 퍼널 */}
          {cohortTab === 'conversion' && (
            <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
              <div className="mb-4 text-sm font-black text-cookie-brown">코호트별 전환 퍼널</div>
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={cohortData.conversion || []} margin={{ top: 20, right: 30, left: 0, bottom: 40 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#FFD93D40" />
                  <XAxis dataKey="cohort" tick={{ fill: '#5C4A3D', fontSize: 10 }} angle={-15} textAnchor="end" />
                  <YAxis tick={{ fill: '#5C4A3D', fontSize: 11 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <Bar dataKey="registered" name="가입" fill="#60A5FA" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="activated" name="활성화" fill="#4ADE80" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="engaged" name="참여" fill="#FFD93D" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="converted" name="전환" fill="#F472B6" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="retained" name="유지" fill="#A78BFA" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
          </>
          )}
        </div>
      )}

      {/* 트렌드 분석 */}
      {activeTab === 'trend' && (
        <div className="space-y-6">
          {!trendData ? (
            <div className="text-center py-16 rounded-3xl border-2 border-cookie-orange/20 bg-white/80">
              <TrendingUp size={48} className="mx-auto mb-3 text-cookie-brown/30" />
              <p className="text-sm font-semibold text-cookie-brown/50">트렌드 데이터를 불러올 수 없습니다</p>
              <p className="text-xs text-cookie-brown/40 mt-1">백엔드 API 연결을 확인하세요</p>
            </div>
          ) : (
          <>
          {/* KPI 요약 카드 */}
          <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
            {(trendData.kpis || []).map((kpi, idx) => (
              <div key={idx} className="rounded-2xl border-2 border-cookie-orange/20 bg-white/80 p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-bold text-cookie-brown/60">{kpi.name}</span>
                  <span className={`flex items-center gap-1 text-xs font-bold ${
                    kpi.change >= 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {kpi.change >= 0 ? <ArrowUpRight size={12} /> : <ArrowDownRight size={12} />}
                    {kpi.change >= 0 ? '+' : ''}{kpi.change}%
                  </span>
                </div>
                <div className="text-2xl font-black text-cookie-brown">
                  {kpi.name.includes('ARPU') ? '₩' : ''}{typeof kpi.current === 'number' ? kpi.current.toLocaleString() : kpi.current}{kpi.name.includes('률') || kpi.name.includes('전환') ? '%' : ''}
                </div>
                <div className="text-xs text-cookie-brown/50">이전: {kpi.previous.toLocaleString()}</div>
              </div>
            ))}
          </div>

          {/* 일별 메트릭 차트 */}
          {(trendData.daily_metrics?.length > 0) && (
          <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
            <div className="mb-4 text-sm font-black text-cookie-brown">일별 핵심 지표 추이</div>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={trendData.daily_metrics}>
                <CartesianGrid strokeDasharray="3 3" stroke="#FFD93D40" />
                <XAxis dataKey="date" tick={{ fill: '#5C4A3D', fontSize: 11 }} />
                <YAxis yAxisId="left" tick={{ fill: '#5C4A3D', fontSize: 11 }} />
                <YAxis yAxisId="right" orientation="right" tick={{ fill: '#5C4A3D', fontSize: 11 }} />
                <Tooltip content={<CustomTooltip />} />
                <Legend />
                <Line yAxisId="left" type="monotone" dataKey="dau" name="DAU" stroke="#FF8C42" strokeWidth={2} dot={{ r: 4 }} />
                <Line yAxisId="left" type="monotone" dataKey="new_users" name="신규가입" stroke="#4ADE80" strokeWidth={2} dot={{ r: 4 }} />
                <Line yAxisId="right" type="monotone" dataKey="sessions" name="세션수" stroke="#60A5FA" strokeWidth={2} dot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          )}

          {/* 예측 & 상관관계 */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* DAU 예측 */}
            <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
              <div className="flex items-center gap-2 mb-4">
                <Brain size={18} className="text-cookie-orange" />
                <span className="text-sm font-black text-cookie-brown">DAU 예측 (5일)</span>
              </div>
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={trendData.forecast || []}>
                  <defs>
                    <linearGradient id="colorForecast" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#A78BFA" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#A78BFA" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#FFD93D40" />
                  <XAxis dataKey="date" tick={{ fill: '#5C4A3D', fontSize: 11 }} />
                  <YAxis tick={{ fill: '#5C4A3D', fontSize: 11 }} domain={['dataMin - 20', 'dataMax + 20']} />
                  <Tooltip content={<CustomTooltip />} />
                  <Area type="monotone" dataKey="upper" name="상한" stroke="transparent" fill="#A78BFA" fillOpacity={0.2} />
                  <Area type="monotone" dataKey="lower" name="하한" stroke="transparent" fill="transparent" />
                  <Line type="monotone" dataKey="predicted_dau" name="예측 DAU" stroke="#A78BFA" strokeWidth={2} strokeDasharray="5 5" dot={{ r: 4 }} />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* 상관관계 분석 */}
            <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
              <div className="flex items-center gap-2 mb-4">
                <BarChart3 size={18} className="text-cookie-orange" />
                <span className="text-sm font-black text-cookie-brown">지표 상관관계</span>
              </div>
              <div className="space-y-3">
                {(trendData.correlation || []).map((item, idx) => {
                  const corr = item.correlation ?? 0;
                  return (
                  <div key={idx} className="flex items-center gap-3">
                    <div className="flex-1">
                      <div className="flex justify-between mb-1">
                        <span className="text-xs font-semibold text-cookie-brown">{item.var1 || item.metric1} ↔ {item.var2 || item.metric2}</span>
                        <span className={`text-xs font-bold ${
                          corr >= 0.8 ? 'text-green-600' :
                          corr >= 0.6 ? 'text-yellow-600' : 'text-red-600'
                        }`}>
                          {corr.toFixed(2)}
                        </span>
                      </div>
                      <div className="h-2 bg-cookie-beige rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full ${
                            corr >= 0.8 ? 'bg-green-500' :
                            corr >= 0.6 ? 'bg-yellow-500' : 'bg-red-500'
                          }`}
                          style={{ width: `${corr * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>
                  );
                })}
              </div>
            </div>
          </div>
          </>
          )}
        </div>
      )}
    </div>
  );
}
