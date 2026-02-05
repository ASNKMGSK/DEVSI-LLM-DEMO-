// components/panels/AnalysisPanel.js
// CookieRun AI Platform - 상세 분석 패널

import { useEffect, useMemo, useState, useRef, useCallback } from 'react';
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
  { key: 'investment', label: '투자 최적화', icon: DollarSign },
];

// 기간 옵션
const DATE_OPTIONS = [
  { value: '7d', label: '최근 7일' },
  { value: '30d', label: '최근 30일' },
  { value: '90d', label: '최근 90일' },
];

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

  // 자동완성 관련 상태
  const [autocompleteResults, setAutocompleteResults] = useState([]);
  const [showAutocomplete, setShowAutocomplete] = useState(false);
  const [autocompleteLoading, setAutocompleteLoading] = useState(false);
  const autocompleteRef = useRef(null);
  const searchInputRef = useRef(null);

  // 자동완성 debounce 타이머
  const autocompleteTimerRef = useRef(null);

  // 새로운 분석 데이터 상태
  const [anomalyData, setAnomalyData] = useState(null);
  const [predictionData, setPredictionData] = useState(null);
  const [cohortData, setCohortData] = useState(null);
  const [trendData, setTrendData] = useState(null);
  const [predictionTab, setPredictionTab] = useState('churn'); // churn, revenue, engagement
  const [cohortTab, setCohortTab] = useState('retention'); // retention, ltv, conversion

  // 투자 최적화 상태
  const [investmentUser, setInvestmentUser] = useState('');
  const [investmentUserInput, setInvestmentUserInput] = useState(''); // 직접 입력
  const [investmentUserStatus, setInvestmentUserStatus] = useState(null);
  const [investmentResult, setInvestmentResult] = useState(null);
  const [investmentOptimizing, setInvestmentOptimizing] = useState(false);
  const [investmentLoading, setInvestmentLoading] = useState(false);

  // 투자 최적화 예시 유저 (3개)
  const INVESTMENT_EXAMPLE_USERS = [
    { id: 'U000001', name: '헤비유저', description: 'VIP 8, 고자원' },
    { id: 'U000050', name: '일반유저', description: 'VIP 5, 중간 자원' },
    { id: 'U000100', name: '신규유저', description: 'VIP 2, 낮은 자원' },
  ];

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

  // 투자 최적화: 유저 상태 로드
  const loadInvestmentUserStatus = async (userId) => {
    setInvestmentLoading(true);
    try {
      const res = await apiCall({
        endpoint: `/api/investment/user/${userId}`,
        method: 'GET',
        auth,
        timeoutMs: 30000,
      });

      if (res?.status === 'SUCCESS') {
        setInvestmentUserStatus(res.data);
        toast.success(`${userId} 유저 정보를 불러왔습니다`);
      } else {
        setInvestmentUserStatus(null);
        toast.error('유저 정보를 불러올 수 없습니다');
      }
    } catch (error) {
      console.error('Failed to load user status:', error);
      setInvestmentUserStatus(null);
      toast.error('유저 정보 조회에 실패했습니다. 백엔드 연결을 확인하세요.');
    } finally {
      setInvestmentLoading(false);
    }
  };

  // 투자 최적화: 최적화 실행
  const runInvestmentOptimization = async () => {
    if (!investmentUserStatus) return;

    setInvestmentOptimizing(true);
    try {
      const res = await apiCall({
        endpoint: '/api/investment/optimize',
        method: 'POST',
        auth,
        data: { user_id: investmentUser, top_n: 10 },
        timeoutMs: 60000,
      });

      if (res?.status === 'SUCCESS') {
        setInvestmentResult(res.data);
        toast.success('P-PSO 최적화가 완료되었습니다!');
      } else {
        setInvestmentResult(null);
        toast.error('최적화에 실패했습니다');
      }
    } catch (error) {
      console.error('Optimization failed:', error);
      setInvestmentResult(null);
      toast.error('최적화 실행에 실패했습니다. 백엔드 연결을 확인하세요.');
    } finally {
      setInvestmentOptimizing(false);
    }
  };

  // 투자 최적화: 예시 유저 선택 핸들러
  const handleInvestmentExampleSelect = (userId) => {
    setInvestmentUser(userId);
    setInvestmentUserInput('');
    setInvestmentResult(null);
    loadInvestmentUserStatus(userId);
  };

  // 투자 최적화: 직접 입력 조회 핸들러
  const handleInvestmentDirectSearch = () => {
    const trimmed = investmentUserInput.trim();
    if (!trimmed) {
      toast.error('유저 ID를 입력해주세요');
      return;
    }
    setInvestmentUser(trimmed);
    setInvestmentResult(null);
    loadInvestmentUserStatus(trimmed);
  };

  // 투자 최적화: Enter 키 처리
  const handleInvestmentInputKeyDown = (e) => {
    if (e.key === 'Enter') {
      handleInvestmentDirectSearch();
    }
  };

  // 자동완성 검색 (debounced)
  const fetchAutocomplete = useCallback(async (query) => {
    if (!query || query.length < 1) {
      setAutocompleteResults([]);
      setShowAutocomplete(false);
      return;
    }

    setAutocompleteLoading(true);
    try {
      const res = await apiCall({
        endpoint: `/api/users/autocomplete?q=${encodeURIComponent(query)}&limit=8`,
        auth,
        timeoutMs: 5000,
      });

      if (res?.status === 'SUCCESS' && res.users) {
        setAutocompleteResults(res.users);
        setShowAutocomplete(res.users.length > 0);
      } else {
        // fallback: quickSelectUsers에서 필터링
        const filtered = quickSelectUsers.filter(u =>
          u.toLowerCase().includes(query.toLowerCase())
        );
        setAutocompleteResults(filtered.map(id => ({ id, name: id })));
        setShowAutocomplete(filtered.length > 0);
      }
    } catch (e) {
      // 오류 시 빠른 선택에서 필터링
      const filtered = quickSelectUsers.filter(u =>
        u.toLowerCase().includes(query.toLowerCase())
      );
      setAutocompleteResults(filtered.map(id => ({ id, name: id })));
      setShowAutocomplete(filtered.length > 0);
    } finally {
      setAutocompleteLoading(false);
    }
  }, [apiCall, auth, quickSelectUsers]);

  // 자동완성 입력 핸들러 (debounce)
  const handleSearchInputChange = (e) => {
    const value = e.target.value;
    setSearchQuery(value);

    // 이전 타이머 취소
    if (autocompleteTimerRef.current) {
      clearTimeout(autocompleteTimerRef.current);
    }

    // 300ms 후 자동완성 검색
    autocompleteTimerRef.current = setTimeout(() => {
      fetchAutocomplete(value);
    }, 300);
  };

  // 자동완성 항목 선택
  const handleAutocompleteSelect = (user) => {
    setSearchQuery(user.id);
    setShowAutocomplete(false);
    // 선택 후 바로 검색 실행
    setTimeout(() => {
      searchInputRef.current?.blur();
      handleUserSearchDirect(user.id);
    }, 50);
  };

  // 직접 검색 (특정 ID로)
  const handleUserSearchDirect = async (userId) => {
    if (!userId?.trim()) return;
    setLoading(true);
    setShowAutocomplete(false);

    // 기간을 일수로 변환
    const daysMap = { '7d': 7, '30d': 30, '90d': 90 };
    const days = daysMap[dateRange] || 7;

    try {
      const res = await apiCall({
        endpoint: `/api/users/search?q=${encodeURIComponent(userId)}&days=${days}`,
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
      toast.error('유저 검색에 실패했습니다');
      setSelectedUser(null);
    }
    setLoading(false);
  };

  // 클릭 외부 감지 - 자동완성 닫기
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (autocompleteRef.current && !autocompleteRef.current.contains(e.target)) {
        setShowAutocomplete(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // 유저 검색
  const handleUserSearch = async () => {
    if (!searchQuery.trim()) {
      toast.error('유저 ID를 입력하세요');
      return;
    }
    setLoading(true);

    // 기간을 일수로 변환
    const daysMap = { '7d': 7, '30d': 30, '90d': 90 };
    const days = daysMap[dateRange] || 7;

    try {
      // API 호출 시도
      const res = await apiCall({
        endpoint: `/api/users/search?q=${encodeURIComponent(searchQuery)}&days=${days}`,
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

  // 기간 변경 시 선택된 유저가 있으면 자동 재검색
  useEffect(() => {
    if (selectedUser?.id && auth) {
      // 기간을 일수로 변환
      const daysMap = { '7d': 7, '30d': 30, '90d': 90 };
      const days = daysMap[dateRange] || 7;

      const refetchUser = async () => {
        try {
          const res = await apiCall({
            endpoint: `/api/users/search?q=${encodeURIComponent(selectedUser.id)}&days=${days}`,
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
          }
        } catch (e) {
          console.log('유저 데이터 재조회 실패');
        }
      };

      refetchUser();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dateRange]);

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
              <div className="flex-1">
                <input
                  ref={searchInputRef}
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      handleUserSearch();
                    }
                  }}
                  placeholder="유저 ID 또는 닉네임 입력 (예: U000001)"
                  className="w-full px-4 py-2.5 rounded-xl border-2 border-cookie-orange/20 bg-white text-sm text-cookie-brown placeholder:text-cookie-brown/40 outline-none focus:border-cookie-orange transition"
                />
              </div>
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

      {/* 투자 최적화 */}
      {activeTab === 'investment' && (
        <div className="space-y-6">
          {/* 헤더 */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <DollarSign size={20} className="text-cookie-orange" />
              <span className="text-lg font-black text-cookie-brown">P-PSO 리소스 투자 최적화</span>
            </div>
            <button
              onClick={() => loadInvestmentUserStatus(investmentUser)}
              disabled={investmentLoading || !investmentUser}
              className="p-2.5 rounded-xl border-2 border-cookie-orange/20 hover:border-cookie-orange hover:bg-cookie-orange/10 transition-all disabled:opacity-50"
            >
              <RefreshCw className={`w-5 h-5 text-cookie-orange ${investmentLoading ? 'animate-spin' : ''}`} />
            </button>
          </div>

          {/* 유저 선택 UI */}
          <div className="bg-white rounded-2xl p-5 border-2 border-cookie-orange/10 shadow-sm">
            <h3 className="text-sm font-bold text-cookie-brown mb-4 flex items-center gap-2">
              <Users size={16} className="text-cookie-orange" />
              유저 선택
            </h3>

            <div className="flex flex-col gap-4">
              {/* 예시 유저 버튼들 */}
              <div className="flex flex-wrap gap-3">
                <span className="text-sm text-cookie-brown/70 self-center mr-2">예시:</span>
                {INVESTMENT_EXAMPLE_USERS.map((user) => (
                  <button
                    key={user.id}
                    onClick={() => handleInvestmentExampleSelect(user.id)}
                    className={`px-4 py-2.5 rounded-xl border-2 transition-all flex flex-col items-start ${
                      investmentUser === user.id
                        ? 'border-cookie-orange bg-cookie-orange/10 text-cookie-brown'
                        : 'border-cookie-orange/20 hover:border-cookie-orange/40 bg-white text-cookie-brown/80'
                    }`}
                  >
                    <span className="font-bold text-sm">{user.name}</span>
                    <span className="text-xs text-cookie-brown/60">{user.id} - {user.description}</span>
                  </button>
                ))}
              </div>

              {/* 구분선 */}
              <div className="flex items-center gap-3">
                <div className="flex-1 h-px bg-cookie-orange/20" />
                <span className="text-sm text-cookie-brown/50">또는 직접 입력</span>
                <div className="flex-1 h-px bg-cookie-orange/20" />
              </div>

              {/* 직접 입력 */}
              <div className="flex items-center gap-3">
                <input
                  type="text"
                  value={investmentUserInput}
                  onChange={(e) => setInvestmentUserInput(e.target.value)}
                  onKeyDown={handleInvestmentInputKeyDown}
                  placeholder="유저 ID 입력 (예: U000123)"
                  className="flex-1 px-4 py-3 rounded-xl border-2 border-cookie-orange/20 bg-white text-cookie-brown font-medium placeholder:text-cookie-brown/40 focus:border-cookie-orange focus:ring-2 focus:ring-cookie-orange/20 outline-none transition-all"
                />
                <button
                  onClick={handleInvestmentDirectSearch}
                  disabled={investmentLoading || !investmentUserInput.trim()}
                  className="px-6 py-3 rounded-xl bg-cookie-orange text-white font-bold hover:bg-cookie-orange/90 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                >
                  <Search size={18} />
                  조회
                </button>
              </div>

              {/* 현재 선택된 유저 표시 */}
              {investmentUser && (
                <div className="flex items-center gap-2 text-sm text-cookie-brown/70 bg-cookie-yellow/10 px-4 py-2 rounded-xl">
                  <Target size={16} className="text-cookie-orange" />
                  현재 선택: <span className="font-bold text-cookie-brown">{investmentUser}</span>
                </div>
              )}
            </div>
          </div>

          {/* 유저 현황 */}
          {investmentUserStatus && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* 리소스 현황 */}
              <div className="bg-white rounded-2xl p-5 border-2 border-cookie-orange/10 shadow-sm">
                <h3 className="text-sm font-bold text-cookie-brown mb-4 flex items-center gap-2">
                  <DollarSign size={16} className="text-cookie-orange" />
                  보유 리소스
                </h3>
                <div className="space-y-3">
                  {Object.entries(investmentUserStatus.resources || {}).map(([key, value]) => (
                    <div key={key} className="flex items-center justify-between p-2 rounded-lg bg-gray-50">
                      <span className="text-sm text-cookie-brown">{key === 'exp_jelly' ? '경험치 젤리' : key === 'coin' ? '코인' : key === 'skill_powder' ? '스킬 파우더' : key === 'soul_stone' ? '소울스톤' : key}</span>
                      <span className="font-bold text-cookie-brown">{value?.toLocaleString() || 0}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* 쿠키 요약 */}
              <div className="lg:col-span-2 bg-white rounded-2xl p-5 border-2 border-cookie-orange/10 shadow-sm">
                <h3 className="text-sm font-bold text-cookie-brown mb-4 flex items-center gap-2">
                  <span className="text-lg">🍪</span>
                  보유 쿠키 현황
                </h3>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <div className="p-3 rounded-xl bg-gradient-to-br from-cookie-orange/10 to-white border border-cookie-orange/10">
                    <div className="text-xs text-cookie-brown/70">총 쿠키</div>
                    <div className="text-xl font-bold text-cookie-brown">{investmentUserStatus.cookies?.length || 0}</div>
                  </div>
                  <div className="p-3 rounded-xl bg-gradient-to-br from-cookie-yellow/10 to-white border border-cookie-yellow/10">
                    <div className="text-xs text-cookie-brown/70">평균 레벨</div>
                    <div className="text-xl font-bold text-cookie-brown">
                      {Math.round((investmentUserStatus.cookies?.reduce((sum, c) => sum + (c.level || 0), 0) || 0) / (investmentUserStatus.cookies?.length || 1))}
                    </div>
                  </div>
                  <div className="p-3 rounded-xl bg-gradient-to-br from-green-100 to-white border border-green-100">
                    <div className="text-xs text-cookie-brown/70">최대 레벨</div>
                    <div className="text-xl font-bold text-cookie-brown">
                      {Math.max(...(investmentUserStatus.cookies?.map(c => c.level || 0) || [0]))}
                    </div>
                  </div>
                  <div className="p-3 rounded-xl bg-gradient-to-br from-purple-100 to-white border border-purple-100">
                    <div className="text-xs text-cookie-brown/70">총 전투력</div>
                    <div className="text-xl font-bold text-cookie-brown">{investmentUserStatus.total_power?.toLocaleString() || '계산중'}</div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* 최적화 버튼 */}
          <div className="flex justify-center">
            <button
              onClick={runInvestmentOptimization}
              disabled={investmentOptimizing || !investmentUserStatus}
              className="px-8 py-4 bg-gradient-to-r from-cookie-orange to-cookie-yellow text-white font-bold text-lg rounded-2xl shadow-lg hover:shadow-xl transform hover:scale-[1.02] transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-3"
            >
              {investmentOptimizing ? (
                <>
                  <RefreshCw className="w-6 h-6 animate-spin" />
                  P-PSO 최적화 실행 중...
                </>
              ) : (
                <>
                  <Target size={24} />
                  투자 최적화 실행
                </>
              )}
            </button>
          </div>

          {/* 최적화 결과 */}
          {investmentResult && (
            <div className="bg-gradient-to-br from-cookie-yellow/10 via-white to-cookie-orange/10 rounded-2xl p-6 border-2 border-cookie-orange/20 shadow-lg">
              <h3 className="text-lg font-bold text-cookie-brown mb-4 flex items-center gap-2">
                <TrendingUp size={20} className="text-cookie-orange" />
                최적화 결과 - 개인화된 투자 추천
              </h3>

              {/* 예상 효과 */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
                <div className="p-4 rounded-xl bg-white border border-gray-100 shadow-sm">
                  <div className="text-xs text-cookie-brown/70 mb-1">예상 승률 증가</div>
                  <div className="text-xl font-bold text-green-600 flex items-center gap-1">
                    +{Number(investmentResult.total_win_rate_gain || 0).toFixed(1)}%
                    <ArrowUpRight size={18} />
                  </div>
                </div>
                <div className="p-4 rounded-xl bg-white border border-gray-100 shadow-sm">
                  <div className="text-xs text-cookie-brown/70 mb-1">추천 개수</div>
                  <div className="text-xl font-bold text-blue-600">{investmentResult.recommendations?.length || 0}개</div>
                </div>
                <div className="p-4 rounded-xl bg-white border border-gray-100 shadow-sm">
                  <div className="text-xs text-cookie-brown/70 mb-1">평균 효율</div>
                  <div className="text-xl font-bold text-pink-600">{investmentResult.recommendations?.length > 0
                    ? (investmentResult.recommendations.reduce((sum, r) => sum + Number(r.efficiency || 0), 0) / investmentResult.recommendations.length * 100).toFixed(1)
                    : 0}%</div>
                </div>
                <div className="p-4 rounded-xl bg-white border border-gray-100 shadow-sm">
                  <div className="text-xs text-cookie-brown/70 mb-1">최적화 방식</div>
                  <div className="text-xl font-bold text-purple-600">P-PSO</div>
                </div>
              </div>

              {/* 추천 리스트 */}
              <div className="bg-white rounded-xl p-4 border border-cookie-orange/10">
                <h4 className="font-bold text-cookie-brown mb-4">우선순위별 투자 추천</h4>
                <div className="space-y-2">
                  {investmentResult.recommendations?.slice(0, 8).map((rec, idx) => (
                    <div key={idx} className="flex items-center gap-3 p-3 rounded-xl bg-gradient-to-r from-gray-50 to-white border border-gray-100 hover:border-cookie-orange/30 transition-colors">
                      <div className={`w-7 h-7 rounded-lg flex items-center justify-center text-white font-bold text-sm shadow-sm ${
                        idx === 0 ? 'bg-gradient-to-br from-amber-500 to-yellow-500' :
                        idx === 1 ? 'bg-gradient-to-br from-gray-400 to-gray-500' :
                        idx === 2 ? 'bg-gradient-to-br from-orange-400 to-orange-500' :
                        'bg-gradient-to-br from-cookie-orange to-cookie-yellow'
                      }`}>
                        {idx + 1}
                      </div>
                      <div className="flex-1">
                        <div className="font-medium text-cookie-brown">{rec.cookie_name}</div>
                        <div className="text-xs text-cookie-brown/60">{rec.upgrade_type} {rec.from_level} → {rec.to_level}</div>
                      </div>
                      <div className="text-right">
                        <div className="font-bold text-green-600">+{Number(rec.win_rate_gain || 0).toFixed(1)}%</div>
                        <div className="text-xs text-cookie-brown/50">승률 증가</div>
                      </div>
                      <div className="text-right">
                        <div className="font-medium text-cookie-brown text-sm">
                          {(() => {
                            const cost = rec.cost;
                            const formatNum = (n) => n >= 1000 ? `${(n / 1000).toFixed(0)}K` : `${n}`;
                            if (!cost || typeof cost !== 'object') return '0';
                            if (rec.upgrade_type === 'cookie_level') {
                              return `${formatNum(Number(cost.exp_jelly || 0))}/${formatNum(Number(cost.coin || 0))}`;
                            } else if (rec.upgrade_type === 'skill_level') {
                              return `${formatNum(Number(cost.skill_powder || 0))}/${formatNum(Number(cost.coin || 0))}`;
                            } else if (rec.upgrade_type === 'ascension') {
                              return `${Number(cost.soul_stone || 0)}개/${formatNum(Number(cost.coin || 0))}`;
                            }
                            return formatNum(Number(cost.coin || 0));
                          })()}
                        </div>
                        <div className="text-xs text-cookie-brown/50">
                          {rec.upgrade_type === 'cookie_level' ? '젤리/코인' :
                           rec.upgrade_type === 'skill_level' ? '파우더/코인' :
                           rec.upgrade_type === 'ascension' ? '소울/코인' : '비용'}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* 로딩 중 */}
          {investmentLoading && !investmentUserStatus && (
            <div className="text-center py-16">
              <RefreshCw size={48} className="mx-auto mb-3 text-cookie-orange animate-spin" />
              <p className="text-sm font-semibold text-cookie-brown/50">유저 정보를 불러오는 중...</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
