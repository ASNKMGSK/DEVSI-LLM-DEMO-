// components/panels/DashboardPanel.js
// CookieRun AI Platform - 대시보드 패널 (Recharts 버전)

import { useEffect, useMemo, useState } from 'react';
import toast from 'react-hot-toast';
import KpiCard from '@/components/KpiCard';
import EmptyState from '@/components/EmptyState';
import { SkeletonCard } from '@/components/Skeleton';
import {
  Cookie, Users, Globe, BarChart3, TrendingUp, RefreshCw,
  AlertTriangle, Zap, ArrowUpRight, ArrowDownRight, Brain, Target
} from 'lucide-react';
import SectionHeader from '@/components/SectionHeader';
import {
  PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, AreaChart, Area, RadialBarChart, RadialBar
} from 'recharts';

// CookieRun 테마 색상
const COLORS = {
  primary: ['#FF8C42', '#FFD93D', '#4ADE80', '#60A5FA', '#F472B6', '#A78BFA'],
  grades: {
    커먼: '#9CA3AF',
    레어: '#60A5FA',
    에픽: '#F472B6',
    레전더리: '#FBBF24',
    에인션트: '#FF8C42',
  },
};

// 샘플 데이터 (API 실패 시 사용)
const SAMPLE_DATA = {
  status: 'SUCCESS',
  cookie_stats: {
    total: 85,
    by_grade: {
      커먼: 25,
      레어: 28,
      에픽: 18,
      레전더리: 10,
      에인션트: 4,
    },
  },
  user_stats: {
    total: 12450,
    anomaly_count: 23,
    segments: {
      '하드코어 게이머': 2840,
      '캐주얼 유저': 5620,
      '신규 유저': 2100,
      '복귀 유저': 890,
      '고래 유저': 1000,
    },
  },
  translation_stats: {
    total: 4520,
    avg_quality: 94.2,
    by_language: {
      en: 1200,
      ja: 980,
      zh: 870,
      'zh-TW': 450,
      th: 380,
      id: 320,
      de: 220,
      fr: 100,
    },
  },
  event_stats: {
    total: 156780,
    by_type: {
      로그인: 45000,
      스테이지클리어: 38000,
      가챠: 22000,
      아이템구매: 18000,
      길드활동: 15000,
      PvP: 18780,
    },
  },
  daily_active_users: [
    { date: '01/25', users: 8200 },
    { date: '01/26', users: 8450 },
    { date: '01/27', users: 9100 },
    { date: '01/28', users: 8800 },
    { date: '01/29', users: 9500 },
    { date: '01/30', users: 10200 },
    { date: '01/31', users: 9800 },
  ],
};

// 커스텀 툴팁
const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-xl border-2 border-cookie-orange/20 bg-white/95 px-3 py-2 shadow-lg backdrop-blur">
      <p className="text-xs font-bold text-cookie-brown">{label}</p>
      {payload.map((entry, idx) => (
        <p key={idx} className="text-sm font-semibold" style={{ color: entry.color }}>
          {entry.name}: {typeof entry.value === 'number' ? entry.value.toLocaleString() : entry.value}
        </p>
      ))}
    </div>
  );
};

// 파이 차트용 커스텀 툴팁
const PieTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const data = payload[0];
  return (
    <div className="rounded-xl border-2 border-cookie-orange/20 bg-white/95 px-3 py-2 shadow-lg backdrop-blur">
      <p className="text-xs font-bold text-cookie-brown">{data.name}</p>
      <p className="text-sm font-semibold" style={{ color: data.payload.fill }}>
        {data.value.toLocaleString()}명 ({((data.value / data.payload.total) * 100).toFixed(1)}%)
      </p>
    </div>
  );
};

// 커스텀 라벨
const renderCustomLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent, name }) => {
  if (percent < 0.05) return null;
  const RADIAN = Math.PI / 180;
  const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
  const x = cx + radius * Math.cos(-midAngle * RADIAN);
  const y = cy + radius * Math.sin(-midAngle * RADIAN);

  return (
    <text
      x={x}
      y={y}
      fill="#5C4A3D"
      textAnchor={x > cx ? 'start' : 'end'}
      dominantBaseline="central"
      className="text-[10px] font-bold"
    >
      {`${name} ${(percent * 100).toFixed(0)}%`}
    </text>
  );
};

export default function DashboardPanel({ auth, selectedCookie, apiCall }) {
  const [dashboard, setDashboard] = useState(null);
  const [insights, setInsights] = useState([]);
  const [loading, setLoading] = useState(false);

  const loadData = async () => {
    setLoading(true);

    // 대시보드 요약과 인사이트를 병렬로 가져오기
    const [summaryRes, insightsRes] = await Promise.all([
      apiCall({
        endpoint: '/api/dashboard/summary',
        auth,
        timeoutMs: 30000,
      }),
      apiCall({
        endpoint: '/api/dashboard/insights',
        auth,
        timeoutMs: 10000,
      }),
    ]);

    setLoading(false);

    if (summaryRes?.status === 'SUCCESS') {
      setDashboard(summaryRes);
    } else {
      setDashboard(null);
      toast.error('대시보드 데이터를 불러올 수 없습니다');
    }

    if (insightsRes?.status === 'SUCCESS' && insightsRes.insights) {
      setInsights(insightsRes.insights);
    }
  };

  useEffect(() => {
    loadData();
  }, [auth, apiCall]);

  // 세그먼트 분포 차트 데이터
  const segmentData = useMemo(() => {
    if (!dashboard?.user_stats?.segments) return [];
    const segments = dashboard.user_stats.segments;
    const total = Object.values(segments).reduce((a, b) => a + b, 0);
    return Object.entries(segments).map(([name, value], idx) => ({
      name,
      value,
      total,
      fill: COLORS.primary[idx % COLORS.primary.length],
    }));
  }, [dashboard]);

  // 이벤트 통계 차트 데이터
  const eventData = useMemo(() => {
    if (!dashboard?.event_stats?.by_type) return [];
    return Object.entries(dashboard.event_stats.by_type).map(([name, value], idx) => ({
      name,
      value,
      fill: COLORS.primary[idx % COLORS.primary.length],
    }));
  }, [dashboard]);

  // 쿠키 등급 데이터
  const gradeData = useMemo(() => {
    if (!dashboard?.cookie_stats?.by_grade) return [];
    return Object.entries(dashboard.cookie_stats.by_grade).map(([name, value]) => ({
      name,
      value,
      fill: COLORS.grades[name] || '#FF8C42',
    }));
  }, [dashboard]);

  // DAU 데이터
  const dauData = useMemo(() => {
    return dashboard?.daily_active_users || [];
  }, [dashboard]);

  // 번역 언어 데이터
  const langData = useMemo(() => {
    if (!dashboard?.translation_stats?.by_language) return [];
    const langNames = {
      en: '영어', ja: '일본어', zh: '중국어', 'zh-TW': '번체',
      th: '태국어', id: '인니어', de: '독일어', fr: '프랑스어',
      es: '스페인어', pt: '포르투갈어',
    };
    return Object.entries(dashboard.translation_stats.by_language)
      .map(([code, value], idx) => ({
        name: langNames[code] || code,
        value,
        fill: COLORS.primary[idx % COLORS.primary.length],
      }))
      .sort((a, b) => b.value - a.value);
  }, [dashboard]);

  return (
    <div>
      <SectionHeader
        title="CookieRun AI 대시보드"
        subtitle="플랫폼 현황 요약"
        right={
          <div className="flex items-center gap-2">
            <button
              onClick={loadData}
              disabled={loading}
              className="rounded-full border-2 border-cookie-orange/20 bg-white/80 p-1.5 hover:bg-cookie-beige transition disabled:opacity-50"
            >
              <RefreshCw size={14} className={`text-cookie-brown ${loading ? 'animate-spin' : ''}`} />
            </button>
            {!loading && (
              <span className={`rounded-full border-2 px-2 py-1 text-[10px] font-black ${
                dashboard
                  ? 'border-green-400/50 bg-green-50 text-green-700'
                  : 'border-red-400/50 bg-red-50 text-red-700'
              }`}>
                {dashboard ? 'LIVE' : 'NO DATA'}
              </span>
            )}
          </div>
        }
      />

      {loading && !dashboard ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
          <SkeletonCard />
          <SkeletonCard />
          <SkeletonCard />
          <SkeletonCard />
        </div>
      ) : null}

      {dashboard ? (
        <>
          {/* KPI 카드 */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <KpiCard
              title="쿠키 캐릭터"
              value={`${dashboard.cookie_stats?.total || 0}개`}
              subtitle={`에인션트: ${dashboard.cookie_stats?.by_grade?.에인션트 || 0}`}
              icon={<Cookie size={18} className="text-cookie-brown" />}
              tone="yellow"
            />
            <KpiCard
              title="전체 유저"
              value={`${(dashboard.user_stats?.total || 0).toLocaleString()}명`}
              subtitle={`이상 유저: ${dashboard.user_stats?.anomaly_count || 0}`}
              icon={<Users size={18} className="text-cookie-brown" />}
              tone="orange"
            />
            <KpiCard
              title="번역 데이터"
              value={`${(dashboard.translation_stats?.total || 0).toLocaleString()}건`}
              subtitle={`평균 품질: ${dashboard.translation_stats?.avg_quality || '-'}%`}
              icon={<Globe size={18} className="text-cookie-brown" />}
              tone="cream"
            />
            <KpiCard
              title="게임 이벤트"
              value={`${(dashboard.event_stats?.total || 0).toLocaleString()}건`}
              subtitle="최근 30일"
              icon={<BarChart3 size={18} className="text-cookie-brown" />}
              tone="green"
            />
          </div>

          {/* DAU 트렌드 차트 */}
          <div className="mb-6 rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
            <div className="flex items-center gap-2 mb-4">
              <TrendingUp size={18} className="text-cookie-orange" />
              <span className="text-sm font-black text-cookie-brown">일일 활성 사용자 (DAU)</span>
            </div>
            {dauData.length > 0 ? (
            <ResponsiveContainer width="100%" height={240}>
              <AreaChart data={dauData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorUsers" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#FF8C42" stopOpacity={0.4}/>
                    <stop offset="95%" stopColor="#FF8C42" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#FFD93D40" />
                <XAxis
                  dataKey="date"
                  tick={{ fill: '#5C4A3D', fontSize: 11 }}
                  tickLine={{ stroke: '#FFD93D60' }}
                  axisLine={{ stroke: '#FFD93D60' }}
                />
                <YAxis
                  tick={{ fill: '#5C4A3D', fontSize: 11 }}
                  tickLine={{ stroke: '#FFD93D60' }}
                  axisLine={{ stroke: '#FFD93D60' }}
                  tickFormatter={(v) => `${(v/1000).toFixed(0)}K`}
                />
                <Tooltip content={<CustomTooltip />} />
                <Area
                  type="monotone"
                  dataKey="users"
                  name="사용자"
                  stroke="#FF8C42"
                  strokeWidth={3}
                  fillOpacity={1}
                  fill="url(#colorUsers)"
                  dot={{ fill: '#FF8C42', strokeWidth: 2, r: 4 }}
                  activeDot={{ r: 6, stroke: '#fff', strokeWidth: 2 }}
                />
              </AreaChart>
            </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-[240px] text-sm text-cookie-brown/60">
                DAU 데이터 없음
              </div>
            )}
          </div>

          {/* 메인 차트 그리드 */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
            {/* 유저 세그먼트 분포 - 파이 차트 */}
            <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
              <div className="mb-4 text-sm font-black text-cookie-brown">유저 세그먼트 분포</div>
              {segmentData.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={segmentData}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={100}
                      paddingAngle={3}
                      dataKey="value"
                      labelLine={false}
                      label={renderCustomLabel}
                      animationBegin={0}
                      animationDuration={800}
                    >
                      {segmentData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.fill} stroke="#fff" strokeWidth={2} />
                      ))}
                    </Pie>
                    <Tooltip content={<PieTooltip />} />
                    <Legend
                      verticalAlign="bottom"
                      height={36}
                      formatter={(value) => <span className="text-xs font-semibold text-cookie-brown">{value}</span>}
                    />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex items-center justify-center h-[300px] text-sm text-cookie-brown/60">
                  세그먼트 데이터 없음
                </div>
              )}
            </div>

            {/* 이벤트 타입별 통계 - 바 차트 */}
            <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
              <div className="mb-4 text-sm font-black text-cookie-brown">이벤트 타입별 통계</div>
              {eventData.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={eventData} margin={{ top: 10, right: 20, left: 0, bottom: 40 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#FFD93D40" vertical={false} />
                    <XAxis
                      dataKey="name"
                      tick={{ fill: '#5C4A3D', fontSize: 11 }}
                      tickLine={false}
                      axisLine={{ stroke: '#FFD93D60' }}
                      angle={-20}
                      textAnchor="end"
                      interval={0}
                    />
                    <YAxis
                      tick={{ fill: '#5C4A3D', fontSize: 11 }}
                      tickLine={false}
                      axisLine={false}
                      tickFormatter={(v) => `${(v/1000).toFixed(0)}K`}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar
                      dataKey="value"
                      name="이벤트"
                      radius={[8, 8, 0, 0]}
                      animationDuration={800}
                    >
                      {eventData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.fill} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex items-center justify-center h-[300px] text-sm text-cookie-brown/60">
                  이벤트 데이터 없음
                </div>
              )}
            </div>
          </div>

          {/* 쿠키 등급별 분포 - Radial Bar Chart */}
          {gradeData.length > 0 && (
            <div className="mb-6 rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
              <div className="mb-4 text-sm font-black text-cookie-brown">쿠키 등급별 분포</div>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Radial 차트 */}
                <ResponsiveContainer width="100%" height={250}>
                  <RadialBarChart
                    cx="50%"
                    cy="50%"
                    innerRadius="20%"
                    outerRadius="90%"
                    data={gradeData}
                    startAngle={180}
                    endAngle={0}
                  >
                    <RadialBar
                      minAngle={15}
                      background
                      clockWise
                      dataKey="value"
                      cornerRadius={10}
                      animationDuration={800}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend
                      iconSize={10}
                      layout="horizontal"
                      verticalAlign="bottom"
                      align="center"
                      formatter={(value) => <span className="text-xs font-semibold text-cookie-brown">{value}</span>}
                    />
                  </RadialBarChart>
                </ResponsiveContainer>

                {/* 등급 카드 그리드 */}
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 content-center">
                  {gradeData.map(({ name, value, fill }) => (
                    <div
                      key={name}
                      className="rounded-2xl border-2 p-4 text-center transition-all hover:scale-105 hover:shadow-md"
                      style={{
                        borderColor: `${fill}50`,
                        background: `linear-gradient(135deg, ${fill}10 0%, ${fill}05 100%)`
                      }}
                    >
                      <div className="text-xs font-bold" style={{ color: fill }}>{name}</div>
                      <div className="text-2xl font-black text-cookie-brown">{value}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* 번역 언어별 통계 */}
          {langData.length > 0 && (
            <div className="mb-6 rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-5 shadow-sm backdrop-blur">
              <div className="mb-4 text-sm font-black text-cookie-brown">번역 언어별 통계</div>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={langData} layout="vertical" margin={{ top: 5, right: 30, left: 50, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#FFD93D40" horizontal={false} />
                  <XAxis
                    type="number"
                    tick={{ fill: '#5C4A3D', fontSize: 11 }}
                    tickLine={false}
                    axisLine={false}
                  />
                  <YAxis
                    type="category"
                    dataKey="name"
                    tick={{ fill: '#5C4A3D', fontSize: 11 }}
                    tickLine={false}
                    axisLine={false}
                    width={50}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar
                    dataKey="value"
                    name="번역 건수"
                    radius={[0, 8, 8, 0]}
                    animationDuration={800}
                  >
                    {langData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* AI 인사이트 & 빠른 액션 */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* AI 인사이트 */}
            <div className="rounded-3xl border-2 border-purple-200 bg-gradient-to-br from-purple-50 to-white p-5 shadow-sm">
              <div className="flex items-center gap-2 mb-4">
                <Brain size={18} className="text-purple-600" />
                <span className="text-sm font-black text-purple-900">AI 인사이트</span>
                <span className="ml-auto px-2 py-0.5 rounded-full bg-purple-500 text-white text-[10px] font-bold">
                  LIVE
                </span>
              </div>
              <div className="space-y-3">
                {insights.length > 0 ? insights.map((insight, idx) => {
                  const iconConfig = {
                    positive: { bg: 'bg-green-100', icon: <ArrowUpRight size={14} className="text-green-600" /> },
                    warning: { bg: 'bg-yellow-100', icon: <Target size={14} className="text-yellow-600" /> },
                    neutral: { bg: 'bg-blue-100', icon: <Zap size={14} className="text-blue-600" /> },
                  };
                  const config = iconConfig[insight.type] || iconConfig.neutral;

                  return (
                    <div key={idx} className="flex items-start gap-3 p-3 rounded-2xl bg-white/80 border border-purple-100">
                      <div className={`w-8 h-8 rounded-full ${config.bg} flex items-center justify-center flex-shrink-0`}>
                        {config.icon}
                      </div>
                      <div>
                        <div className="text-sm font-bold text-cookie-brown">{insight.title}</div>
                        <div className="text-xs text-cookie-brown/70">{insight.description}</div>
                      </div>
                    </div>
                  );
                }) : (
                  <div className="flex items-center justify-center p-4 text-sm text-cookie-brown/50">
                    인사이트 로딩 중...
                  </div>
                )}
              </div>
            </div>

            {/* 실시간 알림 */}
            <div className="rounded-3xl border-2 border-red-200 bg-gradient-to-br from-red-50 to-white p-5 shadow-sm">
              <div className="flex items-center gap-2 mb-4">
                <AlertTriangle size={18} className="text-red-600" />
                <span className="text-sm font-black text-red-900">실시간 알림</span>
                <span className="ml-auto px-2 py-0.5 rounded-full bg-red-500 text-white text-[10px] font-bold">
                  {dashboard?.user_stats?.anomaly_count || 3}
                </span>
              </div>
              <div className="space-y-3">
                <div className="flex items-center gap-3 p-3 rounded-2xl bg-white/80 border border-red-100">
                  <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                  <div className="flex-1">
                    <div className="text-sm font-bold text-cookie-brown">비정상 결제 패턴 감지</div>
                    <div className="text-xs text-cookie-brown/70">U000523 - 24시간 내 15회 결제 시도</div>
                  </div>
                  <span className="text-[10px] text-cookie-brown/50">10분 전</span>
                </div>
                <div className="flex items-center gap-3 p-3 rounded-2xl bg-white/80 border border-orange-100">
                  <div className="w-2 h-2 rounded-full bg-orange-500" />
                  <div className="flex-1">
                    <div className="text-sm font-bold text-cookie-brown">봇 의심 행동</div>
                    <div className="text-xs text-cookie-brown/70">U000891 - 패턴화된 반복 행동 감지</div>
                  </div>
                  <span className="text-[10px] text-cookie-brown/50">25분 전</span>
                </div>
                <div className="flex items-center gap-3 p-3 rounded-2xl bg-white/80 border border-yellow-100">
                  <div className="w-2 h-2 rounded-full bg-yellow-500" />
                  <div className="flex-1">
                    <div className="text-sm font-bold text-cookie-brown">계정 공유 의심</div>
                    <div className="text-xs text-cookie-brown/70">U000234 - 다중 기기/IP 동시 접속</div>
                  </div>
                  <span className="text-[10px] text-cookie-brown/50">1시간 전</span>
                </div>
              </div>
            </div>
          </div>
        </>
      ) : (
        !loading && <EmptyState title="데이터가 없습니다" desc="백엔드 API 연결을 확인하세요." />
      )}
    </div>
  );
}
