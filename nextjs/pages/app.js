// pages/app.js - CookieRun AI Platform
// 데브시스터즈 기술혁신 프로젝트

import { useCallback, useEffect, useMemo, useState } from 'react';
import { useRouter } from 'next/router';

import Layout from '@/components/Layout';
import Tabs from '@/components/Tabs';

import AgentPanel from '@/components/panels/AgentPanel';
import DashboardPanel from '@/components/panels/DashboardPanel';
import AnalysisPanel from '@/components/panels/AnalysisPanel';
import ModelsPanel from '@/components/panels/ModelsPanel';
import SettingsPanel from '@/components/panels/SettingsPanel';
import UsersPanel from '@/components/panels/UsersPanel';
import LogsPanel from '@/components/panels/LogsPanel';
import RagPanel from '@/components/panels/RagPanel';

import { apiCall as apiCallRaw } from '@/lib/api';
import {
  loadFromStorage,
  saveToStorage,
  loadFromSession,
  removeFromSession,
  STORAGE_KEYS,
} from '@/lib/storage';

// CookieRun AI Platform 예시 질문 (agent/tools.py AVAILABLE_TOOLS 기반)
const EXAMPLE_QUESTIONS = {
  '🍪 쿠키 & 세계관 (RAG)': [
    '쿠키런 킹덤 세계관 시대적 배경이 뭐야?',
    '비스트이스트는 어떤 존재들과 관련돼?',
    '빛의 신이랑 쿠키 세계 관계 설명해줘',
    '베이킹 마법은 어떤 존재들이 사용해?',
    '쿠키 등급 체계 종류 알려줘',
    '클로티드 크림 쿠키 출신 국가가 어디야?',
    '찬란한 영웅들의 신전은 어떤 장소야?',
    '골드치즈 왕국은 어느 대륙에 위치해 있어?',
    '쿠키 전투 배치(전방/중앙/후방) 규칙 알려줘',
    '쿠키런 보물 종류랑 등급 체계 알려줘',
    '소울 잼이랑 고대 영웅 쿠키 관계 설명해줘',
    '어둠의 마녀는 누구야?',
    '고대의 영웅 쿠키 5명 알려줘',
    '고대 영웅 쿠키들이 지닌 빛은 각각 뭐야?',
  ],
  '🌐 번역': [
    '"용감한 쿠키가 오븐에서 탈출했어요!" 영어로 번역해줘',
    '"다크카카오 왕국이 무너졌다" 일본어로 번역해줘',
    '세계관 용어집 보여줘',
    '번역 품질 통계 보여줘',
    '"스테이지 클리어 보상" 카테고리 분류해줘',
  ],
  '🔮 AI 예측 분석': [
    'U000001 유저 이탈 확률 예측해줘',
    'U000100 이탈 위험도 분석해줘',
    'U000557 이탈할 것 같아?',
    '전체 이탈 예측 분석 결과 보여줘',
    '고위험 이탈 유저 5명 보여줘',
    '중위험 이탈 유저 현황 알려줘',
    '저위험 이탈 유저 몇 명이야?',
    '이탈 요인 상위 5개 뭐야?',
    '용감한 쿠키 PvP 승률 알려줘',
    '퓨어바닐라 쿠키 승률이랑 스탯 보여줘',
    '에스프레소맛 쿠키 승률 티어 뭐야?',
    '다크카카오 쿠키 승률 분석해줘',
    'U000001 투자 최적화 추천해줘',
    'U000100 승률 최대화 투자 전략 알려줘',
    'U000050 쿠키 육성 추천해줘',
  ],
  '📈 비즈니스 KPI': [
    '최근 7일 KPI 트렌드 분석해줘',
    '최근 14일 DAU 변화율 알려줘',
    '지난주 ARPU 변화 분석해줘',
    '신규 유저 가입 추이 알려줘',
    '결제 전환율 변화 분석해줘',
    '코호트 리텐션 분석 보여줘',
    '2024-11 코호트 리텐션 어때?',
    'Week 4 평균 리텐션 얼마야?',
    '최근 코호트 Week 1 리텐션 알려줘',
    '이번 달 매출 예측해줘',
    '최근 30일 매출 분석해줘',
    'ARPU랑 ARPPU 알려줘',
    'whale/dolphin/minnow 분포 보여줘',
    '월간 매출 성장률 알려줘',
    '대시보드 전체 현황 요약해줘',
  ],
  '👤 유저 분석': [
    'U000001 유저 분석해줘',
    'U000557 유저 프로필 알려줘',
    'U000100 행동 패턴 분석해줘',
    'U000050 유저 상세 정보 보여줘',
    '유저 세그먼트별 통계 보여줘',
    '하드코어 게이머 세그먼트 몇 명이야?',
    'PvP 전문가 유저 통계 알려줘',
    '캐주얼 유저 세그먼트 현황 알려줘',
    '이상 유저 전체 통계 보여줘',
    '세그먼트별 이상 유저 비율 알려줘',
    'U000001 최근 30일 활동 리포트',
    'U000100 최근 7일 활동 보여줘',
    '최근 30일 게임 이벤트 통계 보여줘',
    '스테이지 클리어 이벤트 현황 알려줘',
    '가챠 이벤트 통계 보여줘',
  ],
};

const DEFAULT_SETTINGS = {
  apiKey: '',
  selectedModel: 'gpt-4o-mini',
  maxTokens: 8000,
  temperature: 0.3,
  systemPrompt: '',
  ragMode: 'rag', // 'rag' | 'lightrag' | 'k2rag' | 'auto'
};

function formatTimestamp(d) {
  const pad = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(
    d.getMinutes()
  )}:${pad(d.getSeconds())}`;
}

export default function AppPage() {
  const router = useRouter();

  const [auth, setAuth] = useState(null);
  const [cookies, setCookies] = useState([]);
  const [kingdoms, setKingdoms] = useState([]);
  const [selectedCookie, setSelectedCookie] = useState(null);

  const [settings, setSettings] = useState(null);  // null로 시작, localStorage에서 로드 후 설정
  const [settingsLoaded, setSettingsLoaded] = useState(false);

  const [agentMessages, setAgentMessages] = useState([]);
  const [activityLog, setActivityLog] = useState([]);
  const [totalQueries, setTotalQueries] = useState(0);

  const [activeTab, setActiveTab] = useState('agent');

  const isAdmin = auth?.user_role === '관리자';

  const tabs = useMemo(() => {
    if (isAdmin) {
      return [
        { key: 'agent', label: '🤖 AI 에이전트' },
        { key: 'dashboard', label: '📊 대시보드' },
        { key: 'analysis', label: '📈 분석' },
        { key: 'models', label: '🧠 ML 모델' },
        { key: 'rag', label: '📚 RAG 문서' },
        { key: 'settings', label: '⚙️ LLM 설정' },
        { key: 'users', label: '👥 사용자' },
        { key: 'logs', label: '📋 로그' },
      ];
    }
    return [
      { key: 'agent', label: '🤖 AI 에이전트' },
      { key: 'dashboard', label: '📊 대시보드' },
      { key: 'analysis', label: '📈 분석' },
    ];
  }, [isAdmin]);

  const apiCall = useCallback((args) => apiCallRaw(args), []);

  const addLog = useCallback(
    (action, detail) => {
      const row = {
        시간: formatTimestamp(new Date()),
        사용자: auth?.username || '-',
        작업: action,
        상세: detail,
      };
      setActivityLog((prev) => [...prev, row]);
    },
    [auth?.username]
  );

  const safeReplace = useCallback(
    (path) => {
      if (!router.isReady) return;
      const cur = router.asPath || '';
      if (cur === path) return;
      router.replace(path);
    },
    [router]
  );

  const onLogout = useCallback(() => {
    removeFromSession(STORAGE_KEYS.AUTH);
    safeReplace('/login');
  }, [safeReplace]);

  const clearLog = useCallback(() => {
    setActivityLog([]);
  }, []);

  // 앱 페이지 90% 배율 적용 (로그인 페이지는 100%)
  useEffect(() => {
    document.documentElement.style.zoom = '0.9';
    return () => {
      document.documentElement.style.zoom = '1';
    };
  }, []);

  // 세션 초기 로드
  useEffect(() => {
    if (!router.isReady) return;

    const a = loadFromSession(STORAGE_KEYS.AUTH, null);
    if (!a?.username || !a?.password) {
      safeReplace('/login');
      return;
    }
    setAuth(a);

    const savedSettings = loadFromStorage(STORAGE_KEYS.SETTINGS, null);
    // 저장된 설정과 기본값 병합 (저장된 값 우선)
    const mergedSettings = { ...DEFAULT_SETTINGS, ...(savedSettings || {}) };
    if (!mergedSettings.apiKey || mergedSettings.apiKey.trim() === '') {
      mergedSettings.apiKey = DEFAULT_SETTINGS.apiKey;
    }
    setSettings(mergedSettings);
    setSettingsLoaded(true);  // 로드 완료 표시

    setAgentMessages(loadFromStorage(STORAGE_KEYS.AGENT_MESSAGES, []));
    setActivityLog(loadFromStorage(STORAGE_KEYS.ACTIVITY_LOG, []));
    setTotalQueries(loadFromStorage(STORAGE_KEYS.TOTAL_QUERIES, 0));
  }, [router.isReady, safeReplace]);

  // 시스템 프롬프트 로드 (백엔드 중앙 관리)
  useEffect(() => {
    if (!auth?.username || !auth?.password) return;

    const cur = settings?.systemPrompt ? String(settings.systemPrompt).trim() : '';
    if (cur.length > 0) return;

    let mounted = true;

    async function loadSystemPrompt() {
      try {
        // 백엔드에서 시스템 프롬프트 로드
        const res = await apiCall({
          endpoint: '/api/settings/prompt',
          method: 'GET',
          auth,
          timeoutMs: 30000,
        });

        if (!mounted) return;

        const data = res?.data || res || {};
        const prompt = data?.systemPrompt || data?.system_prompt || '';
        const promptStr = String(prompt || '').trim();

        if (promptStr.length > 0) {
          setSettings((prev) => ({ ...prev, systemPrompt: promptStr }));
        }
      } catch (e) {
        // 백엔드 연결 실패 시 /api/settings/default 시도
        try {
          const fallback = await apiCall({
            endpoint: '/api/settings/default',
            method: 'GET',
            auth,
            timeoutMs: 30000,
          });

          if (!mounted) return;

          const prompt = fallback?.data?.systemPrompt || fallback?.data?.system_prompt || '';
          const promptStr = String(prompt || '').trim();

          if (promptStr.length > 0) {
            setSettings((prev) => ({ ...prev, systemPrompt: promptStr }));
          }
        } catch (e2) {}
      }
    }

    loadSystemPrompt();

    return () => {
      mounted = false;
    };
  }, [apiCall, auth, settings?.systemPrompt]);

  // 쿠키/왕국 데이터 로드
  useEffect(() => {
    if (!auth?.username || !auth?.password) return;

    let mounted = true;

    async function loadCookies() {
      try {
        const res = await apiCall({ endpoint: '/api/cookies', auth, timeoutMs: 30000 });
        if (!mounted) return;

        if (res?.status === 'SUCCESS' && Array.isArray(res.cookies)) {
          setCookies(res.cookies);
          if (!selectedCookie && res.cookies.length > 0) {
            setSelectedCookie(res.cookies[0].id);
          }
        }
      } catch (e) {
        console.error('Failed to load cookies:', e);
      }
    }

    async function loadKingdoms() {
      try {
        const res = await apiCall({ endpoint: '/api/kingdoms', auth, timeoutMs: 30000 });
        if (!mounted) return;

        if (res?.status === 'SUCCESS' && Array.isArray(res.kingdoms)) {
          setKingdoms(res.kingdoms);
        }
      } catch (e) {
        console.error('Failed to load kingdoms:', e);
      }
    }

    loadCookies();
    loadKingdoms();

    return () => {
      mounted = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiCall, auth]);

  // 스토리지 저장 (로드 완료 후에만 저장하여 기존 설정 보존)
  useEffect(() => {
    if (settingsLoaded && settings) {
      saveToStorage(STORAGE_KEYS.SETTINGS, settings);
    }
  }, [settings, settingsLoaded]);

  useEffect(() => {
    saveToStorage(STORAGE_KEYS.AGENT_MESSAGES, agentMessages);
  }, [agentMessages]);

  useEffect(() => {
    saveToStorage(STORAGE_KEYS.ACTIVITY_LOG, activityLog);
  }, [activityLog]);

  useEffect(() => {
    saveToStorage(STORAGE_KEYS.TOTAL_QUERIES, totalQueries);
  }, [totalQueries]);

  const onExampleQuestion = useCallback((q) => {
    setActiveTab('agent');
    if (typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent('cookierun_example_question', { detail: { q } }));
    }
  }, []);

  if (!auth) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-cookie-yellow/30 via-white to-cookie-orange/20 flex items-center justify-center">
        <div className="text-center">
          <div className="relative inline-block">
            <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-cookie-yellow to-cookie-orange shadow-lg flex items-center justify-center animate-bounce">
              <span className="text-5xl">🍪</span>
            </div>
            <div className="absolute -bottom-2 left-1/2 -translate-x-1/2 w-14 h-3 bg-cookie-brown/20 rounded-full blur-sm animate-pulse"></div>
          </div>
          <div className="mt-6 text-cookie-brown font-bold text-lg">로딩 중...</div>
          <div className="mt-2 flex justify-center gap-1">
            <span className="w-2 h-2 bg-amber-700 rounded-full animate-bounce [animation-delay:-0.3s]"></span>
            <span className="w-2 h-2 bg-amber-700 rounded-full animate-bounce [animation-delay:-0.15s]"></span>
            <span className="w-2 h-2 bg-amber-700 rounded-full animate-bounce"></span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <Layout
      auth={auth}
      cookies={cookies}
      kingdoms={kingdoms}
      selectedCookie={selectedCookie}
      setSelectedCookie={setSelectedCookie}
      exampleQuestions={EXAMPLE_QUESTIONS}
      onExampleQuestion={onExampleQuestion}
      onLogout={onLogout}
    >
      <div className="mb-4">
        <div className="flex items-center gap-3">
          <span className="text-4xl">🍪</span>
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-2xl font-bold text-cookie-brown">CookieRun AI Platform</h1>
              {settings?.selectedModel?.includes("mini") && (
                <span className="text-sm bg-amber-100 text-amber-700 px-3 py-1.5 rounded-full font-bold whitespace-nowrap">
                  ⚠️ GPT-4o → mini 전환으로 응답 품질이 다소 낮아질 수 있습니다
                </span>
              )}
            </div>
            <p className="text-sm text-cookie-brown/70">세계관 번역 · AI 에이전트 · 지식 검색 시스템</p>
          </div>
        </div>
        <div className="mt-2 flex items-center gap-2">
          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-bold bg-cookie-yellow/30 text-cookie-brown">
            GPT-4 기반
          </span>
          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-bold bg-cookie-orange/20 text-cookie-orange">
            DEVSISTERS
          </span>
        </div>
      </div>

      <Tabs tabs={tabs} active={activeTab} onChange={setActiveTab} />

      {activeTab === 'agent' ? (
        <ExampleQuestionBridge>
          <AgentPanel
            auth={auth}
            selectedCookie={selectedCookie}
            addLog={addLog}
            settings={settings}
            setSettings={setSettings}
            agentMessages={agentMessages}
            setAgentMessages={setAgentMessages}
            totalQueries={totalQueries}
            setTotalQueries={setTotalQueries}
            apiCall={apiCall}
          />
        </ExampleQuestionBridge>
      ) : null}

      {activeTab === 'dashboard' ? (
        <DashboardPanel auth={auth} selectedCookie={selectedCookie} apiCall={apiCall} />
      ) : null}

      {activeTab === 'analysis' ? <AnalysisPanel auth={auth} apiCall={apiCall} /> : null}

      {activeTab === 'models' && isAdmin ? <ModelsPanel auth={auth} apiCall={apiCall} /> : null}

      {activeTab === 'rag' && isAdmin ? <RagPanel auth={auth} apiCall={apiCall} addLog={addLog} settings={settings} setSettings={setSettings} /> : null}

      {activeTab === 'settings' && isAdmin ? (
        <SettingsPanel settings={settings} setSettings={setSettings} addLog={addLog} apiCall={apiCall} auth={auth} />
      ) : null}

      {activeTab === 'users' && isAdmin ? <UsersPanel auth={auth} apiCall={apiCall} /> : null}

      {activeTab === 'logs' && isAdmin ? (
        <LogsPanel activityLog={activityLog} clearLog={clearLog} />
      ) : null}
    </Layout>
  );
}

function ExampleQuestionBridge({ children }) {
  useEffect(() => {
    function handler(ev) {
      const q = ev?.detail?.q;
      if (!q) return;
      window.dispatchEvent(new CustomEvent('cookierun_send_question', { detail: { q } }));
    }
    window.addEventListener('cookierun_example_question', handler);
    return () => window.removeEventListener('cookierun_example_question', handler);
  }, []);

  return children;
}
