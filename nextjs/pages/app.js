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

// CookieRun AI Platform 예시 질문
const EXAMPLE_QUESTIONS = {
  '🌐 세계관 번역': [
    '"용감한 쿠키가 오븐에서 탈출했어요!" 영어로 번역해줘',
    '소울잼에 대한 설명을 일본어로 번역해줘',
    '"다크엔챈트리스 쿠키가 나타났다!" 중국어로 번역',
    '세계관 용어집 보여줘',
    '번역 품질 통계 확인',
  ],
  '🍪 쿠키 정보': [
    '용감한 쿠키 정보 알려줘',
    '에인션트 등급 쿠키 목록',
    '마법 타입 쿠키들 보여줘',
    '순수 바닐라 쿠키 스킬 설명',
    '레전더리 쿠키 전체 목록',
  ],
  '🏰 왕국 & 세계관': [
    '쿠키 왕국 정보',
    '다크카카오 왕국은 어떤 곳이야?',
    '소울잼이 뭐야?',
    '에인션트 쿠키들 설명해줘',
    '전체 왕국 목록 보여줘',
  ],
  '👤 유저 분석': [
    '유저 세그먼트 통계 보여줘',
    'U0001 유저 분석해줘',
    '하드코어 게이머 세그먼트 특징',
    '이상 행동 유저 탐지',
    '게임 이벤트 통계',
  ],
  '📊 대시보드': [
    '전체 현황 요약해줘',
    '번역 품질 현황',
    '유저 세그먼트 분포',
    '최근 게임 활동 통계',
  ],
};

const DEFAULT_SETTINGS = {
  apiKey: '',
  selectedModel: 'gpt-4o',
  maxTokens: 4000,
  systemPrompt: '',
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

  const [settings, setSettings] = useState(DEFAULT_SETTINGS);

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

  // 세션 초기 로드
  useEffect(() => {
    if (!router.isReady) return;

    const a = loadFromSession(STORAGE_KEYS.AUTH, null);
    if (!a?.username || !a?.password) {
      safeReplace('/login');
      return;
    }
    setAuth(a);

    const savedSettings = loadFromStorage(STORAGE_KEYS.SETTINGS, DEFAULT_SETTINGS);
    // API Key가 비어있으면 기본값 사용
    const mergedSettings = { ...DEFAULT_SETTINGS, ...(savedSettings || {}) };
    if (!mergedSettings.apiKey || mergedSettings.apiKey.trim() === '') {
      mergedSettings.apiKey = DEFAULT_SETTINGS.apiKey;
    }
    setSettings(mergedSettings);

    setAgentMessages(loadFromStorage(STORAGE_KEYS.AGENT_MESSAGES, []));
    setActivityLog(loadFromStorage(STORAGE_KEYS.ACTIVITY_LOG, []));
    setTotalQueries(loadFromStorage(STORAGE_KEYS.TOTAL_QUERIES, 0));
  }, [router.isReady, safeReplace]);

  // 시스템 프롬프트 로드
  useEffect(() => {
    if (!auth?.username || !auth?.password) return;

    const cur = settings?.systemPrompt ? String(settings.systemPrompt).trim() : '';
    if (cur.length > 0) return;

    let mounted = true;

    async function loadDefaultPrompt() {
      try {
        const res = await apiCall({
          endpoint: '/api/settings/default',
          method: 'GET',
          auth,
          timeoutMs: 30000,
        });

        if (!mounted) return;

        const prompt = res?.system_prompt || res?.data?.system_prompt || res?.data?.systemPrompt || '';
        const promptStr = String(prompt || '').trim();

        if (promptStr.length > 0) {
          setSettings((prev) => ({ ...prev, systemPrompt: promptStr }));
        }
      } catch (e) {}
    }

    loadDefaultPrompt();

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

  // 스토리지 저장
  useEffect(() => {
    saveToStorage(STORAGE_KEYS.SETTINGS, settings);
  }, [settings]);

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
      window.dispatchEvent(new CustomEvent('danal_example_question', { detail: { q } }));
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
            <h1 className="text-2xl font-bold text-cookie-brown">CookieRun AI Platform</h1>
            <p className="text-sm text-cookie-brown/70">세계관 번역 · 멀티 에이전트 · 지식 검색 시스템</p>
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

      {activeTab === 'rag' && isAdmin ? <RagPanel auth={auth} apiCall={apiCall} addLog={addLog} /> : null}

      {activeTab === 'settings' && isAdmin ? (
        <SettingsPanel settings={settings} setSettings={setSettings} addLog={addLog} />
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
      window.dispatchEvent(new CustomEvent('danal_send_question', { detail: { q } }));
    }
    window.addEventListener('danal_example_question', handler);
    return () => window.removeEventListener('danal_example_question', handler);
  }, []);

  return children;
}
