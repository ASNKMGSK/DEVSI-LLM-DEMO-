// components/panels/AgentPanel.js
// CookieRun AI Platform - 에이전트 패널

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import toast from 'react-hot-toast';
import { motion } from 'framer-motion';
import EmptyState from '@/components/EmptyState';
import SectionHeader from '@/components/SectionHeader';
import { ArrowUpRight, Sparkles, Zap, Loader2, Cookie } from 'lucide-react';
import { fetchEventSource } from '@microsoft/fetch-event-source';

// CookieRun 테마 버튼 스타일 - 진한 브라운/오렌지
const cookieBtn =
  'w-full rounded-2xl border-2 border-amber-700/40 bg-gradient-to-r from-amber-700 via-amber-800 to-amber-700 px-4 py-3 text-sm font-extrabold text-white shadow-lg transition hover:from-amber-800 hover:via-amber-900 hover:to-amber-800 active:translate-y-[1px] disabled:opacity-60 disabled:cursor-not-allowed';

const cookieBtnSecondary =
  'w-full rounded-2xl border-2 border-cookie-brown/30 bg-cookie-beige px-4 py-3 text-sm font-extrabold text-cookie-brown shadow-sm transition hover:bg-cookie-brown/10 active:translate-y-[1px] disabled:opacity-60 disabled:cursor-not-allowed';

const cookieBtnInline =
  'rounded-2xl border-2 border-amber-700/40 bg-gradient-to-r from-amber-700 via-amber-800 to-amber-700 px-4 py-3 text-sm font-extrabold text-white shadow-lg transition hover:from-amber-800 hover:via-amber-900 hover:to-amber-800 active:translate-y-[1px] disabled:opacity-60 disabled:cursor-not-allowed inline-flex items-center justify-center gap-2 whitespace-nowrap';

const cookieBtnSecondaryInline =
  'rounded-2xl border-2 border-cookie-brown/30 bg-cookie-beige px-4 py-3 text-sm font-extrabold text-cookie-brown shadow-sm transition hover:bg-cookie-brown/10 active:translate-y-[1px] disabled:opacity-60 disabled:cursor-not-allowed inline-flex items-center justify-center gap-2 whitespace-nowrap';

const SEEN_KEY = 'cookierun_seen_example_hint';

const DEFAULT_FALLBACK_SYSTEM_PROMPT = [];

const WAITING_PLACEHOLDER = ['답변 생성 중입니다.', '잠시 기다려주세요.'].join('\n');

function basicAuthHeader(username, password) {
  return 'Basic ' + btoa(`${username}:${password}`);
}

function newMsgId() {
  return `${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function ToolCalls({ toolCalls }) {
  if (!toolCalls?.length) return null;
  return (
    <details className="details mt-2">
      <summary>도구 실행 결과</summary>
      <div className="mt-2 space-y-3">
        {toolCalls.map((tc, idx) => {
          const ok = tc?.result?.status === 'SUCCESS';
          return (
            <div
              key={idx}
              className="rounded-2xl border-2 border-cookie-orange/20 bg-white/80 p-3 shadow-sm backdrop-blur"
            >
              <div className="flex items-center justify-between">
                <div className="font-extrabold text-cookie-brown">{tc.tool}</div>
                <span className={ok ? 'badge badge-success' : 'badge badge-danger'}>
                  {ok ? '성공' : '실패'}
                </span>
              </div>
              <pre className="mt-2 overflow-auto rounded-xl bg-cookie-yellow/10 p-3 text-xs text-cookie-brown">
                {JSON.stringify(tc.result, null, 2)}
              </pre>
            </div>
          );
        })}
      </div>
    </details>
  );
}

function Chip({ label, onClick }) {
  return (
    <button
      className="inline-flex items-center gap-2 rounded-full border-2 border-cookie-orange/20 bg-white/80 px-3 py-1.5 text-xs font-extrabold text-cookie-brown hover:bg-cookie-yellow/20 hover:border-cookie-orange/40 hover:shadow-sm transition active:translate-y-[1px] whitespace-nowrap"
      onClick={onClick}
      title="클릭하면 질문이 바로 전송됩니다"
      type="button"
    >
      <Cookie size={14} className="text-cookie-orange" />
      <span className="max-w-[220px] truncate">{label}</span>
      <ArrowUpRight size={14} className="text-cookie-brown/50" />
    </button>
  );
}

function TypingDots() {
  return (
    <div className="flex items-center gap-1 py-1">
      <span className="h-2 w-2 rounded-full bg-cookie-orange animate-bounce [animation-delay:-0.2s]" />
      <span className="h-2 w-2 rounded-full bg-cookie-orange animate-bounce [animation-delay:-0.1s]" />
      <span className="h-2 w-2 rounded-full bg-cookie-orange animate-bounce" />
      <span className="ml-2 text-xs text-cookie-brown/60">답변 생성 중…</span>
    </div>
  );
}

function TopProgressBar({ active }) {
  if (!active) return null;
  return (
    <div className="mb-3 h-1 w-full overflow-hidden rounded-full bg-cookie-yellow/30">
      <div className="h-full w-1/3 animate-[danal_progress_1.2s_ease-in-out_infinite] bg-cookie-orange" />
    </div>
  );
}

function useRemarkGfm() {
  const [remarkGfm, setRemarkGfm] = useState(null);

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const mod = await import('remark-gfm');
        if (!mounted) return;
        setRemarkGfm(() => (mod?.default ? mod.default : mod));
      } catch (e) {
        if (!mounted) return;
        setRemarkGfm(null);
      }
    })();
    return () => {
      mounted = false;
    };
  }, []);

  return remarkGfm;
}

function MarkdownMessage({ content }) {
  const remarkGfm = useRemarkGfm();
  const remarkPlugins = useMemo(() => (remarkGfm ? [remarkGfm] : []), [remarkGfm]);

  return (
    <ReactMarkdown
      remarkPlugins={remarkPlugins}
      components={{
        table: ({ node, ...props }) => (
          <div className="overflow-x-auto -mx-1 my-2">
            <table className="w-full border-collapse" {...props} />
          </div>
        ),
        thead: ({ node, ...props }) => <thead className="bg-cookie-yellow/20" {...props} />,
        th: ({ node, ...props }) => (
          <th
            className="border-2 border-cookie-orange/20 px-3 py-2 text-left text-xs font-extrabold text-cookie-brown"
            {...props}
          />
        ),
        td: ({ node, ...props }) => (
          <td
            className="border border-cookie-orange/15 px-3 py-2 align-top text-xs text-cookie-brown whitespace-nowrap"
            {...props}
          />
        ),
        pre: ({ node, ...props }) => (
          <pre className="overflow-x-auto rounded-xl bg-cookie-yellow/10 p-3 text-xs text-cookie-brown" {...props} />
        ),
        code: ({ node, inline, className, children, ...props }) => {
          if (inline) {
            return (
              <code className="rounded bg-cookie-yellow/20 px-1 py-0.5 text-[11px] text-cookie-brown" {...props}>
                {children}
              </code>
            );
          }
          return (
            <code className={className} {...props}>
              {children}
            </code>
          );
        },
        a: ({ node, ...props }) => (
          <a
            {...props}
            target="_blank"
            rel="noopener noreferrer"
            className="font-extrabold text-cookie-orange underline underline-offset-2 hover:text-cookie-brown"
          />
        ),
      }}
    >
      {content || ''}
    </ReactMarkdown>
  );
}

export default function AgentPanel({
  auth,
  selectedCookie,
  addLog,
  settings,
  setSettings,
  agentMessages,
  setAgentMessages,
  totalQueries,
  setTotalQueries,
  apiCall,
}) {
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [quickResult, setQuickResult] = useState(null);

  const chatBoxRef = useRef(null);
  const scrollRef = useRef(null);

  const abortRef = useRef(null);
  const timeoutRef = useRef(null);

  const stoppedRef = useRef(false);
  const runIdRef = useRef(0);
  const activeAssistantIdRef = useRef(null);

  const canSend = useMemo(() => !!input?.trim() && !loading, [input, loading]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const seen = window.localStorage.getItem(SEEN_KEY);
    if (!seen) toast('왼쪽 예시 질문을 클릭하면 바로 분석이 시작됩니다', { icon: '🍪' });
  }, []);

  function markSeen() {
    if (typeof window === 'undefined') return;
    window.localStorage.setItem(SEEN_KEY, '1');
  }

  // CookieRun 관련 추천 질문 (데이터 분석 강화)
  const chips = useMemo(() => {
    const cookieId = selectedCookie || 'CK001';
    return [
      // 예측 분석
      '이탈 예측 분석 보여줘',
      '매출 예측 현황',
      '코호트 리텐션 분석',
      'KPI 트렌드 분석',
      // 데이터 분석
      '이상 행동 유저 탐지 현황',
      '유저 세그먼트 통계',
      '대시보드 전체 현황',
      // 쿠키 관련
      `${cookieId} 쿠키 정보 알려줘`,
      '에인션트 등급 쿠키 목록',
      // 세계관 정보
      '다크카카오 왕국 정보',
    ];
  }, [selectedCookie]);

  const shouldAutoScrollRef = useRef(true);

  const updateAutoScrollFlag = useCallback(() => {
    const el = chatBoxRef.current;
    if (!el) return;
    const threshold = 80;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    shouldAutoScrollRef.current = distanceFromBottom <= threshold;
  }, []);

  useEffect(() => {
    const el = chatBoxRef.current;
    if (!el) return;
    el.addEventListener('scroll', updateAutoScrollFlag, { passive: true });
    return () => el.removeEventListener('scroll', updateAutoScrollFlag);
  }, [updateAutoScrollFlag]);

  useEffect(() => {
    const el = chatBoxRef.current;
    if (!el) return;
    if (!shouldAutoScrollRef.current) return;
    el.scrollTop = el.scrollHeight;
  }, [agentMessages, loading]);

  const stopStream = useCallback(() => {
    setLoading(false);

    try {
      runIdRef.current += 1;
      stoppedRef.current = true;

      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }

      if (abortRef.current) {
        abortRef.current.abort();
        abortRef.current = null;
      }

      const aid = activeAssistantIdRef.current;

      setAgentMessages((prev) => {
        const arr = prev || [];

        let targetId = aid;
        if (!targetId) {
          const lastPending = [...arr].reverse().find((m) => m?.role === 'assistant' && m?._pending);
          targetId = lastPending?._id || null;
        }
        if (!targetId) return arr;

        const idx = arr.findIndex((m) => m?._id === targetId);
        if (idx < 0) return arr;

        const msg = arr[idx] || {};
        const content = String(msg.content || '').trim();
        const isPending = !!msg._pending;
        const isOnlyWaiting = content === String(WAITING_PLACEHOLDER).trim();

        if (!content || isPending || isOnlyWaiting) return arr.filter((m) => m?._id !== targetId);

        return arr.map((m) => {
          if (m?._id !== targetId) return m;
          const cur = String(m.content || '');
          return { ...m, content: cur + '\n\n[중단됨]', _pending: false };
        });
      });

      activeAssistantIdRef.current = null;
    } catch (e) {
      activeAssistantIdRef.current = null;
    } finally {
      setLoading(false);
    }
  }, [setAgentMessages]);

  const userKey = useMemo(() => String(auth?.username || '').trim(), [auth?.username]);
  const prevUserKeyRef = useRef(userKey);

  useEffect(() => {
    if (prevUserKeyRef.current === userKey) return;

    prevUserKeyRef.current = userKey;

    stopStream();
    setAgentMessages([]);
    setTotalQueries(0);
    setQuickResult(null);
    setInput('');
    setLoading(false);
  }, [userKey, stopStream, setAgentMessages, setTotalQueries]);

  const sendQuestion = useCallback(
    async (question) => {
      const q = String(question || '').trim();
      if (!q) return;

      markSeen();
      stopStream();

      stoppedRef.current = false;
      runIdRef.current += 1;
      const myRunId = runIdRef.current;

      setLoading(true);
      addLog('질문', q.slice(0, 30));

      const userMsg = { _id: newMsgId(), role: 'user', content: q };
      const assistantId = newMsgId();
      activeAssistantIdRef.current = assistantId;

      const assistantMsg = {
        _id: assistantId,
        role: 'assistant',
        content: WAITING_PLACEHOLDER,
        tool_calls: [],
        _pending: true,
      };

      setAgentMessages((prev) => [...(prev || []), userMsg, assistantMsg]);

      const systemPromptToSend =
        settings?.systemPrompt && String(settings.systemPrompt).trim().length > 0
          ? String(settings.systemPrompt)
          : DEFAULT_FALLBACK_SYSTEM_PROMPT;

      const username = auth?.username || '';
      const password = auth?.password || '';

      const ctrl = new AbortController();
      abortRef.current = ctrl;

      const timeoutMs = 60000;
      timeoutRef.current = setTimeout(() => {
        try {
          stoppedRef.current = true;
          ctrl.abort();
        } catch (e) {}
      }, timeoutMs);

      let deltaBuf = '';
      let flushTimer = null;

      const flushDelta = () => {
        if (!deltaBuf) return;
        const chunk = deltaBuf;
        deltaBuf = '';

        setAgentMessages((prev) =>
          (prev || []).map((m) => {
            if (m?._id !== assistantId) return m;

            const isPending = !!m?._pending;
            if (isPending) return { ...m, content: chunk, _pending: false };
            return { ...m, content: String(m.content || '') + chunk, _pending: false };
          })
        );
      };

      const isStale = () =>
        myRunId !== runIdRef.current ||
        stoppedRef.current ||
        ctrl.signal.aborted ||
        activeAssistantIdRef.current !== assistantId;

      try {
        await fetchEventSource(`/api/agent/stream`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Accept: 'text/event-stream',
            Authorization: basicAuthHeader(username, password),
          },
          body: JSON.stringify({
            user_input: q,
            cookie_id: selectedCookie || null,
            api_key: settings.apiKey || '',
            model: settings.selectedModel || 'gpt-4o',
            max_tokens: Number(settings.maxTokens ?? 4000),
            system_prompt: systemPromptToSend,
            debug: true,
          }),
          signal: ctrl.signal,

          async onopen(res) {
            if (isStale()) return;
            const ct = res.headers.get('content-type') || '';
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            if (!ct.includes('text/event-stream')) throw new Error('Not an SSE response');
          },

          onmessage(ev) {
            if (isStale()) return;

            let data = {};
            try {
              data = ev.data ? JSON.parse(ev.data) : {};
            } catch (e) {
              return;
            }

            if (ev.event === 'delta') {
              const delta = String(data.delta || '');
              if (!delta) return;

              deltaBuf += delta;

              if (!flushTimer) {
                flushTimer = setTimeout(() => {
                  flushTimer = null;
                  if (isStale()) return;
                  flushDelta();
                }, 50);
              }
              return;
            }

            if (ev.event === 'done') {
              if (isStale()) return;

              if (flushTimer) {
                clearTimeout(flushTimer);
                flushTimer = null;
              }
              flushDelta();

              const ok = !!data.ok;
              const finalText = String(data.final || '');
              const toolCalls = Array.isArray(data.tool_calls) ? data.tool_calls : [];

              setAgentMessages((prev) =>
                (prev || []).map((m) => {
                  if (m?._id !== assistantId) return m;
                  return {
                    ...m,
                    content: finalText || String(m.content || ''),
                    tool_calls: toolCalls,
                    _pending: false,
                  };
                })
              );

              setTotalQueries((prev) => (prev || 0) + 1);
              setLoading(false);

              if (timeoutRef.current) {
                clearTimeout(timeoutRef.current);
                timeoutRef.current = null;
              }
              abortRef.current = null;
              activeAssistantIdRef.current = null;

              if (ok) toast.success('분석 완료');
              else toast.error('요청 실패: 백엔드/네트워크를 확인하세요');
              return;
            }

            if (ev.event === 'error') {
              if (isStale()) return;

              if (flushTimer) {
                clearTimeout(flushTimer);
                flushTimer = null;
              }
              flushDelta();

              const msg = data?.message ? String(data.message) : '스트리밍 오류';

              setAgentMessages((prev) =>
                (prev || []).map((m) => {
                  if (m?._id !== assistantId) return m;
                  const cur = String(m.content || '');
                  return { ...m, content: cur + `\n\n[오류]\n${msg}`, _pending: false };
                })
              );

              toast.error(msg);
              return;
            }
          },

          onerror(err) {
            throw err;
          },

          onclose() {
            if (isStale()) return;
            throw new Error('SSE closed');
          },
        });
      } catch (e) {
        if (isStale()) {
          setLoading(false);
          return;
        }

        if (flushTimer) {
          clearTimeout(flushTimer);
          flushTimer = null;
        }
        flushDelta();

        const msg = String(e || '요청 실패');

        setAgentMessages((prev) =>
          (prev || []).map((m) => {
            if (m?._id !== assistantId) return m;
            const cur = String(m.content || '');
            return { ...m, content: cur + `\n\n[오류]\n${msg}`, _pending: false };
          })
        );

        setLoading(false);
        toast.error('요청 실패');
      } finally {
        if (flushTimer) {
          clearTimeout(flushTimer);
          flushTimer = null;
        }

        if (timeoutRef.current) {
          clearTimeout(timeoutRef.current);
          timeoutRef.current = null;
        }
        abortRef.current = null;

        if (activeAssistantIdRef.current === assistantId) {
          activeAssistantIdRef.current = null;
        }
      }
    },
    [addLog, auth, settings, setAgentMessages, setTotalQueries, stopStream, selectedCookie]
  );

  useEffect(() => {
    function handler(ev) {
      const q = ev?.detail?.q;
      if (!q) return;
      sendQuestion(q);
    }
    window.addEventListener('danal_send_question', handler);
    return () => window.removeEventListener('danal_send_question', handler);
  }, [sendQuestion]);

  async function runQuick(endpoint, method = 'GET', payload = null) {
    setQuickResult(null);

    const res = await apiCall({
      endpoint,
      method,
      auth,
      data: payload,
      timeoutMs: 60000,
    });

    setQuickResult(res);
    addLog('빠른분석', endpoint);
  }

  return (
    <div className="grid grid-cols-12 gap-4">
      <div className="col-span-12 xl:col-span-9">
        <SectionHeader
          title="AI 에이전트"
          subtitle="GPT + ML 기반 쿠키런 분석"
          right={<span className="badge">쿼리 {totalQueries || 0}</span>}
        />

        <div className="card">
          <div ref={chatBoxRef} className="max-h-[62vh] md:max-h-[70vh] overflow-auto pr-1">
            {(agentMessages || []).map((m, idx) => {
              const isUser = m.role === 'user';
              const isPending = !!m?._pending;

              return (
                <motion.div
                  key={m?._id || idx}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.18 }}
                  className={isUser ? 'flex justify-end mb-3' : 'flex justify-start mb-3'}
                >
                  <div
                    className={
                      isUser
                        ? 'chat-bubble chat-bubble-user w-full md:max-w-[78%]'
                        : 'chat-bubble chat-bubble-ai w-full md:max-w-[78%]'
                    }
                  >
                    <div className="text-[11px] font-extrabold text-cookie-brown/60 mb-2 flex items-center justify-between">
                      <span>{isUser ? auth?.username || 'USER' : 'COOKIERUN AI'}</span>

                      {!isUser && isPending ? (
                        <span className="inline-flex items-center gap-2 text-cookie-orange">
                          <span className="h-3 w-3 rounded-full border-2 border-cookie-yellow border-t-cookie-orange animate-spin" />
                          <span className="text-[10px]">streaming</span>
                        </span>
                      ) : null}
                    </div>

                    <div className="prose prose-sm max-w-none">
                      {!isUser && isPending ? <TypingDots /> : <MarkdownMessage content={m.content || ''} />}
                    </div>

                    <ToolCalls toolCalls={m.tool_calls} />
                  </div>
                </motion.div>
              );
            })}

            {!agentMessages?.length ? (
              <EmptyState
                title="대화를 시작해보세요"
                desc="왼쪽 예시 질문을 누르거나 아래 추천 질문을 클릭하면 바로 시작됩니다."
              />
            ) : null}

            <div ref={scrollRef} />
          </div>

          <div className="mt-3 flex flex-wrap gap-2">
            {chips.map((c) => (
              <Chip
                key={c}
                label={c}
                onClick={() => {
                  sendQuestion(c);
                  setInput('');
                }}
              />
            ))}
          </div>

          <div className="mt-3 flex flex-col md:flex-row gap-2">
            <input
              className="input"
              placeholder="질문 입력 (Enter로 전송)"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && canSend) {
                  sendQuestion(input);
                  setInput('');
                }
              }}
            />

            <button
              className={`${cookieBtnInline} w-[140px]`}
              onClick={() => {
                sendQuestion(input);
                setInput('');
              }}
              disabled={!canSend}
              type="button"
            >
              {loading ? <Loader2 size={16} className="animate-spin" /> : <Zap size={16} />}
              {loading ? '분석중...' : '전송'}
            </button>

            <button
              className={`${cookieBtnSecondaryInline} w-[140px]`}
              onClick={() => {
                stopStream();
                toast('중단됨');
              }}
              disabled={!loading}
              title="스트리밍 중단"
              type="button"
            >
              중단
            </button>
          </div>
        </div>
      </div>

      <div className="col-span-12 xl:col-span-3">
        <div className="card">
          <div className="card-header">빠른 분석</div>
          <div className="text-sm text-cookie-brown/70 mb-3">
            쿠키런 AI 도구 호출
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-1 gap-2">
            <button
              className={cookieBtn}
              onClick={() => runQuick('/api/cookies')}
              type="button"
            >
              쿠키 목록
            </button>
            <button
              className={cookieBtn}
              onClick={() => runQuick('/api/kingdoms')}
              type="button"
            >
              왕국 목록
            </button>
            <button
              className={cookieBtn}
              onClick={() => runQuick('/api/translate/terms')}
              type="button"
            >
              번역 용어집
            </button>
            <button
              className={cookieBtn}
              onClick={() => runQuick('/api/users/segments/statistics')}
              type="button"
            >
              세그먼트 통계
            </button>
          </div>

          <div className="mt-3">
            <button className={cookieBtnSecondary} onClick={() => setAgentMessages([])} type="button">
              대화 초기화
            </button>
          </div>

          {quickResult ? (
            <pre className="mt-3 max-h-[45vh] overflow-auto rounded-2xl bg-cookie-yellow/10 p-3 text-xs text-cookie-brown">
              {JSON.stringify(quickResult, null, 2)}
            </pre>
          ) : (
            <div className="mt-3 text-xs text-cookie-brown/60">버튼을 클릭하면 API 호출 결과를 확인할 수 있어요.</div>
          )}
        </div>

        <div className="card mt-4">
          <div className="card-header">LLM 설정 요약</div>
          <div className="text-sm text-cookie-brown/70 space-y-1">
            <div>
              <span className="text-cookie-brown/50">모델</span>: <span className="font-mono">{settings.selectedModel}</span>
            </div>
            <div>
              <span className="text-cookie-brown/50">Max Tokens</span>: <span className="font-mono">{settings.maxTokens}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
