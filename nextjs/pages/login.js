import { useEffect, useState } from 'react';
import { useRouter } from 'next/router';
import { motion } from 'framer-motion';
import { apiCall } from '@/lib/api';
import { saveToSession, loadFromSession, STORAGE_KEYS } from '@/lib/storage';
import { User, Lock, ChevronDown } from 'lucide-react';

export default function LoginPage() {
  const router = useRouter();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState('');
  const [showAccounts, setShowAccounts] = useState(false);

  useEffect(() => {
    const auth = loadFromSession(STORAGE_KEYS.AUTH, null);
    if (auth?.username && auth?.password) router.replace('/app');
  }, [router]);

  async function onLogin() {
    setErr('');
    setLoading(true);

    const res = await apiCall({
      endpoint: '/api/login',
      method: 'POST',
      auth: { username, password },
      timeoutMs: 30000,
    });

    setLoading(false);

    if (res?.status === 'SUCCESS') {
      const auth = {
        username,
        password,
        user_name: res.user_name,
        user_role: res.user_role,
      };
      saveToSession(STORAGE_KEYS.AUTH, auth);
      router.replace('/app');
    } else {
      setErr('아이디 또는 비밀번호가 틀렸습니다');
    }
  }

  function fillAccount(user, pass) {
    setUsername(user);
    setPassword(pass);
  }

  const accounts = [
    { label: '관리자', user: 'admin', pass: 'admin123', role: 'Admin' },
    { label: '번역가', user: 'translator', pass: 'trans123', role: 'Translator' },
    { label: '분석가', user: 'analyst', pass: 'analyst123', role: 'Analyst' },
    { label: '사용자', user: 'user', pass: 'user123', role: 'User' },
  ];

  return (
    <div className="min-h-screen flex items-center justify-center px-4 bg-[#F8F7F4]">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: 'easeOut' }}
        className="w-full max-w-sm"
      >
        {/* 헤더 */}
        <div className="text-center mb-8">
          <motion.div
            initial={{ scale: 0.8 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.1, type: 'spring', stiffness: 200 }}
            className="mb-4"
          >
            <span className="text-6xl">🍪</span>
          </motion.div>
          <h1 className="text-xl font-semibold text-cookie-brown">CookieRun AI Platform</h1>
          <p className="text-sm text-cookie-brown/60 mt-1">세계관 번역 · 멀티 에이전트 · 지식 검색</p>
          <div className="mt-3 inline-flex items-center gap-1.5 bg-cookie-beige px-3 py-1 rounded-full">
            <span className="text-xs font-medium text-cookie-brown/70">DEVSISTERS</span>
          </div>
        </div>

        {/* 로그인 카드 */}
        <div className="bg-white rounded-2xl border border-[rgba(92,74,61,0.08)] shadow-soft p-6">
          <div className="space-y-4">
            {/* 아이디 입력 */}
            <div>
              <label className="text-sm font-medium text-cookie-brown/80 mb-1.5 block">아이디</label>
              <div className="relative">
                <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-cookie-brown/40" />
                <input
                  className="w-full pl-10 pr-4 py-2.5 rounded-xl border border-[rgba(92,74,61,0.12)] bg-white text-sm text-cookie-brown placeholder:text-cookie-brown/40 outline-none transition-all focus:border-cookie-orange focus:ring-2 focus:ring-cookie-orange/10"
                  placeholder="아이디를 입력하세요"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  autoComplete="username"
                />
              </div>
            </div>

            {/* 비밀번호 입력 */}
            <div>
              <label className="text-sm font-medium text-cookie-brown/80 mb-1.5 block">비밀번호</label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-cookie-brown/40" />
                <input
                  className="w-full pl-10 pr-4 py-2.5 rounded-xl border border-[rgba(92,74,61,0.12)] bg-white text-sm text-cookie-brown placeholder:text-cookie-brown/40 outline-none transition-all focus:border-cookie-orange focus:ring-2 focus:ring-cookie-orange/10"
                  placeholder="비밀번호를 입력하세요"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  autoComplete="current-password"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && username && password) onLogin();
                  }}
                />
              </div>
            </div>

            {/* 에러 메시지 */}
            {err && (
              <motion.div
                initial={{ opacity: 0, y: -8 }}
                animate={{ opacity: 1, y: 0 }}
                className="rounded-lg bg-red-50 border border-red-100 px-3 py-2 text-sm text-red-600"
              >
                {err}
              </motion.div>
            )}

            {/* 로그인 버튼 */}
            <button
              onClick={onLogin}
              disabled={loading || !username || !password}
              className="w-full py-2.5 rounded-xl bg-gradient-to-r from-amber-700 to-amber-800 text-white font-semibold text-sm shadow-lg transition-all hover:from-amber-800 hover:to-amber-900 hover:-translate-y-0.5 active:translate-y-0 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:translate-y-0"
            >
              {loading ? (
                <span className="inline-flex items-center gap-2">
                  <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  로그인 중...
                </span>
              ) : (
                '로그인'
              )}
            </button>

            {/* 테스트 계정 토글 */}
            <div className="pt-2">
              <button
                onClick={() => setShowAccounts(!showAccounts)}
                className="w-full flex items-center justify-between px-3 py-2 rounded-lg hover:bg-cookie-beige transition-colors text-sm text-cookie-brown/70"
              >
                <span className="font-medium">테스트 계정</span>
                <ChevronDown className={`w-4 h-4 transition-transform ${showAccounts ? 'rotate-180' : ''}`} />
              </button>

              {showAccounts && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="mt-2 space-y-1.5"
                >
                  {accounts.map((acc) => (
                    <button
                      key={acc.user}
                      onClick={() => fillAccount(acc.user, acc.pass)}
                      className="w-full flex items-center justify-between px-3 py-2 rounded-lg border border-transparent hover:border-cookie-orange/20 hover:bg-cookie-light transition-all text-left group"
                    >
                      <div>
                        <span className="text-sm font-medium text-cookie-brown">{acc.label}</span>
                        <span className="text-xs text-cookie-brown/50 ml-2">{acc.user}</span>
                      </div>
                      <span className="text-[10px] px-2 py-0.5 rounded-full bg-cookie-beige text-cookie-brown/60 group-hover:bg-cookie-orange/10 group-hover:text-cookie-orange transition-colors">
                        {acc.role}
                      </span>
                    </button>
                  ))}
                </motion.div>
              )}
            </div>
          </div>
        </div>

        {/* 푸터 */}
        <p className="mt-6 text-center text-xs text-cookie-brown/40">
          © 2024 DEVSISTERS · CookieRun AI Platform
        </p>
      </motion.div>
    </div>
  );
}
