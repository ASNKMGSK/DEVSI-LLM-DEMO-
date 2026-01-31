// components/Sidebar.js
// CookieRun AI 플랫폼 사이드바
import { useEffect, useState } from 'react';
import {
  LogOut,
  ChevronDown,
  Cookie,
  Globe,
  Users,
  BarChart3,
  Search,
  MessageSquare,
  Sparkles,
  Crown,
  Languages,
  X
} from 'lucide-react';
import { AnimatePresence, motion } from 'framer-motion';

const SEEN_KEY = 'cookierun_seen_example_hint';

function SidebarContent({
  auth,
  cookies,
  kingdoms,
  selectedCookie,
  setSelectedCookie,
  exampleQuestions,
  onExampleQuestion,
  onLogout,
  onClose,
  isMobile,
}) {
  const [hintActive, setHintActive] = useState(false);
  const [openCats, setOpenCats] = useState({});

  // 카테고리별 스타일 (쿠키런 테마)
  const CAT_STYLES = [
    { card: 'from-amber-50/70 to-white/70 border-amber-200/70 border-l-amber-400', icon: Languages },
    { card: 'from-orange-50/70 to-white/70 border-orange-200/70 border-l-orange-400', icon: Cookie },
    { card: 'from-yellow-50/70 to-white/70 border-yellow-200/70 border-l-yellow-400', icon: Search },
    { card: 'from-rose-50/70 to-white/70 border-rose-200/70 border-l-rose-400', icon: Users },
    { card: 'from-indigo-50/70 to-white/70 border-indigo-200/70 border-l-indigo-400', icon: BarChart3 },
  ];

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const seen = window.localStorage.getItem(SEEN_KEY);
    setHintActive(!seen);
  }, []);

  useEffect(() => {
    if (!hintActive) return;
    const examples = exampleQuestions || {};
    const keys = Object.keys(examples);
    if (!keys.length) return;
    const next = {};
    for (const k of keys) next[k] = true;
    setOpenCats(next);
  }, [hintActive, exampleQuestions]);

  function markSeen() {
    if (typeof window === 'undefined') return;
    window.localStorage.setItem(SEEN_KEY, '1');
    setHintActive(false);
  }

  function clickExample(q) {
    markSeen();
    onExampleQuestion(q);
    if (isMobile) onClose?.();
  }

  function toggleCat(cat) {
    setOpenCats((prev) => ({ ...prev, [cat]: !prev?.[cat] }));
  }

  const accordionVariants = {
    open: {
      height: 'auto',
      opacity: 1,
      transition: { duration: 0.24, ease: 'easeOut', when: 'beforeChildren', staggerChildren: 0.03 },
    },
    closed: {
      height: 0,
      opacity: 0,
      transition: { duration: 0.18, ease: 'easeIn', when: 'afterChildren' },
    },
  };

  const itemVariants = {
    open: { opacity: 1, y: 0, transition: { duration: 0.18, ease: 'easeOut' } },
    closed: { opacity: 0, y: -6, transition: { duration: 0.12, ease: 'easeIn' } },
  };

  const examples = exampleQuestions || {};

  return (
    <div className={isMobile ? 'h-full overflow-auto px-4 py-5' : 'px-4 py-5'}>
      {/* 모바일 닫기 버튼 */}
      {isMobile && (
        <div className="flex justify-end mb-2">
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-cookie-orange/10 transition-colors"
          >
            <X className="w-5 h-5 text-cookie-brown" />
          </button>
        </div>
      )}

      {/* 로고 영역 */}
      <div className="pb-4 mb-4 border-b border-cookie-orange/20">
        <div className="flex items-start justify-between gap-2">
          <div className="inline-flex items-center gap-3">
            {/* 쿠키런 스타일 로고 */}
            <div className="h-12 w-12 rounded-2xl bg-gradient-to-br from-cookie-yellow via-cookie-orange to-cookie-yellow shadow-lg flex items-center justify-center">
              <Cookie className="w-7 h-7 text-white" />
            </div>
            <div>
              <h2 className="text-base font-black text-cookie-brown leading-tight">쿠키런 AI</h2>
              <p className="text-xs font-semibold text-cookie-orange">DEVSISTERS</p>
            </div>
          </div>
        </div>
      </div>

      {/* 기능 소개 배지 */}
      <div className="mb-4 flex flex-wrap gap-2">
        <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full bg-cookie-yellow/30 text-cookie-brown text-xs font-medium">
          <Globe className="w-3 h-3" /> 번역
        </span>
        <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full bg-cookie-orange/20 text-cookie-orange text-xs font-medium">
          <Search className="w-3 h-3" /> 검색
        </span>
        <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full bg-amber-100 text-amber-700 text-xs font-medium">
          <BarChart3 className="w-3 h-3" /> 분석
        </span>
      </div>

      {/* 사용자 정보 */}
      {auth?.username && (
        <div className="mb-4 p-3 rounded-xl bg-gradient-to-r from-cookie-yellow/20 to-cookie-orange/10 border border-cookie-orange/20">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-cookie-yellow to-cookie-orange flex items-center justify-center">
                <Users className="w-4 h-4 text-white" />
              </div>
              <div>
                <p className="text-sm font-bold text-cookie-brown">{auth.user_name || auth.username}</p>
                <p className="text-xs text-cookie-orange">{auth.user_role || '사용자'}</p>
              </div>
            </div>
            <button
              onClick={onLogout}
              className="p-2 rounded-lg hover:bg-cookie-orange/10 transition-colors"
              title="로그아웃"
            >
              <LogOut className="w-4 h-4 text-cookie-brown/60" />
            </button>
          </div>
        </div>
      )}

      {/* 예시 질문 섹션 */}
      <div className="space-y-3">
        <div className="flex items-center gap-2 mb-2">
          <Sparkles className="w-4 h-4 text-cookie-orange" />
          <span className="text-sm font-bold text-cookie-brown">이렇게 물어보세요</span>
          {hintActive && (
            <span className="ml-auto text-xs px-2 py-0.5 rounded-full bg-cookie-orange text-white animate-pulse">
              NEW
            </span>
          )}
        </div>

        {Object.entries(examples).map(([cat, questions], catIdx) => {
          const style = CAT_STYLES[catIdx % CAT_STYLES.length];
          const IconComponent = style.icon;
          const isOpen = openCats[cat];

          return (
            <div key={cat} className="rounded-xl overflow-hidden">
              {/* 카테고리 헤더 */}
              <button
                onClick={() => toggleCat(cat)}
                className={`w-full px-3 py-2.5 flex items-center justify-between bg-gradient-to-r ${style.card} border border-l-4 rounded-xl transition-all hover:shadow-sm`}
              >
                <div className="flex items-center gap-2">
                  <IconComponent className="w-4 h-4 text-cookie-brown/70" />
                  <span className="text-sm font-bold text-cookie-brown">{cat}</span>
                  <span className="text-xs text-cookie-brown/50">({questions.length})</span>
                </div>
                <ChevronDown
                  className={`w-4 h-4 text-cookie-brown/50 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`}
                />
              </button>

              {/* 질문 목록 */}
              <AnimatePresence initial={false}>
                {isOpen && (
                  <motion.div
                    initial="closed"
                    animate="open"
                    exit="closed"
                    variants={accordionVariants}
                    className="overflow-hidden"
                  >
                    <div className="pt-2 space-y-1.5">
                      {questions.map((q, idx) => (
                        <motion.button
                          key={idx}
                          variants={itemVariants}
                          onClick={() => clickExample(q)}
                          className="w-full text-left px-3 py-2 rounded-lg text-sm text-cookie-brown/80 hover:bg-cookie-yellow/20 hover:text-cookie-brown transition-colors border border-transparent hover:border-cookie-orange/20"
                        >
                          {q}
                        </motion.button>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          );
        })}
      </div>

      {/* 하단 정보 */}
      <div className="mt-6 pt-4 border-t border-cookie-orange/20">
        <div className="text-center">
          <p className="text-xs text-cookie-brown/50 mb-2">Powered by</p>
          <div className="flex items-center justify-center gap-2">
            <Crown className="w-4 h-4 text-cookie-orange" />
            <span className="text-sm font-bold bg-gradient-to-r from-cookie-orange to-cookie-yellow bg-clip-text text-transparent">
              DEVSISTERS
            </span>
          </div>
          <p className="text-xs text-cookie-brown/40 mt-1">ML Engineer Portfolio</p>
        </div>
      </div>
    </div>
  );
}

export default function Sidebar({
  auth,
  cookies,
  kingdoms,
  selectedCookie,
  setSelectedCookie,
  exampleQuestions,
  onExampleQuestion,
  onLogout,
  open,
  onClose,
}) {
  return (
    <>
      {/* 데스크탑 사이드바 */}
      <aside className="hidden xl:block sticky top-20 h-fit rounded-[32px] border-2 border-cookie-orange/10 bg-white/80 backdrop-blur-sm shadow-lg overflow-hidden">
        <SidebarContent
          auth={auth}
          cookies={cookies}
          kingdoms={kingdoms}
          selectedCookie={selectedCookie}
          setSelectedCookie={setSelectedCookie}
          exampleQuestions={exampleQuestions}
          onExampleQuestion={onExampleQuestion}
          onLogout={onLogout}
          isMobile={false}
        />
      </aside>

      {/* 모바일 사이드바 */}
      <AnimatePresence>
        {open && (
          <>
            {/* 오버레이 */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={onClose}
              className="fixed inset-0 bg-black/30 z-40 xl:hidden"
            />
            {/* 사이드바 */}
            <motion.aside
              initial={{ x: -320 }}
              animate={{ x: 0 }}
              exit={{ x: -320 }}
              transition={{ type: 'spring', damping: 25, stiffness: 200 }}
              className="fixed left-0 top-0 bottom-0 w-80 bg-gradient-to-b from-cookie-yellow/10 via-white to-cookie-orange/10 backdrop-blur-md z-50 xl:hidden shadow-2xl overflow-auto"
            >
              <SidebarContent
                auth={auth}
                cookies={cookies}
                kingdoms={kingdoms}
                selectedCookie={selectedCookie}
                setSelectedCookie={setSelectedCookie}
                exampleQuestions={exampleQuestions}
                onExampleQuestion={onExampleQuestion}
                onLogout={onLogout}
                onClose={onClose}
                isMobile={true}
              />
            </motion.aside>
          </>
        )}
      </AnimatePresence>
    </>
  );
}
