// Layout.js - CookieRun AI Platform
import { useMemo, useState } from 'react';
import Sidebar from '@/components/Sidebar';
import Topbar from '@/components/Topbar';
import { Noto_Sans_KR } from 'next/font/google';
import { FEATURED_COOKIES } from '@/lib/cookieImages';

const notoSansKr = Noto_Sans_KR({
  subsets: ['latin'],
  weight: ['400', '500', '600', '700'],
  display: 'swap',
});

export default function Layout({
  auth,
  cookies,
  kingdoms,
  selectedCookie,
  setSelectedCookie,
  exampleQuestions,
  onExampleQuestion,
  onLogout,
  children,
}) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [showWelcomePopup, setShowWelcomePopup] = useState(true); // 로그인 시 팝업 표시

  const username = useMemo(() => auth?.username || 'USER', [auth?.username]);

  return (
    <div
      className={`${notoSansKr.className} antialiased min-h-screen bg-gradient-to-br from-cookie-yellow/20 via-white to-cookie-orange/10`}
    >
      {/* 배경 장식 */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        {/* 그라데이션 블러 */}
        <div className="absolute top-20 left-20 w-40 h-40 bg-cookie-yellow/20 rounded-full blur-3xl"></div>
        <div className="absolute bottom-40 right-20 w-60 h-60 bg-cookie-orange/15 rounded-full blur-3xl"></div>
        <div className="absolute top-1/2 left-1/3 w-32 h-32 bg-cookie-yellow/25 rounded-full blur-2xl"></div>

        {/* 플로팅 쿠키 캐릭터 장식 */}
        {FEATURED_COOKIES.slice(0, 5).map((cookie, idx) => {
          const positions = [
            { top: '15%', right: '8%', size: 'w-16 h-16' },
            { top: '45%', left: '5%', size: 'w-14 h-14' },
            { bottom: '20%', right: '15%', size: 'w-12 h-12' },
            { top: '70%', left: '12%', size: 'w-10 h-10' },
            { top: '25%', left: '85%', size: 'w-12 h-12' },
          ];
          const pos = positions[idx];
          return (
            <div
              key={cookie.name}
              className={`absolute ${pos.size} opacity-30 cookie-float`}
              style={{
                top: pos.top,
                bottom: pos.bottom,
                left: pos.left,
                right: pos.right,
                animationDelay: `${idx * 0.5}s`,
              }}
            >
              <img
                src={cookie.image}
                alt={cookie.nameKr}
                className="w-full h-full object-contain drop-shadow-lg"
                onError={(e) => {
                  e.target.style.display = 'none';
                }}
              />
            </div>
          );
        })}
      </div>

      <Topbar
        username={username}
        onOpenSidebar={() => setSidebarOpen(true)}
        onLogout={onLogout}
      />

      <div className="relative z-10 mx-auto max-w-[1400px] px-3 sm:px-4">
        <div className="grid grid-cols-12 gap-4 pb-10 pt-3">
          <div className="col-span-12 xl:col-span-3 relative">
            <Sidebar
              auth={auth}
              cookies={cookies}
              kingdoms={kingdoms}
              selectedCookie={selectedCookie}
              setSelectedCookie={setSelectedCookie}
              exampleQuestions={exampleQuestions}
              onExampleQuestion={onExampleQuestion}
              onLogout={onLogout}
              open={sidebarOpen}
              onClose={() => setSidebarOpen(false)}
              showWelcomePopup={showWelcomePopup}
              onCloseWelcomePopup={() => setShowWelcomePopup(false)}
            />
          </div>

          <main className="col-span-12 xl:col-span-9">
            <div className="rounded-[32px] border-2 border-cookie-orange/10 bg-white/80 p-4 shadow-xl backdrop-blur-sm md:p-5">
              {children}
            </div>
          </main>
        </div>

      </div>
    </div>
  );
}
