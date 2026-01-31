// Layout.js - CookieRun AI Platform
import { useMemo, useState } from 'react';
import Sidebar from '@/components/Sidebar';
import Topbar from '@/components/Topbar';
import { Noto_Sans_KR } from 'next/font/google';

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

  const username = useMemo(() => auth?.username || 'USER', [auth?.username]);

  return (
    <div
      className={`${notoSansKr.className} antialiased min-h-screen bg-gradient-to-br from-cookie-yellow/20 via-white to-cookie-orange/10`}
    >
      {/* 배경 장식 */}
      <div className="pointer-events-none fixed inset-0">
        <div className="absolute top-20 left-20 w-40 h-40 bg-cookie-yellow/20 rounded-full blur-3xl"></div>
        <div className="absolute bottom-40 right-20 w-60 h-60 bg-cookie-orange/15 rounded-full blur-3xl"></div>
        <div className="absolute top-1/2 left-1/3 w-32 h-32 bg-cookie-yellow/25 rounded-full blur-2xl"></div>
      </div>

      <Topbar
        username={username}
        onOpenSidebar={() => setSidebarOpen(true)}
        onLogout={onLogout}
      />

      <div className="relative z-10 mx-auto max-w-[1400px] px-3 sm:px-4">
        <div className="grid grid-cols-12 gap-4 pb-10 pt-3">
          <div className="col-span-12 xl:col-span-3">
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
