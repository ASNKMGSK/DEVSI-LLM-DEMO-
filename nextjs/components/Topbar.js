import { LogOut, Menu } from 'lucide-react';
import { COOKIE_IMAGES } from '@/lib/cookieImages';

export default function Topbar({ username, onOpenSidebar, onLogout }) {
  return (
    <header className="sticky top-0 z-40">
      <div className="mx-auto max-w-[1320px] px-3 sm:px-4">
        <div className="mt-3 rounded-3xl border-2 border-cookie-orange/20 bg-white/80 px-3 py-2 shadow-lg backdrop-blur">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={onOpenSidebar}
                className="inline-flex items-center justify-center rounded-2xl border-2 border-cookie-orange/20 bg-white/80 p-2 text-cookie-brown shadow-sm hover:bg-cookie-yellow/20 active:translate-y-[1px] xl:hidden"
                aria-label="Open menu"
              >
                <Menu size={18} />
              </button>

              <div className="flex items-center gap-2 group cursor-pointer">
                <div className="h-9 w-9 rounded-2xl bg-gradient-to-br from-cookie-yellow via-cookie-orange to-cookie-yellow shadow-cookie flex items-center justify-center overflow-hidden transition-transform duration-300 group-hover:scale-110 group-hover:rotate-12">
                  <img
                    src={COOKIE_IMAGES['GingerBrave']}
                    alt="GingerBrave"
                    className="w-8 h-8 object-contain"
                    onError={(e) => {
                      e.target.style.display = 'none';
                      e.target.parentElement.innerHTML = '🍪';
                    }}
                  />
                </div>
                <div>
                  <div className="text-xs font-extrabold tracking-wide cookie-text">
                    CookieRun AI Platform
                  </div>
                  <div className="text-[11px] font-semibold text-cookie-orange/80">
                    {username}
                  </div>
                </div>
              </div>
            </div>

            <button
              type="button"
              onClick={onLogout}
              className="inline-flex items-center gap-2 rounded-2xl border-2 border-cookie-orange/20 bg-white/80 px-3 py-2 text-xs font-extrabold text-cookie-brown shadow-sm hover:bg-cookie-yellow/20 active:translate-y-[1px]"
              title="로그아웃"
            >
              <LogOut size={16} />
              로그아웃
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}
