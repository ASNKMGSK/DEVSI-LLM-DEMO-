import { Cookie } from 'lucide-react';

export default function EmptyState({ title = '데이터가 없습니다', desc = '조건을 바꿔 다시 시도해보세요.' }) {
  return (
    <div className="rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-6 shadow-sm backdrop-blur">
      <div className="flex items-center gap-3">
        <div className="rounded-2xl border-2 border-cookie-orange/20 bg-gradient-to-br from-cookie-yellow/30 via-cookie-orange/20 to-cookie-yellow/30 p-3 shadow-sm">
          <Cookie className="text-cookie-brown" size={18} />
        </div>
        <div>
          <div className="text-sm font-black text-cookie-brown">{title}</div>
          <div className="text-xs font-semibold text-cookie-brown/60">{desc}</div>
        </div>
      </div>
    </div>
  );
}
