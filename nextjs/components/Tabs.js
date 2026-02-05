import { cn } from '@/lib/cn';

export default function Tabs({ tabs = [], active, onChange }) {
  return (
    <div className="mb-4">
      <div className="flex flex-wrap gap-2 rounded-3xl border-2 border-cookie-orange/20 bg-white/80 p-2 shadow-sm backdrop-blur">
        {tabs.map((t) => {
          const isActive = t.key === active;
          return (
            <button
              key={t.key}
              type="button"
              onClick={() => onChange(t.key)}
              className={cn(
                'rounded-2xl px-4 py-2 text-sm font-black transition active:translate-y-[1px]',
                isActive
                  ? 'bg-gradient-to-br from-cookie-yellow via-cookie-orange to-cookie-yellow text-cookie-brown shadow-cookie'
                  : 'bg-white/70 text-cookie-brown/60 hover:bg-cookie-yellow/20'
              )}
            >
              {t.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
