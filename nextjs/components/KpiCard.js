import React from 'react';
import { cn } from '@/lib/cn';

export default function KpiCard({
  title,
  value,
  subtitle,
  icon = null,
  tone = 'yellow', // yellow | orange | cream | green | blue | pink
  className = '',
}) {
  const toneMap = {
    yellow: 'from-cookie-yellow/20 to-cookie-orange/10 border-cookie-yellow/40',
    orange: 'from-cookie-orange/20 to-cookie-yellow/10 border-cookie-orange/40',
    cream: 'from-cookie-cream to-white border-cookie-orange/20',
    green: 'from-emerald-50 to-teal-50 border-emerald-200/70',
    blue: 'from-sky-50 to-cyan-50 border-sky-200/70',
    pink: 'from-rose-50 to-pink-50 border-rose-200/70',
  };

  return (
    <div
      className={cn(
        'rounded-3xl border-2 bg-gradient-to-br p-4 shadow-[0_10px_30px_-18px_rgba(110,76,30,0.25)] backdrop-blur',
        toneMap[tone] || toneMap.yellow,
        className
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-[11px] font-extrabold tracking-wide text-cookie-brown/70">{title}</div>
          <div className="mt-1 text-2xl font-black text-cookie-brown">{value}</div>
          {subtitle ? <div className="mt-1 text-xs font-semibold text-cookie-brown/60">{subtitle}</div> : null}
        </div>

        {icon ? (
          <div className="shrink-0 rounded-2xl border-2 border-cookie-orange/20 bg-white/70 p-2 shadow-sm">
            {icon}
          </div>
        ) : null}
      </div>
    </div>
  );
}
