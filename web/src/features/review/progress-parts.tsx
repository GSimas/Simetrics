import { Check } from 'lucide-react';

import type { StepProgress } from '@/core/review/progress';
import { cn } from '@/lib/utils';

/** Anel de progresso; completo, vira um check aceso. */
export function ProgressRing({ progress, className }: { progress: StepProgress; className?: string }) {
  const done = progress.state === 'done';
  const radius = 7;
  const circumference = 2 * Math.PI * radius;
  return (
    <span className={cn('relative inline-grid size-4.5 shrink-0 place-items-center', className)} aria-hidden>
      <svg viewBox="0 0 18 18" className="absolute inset-0 -rotate-90">
        <circle cx="9" cy="9" r={radius} fill="none" strokeWidth="2" className="stroke-border" />
        <circle
          cx="9"
          cy="9"
          r={radius}
          fill="none"
          strokeWidth="2"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={circumference * (1 - progress.ratio)}
          className={cn(
            'transition-[stroke-dashoffset] duration-500 ease-out',
            progress.state === 'off' || progress.state === 'waiting' ? 'stroke-muted-foreground/40' : 'stroke-highlight',
          )}
        />
      </svg>
      {done && (
        <span className="grid size-4.5 place-items-center rounded-full bg-highlight text-ink animate-in zoom-in-50 duration-300">
          <Check className="size-3" strokeWidth={3} />
        </span>
      )}
    </span>
  );
}
