import { useState, type ReactNode } from 'react';
import { Info, type LucideIcon } from 'lucide-react';

import { cn } from '@/lib/utils';

export type KpiTone =
  | 'default'
  | 'accent'
  | 'blue'
  | 'purple'
  | 'emerald'
  | 'amber'
  | 'indigo'
  | 'cyan'
  | 'warning';

export interface KpiCardProps {
  title: string;
  value: string | number | ReactNode;
  subtitle?: string;
  Icon?: LucideIcon;
  info?: string;
  /** Define a paleta de cores do cartão para enriquecer a visualização. */
  tone?: KpiTone;
  className?: string;
}

const TONE_STYLES: Record<
  KpiTone,
  { container: string; iconBg: string; iconColor: string; valueColor: string }
> = {
  default: {
    container: 'bg-card border-border/80 hover:border-primary/40',
    iconBg: 'bg-primary/10',
    iconColor: 'text-primary',
    valueColor: 'text-foreground',
  },
  accent: {
    container:
      'bg-gradient-to-br from-blue-500/[0.08] via-card to-card border-blue-200/80 dark:border-blue-900/50 hover:border-blue-400',
    iconBg: 'bg-blue-100 dark:bg-blue-950/80',
    iconColor: 'text-blue-600 dark:text-blue-400',
    valueColor: 'text-foreground',
  },
  blue: {
    container:
      'bg-gradient-to-br from-blue-500/[0.08] via-card to-card border-blue-200/80 dark:border-blue-900/50 hover:border-blue-400',
    iconBg: 'bg-blue-100 dark:bg-blue-950/80',
    iconColor: 'text-blue-600 dark:text-blue-400',
    valueColor: 'text-foreground',
  },
  purple: {
    container:
      'bg-gradient-to-br from-purple-500/[0.08] via-card to-card border-purple-200/80 dark:border-purple-900/50 hover:border-purple-400',
    iconBg: 'bg-purple-100 dark:bg-purple-950/80',
    iconColor: 'text-purple-600 dark:text-purple-400',
    valueColor: 'text-foreground',
  },
  emerald: {
    container:
      'bg-gradient-to-br from-emerald-500/[0.08] via-card to-card border-emerald-200/80 dark:border-emerald-900/50 hover:border-emerald-400',
    iconBg: 'bg-emerald-100 dark:bg-emerald-950/80',
    iconColor: 'text-emerald-600 dark:text-emerald-400',
    valueColor: 'text-foreground',
  },
  amber: {
    container:
      'bg-gradient-to-br from-amber-500/[0.08] via-card to-card border-amber-200/80 dark:border-amber-900/50 hover:border-amber-400',
    iconBg: 'bg-amber-100 dark:bg-amber-950/80',
    iconColor: 'text-amber-600 dark:text-amber-400',
    valueColor: 'text-foreground',
  },
  indigo: {
    container:
      'bg-gradient-to-br from-indigo-500/[0.08] via-card to-card border-indigo-200/80 dark:border-indigo-900/50 hover:border-indigo-400',
    iconBg: 'bg-indigo-100 dark:bg-indigo-950/80',
    iconColor: 'text-indigo-600 dark:text-indigo-400',
    valueColor: 'text-foreground',
  },
  cyan: {
    container:
      'bg-gradient-to-br from-cyan-500/[0.08] via-card to-card border-cyan-200/80 dark:border-cyan-900/50 hover:border-cyan-400',
    iconBg: 'bg-cyan-100 dark:bg-cyan-950/80',
    iconColor: 'text-cyan-600 dark:text-cyan-400',
    valueColor: 'text-foreground',
  },
  warning: {
    container:
      'bg-gradient-to-br from-amber-500/[0.12] via-card to-card border-amber-300 dark:border-amber-900 hover:border-amber-400',
    iconBg: 'bg-amber-100 dark:bg-amber-950',
    iconColor: 'text-amber-700 dark:text-amber-400',
    valueColor: 'text-foreground',
  },
};

export function KpiCard({
  title,
  value,
  subtitle,
  Icon,
  info,
  tone = 'default',
  className,
}: KpiCardProps) {
  const [showTooltip, setShowTooltip] = useState(false);

  const formatted =
    typeof value === 'number'
      ? value.toLocaleString('pt-BR', { maximumFractionDigits: 2 })
      : value;

  const style = TONE_STYLES[tone] ?? TONE_STYLES.default;

  return (
    <div
      className={cn(
        'group relative overflow-visible rounded-xl border p-4 shadow-xs transition-all duration-200 hover:-translate-y-0.5 hover:shadow-md',
        style.container,
        className,
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-1.5 min-w-0">
          <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground truncate">
            {title}
          </p>
          {info && (
            <div
              className="relative inline-flex items-center"
              onMouseEnter={() => setShowTooltip(true)}
              onMouseLeave={() => setShowTooltip(false)}
            >
              <button
                type="button"
                className="text-muted-foreground/70 transition-colors hover:text-foreground focus:outline-hidden"
                title={info}
                aria-label={info}
                onClick={(e) => {
                  e.stopPropagation();
                  setShowTooltip((prev) => !prev);
                }}
              >
                <Info className="size-3.5" />
              </button>

              {showTooltip && (
                <div className="pointer-events-none absolute left-1/2 bottom-full mb-2 z-50 -translate-x-1/2 w-56 rounded-lg border border-border/90 bg-popover p-2.5 text-[11px] font-normal normal-case leading-snug text-popover-foreground shadow-xl backdrop-blur-xs animate-in fade-in-0 zoom-in-95">
                  <p>{info}</p>
                </div>
              )}
            </div>
          )}
        </div>

        {Icon && (
          <div
            className={cn(
              'grid size-8 shrink-0 place-items-center rounded-lg transition-transform duration-200 group-hover:scale-110',
              style.iconBg,
              style.iconColor,
            )}
          >
            <Icon className="size-4" aria-hidden />
          </div>
        )}
      </div>

      <p className={cn('mt-2 text-2xl font-bold tabular-nums leading-none tracking-tight', style.valueColor)}>
        {formatted}
      </p>
      {subtitle && <p className="mt-2 text-xs text-muted-foreground/90">{subtitle}</p>}
    </div>
  );
}
