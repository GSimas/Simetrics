import type { ReactNode } from 'react';
import type { LucideIcon } from 'lucide-react';

import { InfoTip } from '@/components/InfoTip';

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

/**
 * Na linguagem Scientata os cartões são planos (fundo do cartão + linha fina); o tom só
 * colore o ícone e o filete de destaque que aparece ao passar o mouse.
 */
const TONE_ICON: Record<KpiTone, string> = {
  default: 'text-highlight',
  accent: 'text-highlight',
  blue: 'text-blue-600 dark:text-blue-400',
  purple: 'text-purple-600 dark:text-purple-400',
  emerald: 'text-emerald-600 dark:text-emerald-400',
  amber: 'text-amber-600 dark:text-amber-400',
  indigo: 'text-indigo-600 dark:text-indigo-400',
  cyan: 'text-cyan-600 dark:text-cyan-400',
  warning: 'text-amber-600 dark:text-amber-400',
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
  const formatted =
    typeof value === 'number'
      ? value.toLocaleString('pt-BR', { maximumFractionDigits: 2 })
      : value;

  const iconColor = TONE_ICON[tone] ?? TONE_ICON.default;

  return (
    <div
      className={cn(
        'group relative overflow-visible border bg-card p-4 transition-colors duration-200 hover:border-highlight/60',
        tone === 'warning' && 'border-amber-500/50',
        className,
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex min-h-8 min-w-0 items-center gap-1.5">
          <p className="eyebrow truncate">
            {title}
          </p>
          {info && <InfoTip label={title}>{info}</InfoTip>}
        </div>

        {Icon && (
          <div
            className={cn(
              'grid size-8 shrink-0 place-items-center border border-border transition-colors duration-200 group-hover:border-current',
              iconColor,
            )}
          >
            <Icon className="size-4" aria-hidden />
          </div>
        )}
      </div>

      <p className="mt-3 text-[1.75rem] font-medium tabular-nums leading-none tracking-[-0.04em] text-foreground">
        {formatted}
      </p>
      {subtitle && <p className="mt-2 text-xs text-muted-foreground/90">{subtitle}</p>}
    </div>
  );
}
