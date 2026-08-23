import type { LucideIcon } from 'lucide-react';

import { cn } from '@/lib/utils';

/**
 * Cartão de indicador — ⇄ `create_kpi_card` (Geral.py:57).
 *
 * No Streamlit isso era HTML cru com estilos embutidos, injetado via
 * `unsafe_allow_html`. Aqui vira um componente, e as cores saem dos tokens do tema em
 * vez de hexadecimais fixos — o que faz o modo escuro funcionar sem duplicação.
 */

export interface KpiCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  Icon?: LucideIcon;
  /** Destaca o cartão quando o indicador merece atenção. */
  tone?: 'default' | 'accent' | 'warning';
  className?: string;
}

const TONES = {
  default: 'bg-card',
  accent: 'bg-primary/5 border-primary/20',
  warning: 'bg-amber-500/5 border-amber-500/20',
} as const;

export function KpiCard({
  title,
  value,
  subtitle,
  Icon,
  tone = 'default',
  className,
}: KpiCardProps) {
  const formatted =
    typeof value === 'number' ? value.toLocaleString('pt-BR', { maximumFractionDigits: 2 }) : value;

  return (
    <div className={cn('rounded-lg border p-4 shadow-sm', TONES[tone], className)}>
      <div className="flex items-start justify-between gap-2">
        <p className="text-xs font-medium text-muted-foreground">{title}</p>
        {Icon && <Icon className="size-4 shrink-0 text-muted-foreground" aria-hidden />}
      </div>
      <p className="mt-2 text-2xl font-bold tabular-nums leading-none">{formatted}</p>
      {subtitle && <p className="mt-1.5 text-xs text-muted-foreground">{subtitle}</p>}
    </div>
  );
}
