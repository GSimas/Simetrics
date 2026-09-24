import { useState, type KeyboardEvent, type ReactNode, type RefObject } from 'react';
import { RotateCcw } from 'lucide-react';

import { ExpandChartButton } from '@/components/charts/ExpandChartButton';
import { CHART_BUTTON_CLASS, ExportImageButton } from '@/components/charts/ExportImageButton';
import { imageFromSvgElement } from '@/lib/export-image';
import { cn } from '@/lib/utils';
import { useLocale } from '@/state/locale.store';
import type { TooltipState } from './hooks';
import { LEGEND_FONT, type LegendLayout } from './legend';
import { FONT_SANS } from './text';

/**
 * Componentes comuns dos gráficos SVG: a dica flutuante, a legenda desenhada dentro do
 * SVG (para sair na exportação) e a moldura com os botões de ampliar e exportar. Hooks,
 * legenda e texto ficam em `hooks.tsx`, `legend.ts` e `text.ts`.
 */

function Tooltip({ state, width }: { state: TooltipState; width: number }) {
  // Na metade direita a dica abre para a esquerda do cursor, para não sair do gráfico.
  const flipX = state.x > width * 0.55;
  return (
    <div
      className="pointer-events-none absolute z-20 max-w-72 border border-border bg-popover/95 px-3 py-2 text-xs shadow-lg backdrop-blur-sm"
      style={{
        left: state.x,
        top: state.y,
        transform: `translate(${flipX ? 'calc(-100% - 14px)' : '14px'}, calc(-50%))`,
      }}
    >
      {state.content}
    </div>
  );
}

/** Linha "rótulo: valor" padronizada das dicas. */
export function TipRow({ label, value, color }: { label: ReactNode; value: ReactNode; color?: string | undefined }) {
  return (
    <p className="flex items-center gap-2 text-muted-foreground">
      {color && <span className="size-2 shrink-0 rounded-full" style={{ background: color }} />}
      <span className="min-w-0 flex-1 truncate">{label}</span>
      <span className="tabular-nums text-foreground">{value}</span>
    </p>
  );
}

/**
 * Legenda desenhada no SVG. Com `onToggle`, cada item é um botão de alternância também
 * por teclado (Tab + Enter/Espaço) — o SVG que a contém usa `role="figure"`, que (ao
 * contrário de `img`) deixa os controles internos acessíveis.
 */
export function SvgLegend({
  layout,
  x,
  y,
  onToggle,
}: {
  layout: LegendLayout;
  x: number;
  y: number;
  onToggle?: (key: string) => void;
}) {
  const t = useLocale((state) => state.t);
  const [focused, setFocused] = useState<string | null>(null);
  return (
    <g transform={`translate(${x} ${y})`} fontFamily={FONT_SANS} fontSize={LEGEND_FONT}>
      {layout.items.map(({ item, x: ix, y: iy, width, text }) => (
        <g
          key={item.key}
          transform={`translate(${ix} ${iy})`}
          opacity={item.muted ? 0.35 : 1}
          className={onToggle ? 'cursor-pointer outline-none' : undefined}
          onClick={onToggle ? () => onToggle(item.key) : undefined}
          {...(onToggle
            ? {
                role: 'button',
                tabIndex: 0,
                'aria-pressed': !item.muted,
                'aria-label': `${t('chart_legend_toggle')}: ${item.label}`,
                onKeyDown: (event: KeyboardEvent<SVGGElement>) => {
                  if (event.key !== 'Enter' && event.key !== ' ') return;
                  event.preventDefault();
                  onToggle(item.key);
                },
                onFocus: () => setFocused(item.key),
                onBlur: () => setFocused(null),
              }
            : {})}
        >
          <title>{item.label}</title>
          {/* Anel de foco: SVG não desenha o outline do navegador em <g>. */}
          {focused === item.key && (
            <rect
              x={-3}
              y={-3}
              width={width + 6}
              height={20}
              rx={3}
              fill="none"
              stroke="var(--ring)"
              strokeWidth={1.5}
            />
          )}
          {item.line ? (
            <line x1={0} x2={14} y1={7} y2={7} stroke={item.color} strokeWidth={2.5} strokeDasharray={item.dash} />
          ) : (
            <rect x={2} y={2} width={10} height={10} rx={2} fill={item.color} />
          )}
          <text x={18} y={7} dy="0.35em" fill="var(--foreground)">
            {text}
          </text>
        </g>
      ))}
    </g>
  );
}

export interface ChartFrameProps {
  exportName: string;
  svgRef: RefObject<SVGSVGElement | null>;
  containerRef: RefObject<HTMLDivElement | null>;
  width: number;
  tooltip: TooltipState | null;
  /** Instância ampliada do mesmo gráfico; sem ela, o botão de ampliar não aparece. */
  expandedContent?: ReactNode | undefined;
  /** Desfaz o zoom; o botão só aparece quando há zoom a desfazer. */
  onResetZoom?: (() => void) | undefined;
  /** Controles extras na barra, à esquerda dos botões. */
  toolbar?: ReactNode;
  className?: string | undefined;
  children: ReactNode;
}

export function ChartFrame({
  exportName,
  svgRef,
  containerRef,
  width,
  tooltip,
  expandedContent,
  onResetZoom,
  toolbar,
  className,
  children,
}: ChartFrameProps) {
  const t = useLocale((state) => state.t);
  return (
    <div className={cn('min-w-0 space-y-2', className)}>
      <div className="flex flex-wrap items-center justify-end gap-1.5">
        {toolbar && <div className="mr-auto flex min-w-0 flex-wrap items-center gap-2">{toolbar}</div>}
        {onResetZoom && (
          <button
            type="button"
            onClick={onResetZoom}
            title={t('chart_zoom_reset')}
            aria-label={t('chart_zoom_reset')}
            className={CHART_BUTTON_CLASS}
          >
            <RotateCcw className="size-4" aria-hidden />
          </button>
        )}
        {expandedContent && <ExpandChartButton>{expandedContent}</ExpandChartButton>}
        <ExportImageButton
          filename={exportName}
          getImage={() => (svgRef.current ? imageFromSvgElement(svgRef.current) : null)}
        />
      </div>
      <div ref={containerRef} className="relative min-w-0 w-full">
        {children}
        {tooltip && <Tooltip state={tooltip} width={width} />}
      </div>
    </div>
  );
}
