import { useMemo, useRef, useState, type PointerEvent } from 'react';

import { ExpandChartButton } from '@/components/charts/ExpandChartButton';
import { ExportImageButton } from '@/components/charts/ExportImageButton';
import type { LotkaDistribution } from '@/core/scientometrics';
import { PALETTE } from '@/features/overview/viz-shared';
import { imageFromSvgElement } from '@/lib/export-image';
import { cn } from '@/lib/utils';
import { useLocale } from '@/state/locale.store';
import { withChartBoundary } from '@/components/with-chart-boundary';

/**
 * Lei de Lotka em SVG: proporção de autores (eixo Y) por número de artigos publicados
 * (eixo X), observada contra a curva teórica c/x². Passar o mouse mostra os dois valores
 * no número de artigos mais próximo.
 */
export interface LotkaChartProps {
  lotka: LotkaDistribution;
  exportName?: string;
  className?: string;
  /** Dentro da janela ampliada: sem o botão de ampliar e com altura da tela. */
  expanded?: boolean;
}

const WIDTH = 800;
const HEIGHT = 420;
const MARGIN = { top: 20, right: 24, bottom: 56, left: 68 };
const PLOT_W = WIDTH - MARGIN.left - MARGIN.right;
const PLOT_H = HEIGHT - MARGIN.top - MARGIN.bottom;

/** Passo "redondo" (1, 2, 5 × 10ⁿ) para cerca de `count` marcas até `max`. */
function niceStep(max: number, count: number): number {
  const raw = max / Math.max(1, count);
  const power = 10 ** Math.floor(Math.log10(raw));
  const unit = raw / power;
  return (unit <= 1 ? 1 : unit <= 2 ? 2 : unit <= 5 ? 5 : 10) * power;
}

const ticks = (max: number, step: number, start = 0): number[] => {
  const values: number[] = [];
  for (let value = start; value <= max + step * 1e-9; value += step) values.push(Number(value.toFixed(10)));
  return values;
};

const percent = (value: number): string =>
  `${(value * 100).toLocaleString('pt-BR', { maximumFractionDigits: value < 0.01 ? 2 : 1 })}%`;

function LotkaChart(props: LotkaChartProps) {
  const { lotka, exportName = 'lei-de-lotka', className, expanded } = props;
  const t = useLocale((state) => state.t);
  const svgRef = useRef<SVGSVGElement>(null);
  const [hovered, setHovered] = useState<number | null>(null);

  const chart = useMemo(() => {
    const maxX = Math.max(1, ...lotka.theoretical.map((point) => point.articles), ...lotka.observed.map((point) => point.articles));
    const maxY = Math.max(...lotka.theoretical.map((point) => point.frequency), ...lotka.observed.map((point) => point.frequency), 0.01);
    const yStep = niceStep(maxY, 5);
    const yMax = Math.ceil(maxY / yStep) * yStep;
    // Eixo X de 1 ao máximo; com um único valor, a escala ganha folga para não dividir por zero.
    const xMin = 1;
    const xMax = Math.max(2, maxX);
    const sx = (x: number): number => MARGIN.left + ((x - xMin) / (xMax - xMin)) * PLOT_W;
    const sy = (y: number): number => MARGIN.top + PLOT_H - (y / yMax) * PLOT_H;
    const line = (points: LotkaDistribution['observed']): string =>
      points.map((point, index) => `${index ? 'L' : 'M'}${sx(point.articles).toFixed(1)},${sy(point.frequency).toFixed(1)}`).join('');
    const xStep = Math.max(1, niceStep(xMax - xMin, 10));
    return {
      xMin,
      xMax,
      sx,
      sy,
      observedPath: line(lotka.observed),
      theoreticalPath: line(lotka.theoretical),
      // A marca 0 vira 1: o eixo começa em quem publicou um artigo.
      xTicks: ticks(xMax, xStep).map((value) => Math.max(value, xMin)),
      yTicks: ticks(yMax, yStep),
      observedBy: new Map(lotka.observed.map((point) => [point.articles, point.frequency])),
      theoreticalBy: new Map(lotka.theoretical.map((point) => [point.articles, point.frequency])),
    };
  }, [lotka]);

  const handlePointerMove = (event: PointerEvent<SVGSVGElement>): void => {
    const ctm = svgRef.current?.getScreenCTM();
    if (!ctm) return;
    const { x } = new DOMPoint(event.clientX, event.clientY).matrixTransform(ctm.inverse());
    const articles = Math.round(chart.xMin + ((x - MARGIN.left) / PLOT_W) * (chart.xMax - chart.xMin));
    setHovered(Math.min(chart.xMax, Math.max(chart.xMin, articles)));
  };

  const observed = hovered !== null ? chart.observedBy.get(hovered) : undefined;
  const theoretical = hovered !== null ? chart.theoreticalBy.get(hovered) : undefined;
  const series = [
    { label: t('lotka_observed'), color: PALETTE[0], dash: undefined },
    { label: t('lotka_theoretical'), color: PALETTE[1], dash: '7 5' },
  ];

  return (
    <div className={cn('space-y-3', className)}>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap gap-x-4 gap-y-1.5">
          {series.map((item) => (
            <span key={item.label} className="flex items-center gap-2 text-sm">
              <svg width="22" height="4" aria-hidden>
                <line x1="0" y1="2" x2="22" y2="2" stroke={item.color} strokeWidth="2.5" strokeDasharray={item.dash} />
              </svg>
              {item.label}
            </span>
          ))}
        </div>
        <div className="flex gap-1.5">
          {!expanded && (
            <ExpandChartButton>
              <LotkaChart {...props} expanded />
            </ExpandChartButton>
          )}
          <ExportImageButton
            filename={exportName}
            getImage={() => (svgRef.current ? imageFromSvgElement(svgRef.current) : null)}
          />
        </div>
      </div>

      <div className="relative">
        {hovered !== null && (
          <div
            className="pointer-events-none absolute top-2 z-10 -translate-x-1/2 border border-border bg-popover/95 px-3 py-2 text-xs shadow-lg animate-in fade-in-0"
            style={{ left: `${Math.min(88, Math.max(12, (chart.sx(hovered) / WIDTH) * 100))}%` }}
          >
            <p className="mb-1 font-semibold">
              {hovered} {hovered === 1 ? t('lotka_article') : t('lotka_articles')}
            </p>
            <p className="text-muted-foreground">
              {t('lotka_observed')}:{' '}
              <span className="tabular-nums text-foreground">{observed !== undefined ? percent(observed) : '—'}</span>
            </p>
            <p className="text-muted-foreground">
              {t('lotka_theoretical')}:{' '}
              <span className="tabular-nums text-foreground">{theoretical !== undefined ? percent(theoretical) : '—'}</span>
            </p>
          </div>
        )}

        <svg
          ref={svgRef}
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          className={cn('block w-full select-none', expanded ? 'h-[74dvh]' : 'h-auto')}
          role="img"
          aria-label={t('lotka_title')}
          xmlns="http://www.w3.org/2000/svg"
          fontFamily="Manrope, system-ui, sans-serif"
          onPointerMove={handlePointerMove}
          onPointerLeave={() => setHovered(null)}
        >
          <g fontFamily='"DM Mono", ui-monospace, monospace' fontSize={11} fill="var(--muted-foreground)">
            {chart.yTicks.map((value) => (
              <g key={`y${value}`}>
                <line x1={MARGIN.left} x2={WIDTH - MARGIN.right} y1={chart.sy(value)} y2={chart.sy(value)} stroke="var(--border)" strokeWidth={value === 0 ? 1 : 0.6} />
                <text x={MARGIN.left - 10} y={chart.sy(value)} dy="0.32em" textAnchor="end">
                  {percent(value)}
                </text>
              </g>
            ))}
            {chart.xTicks.map((value) => (
              <text key={`x${value}`} x={chart.sx(value)} y={HEIGHT - MARGIN.bottom + 20} textAnchor="middle">
                {value}
              </text>
            ))}
          </g>

          <text x={MARGIN.left + PLOT_W / 2} y={HEIGHT - 10} textAnchor="middle" fontSize={12} fill="var(--foreground)">
            {t('lotka_x_axis')}
          </text>
          <text
            transform={`translate(16 ${MARGIN.top + PLOT_H / 2}) rotate(-90)`}
            textAnchor="middle"
            fontSize={12}
            fill="var(--foreground)"
          >
            {t('lotka_y_axis')}
          </text>

          {hovered !== null && (
            <line x1={chart.sx(hovered)} x2={chart.sx(hovered)} y1={MARGIN.top} y2={MARGIN.top + PLOT_H} stroke="var(--muted-foreground)" strokeDasharray="3 3" />
          )}

          <path d={chart.theoreticalPath} fill="none" stroke={PALETTE[1]} strokeWidth={2} strokeDasharray="7 5" />
          <path d={chart.observedPath} fill="none" stroke={PALETTE[0]} strokeWidth={2.5} strokeLinejoin="round" />
          {lotka.observed.map((point) => (
            <circle
              key={point.articles}
              cx={chart.sx(point.articles)}
              cy={chart.sy(point.frequency)}
              r={hovered === point.articles ? 5 : 3}
              fill={PALETTE[0]}
              stroke="var(--background)"
              strokeWidth={1}
              className="transition-[r] duration-150"
            />
          ))}

          {/* Área de captura do mouse sobre todo o gráfico. */}
          <rect x={MARGIN.left} y={MARGIN.top} width={PLOT_W} height={PLOT_H} fill="transparent" />
        </svg>
      </div>
    </div>
  );
}

// Um gráfico que quebre com dado atípico não derruba a aba (ver with-chart-boundary).
export default withChartBoundary(LotkaChart);
