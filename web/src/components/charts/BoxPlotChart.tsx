import { useMemo, useRef, useState } from 'react';

import { expandedHeight } from '@/components/charts/ExpandChartButton';
import { ChartFrame, TipRow } from '@/components/charts/svg/chart-kit';
import { useElementWidth, useTooltip } from '@/components/charts/svg/hooks';
import { fitText, FONT_MONO, FONT_SANS, measureText } from '@/components/charts/svg/text';
import {
  formatNumber,
  formatTick,
  linearScale,
  linearTicks,
  logScale,
  logTicks,
  niceDomain,
  seededRandom,
} from '@/components/charts/svg/scale';
import { useLocale } from '@/state/locale.store';
import { withChartBoundary } from '@/components/with-chart-boundary';

/**
 * Boxplot em SVG com todos os pontos visíveis (jitter), no lugar do traço `box` do
 * Plotly. Caixa = 1º a 3º quartil, traço = mediana, bigodes = último valor dentro de
 * 1,5 × IQR; o que passa disso aparece como ponto isolado — os outliers.
 */
export interface BoxSeries {
  name: string;
  color: string;
  values: readonly number[];
  /** Rótulo de cada valor (o documento, o autor…), mostrado na dica do ponto. */
  labels?: readonly string[];
}

export interface BoxPlotChartProps {
  series: readonly BoxSeries[];
  yLabel: string;
  log?: boolean;
  height?: number;
  exportName: string;
  onSeriesClick?: ((name: string) => void) | undefined;
  expanded?: boolean;
  className?: string;
}

interface BoxStats {
  n: number;
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  mean: number;
  lowWhisker: number;
  highWhisker: number;
}

/** Quantil com interpolação linear — o mesmo método padrão do Plotly e do NumPy. */
function quantile(sorted: readonly number[], p: number): number {
  if (sorted.length === 0) return 0;
  const position = (sorted.length - 1) * p;
  const low = Math.floor(position);
  const high = Math.ceil(position);
  const a = sorted[low] as number;
  const b = sorted[high] as number;
  return a + (b - a) * (position - low);
}

function stats(values: readonly number[]): BoxStats | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const q1 = quantile(sorted, 0.25);
  const q3 = quantile(sorted, 0.75);
  const iqr = q3 - q1;
  const inside = sorted.filter((value) => value >= q1 - 1.5 * iqr && value <= q3 + 1.5 * iqr);
  return {
    n: sorted.length,
    min: sorted[0] as number,
    max: sorted[sorted.length - 1] as number,
    q1,
    q3,
    median: quantile(sorted, 0.5),
    mean: sorted.reduce((sum, value) => sum + value, 0) / sorted.length,
    lowWhisker: inside[0] ?? q1,
    highWhisker: inside[inside.length - 1] ?? q3,
  };
}

function BoxPlotChart(props: BoxPlotChartProps) {
  const { series, yLabel, log = false, height = 440, exportName, onSeriesClick, expanded, className } = props;
  const t = useLocale((state) => state.t);
  const locale = useLocale((state) => state.locale);
  const svgRef = useRef<SVGSVGElement>(null);
  const [containerRef, width] = useElementWidth<HTMLDivElement>();
  const { tooltip, show, hide } = useTooltip(containerRef);
  const [hovered, setHovered] = useState<string | null>(null);

  // Na escala log, zero e negativos não têm posição: ficam fora, como no Plotly.
  const prepared = useMemo(
    () =>
      series.map((entry) => {
        const points = entry.values
          .map((value, index) => ({ value, label: entry.labels?.[index] ?? '' }))
          .filter((point) => Number.isFinite(point.value) && (!log || point.value > 0));
        return { ...entry, points, stats: stats(points.map((point) => point.value)) };
      }),
    [series, log],
  );

  const all = prepared.flatMap((entry) => entry.points.map((point) => point.value));
  const dataMin = all.length ? Math.min(...all) : 0;
  const dataMax = all.length ? Math.max(...all) : 1;

  const yDomain: [number, number] = log
    ? [10 ** Math.floor(Math.log10(Math.max(dataMin, 1e-9))), 10 ** Math.ceil(Math.log10(Math.max(dataMax, 1e-9)))]
    : niceDomain(Math.min(0, dataMin), dataMax, 5);
  if (log && yDomain[0] === yDomain[1]) yDomain[1] = yDomain[0] * 10;
  const yTicks = log ? logTicks(yDomain[0], yDomain[1]) : linearTicks(yDomain[0], yDomain[1], 5);
  const tickWidth = Math.max(...yTicks.map((value) => measureText(formatTick(value, locale), 11, FONT_MONO)), 10);

  const left = Math.round(tickWidth + 40);
  const right = 16;
  const plotWidth = Math.max(60, width - left - right);
  const band = plotWidth / Math.max(1, prepared.length);
  // Rótulos longos giram quando não cabem deitados na faixa de cada caixa.
  const longest = Math.max(...prepared.map((entry) => measureText(entry.name, 11)), 0);
  const rotate = longest > band - 8;
  const labelSpace = rotate ? Math.min(120, longest * 0.72 + 12) : 22;
  const top = 12;
  const plotHeight = Math.max(100, height - top - labelSpace - 34);
  const sy = (log ? logScale : linearScale)(yDomain, [top + plotHeight, top]);
  const cx = (index: number): number => left + band * (index + 0.5);
  const boxWidth = Math.min(90, band * 0.5);

  // Resumo para leitores de tela: mediana e tamanho de cada grupo.
  const summary = prepared
    .filter((entry) => entry.stats)
    .map((entry) =>
      t('chart_summary_box')
        .replace('{name}', entry.name)
        .replace('{median}', formatNumber(entry.stats?.median ?? 0, undefined, locale))
        .replace('{count}', String(entry.stats?.n ?? 0)),
    )
    .join('; ');

  return (
    <ChartFrame
      exportName={exportName}
      svgRef={svgRef}
      containerRef={containerRef}
      width={width}
      tooltip={tooltip}
      className={className}
      expandedContent={
        expanded ? undefined : <BoxPlotChart {...props} expanded height={Math.max(height, expandedHeight())} />
      }
    >
      <svg
        ref={svgRef}
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        className="block select-none"
        role="img"
        aria-label={`${t('visual_tab_boxplot')} — ${yLabel}`}
        xmlns="http://www.w3.org/2000/svg"
        fontFamily={FONT_SANS}
        onPointerLeave={() => {
          setHovered(null);
          hide();
        }}
      >
        <desc>{summary}</desc>
        <g fontFamily={FONT_MONO} fontSize={11} fill="var(--muted-foreground)" aria-hidden>
          {yTicks.map((value) => (
            <g key={`y${value}`}>
              <line x1={left} x2={left + plotWidth} y1={sy(value)} y2={sy(value)} stroke="var(--border)" strokeWidth={0.6} />
              <text x={left - 8} y={sy(value)} dy="0.32em" textAnchor="end">
                {formatTick(value, locale)}
              </text>
            </g>
          ))}
        </g>
        <text
          transform={`translate(14 ${top + plotHeight / 2}) rotate(-90)`}
          textAnchor="middle"
          fontSize={12}
          fill="var(--foreground)"
        >
          {yLabel}
        </text>

        {prepared.map((entry, index) => {
          const s = entry.stats;
          const x = cx(index);
          const dimmed = hovered !== null && hovered !== entry.name;
          const random = seededRandom(index + 1);
          const label = rotate ? fitText(entry.name, 120, 11) : fitText(entry.name, band - 6, 11);
          const describe = (event: { clientX: number; clientY: number }): void => {
            if (!s) return;
            setHovered(entry.name);
            show(
              event,
              <>
                <p className="mb-1 font-semibold">{entry.name}</p>
                <TipRow label="n" value={formatNumber(s.n, undefined, locale)} />
                <TipRow label={locale === 'en' ? 'Maximum' : 'Máximo'} value={formatNumber(s.max, undefined, locale)} />
                <TipRow label={locale === 'en' ? '3rd quartile' : '3º quartil'} value={formatNumber(s.q3, undefined, locale)} />
                <TipRow label={locale === 'en' ? 'Median' : 'Mediana'} value={formatNumber(s.median, undefined, locale)} />
                <TipRow label={locale === 'en' ? 'Mean' : 'Média'} value={formatNumber(s.mean, undefined, locale)} />
                <TipRow label={locale === 'en' ? '1st quartile' : '1º quartil'} value={formatNumber(s.q1, undefined, locale)} />
                <TipRow label={locale === 'en' ? 'Minimum' : 'Mínimo'} value={formatNumber(s.min, undefined, locale)} />
              </>,
            );
          };
          return (
            <g key={entry.name} opacity={dimmed ? 0.35 : 1} className="transition-opacity duration-200">
              {s && (
                <g
                  stroke={entry.color}
                  className={onSeriesClick ? 'cursor-pointer' : undefined}
                  onPointerMove={describe}
                  onClick={onSeriesClick ? () => onSeriesClick(entry.name) : undefined}
                >
                  <line x1={x} x2={x} y1={sy(s.highWhisker)} y2={sy(s.q3)} strokeWidth={1.5} />
                  <line x1={x} x2={x} y1={sy(s.q1)} y2={sy(s.lowWhisker)} strokeWidth={1.5} />
                  <line x1={x - boxWidth / 4} x2={x + boxWidth / 4} y1={sy(s.highWhisker)} y2={sy(s.highWhisker)} strokeWidth={1.5} />
                  <line x1={x - boxWidth / 4} x2={x + boxWidth / 4} y1={sy(s.lowWhisker)} y2={sy(s.lowWhisker)} strokeWidth={1.5} />
                  <rect
                    x={x - boxWidth / 2}
                    y={sy(s.q3)}
                    width={boxWidth}
                    height={Math.max(1, sy(s.q1) - sy(s.q3))}
                    fill={entry.color}
                    fillOpacity={0.18}
                    strokeWidth={1.5}
                  />
                  <line x1={x - boxWidth / 2} x2={x + boxWidth / 2} y1={sy(s.median)} y2={sy(s.median)} strokeWidth={2.5} />
                </g>
              )}
              {entry.points.map((point, pointIndex) => (
                <circle
                  key={pointIndex}
                  cx={x + (random() - 0.5) * boxWidth * 0.9}
                  cy={sy(point.value)}
                  r={3}
                  fill={entry.color}
                  fillOpacity={0.6}
                  className={onSeriesClick ? 'cursor-pointer' : undefined}
                  onPointerMove={(event) => {
                    event.stopPropagation();
                    setHovered(entry.name);
                    show(
                      event,
                      <>
                        {point.label && <p className="mb-1 font-semibold break-words">{point.label}</p>}
                        <TipRow label={entry.name} value={formatNumber(point.value, undefined, locale)} color={entry.color} />
                      </>,
                    );
                  }}
                  onClick={onSeriesClick ? () => onSeriesClick(entry.name) : undefined}
                />
              ))}
              <text
                transform={
                  rotate
                    ? `translate(${x} ${top + plotHeight + 12}) rotate(-35)`
                    : `translate(${x} ${top + plotHeight + 18})`
                }
                textAnchor={rotate ? 'end' : 'middle'}
                fontSize={11}
                fill="var(--foreground)"
                className={onSeriesClick ? 'cursor-pointer' : undefined}
                onClick={onSeriesClick ? () => onSeriesClick(entry.name) : undefined}
              >
                <title>{entry.name}</title>
                {label}
              </text>
            </g>
          );
        })}
      </svg>
    </ChartFrame>
  );
}

// Um gráfico que quebre com dado atípico não derruba a aba (ver with-chart-boundary).
export default withChartBoundary(BoxPlotChart);
