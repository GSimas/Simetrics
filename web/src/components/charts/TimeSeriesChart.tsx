import { useMemo, useRef, useState, type PointerEvent } from 'react';

import { expandedHeight } from '@/components/charts/ExpandChartButton';
import { ChartFrame, SvgLegend, TipRow } from '@/components/charts/svg/chart-kit';
import {
  useBrush,
  useElementWidth,
  useTooltip,
  type PlotArea,
} from '@/components/charts/svg/hooks';
import { layoutLegend } from '@/components/charts/svg/legend';
import { FONT_MONO, FONT_SANS, measureText } from '@/components/charts/svg/text';
import { formatNumber, formatTick, linearScale, linearTicks, niceDomain } from '@/components/charts/svg/scale';
import { cn } from '@/lib/utils';
import { useLocale } from '@/state/locale.store';
import { withChartBoundary } from '@/components/with-chart-boundary';

/**
 * Série temporal por ano em SVG: barras agrupadas, empilhadas ou linhas. Substitui o
 * Plotly na produção ao longo do tempo e na linha do tempo do Motor de Busca.
 *
 * - Passar o mouse destaca o ano e lista o valor de cada série.
 * - Clicar numa barra (ou ponto) chama `onSeriesClick` com o nome da série.
 * - Clicar na legenda mostra ou esconde a série.
 * - Arrastar sobre o gráfico dá zoom num período; duplo clique ou o botão desfazem.
 */
export interface TimeSeries {
  name: string;
  color: string;
  points: readonly { x: number; y: number }[];
}

export type TimeSeriesMode = 'bars-grouped' | 'bars-stacked' | 'line';

export interface TimeSeriesChartProps {
  series: readonly TimeSeries[];
  mode: TimeSeriesMode;
  xLabel: string;
  yLabel: string;
  /** Unidade dos valores na dica ("documentos"). */
  unit?: string;
  height?: number;
  exportName: string;
  onSeriesClick?: ((name: string) => void) | undefined;
  expanded?: boolean;
  className?: string;
}

const MAX_TIP_ROWS = 10;

function TimeSeriesChart(props: TimeSeriesChartProps) {
  const {
    series,
    mode,
    xLabel,
    yLabel,
    unit = '',
    height = 440,
    exportName,
    onSeriesClick,
    expanded,
    className,
  } = props;
  const t = useLocale((state) => state.t);
  const locale = useLocale((state) => state.locale);
  const svgRef = useRef<SVGSVGElement>(null);
  const [containerRef, width] = useElementWidth<HTMLDivElement>();
  const { tooltip, show, hide } = useTooltip(containerRef);
  const [hidden, setHidden] = useState<ReadonlySet<string>>(new Set());
  const [zoom, setZoom] = useState<[number, number] | null>(null);
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  // Todos os anos do intervalo, inclusive os sem documento: um buraco na série é informação.
  const years = useMemo(() => {
    const all = series.flatMap((entry) => entry.points.map((point) => point.x));
    if (all.length === 0) return [];
    const min = Math.min(...all);
    const max = Math.max(...all);
    return Array.from({ length: max - min + 1 }, (_, index) => min + index);
  }, [series]);

  const valueOf = useMemo(() => {
    const maps = new Map(series.map((entry) => [entry.name, new Map(entry.points.map((point) => [point.x, point.y]))]));
    return (name: string, year: number): number => maps.get(name)?.get(year) ?? 0;
  }, [series]);

  const visible = series.filter((entry) => !hidden.has(entry.name));
  const [from, to] = zoom ?? [0, Math.max(0, years.length - 1)];
  const shownYears = years.slice(from, to + 1);

  const legend = useMemo(
    () =>
      layoutLegend(
        series.length > 1
          ? series.map((entry) => ({
              key: entry.name,
              label: entry.name,
              color: entry.color,
              line: mode === 'line',
              muted: hidden.has(entry.name),
            }))
          : [],
        Math.max(120, width - 24),
      ),
    [series, width, mode, hidden],
  );

  const stacked = mode === 'bars-stacked';
  const yMaxRaw = Math.max(
    1,
    ...shownYears.map((year) =>
      stacked
        ? visible.reduce((sum, entry) => sum + valueOf(entry.name, year), 0)
        : Math.max(0, ...visible.map((entry) => valueOf(entry.name, year))),
    ),
  );
  const [, yMax] = niceDomain(0, yMaxRaw, 5);
  const yTicks = linearTicks(0, yMax, 5);
  const tickWidth = Math.max(...yTicks.map((value) => measureText(formatTick(value, locale), 11, FONT_MONO)));

  const area: PlotArea = {
    left: Math.round(tickWidth + 40),
    top: legend.height + 12,
    width: Math.max(40, width - tickWidth - 40 - 16),
    height: Math.max(80, height - legend.height - 12 - 52),
  };
  const band = area.width / Math.max(1, shownYears.length);
  const sy = linearScale([0, yMax], [area.top + area.height, area.top]);
  const cx = (index: number): number => area.left + band * (index + 0.5);

  const labelEvery = Math.max(1, Math.ceil(shownYears.length / Math.max(1, Math.floor(area.width / 46))));

  const brush = useBrush(svgRef, area, 'x', (rect) => {
    const i0 = Math.max(0, Math.floor((rect.x0 - area.left) / band));
    const i1 = Math.min(shownYears.length - 1, Math.floor((rect.x1 - area.left) / band));
    if (i1 - i0 >= 1) setZoom([from + i0, from + i1]);
  });

  const indexAt = (event: PointerEvent<SVGSVGElement>): number | null => {
    const ctm = svgRef.current?.getScreenCTM();
    if (!ctm) return null;
    const { x, y } = new DOMPoint(event.clientX, event.clientY).matrixTransform(ctm.inverse());
    if (x < area.left || x > area.left + area.width || y < area.top || y > area.top + area.height) return null;
    return Math.min(shownYears.length - 1, Math.max(0, Math.floor((x - area.left) / band)));
  };

  const handleMove = (event: PointerEvent<SVGSVGElement>): void => {
    brush.handlers.onPointerMove(event);
    const index = indexAt(event);
    setHoverIndex(index);
    if (index === null) {
      hide();
      return;
    }
    const year = shownYears[index] as number;
    const rows = visible
      .map((entry) => ({ entry, value: valueOf(entry.name, year) }))
      .sort((a, b) => b.value - a.value);
    const total = rows.reduce((sum, row) => sum + row.value, 0);
    show(
      event,
      <>
        <p className="mb-1 font-semibold tabular-nums">{year}</p>
        {rows.slice(0, MAX_TIP_ROWS).map(({ entry, value }) => (
          <TipRow key={entry.name} label={entry.name} value={formatNumber(value, undefined, locale)} color={entry.color} />
        ))}
        {rows.length > MAX_TIP_ROWS && <p className="text-muted-foreground">+{rows.length - MAX_TIP_ROWS}</p>}
        {rows.length > 1 && (
          <p className="mt-1 border-t border-border pt-1 text-muted-foreground">
            Total: <span className="tabular-nums text-foreground">{formatNumber(total, undefined, locale)}</span> {unit}
          </p>
        )}
        {rows.length === 1 && unit && <p className="text-muted-foreground">{unit}</p>}
      </>,
    );
  };

  const toggle = (name: string): void =>
    setHidden((current) => {
      const next = new Set(current);
      if (next.has(name)) next.delete(name);
      else if (next.size < series.length - 1) next.add(name);
      return next;
    });

  const clickable = onSeriesClick ? 'cursor-pointer' : undefined;
  const inner = band * 0.78;

  const marks = visible.map((entry, seriesIndex) => {
    if (mode === 'line') {
      const points = shownYears.map((year, index) => [cx(index), sy(valueOf(entry.name, year))] as const);
      const path = points.map(([x, y], index) => `${index ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`).join('');
      return (
        <g key={entry.name}>
          <path d={path} fill="none" stroke={entry.color} strokeWidth={2} strokeLinejoin="round" />
          {points.map(([x, y], index) => (
            <circle
              key={index}
              cx={x}
              cy={y}
              r={hoverIndex === index ? 4.5 : band > 10 ? 3 : 0}
              fill={entry.color}
              stroke="var(--background)"
              strokeWidth={1}
              className={clickable}
              onClick={onSeriesClick ? () => onSeriesClick(entry.name) : undefined}
            />
          ))}
        </g>
      );
    }

    return (
      <g key={entry.name} fill={entry.color}>
        {shownYears.map((year, index) => {
          const value = valueOf(entry.name, year);
          if (value <= 0) return null;
          let x: number;
          let w: number;
          let y0: number;
          if (stacked) {
            const below = visible.slice(0, seriesIndex).reduce((sum, other) => sum + valueOf(other.name, year), 0);
            x = cx(index) - inner / 2;
            w = inner;
            y0 = below;
          } else {
            w = inner / visible.length;
            x = cx(index) - inner / 2 + seriesIndex * w;
            y0 = 0;
          }
          const top = sy(y0 + value);
          const bottom = sy(y0);
          return (
            <rect
              key={year}
              x={x}
              y={top}
              width={Math.max(0.5, w - (visible.length > 1 && !stacked ? 0.5 : 0))}
              height={Math.max(0.5, bottom - top)}
              opacity={hoverIndex === null || hoverIndex === index ? 1 : 0.55}
              className={clickable}
              onClick={onSeriesClick ? () => onSeriesClick(entry.name) : undefined}
            />
          );
        })}
      </g>
    );
  });

  // Resumo para leitores de tela (vai no <desc> do SVG — e no arquivo exportado).
  const summary = (() => {
    if (years.length === 0) return '';
    const totals = years.map((year) => series.reduce((sum, entry) => sum + valueOf(entry.name, year), 0));
    const total = totals.reduce((sum, value) => sum + value, 0);
    const peak = totals.indexOf(Math.max(...totals));
    return `${t('chart_summary_series')
      .replace('{count}', String(series.length))
      .replace('{from}', String(years[0]))
      .replace('{to}', String(years[years.length - 1]))} ${t('chart_summary_total')
      .replace('{total}', formatNumber(total, undefined, locale))
      .replace('{peak}', String(years[peak]))
      .replace('{value}', formatNumber(totals[peak] ?? 0, undefined, locale))}`;
  })();

  return (
    <ChartFrame
      exportName={exportName}
      svgRef={svgRef}
      containerRef={containerRef}
      width={width}
      tooltip={tooltip}
      className={className}
      onResetZoom={zoom ? () => setZoom(null) : undefined}
      toolbar={<span className="eyebrow hidden sm:inline">{t('chart_brush_hint')}</span>}
      expandedContent={
        expanded ? undefined : <TimeSeriesChart {...props} expanded height={Math.max(height, expandedHeight())} />
      }
    >
      {years.length === 0 ? null : (
        <svg
          ref={svgRef}
          width={width}
          height={height}
          viewBox={`0 0 ${width} ${height}`}
          className={cn('block select-none', onSeriesClick && 'touch-pan-y')}
          role={series.length > 1 ? 'figure' : 'img'}
          aria-label={`${yLabel} × ${xLabel}`}
          xmlns="http://www.w3.org/2000/svg"
          fontFamily={FONT_SANS}
          {...brush.handlers}
          onPointerMove={handleMove}
          onPointerLeave={() => {
            setHoverIndex(null);
            hide();
          }}
          onDoubleClick={() => setZoom(null)}
        >
          <desc>{summary}</desc>
          <SvgLegend layout={legend} x={12} y={4} onToggle={toggle} />

          <g fontFamily={FONT_MONO} fontSize={11} fill="var(--muted-foreground)" aria-hidden>
            {/* Grade secundária: meio-passos no eixo y (sem rótulo, para não poluir o eixo)
                e uma vertical por ano rotulado. */}
            <g stroke="var(--muted-foreground)" strokeOpacity={0.22} strokeWidth={0.7} strokeDasharray="3 3">
              {yTicks.slice(1).map((value, index) => {
                const y = sy((value + (yTicks[index] as number)) / 2);
                return <line key={`ym${value}`} x1={area.left} x2={area.left + area.width} y1={y} y2={y} />;
              })}
              {shownYears.map((year, index) =>
                index % labelEvery === 0 ? (
                  <line key={`xg${year}`} x1={cx(index)} x2={cx(index)} y1={area.top} y2={area.top + area.height} />
                ) : null,
              )}
            </g>
            {yTicks.map((value) => (
              <g key={`y${value}`}>
                <line
                  x1={area.left}
                  x2={area.left + area.width}
                  y1={sy(value)}
                  y2={sy(value)}
                  stroke="var(--border)"
                  strokeWidth={value === 0 ? 1 : 0.6}
                />
                <text x={area.left - 8} y={sy(value)} dy="0.32em" textAnchor="end">
                  {formatTick(value, locale)}
                </text>
              </g>
            ))}
            {shownYears.map((year, index) =>
              index % labelEvery === 0 ? (
                <text key={`x${year}`} x={cx(index)} y={area.top + area.height + 18} textAnchor="middle">
                  {year}
                </text>
              ) : null,
            )}
          </g>

          <text
            x={area.left + area.width / 2}
            y={height - 8}
            textAnchor="middle"
            fontSize={12}
            fill="var(--foreground)"
          >
            {xLabel}
          </text>
          <text
            transform={`translate(14 ${area.top + area.height / 2}) rotate(-90)`}
            textAnchor="middle"
            fontSize={12}
            fill="var(--foreground)"
          >
            {yLabel}
          </text>

          {hoverIndex !== null && (
            <rect
              x={area.left + band * hoverIndex}
              y={area.top}
              width={band}
              height={area.height}
              fill="var(--foreground)"
              fillOpacity={0.05}
              pointerEvents="none"
            />
          )}

          {/* Captura o mouse nos vãos entre as barras. */}
          <rect x={area.left} y={area.top} width={area.width} height={area.height} fill="transparent" />
          {marks}
          {brush.overlay}
        </svg>
      )}
    </ChartFrame>
  );
}

// Um gráfico que quebre com dado atípico não derruba a aba (ver with-chart-boundary).
export default withChartBoundary(TimeSeriesChart);
