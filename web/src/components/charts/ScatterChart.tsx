import { useId, useMemo, useRef, useState, type ReactNode } from 'react';

import { expandedHeight } from '@/components/charts/ExpandChartButton';
import { ChartFrame, SvgLegend } from '@/components/charts/svg/chart-kit';
import {
  useBrush,
  useElementWidth,
  useTooltip,
  type PlotArea,
} from '@/components/charts/svg/hooks';
import { layoutLegend, type LegendItem } from '@/components/charts/svg/legend';
import { fitText, FONT_MONO, FONT_SANS, measureText } from '@/components/charts/svg/text';
import { formatNumber, formatTick, linearScale, linearTicks, niceStep } from '@/components/charts/svg/scale';
import { useLocale } from '@/state/locale.store';
import { withChartBoundary } from '@/components/with-chart-boundary';

/**
 * Dispersão em SVG — bolhas com rótulo, ligações entre pontos, linhas de referência,
 * quadrantes e escala de cor. Cobre o que o Plotly desenhava na genética dos termos, no
 * mapa conceitual 2D, no mapa temático e no historiograph.
 *
 * Os rótulos não se atropelam: são colocados do maior ponto para o menor e um rótulo que
 * cairia sobre outro fica oculto (aparece ao passar o mouse). Arrastar um retângulo dá
 * zoom; duplo clique ou o botão desfazem.
 */
export interface ScatterPoint {
  id: string;
  x: number;
  y: number;
  /** Raio em pixels. */
  r: number;
  color: string;
  label?: string;
  /** Série a que pertence — a legenda mostra e esconde séries inteiras. */
  group?: string;
  opacity?: number;
  tooltip: ReactNode;
  onClick?: (() => void) | undefined;
}

export interface ColorScaleLegend {
  label: string;
  min: number;
  max: number;
  /** Cores do gradiente, do menor para o maior valor. */
  colors: readonly string[];
}

export interface ScatterChartProps {
  points: readonly ScatterPoint[];
  edges?: readonly { from: string; to: string }[];
  xLabel: string;
  yLabel?: string;
  labelPlacement?: 'top' | 'center';
  /** Eixo X só com inteiros (anos). */
  integerX?: boolean;
  hideYAxis?: boolean;
  /** Linhas tracejadas de referência (médias do mapa temático). */
  reference?: { x?: number; y?: number };
  /** Nome de cada quadrante, nos cantos do gráfico. */
  quadrants?: { topLeft: string; topRight: string; bottomLeft: string; bottomRight: string };
  legend?: readonly LegendItem[];
  colorScale?: ColorScaleLegend;
  height?: number;
  exportName: string;
  ariaLabel: string;
  expanded?: boolean;
  className?: string;
}

interface Domain {
  x: [number, number];
  y: [number, number];
}

const LABEL_FONT = 10;

function paddedDomain(values: readonly number[], pad = 0.08): [number, number] {
  if (values.length === 0) return [0, 1];
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (min === max) {
    min -= Math.abs(min) * 0.1 || 1;
    max += Math.abs(max) * 0.1 || 1;
  }
  const span = max - min;
  return [min - span * pad, max + span * pad];
}

function ScatterChart(props: ScatterChartProps) {
  const {
    points,
    edges = [],
    xLabel,
    yLabel,
    labelPlacement = 'top',
    integerX = false,
    hideYAxis = false,
    reference,
    quadrants,
    legend: legendItems = [],
    colorScale,
    height = 480,
    exportName,
    ariaLabel,
    expanded,
    className,
  } = props;
  const t = useLocale((state) => state.t);
  const locale = useLocale((state) => state.locale);
  const clipId = useId().replace(/:/g, '');
  const svgRef = useRef<SVGSVGElement>(null);
  const [containerRef, width] = useElementWidth<HTMLDivElement>();
  const { tooltip, show, hide } = useTooltip(containerRef);
  const [zoom, setZoom] = useState<Domain | null>(null);
  const [hovered, setHovered] = useState<string | null>(null);
  const [hiddenGroups, setHiddenGroups] = useState<ReadonlySet<string>>(new Set());

  const visible = points.filter((point) => !point.group || !hiddenGroups.has(point.group));

  const fullDomain = useMemo<Domain>(
    () => ({
      x: paddedDomain(points.map((point) => point.x)),
      y: paddedDomain(points.map((point) => point.y), 0.12),
    }),
    [points],
  );
  const domain = zoom ?? fullDomain;

  const legend = useMemo(
    () =>
      layoutLegend(
        legendItems.map((item) => ({ ...item, muted: hiddenGroups.has(item.key) })),
        Math.max(120, width - 24),
      ),
    [legendItems, width, hiddenGroups],
  );

  const yTicks = hideYAxis ? [] : linearTicks(domain.y[0], domain.y[1], 6);
  const tickWidth = hideYAxis ? 0 : Math.max(...yTicks.map((value) => measureText(formatTick(value, locale), 11, FONT_MONO)), 10);
  const colorBarSpace = colorScale ? 78 : 0;

  const area: PlotArea = {
    left: hideYAxis ? 16 : Math.round(tickWidth + (yLabel ? 40 : 18)),
    top: legend.height + 12,
    width: 0,
    height: Math.max(100, height - legend.height - 12 - 50),
  };
  area.width = Math.max(60, width - area.left - 16 - colorBarSpace);

  const sx = linearScale(domain.x, [area.left, area.left + area.width]);
  const sy = linearScale(domain.y, [area.top + area.height, area.top]);

  const xTicks = (() => {
    const approx = Math.max(2, Math.floor(area.width / 70));
    if (!integerX) return linearTicks(domain.x[0], domain.x[1], approx);
    const step = Math.max(1, Math.round(niceStep(domain.x[1] - domain.x[0], approx)));
    const start = Math.ceil(domain.x[0] / step) * step;
    const ticks: number[] = [];
    for (let value = start; value <= domain.x[1]; value += step) ticks.push(value);
    return ticks;
  })();

  const brush = useBrush(svgRef, area, 'xy', (rect) =>
    setZoom({
      x: [sx.invert(rect.x0), sx.invert(rect.x1)],
      y: [sy.invert(rect.y1), sy.invert(rect.y0)],
    }),
  );

  const positions = new Map(visible.map((point) => [point.id, { x: sx(point.x), y: sy(point.y), point }]));

  // Rótulos sem sobreposição, do maior ponto para o menor.
  // Quebras de linha ("\n") viram linhas separadas, centradas no ponto.
  const shownLabels = new Map<string, { lines: string[]; x: number; y: number }>();
  {
    const boxes: { x0: number; y0: number; x1: number; y1: number }[] = [];
    const bySize = [...positions.values()].sort((a, b) => b.point.r - a.point.r);
    for (const { x, y, point } of bySize) {
      if (!point.label) continue;
      if (x < area.left || x > area.left + area.width || y < area.top || y > area.top + area.height) continue;
      const maxWidth = labelPlacement === 'center' ? Math.max(60, point.r * 2.6) : 150;
      const lines = point.label.split('\n').map((line) => fitText(line, maxWidth, LABEL_FONT));
      const w = Math.max(...lines.map((line) => measureText(line, LABEL_FONT)));
      const ly = labelPlacement === 'center' ? y : y - point.r - 4;
      const box = { x0: x - w / 2 - 2, x1: x + w / 2 + 2, y0: ly - LABEL_FONT * lines.length, y1: ly + 2 };
      const collides = boxes.some(
        (other) => box.x0 < other.x1 && box.x1 > other.x0 && box.y0 < other.y1 && box.y1 > other.y0,
      );
      if (collides && labelPlacement !== 'center') continue;
      boxes.push(box);
      shownLabels.set(point.id, { lines, x, y: ly });
    }
  }

  const linked = useMemo(() => {
    if (!hovered) return null;
    const ids = new Set([hovered]);
    for (const edge of edges) {
      if (edge.from === hovered) ids.add(edge.to);
      if (edge.to === hovered) ids.add(edge.from);
    }
    return ids;
  }, [hovered, edges]);

  const toggleGroup = (key: string): void =>
    setHiddenGroups((current) => {
      const next = new Set(current);
      if (next.has(key)) next.delete(key);
      else if (next.size < legendItems.length - 1) next.add(key);
      return next;
    });

  const ordered = [...positions.values()].sort((a, b) => b.point.r - a.point.r);
  const gradientId = `${clipId}-scale`;

  // Resumo para leitores de tela: quantos pontos e os maiores.
  const summary = `${t('chart_summary_points').replace('{count}', String(points.length))} ${t('chart_summary_largest')}: ${[
    ...points,
  ]
    .sort((a, b) => b.r - a.r)
    .slice(0, 5)
    .map((point) => (point.label ?? point.id).replace(/\n/g, ' '))
    .join(', ')}.`;

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
        expanded ? undefined : <ScatterChart {...props} expanded height={Math.max(height, expandedHeight())} />
      }
    >
      <svg
        ref={svgRef}
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        className="block select-none"
        role={legendItems.length > 0 ? 'figure' : 'img'}
        aria-label={ariaLabel}
        xmlns="http://www.w3.org/2000/svg"
        fontFamily={FONT_SANS}
        {...brush.handlers}
        onDoubleClick={() => setZoom(null)}
        onPointerLeave={() => {
          setHovered(null);
          hide();
        }}
      >
        <desc>{summary}</desc>
        <defs>
          <clipPath id={clipId}>
            <rect x={area.left} y={area.top} width={area.width} height={area.height} />
          </clipPath>
          {colorScale && (
            <linearGradient id={gradientId} x1="0" y1="1" x2="0" y2="0">
              {colorScale.colors.map((color, index) => (
                <stop key={color + index} offset={index / Math.max(1, colorScale.colors.length - 1)} stopColor={color} />
              ))}
            </linearGradient>
          )}
        </defs>

        <SvgLegend layout={legend} x={12} y={4} onToggle={toggleGroup} />

        <g fontFamily={FONT_MONO} fontSize={11} fill="var(--muted-foreground)" aria-hidden>
          {yTicks.map((value) => (
            <g key={`y${value}`}>
              <line x1={area.left} x2={area.left + area.width} y1={sy(value)} y2={sy(value)} stroke="var(--border)" strokeWidth={0.6} />
              <text x={area.left - 8} y={sy(value)} dy="0.32em" textAnchor="end">
                {formatTick(value, locale)}
              </text>
            </g>
          ))}
          {xTicks.map((value) => (
            <g key={`x${value}`}>
              <line x1={sx(value)} x2={sx(value)} y1={area.top} y2={area.top + area.height} stroke="var(--border)" strokeWidth={0.6} />
              <text x={sx(value)} y={area.top + area.height + 18} textAnchor="middle">
                {integerX ? String(value) : formatTick(value, locale)}
              </text>
            </g>
          ))}
        </g>
        <rect x={area.left} y={area.top} width={area.width} height={area.height} fill="none" stroke="var(--border)" />

        <text x={area.left + area.width / 2} y={height - 8} textAnchor="middle" fontSize={12} fill="var(--foreground)">
          {xLabel}
        </text>
        {yLabel && !hideYAxis && (
          <text
            transform={`translate(14 ${area.top + area.height / 2}) rotate(-90)`}
            textAnchor="middle"
            fontSize={12}
            fill="var(--foreground)"
          >
            {yLabel}
          </text>
        )}

        {/* Área de captura do arraste (zoom) sob os pontos. */}
        <rect x={area.left} y={area.top} width={area.width} height={area.height} fill="transparent" />

        <g clipPath={`url(#${clipId})`}>
          {reference?.x !== undefined && (
            <line
              x1={sx(reference.x)}
              x2={sx(reference.x)}
              y1={area.top}
              y2={area.top + area.height}
              stroke="var(--muted-foreground)"
              strokeOpacity={0.6}
              strokeDasharray="5 4"
            />
          )}
          {reference?.y !== undefined && (
            <line
              x1={area.left}
              x2={area.left + area.width}
              y1={sy(reference.y)}
              y2={sy(reference.y)}
              stroke="var(--muted-foreground)"
              strokeOpacity={0.6}
              strokeDasharray="5 4"
            />
          )}

          <g fill="none">
            {edges.map((edge, index) => {
              const a = positions.get(edge.from);
              const b = positions.get(edge.to);
              if (!a || !b) return null;
              const on = !linked || (linked.has(edge.from) && linked.has(edge.to) && (edge.from === hovered || edge.to === hovered));
              return (
                <line
                  key={index}
                  x1={a.x}
                  y1={a.y}
                  x2={b.x}
                  y2={b.y}
                  stroke={on && linked ? 'var(--highlight)' : 'var(--muted-foreground)'}
                  strokeOpacity={linked ? (on ? 0.9 : 0.08) : 0.4}
                  strokeWidth={on && linked ? 1.6 : 1}
                  className="transition-[stroke-opacity] duration-200"
                />
              );
            })}
          </g>

          {ordered.map(({ x, y, point }) => {
            const dimmed = linked !== null && !linked.has(point.id);
            return (
              <circle
                key={point.id}
                cx={x}
                cy={y}
                r={hovered === point.id ? point.r + 2 : point.r}
                fill={point.color}
                fillOpacity={dimmed ? 0.15 : (point.opacity ?? 0.8)}
                stroke="var(--background)"
                strokeWidth={1}
                className={point.onClick ? 'cursor-pointer transition-[fill-opacity] duration-200' : 'transition-[fill-opacity] duration-200'}
                onPointerMove={(event) => {
                  setHovered(point.id);
                  show(event, point.tooltip);
                }}
                onPointerLeave={() => {
                  setHovered(null);
                  hide();
                }}
                onClick={point.onClick}
              />
            );
          })}

          <g fontSize={LABEL_FONT} fill="var(--foreground)" pointerEvents="none">
            {ordered.map(({ x, y, point }) => {
              const label = shownLabels.get(point.id);
              const forced = hovered === point.id && !label && point.label;
              if (!label && !forced) return null;
              const dimmed = linked !== null && !linked.has(point.id);
              const lines = label?.lines ?? (point.label ?? '').split('\n');
              const lx = label?.x ?? x;
              // No centro, o bloco de linhas fica centrado na vertical; em cima, a última
              // linha encosta no ponto.
              const firstDy =
                labelPlacement === 'center' ? `${0.35 - ((lines.length - 1) * 1.15) / 2}em` : `${-(lines.length - 1) * 1.15}em`;
              return (
                <text
                  key={point.id}
                  x={lx}
                  y={label?.y ?? y - point.r - 4}
                  textAnchor="middle"
                  opacity={dimmed ? 0.25 : 1}
                  fontWeight={hovered === point.id || labelPlacement === 'center' ? 700 : 500}
                  paintOrder="stroke"
                  stroke="var(--background)"
                  strokeWidth={labelPlacement === 'center' ? 0 : 3}
                  strokeOpacity={0.7}
                >
                  {lines.map((line, index) => (
                    <tspan key={index} x={lx} dy={index === 0 ? firstDy : '1.15em'}>
                      {line}
                    </tspan>
                  ))}
                </text>
              );
            })}
          </g>
        </g>

        {quadrants && (
          <g fontSize={11} fontWeight={700} fill="var(--muted-foreground)" pointerEvents="none">
            <text x={area.left + 8} y={area.top + 16}>
              {quadrants.topLeft}
            </text>
            <text x={area.left + area.width - 8} y={area.top + 16} textAnchor="end">
              {quadrants.topRight}
            </text>
            <text x={area.left + 8} y={area.top + area.height - 8}>
              {quadrants.bottomLeft}
            </text>
            <text x={area.left + area.width - 8} y={area.top + area.height - 8} textAnchor="end">
              {quadrants.bottomRight}
            </text>
          </g>
        )}

        {colorScale && (
          <g transform={`translate(${area.left + area.width + 20} ${area.top})`} fontFamily={FONT_MONO} fontSize={10}>
            <rect width={12} height={area.height * 0.6} fill={`url(#${gradientId})`} rx={2} />
            <text x={18} y={8} fill="var(--muted-foreground)">
              {formatNumber(colorScale.max, 0, locale)}
            </text>
            <text x={18} y={area.height * 0.6} fill="var(--muted-foreground)">
              {formatNumber(colorScale.min, 0, locale)}
            </text>
            <text
              transform={`translate(-6 ${area.height * 0.3}) rotate(-90)`}
              textAnchor="middle"
              fontFamily={FONT_SANS}
              fontSize={11}
              fill="var(--foreground)"
            >
              {colorScale.label}
            </text>
          </g>
        )}

        {brush.overlay}
      </svg>
    </ChartFrame>
  );
}

// Um gráfico que quebre com dado atípico não derruba a aba (ver with-chart-boundary).
export default withChartBoundary(ScatterChart);
