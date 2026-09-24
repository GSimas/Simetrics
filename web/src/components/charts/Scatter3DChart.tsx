import { useEffect, useMemo, useRef, useState, type PointerEvent, type ReactNode } from 'react';

import { expandedHeight } from '@/components/charts/ExpandChartButton';
import { ChartFrame, SvgLegend } from '@/components/charts/svg/chart-kit';
import { useElementWidth, useTooltip } from '@/components/charts/svg/hooks';
import { layoutLegend, type LegendItem } from '@/components/charts/svg/legend';
import { fitText, FONT_MONO, FONT_SANS, measureText } from '@/components/charts/svg/text';
import { clamp } from '@/components/charts/svg/scale';
import { useLocale } from '@/state/locale.store';
import { withChartBoundary } from '@/components/with-chart-boundary';

/**
 * Dispersão 3D em SVG, no lugar do `scatter3d` (WebGL) do Plotly.
 *
 * Projeção ortográfica de um cubo normalizado: cada eixo vai de -1 a 1, como o
 * `aspectmode: 'cube'`. Arrastar gira (horizontal = em torno do eixo vertical, vertical =
 * inclinação), a roda aproxima, e os pontos são desenhados de trás para a frente — os da
 * frente ficam maiores e mais opacos, o que dá a leitura de profundidade.
 */
export interface Point3D {
  id: string;
  x: number;
  y: number;
  z: number;
  r: number;
  color: string;
  label?: string;
  group?: string;
  tooltip: ReactNode;
  onClick?: (() => void) | undefined;
}

export interface Scatter3DChartProps {
  points: readonly Point3D[];
  axisLabels: readonly [string, string, string];
  legend?: readonly LegendItem[];
  height?: number;
  exportName: string;
  ariaLabel: string;
  expanded?: boolean;
  className?: string;
}

interface View {
  yaw: number;
  pitch: number;
  zoom: number;
}

const DEFAULT_VIEW: View = { yaw: -0.65, pitch: 0.42, zoom: 1 };
const LABEL_FONT = 10;

/** Arestas do cubo [-1, 1]³ por índice de vértice. */
const CUBE_EDGES: readonly (readonly [number, number])[] = [
  [0, 1], [1, 3], [3, 2], [2, 0],
  [4, 5], [5, 7], [7, 6], [6, 4],
  [0, 4], [1, 5], [2, 6], [3, 7],
];
const CUBE_VERTICES = [-1, 1].flatMap((x) => [-1, 1].flatMap((y) => [-1, 1].map((z) => [x, y, z] as const)));

function normalize(values: readonly number[]): (value: number) => number {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  return (value) => ((value - min) / span) * 2 - 1;
}

function Scatter3DChart(props: Scatter3DChartProps) {
  const { points, axisLabels, legend: legendItems = [], height = 620, exportName, ariaLabel, expanded, className } = props;
  const t = useLocale((state) => state.t);
  const svgRef = useRef<SVGSVGElement>(null);
  const [containerRef, width] = useElementWidth<HTMLDivElement>();
  const { tooltip, show, hide } = useTooltip(containerRef);
  const [view, setView] = useState<View>(DEFAULT_VIEW);
  const [hovered, setHovered] = useState<string | null>(null);
  const [hiddenGroups, setHiddenGroups] = useState<ReadonlySet<string>>(new Set());
  const drag = useRef<{ id: number; x: number; y: number; view: View; moved: boolean } | null>(null);
  const suppressClick = useRef(false);

  const legend = useMemo(
    () =>
      layoutLegend(
        legendItems.map((item) => ({ ...item, muted: hiddenGroups.has(item.key) })),
        Math.max(120, width - 24),
      ),
    [legendItems, width, hiddenGroups],
  );

  const normalized = useMemo(() => {
    const nx = normalize(points.map((point) => point.x));
    const ny = normalize(points.map((point) => point.y));
    const nz = normalize(points.map((point) => point.z));
    return points.map((point) => ({ point, v: [nx(point.x), ny(point.y), nz(point.z)] as const }));
  }, [points]);

  const top = legend.height + 8;
  const plotHeight = height - top - 8;
  const cx = width / 2;
  const cy = top + plotHeight / 2;
  // O cubo girado cabe num círculo de raio √3; a escala deixa folga para os rótulos.
  const scale = (Math.min(width, plotHeight) / 2 / 1.9) * view.zoom;

  const project = (x: number, y: number, z: number): { sx: number; sy: number; depth: number } => {
    // Dados: x → horizontal, y → profundidade, z → vertical (como no Plotly).
    const cosY = Math.cos(view.yaw);
    const sinY = Math.sin(view.yaw);
    const rx = x * cosY - y * sinY;
    const ry = x * sinY + y * cosY;
    const cosP = Math.cos(view.pitch);
    const sinP = Math.sin(view.pitch);
    const up = z * cosP - ry * sinP;
    const depth = z * sinP + ry * cosP;
    return { sx: cx + rx * scale, sy: cy - up * scale, depth };
  };

  const cube = CUBE_VERTICES.map(([x, y, z]) => project(x, y, z));
  const projected = normalized
    .filter(({ point }) => !point.group || !hiddenGroups.has(point.group))
    .map(({ point, v }) => ({ point, ...project(v[0], v[1], v[2]) }))
    .sort((a, b) => b.depth - a.depth);

  const depthRange = projected.length ? [Math.min(...projected.map((p) => p.depth)), Math.max(...projected.map((p) => p.depth))] : [0, 1];
  const nearness = (depth: number): number =>
    1 - (depth - (depthRange[0] as number)) / (((depthRange[1] as number) - (depthRange[0] as number)) || 1);

  // Rótulos dos pontos da frente primeiro; os que colidiriam ficam ocultos.
  const labels = new Map<string, string>();
  {
    const boxes: { x0: number; x1: number; y0: number; y1: number }[] = [];
    for (const item of [...projected].reverse()) {
      if (!item.point.label) continue;
      const text = fitText(item.point.label, 130, LABEL_FONT);
      const w = measureText(text, LABEL_FONT);
      const y = item.sy - item.point.r - 4;
      const box = { x0: item.sx - w / 2, x1: item.sx + w / 2, y0: y - LABEL_FONT, y1: y + 2 };
      if (boxes.some((o) => box.x0 < o.x1 && box.x1 > o.x0 && box.y0 < o.y1 && box.y1 > o.y0)) continue;
      boxes.push(box);
      labels.set(item.point.id, text);
    }
  }

  const axisEnds = [
    { label: axisLabels[0], from: project(-1, -1, -1), to: project(1, -1, -1) },
    { label: axisLabels[1], from: project(-1, -1, -1), to: project(-1, 1, -1) },
    { label: axisLabels[2], from: project(-1, -1, -1), to: project(-1, -1, 1) },
  ];

  // `wheel` precisa de listener não passivo para não rolar a página enquanto aproxima.
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const onWheel = (event: WheelEvent): void => {
      event.preventDefault();
      setView((current) => ({ ...current, zoom: clamp(current.zoom * Math.exp(-event.deltaY * 0.0015), 0.5, 4) }));
    };
    svg.addEventListener('wheel', onWheel, { passive: false });
    return () => svg.removeEventListener('wheel', onWheel);
  }, []);

  const onPointerDown = (event: PointerEvent<SVGSVGElement>): void => {
    if (event.button !== 0) return;
    suppressClick.current = false;
    drag.current = { id: event.pointerId, x: event.clientX, y: event.clientY, view, moved: false };
  };
  const onPointerMove = (event: PointerEvent<SVGSVGElement>): void => {
    const current = drag.current;
    if (!current || current.id !== event.pointerId) return;
    const dx = event.clientX - current.x;
    const dy = event.clientY - current.y;
    if (!current.moved && Math.hypot(dx, dy) < 4) return;
    if (!current.moved) event.currentTarget.setPointerCapture(event.pointerId);
    current.moved = true;
    hide();
    setView({
      ...current.view,
      yaw: current.view.yaw + dx * 0.01,
      pitch: clamp(current.view.pitch + dy * 0.01, -1.45, 1.45),
    });
  };
  const endDrag = (): void => {
    suppressClick.current = drag.current?.moved ?? false;
    drag.current = null;
  };

  const toggleGroup = (key: string): void =>
    setHiddenGroups((current) => {
      const next = new Set(current);
      if (next.has(key)) next.delete(key);
      else if (next.size < legendItems.length - 1) next.add(key);
      return next;
    });

  const changed = view.yaw !== DEFAULT_VIEW.yaw || view.pitch !== DEFAULT_VIEW.pitch || view.zoom !== DEFAULT_VIEW.zoom;

  // Resumo para leitores de tela: quantos pontos e os maiores.
  const summary = `${t('chart_summary_points').replace('{count}', String(points.length))} ${t('chart_summary_largest')}: ${[
    ...points,
  ]
    .sort((a, b) => b.r - a.r)
    .slice(0, 5)
    .map((point) => point.label ?? point.id)
    .join(', ')}.`;

  return (
    <ChartFrame
      exportName={exportName}
      svgRef={svgRef}
      containerRef={containerRef}
      width={width}
      tooltip={tooltip}
      className={className}
      onResetZoom={changed ? () => setView(DEFAULT_VIEW) : undefined}
      toolbar={<span className="eyebrow">{t('chart_3d_hint')}</span>}
      expandedContent={
        expanded ? undefined : <Scatter3DChart {...props} expanded height={Math.max(height, expandedHeight())} />
      }
    >
      <svg
        ref={svgRef}
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        className="block cursor-grab touch-pan-y select-none active:cursor-grabbing"
        role={legendItems.length > 0 ? 'figure' : 'img'}
        aria-label={ariaLabel}
        xmlns="http://www.w3.org/2000/svg"
        fontFamily={FONT_SANS}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onClickCapture={(event) => {
          // Soltar depois de girar não conta como clique num ponto.
          if (!suppressClick.current) return;
          suppressClick.current = false;
          event.stopPropagation();
        }}
        onPointerLeave={() => {
          setHovered(null);
          hide();
        }}
        onDoubleClick={() => setView(DEFAULT_VIEW)}
      >
        <desc>{summary}</desc>
        <SvgLegend layout={legend} x={12} y={4} onToggle={toggleGroup} />

        <g stroke="var(--border)" strokeWidth={0.8} fill="none">
          {CUBE_EDGES.map(([a, b]) => {
            const p = cube[a];
            const q = cube[b];
            if (!p || !q) return null;
            return <line key={`${a}-${b}`} x1={p.sx} y1={p.sy} x2={q.sx} y2={q.sy} />;
          })}
        </g>
        <g fontFamily={FONT_MONO} fontSize={11} fill="var(--muted-foreground)" aria-hidden>
          {axisEnds.map(({ label, from, to }) => (
            <g key={label}>
              <line x1={from.sx} y1={from.sy} x2={to.sx} y2={to.sy} stroke="var(--muted-foreground)" strokeWidth={1.4} />
              <text x={to.sx + (to.sx - cx) * 0.06} y={to.sy + (to.sy - cy) * 0.06} textAnchor="middle" dy="0.35em">
                {label}
              </text>
            </g>
          ))}
        </g>

        {projected.map(({ point, sx, sy, depth }) => {
          const near = nearness(depth);
          const r = point.r * (0.7 + 0.5 * near) * Math.sqrt(view.zoom);
          return (
            <g key={point.id}>
              <circle
                cx={sx}
                cy={sy}
                r={hovered === point.id ? r + 2 : r}
                fill={point.color}
                fillOpacity={0.35 + 0.55 * near}
                stroke="var(--background)"
                strokeWidth={1}
                className={point.onClick ? 'cursor-pointer' : undefined}
                onPointerMove={(event) => {
                  if (drag.current?.moved) return;
                  setHovered(point.id);
                  show(event, point.tooltip);
                }}
                onPointerLeave={() => {
                  setHovered(null);
                  hide();
                }}
                onClick={point.onClick}
              />
              {(labels.has(point.id) || hovered === point.id) && point.label && (
                <text
                  x={sx}
                  y={sy - r - 4}
                  textAnchor="middle"
                  fontSize={LABEL_FONT}
                  fill="var(--foreground)"
                  opacity={0.45 + 0.55 * near}
                  fontWeight={hovered === point.id ? 700 : 500}
                  paintOrder="stroke"
                  stroke="var(--background)"
                  strokeWidth={3}
                  strokeOpacity={0.7}
                  pointerEvents="none"
                >
                  {labels.get(point.id) ?? point.label}
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </ChartFrame>
  );
}

// Um gráfico que quebre com dado atípico não derruba a aba (ver with-chart-boundary).
export default withChartBoundary(Scatter3DChart);
