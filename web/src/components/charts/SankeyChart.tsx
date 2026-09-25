import { useMemo, useRef, useState } from 'react';

import { expandedHeight } from '@/components/charts/ExpandChartButton';
import { ChartFrame, TipRow } from '@/components/charts/svg/chart-kit';
import { useElementWidth, useTooltip } from '@/components/charts/svg/hooks';
import { fitText, FONT_MONO, FONT_SANS } from '@/components/charts/svg/text';
import { formatNumber } from '@/components/charts/svg/scale';
import { useLocale } from '@/state/locale.store';
import { withChartBoundary } from '@/components/with-chart-boundary';

/**
 * Diagrama de Sankey em SVG, no lugar do traço `sankey` do Plotly.
 *
 * Layout próprio, em colunas: a altura de cada nó é o maior entre o que entra e o que
 * sai; a primeira coluna é ordenada pelo peso e as seguintes pelo baricentro das origens
 * (a média da altura dos nós que chegam nele), o que desembaraça boa parte dos
 * cruzamentos. Passar o mouse num nó acende só os fluxos dele; clicar abre o perfil.
 */
export interface SankeyChartNode {
  term: string;
  column: number;
  color: string;
}

export interface SankeyChartLink {
  source: number;
  target: number;
  value: number;
  color: string;
  /** Descrição do tipo de fluxo, na dica. */
  kind?: string;
}

export interface SankeyChartProps {
  nodes: readonly SankeyChartNode[];
  links: readonly SankeyChartLink[];
  columnLabels: readonly string[];
  height?: number;
  exportName: string;
  onNodeClick?: ((term: string) => void) | undefined;
  expanded?: boolean;
  className?: string;
}

const NODE_WIDTH = 14;
const NODE_PAD = 10;
const HEADER = 28;
const LABEL_FONT = 11;
/** Passo mínimo entre nós: um nó sem fluxo ainda precisa de espaço para o rótulo. */
const MIN_STEP = LABEL_FONT + 5;

interface PlacedNode {
  index: number;
  node: SankeyChartNode;
  value: number;
  x: number;
  y: number;
  h: number;
  inValue: number;
  outValue: number;
}

interface PlacedLink {
  index: number;
  link: SankeyChartLink;
  path: string;
  thickness: number;
}

function layout(
  nodes: readonly SankeyChartNode[],
  links: readonly SankeyChartLink[],
  width: number,
  height: number,
  columns: number,
): { placed: PlacedNode[]; paths: PlacedLink[]; columnX: number[] } {
  const inValue = new Array<number>(nodes.length).fill(0);
  const outValue = new Array<number>(nodes.length).fill(0);
  for (const link of links) {
    outValue[link.source] = (outValue[link.source] ?? 0) + link.value;
    inValue[link.target] = (inValue[link.target] ?? 0) + link.value;
  }

  // Rótulos: à direita do nó nas colunas iniciais, à esquerda na última.
  const labelRoom = Math.min(180, Math.max(90, width * 0.16));
  const left = 8;
  const right = width - labelRoom - 8;
  const columnX = Array.from({ length: columns }, (_, column) =>
    columns === 1 ? left : left + ((right - left - NODE_WIDTH) * column) / (columns - 1),
  );
  // A última coluna leva o rótulo à esquerda; o espaço à direita volta para o desenho.
  if (columns > 1) columnX[columns - 1] = width - 8 - NODE_WIDTH;

  const byColumn: number[][] = Array.from({ length: columns }, () => []);
  nodes.forEach((node, index) => byColumn[Math.min(columns - 1, node.column)]?.push(index));

  const valueOf = (index: number): number => Math.max(inValue[index] ?? 0, outValue[index] ?? 0, 1e-6);
  const available = height - HEADER - 8;
  // Duas passadas: a primeira escala acha os nós pequenos demais para o rótulo; eles
  // passam a ocupar o passo mínimo, e a segunda escala divide o resto entre os demais.
  const scale = Math.min(
    ...byColumn
      .filter((column) => column.length > 0)
      .map((column) => {
        const total = column.reduce((sum, index) => sum + valueOf(index), 0);
        const first = (available - NODE_PAD * (column.length - 1)) / total;
        const small = column.filter((index) => valueOf(index) * first + NODE_PAD < MIN_STEP);
        if (small.length === 0) return first;
        const large = column.filter((index) => !small.includes(index));
        const largeTotal = large.reduce((sum, index) => sum + valueOf(index), 0);
        if (largeTotal === 0) return first;
        return Math.max(0.0001, (available - small.length * MIN_STEP - NODE_PAD * large.length) / largeTotal);
      }),
  );

  const center = new Map<number, number>();
  const placed = new Map<number, PlacedNode>();

  byColumn.forEach((column, columnIndex) => {
    const order =
      columnIndex === 0
        ? [...column].sort((a, b) => valueOf(b) - valueOf(a))
        : [...column].sort((a, b) => barycenter(a) - barycenter(b));
    const step = (index: number): number => Math.max(valueOf(index) * scale + NODE_PAD, MIN_STEP);
    const total = order.reduce((sum, index) => sum + step(index), 0) - NODE_PAD;
    let y = HEADER + Math.max(0, (available - total) / 2);
    for (const index of order) {
      const h = Math.max(2, valueOf(index) * scale);
      placed.set(index, {
        index,
        node: nodes[index] as SankeyChartNode,
        value: valueOf(index),
        x: columnX[columnIndex] as number,
        y,
        h,
        inValue: inValue[index] ?? 0,
        outValue: outValue[index] ?? 0,
      });
      center.set(index, y + h / 2);
      y += step(index);
    }
  });

  function barycenter(index: number): number {
    let weighted = 0;
    let total = 0;
    for (const link of links) {
      if (link.target !== index) continue;
      const y = center.get(link.source);
      if (y === undefined) continue;
      weighted += y * link.value;
      total += link.value;
    }
    return total > 0 ? weighted / total : Number.MAX_SAFE_INTEGER - valueOf(index);
  }

  // Cada fluxo ocupa uma faixa dentro do nó de origem e do de destino, na ordem vertical
  // do outro lado — assim as faixas saem e chegam sem se cruzar dentro do nó.
  const sourceOffset = new Map<number, number>();
  const targetOffset = new Map<number, number>();
  const sorted = links
    .map((link, index) => ({ link, index }))
    .filter(({ link }) => placed.has(link.source) && placed.has(link.target));
  const bySourceOrder = [...sorted].sort(
    (a, b) => (placed.get(a.link.target)?.y ?? 0) - (placed.get(b.link.target)?.y ?? 0),
  );
  const byTargetOrder = [...sorted].sort(
    (a, b) => (placed.get(a.link.source)?.y ?? 0) - (placed.get(b.link.source)?.y ?? 0),
  );
  const sy0 = new Map<number, number>();
  const ty0 = new Map<number, number>();
  for (const { link, index } of bySourceOrder) {
    const node = placed.get(link.source) as PlacedNode;
    const offset = sourceOffset.get(link.source) ?? 0;
    sy0.set(index, node.y + offset);
    sourceOffset.set(link.source, offset + link.value * scale);
  }
  for (const { link, index } of byTargetOrder) {
    const node = placed.get(link.target) as PlacedNode;
    const offset = targetOffset.get(link.target) ?? 0;
    ty0.set(index, node.y + offset);
    targetOffset.set(link.target, offset + link.value * scale);
  }

  const paths = sorted.map(({ link, index }): PlacedLink => {
    const source = placed.get(link.source) as PlacedNode;
    const target = placed.get(link.target) as PlacedNode;
    const thickness = Math.max(0.8, link.value * scale);
    const x0 = source.x + NODE_WIDTH;
    const x1 = target.x;
    const a = sy0.get(index) as number;
    const b = ty0.get(index) as number;
    const mid = (x0 + x1) / 2;
    const path =
      `M${x0},${a}C${mid},${a} ${mid},${b} ${x1},${b}` +
      `L${x1},${b + thickness}C${mid},${b + thickness} ${mid},${a + thickness} ${x0},${a + thickness}Z`;
    return { index, link, path, thickness };
  });

  return { placed: [...placed.values()], paths, columnX };
}

function SankeyChart(props: SankeyChartProps) {
  const { nodes, links, columnLabels, height = 620, exportName, onNodeClick, expanded, className } = props;
  const t = useLocale((state) => state.t);
  const locale = useLocale((state) => state.locale);
  const svgRef = useRef<SVGSVGElement>(null);
  const [containerRef, width] = useElementWidth<HTMLDivElement>();
  const { tooltip, show, hide } = useTooltip(containerRef);
  const [focusNode, setFocusNode] = useState<number | null>(null);
  const [focusLink, setFocusLink] = useState<number | null>(null);

  const columns = Math.max(columnLabels.length, ...nodes.map((node) => node.column + 1), 1);
  const chart = useMemo(() => layout(nodes, links, width, height, columns), [nodes, links, width, height, columns]);

  const lastColumn = columns - 1;
  const labelWidth = Math.min(180, Math.max(90, width * 0.16)) - 8;

  const linkActive = (entry: PlacedLink): boolean =>
    focusLink !== null
      ? entry.index === focusLink
      : focusNode !== null
        ? entry.link.source === focusNode || entry.link.target === focusNode
        : true;

  const reset = (): void => {
    setFocusNode(null);
    setFocusLink(null);
    hide();
  };

  // Resumo para leitores de tela: tamanho do fluxo e os períodos.
  const summary = `${t('chart_summary_flows')
    .replace('{nodes}', String(nodes.length))
    .replace('{links}', String(links.length))} ${columnLabels.join(' → ')}.`;

  return (
    <ChartFrame
      exportName={exportName}
      svgRef={svgRef}
      containerRef={containerRef}
      width={width}
      tooltip={tooltip}
      className={className}
      expandedContent={
        expanded ? undefined : <SankeyChart {...props} expanded height={Math.max(height, expandedHeight())} />
      }
    >
      <svg
        ref={svgRef}
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        className="block select-none"
        role="img"
        aria-label={t('visual_tab_sankey')}
        xmlns="http://www.w3.org/2000/svg"
        fontFamily={FONT_SANS}
        onPointerLeave={reset}
      >
        <desc>{summary}</desc>
        <g fontFamily={FONT_MONO} fontSize={11} fill="var(--muted-foreground)" aria-hidden>
          {columnLabels.map((label, column) => (
            <text
              key={label}
              x={(chart.columnX[column] ?? 0) + (column === lastColumn ? NODE_WIDTH : 0)}
              y={14}
              textAnchor={column === lastColumn && columns > 1 ? 'end' : 'start'}
            >
              {label}
            </text>
          ))}
        </g>

        <g>
          {chart.paths.map((entry) => {
            const active = linkActive(entry);
            return (
              <path
                key={entry.index}
                d={entry.path}
                fill={entry.link.color}
                opacity={active ? 1 : 0.2}
                className="transition-opacity duration-200"
                onPointerMove={(event) => {
                  setFocusLink(entry.index);
                  setFocusNode(null);
                  const source = nodes[entry.link.source];
                  const target = nodes[entry.link.target];
                  show(
                    event,
                    <>
                      <p className="mb-1 font-semibold break-words">
                        {source?.term} → {target?.term}
                      </p>
                      <TipRow label={entry.link.kind ?? (locale === 'en' ? 'Flow' : 'Fluxo')} value={formatNumber(entry.link.value, undefined, locale)} color={entry.link.color} />
                    </>,
                  );
                }}
                onPointerLeave={() => setFocusLink(null)}
              />
            );
          })}
        </g>

        <g>
          {chart.placed.map((item) => {
            const onLeft = item.node.column === lastColumn && columns > 1;
            const dimmed =
              focusNode !== null &&
              focusNode !== item.index &&
              !links.some(
                (link) =>
                  (link.source === focusNode && link.target === item.index) ||
                  (link.target === focusNode && link.source === item.index),
              );
            return (
              <g
                key={item.index}
                opacity={dimmed ? 0.35 : 1}
                className={onNodeClick ? 'cursor-pointer transition-opacity duration-200' : 'transition-opacity duration-200'}
                onPointerMove={(event) => {
                  setFocusNode(item.index);
                  setFocusLink(null);
                  show(
                    event,
                    <>
                      <p className="mb-1 font-semibold break-words">{item.node.term}</p>
                      <p className="eyebrow mb-1">{columnLabels[item.node.column]}</p>
                      <TipRow label={locale === 'en' ? 'Inflow' : 'Entrada'} value={formatNumber(item.inValue, undefined, locale)} />
                      <TipRow label={locale === 'en' ? 'Outflow' : 'Saída'} value={formatNumber(item.outValue, undefined, locale)} />
                    </>,
                  );
                }}
                onClick={onNodeClick ? () => onNodeClick(item.node.term) : undefined}
              >
                <rect
                  x={item.x}
                  y={item.y}
                  width={NODE_WIDTH}
                  height={item.h}
                  fill={item.node.color}
                  stroke="var(--background)"
                  strokeWidth={0.5}
                />
                <text
                  x={onLeft ? item.x - 6 : item.x + NODE_WIDTH + 6}
                  y={item.y + item.h / 2}
                  dy="0.35em"
                  textAnchor={onLeft ? 'end' : 'start'}
                  fontSize={LABEL_FONT}
                  fill="var(--foreground)"
                  fontWeight={focusNode === item.index ? 700 : 500}
                  paintOrder="stroke"
                  stroke="var(--background)"
                  strokeWidth={3}
                  strokeOpacity={0.8}
                >
                  {fitText(item.node.term, labelWidth, LABEL_FONT)}
                </text>
              </g>
            );
          })}
        </g>
      </svg>
    </ChartFrame>
  );
}

// Um gráfico que quebre com dado atípico não derruba a aba (ver with-chart-boundary).
export default withChartBoundary(SankeyChart);
