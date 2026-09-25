import { useMemo, useRef, useState } from 'react';

import { ChartSearch } from '@/components/charts/ChartSearch';
import { matchKeys } from '@/components/charts/chart-search';
import { ExpandChartButton } from '@/components/charts/ExpandChartButton';
import { ExportImageButton } from '@/components/charts/ExportImageButton';
import { useSvgZoom, ZoomControls } from '@/components/charts/svg-zoom';
import { chordLayout } from '@/core/graph/chord';
import { imageFromSvgElement } from '@/lib/export-image';
import { numberLocale } from '@/lib/i18n/labels';
import { cn } from '@/lib/utils';
import { useLocale } from '@/state/locale.store';
import { withChartBoundary } from '@/components/with-chart-boundary';

/**
 * Grafo radial em diagrama de cordas: nós num círculo, agrupados por comunidade, rótulos
 * girados para fora e as ligações como cordas curvas pelo centro.
 *
 * SVG próprio porque o Sigma não gira rótulos ao longo do raio nem desenha cordas curvas. Passar o mouse sobre um nó mostra suas ligações; o primeiro clique fixa
 * o destaque, o segundo no mesmo nó chama `onNodeClick` (abrir o perfil no Motor de
 * Busca). Clicar no fundo desfaz o destaque. Roda do mouse ou botões aproximam; arrastar
 * move o diagrama. A busca destaca os nós cujo rótulo contém o texto digitado.
 */
export interface RadialNode {
  key: string;
  label: string;
  /** Tamanho do nó e ordem dentro do grupo. */
  weight: number;
  group?: number;
  color: string;
}

export interface RadialEdge {
  source: string;
  target: string;
  weight: number;
}

export interface RadialGraphProps {
  nodes: readonly RadialNode[];
  edges: readonly RadialEdge[];
  /** Unidade do peso no destaque de hover ("documentos"). */
  weightLabel: string;
  legend?: readonly { label: string; color: string }[] | undefined;
  onNodeClick?: (key: string) => void;
  exportName?: string;
  className?: string;
  /** Dentro da janela ampliada: sem o botão de ampliar e com altura da tela. */
  expanded?: boolean;
}

// Margem em volta do círculo para os rótulos: até MAX_LABEL caracteres na maior fonte
// (~0,55 em por caractere) ficam dentro do quadro, sem corte nas bordas.
const SIZE = 1100;
const CENTER = SIZE / 2;
const RADIUS = 300;
const LABEL_GAP = 12;
const MAX_LABEL = 30;

function truncate(text: string): string {
  return text.length > MAX_LABEL ? `${text.slice(0, MAX_LABEL - 1)}…` : text;
}

function RadialGraph(props: RadialGraphProps) {
  const { nodes, edges, weightLabel, legend, onNodeClick, className, expanded } = props;
  const t = useLocale((state) => state.t);
  const locale = useLocale((state) => state.locale);
  const exportName = props.exportName ?? (locale === 'en' ? 'radial-graph' : 'grafo-radial');
  const svgRef = useRef<SVGSVGElement>(null);
  const zoom = useSvgZoom(svgRef, SIZE, SIZE);
  const [hovered, setHovered] = useState<string | null>(null);
  const [chosen, setSelected] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const matches = useMemo(() => matchKeys(nodes, query, (node) => node.key, (node) => node.label), [nodes, query]);
  // Com outros dados (outro tipo de rede), um nó escolhido antes pode não existir mais.
  const selected = chosen !== null && nodes.some((node) => node.key === chosen) ? chosen : null;
  // O destaque fixo (clique) tem prioridade sobre o de passagem (hover).
  const focus = selected ?? hovered;

  const handleNodeClick = (key: string): void => {
    if (selected === key) onNodeClick?.(key);
    else setSelected(key);
  };

  const layout = useMemo(() => {
    const byKey = new Map(nodes.map((node) => [node.key, node]));
    const positions = chordLayout(
      nodes.map((node) => ({ key: node.key, weight: node.weight, group: node.group ?? 0 })),
    );
    const maxWeight = Math.max(...nodes.map((node) => node.weight), 1);
    const maxEdge = Math.max(...edges.map((edge) => edge.weight), 1);
    // Mais nós, letra menor: o perímetro é fixo.
    const fontSize = nodes.length <= 40 ? 15 : nodes.length <= 70 ? 12 : 10;

    const placed = new Map(
      positions.map((position) => {
        const node = byKey.get(position.key)!;
        return [
          position.key,
          {
            ...position,
            node,
            px: CENTER + RADIUS * position.x,
            py: CENTER + RADIUS * position.y,
            r: 3 + (node.weight / maxWeight) * 11,
          },
        ];
      }),
    );

    const chords = edges
      .filter((edge) => placed.has(edge.source) && placed.has(edge.target))
      .map((edge) => {
        const from = placed.get(edge.source)!;
        const to = placed.get(edge.target)!;
        // Controle da curva puxado para o centro: cordas entre nós distantes passam perto
        // dele, entre vizinhos fazem um arco raso junto à borda.
        const cx = CENTER + ((from.px + to.px) / 2 - CENTER) * 0.2;
        const cy = CENTER + ((from.py + to.py) / 2 - CENTER) * 0.2;
        return {
          ...edge,
          d: `M${from.px.toFixed(1)},${from.py.toFixed(1)} Q${cx.toFixed(1)},${cy.toFixed(1)} ${to.px.toFixed(1)},${to.py.toFixed(1)}`,
          color: from.node.color,
          width: 0.6 + (edge.weight / maxEdge) * 3.4,
        };
      });

    return { placed: [...placed.values()], chords, fontSize };
  }, [nodes, edges]);

  const neighbours = useMemo(() => {
    // Sem nó em foco, a busca decide o destaque.
    if (!focus) return matches && matches.size > 0 ? matches : null;
    const keys = new Set([focus]);
    for (const edge of edges) {
      if (edge.source === focus) keys.add(edge.target);
      if (edge.target === focus) keys.add(edge.source);
    }
    return keys;
  }, [focus, edges, matches]);

  const hoveredNode = focus ? layout.placed.find((item) => item.key === focus) : undefined;
  const hoveredDegree = neighbours ? neighbours.size - 1 : 0;


  return (
    <div className={cn('space-y-3', className)}>
      <ChartSearch value={query} onChange={setQuery} found={matches?.size ?? null} />
      <div className="flex flex-wrap items-center justify-between gap-3">
        {legend && legend.length > 0 ? (
          <div className="flex flex-wrap gap-x-4 gap-y-1.5">
            {legend.map((item) => (
              <span key={item.label} className="flex items-center gap-2 text-sm">
                <span className="size-2.5 rounded-full" style={{ background: item.color }} />
                {item.label}
              </span>
            ))}
          </div>
        ) : (
          <span />
        )}
        <div className="flex gap-1.5">
          {!expanded && (
            <ExpandChartButton>
              <RadialGraph {...props} expanded />
            </ExpandChartButton>
          )}
          <ExportImageButton
            filename={exportName}
            getImage={() => (svgRef.current ? imageFromSvgElement(svgRef.current) : null)}
          />
        </div>
      </div>

      <div className="relative overflow-hidden border border-border">
        {hoveredNode && (
          <div className="pointer-events-none absolute left-3 top-3 z-10 max-w-xs border border-border bg-popover/95 p-3 text-xs shadow-lg animate-in fade-in-0">
            <p className="mb-1 font-semibold break-words">{hoveredNode.node.label}</p>
            <p className="text-muted-foreground">
              <span className="tabular-nums text-foreground">{hoveredNode.node.weight.toLocaleString(numberLocale(locale))}</span>{' '}
              {weightLabel} ·{' '}
              <span className="tabular-nums text-foreground">{hoveredDegree}</span> {t('radial_links')}
            </p>
            {onNodeClick && (
              <p className="eyebrow mt-2 text-highlight">
                {selected === hoveredNode.key ? t('map_click_open') : t('radial_click_hint')}
              </p>
            )}
          </div>
        )}

        <svg
          ref={svgRef}
          viewBox={`0 0 ${SIZE} ${SIZE}`}
          className={cn('mx-auto block w-full select-none', expanded ? 'h-[74dvh]' : 'h-auto max-w-[820px]')}
          role="img"
          aria-label={t('radial_aria').replace('{count}', String(nodes.length))}
          xmlns="http://www.w3.org/2000/svg"
          fontFamily="Manrope, system-ui, sans-serif"
          onClick={(event) => {
            // Clique fora de um nó desfaz o destaque fixo.
            if (event.target === event.currentTarget) setSelected(null);
          }}
          {...zoom.svgProps}
        >
          <g style={zoom.style}>
          <g fill="none">
            {layout.chords.map((chord) => {
              const active = focus
                ? chord.source === focus || chord.target === focus
                : !neighbours || neighbours.has(chord.source) || neighbours.has(chord.target);
              return (
                <path
                  key={`${chord.source}→${chord.target}`}
                  d={chord.d}
                  stroke={chord.color}
                  strokeWidth={chord.width / zoom.k}
                  strokeOpacity={neighbours ? (active ? 0.85 : 0.04) : 0.28}
                  className="transition-[stroke-opacity] duration-200"
                />
              );
            })}
          </g>

          {layout.placed.map((item) => {
            const degrees = (item.angle * 180) / Math.PI;
            // Lado esquerdo do círculo: o texto giraria de cabeça para baixo, então vira
            // 180° e passa a crescer para a esquerda, a partir do nó.
            const flip = Math.cos(item.angle) < 0;
            const dimmed = neighbours !== null && !neighbours.has(item.key);
            // O nó cresce menos que o zoom; o rótulo acompanha a borda do nó.
            const r = item.r / Math.sqrt(zoom.k);
            return (
              <g
                key={item.key}
                className={cn(
                  'transition-opacity duration-200',
                  onNodeClick && 'cursor-pointer',
                  dimmed ? 'opacity-25' : 'opacity-100',
                )}
                onMouseEnter={() => setHovered(item.key)}
                onMouseLeave={() => setHovered(null)}
                onClick={() => handleNodeClick(item.key)}
              >
                <circle
                  cx={item.px}
                  cy={item.py}
                  r={r}
                  fill={item.node.color}
                  stroke={selected === item.key || matches?.has(item.key) ? 'var(--foreground)' : 'none'}
                  strokeWidth={2 / zoom.k}
                />
                <text
                  transform={`translate(${item.px} ${item.py}) rotate(${flip ? degrees + 180 : degrees})`}
                  x={flip ? -(r + LABEL_GAP) : r + LABEL_GAP}
                  dy="0.35em"
                  textAnchor={flip ? 'end' : 'start'}
                  fontSize={layout.fontSize}
                  fill="var(--foreground)"
                  fontWeight={focus === item.key || matches?.has(item.key) ? 700 : 500}
                >
                  {truncate(item.node.label)}
                </text>
              </g>
            );
          })}
          </g>
        </svg>
        <ZoomControls zoom={zoom} />
      </div>
    </div>
  );
}

// Um gráfico que quebre com dado atípico não derruba a aba (ver with-chart-boundary).
export default withChartBoundary(RadialGraph);
