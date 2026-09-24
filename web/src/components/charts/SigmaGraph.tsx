import { useEffect, useRef, useState } from 'react';
import Graph from 'graphology';
import forceAtlas2 from 'graphology-layout-forceatlas2';
import Sigma from 'sigma';

import type { RenderEdge, RenderNode } from '@/core/graph';
import { communityColor } from '@/features/overview/viz-shared';
import { ExpandChartButton, expandedHeight } from '@/components/charts/ExpandChartButton';
import { ExportImageButton } from '@/components/charts/ExportImageButton';
import type { ChartImage } from '@/lib/export-image';
import { cn } from '@/lib/utils';
import { useLocale } from '@/state/locale.store';
import { getGraphWorker } from '@/workers/client';
import { withChartBoundary } from '@/components/with-chart-boundary';

/**
 * Renderizador de rede com Sigma.js — substitui o `streamlit_agraph`.
 *
 * O Sigma desenha em WebGL, então milhares de nós continuam fluidos, ao contrário do
 * canvas do agraph.
 */

export interface SigmaGraphProps {
  nodes: readonly RenderNode[];
  edges: readonly RenderEdge[];
  /** Altura da área de desenho. */
  height?: number;
  className?: string;
  onNodeClick?: (key: string) => void;
  /** Nome do arquivo ao exportar a imagem. */
  exportName?: string;
  /** Dentro da janela ampliada: sem o botão de ampliar e com altura da tela. */
  expanded?: boolean;
}

type Positions = Record<string, [number, number]>;

/**
 * Cor das arestas: tinta sobre o papel no tema claro, papel sobre a tinta no escuro, com
 * contraste de ~2:1 contra o fundo. Translúcidas, a sobreposição ainda mostra onde a rede
 * é densa.
 *
 * O WebGL do Sigma mistura esperando cor pré-multiplicada (ONE, ONE_MINUS_SRC_ALPHA), mas
 * escreve a cor como a recebe: uma aresta escura translúcida saía mais clara que o papel
 * (linhas brancas no tema claro) e uma clara saía quase opaca no escuro. Por isso o WebGL
 * recebe a versão pré-multiplicada (rgb × alfa), que resulta exatamente na cor pretendida;
 * o SVG exportado, que mistura do jeito normal, recebe a original.
 */
function edgeColors(isDark: boolean): { webgl: string; svg: string } {
  const [r, g, b, alpha] = isDark ? [240, 238, 230, 0.3] : [7, 17, 15, 0.35];
  const pre = (channel: number): number => Math.round(channel * alpha);
  return {
    webgl: `rgba(${pre(r)}, ${pre(g)}, ${pre(b)}, ${alpha})`,
    svg: `rgba(${r}, ${g}, ${b}, ${alpha})`,
  };
}

/** Resumo para leitores de tela: tamanho da rede e os nós mais conectados. */
function describeGraph(nodes: readonly RenderNode[], edges: readonly RenderEdge[], template: string, mostConnected: string): string {
  const top = [...nodes]
    .sort((a, b) => b.degreeAbsolute - a.degreeAbsolute)
    .slice(0, 5)
    .map((node) => node.label);
  const summary = template.replace('{nodes}', String(nodes.length)).replace('{edges}', String(edges.length));
  if (top.length === 0) return summary;
  const list = top.join(', ');
  return `${summary} ${mostConnected}: ${list}${list.endsWith('.') ? '' : '.'}`;
}

/** Mesmas configurações do layout do worker — usadas só se o worker falhar. */
function layoutOnMainThread(nodes: readonly RenderNode[], edges: readonly RenderEdge[]): Positions {
  const graph = new Graph({ type: 'undirected', multi: false });
  nodes.forEach((node, index) => {
    const angle = (2 * Math.PI * index) / nodes.length;
    graph.addNode(node.key, { x: Math.cos(angle), y: Math.sin(angle) });
  });
  for (const edge of edges) {
    if (!graph.hasNode(edge.source) || !graph.hasNode(edge.target)) continue;
    if (graph.hasEdge(edge.source, edge.target)) continue;
    graph.addEdge(edge.source, edge.target);
  }
  if (graph.order > 1) {
    forceAtlas2.assign(graph, {
      iterations: 260,
      settings: {
        ...forceAtlas2.inferSettings(graph),
        gravity: 1.1,
        scalingRatio: 12,
        barnesHutOptimize: graph.order > 200,
      },
    });
  }
  const positions: Positions = {};
  graph.forEachNode((key, attributes) => {
    positions[key] = [attributes['x'] as number, attributes['y'] as number];
  });
  return positions;
}

/**
 * Layout por rede, calculado uma vez: trocar o tema ou abrir a janela ampliada reusa as
 * posições em vez de rodar o ForceAtlas2 de novo.
 */
const layoutCache = new WeakMap<readonly RenderNode[], { edges: readonly RenderEdge[]; positions: Promise<Positions> }>();

function requestLayout(nodes: readonly RenderNode[], edges: readonly RenderEdge[]): Promise<Positions> {
  const cached = layoutCache.get(nodes);
  if (cached && cached.edges === edges) return cached.positions;
  const positions = getGraphWorker()
    .layout(
      nodes.map((node) => node.key),
      edges.map((edge) => [edge.source, edge.target] as const),
    )
    .catch(() => layoutOnMainThread(nodes, edges));
  layoutCache.set(nodes, { edges, positions });
  return positions;
}

function SigmaGraph(props: SigmaGraphProps) {
  const { nodes, edges, height = 560, className, onNodeClick, exportName = 'rede', expanded } = props;
  const containerRef = useRef<HTMLDivElement>(null);
  const sigmaRef = useRef<Sigma | null>(null);
  const clickHandlerRef = useRef(onNodeClick);
  const [hovered, setHovered] = useState<RenderNode | null>(null);
  const t = useLocale((state) => state.t);
  const [layout, setLayout] = useState<{ nodes: readonly RenderNode[]; positions: Positions } | null>(null);
  const positions = layout?.nodes === nodes ? layout.positions : null;
  const [isDark, setIsDark] = useState<boolean>(() =>
    typeof document !== 'undefined' ? document.documentElement.classList.contains('dark') : false,
  );

  // O handler fica numa ref para que trocá-lo não force a reconstrução do grafo.
  useEffect(() => {
    clickHandlerRef.current = onNodeClick;
  }, [onNodeClick]);

  // Monitora alterações de tema (dark/light) no elemento raiz para sincronizar os contrastes do WebGL
  useEffect(() => {
    if (typeof document === 'undefined') return;
    const observer = new MutationObserver(() => {
      setIsDark(document.documentElement.classList.contains('dark'));
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    return () => observer.disconnect();
  }, []);

  // O ForceAtlas2 roda no worker de grafo; a aba não trava enquanto ele calcula.
  useEffect(() => {
    if (nodes.length === 0) return;
    let cancelled = false;
    void requestLayout(nodes, edges).then((result) => {
      if (!cancelled) setLayout({ nodes, positions: result });
    });
    return () => {
      cancelled = true;
    };
  }, [nodes, edges]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || nodes.length === 0 || !positions) return;

    const graph = new Graph({ type: 'undirected', multi: false });

    // Cores de alto contraste adaptadas ao modo claro / escuro
    const labelColor = isDark ? '#f0eee6' : '#07110f';
    const highlightColor = isDark ? '#b8ff4a' : '#236e5e';
    const hoverBackground = isDark ? '#0e1d19' : '#f7f6f1';
    // Rótulos sobre um fundo do tema: numa rede densa as arestas se somam num branco
    // quase sólido e engoliam o texto claro do modo escuro.
    const labelBackground = isDark ? 'rgba(7, 17, 15, 0.82)' : 'rgba(247, 246, 241, 0.88)';
    const edgeColor = edgeColors(isDark).webgl;

    // Posições já calculadas pelo ForceAtlas2 (no worker).
    nodes.forEach((node) => {
      const [x, y] = positions[node.key] ?? [0, 0];
      graph.addNode(node.key, {
        label: node.label,
        size: Math.max(4, node.size / 4),
        color: communityColor(node.community),
        x,
        y,
      });
    });

    for (const edge of edges) {
      if (!graph.hasNode(edge.source) || !graph.hasNode(edge.target)) continue;
      if (graph.hasEdge(edge.source, edge.target)) continue;
      graph.addEdge(edge.source, edge.target, {
        size: Math.max(1, Math.min(3.5, 0.6 + Math.log2(edge.weight + 1))),
        color: edgeColor,
      });
    }

    const renderer = new Sigma(graph, container, {
      renderLabels: true,
      labelDensity: 0.6,
      labelRenderedSizeThreshold: 3,
      labelFont: 'Manrope, system-ui, -apple-system, sans-serif',
      labelWeight: '600',
      labelSize: 12,
      labelColor: { color: labelColor },
      defaultEdgeColor: edgeColor,
      minEdgeThickness: 1.5,
      minCameraRatio: 0.05,
      maxCameraRatio: 12,
      defaultDrawNodeLabel: (context, data, settings) => {
        if (!data.label) return;
        const size = settings.labelSize;
        context.font = `${settings.labelWeight} ${size}px ${settings.labelFont}`;
        const x = data.x + data.size + 4;
        const width = context.measureText(data.label).width;
        context.fillStyle = labelBackground;
        context.fillRect(x - 3, data.y - size / 2 - 3, width + 6, size + 6);
        context.fillStyle = labelColor;
        context.fillText(data.label, x, data.y + size / 3);
      },
      // O hover padrão do Sigma pinta uma caixa branca sob o rótulo — um clarão no tema
      // escuro. Aqui o nó ganha um anel no tom de destaque e o rótulo, um fundo do tema.
      defaultDrawNodeHover: (context, data, settings) => {
        context.beginPath();
        context.arc(data.x, data.y, data.size + 3, 0, Math.PI * 2);
        context.strokeStyle = highlightColor;
        context.lineWidth = 2;
        context.stroke();

        if (!data.label) return;
        const size = settings.labelSize;
        context.font = `${settings.labelWeight} ${size}px ${settings.labelFont}`;
        const x = data.x + data.size + 6;
        const width = context.measureText(data.label).width;
        context.fillStyle = hoverBackground;
        context.fillRect(x - 5, data.y - size / 2 - 5, width + 10, size + 10);
        context.fillStyle = labelColor;
        context.fillText(data.label, x, data.y + size / 3);
      },
    });
    sigmaRef.current = renderer;

    const byKey = new Map(nodes.map((node) => [node.key, node]));
    renderer.on('enterNode', ({ node }) => {
      setHovered(byKey.get(node) ?? null);
      if (clickHandlerRef.current) container.style.cursor = 'pointer';
    });
    renderer.on('leaveNode', () => {
      setHovered(null);
      container.style.cursor = '';
    });
    renderer.on('clickNode', ({ node }) => clickHandlerRef.current?.(node));

    return () => {
      renderer.kill();
      sigmaRef.current = null;
    };
  }, [nodes, edges, isDark, positions]);

  /**
   * SVG montado a partir do grafo: o Sigma desenha em WebGL, sem SVG próprio. Posições
   * convertidas pela câmera atual, então o arquivo sai como o grafo está na tela.
   */
  const getImage = (): ChartImage | null => {
    const renderer = sigmaRef.current;
    const container = containerRef.current;
    if (!renderer || !container) return null;

    const width = Math.max(1, Math.round(container.clientWidth));
    const heightPx = Math.max(1, Math.round(container.clientHeight));
    const labelColor = isDark ? '#f0eee6' : '#07110f';
    const labelBackground = isDark ? 'rgba(7, 17, 15, 0.82)' : 'rgba(247, 246, 241, 0.88)';
    const escape = (text: string): string =>
      text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

    const graph = renderer.getGraph();
    const placed = new Map<string, { x: number; y: number; r: number; color: string; label: string }>();
    graph.forEachNode((key) => {
      const data = renderer.getNodeDisplayData(key);
      if (!data || data.hidden) return;
      const point = renderer.framedGraphToViewport({ x: data.x, y: data.y });
      placed.set(key, {
        x: point.x,
        y: point.y,
        r: renderer.scaleSize(data.size),
        color: data.color,
        label: data.label ?? '',
      });
    });

    const parts: string[] = [];
    const edgeStroke = edgeColors(isDark).svg;
    graph.forEachEdge((edge, _attributes, source, target) => {
      const from = placed.get(source);
      const to = placed.get(target);
      const data = renderer.getEdgeDisplayData(edge);
      if (!from || !to || !data || data.hidden) return;
      parts.push(
        `<line x1="${from.x.toFixed(1)}" y1="${from.y.toFixed(1)}" x2="${to.x.toFixed(1)}" y2="${to.y.toFixed(1)}" stroke="${edgeStroke}" stroke-width="${renderer.scaleSize(data.size).toFixed(2)}"/>`,
      );
    });
    for (const node of placed.values()) {
      parts.push(`<circle cx="${node.x.toFixed(1)}" cy="${node.y.toFixed(1)}" r="${node.r.toFixed(1)}" fill="${node.color}"/>`);
    }
    for (const node of placed.values()) {
      if (!node.label) continue;
      const x = node.x + node.r + 4;
      const approxWidth = node.label.length * 6.6;
      parts.push(
        `<rect x="${(x - 3).toFixed(1)}" y="${(node.y - 9).toFixed(1)}" width="${(approxWidth + 6).toFixed(1)}" height="18" fill="${labelBackground}"/>` +
          `<text x="${x.toFixed(1)}" y="${(node.y + 4).toFixed(1)}" font-size="12" font-weight="600" fill="${labelColor}">${escape(node.label)}</text>`,
      );
    }

    const svg =
      `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${heightPx}" viewBox="0 0 ${width} ${heightPx}" font-family="Manrope, system-ui, sans-serif">` +
      parts.join('') +
      '</svg>';
    return { svg, width, height: heightPx };
  };

  return (
    <div className={cn('relative w-full overflow-hidden rounded-lg border', className)}>
      <div className="absolute right-2 top-2 z-10 flex gap-1.5">
        {!expanded && (
          <ExpandChartButton>
            <SigmaGraph {...props} height={Math.max(height, expandedHeight())} expanded />
          </ExpandChartButton>
        )}
        <ExportImageButton filename={exportName} getImage={getImage} />
      </div>
      {/* Canvas WebGL: sem texto próprio, então o resumo vai no nome acessível. */}
      <div
        ref={containerRef}
        style={{ height }}
        role="img"
        aria-busy={nodes.length > 0 && !positions}
        aria-label={describeGraph(nodes, edges, t('sigma_aria'), t('chart_most_connected'))}
      />

      {nodes.length === 0 && (
        <div className="absolute inset-0 grid place-items-center text-sm text-muted-foreground">
          Nenhum nó para exibir com os filtros atuais.
        </div>
      )}

      {hovered && (
        <div className="pointer-events-none absolute left-3 top-3 max-w-xs rounded-md border bg-popover/95 p-3 text-xs shadow-lg backdrop-blur">
          <p className="mb-1 font-semibold break-words">{hovered.label}</p>
          <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-muted-foreground">
            <dt>Documentos</dt>
            <dd className="text-foreground tabular-nums">{hovered.count}</dd>
            <dt>Grau absoluto</dt>
            <dd className="text-foreground tabular-nums">{hovered.degreeAbsolute}</dd>
            <dt>Eigenvector</dt>
            <dd className="text-foreground tabular-nums">{hovered.eigenvector}</dd>
            <dt>Betweenness</dt>
            <dd className="text-foreground tabular-nums">{hovered.betweenness}</dd>
            <dt>Closeness</dt>
            <dd className="text-foreground tabular-nums">{hovered.closeness}</dd>
          </dl>
          {onNodeClick && <p className="eyebrow mt-2 text-highlight">Clique para abrir o perfil</p>}
        </div>
      )}
    </div>
  );
}

// Um gráfico que quebre com dado atípico não derruba a aba (ver with-chart-boundary).
export default withChartBoundary(SigmaGraph);
