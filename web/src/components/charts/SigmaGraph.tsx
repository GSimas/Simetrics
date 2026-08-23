import { useEffect, useRef, useState } from 'react';
import Graph from 'graphology';
import forceAtlas2 from 'graphology-layout-forceatlas2';
import Sigma from 'sigma';

import type { RenderEdge, RenderNode } from '@/core/graph';
import { cn } from '@/lib/utils';

/**
 * Renderizador de rede com Sigma.js — substitui o `streamlit_agraph`.
 *
 * O Sigma desenha em WebGL, então milhares de nós continuam fluidos, ao contrário do
 * canvas do agraph.
 */

/**
 * Paleta das comunidades detectadas pelo Louvain.
 *
 * Cores qualitativas e distinguíveis entre si, incluindo para as formas mais comuns de
 * daltonismo — a cor aqui codifica pertencimento a um agrupamento, não intensidade, então
 * um gradiente seria a escolha errada.
 */
const COMMUNITY_COLORS = [
  '#1273B9', '#E8734A', '#3FA96C', '#A05FC4', '#D8A13A',
  '#4BAFC9', '#D45D79', '#7A8B99', '#8C6D46', '#5C6BC0',
] as const;

const EDGE_COLOR = 'rgba(130, 140, 150, 0.28)';

export interface SigmaGraphProps {
  nodes: readonly RenderNode[];
  edges: readonly RenderEdge[];
  /** Altura da área de desenho. */
  height?: number;
  className?: string;
  onNodeClick?: (key: string) => void;
}

export default function SigmaGraph({
  nodes,
  edges,
  height = 560,
  className,
  onNodeClick,
}: SigmaGraphProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const sigmaRef = useRef<Sigma | null>(null);
  const clickHandlerRef = useRef(onNodeClick);
  const [hovered, setHovered] = useState<RenderNode | null>(null);

  // O handler fica numa ref para que trocá-lo não force a reconstrução do grafo.
  useEffect(() => {
    clickHandlerRef.current = onNodeClick;
  }, [onNodeClick]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || nodes.length === 0) return;

    const graph = new Graph({ type: 'undirected', multi: false });

    // Posição inicial em círculo. O ForceAtlas2 não sai do lugar se todos os nós começarem
    // sobrepostos: com forças simétricas, o deslocamento resultante é zero.
    nodes.forEach((node, index) => {
      const angle = (2 * Math.PI * index) / nodes.length;
      graph.addNode(node.key, {
        label: node.label,
        size: Math.max(4, node.size / 4),
        color: COMMUNITY_COLORS[node.community % COMMUNITY_COLORS.length] as string,
        x: Math.cos(angle),
        y: Math.sin(angle),
      });
    });

    for (const edge of edges) {
      if (!graph.hasNode(edge.source) || !graph.hasNode(edge.target)) continue;
      if (graph.hasEdge(edge.source, edge.target)) continue;
      graph.addEdge(edge.source, edge.target, {
        size: Math.min(4, 0.4 + Math.log2(edge.weight + 1)),
        color: EDGE_COLOR,
      });
    }

    // Layout síncrono: as redes visualizadas são recortadas por top-N (dezenas de nós),
    // então algumas centenas de iterações levam milissegundos.
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

    const renderer = new Sigma(graph, container, {
      renderLabels: true,
      labelDensity: 0.5,
      labelRenderedSizeThreshold: 5,
      defaultEdgeColor: EDGE_COLOR,
      minCameraRatio: 0.05,
      maxCameraRatio: 12,
    });
    sigmaRef.current = renderer;

    const byKey = new Map(nodes.map((node) => [node.key, node]));
    renderer.on('enterNode', ({ node }) => setHovered(byKey.get(node) ?? null));
    renderer.on('leaveNode', () => setHovered(null));
    renderer.on('clickNode', ({ node }) => clickHandlerRef.current?.(node));

    return () => {
      renderer.kill();
      sigmaRef.current = null;
    };
  }, [nodes, edges]);

  return (
    <div className={cn('relative w-full overflow-hidden rounded-lg border', className)}>
      <div ref={containerRef} style={{ height }} />

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
        </div>
      )}
    </div>
  );
}
