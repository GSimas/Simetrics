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

export default function SigmaGraph(props: SigmaGraphProps) {
  const { nodes, edges, height = 560, className, onNodeClick, exportName = 'rede', expanded } = props;
  const containerRef = useRef<HTMLDivElement>(null);
  const sigmaRef = useRef<Sigma | null>(null);
  const clickHandlerRef = useRef(onNodeClick);
  const [hovered, setHovered] = useState<RenderNode | null>(null);
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

  useEffect(() => {
    const container = containerRef.current;
    if (!container || nodes.length === 0) return;

    const graph = new Graph({ type: 'undirected', multi: false });

    // Cores de alto contraste adaptadas ao modo claro / escuro
    const labelColor = isDark ? '#f0eee6' : '#07110f';
    const highlightColor = isDark ? '#b8ff4a' : '#236e5e';
    const hoverBackground = isDark ? '#0e1d19' : '#f7f6f1';
    // Rótulos sobre um fundo do tema: numa rede densa as arestas se somam num branco
    // quase sólido e engoliam o texto claro do modo escuro.
    const labelBackground = isDark ? 'rgba(7, 17, 15, 0.82)' : 'rgba(247, 246, 241, 0.88)';
    // Arestas translúcidas: a sobreposição ainda mostra onde a rede é densa, sem virar
    // uma mancha que esconde nós e rótulos.
    const edgeColor = isDark ? 'rgba(156, 167, 162, 0.2)' : 'rgba(61, 72, 67, 0.22)';

    // Posição inicial em círculo para o ForceAtlas2, que não sai do lugar se todos os nós
    // começarem sobrepostos (com forças simétricas, o deslocamento resultante é zero).
    nodes.forEach((node, index) => {
      const angle = (2 * Math.PI * index) / nodes.length;
      graph.addNode(node.key, {
        label: node.label,
        size: Math.max(4, node.size / 4),
        color: communityColor(node.community),
        x: Math.cos(angle),
        y: Math.sin(angle),
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
  }, [nodes, edges, isDark]);

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
    graph.forEachEdge((edge, _attributes, source, target) => {
      const from = placed.get(source);
      const to = placed.get(target);
      const data = renderer.getEdgeDisplayData(edge);
      if (!from || !to || !data || data.hidden) return;
      parts.push(
        `<line x1="${from.x.toFixed(1)}" y1="${from.y.toFixed(1)}" x2="${to.x.toFixed(1)}" y2="${to.y.toFixed(1)}" stroke="${data.color}" stroke-width="${renderer.scaleSize(data.size).toFixed(2)}"/>`,
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
          {onNodeClick && <p className="eyebrow mt-2 text-highlight">Clique para abrir o perfil</p>}
        </div>
      )}
    </div>
  );
}
