import { useMemo } from 'react';

import type { RenderEdge, RenderNode } from '@/core/graph';
import { communityColor } from '@/features/overview/viz-shared';
import { useAsyncResult, identityKey } from '@/lib/use-async-result';
import { useLocale } from '@/state/locale.store';
import { getGraphWorker } from '@/workers/client';

/**
 * Rede de coocorrência em SVG, para o relatório.
 *
 * Na aba Redes o grafo é WebGL (Sigma), que não vira SVG na página. Aqui vão as mesmas
 * posições (ForceAtlas2 no worker de grafo), as mesmas cores de comunidade e rótulos só
 * nos nós maiores — uma figura estática, legível impressa.
 */
const WIDTH = 960;
const HEIGHT = 560;
const PAD = 40;
const LABELED = 24;

export function NetworkSvg({ nodes, edges }: { nodes: readonly RenderNode[]; edges: readonly RenderEdge[] }) {
  const en = useLocale((s) => s.locale === 'en');
  const { data: positions } = useAsyncResult(`report network ${identityKey(nodes)}`, () =>
    getGraphWorker().layout(
      nodes.map((node) => node.key),
      edges.map((edge) => [edge.source, edge.target] as const),
    ),
  );

  const drawing = useMemo(() => {
    if (!positions) return null;
    const placed = nodes.filter((node) => positions[node.key]);
    const xs = placed.map((node) => positions[node.key]![0]);
    const ys = placed.map((node) => positions[node.key]![1]);
    const [minX, maxX, minY, maxY] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const scale = Math.min((WIDTH - PAD * 2) / (maxX - minX || 1), (HEIGHT - PAD * 2) / (maxY - minY || 1));
    const offsetX = (WIDTH - (maxX - minX) * scale) / 2;
    const offsetY = (HEIGHT - (maxY - minY) * scale) / 2;
    const point = (key: string) => {
      const [x, y] = positions[key]!;
      return { x: offsetX + (x - minX) * scale, y: offsetY + (y - minY) * scale };
    };
    const labeled = new Set([...placed].sort((a, b) => b.size - a.size).slice(0, LABELED).map((node) => node.key));
    return {
      lines: edges
        .filter((edge) => positions[edge.source] && positions[edge.target])
        .map((edge) => ({ key: `${edge.source}→${edge.target}`, from: point(edge.source), to: point(edge.target), weight: edge.weight })),
      circles: placed.map((node) => ({
        node,
        ...point(node.key),
        r: Math.max(3, Math.min(16, node.size / 5)),
        labeled: labeled.has(node.key),
      })),
    };
  }, [positions, nodes, edges]);

  if (!drawing) return <div className="h-56 animate-pulse bg-muted/60" aria-busy="true" />;

  return (
    <svg
      role="img"
      aria-label={en ? `${nodes.length} nodes, ${edges.length} edges` : `${nodes.length} nós, ${edges.length} arestas`}
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      width={WIDTH}
      height={HEIGHT}
      className="h-auto w-full"
      fontFamily="Manrope, system-ui, sans-serif"
    >
      <g stroke="var(--muted-foreground)" strokeOpacity={0.28}>
        {drawing.lines.map((line) => (
          <line
            key={line.key}
            x1={line.from.x}
            y1={line.from.y}
            x2={line.to.x}
            y2={line.to.y}
            strokeWidth={Math.max(0.6, Math.min(3, 0.4 + Math.log2(line.weight + 1) * 0.6))}
          />
        ))}
      </g>
      <g>
        {drawing.circles.map(({ node, x, y, r }) => (
          <circle key={node.key} cx={x} cy={y} r={r} fill={communityColor(node.community)} stroke="var(--background)" strokeWidth={1} />
        ))}
      </g>
      <g fontSize={12} fontWeight={600} fill="var(--foreground)" paintOrder="stroke" stroke="var(--background)" strokeWidth={3}>
        {drawing.circles
          .filter((circle) => circle.labeled)
          .map(({ node, x, y, r }) => (
            <text key={node.key} x={x + r + 4} y={y + 4}>
              {node.label}
            </text>
          ))}
      </g>
    </svg>
  );
}
