import louvain from 'graphology-communities-louvain';

import type { Dataset, NodeKind, SnaNodeMetrics } from '@/lib/types';
import { pyRound } from '../stats';
import {
  buildCooccurrenceGraph,
  buildHeterogeneousGraph,
  resolveCooccurrenceColumn,
  type CooccurrenceKind,
} from './build';
import {
  betweennessCentrality,
  closenessCentrality,
  degreeCentrality,
  eigenvectorCentrality,
} from './centrality';
import { degrees, toCompact, type CompactGraph } from './compact';
import { globalMetrics, type GlobalMetrics } from './metrics';

export * from './build';
export * from './centrality';
export * from './compact';
export * from './metrics';

/**
 * Orçamento de operações do Brandes exato (aproximadamente V × E).
 *
 * Abaixo dele o betweenness é calculado de forma exata; acima, cai para amostragem
 * semeada. O limite foi calibrado para o grafo heterogêneo dos dados de exemplo
 * (3.239 nós, 3.806 arestas ≈ 12M) rodar exato em bem menos de um segundo.
 */
const EXACT_BETWEENNESS_BUDGET = 20_000_000;

/** Número de fontes amostradas quando o cálculo exato é caro demais — o `k` do Python. */
const BETWEENNESS_SAMPLE_SIZE = 100;

export interface BetweennessPlan {
  exact: boolean;
  sampleSize: number | null;
}

/** Decide entre Brandes exato e amostrado a partir do tamanho do grafo. */
export function planBetweenness(graph: CompactGraph): BetweennessPlan {
  const cost = graph.order * Math.max(graph.size, 1);
  if (cost <= EXACT_BETWEENNESS_BUDGET) return { exact: true, sampleSize: null };
  return { exact: false, sampleSize: Math.min(BETWEENNESS_SAMPLE_SIZE, graph.order) };
}

export interface SnaReport {
  nodes: SnaNodeMetrics[];
  global: GlobalMetrics;
  /** Falso quando o betweenness foi estimado por amostragem. */
  betweennessExact: boolean;
}

/**
 * Análise completa do grafo heterogêneo — ⇄ `gerar_tabela_metricas_completas`
 * (utils.py:2142).
 */
export function analyzeHeterogeneousNetwork(
  rows: Dataset,
  onProgress?: (ratio: number, phase: string) => void,
): SnaReport {
  onProgress?.(0, 'Mapeando topologia');
  const { graph, nodeTypes } = buildHeterogeneousGraph(rows, (ratio) =>
    onProgress?.(ratio * 0.4, 'Mapeando topologia'),
  );

  const compact = toCompact(graph);
  if (compact.order === 0) {
    return { nodes: [], global: globalMetrics(compact), betweennessExact: true };
  }

  onProgress?.(0.45, 'Calculando centralidades');
  const absoluteDegrees = degrees(compact);
  const degreeCent = degreeCentrality(compact);
  const closeness = closenessCentrality(compact);
  const eigenvector = eigenvectorCentrality(compact);

  const plan = planBetweenness(compact);
  onProgress?.(0.6, plan.exact ? 'Betweenness (exato)' : 'Betweenness (amostrado)');
  const betweenness = betweennessCentrality(compact, {
    sampleSize: plan.sampleSize,
    onProgress: (ratio) => onProgress?.(0.6 + ratio * 0.25, 'Betweenness'),
  });

  onProgress?.(0.9, 'Métricas de ecologia profunda');
  const global = globalMetrics(compact, betweenness);

  const nodes: SnaNodeMetrics[] = compact.nodes.map((key, index) => ({
    item: key,
    kind: (nodeTypes.get(key) ?? 'Outro') as NodeKind | 'Outro',
    degreeAbsolute: absoluteDegrees[index] as number,
    degreeCentrality: pyRound(degreeCent[index] as number, 4),
    eigenvector: pyRound(eigenvector[index] as number, 4),
    betweenness: pyRound(betweenness[index] as number, 4),
    closeness: pyRound(closeness[index] as number, 4),
  }));

  nodes.sort((left, right) => right.degreeAbsolute - left.degreeAbsolute);
  onProgress?.(1, 'Concluído');

  return { nodes, global, betweennessExact: plan.exact };
}

/** Um nó pronto para o Sigma.js renderizar. */
export interface RenderNode {
  key: string;
  label: string;
  /** Documentos em que a entidade aparece. */
  count: number;
  size: number;
  community: number;
  degreeAbsolute: number;
  degreeCentrality: number;
  eigenvector: number;
  betweenness: number;
  closeness: number;
}

export interface RenderEdge {
  source: string;
  target: string;
  weight: number;
}

export interface CooccurrenceReport {
  nodes: RenderNode[];
  edges: RenderEdge[];
  global: GlobalMetrics;
  communityCount: number;
}

/** Métrica que controla o tamanho visual dos nós. */
export type SizeMetric =
  | 'Tamanho Fixo'
  | 'Grau Absoluto'
  | 'Centralidade (Eigen)'
  | 'Betweenness'
  | 'Closeness';

const FIXED_NODE_SIZE = 25;
const MIN_NODE_SIZE = 15;
const MAX_NODE_SIZE = 55;

/** Escala linear entre os limites visuais — ⇄ `get_scaled_size` (utils.py:2223). */
function scaleSize(value: number, min: number, max: number): number {
  if (max === min) return MIN_NODE_SIZE;
  return MIN_NODE_SIZE + ((value - min) * (MAX_NODE_SIZE - MIN_NODE_SIZE)) / (max - min);
}

/**
 * Rede de coocorrência com centralidades, comunidades e tamanhos prontos para renderizar.
 *
 * As comunidades vêm do Louvain (`graphology-communities-louvain`), no lugar do
 * `greedy_modularity_communities` do NetworkX. Ambos otimizam modularidade; o Louvain é
 * substancialmente mais rápido e é o algoritmo que ferramentas como o VOSviewer usam.
 */
export function analyzeCooccurrenceNetwork(
  rows: Dataset,
  kind: CooccurrenceKind,
  topN: number,
  sizeMetric: SizeMetric = 'Tamanho Fixo',
): CooccurrenceReport {
  const column = resolveCooccurrenceColumn(rows, kind);
  if (!column) {
    const empty = toCompact(buildCooccurrenceGraph([], 'none'));
    return { nodes: [], edges: [], global: globalMetrics(empty), communityCount: 0 };
  }

  const graph = buildCooccurrenceGraph(rows, column, { topN });
  const compact = toCompact(graph);

  const absoluteDegrees = degrees(compact);
  const degreeCent = degreeCentrality(compact);
  const closeness = closenessCentrality(compact);
  const eigenvector = eigenvectorCentrality(compact);
  const plan = planBetweenness(compact);
  const betweenness = betweennessCentrality(compact, { sampleSize: plan.sampleSize });

  // Louvain precisa de ao menos uma aresta para particionar.
  const communities: Record<string, number> =
    compact.size > 0 ? louvain(graph, { resolution: 1 }) : {};

  const metricValues: Record<Exclude<SizeMetric, 'Tamanho Fixo'>, Float64Array | Int32Array> = {
    'Grau Absoluto': absoluteDegrees,
    'Centralidade (Eigen)': eigenvector,
    Betweenness: betweenness,
    Closeness: closeness,
  };

  const selected = sizeMetric === 'Tamanho Fixo' ? null : metricValues[sizeMetric];
  const minValue = selected ? Math.min(...selected) : 0;
  const maxValue = selected ? Math.max(...selected) : 0;

  const nodes: RenderNode[] = compact.nodes.map((key, index) => ({
    key,
    label: key,
    count: Number(graph.getNodeAttribute(key, 'count') ?? 0),
    size: selected ? scaleSize(selected[index] as number, minValue, maxValue) : FIXED_NODE_SIZE,
    community: communities[key] ?? 0,
    degreeAbsolute: absoluteDegrees[index] as number,
    degreeCentrality: pyRound(degreeCent[index] as number, 4),
    eigenvector: pyRound(eigenvector[index] as number, 4),
    betweenness: pyRound(betweenness[index] as number, 4),
    closeness: pyRound(closeness[index] as number, 4),
  }));

  const edges: RenderEdge[] = [];
  graph.forEachEdge((_edge, attributes, source, target) => {
    edges.push({ source, target, weight: Number(attributes['weight'] ?? 1) });
  });

  nodes.sort((left, right) => right.degreeAbsolute - left.degreeAbsolute);

  return {
    nodes,
    edges,
    global: globalMetrics(compact, betweenness),
    communityCount: new Set(Object.values(communities)).size,
  };
}
