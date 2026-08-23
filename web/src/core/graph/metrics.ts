import { linearFit, mean, spearman, stdPopulation } from '../stats';
import { betweennessCentrality, eigenvectorCentrality, pagerank } from './centrality';
import { degrees, type CompactGraph } from './compact';

/**
 * Métricas globais da rede — ⇄ `_calcular_metricas_globais_sna` (utils.py:2059).
 * O painel de "ecologia profunda" da aba de Redes.
 */

/** Acima deste tamanho, a eficiência global (O(V·E)) é suprimida — como no Python. */
const EFFICIENCY_NODE_LIMIT = 1500;

/** Densidade — ⇄ `nx.density`: arestas existentes sobre arestas possíveis. */
export function density(graph: CompactGraph): number {
  const { order, size } = graph;
  if (order < 2) return 0;
  return (2 * size) / (order * (order - 1));
}

/**
 * Coeficiente de agrupamento médio — ⇄ `nx.average_clustering`.
 *
 * Para cada nó, a fração de pares de vizinhos que também são vizinhos entre si. Nós com
 * grau menor que 2 contam como zero, que é a convenção do NetworkX.
 */
export function averageClustering(graph: CompactGraph): number {
  const { order, offsets, targets } = graph;
  if (order === 0) return 0;

  // Marcação por época: `neighborMark[u] === v` significa "u é vizinho de v", sem precisar
  // limpar o array a cada nó.
  const neighborMark = new Int32Array(order).fill(-1);
  let total = 0;

  for (let v = 0; v < order; v += 1) {
    const start = offsets[v] as number;
    const end = offsets[v + 1] as number;

    // Laços próprios não são vizinhança: o NetworkX faz `set(v_nbrs) - {v}` antes de
    // contar. Ignorá-los aqui é obrigatório — contá-los produz clustering acima de 1,
    // que é matematicamente impossível.
    let realDegree = 0;
    for (let e = start; e < end; e += 1) {
      const u = targets[e] as number;
      if (u === v) continue;
      neighborMark[u] = v;
      realDegree += 1;
    }
    if (realDegree < 2) continue;

    let links = 0;
    for (let e = start; e < end; e += 1) {
      const u = targets[e] as number;
      if (u === v) continue;

      const uEnd = offsets[u + 1] as number;
      for (let f = offsets[u] as number; f < uEnd; f += 1) {
        const w = targets[f] as number;
        // Descarta o laço próprio de u e o retorno para o próprio v.
        if (w === u || w === v) continue;
        if ((neighborMark[w] as number) === v) links += 1;
      }
    }

    // `links` conta cada triângulo duas vezes, o que casa com o 2·T(v) da fórmula.
    total += links / (realDegree * (realDegree - 1));
  }

  return total / order;
}

/**
 * Entropia de Shannon da distribuição de graus — ⇄ utils.py:2076.
 *
 * Mede a desordem topológica: entropia alta indica graus heterogêneos (rede resiliente,
 * com muitos papéis distintos); baixa indica uniformidade.
 *
 * O Python divide as contagens pelo número de NÓS, não pela soma das contagens — o que dá
 * no mesmo, já que todo nó tem exatamente um grau. Mantido para deixar a equivalência
 * explícita.
 */
export function degreeEntropy(graph: CompactGraph): number {
  const { order } = graph;
  if (order === 0) return 0;

  const nodeDegrees = degrees(graph);
  const histogram = new Map<number, number>();
  for (let i = 0; i < order; i += 1) {
    const degree = nodeDegrees[i] as number;
    histogram.set(degree, (histogram.get(degree) ?? 0) + 1);
  }

  let entropy = 0;
  for (const count of histogram.values()) {
    const probability = count / order;
    entropy -= probability * Math.log2(probability);
  }

  return entropy;
}

/**
 * Eficiência global — ⇄ `nx.global_efficiency`: média do inverso das distâncias entre
 * todos os pares. Custa O(V·E), daí o teto de nós.
 */
export function globalEfficiency(graph: CompactGraph): number {
  const { order, offsets, targets } = graph;
  if (order < 2) return 0;

  const distance = new Int32Array(order);
  const queue = new Int32Array(order);
  let total = 0;

  for (let source = 0; source < order; source += 1) {
    distance.fill(-1);
    distance[source] = 0;

    let queueHead = 0;
    let queueTail = 0;
    queue[queueTail++] = source;

    while (queueHead < queueTail) {
      const v = queue[queueHead++] as number;
      const end = offsets[v + 1] as number;

      for (let e = offsets[v] as number; e < end; e += 1) {
        const w = targets[e] as number;
        if ((distance[w] as number) >= 0) continue;

        distance[w] = (distance[v] as number) + 1;
        total += 1 / (distance[w] as number);
        queue[queueTail++] = w;
      }
    }
  }

  return total / (order * (order - 1));
}

/**
 * Assortatividade de grau — ⇄ `nx.degree_assortativity_coefficient`.
 *
 * Correlação de Pearson entre os graus das duas pontas de cada aresta. Positiva significa
 * que hubs se conectam a hubs; negativa, que hubs se cercam de nós periféricos — o padrão
 * típico de redes de coautoria.
 */
export function degreeAssortativity(graph: CompactGraph): number {
  const { order, offsets, targets } = graph;
  if (graph.size === 0) return Number.NaN;

  const nodeDegrees = degrees(graph);
  const left: number[] = [];
  const right: number[] = [];

  // Cada aresta entra nas duas orientações, como o `for u in G: for v in G[u]` do NetworkX.
  // Laços próprios aparecem uma vez só nos dois — o CSR já segue essa convenção.
  for (let v = 0; v < order; v += 1) {
    const end = offsets[v + 1] as number;
    for (let e = offsets[v] as number; e < end; e += 1) {
      left.push(nodeDegrees[v] as number);
      right.push(nodeDegrees[targets[e] as number] as number);
    }
  }

  const meanLeft = mean(left);
  const meanRight = mean(right);

  let covariance = 0;
  for (let i = 0; i < left.length; i += 1) {
    covariance += ((left[i] as number) - meanLeft) * ((right[i] as number) - meanRight);
  }
  covariance /= left.length;

  const deviation = stdPopulation(left) * stdPopulation(right);
  return deviation === 0 ? Number.NaN : covariance / deviation;
}

/**
 * Expoente da lei de potência, por regressão log-log no histograma de graus — ⇄
 * utils.py:2111. Valores próximos de 2-3 indicam rede livre de escala.
 */
export function powerLawExponent(graph: CompactGraph): number {
  const nodeDegrees = degrees(graph);
  const histogram = new Map<number, number>();

  for (let i = 0; i < graph.order; i += 1) {
    const degree = nodeDegrees[i] as number;
    // Grau zero não tem logaritmo; o `np.nonzero` do Python o descarta pelo mesmo motivo.
    if (degree === 0) continue;
    histogram.set(degree, (histogram.get(degree) ?? 0) + 1);
  }

  const points = [...histogram.entries()].sort((left, right) => left[0] - right[0]);
  if (points.length <= 2) return 0;

  const logDegree = points.map(([degree]) => Math.log10(degree));
  const logCount = points.map(([, count]) => Math.log10(count));

  return Math.abs(linearFit(logDegree, logCount)[0]);
}

export interface GlobalMetrics {
  density: number;
  clustering: number;
  entropy: number;
  /** String quando suprimida por custo, seguindo o comportamento do Python. */
  efficiency: number | string;
  meanDegree: number;
  stdDegree: number;
  minDegree: number;
  maxDegree: number;
  meanPageRank: number;
  meanEigenvector: number;
  powerLawExponent: number;
  assortativity: number;
  spearmanDegreeBetweenness: number;
  nodeCount: number;
  edgeCount: number;
  componentCount: number;
}

/** Número de componentes conexos, por varredura em largura. */
export function connectedComponents(graph: CompactGraph): number {
  const { order, offsets, targets } = graph;
  const seen = new Uint8Array(order);
  const queue = new Int32Array(order);
  let components = 0;

  for (let source = 0; source < order; source += 1) {
    if (seen[source]) continue;

    components += 1;
    seen[source] = 1;

    let queueHead = 0;
    let queueTail = 0;
    queue[queueTail++] = source;

    while (queueHead < queueTail) {
      const v = queue[queueHead++] as number;
      const end = offsets[v + 1] as number;
      for (let e = offsets[v] as number; e < end; e += 1) {
        const w = targets[e] as number;
        if (seen[w]) continue;
        seen[w] = 1;
        queue[queueTail++] = w;
      }
    }
  }

  return components;
}

/**
 * Painel completo de métricas globais.
 *
 * @param betweenness Betweenness já calculado, reaproveitado para a correlação de
 *   Spearman. Recalcular aqui dobraria a parte mais cara da análise.
 */
export function globalMetrics(
  graph: CompactGraph,
  betweenness?: Float64Array,
): GlobalMetrics {
  if (graph.order === 0) {
    return {
      density: 0, clustering: 0, entropy: 0, efficiency: 0,
      meanDegree: 0, stdDegree: 0, minDegree: 0, maxDegree: 0,
      meanPageRank: 0, meanEigenvector: 0, powerLawExponent: 0,
      assortativity: Number.NaN, spearmanDegreeBetweenness: Number.NaN,
      nodeCount: 0, edgeCount: 0, componentCount: 0,
    };
  }

  const nodeDegrees = Array.from(degrees(graph));
  const between = betweenness ?? betweennessCentrality(graph, { sampleSize: 100 });

  return {
    density: density(graph),
    clustering: averageClustering(graph),
    entropy: degreeEntropy(graph),
    efficiency:
      graph.order < EFFICIENCY_NODE_LIMIT ? globalEfficiency(graph) : 'N/A (Grafo Denso)',
    meanDegree: mean(nodeDegrees),
    stdDegree: stdPopulation(nodeDegrees),
    minDegree: Math.min(...nodeDegrees),
    maxDegree: Math.max(...nodeDegrees),
    meanPageRank: mean(Array.from(pagerank(graph))),
    meanEigenvector: mean(Array.from(eigenvectorCentrality(graph))),
    powerLawExponent: powerLawExponent(graph),
    assortativity: degreeAssortativity(graph),
    spearmanDegreeBetweenness: spearman(nodeDegrees, Array.from(between)),
    nodeCount: graph.order,
    edgeCount: graph.size,
    componentCount: connectedComponents(graph),
  };
}
