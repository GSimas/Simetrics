import type Graph from 'graphology';

/**
 * Representação CSR (compressed sparse row) de um grafo não direcionado.
 *
 * Os algoritmos de centralidade percorrem a vizinhança milhões de vezes; a API de
 * objetos do Graphology cobra alocação e hashing a cada acesso. Aqui a vizinhança vira
 * dois arrays tipados contíguos, e o laço interno vira aritmética de índices.
 *
 * `offsets[i]` até `offsets[i + 1]` delimitam os vizinhos do nó `i` dentro de `targets`.
 */
export interface CompactGraph {
  /** Número de nós. */
  order: number;
  /** Número de arestas (não direcionadas, contadas uma vez). */
  size: number;
  /** Índice numérico -> chave original do nó. */
  nodes: string[];
  /** Chave original -> índice numérico. */
  index: Map<string, number>;
  /** Ponteiros de linha CSR, comprimento `order + 1`. */
  offsets: Int32Array;
  /** Vizinhos concatenados, comprimento `2 * size`. */
  targets: Int32Array;
  /** Pesos paralelos a `targets`. */
  weights: Float64Array;
  /**
   * Laços próprios por nó.
   *
   * Guardados à parte porque as duas convenções do NetworkX divergem: `G.degree(v)` conta
   * um laço próprio como 2, mas a matriz de adjacência registra 1. O CSR segue a matriz
   * (uma entrada), e `degrees()` soma este contador para reproduzir o grau.
   */
  selfLoops: Int32Array;
}

/**
 * Converte um grafo do Graphology para CSR.
 *
 * Laços próprios entram UMA vez em `targets` — a convenção da matriz de adjacência, que é
 * o que os métodos espectrais consomem — e são contabilizados à parte em `selfLoops` para
 * que `degrees()` possa reproduzir a convenção de grau do NetworkX.
 */
export function toCompact(graph: Graph, weightAttribute = 'weight'): CompactGraph {
  const nodes = graph.nodes();
  const order = nodes.length;

  const index = new Map<string, number>();
  for (let i = 0; i < order; i += 1) index.set(nodes[i] as string, i);

  const degrees = new Int32Array(order);
  let size = 0;

  const selfLoops = new Int32Array(order);

  graph.forEachEdge((_edge, _attributes, source, target) => {
    const from = index.get(source) as number;
    const to = index.get(target) as number;

    if (from === to) {
      degrees[from] = (degrees[from] as number) + 1;
      selfLoops[from] = (selfLoops[from] as number) + 1;
      size += 1;
      return;
    }

    degrees[from] = (degrees[from] as number) + 1;
    degrees[to] = (degrees[to] as number) + 1;
    size += 1;
  });

  const offsets = new Int32Array(order + 1);
  for (let i = 0; i < order; i += 1) {
    offsets[i + 1] = (offsets[i] as number) + (degrees[i] as number);
  }

  // Cada aresta ocupa duas posições, exceto os laços próprios, que ocupam uma.
  let selfLoopTotal = 0;
  for (let i = 0; i < order; i += 1) selfLoopTotal += selfLoops[i] as number;

  const entryCount = size * 2 - selfLoopTotal;
  const targets = new Int32Array(entryCount);
  const weights = new Float64Array(entryCount);
  const cursor = offsets.slice(0, order);

  graph.forEachEdge((_edge, attributes, source, target) => {
    const from = index.get(source) as number;
    const to = index.get(target) as number;
    const weight = Number((attributes as Record<string, unknown>)[weightAttribute] ?? 1) || 1;

    targets[cursor[from] as number] = to;
    weights[cursor[from] as number] = weight;
    cursor[from] = (cursor[from] as number) + 1;

    if (from === to) return;

    targets[cursor[to] as number] = from;
    weights[cursor[to] as number] = weight;
    cursor[to] = (cursor[to] as number) + 1;
  });

  return { order, size, nodes, index, offsets, targets, weights, selfLoops };
}

/**
 * Graus absolutos por índice de nó, na convenção do NetworkX: um laço próprio soma 2.
 * Os vizinhos vêm do CSR e o laço extra vem do contador dedicado.
 */
export function degrees(graph: CompactGraph): Int32Array {
  const result = new Int32Array(graph.order);
  for (let i = 0; i < graph.order; i += 1) {
    result[i] =
      (graph.offsets[i + 1] as number) -
      (graph.offsets[i] as number) +
      (graph.selfLoops[i] as number);
  }
  return result;
}

/**
 * PRNG determinístico (mulberry32).
 *
 * `Math.random` não aceita semente, e a amostragem do betweenness precisa ser
 * reproduzível — é justamente o que falta na versão Python, onde o mesmo grafo produz
 * valores diferentes a cada execução.
 */
export function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Amostra `count` índices distintos em [0, order), de forma determinística. */
export function sampleIndices(order: number, count: number, seed: number): Int32Array {
  if (count >= order) return Int32Array.from({ length: order }, (_, i) => i);

  const random = seededRandom(seed);
  const pool = Int32Array.from({ length: order }, (_, i) => i);

  // Fisher-Yates parcial: embaralha só os `count` primeiros.
  for (let i = 0; i < count; i += 1) {
    const j = i + Math.floor(random() * (order - i));
    const temp = pool[i] as number;
    pool[i] = pool[j] as number;
    pool[j] = temp;
  }

  return pool.slice(0, count);
}
