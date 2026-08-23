import { degrees, sampleIndices, type CompactGraph } from './compact';

/**
 * Centralidades sobre a representação CSR.
 *
 * Reimplementadas em vez de delegadas ao `graphology-metrics` por dois motivos: ele não
 * oferece betweenness amostrado (só Brandes completo, inviável nos grafos maiores), e
 * precisamos casar exatamente as fórmulas do NetworkX para a paridade valer.
 */

/** Grau normalizado pelo máximo possível — ⇄ `nx.degree_centrality`. */
export function degreeCentrality(graph: CompactGraph): Float64Array {
  const result = new Float64Array(graph.order);
  if (graph.order <= 1) return result;

  const raw = degrees(graph);
  const scale = 1 / (graph.order - 1);
  for (let i = 0; i < graph.order; i += 1) result[i] = (raw[i] as number) * scale;

  return result;
}

export interface BetweennessOptions {
  /**
   * Número de nós-fonte a amostrar. `null` calcula exato (Brandes completo).
   * A escala é corrigida por `n / k`, como no NetworkX.
   */
  sampleSize?: number | null;
  /** Semente da amostragem, garantindo reprodutibilidade entre execuções. */
  seed?: number;
  onProgress?: (ratio: number) => void;
}

/**
 * Betweenness de Brandes para grafos não ponderados — ⇄ `nx.betweenness_centrality`.
 *
 * Complexidade O(V·E). Com amostragem, O(k·E).
 *
 * Sobre a amostragem: o NetworkX aceita `k` mas, com `seed=None` — que é como o app
 * Python chama —, sorteia as fontes pelo estado global do `random`. O resultado muda a
 * cada execução: no grafo de exemplo, 33% dos nós mudam de valor entre duas rodadas
 * seguidas, e o próprio ranking do topo se altera. Aqui a amostra é semeada, então o
 * mesmo grafo sempre produz os mesmos números.
 */
export function betweennessCentrality(
  graph: CompactGraph,
  options: BetweennessOptions = {},
): Float64Array {
  const { order, offsets, targets } = graph;
  const betweenness = new Float64Array(order);
  if (order <= 2) return betweenness;

  const sampleSize = options.sampleSize ?? null;
  const sources =
    sampleSize === null || sampleSize >= order
      ? null
      : sampleIndices(order, sampleSize, options.seed ?? 42);

  const sourceCount = sources ? sources.length : order;

  // Buffers reaproveitados entre as fontes: realocar por iteração dominaria o tempo.
  const sigma = new Float64Array(order);
  const distance = new Int32Array(order);
  const delta = new Float64Array(order);
  const queue = new Int32Array(order);
  const stack = new Int32Array(order);

  // Predecessores em CSR dinâmico: `predHead`/`predNext` formam listas encadeadas sobre
  // um pool único, evitando um array de arrays com milhares de alocações por fonte.
  const predHead = new Int32Array(order);
  const predNext = new Int32Array(targets.length);
  const predNode = new Int32Array(targets.length);

  for (let s = 0; s < sourceCount; s += 1) {
    const source = sources ? (sources[s] as number) : s;

    sigma.fill(0);
    delta.fill(0);
    distance.fill(-1);
    predHead.fill(-1);

    let predCursor = 0;
    sigma[source] = 1;
    distance[source] = 0;

    let queueHead = 0;
    let queueTail = 0;
    let stackTop = 0;
    queue[queueTail++] = source;

    // Fase 1 — BFS acumulando contagem de caminhos mínimos.
    while (queueHead < queueTail) {
      const v = queue[queueHead++] as number;
      stack[stackTop++] = v;

      const end = offsets[v + 1] as number;
      for (let e = offsets[v] as number; e < end; e += 1) {
        const w = targets[e] as number;

        if ((distance[w] as number) < 0) {
          distance[w] = (distance[v] as number) + 1;
          queue[queueTail++] = w;
        }

        if ((distance[w] as number) === (distance[v] as number) + 1) {
          sigma[w] = (sigma[w] as number) + (sigma[v] as number);
          predNode[predCursor] = v;
          predNext[predCursor] = predHead[w] as number;
          predHead[w] = predCursor;
          predCursor += 1;
        }
      }
    }

    // Fase 2 — acumulação de dependências, da folha para a raiz.
    while (stackTop > 0) {
      const w = stack[--stackTop] as number;
      const coefficient = (1 + (delta[w] as number)) / (sigma[w] as number);

      for (let p = predHead[w] as number; p !== -1; p = predNext[p] as number) {
        const v = predNode[p] as number;
        delta[v] = (delta[v] as number) + (sigma[v] as number) * coefficient;
      }

      if (w !== source) betweenness[w] = (betweenness[w] as number) + (delta[w] as number);
    }

    if (options.onProgress && s % 64 === 0) options.onProgress(s / sourceCount);
  }

  // Normalização do NetworkX para grafos não direcionados: 1 / ((n-1)(n-2)), com o fator
  // n/k compensando a amostragem.
  let scale = 1 / ((order - 1) * (order - 2));
  if (sources) scale *= order / sources.length;

  for (let i = 0; i < order; i += 1) betweenness[i] = (betweenness[i] as number) * scale;

  return betweenness;
}

/**
 * Closeness com a correção de Wasserman-Faust — ⇄ `nx.closeness_centrality`
 * (`wf_improved=True` é o padrão do NetworkX).
 *
 *     C(v) = (alcançáveis - 1) / soma_das_distâncias × (alcançáveis - 1) / (n - 1)
 *
 * O segundo fator penaliza nós presos em componentes pequenos. Sem ele, um nó isolado
 * num par teria closeness máximo — e o grafo bibliométrico tem dezenas de componentes.
 */
export function closenessCentrality(graph: CompactGraph): Float64Array {
  const { order, offsets, targets } = graph;
  const result = new Float64Array(order);
  if (order <= 1) return result;

  const distance = new Int32Array(order);
  const queue = new Int32Array(order);

  for (let source = 0; source < order; source += 1) {
    distance.fill(-1);
    distance[source] = 0;

    let queueHead = 0;
    let queueTail = 0;
    let reachable = 1;
    let totalDistance = 0;

    queue[queueTail++] = source;

    while (queueHead < queueTail) {
      const v = queue[queueHead++] as number;
      const end = offsets[v + 1] as number;

      for (let e = offsets[v] as number; e < end; e += 1) {
        const w = targets[e] as number;
        if ((distance[w] as number) >= 0) continue;

        distance[w] = (distance[v] as number) + 1;
        totalDistance += distance[w] as number;
        reachable += 1;
        queue[queueTail++] = w;
      }
    }

    if (totalDistance > 0) {
      result[source] = ((reachable - 1) / totalDistance) * ((reachable - 1) / (order - 1));
    }
  }

  return result;
}

export interface EigenvectorOptions {
  maxIterations?: number;
  tolerance?: number;
}

/**
 * Centralidade de autovetor por iteração de potência com deslocamento espectral.
 *
 * O deslocamento não é refinamento: sem ele o método não converge neste grafo.
 *
 * O grafo heterogêneo é BIPARTIDO — documentos só se ligam a autores, países e venues,
 * nunca entre si. O espectro de um grafo bipartido é simétrico: para todo autovalor λ
 * existe −λ. Logo |λ₂/λ₁| = 1 e a iteração de potência pura oscila entre os dois lados
 * indefinidamente, em vez de convergir. Medido nos dados de exemplo, 200 iterações erram
 * por 0,26 e mesmo 20.000 ainda não fecham.
 *
 * Iterar em (A + σI) preserva os autovetores e desloca o espectro. Com σ igual à
 * estimativa corrente de λ₁ pelo quociente de Rayleigh, o autovalor −λ₁ vai para perto de
 * zero e a razão de convergência despenca. A convergência passa a acontecer em dezenas de
 * iterações.
 *
 * ⇄ `nx.eigenvector_centrality_numpy`, que normaliza em L2 e fixa o sinal pela soma.
 * Atenção: o NetworkX LEVANTA `AmbiguousSolution` em grafo desconexo e se recusa a
 * calcular; o app captura a exceção e preenche a coluna inteira com zero (utils.py:2038).
 * Aqui o resultado é o autovetor dominante, concentrado no componente de maior raio
 * espectral.
 */
export function eigenvectorCentrality(
  graph: CompactGraph,
  options: EigenvectorOptions = {},
): Float64Array {
  const { order, offsets, targets } = graph;
  const maxIterations = options.maxIterations ?? 1000;
  const tolerance = options.tolerance ?? 1e-12;

  if (order === 0) return new Float64Array(0);

  let current = new Float64Array(order).fill(1 / Math.sqrt(order));
  let next = new Float64Array(order);

  for (let iteration = 0; iteration < maxIterations; iteration += 1) {
    next.fill(0);

    for (let v = 0; v < order; v += 1) {
      const value = current[v] as number;
      if (value === 0) continue;

      const end = offsets[v + 1] as number;
      for (let e = offsets[v] as number; e < end; e += 1) {
        const w = targets[e] as number;
        next[w] = (next[w] as number) + value;
      }
    }

    // Quociente de Rayleigh: com `current` unitário, x·Ax estima λ₁.
    let rayleigh = 0;
    for (let i = 0; i < order; i += 1) rayleigh += (current[i] as number) * (next[i] as number);

    const shift = Math.abs(rayleigh);
    if (shift > 0) {
      for (let i = 0; i < order; i += 1) {
        next[i] = (next[i] as number) + shift * (current[i] as number);
      }
    }

    let squaredNorm = 0;
    for (let i = 0; i < order; i += 1) squaredNorm += (next[i] as number) ** 2;
    const norm = Math.sqrt(squaredNorm);

    // Norma zero indica grafo sem arestas: centralidade indefinida para todos.
    if (norm === 0) return new Float64Array(order);

    let drift = 0;
    for (let i = 0; i < order; i += 1) {
      next[i] = (next[i] as number) / norm;
      drift += Math.abs((next[i] as number) - (current[i] as number));
    }

    const swap = current;
    current = next;
    next = swap;

    if (drift < order * tolerance) break;
  }

  // Sinal canônico: o NetworkX força a soma do vetor a ser positiva.
  let total = 0;
  for (let i = 0; i < order; i += 1) total += current[i] as number;
  if (total < 0) {
    for (let i = 0; i < order; i += 1) current[i] = -(current[i] as number);
  }

  return current;
}

export interface PageRankOptions {
  alpha?: number;
  maxIterations?: number;
  tolerance?: number;
}

/** PageRank — ⇄ `nx.pagerank`. Nós sem vizinhos redistribuem sua massa uniformemente. */
export function pagerank(graph: CompactGraph, options: PageRankOptions = {}): Float64Array {
  const { order, offsets, targets } = graph;
  const alpha = options.alpha ?? 0.85;
  const maxIterations = options.maxIterations ?? 100;
  const tolerance = options.tolerance ?? 1e-6;

  if (order === 0) return new Float64Array(0);

  // O divisor é o número de entradas do CSR, e não `degrees()`.
  //
  // Os dois diferem em nós com laço próprio: `degrees()` conta o laço como 2, seguindo a
  // convenção do NetworkX, mas o CSR guarda uma entrada só. Dividir por 2 distribuindo
  // uma única aresta vaza massa a cada iteração, e o PageRank deixa de somar 1.
  const outDegrees = new Int32Array(order);
  for (let v = 0; v < order; v += 1) {
    outDegrees[v] = (offsets[v + 1] as number) - (offsets[v] as number);
  }

  let current = new Float64Array(order).fill(1 / order);
  const next = new Float64Array(order);

  for (let iteration = 0; iteration < maxIterations; iteration += 1) {
    next.fill(0);

    // Massa presa em nós isolados (dangling) volta uniformemente para a rede.
    let dangling = 0;
    for (let v = 0; v < order; v += 1) {
      if ((outDegrees[v] as number) === 0) dangling += current[v] as number;
    }

    const teleport = (1 - alpha) / order + (alpha * dangling) / order;

    for (let v = 0; v < order; v += 1) {
      const degree = outDegrees[v] as number;
      if (degree === 0) continue;

      const share = (alpha * (current[v] as number)) / degree;
      const end = offsets[v + 1] as number;
      for (let e = offsets[v] as number; e < end; e += 1) {
        const w = targets[e] as number;
        next[w] = (next[w] as number) + share;
      }
    }

    let drift = 0;
    for (let i = 0; i < order; i += 1) {
      next[i] = (next[i] as number) + teleport;
      drift += Math.abs((next[i] as number) - (current[i] as number));
    }

    current = Float64Array.from(next);
    if (drift < order * tolerance) break;
  }

  return current;
}
