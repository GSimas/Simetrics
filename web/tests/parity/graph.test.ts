import { readFileSync } from 'node:fs';
import { fileURLToPath, URL } from 'node:url';
import { describe, expect, it } from 'vitest';

import {
  betweennessCentrality,
  buildHeterogeneousGraph,
  closenessCentrality,
  degreeCentrality,
  eigenvectorCentrality,
  globalMetrics,
  toCompact,
} from '@/core/graph';
import { degrees } from '@/core/graph/compact';
import { processRisFiles, type RisSource } from '@/core/parsers/pipeline-ris';
import type { Dataset } from '@/lib/types';

/**
 * Paridade das métricas de rede com o NetworkX.
 *
 * Gere o oráculo com `.venv/bin/python scripts/export_graph_golden.py`.
 *
 * O betweenness do golden é o EXATO (`k=None`). O app Python usa `k=100` sem `seed`, que
 * sorteia as fontes pelo estado global do `random` e devolve valores diferentes a cada
 * execução — 33% dos nós mudam entre duas rodadas seguidas no grafo de exemplo. Não há
 * como comparar contra esse valor, nem faria sentido: comparamos contra o exato, que é o
 * que a nossa implementação calcula neste tamanho de grafo.
 */

interface GraphGolden {
  _meta: {
    arquivos: Record<string, string>;
    baseYear: number;
    eigenvectorFalhaNoGrafoTodo: boolean;
  };
  eigenvectorMaiorComponente: { nodeCount: number; values: Record<string, number> };
  estrutura: { nodeCount: number; edgeCount: number; componentCount: number };
  global: Record<string, number | null>;
  nodes: {
    item: string;
    kind: string;
    degreeAbsolute: number;
    degreeCentrality: number;
    betweenness: number;
    closeness: number;
  }[];
}

const repoRoot = fileURLToPath(new URL('../../..', import.meta.url));
const golden = JSON.parse(
  readFileSync(new URL('./graph-golden.json', import.meta.url), 'utf8'),
) as GraphGolden;

const sources: RisSource[] = Object.entries(golden._meta.arquivos).map(([name, database]) => ({
  name,
  database,
  text: readFileSync(`${repoRoot}/${name}`, 'utf8'),
}));

const dataset: Dataset = processRisFiles(sources);
const { graph, nodeTypes } = buildHeterogeneousGraph(dataset);
const compact = toCompact(graph);

describe('construção do grafo heterogêneo', () => {
  it('produz a mesma topologia que o NetworkX', () => {
    expect(compact.order).toBe(golden.estrutura.nodeCount);
    expect(compact.size).toBe(golden.estrutura.edgeCount);
  });

  it('classifica os nós nos mesmos tipos', () => {
    const expectedKinds = new Map(golden.nodes.map((node) => [node.item, node.kind]));
    const divergentes = compact.nodes.filter(
      (key) => (nodeTypes.get(key) ?? 'Outro') !== expectedKinds.get(key),
    );

    expect(divergentes.slice(0, 5)).toEqual([]);
  });
});

describe('centralidades por nó', () => {
  const absoluteDegrees = degrees(compact);
  const degreeCent = degreeCentrality(compact);
  const closeness = closenessCentrality(compact);
  const eigenvector = eigenvectorCentrality(compact);
  // Exato, para ser comparável ao golden.
  const betweenness = betweennessCentrality(compact, { sampleSize: null });

  const byKey = new Map(compact.nodes.map((key, index) => [key, index]));

  it('grau absoluto: idêntico', () => {
    for (const expected of golden.nodes) {
      const index = byKey.get(expected.item);
      expect(index, `nó ausente: ${expected.item}`).toBeDefined();
      expect(absoluteDegrees[index as number], expected.item).toBe(expected.degreeAbsolute);
    }
  });

  it('centralidade de grau: idêntica', () => {
    for (const expected of golden.nodes) {
      const index = byKey.get(expected.item) as number;
      expect(degreeCent[index], expected.item).toBeCloseTo(expected.degreeCentrality, 12);
    }
  });

  it('closeness (Wasserman-Faust): idêntico', () => {
    for (const expected of golden.nodes) {
      const index = byKey.get(expected.item) as number;
      expect(closeness[index], expected.item).toBeCloseTo(expected.closeness, 10);
    }
  });

  it('betweenness exato de Brandes: idêntico', () => {
    for (const expected of golden.nodes) {
      const index = byKey.get(expected.item) as number;
      expect(betweenness[index], expected.item).toBeCloseTo(expected.betweenness, 10);
    }
  });

  /**
   * O autovetor precisa de um oráculo diferente dos demais.
   *
   * `nx.eigenvector_centrality_numpy` LEVANTA `AmbiguousSolution` em grafo desconexo — o
   * NetworkX se recusa a calcular, e o grafo bibliométrico tem 49 componentes. O app
   * engole a exceção com `except: eigen_cent = {n: 0 ...}` (utils.py:2038), então a coluna
   * "Centralidade (Eigen)" da tabela SNA hoje é inteiramente zero.
   *
   * A iteração de potência daqui converge para o autovetor dominante, que vive no
   * componente de maior raio espectral. Validamos contra o NetworkX rodando no MAIOR
   * COMPONENTE isolado, onde ele calcula normalmente.
   */
  it('autovetor: bate com o NetworkX no maior componente conexo', () => {
    const expected = golden.eigenvectorMaiorComponente.values;
    const nodesInComponent = Object.keys(expected);

    expect(nodesInComponent.length).toBe(golden.eigenvectorMaiorComponente.nodeCount);

    // O autovetor dominante é zero fora do seu componente, então os valores do grafo
    // inteiro restritos ao componente já estão normalizados em L2.
    let maxError = 0;
    for (const key of nodesInComponent) {
      const index = byKey.get(key) as number;
      maxError = Math.max(maxError, Math.abs((eigenvector[index] as number) - (expected[key] as number)));
    }

    expect(maxError).toBeLessThan(1e-6);
  });

  it('atribui zero fora do componente dominante', () => {
    const inComponent = new Set(Object.keys(golden.eigenvectorMaiorComponente.values));
    for (const [key, index] of byKey) {
      if (inComponent.has(key)) continue;
      expect(Math.abs(eigenvector[index] as number), key).toBeLessThan(1e-9);
    }
  });
});

describe('betweenness amostrado', () => {
  it('é determinístico entre execuções, ao contrário do NetworkX sem seed', () => {
    const first = betweennessCentrality(compact, { sampleSize: 100, seed: 42 });
    const second = betweennessCentrality(compact, { sampleSize: 100, seed: 42 });

    expect(Array.from(first)).toEqual(Array.from(second));
  });

  it('aproxima razoavelmente o valor exato nos nós de maior centralidade', () => {
    const exact = betweennessCentrality(compact, { sampleSize: null });
    const sampled = betweennessCentrality(compact, { sampleSize: 400, seed: 42 });

    const topIndices = Array.from(exact)
      .map((value, index) => ({ value, index }))
      .sort((left, right) => right.value - left.value)
      .slice(0, 10)
      .map((entry) => entry.index);

    for (const index of topIndices) {
      const expected = exact[index] as number;
      const actual = sampled[index] as number;
      // A amostragem carrega erro real; o teste garante a ordem de grandeza, não o valor.
      expect(Math.abs(actual - expected), compact.nodes[index]).toBeLessThan(
        Math.max(expected * 0.6, 0.01),
      );
    }
  });
});

describe('métricas globais', () => {
  const exact = betweennessCentrality(compact, { sampleSize: null });
  const metrics = globalMetrics(compact, exact);

  it('conta nós, arestas e componentes como o NetworkX', () => {
    expect(metrics.nodeCount).toBe(golden.estrutura.nodeCount);
    expect(metrics.edgeCount).toBe(golden.estrutura.edgeCount);
    expect(metrics.componentCount).toBe(golden.estrutura.componentCount);
  });

  it.each([
    ['density', 12],
    ['clustering', 10],
    ['entropy', 10],
    ['meanDegree', 10],
    ['stdDegree', 10],
    ['minDegree', 12],
    ['maxDegree', 12],
    ['powerLawExponent', 8],
    ['assortativity', 8],
  ])('%s bate com o NetworkX', (key, precision) => {
    const expected = golden.global[key];
    expect(expected, `${key} ausente no golden`).not.toBeNull();
    expect(metrics[key as keyof typeof metrics] as number).toBeCloseTo(
      expected as number,
      precision as number,
    );
  });

  it('eficiência global bate quando o grafo é pequeno o suficiente para calculá-la', () => {
    if (golden.global['efficiency'] === null) {
      // Grafo grande: o Python suprime o cálculo, e nós também.
      expect(typeof metrics.efficiency).toBe('string');
      return;
    }
    expect(metrics.efficiency as number).toBeCloseTo(golden.global['efficiency'] as number, 10);
  });

  /**
   * Spearman tolera erro maior que as demais métricas, por um motivo concreto: 2.040 dos
   * 3.239 nós têm betweenness exatamente zero. Com esse tanto de empate, uma diferença de
   * um único bit no acúmulo de ponto flutuante muda quais nós empatam, muda os postos
   * médios e desloca o coeficiente na sexta casa. O grau e o betweenness em si já são
   * verificados com precisão bem maior acima.
   */
  it('correlação de Spearman entre grau e betweenness bate com o pandas', () => {
    expect(metrics.spearmanDegreeBetweenness).toBeCloseTo(
      golden.global['spearmanDegreeBetweenness'] as number,
      4,
    );
  });

  it('PageRank médio bate com o NetworkX', () => {
    expect(metrics.meanPageRank).toBeCloseTo(golden.global['meanPageRank'] as number, 8);
  });
});
