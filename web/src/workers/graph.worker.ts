import * as Comlink from 'comlink';
import Graph from 'graphology';
import forceAtlas2 from 'graphology-layout-forceatlas2';

import { withDatasetCache } from './dataset-cache';

import {
  analyzeCooccurrenceNetwork,
  analyzeHeterogeneousNetwork,
  type CooccurrenceKind,
  type CooccurrenceReport,
  type SizeMetric,
  type SnaReport,
} from '@/core/graph';
import type { Dataset, WorkerProgress } from '@/lib/types';

/**
 * Worker topológico: centralidades e métricas de rede.
 *
 * É o mais pesado dos três. O grafo heterogêneo dos dados de exemplo tem 3.239 nós e
 * 3.806 arestas, e o betweenness exato de Brandes é O(V·E) — no Python, com NetworkX,
 * isso leva 9 segundos. Fora da thread principal, a UI segue respondendo enquanto roda.
 */

export type ProgressCallback = (progress: WorkerProgress) => void;

/** Iterações do ForceAtlas2 — as mesmas que o grafo usava na thread principal. */
const LAYOUT_ITERATIONS = 260;

const api = {
  /** Ecossistema completo: documentos, autores, países e venues. */
  heterogeneous(dataset: Dataset, onProgress?: ProgressCallback): SnaReport {
    return analyzeHeterogeneousNetwork(dataset, (ratio, phase) =>
      onProgress?.({ phase, ratio }),
    );
  },

  /** Rede de coocorrência pronta para renderizar, com comunidades do Louvain. */
  cooccurrence(
    dataset: Dataset,
    kind: CooccurrenceKind,
    topN: number,
    sizeMetric: SizeMetric,
  ): CooccurrenceReport {
    return analyzeCooccurrenceNetwork(dataset, kind, topN, sizeMetric);
  },

  /**
   * Posições do ForceAtlas2 para o grafo de forças. Rodava na thread principal (260
   * iterações, ~300 ms travando a aba Redes). O cálculo é determinístico — mesma ordem de
   * nós, mesmas arestas, mesmo ponto de partida em círculo —, então as posições saem
   * idênticas às de antes.
   */
  // Arestas como tuplas: um array de objetos seria tomado por uma base pelo cache do cliente.
  layout(nodeKeys: readonly string[], edges: readonly (readonly [string, string])[]): Record<string, [number, number]> {
    const graph = new Graph({ type: 'undirected', multi: false });
    nodeKeys.forEach((key, index) => {
      const angle = (2 * Math.PI * index) / nodeKeys.length;
      graph.addNode(key, { x: Math.cos(angle), y: Math.sin(angle) });
    });
    for (const [source, target] of edges) {
      if (!graph.hasNode(source) || !graph.hasNode(target)) continue;
      if (graph.hasEdge(source, target)) continue;
      graph.addEdge(source, target);
    }
    if (graph.order > 1) {
      forceAtlas2.assign(graph, {
        iterations: LAYOUT_ITERATIONS,
        settings: {
          ...forceAtlas2.inferSettings(graph),
          gravity: 1.1,
          scalingRatio: 12,
          barnesHutOptimize: graph.order > 200,
        },
      });
    }
    const positions: Record<string, [number, number]> = {};
    graph.forEachNode((key, attributes) => {
      positions[key] = [attributes['x'] as number, attributes['y'] as number];
    });
    return positions;
  },
};

export type GraphWorkerApi = typeof api;

// A base chega uma vez por versão; as chamadas trazem só a referência.
Comlink.expose(withDatasetCache(api));
