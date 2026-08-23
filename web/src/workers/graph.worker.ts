import * as Comlink from 'comlink';

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
};

export type GraphWorkerApi = typeof api;

Comlink.expose(api);
