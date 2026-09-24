/**
 * Cache da base dentro do worker.
 *
 * Sem ele, toda chamada ao worker clona a base inteira na thread principal (clonagem
 * estrutural do `postMessage`): ~11 ms para 973 documentos, ~140 ms para 10 mil — em cada
 * passo de um slider, em cada troca de aba. Com ele, a base atravessa uma vez por versão e
 * as chamadas seguintes mandam só uma referência (`DatasetRef`).
 *
 * O cliente (`client.ts`) espelha este LRU: os dois aplicam a mesma sequência de operações,
 * na mesma ordem (a ordem das mensagens de uma porta é garantida), então sabem sempre quais
 * bases o worker tem.
 */
export const DATASET_CACHE_SIZE = 2;
export const DATASET_CACHE_MISS = 'SIMETRICS_DATASET_CACHE_MISS';

export interface DatasetRef {
  __datasetRef: number;
}

export function isDatasetRef(value: unknown): value is DatasetRef {
  return typeof value === 'object' && value !== null && '__datasetRef' in value;
}

/** LRU pequeno: a base mais recente na frente. */
export class DatasetLru<T> {
  private entries: { id: number; value: T }[] = [];

  set(id: number, value: T): void {
    this.entries = [{ id, value }, ...this.entries.filter((entry) => entry.id !== id)].slice(0, DATASET_CACHE_SIZE);
  }

  /** Lê e marca como usada — a mesma atualização de recência nos dois lados. */
  touch(id: number): T | undefined {
    const index = this.entries.findIndex((entry) => entry.id === id);
    if (index < 0) return undefined;
    const [entry] = this.entries.splice(index, 1);
    this.entries.unshift(entry!);
    return entry!.value;
  }

  clear(): void {
    this.entries = [];
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any -- métodos de API com assinaturas quaisquer
type AnyApi = Record<string, (...args: any[]) => unknown>;

/** Envolve a API do worker: resolve `DatasetRef` → base guardada, e aceita a base nova. */
export function withDatasetCache<T extends AnyApi>(api: T): T & { __setDataset: (id: number, dataset: unknown) => void } {
  const cache = new DatasetLru<unknown>();
  const resolve = (arg: unknown): unknown => {
    if (!isDatasetRef(arg)) return arg;
    const dataset = cache.touch(arg.__datasetRef);
    if (dataset === undefined) throw new Error(DATASET_CACHE_MISS);
    return dataset;
  };
  const wrapped: AnyApi = {};
  for (const [name, method] of Object.entries(api)) {
    wrapped[name] = (...args: unknown[]) => method(...args.map(resolve));
  }
  return Object.assign(wrapped, {
    __setDataset: (id: number, dataset: unknown) => cache.set(id, dataset),
  }) as T & { __setDataset: (id: number, dataset: unknown) => void };
}
