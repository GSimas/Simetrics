import * as Comlink from 'comlink';

import type { AiWorkerApi } from './ai.worker';
import type { AnalyticsWorkerApi } from './analytics.worker';
import { DATASET_CACHE_MISS, DatasetLru, type DatasetRef } from './dataset-cache';
import type { GraphWorkerApi } from './graph.worker';
import type { IngestWorkerApi } from './ingest.worker';

/**
 * Ponte tipada entre a UI e os Web Workers.
 *
 * Os workers são instanciados sob demanda e reaproveitados: criar um worker custa dezenas
 * de milissegundos e recarrega todo o bundle do core, então um por tipo é o suficiente.
 *
 * Nos workers de análise, grafo e IA a base não é clonada a cada chamada: ela atravessa
 * uma vez por versão e as chamadas seguintes mandam só uma referência (ver
 * `dataset-cache.ts`). As chamadas continuam idênticas para quem usa — a troca acontece
 * aqui dentro.
 */

interface Handle<T> {
  worker: Worker;
  remote: Comlink.Remote<T>;
}

let ingestWorker: Handle<IngestWorkerApi> | null = null;
let analyticsWorker: Handle<AnalyticsWorkerApi> | null = null;
let graphWorker: Handle<GraphWorkerApi> | null = null;
let aiWorker: Handle<AiWorkerApi> | null = null;

// ---------------------------------------------------------------------------------------
// Base enviada uma vez por versão

const datasetIds = new WeakMap<object, number>();
let nextDatasetId = 1;

function datasetId(dataset: object): number {
  let id = datasetIds.get(dataset);
  if (id === undefined) {
    id = nextDatasetId++;
    datasetIds.set(dataset, id);
  }
  return id;
}

/** Uma base é um array não vazio de documentos (objetos simples, não arrays). */
function isDataset(value: unknown): value is object[] {
  if (!Array.isArray(value) || value.length === 0) return false;
  const first: unknown = value[0];
  return typeof first === 'object' && first !== null && !Array.isArray(first);
}

type SetDataset = (id: number, dataset: unknown) => Promise<void>;

/**
 * Proxy sobre o remote do Comlink: troca cada base pela referência, mandando antes a base
 * se o worker ainda não a tem. O LRU daqui espelha o do worker.
 */
function withDatasetRefs<T>(remote: Comlink.Remote<T>): Comlink.Remote<T> {
  const mirror = new DatasetLru<true>();
  const setDataset = (remote as unknown as { __setDataset: SetDataset }).__setDataset;

  const call = (method: (...args: unknown[]) => Promise<unknown>, args: unknown[]): Promise<unknown> => {
    const mapped = args.map((arg): unknown => {
      if (!isDataset(arg)) return arg;
      const id = datasetId(arg);
      if (!mirror.touch(id)) {
        // Sem await: a base e a chamada saem em ordem pela mesma porta.
        void setDataset(id, arg).catch(() => undefined);
        mirror.set(id, true);
      }
      return { __datasetRef: id } satisfies DatasetRef;
    });
    return method(...mapped);
  };

  return new Proxy(remote as object, {
    get(target, property, receiver) {
      const value = Reflect.get(target, property, receiver) as unknown;
      if (typeof property !== 'string' || property === 'then' || property.startsWith('__') || typeof value !== 'function') {
        return value;
      }
      return (...args: unknown[]) =>
        call(value as (...a: unknown[]) => Promise<unknown>, args).catch((error: unknown) => {
          // O espelho perdeu a sincronia (não deveria): reenvia a base completa uma vez.
          if (error instanceof Error && error.message.includes(DATASET_CACHE_MISS)) {
            mirror.clear();
            return call(value as (...a: unknown[]) => Promise<unknown>, args);
          }
          throw error;
        });
    },
  }) as Comlink.Remote<T>;
}

// ---------------------------------------------------------------------------------------

// A fábrica precisa conter o `new Worker(new URL('./x.worker.ts', import.meta.url))`
// literal: é por esse padrão que o Vite reconhece e empacota o worker. Com a URL montada
// fora do `new Worker`, o arquivo .ts ia cru para o build e o worker quebrava em produção.
function spawn<T>(create: () => Worker, cacheDatasets: boolean): Handle<T> {
  const worker = create();
  const remote = Comlink.wrap<T>(worker);
  return { worker, remote: cacheDatasets ? withDatasetRefs(remote) : remote };
}

export function getIngestWorker(): Comlink.Remote<IngestWorkerApi> {
  ingestWorker ??= spawn<IngestWorkerApi>(
    () => new Worker(new URL('./ingest.worker.ts', import.meta.url), { type: 'module' }),
    false,
  );
  return ingestWorker.remote;
}

export function getAnalyticsWorker(): Comlink.Remote<AnalyticsWorkerApi> {
  analyticsWorker ??= spawn<AnalyticsWorkerApi>(
    () => new Worker(new URL('./analytics.worker.ts', import.meta.url), { type: 'module' }),
    true,
  );
  return analyticsWorker.remote;
}

export function getGraphWorker(): Comlink.Remote<GraphWorkerApi> {
  graphWorker ??= spawn<GraphWorkerApi>(
    () => new Worker(new URL('./graph.worker.ts', import.meta.url), { type: 'module' }),
    true,
  );
  return graphWorker.remote;
}

export function getAiWorker(): Comlink.Remote<AiWorkerApi> {
  aiWorker ??= spawn<AiWorkerApi>(
    () => new Worker(new URL('./ai.worker.ts', import.meta.url), { type: 'module' }),
    true,
  );
  return aiWorker.remote;
}

/**
 * Envolve um callback de progresso para atravessar a fronteira do worker.
 *
 * Funções não sobrevivem à clonagem estrutural; o `Comlink.proxy` cria um canal de
 * mensagens dedicado para elas. Sem isso, passar um callback lança DataCloneError.
 */
export function proxyProgress<T extends (...args: never[]) => void>(callback: T): T {
  return Comlink.proxy(callback) as unknown as T;
}

/**
 * Encerra os workers e libera a memória — ao trocar de base ou limpar o projeto.
 *
 * `releaseProxy` só fecha as portas do Comlink; sem `terminate()` a thread continuava
 * viva, terminando cálculos que ninguém esperava mais e segurando a base antiga.
 */
export function terminateWorkers(): void {
  for (const handle of [ingestWorker, analyticsWorker, graphWorker, aiWorker]) {
    if (!handle) continue;
    handle.remote[Comlink.releaseProxy]();
    handle.worker.terminate();
  }
  ingestWorker = null;
  analyticsWorker = null;
  graphWorker = null;
  aiWorker = null;
}
