import * as Comlink from 'comlink';

import type { AiWorkerApi } from './ai.worker';
import type { AnalyticsWorkerApi } from './analytics.worker';
import type { GraphWorkerApi } from './graph.worker';
import type { IngestWorkerApi } from './ingest.worker';

/**
 * Ponte tipada entre a UI e os Web Workers.
 *
 * Os workers são instanciados sob demanda e reaproveitados: criar um worker custa dezenas
 * de milissegundos e recarrega todo o bundle do core, então um por tipo é o suficiente.
 */

let ingestWorker: Comlink.Remote<IngestWorkerApi> | null = null;
let analyticsWorker: Comlink.Remote<AnalyticsWorkerApi> | null = null;
let graphWorker: Comlink.Remote<GraphWorkerApi> | null = null;
let aiWorker: Comlink.Remote<AiWorkerApi> | null = null;

export function getIngestWorker(): Comlink.Remote<IngestWorkerApi> {
  ingestWorker ??= Comlink.wrap<IngestWorkerApi>(
    new Worker(new URL('./ingest.worker.ts', import.meta.url), { type: 'module' }),
  );
  return ingestWorker;
}

export function getAnalyticsWorker(): Comlink.Remote<AnalyticsWorkerApi> {
  analyticsWorker ??= Comlink.wrap<AnalyticsWorkerApi>(
    new Worker(new URL('./analytics.worker.ts', import.meta.url), { type: 'module' }),
  );
  return analyticsWorker;
}

export function getGraphWorker(): Comlink.Remote<GraphWorkerApi> {
  graphWorker ??= Comlink.wrap<GraphWorkerApi>(
    new Worker(new URL('./graph.worker.ts', import.meta.url), { type: 'module' }),
  );
  return graphWorker;
}

export function getAiWorker(): Comlink.Remote<AiWorkerApi> {
  aiWorker ??= Comlink.wrap<AiWorkerApi>(
    new Worker(new URL('./ai.worker.ts', import.meta.url), { type: 'module' }),
  );
  return aiWorker;
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

/** Encerra os workers e libera a memória — útil ao trocar de base. */
export function terminateWorkers(): void {
  ingestWorker?.[Comlink.releaseProxy]();
  analyticsWorker?.[Comlink.releaseProxy]();
  graphWorker?.[Comlink.releaseProxy]();
  aiWorker?.[Comlink.releaseProxy]();
  ingestWorker = null;
  analyticsWorker = null;
  graphWorker = null;
  aiWorker = null;
}
