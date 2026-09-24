import { lazy, type ComponentType, type LazyExoticComponent } from 'react';

/**
 * `React.lazy` com duas adições:
 *
 * - `preload()`: baixa o chunk antes de o componente ser pedido (ocioso, hover…), para a
 *   primeira abertura da aba não piscar no fallback do Suspense.
 * - Retentativa: uma falha de rede ao baixar o chunk é tentada de novo, com espera
 *   crescente. O `React.lazy` guarda a promessa rejeitada para sempre; sem isso, um soluço
 *   da rede deixaria a aba quebrada até recarregar a página.
 */
const RETRY_DELAYS_MS = [400, 1200];

async function importWithRetry<T>(factory: () => Promise<T>): Promise<T> {
  for (let attempt = 0; ; attempt += 1) {
    try {
      return await factory();
    } catch (error) {
      const delay = RETRY_DELAYS_MS[attempt];
      if (delay === undefined) throw error;
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any -- a mesma restrição do React.lazy
export type PreloadableComponent<T extends ComponentType<any>> = LazyExoticComponent<T> & {
  preload: () => Promise<unknown>;
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any -- a mesma restrição do React.lazy
export function lazyWithPreload<T extends ComponentType<any>>(
  factory: () => Promise<{ default: T }>,
): PreloadableComponent<T> {
  let pending: Promise<{ default: T }> | null = null;
  const load = (): Promise<{ default: T }> => {
    pending ??= importWithRetry(factory).catch((error: unknown) => {
      // Esquece a falha: o próximo `preload` ou render tenta de novo.
      pending = null;
      throw error;
    });
    return pending;
  };
  return Object.assign(lazy(load), { preload: load });
}

/** Erro de chunk que não carregou — só recarregar a página resolve (deploy novo, offline). */
export function isChunkLoadError(error: unknown): boolean {
  const message = error instanceof Error ? `${error.name} ${error.message}` : String(error);
  return /dynamically imported module|Importing a module script failed|Failed to fetch|ChunkLoadError|error loading dynamically/i.test(
    message,
  );
}

/** Roda `task` quando o navegador estiver ocioso (com teto de espera). */
export function whenIdle(task: () => void, timeout = 2000): () => void {
  if (typeof window.requestIdleCallback === 'function') {
    const handle = window.requestIdleCallback(task, { timeout });
    return () => window.cancelIdleCallback(handle);
  }
  // Safari não tem requestIdleCallback; um atraso curto cumpre o mesmo papel.
  const handle = window.setTimeout(task, 600);
  return () => window.clearTimeout(handle);
}
