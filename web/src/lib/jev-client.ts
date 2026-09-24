import type { JevRequestBody, JevResponseBody } from '@/core/hybrid/jev-request';
import { AiError } from './ai-client';

/**
 * Cliente do Jev (TypeSafe).
 *
 * A API do Jev recusa requisições que chegam com cabeçalho `Origin` (403), ou seja, não
 * pode ser chamada do navegador. Todas as chamadas passam por `/api/jev/systemone`: no
 * desenvolvimento, o proxy do Vite (vite.config.ts); em produção, a Netlify Function
 * `jev-systemone`. Os dois removem o `Origin` e repassam a chave — a do usuário, quando
 * ele informou uma, ou a `TYPESAFE_API_KEY` do servidor.
 */

const ENDPOINT = '/api/jev/systemone';
const MAX_ATTEMPTS = 5;

function wait(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener(
      'abort',
      () => {
        clearTimeout(timer);
        reject(new DOMException('Aborted', 'AbortError'));
      },
      { once: true },
    );
  });
}

function isAbort(cause: unknown): boolean {
  return cause instanceof DOMException && cause.name === 'AbortError';
}

/** Espera antes de tentar de novo: `Retry-After` quando vier, senão exponencial com jitter. */
function backoff(attempt: number, response?: Response): number {
  const retryAfter = Number(response?.headers.get('retry-after'));
  if (Number.isFinite(retryAfter) && retryAfter > 0) return retryAfter * 1000;
  return 400 * 2 ** (attempt - 1) + Math.random() * 250;
}

async function readError(response: Response): Promise<string> {
  const body = (await response.json().catch(() => ({}))) as {
    error?: string;
    detail?: { message?: string } | string | { msg?: string }[];
  };
  if (typeof body.error === 'string') return body.error;
  if (typeof body.detail === 'string') return body.detail;
  if (Array.isArray(body.detail)) return body.detail.map((item) => item.msg).filter(Boolean).join('; ');
  return body.detail?.message ?? `Jev: erro HTTP ${response.status}`;
}

/**
 * Uma avaliação no Jev, com novas tentativas em 429/529/5xx e em falha de rede — as SDKs
 * oficiais fazem o mesmo; aqui não usamos a SDK porque a chamada passa pelo proxy.
 */
export async function jevEvaluate(
  body: JevRequestBody | Record<string, unknown>,
  apiKey: string,
  signal?: AbortSignal,
): Promise<JevResponseBody> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (apiKey.trim()) headers['Authorization'] = `Bearer ${apiKey.trim()}`;
  const payload = JSON.stringify(body);

  for (let attempt = 1; ; attempt += 1) {
    let response: Response;
    try {
      response = await fetch(ENDPOINT, { method: 'POST', headers, body: payload, ...(signal ? { signal } : {}) });
    } catch (cause) {
      if (isAbort(cause) || attempt >= MAX_ATTEMPTS) throw cause;
      await wait(backoff(attempt), signal);
      continue;
    }

    if (response.ok) return (await response.json()) as JevResponseBody;

    const transient = response.status === 429 || response.status === 529 || response.status >= 500;
    if (!transient || attempt >= MAX_ATTEMPTS) {
      const message = await readError(response);
      // O Jev responde 403 (não 401) para chave ausente; o proxy responde 401.
      const keyProblem = response.status === 401 || response.status === 403;
      throw new AiError(keyProblem ? `Chave do Jev inválida ou ausente (${message}).` : message);
    }
    await wait(backoff(attempt, response), signal);
  }
}

/**
 * Executa `task` sobre cada item com no máximo `concurrency` em paralelo. A primeira
 * falha interrompe a fila (os itens em voo terminam) e é propagada.
 */
export async function runPool<T, R>(
  items: readonly T[],
  concurrency: number,
  task: (item: T) => Promise<R>,
  onProgress?: (done: number, total: number) => void,
  signal?: AbortSignal,
): Promise<R[]> {
  const results = new Array<R>(items.length);
  let next = 0;
  let done = 0;
  let failure: unknown = null;

  const runner = async () => {
    while (failure === null && next < items.length) {
      if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
      const position = next;
      next += 1;
      try {
        results[position] = await task(items[position] as T);
      } catch (cause) {
        failure ??= cause;
        return;
      }
      done += 1;
      onProgress?.(done, items.length);
    }
  };

  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, runner));
  if (failure !== null) throw failure;
  return results;
}
