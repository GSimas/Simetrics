import type { ChatContext } from '@/workers/ai.worker';

/**
 * Cliente das funções serverless do Gemini.
 *
 * Nada aqui conhece a chave da API — ela vive apenas nas Netlify Functions. Este módulo
 * só monta requisições e consome respostas.
 */

const CHAT_ENDPOINT = '/api/gemini/chat';
const LABEL_ENDPOINT = '/api/gemini/label-cluster';

export interface ChatTurn {
  role: 'user' | 'assistant';
  content: string;
}

/** Erro de API com mensagem já apresentável ao usuário. */
export class GeminiError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'GeminiError';
  }
}

async function readErrorMessage(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { error?: string };
    if (body.error) return body.error;
  } catch {
    // Resposta sem JSON: cai na mensagem genérica abaixo.
  }
  return `A requisição falhou (HTTP ${response.status}).`;
}

export interface ChatStreamOptions {
  question: string;
  history: readonly ChatTurn[];
  context: ChatContext;
  /** Chamado a cada pedaço de texto recebido. */
  onChunk: (text: string) => void;
  signal?: AbortSignal;
}

/**
 * Envia a pergunta e consome a resposta em streaming.
 *
 * O streaming não é enfeite: uma resposta longa ultrapassaria os 10 s que a Netlify
 * concede a uma função síncrona. Em streaming, o primeiro byte sai quase imediatamente e
 * a conexão fica aberta enquanto o texto chega.
 */
export async function streamChat(options: ChatStreamOptions): Promise<void> {
  const response = await fetch(CHAT_ENDPOINT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question: options.question,
      history: options.history,
      documents: options.context.documents,
      aggregate: options.context.aggregate,
    }),
    ...(options.signal ? { signal: options.signal } : {}),
  });

  if (!response.ok) throw new GeminiError(await readErrorMessage(response));
  if (!response.body) throw new GeminiError('A resposta chegou vazia.');

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      // `stream: true` mantém o estado do decodificador entre pedaços, para que um
      // caractere multibyte partido na fronteira não vire caractere de substituição.
      options.onChunk(decoder.decode(value, { stream: true }));
    }
    const tail = decoder.decode();
    if (tail) options.onChunk(tail);
  } finally {
    reader.releaseLock();
  }
}

export interface LabelClusterRequest {
  samples: { title: string; abstract: string }[];
  topTerms: string[];
}

/** Pede ao modelo o nome de um agrupamento temático. */
export async function labelCluster(
  request: LabelClusterRequest,
  signal?: AbortSignal,
): Promise<string> {
  const response = await fetch(LABEL_ENDPOINT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
    ...(signal ? { signal } : {}),
  });

  if (!response.ok) throw new GeminiError(await readErrorMessage(response));

  const body = (await response.json()) as { name?: string };
  if (!body.name) throw new GeminiError('O modelo não devolveu um nome de tema.');

  return body.name;
}
