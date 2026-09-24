import type { ChatPrompt } from '@/core/hybrid/prompts';
import { AiError } from './ai-client';

/**
 * Cliente do modelo gerativo da classificação híbrida.
 *
 * A API do DeepSeek é compatível com a da OpenAI e aceita chamadas diretas do navegador
 * (CORS liberado), então a chave própria vai do navegador para o provedor sem passar por
 * servidor nosso — o mesmo modelo BYOK do chat. Qualquer outro endpoint compatível serve,
 * trocando a URL base. Sem chave própria, a URL base é `/api/hybrid`: o proxy do servidor
 * com a chave do .env e a cota gratuita por dispositivo.
 */

export interface GenerativeConfig {
  apiKey: string;
  model: string;
  baseUrl: string;
}

export interface ChatJsonResult {
  text: string;
  model: string;
  inputTokens: number;
  outputTokens: number;
}

const MAX_ATTEMPTS = 3;

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

/** Pede uma resposta em JSON (modo `json_object`) e devolve o texto cru mais o consumo. */
export async function chatJson(
  config: GenerativeConfig,
  prompt: ChatPrompt,
  signal?: AbortSignal,
  extraHeaders: Record<string, string> = {},
): Promise<ChatJsonResult> {
  const url = `${config.baseUrl.replace(/\/+$/, '')}/chat/completions`;
  const headers: Record<string, string> = { 'Content-Type': 'application/json', ...extraHeaders };
  if (config.apiKey) headers['Authorization'] = `Bearer ${config.apiKey}`;

  const body = JSON.stringify({
    model: config.model,
    messages: [
      { role: 'system', content: prompt.system },
      { role: 'user', content: prompt.user },
    ],
    response_format: { type: 'json_object' },
    temperature: 0.2,
    max_tokens: 8192,
    stream: false,
  });

  for (let attempt = 1; ; attempt += 1) {
    const response = await fetch(url, { method: 'POST', headers, body, ...(signal ? { signal } : {}) });

    if (response.ok) {
      const data = (await response.json()) as {
        model?: string;
        choices?: { message?: { content?: string } }[];
        usage?: { prompt_tokens?: number; completion_tokens?: number };
      };
      const text = data.choices?.[0]?.message?.content ?? '';
      if (!text.trim()) throw new AiError('O modelo gerativo devolveu uma resposta vazia.');
      return {
        text,
        model: data.model ?? config.model,
        inputTokens: data.usage?.prompt_tokens ?? 0,
        outputTokens: data.usage?.completion_tokens ?? 0,
      };
    }

    // Limite de taxa e indisponibilidade são transitórios; o resto (chave, modelo) não.
    // Cota gratuita esgotada também chega como 429, mas não adianta tentar de novo.
    const quotaExhausted = response.headers.has('x-free-limit');
    const transient = !quotaExhausted && (response.status === 429 || response.status >= 500);
    if (!transient || attempt >= MAX_ATTEMPTS) {
      const err = (await response.json().catch(() => ({}))) as { error?: { message?: string } };
      throw new AiError(err.error?.message || `Modelo gerativo: erro HTTP ${response.status}`);
    }
    await wait(1000 * 2 ** attempt, signal);
  }
}
