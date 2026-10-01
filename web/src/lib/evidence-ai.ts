import type { ChatPrompt } from '@/core/hybrid/prompts';
import { openAiCompatibleBaseUrl, PROVIDER_OPTIONS, useAiConfig } from '@/state/ai-config.store';
import { useFreeTier } from '@/state/free-tier.store';
import { useLocale } from '@/state/locale.store';
import { AiError, localized } from './ai-client';
import { chatJson } from './deepseek-client';
import { freeTierHeaders } from './device-id';

/**
 * Para onde vai o texto do artigo quando a IA propõe respostas. Usa a mesma configuração
 * do Simi ("Configurar IA"): com chave própria, o navegador fala direto com o provedor
 * escolhido; sem ela, o DeepSeek do servidor do Simetrics responde e cada artigo conta
 * como uma pergunta gratuita do Simi.
 */

export interface AiDestination {
  /** Nome do provedor como o usuário o reconhece. */
  provider: string;
  model: string;
  /** Chave própria (direto do navegador) ou servidor do Simetrics. */
  ownKey: boolean;
  /** Sem chave própria e sem DeepSeek no servidor, não há a quem enviar. */
  available: boolean;
}

export function aiDestination(): AiDestination {
  const config = useAiConfig.getState().config;
  if (!config.apiKey && config.provider !== 'custom') {
    const deepseek = useFreeTier.getState().status?.deepseek;
    return { provider: 'DeepSeek', model: deepseek?.model ?? 'deepseek', ownKey: false, available: deepseek?.available !== false };
  }
  const label = PROVIDER_OPTIONS.find((option) => option.id === config.provider)?.label ?? config.provider;
  return { provider: label, model: config.model, ownKey: true, available: true };
}

const MAX_OUTPUT_TOKENS = 8192;

async function failure(response: Response, provider: string): Promise<never> {
  const err = (await response.json().catch(() => ({}))) as { error?: { message?: string } };
  throw new AiError(err.error?.message || `${provider}: HTTP ${response.status}`);
}

/** Pede a resposta (JSON no texto) e devolve o texto cru e o modelo que respondeu. */
export async function requestEvidenceJson(prompt: ChatPrompt, signal?: AbortSignal): Promise<{ text: string; model: string }> {
  const config = useAiConfig.getState().config;

  if (!config.apiKey && config.provider !== 'custom') {
    try {
      const result = await chatJson(
        { apiKey: '', model: 'server', baseUrl: '/api/simi' },
        prompt,
        signal,
        freeTierHeaders(crypto.randomUUID(), useLocale.getState().locale),
      );
      return { text: result.text, model: result.model };
    } finally {
      void useFreeTier.getState().refresh();
    }
  }

  const { provider, apiKey, model } = config;

  if (provider === 'gemini') {
    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent?key=${encodeURIComponent(apiKey)}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          systemInstruction: { parts: [{ text: prompt.system }] },
          contents: [{ role: 'user', parts: [{ text: prompt.user }] }],
          generationConfig: { maxOutputTokens: MAX_OUTPUT_TOKENS, temperature: 0.1, responseMimeType: 'application/json' },
        }),
        ...(signal ? { signal } : {}),
      },
    );
    if (!response.ok) return failure(response, 'Gemini');
    const data = (await response.json()) as { candidates?: { content?: { parts?: { text?: string }[] } }[] };
    return { text: data.candidates?.[0]?.content?.parts?.map((part) => part.text ?? '').join('') ?? '', model };
  }

  if (provider === 'claude') {
    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01',
        'anthropic-dangerous-direct-browser-access': 'true',
      },
      body: JSON.stringify({
        model,
        system: prompt.system,
        messages: [{ role: 'user', content: prompt.user }],
        max_tokens: MAX_OUTPUT_TOKENS,
      }),
      ...(signal ? { signal } : {}),
    });
    if (!response.ok) return failure(response, 'Claude');
    const data = (await response.json()) as { content?: { type: string; text?: string }[] };
    return { text: data.content?.find((block) => block.type === 'text')?.text ?? '', model };
  }

  const baseUrl = openAiCompatibleBaseUrl(config);
  if (!baseUrl) throw new AiError(localized(`Provedor sem endpoint configurado: ${provider}`, `Provider has no configured endpoint: ${provider}`));
  const body: Record<string, unknown> = {
    model,
    messages: [
      { role: 'system', content: prompt.system },
      { role: 'user', content: prompt.user },
    ],
    max_tokens: MAX_OUTPUT_TOKENS,
    temperature: 0.1,
  };
  // A OpenAI trocou `max_tokens` por `max_completion_tokens` e, nos modelos de raciocínio,
  // só aceita a temperatura padrão (o mesmo ajuste de `tuneOpenAiBody` no chat).
  if (provider === 'openai') {
    body.max_completion_tokens = body.max_tokens;
    delete body.max_tokens;
    delete body.temperature;
  }
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (apiKey) headers['Authorization'] = `Bearer ${apiKey}`;
  const response = await fetch(`${baseUrl.replace(/\/+$/, '')}/chat/completions`, {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
    ...(signal ? { signal } : {}),
  });
  if (!response.ok) return failure(response, provider);
  const data = (await response.json()) as { model?: string; choices?: { message?: { content?: string } }[] };
  return { text: data.choices?.[0]?.message?.content ?? '', model: data.model ?? model };
}
