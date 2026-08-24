import type { ChatContext } from '@/workers/ai.worker';
import { useAiConfig, type AiConfig } from '@/state/ai-config.store';
import { useLocale } from '@/state/locale.store';

export interface ChatTurn {
  role: 'user' | 'assistant';
  content: string;
}

export class AiError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'AiError';
  }
}

export interface ChatStreamOptions {
  question: string;
  history: readonly ChatTurn[];
  context: ChatContext;
  onChunk: (text: string) => void;
  signal?: AbortSignal;
}

export interface LabelClusterRequest {
  samples: { title: string; abstract: string }[];
  topTerms: string[];
}

/** Limites defensivos de texto */
const MAX_SAMPLES = 8;
const MAX_ABSTRACT = 800;
const MAX_TERMS = 12;
const MAX_NAME_LENGTH = 60;

function buildLabelPrompt(
  samples: LabelClusterRequest['samples'],
  topTerms: string[],
  locale: 'pt' | 'en' = 'pt',
): string {
  const documentBlock = (samples ?? [])
    .slice(0, MAX_SAMPLES)
    .map((sample) => {
      const title = String(sample.title ?? '').slice(0, 300).trim();
      const abstract = String(sample.abstract ?? '').slice(0, MAX_ABSTRACT).trim();
      return `- ${locale === 'pt' ? 'Título' : 'Title'}: ${title || (locale === 'pt' ? 'Sem título' : 'No title')}\n  ${locale === 'pt' ? 'Resumo' : 'Abstract'}: ${abstract || (locale === 'pt' ? 'Sem resumo' : 'No abstract')}`;
    })
    .join('\n\n');

  const termBlock = topTerms.slice(0, MAX_TERMS).join(', ');

  if (locale === 'en') {
    return `You are an expert data scientist specializing in academic literature review.
Below are scientific papers grouped by textual similarity.
Key characteristic terms of this cluster: ${termBlock || 'none available'}
Representative papers:
${documentBlock || 'No papers available.'}

Your task: synthesize the central research theme that unifies this group.
Respond ONLY with the theme title in English, in at most 4 words. No punctuation, no quotes, no prefix like "Theme:".`;
  }

  return `Você é um cientista de dados especialista em revisão de literatura.
Abaixo estão artigos científicos que um algoritmo agrupou por similaridade textual.
Termos mais característicos do agrupamento: ${termBlock || 'não disponíveis'}
Artigos representativos:
${documentBlock || 'Nenhum artigo disponível.'}

Sua tarefa: sintetize o tema central que unifica esta escola de pesquisa.
Responda APENAS com o nome do tema, em português, com no máximo 4 palavras. Sem pontuação final, sem aspas, sem prefixos como "Tema:".`;
}

function sanitizeThemeName(raw: string): string {
  const cleaned = raw
    .replace(/[\n\r]+/g, ' ')
    .replace(/["'*`]/g, '')
    .replace(/^\s*(tema|theme|título|title|nome|name)\s*:\s*/i, '')
    .replace(/\.\s*$/, '')
    .trim();

  if (!cleaned) return '';
  const truncated = cleaned.slice(0, MAX_NAME_LENGTH);
  return truncated.replace(
    /\p{L}[\p{L}\p{M}]*/gu,
    (word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase(),
  );
}

function buildSystemPrompt(
  context: ChatContext,
  locale: 'pt' | 'en' = 'pt',
): string {
  const isEn = locale === 'en';
  return `${isEn ? 'You are a senior academic advisor and scientometrics expert working within the Simetrics platform.' : 'Você é um conselheiro acadêmico sênior, especialista em cienciometria, operando dentro da plataforma Simetrics.'}

## ${isEn ? 'User Dataset Overview' : 'Panorama da base do usuário'}
${JSON.stringify(context.aggregate)}

## ${isEn ? 'Relevant Documents for the current question' : 'Documentos relevantes para a pergunta atual'}
${JSON.stringify(context.documents)}

## ${isEn ? 'Your Tasks' : 'Suas tarefas'}
- ${isEn ? 'Answer strictly based on the data above.' : 'Responder com base exclusivamente nos dados acima.'}
- ${isEn ? 'Recommend foundational papers based on themes, abstracts, and citations.' : 'Recomendar artigos fundamentais considerando tema, resumo e impacto (citações).'}
- ${isEn ? 'Suggest suitable journals for manuscript submission based on the profiles in the dataset.' : 'Sugerir periódicos para submissão a partir do perfil do manuscrito descrito pelo usuário.'}
- ${isEn ? 'Identify leading researchers and experts for collaboration or citation.' : 'Identificar especialistas para parceria ou referência.'}

## ${isEn ? 'Strict Rules' : 'Regras absolutas'}
- ${isEn ? 'Recommend ONLY items present in the data above. Never fabricate titles, authors, or journals.' : 'Recomende SOMENTE itens presentes nos dados acima. Nunca invente títulos, autores ou periódicos.'}
- ${isEn ? 'Cite document titles and author names exactly as they appear.' : 'Cite títulos de documentos e nomes de autores exatamente como aparecem.'}
- ${isEn ? 'If the context lacks sufficient data, explicitly state what is missing.' : 'Se os dados não forem suficientes para responder, aponte explicitamente a lacuna.'}
- ${isEn ? 'Respond in English.' : 'Responda em português.'}`;
}

/** Pede ao modelo o nome de um agrupamento temático (BYOK) */
export async function labelCluster(
  request: LabelClusterRequest,
  signal?: AbortSignal,
): Promise<string> {
  const config = useAiConfig.getState().config;
  const locale = useLocale.getState().locale;
  const prompt = buildLabelPrompt(request.samples, request.topTerms, locale);

  // Se não houver chave configurada, tenta o endpoint nativo Netlify
  if (!config.apiKey && config.provider !== 'custom') {
    return labelClusterServerless(request, signal);
  }

  const rawText = await generateTextWithProvider(config, prompt, signal);
  const name = sanitizeThemeName(rawText);
  if (!name) throw new AiError('O modelo devolveu uma resposta vazia.');
  return name;
}

/** Chamada direta para o endpoint serverless legado se chave não estiver configurada */
async function labelClusterServerless(
  request: LabelClusterRequest,
  signal?: AbortSignal,
): Promise<string> {
  const response = await fetch('/api/gemini/label-cluster', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
    ...(signal ? { signal } : {}),
  });

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    throw new AiError(errorBody.error || `HTTP ${response.status}`);
  }

  const body = (await response.json()) as { name?: string };
  if (!body.name) throw new AiError('O modelo não devolveu um nome de tema.');
  return body.name;
}

/** Executa chat com streaming suportando múltiplos provedores (BYOK) */
export async function streamChat(options: ChatStreamOptions): Promise<void> {
  const config = useAiConfig.getState().config;
  const locale = useLocale.getState().locale;
  const systemPrompt = buildSystemPrompt(options.context, locale);

  // Se não houver chave configurada, tenta a função serverless nativa se existir
  if (!config.apiKey && config.provider !== 'custom') {
    return streamChatServerless(options);
  }

  const { provider, apiKey, model, baseUrl } = config;

  if (provider === 'gemini') {
    await streamGemini(apiKey, model, systemPrompt, options);
  } else if (provider === 'openai' || provider === 'openrouter' || provider === 'custom') {
    const defaultUrl =
      provider === 'openrouter'
        ? 'https://openrouter.ai/api/v1'
        : provider === 'openai'
          ? 'https://api.openai.com/v1'
          : baseUrl || 'http://localhost:11434/v1';

    await streamOpenAiCompatible(apiKey, model, defaultUrl, systemPrompt, options);
  } else if (provider === 'claude') {
    await streamClaude(apiKey, model, systemPrompt, options);
  }
}

/** Streaming via Google Gemini REST API */
async function streamGemini(
  apiKey: string,
  model: string,
  systemPrompt: string,
  options: ChatStreamOptions,
): Promise<void> {
  const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(
    model,
  )}:streamGenerateContent?alt=sse&key=${encodeURIComponent(apiKey)}`;

  const contents = [
    ...options.history.map((turn) => ({
      role: turn.role === 'user' ? 'user' : 'model',
      parts: [{ text: turn.content }],
    })),
    { role: 'user', parts: [{ text: options.question }] },
  ];

  const body = {
    system_instruction: { parts: [{ text: systemPrompt }] },
    contents,
    generationConfig: { temperature: 0.3, maxOutputTokens: 8192 },
  };

  const response = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    ...(options.signal ? { signal: options.signal } : {}),
  });

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    throw new AiError(errorBody.error?.message || `Google Gemini API error (HTTP ${response.status})`);
  }

  if (!response.body) throw new AiError('A resposta chegou vazia.');

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.startsWith('data:')) {
          const jsonStr = trimmed.slice(5).trim();
          if (jsonStr) {
            try {
              const data = JSON.parse(jsonStr);
              const parts = data.candidates?.[0]?.content?.parts;
              if (Array.isArray(parts)) {
                for (const part of parts) {
                  if (typeof part?.text === 'string') {
                    options.onChunk(part.text);
                  }
                }
              }
            } catch {
              // chunk json incompleto ou de heartbeat
            }
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/** Streaming via OpenAI / OpenRouter / Custom compatible API */
async function streamOpenAiCompatible(
  apiKey: string,
  model: string,
  baseUrl: string,
  systemPrompt: string,
  options: ChatStreamOptions,
): Promise<void> {
  const url = `${baseUrl.replace(/\/+$/, '')}/chat/completions`;

  const messages = [
    { role: 'system', content: systemPrompt },
    ...options.history.map((turn) => ({ role: turn.role, content: turn.content })),
    { role: 'user', content: options.question },
  ];

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };
  if (apiKey) {
    headers['Authorization'] = `Bearer ${apiKey}`;
  }

  const response = await fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify({
      model,
      messages,
      temperature: 0.3,
      max_tokens: 8192,
      stream: true,
    }),
    ...(options.signal ? { signal: options.signal } : {}),
  });

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    throw new AiError(errorBody.error?.message || `API error (HTTP ${response.status})`);
  }

  if (!response.body) throw new AiError('A resposta chegou vazia.');

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.startsWith('data:')) {
          const jsonStr = trimmed.slice(5).trim();
          if (jsonStr === '[DONE]') break;
          if (jsonStr) {
            try {
              const data = JSON.parse(jsonStr);
              const delta = data.choices?.[0]?.delta?.content;
              if (delta) options.onChunk(delta);
            } catch {
              // chunk parsing
            }
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/** Streaming via Anthropic Claude API */
async function streamClaude(
  apiKey: string,
  model: string,
  systemPrompt: string,
  options: ChatStreamOptions,
): Promise<void> {
  const url = 'https://api.anthropic.com/v1/messages';

  const messages = [
    ...options.history.map((turn) => ({ role: turn.role, content: turn.content })),
    { role: 'user', content: options.question },
  ];

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
      'anthropic-dangerous-direct-browser-access': 'true',
    },
    body: JSON.stringify({
      model,
      system: systemPrompt,
      messages,
      max_tokens: 8192,
      temperature: 0.3,
      stream: true,
    }),
    ...(options.signal ? { signal: options.signal } : {}),
  });

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    throw new AiError(errorBody.error?.message || `Anthropic Claude API error (HTTP ${response.status})`);
  }

  if (!response.body) throw new AiError('A resposta chegou vazia.');

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.startsWith('data:')) {
          const jsonStr = trimmed.slice(5).trim();
          if (jsonStr) {
            try {
              const data = JSON.parse(jsonStr);
              if (data.type === 'content_block_delta' && data.delta?.text) {
                options.onChunk(data.delta.text);
              }
            } catch {
              // chunk parse
            }
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/** Streaming via Netlify Serverless fallback */
async function streamChatServerless(options: ChatStreamOptions): Promise<void> {
  const response = await fetch('/api/gemini/chat', {
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

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    throw new AiError(errorBody.error || `HTTP ${response.status}`);
  }
  if (!response.body) throw new AiError('A resposta chegou vazia.');

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      options.onChunk(decoder.decode(value, { stream: true }));
    }
    const tail = decoder.decode();
    if (tail) options.onChunk(tail);
  } finally {
    reader.releaseLock();
  }
}

/** Helper genérico para geração síncrona/curta com provedores */
async function generateTextWithProvider(
  config: AiConfig,
  prompt: string,
  signal?: AbortSignal,
): Promise<string> {
  const { provider, apiKey, model, baseUrl } = config;

  if (provider === 'gemini') {
    const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(
      model,
    )}:generateContent?key=${encodeURIComponent(apiKey)}`;

    const response = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [{ role: 'user', parts: [{ text: prompt }] }],
        generationConfig: { maxOutputTokens: 64, temperature: 0.2 },
      }),
      ...(signal ? { signal } : {}),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new AiError(err.error?.message || `Gemini API error (HTTP ${response.status})`);
    }

    const data = await response.json();
    return data.candidates?.[0]?.content?.parts?.[0]?.text ?? '';
  }

  if (provider === 'openai' || provider === 'openrouter' || provider === 'custom') {
    const url = `${(provider === 'openrouter'
      ? 'https://openrouter.ai/api/v1'
      : provider === 'openai'
        ? 'https://api.openai.com/v1'
        : baseUrl || 'http://localhost:11434/v1'
    ).replace(/\/+$/, '')}/chat/completions`;

    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (apiKey) headers['Authorization'] = `Bearer ${apiKey}`;

    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        model,
        messages: [{ role: 'user', content: prompt }],
        max_tokens: 64,
        temperature: 0.2,
      }),
      ...(signal ? { signal } : {}),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new AiError(err.error?.message || `API error (HTTP ${response.status})`);
    }

    const data = await response.json();
    return data.choices?.[0]?.message?.content ?? '';
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
        messages: [{ role: 'user', content: prompt }],
        max_tokens: 64,
        temperature: 0.2,
      }),
      ...(signal ? { signal } : {}),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new AiError(err.error?.message || `Claude API error (HTTP ${response.status})`);
    }

    const data = await response.json();
    return data.content?.[0]?.text ?? '';
  }

  return '';
}

/** Testa uma configuração de IA com uma pergunta simples */
export async function testAiConnection(config: AiConfig): Promise<string> {
  if (!config.apiKey && config.provider !== 'custom') {
    throw new AiError('Chave de API não informada.');
  }
  const result = await generateTextWithProvider(config, 'Responda apenas "OK" para testar a conexão.');
  if (!result || !result.trim()) {
    throw new AiError('O modelo respondeu com texto vazio.');
  }
  return result.trim();
}
