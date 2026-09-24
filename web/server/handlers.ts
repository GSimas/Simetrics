import { randomUUID } from 'node:crypto';

import type { ServerEnv } from './env.ts';
import { consume, readUsage, refund, validId, type Identity, type QuotaPolicy, type QuotaStore } from './quota.ts';

/**
 * Endpoints do Simetrics, escritos uma vez só sobre `Request`/`Response` da web: as
 * Netlify Functions os chamam em produção, e o plugin do Vite (vite.config.ts) no
 * desenvolvimento. Assim o comportamento local é o mesmo do publicado.
 *
 * Rotas:
 *   GET  /api/status                   — o que o servidor oferece e quanto resta da cota
 *   POST /api/simi/chat/completions    — Simi grátis (DeepSeek), 10 perguntas/dispositivo
 *   POST /api/hybrid/chat/completions  — descoberta de categorias grátis, 3 execuções/dispositivo
 *   POST /api/themes/label             — nome de tema do k-means (prompt montado aqui)
 *   POST /api/jev/systemone            — Jev, livre, com a chave do servidor
 *
 * Os dois proxies do DeepSeek falam o formato da OpenAI, então o cliente reaproveita o
 * mesmo código de streaming e de ferramentas que usa com chave própria. O que torna isso
 * seguro é o que o servidor impõe: modelo fixo, teto de tokens, cota por unidade e por IP.
 */

export interface ApiContext {
  env: ServerEnv;
  store: QuotaStore;
  ip: string;
}

type Handler = (request: Request, ctx: ApiContext) => Promise<Response>;
type Locale = 'pt' | 'en';

const MAX_PAYLOAD_BYTES = 1_500_000;
const JEV_MAX_PAYLOAD_BYTES = 200_000;
const JEV_UPSTREAM = 'https://api.typesafe.ai/v1/systemone';

class HttpError extends Error {
  constructor(
    readonly status: number,
    message: string,
    readonly code = 'error',
    readonly headers: Record<string, string> = {},
  ) {
    super(message);
  }
}

function json(status: number, body: unknown, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store', ...headers },
  });
}

/** Erros no formato da OpenAI (`error.message`), que o cliente já sabe exibir. */
function errorResponse(cause: unknown): Response {
  if (cause instanceof HttpError) {
    return json(cause.status, { error: { message: cause.message, code: cause.code } }, cause.headers);
  }
  console.error('Falha na API do Simetrics:', cause instanceof Error ? cause.message : cause);
  return json(502, { error: { message: 'Falha ao contatar o provedor. Tente novamente em instantes.', code: 'upstream' } });
}

function localeOf(request: Request): Locale {
  return request.headers.get('x-simetrics-locale') === 'en' ? 'en' : 'pt';
}

function identityOf(request: Request, ctx: ApiContext): Identity {
  return { device: validId(request.headers.get('x-simetrics-device')), ip: ctx.ip || 'unknown' };
}

async function readJson(request: Request, limit = MAX_PAYLOAD_BYTES): Promise<Record<string, unknown>> {
  if (request.method !== 'POST') throw new HttpError(405, 'Use POST.', 'method');
  const raw = await request.text();
  if (raw.length > limit) throw new HttpError(413, 'Payload grande demais.', 'payload');
  try {
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) throw new Error();
    return parsed as Record<string, unknown>;
  } catch {
    throw new HttpError(400, 'Corpo da requisição não é um JSON válido.', 'json');
  }
}

// ---------------------------------------------------------------------------------------
// Políticas

export function simiPolicy(env: ServerEnv): QuotaPolicy {
  // 1 pergunta = 1 chamada + até 3 rodadas de ferramentas; folga para uma nova tentativa.
  return { scope: 'simi', deviceLimit: env.simiFreeQuestions, ipDailyLimit: env.simiIpDaily, maxRequestsPerUnit: 6 };
}

export function hybridPolicy(env: ServerEnv): QuotaPolicy {
  // 1 execução = descoberta + até 2 esclarecimentos + até 2 expansões, com folga.
  return { scope: 'hybrid', deviceLimit: env.hybridFreeRuns, ipDailyLimit: env.hybridIpDaily, maxRequestsPerUnit: 8 };
}

const QUOTA_MESSAGES: Record<string, Record<Locale, (limit: number) => string>> = {
  'simi:device': {
    pt: (limit) => `Você usou as ${limit} perguntas gratuitas do Simi neste dispositivo. Configure sua própria chave de API em "Configurar IA" para continuar.`,
    en: (limit) => `You have used the ${limit} free Simi questions on this device. Set your own API key in "AI settings" to continue.`,
  },
  'hybrid:device': {
    pt: (limit) => `Você usou as ${limit} classificações híbridas gratuitas neste dispositivo. Informe sua própria chave do modelo gerativo nas configurações para continuar.`,
    en: (limit) => `You have used the ${limit} free hybrid classifications on this device. Set your own generative model key in the settings to continue.`,
  },
  ip: {
    pt: () => 'O limite diário de uso gratuito desta rede foi atingido. Tente amanhã ou use sua própria chave de API.',
    en: () => 'This network has reached the daily free-use limit. Try again tomorrow or use your own API key.',
  },
  unit: {
    pt: () => 'Esta pergunta excedeu o número de chamadas permitidas ao modelo. Faça uma nova pergunta.',
    en: () => 'This question exceeded the allowed number of model calls. Ask a new question.',
  },
  'no-device': {
    pt: () => 'Identificador do dispositivo ausente. Recarregue a página.',
    en: () => 'Missing device identifier. Reload the page.',
  },
};

// ---------------------------------------------------------------------------------------
// Status

const handleStatus: Handler = async (request, ctx) => {
  const identity = identityOf(request, ctx);
  const deepseek = Boolean(ctx.env.deepseekKey);
  const usage = async (policy: QuotaPolicy) => {
    if (!deepseek || policy.deviceLimit === 0) return { limit: 0, used: 0, remaining: 0 };
    const { used, limit } = await readUsage(ctx.store, policy, identity, ctx.env.quotaSalt);
    return { limit, used, remaining: Math.max(0, limit - used) };
  };
  return json(200, {
    deepseek: { available: deepseek, model: deepseek ? ctx.env.deepseekModel : null },
    jev: { available: Boolean(ctx.env.typesafeKey) },
    simi: await usage(simiPolicy(ctx.env)),
    hybrid: await usage(hybridPolicy(ctx.env)),
  });
};

// ---------------------------------------------------------------------------------------
// Proxy do DeepSeek (formato OpenAI)

const ROLES = new Set(['system', 'user', 'assistant', 'tool']);

function validateMessages(value: unknown): unknown[] {
  if (!Array.isArray(value) || value.length === 0 || value.length > 60) {
    throw new HttpError(400, 'Lista de mensagens inválida.', 'messages');
  }
  for (const message of value) {
    const role = (message as { role?: unknown } | null)?.role;
    if (typeof role !== 'string' || !ROLES.has(role)) throw new HttpError(400, 'Mensagem com papel inválido.', 'messages');
  }
  return value;
}

function clampNumber(value: unknown, min: number, max: number, fallback: number): number {
  const number = Number(value);
  return Number.isFinite(number) ? Math.min(max, Math.max(min, number)) : fallback;
}

function deepseekProxy(policyOf: (env: ServerEnv) => QuotaPolicy, options: { maxTokens: number; allowTools: boolean }): Handler {
  return async (request, ctx) => {
    const locale = localeOf(request);
    if (!ctx.env.deepseekKey) {
      throw new HttpError(
        503,
        locale === 'en'
          ? 'Free use is unavailable: DEEPSEEK_API_KEY is not configured on the server.'
          : 'Uso gratuito indisponível: DEEPSEEK_API_KEY não configurada no servidor.',
        'unavailable',
      );
    }

    const body = await readJson(request);
    const messages = validateMessages(body['messages']);
    const unit = validId(request.headers.get('x-simetrics-unit'));
    if (!unit) throw new HttpError(400, 'Identificador da pergunta ausente.', 'unit');

    const policy = policyOf(ctx.env);
    const identity = identityOf(request, ctx);
    const decision = await consume(ctx.store, policy, identity, unit, ctx.env.quotaSalt);
    if (!decision.ok) {
      const messageKey = decision.reason === 'device' ? `${policy.scope}:device` : decision.reason;
      const message = QUOTA_MESSAGES[messageKey]?.[locale](decision.limit) ?? 'Cota esgotada.';
      throw new HttpError(429, message, `quota_${decision.reason}`, {
        'X-Free-Limit': String(decision.limit),
        'X-Free-Used': String(decision.used),
      });
    }

    const tools = options.allowTools && Array.isArray(body['tools']) && body['tools'].length <= 16 ? body['tools'] : undefined;
    const responseFormat = (body['response_format'] as { type?: unknown } | undefined)?.type === 'json_object';
    const upstreamBody = {
      model: ctx.env.deepseekModel,
      messages,
      stream: body['stream'] === true,
      temperature: clampNumber(body['temperature'], 0, 1.5, 0.2),
      max_tokens: Math.round(clampNumber(body['max_tokens'], 1, options.maxTokens, options.maxTokens)),
      ...(tools ? { tools, tool_choice: body['tool_choice'] ?? 'auto' } : {}),
      ...(responseFormat ? { response_format: { type: 'json_object' } } : {}),
    };

    let upstream: Response;
    try {
      upstream = await fetch(`${ctx.env.deepseekBaseUrl}/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${ctx.env.deepseekKey}` },
        body: JSON.stringify(upstreamBody),
      });
    } catch (cause) {
      if (decision.newUnit) await refund(ctx.store, policy, identity, unit, ctx.env.quotaSalt);
      throw cause;
    }

    if (!upstream.ok) {
      if (decision.newUnit) await refund(ctx.store, policy, identity, unit, ctx.env.quotaSalt);
      const detail = (await upstream.json().catch(() => ({}))) as { error?: { message?: string } };
      // Erro de autenticação do provedor diz respeito à chave do servidor, não ao usuário.
      const message =
        upstream.status === 401 || upstream.status === 403
          ? 'A chave DeepSeek do servidor foi recusada pelo provedor.'
          : (detail.error?.message ?? `DeepSeek: erro HTTP ${upstream.status}`);
      throw new HttpError(upstream.status === 429 ? 429 : 502, message, 'upstream');
    }

    return new Response(upstream.body, {
      status: 200,
      headers: {
        'Content-Type': upstream.headers.get('content-type') ?? 'application/json',
        'Cache-Control': 'no-store',
        'X-Free-Limit': String(decision.limit),
        'X-Free-Used': String(decision.used),
      },
    });
  };
}

// ---------------------------------------------------------------------------------------
// Nome de tema do k-means — o prompt é montado aqui, então o endpoint não serve de LLM livre.

const LABEL_MAX_SAMPLES = 8;
const LABEL_MAX_ABSTRACT = 800;
const LABEL_MAX_TERMS = 12;

function buildLabelPrompt(samples: { title?: unknown; abstract?: unknown }[], terms: string[], locale: Locale): string {
  const documents = samples
    .slice(0, LABEL_MAX_SAMPLES)
    .map((sample) => {
      const title = String(sample.title ?? '').slice(0, 300).trim();
      const abstract = String(sample.abstract ?? '').slice(0, LABEL_MAX_ABSTRACT).trim();
      return locale === 'en'
        ? `- Title: ${title || 'No title'}\n  Abstract: ${abstract || 'No abstract'}`
        : `- Título: ${title || 'Sem título'}\n  Resumo: ${abstract || 'Sem resumo'}`;
    })
    .join('\n\n');
  const termBlock = terms.slice(0, LABEL_MAX_TERMS).join(', ');

  return locale === 'en'
    ? `You are an expert data scientist specializing in academic literature review.
Below are scientific papers that an algorithm grouped by textual similarity.
Key characteristic terms of this cluster: ${termBlock || 'none available'}
Representative papers:
${documents || 'No papers available.'}

Synthesize the central research theme that unifies this group.
Respond ONLY with the theme title in English, in at most 4 words. No punctuation, no quotes, no prefix.`
    : `Você é um cientista de dados especialista em revisão de literatura.
Abaixo estão artigos científicos que um algoritmo agrupou por similaridade textual.
Termos mais característicos do agrupamento: ${termBlock || 'não disponíveis'}
Artigos representativos:
${documents || 'Nenhum artigo disponível.'}

Sintetize o tema central que unifica este grupo.
Responda APENAS com o nome do tema, em português, com no máximo 4 palavras. Sem pontuação final, sem aspas, sem prefixos.`;
}

function sanitizeThemeName(raw: string): string {
  const cleaned = raw
    .replace(/[\n\r]+/g, ' ')
    .replace(/["'*`]/g, '')
    .replace(/^\s*(tema|theme|título|title|nome|name)\s*:\s*/i, '')
    .replace(/\.\s*$/, '')
    .trim()
    .slice(0, 60);
  return cleaned.replace(/\p{L}[\p{L}\p{M}]*/gu, (word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase());
}

const handleThemeLabel: Handler = async (request, ctx) => {
  if (!ctx.env.deepseekKey) throw new HttpError(503, 'DEEPSEEK_API_KEY não configurada no servidor.', 'unavailable');
  const body = await readJson(request, 100_000);
  const samples = Array.isArray(body['samples']) ? (body['samples'] as { title?: unknown; abstract?: unknown }[]) : [];
  const terms = Array.isArray(body['topTerms']) ? body['topTerms'].map(String) : [];
  if (samples.length === 0 && terms.length === 0) throw new HttpError(400, 'Envie amostras ou termos.', 'input');
  const locale: Locale = body['locale'] === 'en' ? 'en' : 'pt';

  // Sem cota por dispositivo — só um teto diário por IP, contado como se o IP fosse o
  // dispositivo, num escopo que muda de nome a cada dia.
  const day = new Date().toISOString().slice(0, 10);
  const policy: QuotaPolicy = { scope: `label/${day}`, deviceLimit: ctx.env.labelIpDaily, ipDailyLimit: 0, maxRequestsPerUnit: 1 };
  const decision = await consume(ctx.store, policy, { device: ctx.ip || 'unknown', ip: ctx.ip }, randomUUID(), ctx.env.quotaSalt);
  if (!decision.ok) throw new HttpError(429, QUOTA_MESSAGES['ip']![locale](0), 'quota_ip');

  const upstream = await fetch(`${ctx.env.deepseekBaseUrl}/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${ctx.env.deepseekKey}` },
    body: JSON.stringify({
      model: ctx.env.deepseekModel,
      messages: [{ role: 'user', content: buildLabelPrompt(samples, terms, locale) }],
      max_tokens: 200,
      temperature: 0.2,
    }),
  });
  if (!upstream.ok) throw new HttpError(502, `DeepSeek: erro HTTP ${upstream.status}`, 'upstream');
  const data = (await upstream.json()) as { choices?: { message?: { content?: string } }[] };
  const name = sanitizeThemeName(data.choices?.[0]?.message?.content ?? '');
  if (!name) throw new HttpError(502, 'O modelo devolveu uma resposta vazia.', 'empty');
  return json(200, { name });
};

// ---------------------------------------------------------------------------------------
// Jev — livre para os usuários, com a chave do servidor

const handleJev: Handler = async (request, ctx) => {
  const authorization =
    request.headers.get('authorization')?.trim() || (ctx.env.typesafeKey ? `Bearer ${ctx.env.typesafeKey}` : '');
  if (!authorization) {
    throw new HttpError(
      401,
      'Jev indisponível: nenhuma chave informada e TYPESAFE_API_KEY não configurada no servidor.',
      'unavailable',
    );
  }

  const body = await readJson(request, JEV_MAX_PAYLOAD_BYTES);
  const questions = body['questions'];
  if (typeof questions !== 'object' || questions === null || Object.keys(questions).length === 0 || Object.keys(questions).length > 8) {
    throw new HttpError(400, 'Perguntas do Jev ausentes ou em excesso.', 'questions');
  }

  // Sem `Origin`: a API do Jev recusa (403) requisições que o trazem.
  const upstream = await fetch(JEV_UPSTREAM, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: authorization },
    body: JSON.stringify(body),
  });
  const headers: Record<string, string> = {
    'Content-Type': upstream.headers.get('content-type') ?? 'application/json',
    'Cache-Control': 'no-store',
  };
  const retryAfter = upstream.headers.get('retry-after');
  if (retryAfter) headers['Retry-After'] = retryAfter;
  return new Response(await upstream.text(), { status: upstream.status, headers });
};

// ---------------------------------------------------------------------------------------

const ROUTES: Record<`/${string}`, Handler> = {
  '/api/status': handleStatus,
  '/api/simi/chat/completions': deepseekProxy(simiPolicy, { maxTokens: 8192, allowTools: true }),
  '/api/hybrid/chat/completions': deepseekProxy(hybridPolicy, { maxTokens: 8192, allowTools: false }),
  '/api/themes/label': handleThemeLabel,
  '/api/jev/systemone': handleJev,
};

export const API_PATHS = Object.keys(ROUTES) as `/${string}`[];

/** Atende a requisição, ou devolve `null` se o caminho não for um endpoint do Simetrics. */
export async function handleApi(request: Request, ctx: ApiContext): Promise<Response | null> {
  const handler = ROUTES[new URL(request.url).pathname as `/${string}`];
  if (!handler) return null;
  try {
    return await handler(request, ctx);
  } catch (cause) {
    return errorResponse(cause);
  }
}
