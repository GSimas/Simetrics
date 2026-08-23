import { GoogleGenAI, HarmBlockThreshold, HarmCategory } from '@google/genai';
import type { SafetySetting } from '@google/genai';

/**
 * Base compartilhada pelas funções que falam com o Gemini.
 *
 * Estas funções são o único lugar do sistema que enxerga a `GEMINI_API_KEY`. O bundle do
 * cliente nunca a recebe — é justamente por isso que elas existem, já que todo o resto do
 * processamento acontece no navegador.
 */

/**
 * Modelos configuráveis por variável de ambiente.
 *
 * Ficam em variáveis, e não fixos no código, porque os identificadores do Gemini são
 * descontinuados periodicamente. Quando um deles sair do ar, trocar no painel do Netlify
 * resolve sem novo deploy do código.
 */
export const CHAT_MODEL = process.env['GEMINI_CHAT_MODEL'] ?? 'gemini-2.5-flash';
export const LABEL_MODEL = process.env['GEMINI_LABEL_MODEL'] ?? 'gemini-2.5-flash-lite';

/** Limite de payload aceito, bem abaixo dos 6 MB da plataforma. */
const MAX_PAYLOAD_BYTES = 1_500_000;

export class RequestError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = 'RequestError';
  }
}

/** Cliente do Gemini, ou erro claro quando a chave não está configurada. */
export function createClient(): GoogleGenAI {
  const apiKey = process.env['GEMINI_API_KEY']?.trim();

  if (!apiKey) {
    throw new RequestError(
      'GEMINI_API_KEY não configurada. Defina a variável no painel do Netlify.',
      503,
    );
  }

  return new GoogleGenAI({ apiKey });
}

/** Lê e valida o corpo JSON da requisição. */
export async function readJsonBody<T>(request: Request): Promise<T> {
  if (request.method !== 'POST') {
    throw new RequestError('Método não permitido. Use POST.', 405);
  }

  const raw = await request.text();

  if (raw.length > MAX_PAYLOAD_BYTES) {
    throw new RequestError(
      `Payload de ${Math.round(raw.length / 1024)} kB excede o limite. ` +
        'Reduza o número de documentos de contexto enviados.',
      413,
    );
  }

  try {
    return JSON.parse(raw) as T;
  } catch {
    throw new RequestError('Corpo da requisição não é um JSON válido.', 400);
  }
}

/** Converte qualquer falha numa resposta JSON com status apropriado. */
export function errorResponse(cause: unknown): Response {
  if (cause instanceof RequestError) {
    return Response.json({ error: cause.message }, { status: cause.status });
  }

  // Erros vindos da API do Gemini podem carregar detalhes da conta ou da chave; o cliente
  // recebe uma mensagem genérica e o detalhe fica no log da função.
  console.error('Falha ao contatar o Gemini:', cause);
  return Response.json(
    { error: 'Não foi possível gerar a resposta. Tente novamente em instantes.' },
    { status: 502 },
  );
}

/**
 * Configuração de segurança para textos acadêmicos — ⇄ utils.py:1241.
 *
 * Os filtros são desativados porque literatura científica discute rotineiramente
 * violência, doenças e conflito, e os limiares padrão bloqueiam resumos legítimos de
 * epidemiologia, estudos de conflito e psiquiatria.
 */
export const ACADEMIC_SAFETY_SETTINGS: SafetySetting[] = [
  { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_NONE },
  { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_NONE },
  {
    category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    threshold: HarmBlockThreshold.BLOCK_NONE,
  },
  {
    category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    threshold: HarmBlockThreshold.BLOCK_NONE,
  },
];
