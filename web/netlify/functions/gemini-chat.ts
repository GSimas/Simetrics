import type { Config, Context } from '@netlify/functions';

import {
  ACADEMIC_SAFETY_SETTINGS,
  CHAT_MODEL,
  RequestError,
  createClient,
  errorResponse,
  readJsonBody,
} from './_shared';

/**
 * Assistente Científico, com resposta em streaming.
 *
 * Recebe do navegador a pergunta, o histórico e — o ponto central da arquitetura — apenas
 * os documentos que o BM25 selecionou como relevantes, mais um panorama agregado da base.
 *
 * O app Streamlit injetava a base INTEIRA em JSON no `system_instruction` (Geral.py:2227).
 * Medido na base de exemplo: 2,2 MB para 973 documentos, e mais de 22 MB no teto de
 * 10.000 — muito acima do limite de 6 MB de payload da Netlify. O recorte por BM25 leva
 * os mesmos 40 documentos relevantes em cerca de 48 kB.
 *
 * O streaming resolve a outra metade do problema: uma resposta longa leva mais que os
 * 10 s de uma função síncrona, mas em streaming o primeiro byte sai em menos de um
 * segundo e a conexão permanece aberta enquanto o texto chega.
 */

interface ContextDocument {
  title?: string;
  authors?: string;
  year?: number | null;
  venue?: string;
  citations?: number;
  keywords?: string;
  abstract?: string;
  theme?: string;
}

interface ChatRequest {
  question?: string;
  history?: { role?: string; content?: string }[];
  documents?: ContextDocument[];
  aggregate?: Record<string, unknown>;
}

const MAX_QUESTION_LENGTH = 2000;
const MAX_HISTORY_TURNS = 12;
const MAX_DOCUMENTS = 60;

function buildSystemInstruction(
  documents: ContextDocument[],
  aggregate: Record<string, unknown> | undefined,
): string {
  return `Você é um conselheiro acadêmico sênior, especialista em cienciometria, operando dentro da plataforma Simetrics.

## Panorama da base do usuário
${aggregate ? JSON.stringify(aggregate) : 'Não disponível.'}

## Documentos relevantes para a pergunta atual
${JSON.stringify(documents)}

## Suas tarefas
- Responder com base exclusivamente nos dados acima.
- Recomendar artigos fundamentais considerando tema, resumo e impacto (citações).
- Sugerir periódicos para submissão a partir do perfil do manuscrito descrito pelo usuário.
- Identificar especialistas para parceria ou referência.

## Regras absolutas
- Recomende SOMENTE itens presentes nos dados acima. Nunca invente títulos, autores ou periódicos.
- Cite títulos de documentos e nomes de autores exatamente como aparecem.
- Os documentos listados são o recorte mais relevante à pergunta, e não a base inteira. Para perguntas sobre panorama geral, use a seção de panorama.
- Se os dados não permitirem responder, diga isso claramente em vez de especular.
- Responda em português, com tom acadêmico, analítico e direto.`;
}

export default async (request: Request, _context: Context): Promise<Response> => {
  try {
    const body = await readJsonBody<ChatRequest>(request);

    const question = String(body.question ?? '').trim();
    if (!question) {
      throw new RequestError('A pergunta não pode estar vazia.', 400);
    }
    if (question.length > MAX_QUESTION_LENGTH) {
      throw new RequestError('A pergunta excede o tamanho máximo permitido.', 400);
    }

    const documents = (Array.isArray(body.documents) ? body.documents : []).slice(
      0,
      MAX_DOCUMENTS,
    );

    // O histórico entra no formato de conteúdos do Gemini, alternando papéis.
    const history = (Array.isArray(body.history) ? body.history : [])
      .slice(-MAX_HISTORY_TURNS)
      .filter((turn) => typeof turn.content === 'string' && turn.content.trim())
      .map((turn) => ({
        role: turn.role === 'user' ? 'user' : 'model',
        parts: [{ text: String(turn.content) }],
      }));

    const client = createClient();
    const stream = await client.models.generateContentStream({
      model: CHAT_MODEL,
      contents: [...history, { role: 'user', parts: [{ text: question }] }],
      config: {
        systemInstruction: buildSystemInstruction(documents, body.aggregate),
        safetySettings: [...ACADEMIC_SAFETY_SETTINGS],
        temperature: 0.4,
      },
    });

    const encoder = new TextEncoder();
    const readable = new ReadableStream<Uint8Array>({
      async start(controller) {
        try {
          for await (const chunk of stream) {
            const text = chunk.text;
            if (text) controller.enqueue(encoder.encode(text));
          }
        } catch (cause) {
          // O cabeçalho já foi enviado, então não há como mudar o status: a falha vai
          // como texto ao final do fluxo, e o cliente a exibe junto do que já recebeu.
          console.error('Falha durante o streaming:', cause);
          controller.enqueue(
            encoder.encode('\n\n_(A geração foi interrompida por um erro. Tente novamente.)_'),
          );
        } finally {
          controller.close();
        }
      },
    });

    return new Response(readable, {
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'Cache-Control': 'no-store',
        // Impede que proxies acumulem a resposta e anulem o streaming.
        'X-Accel-Buffering': 'no',
      },
    });
  } catch (cause) {
    return errorResponse(cause);
  }
};

export const config: Config = {
  path: '/api/gemini/chat',
};
