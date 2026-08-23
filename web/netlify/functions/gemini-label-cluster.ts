import type { Config, Context } from '@netlify/functions';

import {
  ACADEMIC_SAFETY_SETTINGS,
  LABEL_MODEL,
  RequestError,
  createClient,
  errorResponse,
  readJsonBody,
} from './_shared';

/**
 * Nomeia UM agrupamento temático.
 *
 * Uma requisição por agrupamento, e não uma para todos, por causa do timeout. O laço
 * equivalente no Python (utils.py:1215) percorre até dez agrupamentos com uma pausa de
 * 2,5 s entre chamadas — cerca de 25 s no total, muito além dos 10 s que a Netlify
 * concede a uma função síncrona. Aqui o cliente orquestra o laço, e cada requisição
 * carrega um prompt curto que responde em um ou dois segundos.
 */

interface LabelRequest {
  samples?: { title?: string; abstract?: string }[];
  topTerms?: string[];
}

/** Limites defensivos: o corpo vem do cliente e não é confiável. */
const MAX_SAMPLES = 8;
const MAX_ABSTRACT = 800;
const MAX_TERMS = 12;
const MAX_NAME_LENGTH = 60;

function buildPrompt(samples: LabelRequest['samples'], topTerms: string[]): string {
  const documentBlock = (samples ?? [])
    .slice(0, MAX_SAMPLES)
    .map((sample) => {
      const title = String(sample.title ?? '').slice(0, 300).trim();
      const abstract = String(sample.abstract ?? '').slice(0, MAX_ABSTRACT).trim();
      return `- Título: ${title || 'Sem título'}\n  Resumo: ${abstract || 'Sem resumo'}`;
    })
    .join('\n\n');

  const termBlock = topTerms.slice(0, MAX_TERMS).join(', ');

  return `Você é um cientista de dados especialista em revisão de literatura.

Abaixo estão artigos científicos que um algoritmo agrupou por similaridade textual.

Termos mais característicos do agrupamento: ${termBlock || 'não disponíveis'}

Artigos representativos:
${documentBlock || 'Nenhum artigo disponível.'}

Sua tarefa: sintetize o tema central que unifica esta escola de pesquisa.

Responda APENAS com o nome do tema, em português, com no máximo 4 palavras.
Sem pontuação final, sem aspas, sem prefixos como "Tema:".`;
}

/** Limpa a resposta do modelo, que às vezes vem com aspas, markdown ou prefixos. */
function sanitizeName(raw: string): string {
  const cleaned = raw
    .replace(/[\n\r]+/g, ' ')
    .replace(/["'*`]/g, '')
    .replace(/^\s*(tema|título|nome)\s*:\s*/i, '')
    .replace(/\.\s*$/, '')
    .trim();

  if (!cleaned) return '';

  const truncated = cleaned.slice(0, MAX_NAME_LENGTH);
  // Título em maiúsculas iniciais, como o `.title()` que o Python aplicava.
  return truncated.replace(
    /\p{L}[\p{L}\p{M}]*/gu,
    (word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase(),
  );
}

export default async (request: Request, _context: Context): Promise<Response> => {
  try {
    const body = await readJsonBody<LabelRequest>(request);

    const samples = Array.isArray(body.samples) ? body.samples : [];
    const topTerms = Array.isArray(body.topTerms) ? body.topTerms.map(String) : [];

    if (samples.length === 0 && topTerms.length === 0) {
      throw new RequestError('Envie ao menos amostras ou termos característicos.', 400);
    }

    const client = createClient();
    const response = await client.models.generateContent({
      model: LABEL_MODEL,
      contents: buildPrompt(samples, topTerms),
      config: {
        safetySettings: [...ACADEMIC_SAFETY_SETTINGS],
        // O nome tem no máximo quatro palavras; um teto baixo evita divagação e corta custo.
        maxOutputTokens: 64,
        temperature: 0.2,
      },
    });

    const name = sanitizeName(response.text ?? '');
    if (!name) {
      throw new RequestError('O modelo devolveu uma resposta vazia.', 502);
    }

    return Response.json({ name });
  } catch (cause) {
    return errorResponse(cause);
  }
};

export const config: Config = {
  path: '/api/gemini/label-cluster',
};
