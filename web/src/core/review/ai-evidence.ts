import type { ChatPrompt } from '@/core/hybrid/prompts';
import { extractionTarget, qualityTarget } from './evidence';
import type { EvidenceTargetKey, ExtractionFieldType, ReviewState } from './types';

/**
 * Pergunta → resposta → trecho/página: o pedido à IA e a leitura da resposta dela.
 *
 * O modelo recebe o texto do PDF com as páginas marcadas e devolve, para cada pergunta, a
 * resposta e o trecho literal que a sustenta. Nada do que ele diz entra direto no
 * formulário: vira proposta, o trecho é procurado no PDF (`locateInPages`) e o revisor confere.
 */

export interface AiQuestion {
  target: EvidenceTargetKey;
  label: string;
  type: ExtractionFieldType | 'quality';
  options: string[];
}

/** As perguntas que a IA pode responder: campos da extração e perguntas da qualidade. */
export function aiQuestions(review: ReviewState, scope: 'extraction' | 'quality'): AiQuestion[] {
  if (scope === 'extraction') {
    return review.extractionFields
      .filter((field) => field.label.trim())
      .map((field) => ({ target: extractionTarget(field.id), label: field.label, type: field.type, options: field.options }));
  }
  return review.qualityQuestions
    .filter((question) => question.text.trim())
    .map((question) => ({
      target: qualityTarget(question.id),
      label: question.text,
      type: 'quality',
      options: review.qualityAnswers.map((answer) => answer.label).filter(Boolean),
    }));
}

/** ~45 mil tokens de artigo: cabe folgado nos modelos atuais e mantém o custo previsível. */
export const MAX_ARTICLE_CHARS = 180_000;

const REFERENCES_HEADING = /^\s*(references|referências|referencias|bibliography|bibliografia|literature cited|works cited)\s*$/im;

export interface PreparedArticle {
  text: string;
  /** O texto passou do limite e foi cortado. */
  truncated: boolean;
  /** As referências bibliográficas ficaram de fora — não respondem às perguntas e custam tokens. */
  droppedReferences: boolean;
  chars: number;
}

/** Texto do artigo com as páginas marcadas, sem as referências e dentro do limite. */
export function prepareArticle(pageTexts: readonly string[], pageLabel: string): PreparedArticle {
  const pages = [...pageTexts];
  let droppedReferences = false;
  // Só procura o título das referências da metade do artigo em diante: antes disso, a
  // palavra solta numa linha é mais provável num sumário.
  for (let index = Math.floor(pages.length / 2); index < pages.length; index += 1) {
    const match = REFERENCES_HEADING.exec(pages[index] ?? '');
    if (match) {
      pages[index] = pages[index]!.slice(0, match.index);
      pages.length = index + 1;
      droppedReferences = true;
      break;
    }
  }
  let text = pages.map((page, index) => `=== ${pageLabel} ${index + 1} ===\n${page.trim()}`).join('\n\n');
  const truncated = text.length > MAX_ARTICLE_CHARS;
  if (truncated) text = text.slice(0, MAX_ARTICLE_CHARS);
  return { text, truncated, droppedReferences, chars: text.length };
}

const TYPE_HINT: Record<AiQuestion['type'], string> = {
  text: 'text',
  number: 'number',
  boolean: 'true | false',
  date: 'YYYY-MM-DD',
  select: 'one of options',
  multiselect: 'array of options',
  quality: 'one of options',
};

export function buildEvidencePrompt(
  questions: readonly AiQuestion[],
  article: PreparedArticle,
  context: { title: string; locale: 'pt' | 'en' },
): ChatPrompt {
  const list = questions.map((question, index) => ({
    id: `q${index + 1}`,
    question: question.label,
    answer_type: TYPE_HINT[question.type],
    ...(question.options.length > 0 ? { options: question.options } : {}),
  }));
  const en = context.locale === 'en';
  const system = en
    ? `You extract data from scientific articles for a systematic review. Answer ONLY from the article text below — never from prior knowledge.
For each question return the answer and 1 or 2 supporting quotes copied VERBATIM from the article (exact words, 1–3 sentences, at most 400 characters each) with the page number shown in the "=== Page N ===" markers.
If the article does not report it, use "answer": null and "quotes": []. For options, answer with the option text exactly as given. Write "answer" and "rationale" in English; quotes stay in the article's language.
Reply with JSON only: {"answers":[{"id":"q1","answer":...,"quotes":[{"text":"...","page":1}],"rationale":"one short sentence"}]}`
    : `Você extrai dados de artigos científicos para uma revisão sistemática. Responda APENAS com base no texto do artigo abaixo — nunca com conhecimento prévio.
Para cada pergunta, devolva a resposta e 1 ou 2 trechos que a sustentam, copiados LITERALMENTE do artigo (as mesmas palavras, 1 a 3 frases, até 400 caracteres cada), com o número da página indicado nos marcadores "=== Página N ===".
Se o artigo não informar, use "answer": null e "quotes": []. Em perguntas com opções, responda com o texto da opção exatamente como veio. Escreva "answer" e "rationale" em português; os trechos ficam no idioma do artigo.
Responda só com JSON: {"answers":[{"id":"q1","answer":...,"quotes":[{"text":"...","page":1}],"rationale":"uma frase curta"}]}`;
  const user = `${en ? 'Questions' : 'Perguntas'}:\n${JSON.stringify(list, null, 1)}\n\n${en ? 'Article' : 'Artigo'}: ${context.title || '—'}\n\n${article.text}`;
  return { system, user };
}

export interface AiAnswer {
  target: EvidenceTargetKey;
  answer: unknown;
  quotes: { text: string; page: number }[];
  rationale: string;
}

/** Lê o JSON do modelo, tolerando cercas de código e texto em volta. Perguntas sem resposta ficam de fora. */
export function parseEvidenceResponse(raw: string, questions: readonly AiQuestion[]): AiAnswer[] {
  const start = raw.indexOf('{');
  const end = raw.lastIndexOf('}');
  if (start < 0 || end <= start) return [];
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw.slice(start, end + 1));
  } catch {
    return [];
  }
  const answers = (parsed as { answers?: unknown }).answers;
  if (!Array.isArray(answers)) return [];

  const result: AiAnswer[] = [];
  for (const item of answers) {
    if (typeof item !== 'object' || item === null) continue;
    const entry = item as { id?: unknown; answer?: unknown; quotes?: unknown; rationale?: unknown };
    const index = typeof entry.id === 'string' ? Number(entry.id.replace(/^q/i, '')) - 1 : -1;
    const question = questions[index];
    if (!question || entry.answer === null || entry.answer === undefined || entry.answer === '') continue;
    const quotes = Array.isArray(entry.quotes)
      ? entry.quotes
          .filter((quote): quote is { text?: unknown; page?: unknown } => typeof quote === 'object' && quote !== null)
          .map((quote) => ({ text: typeof quote.text === 'string' ? quote.text.trim() : '', page: Math.round(Number(quote.page)) || 0 }))
          .filter((quote) => quote.text)
          .slice(0, 2)
      : [];
    result.push({
      target: question.target,
      answer: entry.answer,
      quotes,
      rationale: typeof entry.rationale === 'string' ? entry.rationale.trim() : '',
    });
  }
  return result;
}
