import { truncate, type HybridDoc } from './documents';
import { TaxonomyParseError } from './prompts';
import { OTHER_CATEGORY_ID, type DocumentDecision, type HybridCategory } from './types';

/**
 * Montagem da requisição ao Jev e leitura da resposta — sem rede, para poder ser testado.
 *
 * Uma requisição por documento, com uma única pergunta Choice. O `state` leva só título,
 * palavras-chave e resumo (a documentação do Jev recomenda filtrar o contexto), e as
 * opções usam o formato estruturado `what`/`not_for`/`examples`, que é o que mais ajuda o
 * modelo a separar categorias vizinhas.
 */

const ABSTRACT_LIMIT = 2000;
const KEYWORDS_LIMIT = 300;
export const JEV_QUESTION_ID = 'theme';

export interface JevChoiceQuestion {
  type: 'choice';
  instructions: string;
  criteria: Record<string, { what: string; not_for?: string; examples?: string[] }>;
}

export interface JevRequestBody {
  state: { title: string; keywords?: string; abstract?: string };
  model: string;
  questions: Record<string, JevChoiceQuestion>;
}

export interface JevChoiceAnswer {
  type: 'choice';
  choice: string;
  probabilities: Record<string, number>;
  confidence: number;
}

export interface JevResponseBody {
  model: string;
  answers: Record<string, JevChoiceAnswer | { type: string }>;
  usage?: { input_tokens?: number; output_tokens?: number };
}

const INSTRUCTIONS =
  'Which research theme best describes this scientific document? Judge by the title, keywords and abstract. ' +
  'Choose the "other" option only when none of the listed themes fits the document.';

/**
 * Pergunta Choice da taxonomia. A opção "outro" vai sempre por último: a documentação do
 * Jev nota que as opções são avaliadas na ordem em que aparecem.
 */
export function buildThemeQuestion(categories: readonly HybridCategory[], otherName: string): JevChoiceQuestion {
  const criteria: JevChoiceQuestion['criteria'] = {};
  for (const category of categories) {
    criteria[category.name] = {
      what: category.what || category.name,
      ...(category.notFor ? { not_for: category.notFor } : {}),
      ...(category.examples.length > 0 ? { examples: category.examples } : {}),
    };
  }
  criteria[otherName] = {
    what: 'The document does not belong to any of the listed research themes.',
    not_for: 'Documents that clearly fit one of the listed themes.',
  };
  return { type: 'choice', instructions: INSTRUCTIONS, criteria };
}

export function buildJevRequest(doc: HybridDoc, question: JevChoiceQuestion, model: string): JevRequestBody {
  return {
    state: {
      title: doc.title || '(untitled)',
      ...(doc.keywords ? { keywords: truncate(doc.keywords, KEYWORDS_LIMIT) } : {}),
      ...(doc.abstract ? { abstract: truncate(doc.abstract, ABSTRACT_LIMIT) } : {}),
    },
    model,
    questions: { [JEV_QUESTION_ID]: question },
  };
}

/** Converte a resposta do Jev na decisão do documento (id da categoria + confiança). */
export function parseJevDecision(
  response: JevResponseBody,
  categories: readonly HybridCategory[],
  otherName: string,
): DocumentDecision {
  const answer = response.answers?.[JEV_QUESTION_ID];
  if (!answer || answer.type !== 'choice') {
    throw new TaxonomyParseError('Resposta do Jev sem a pergunta de tema.', 'Jev response is missing the theme question.');
  }
  const { choice, probabilities, confidence } = answer as JevChoiceAnswer;
  const categoryId =
    choice === otherName
      ? OTHER_CATEGORY_ID
      : (categories.find((category) => category.name === choice)?.id ?? OTHER_CATEGORY_ID);

  return {
    categoryId,
    confidence: Number.isFinite(confidence) ? confidence : 0,
    probability: Number.isFinite(probabilities?.[choice]) ? (probabilities[choice] as number) : 0,
  };
}
