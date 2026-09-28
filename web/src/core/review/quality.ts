import type { ScreeningRecord } from './records';
import type { QualityAnswer, ReviewState } from './types';

/**
 * Avaliação de qualidade e seleção final, no modelo do Parsifal: a nota do estudo é a
 * soma dos pesos das respostas; a nota máxima, o número de perguntas vezes o maior peso.
 * Com nota de corte e a opção de excluir ligada, os estudos avaliados abaixo dela saem da
 * seleção final — estudos ainda não avaliados por completo nunca saem.
 */

export function defaultQualityAnswers(locale: 'pt' | 'en', makeId: () => string = () => crypto.randomUUID()): QualityAnswer[] {
  const labels = locale === 'en' ? ['Yes', 'Partially', 'No'] : ['Sim', 'Parcialmente', 'Não'];
  return [
    { id: makeId(), label: labels[0]!, weight: 1 },
    { id: makeId(), label: labels[1]!, weight: 0.5 },
    { id: makeId(), label: labels[2]!, weight: 0 },
  ];
}

export function hasQualityChecklist(review: ReviewState): boolean {
  return review.qualityQuestions.length > 0 && review.qualityAnswers.length > 0;
}

export function maxQualityScore(review: ReviewState): number {
  if (!hasQualityChecklist(review)) return 0;
  const highest = Math.max(...review.qualityAnswers.map((answer) => answer.weight));
  return review.qualityQuestions.length * Math.max(0, highest);
}

export interface QualityScore {
  score: number;
  max: number;
  answered: number;
  total: number;
  complete: boolean;
  /** `null` enquanto não há nota de corte ou a avaliação está incompleta. */
  passes: boolean | null;
}

export function scoreStudy(review: ReviewState, key: string): QualityScore {
  const weights = new Map(review.qualityAnswers.map((answer) => [answer.id, answer.weight]));
  const responses = review.quality[key] ?? {};
  let score = 0;
  let answered = 0;
  for (const question of review.qualityQuestions) {
    const weight = weights.get(responses[question.id] ?? '');
    if (weight === undefined) continue;
    score += weight;
    answered += 1;
  }
  const total = review.qualityQuestions.length;
  const complete = total > 0 && answered === total;
  return {
    score,
    max: maxQualityScore(review),
    answered,
    total,
    complete,
    passes: complete && review.qualityCutoff !== null ? score >= review.qualityCutoff : null,
  };
}

/** Incluído na triagem do texto completo. */
export function isIncludedAfterScreening(review: ReviewState, key: string): boolean {
  return review.decisions[key]?.ft === 'include';
}

export function isExcludedByQuality(review: ReviewState, key: string): boolean {
  if (!review.excludeBelowCutoff || review.qualityCutoff === null || !hasQualityChecklist(review)) return false;
  return scoreStudy(review, key).passes === false;
}

/** Estudos que a avaliação de qualidade e a extração percorrem: os incluídos no texto completo. */
export function includedStudies(records: readonly ScreeningRecord[], review: ReviewState): ScreeningRecord[] {
  return records.filter((record) => isIncludedAfterScreening(review, record.key));
}

/** Seleção final: incluídos que não caíram na nota de corte. */
export function finalSelection(records: readonly ScreeningRecord[], review: ReviewState): ScreeningRecord[] {
  return includedStudies(records, review).filter((record) => !isExcludedByQuality(review, record.key));
}
