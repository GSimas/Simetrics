import { finalSelection, hasQualityChecklist, includedStudies, scoreStudy } from './quality';
import { advancesToFullText } from './flow';
import type { ScreeningRecord } from './records';
import { FRAMEWORK_FIELDS, type ReviewState } from './types';

/**
 * Completude de cada etapa da revisão, para a sensação de avanço na tela e na barra
 * lateral. Protocolo, triagem, qualidade e extração são trabalho do revisor; síntese e
 * PRISMA são saídas, prontas quando o trabalho de que dependem termina.
 */

export const REVIEW_STEPS = ['protocol', 'screening', 'quality', 'extraction', 'synthesis', 'prisma'] as const;
export type ReviewStepId = (typeof REVIEW_STEPS)[number];

export type StepState =
  /** Nada feito ainda. */
  | 'todo'
  | 'active'
  | 'done'
  /** Depende de uma etapa anterior (sem estudos incluídos, por exemplo). */
  | 'waiting'
  /** Checklist de qualidade ou formulário de extração não definidos: a etapa fica de fora. */
  | 'off';

export interface StepProgress {
  state: StepState;
  done: number;
  total: number;
  /** 0–1. */
  ratio: number;
}

export interface ReviewProgress {
  steps: Record<ReviewStepId, StepProgress>;
  /** Média das etapas de trabalho que valem para esta revisão (0–1). */
  overall: number;
}

function progress(done: number, total: number): StepProgress {
  const ratio = total > 0 ? Math.min(1, done / total) : 0;
  return { state: ratio >= 1 ? 'done' : done > 0 ? 'active' : 'todo', done, total, ratio };
}

const waiting: StepProgress = { state: 'waiting', done: 0, total: 0, ratio: 0 };
const off: StepProgress = { state: 'off', done: 0, total: 0, ratio: 0 };

/** Itens do protocolo: título, objetivo, estrutura da pergunta, perguntas, conceitos e critérios. */
function protocolProgress(review: ReviewState): StepProgress {
  const filled = (text: string | undefined) => Boolean(text?.trim());
  const fields = FRAMEWORK_FIELDS[review.framework];
  const checks = [
    filled(review.title),
    filled(review.objective),
    ...(fields.length > 0 ? [fields.every((field) => filled(review.frameworkValues[field]))] : []),
    review.questions.some((question) => filled(question.text)),
    review.concepts.some((concept) => concept.terms.length > 0),
    review.criteria.some((criterion) => criterion.kind === 'inclusion' && filled(criterion.text)),
    review.criteria.some((criterion) => criterion.kind === 'exclusion' && filled(criterion.text)),
  ];
  return progress(checks.filter(Boolean).length, checks.length);
}

/** Decisões tomadas sobre as decisões possíveis: título/resumo de todos + texto completo dos que avançaram. */
function screeningProgress(review: ReviewState, records: readonly ScreeningRecord[]): StepProgress {
  if (records.length === 0) return waiting;
  let done = 0;
  let total = records.length;
  for (const record of records) {
    const screening = review.decisions[record.key];
    if (!screening?.ta) continue;
    done += 1;
    if (!advancesToFullText(screening)) continue;
    total += 1;
    if (screening.ft) done += 1;
  }
  return progress(done, total);
}

export function reviewProgress(review: ReviewState, records: readonly ScreeningRecord[]): ReviewProgress {
  const protocol = protocolProgress(review);
  const screening = screeningProgress(review, records);

  const included = includedStudies(records, review);
  const quality = !hasQualityChecklist(review)
    ? off
    : included.length === 0
      ? waiting
      : progress(included.filter((study) => scoreStudy(review, study.key).complete).length, included.length);

  const selected = finalSelection(records, review);
  const extraction =
    review.extractionFields.length === 0
      ? off
      : selected.length === 0
        ? waiting
        : progress(selected.filter((study) => review.extraction[study.key]?.done).length, selected.length);

  // Saídas: acompanham a etapa de que dependem.
  const synthesisSource = [extraction, quality, screening].find((step) => step.state !== 'off') ?? screening;
  const synthesis = { ...synthesisSource, state: synthesisSource.state === 'done' ? 'done' : 'waiting' } as StepProgress;
  const prisma = { ...screening, state: screening.state === 'done' ? 'done' : 'waiting' } as StepProgress;

  const work = [protocol, screening, quality, extraction].filter((step) => step.state !== 'off');
  const overall = work.reduce((sum, step) => sum + step.ratio, 0) / work.length;

  return { steps: { protocol, screening, quality, extraction, synthesis, prisma }, overall };
}

/** Etapas que funcionam sem base carregada: o protocolo se escreve antes da busca. */
export const STEPS_WITHOUT_DATA: readonly ReviewStepId[] = ['protocol'];

export type NextActionKind =
  | 'title'
  | 'objective'
  | 'framework'
  | 'question'
  | 'concept'
  | 'inclusion'
  | 'exclusion'
  | 'import'
  | 'screen-ta'
  | 'screen-ft'
  | 'quality'
  | 'extraction-form'
  | 'extraction'
  | 'report';

export interface NextAction {
  kind: NextActionKind;
  step: ReviewStepId;
  /** Quantos itens faltam, quando a ação é uma fila (registros, estudos). */
  count?: number;
}

/**
 * O que falta fazer primeiro, na ordem do fluxo: o protocolo campo a campo, a importação,
 * as duas etapas da triagem, a qualidade (se houver checklist), o formulário e a extração,
 * e por fim o relato. Qualidade é opcional (revisões de escopo não avaliam); extração não.
 */
export function nextReviewAction(review: ReviewState, records: readonly ScreeningRecord[]): NextAction {
  const filled = (text: string | undefined) => Boolean(text?.trim());
  if (!filled(review.title)) return { kind: 'title', step: 'protocol' };
  if (!filled(review.objective)) return { kind: 'objective', step: 'protocol' };
  if (FRAMEWORK_FIELDS[review.framework].some((field) => !filled(review.frameworkValues[field])))
    return { kind: 'framework', step: 'protocol' };
  if (!review.questions.some((question) => filled(question.text))) return { kind: 'question', step: 'protocol' };
  if (!review.concepts.some((concept) => concept.terms.length > 0)) return { kind: 'concept', step: 'protocol' };
  if (!review.criteria.some((c) => c.kind === 'inclusion' && filled(c.text))) return { kind: 'inclusion', step: 'protocol' };
  if (!review.criteria.some((c) => c.kind === 'exclusion' && filled(c.text))) return { kind: 'exclusion', step: 'protocol' };
  if (records.length === 0) return { kind: 'import', step: 'screening' };

  let titleAbstract = 0;
  let fullText = 0;
  for (const record of records) {
    const screening = review.decisions[record.key];
    if (!screening?.ta) titleAbstract += 1;
    else if (advancesToFullText(screening) && !screening.ft) fullText += 1;
  }
  if (titleAbstract > 0) return { kind: 'screen-ta', step: 'screening', count: titleAbstract };
  if (fullText > 0) return { kind: 'screen-ft', step: 'screening', count: fullText };

  const { steps } = reviewProgress(review, records);
  if (steps.quality.state === 'todo' || steps.quality.state === 'active')
    return { kind: 'quality', step: 'quality', count: steps.quality.total - steps.quality.done };
  if (review.extractionFields.length === 0) return { kind: 'extraction-form', step: 'protocol' };
  if (steps.extraction.state === 'todo' || steps.extraction.state === 'active')
    return { kind: 'extraction', step: 'extraction', count: steps.extraction.total - steps.extraction.done };
  return { kind: 'report', step: 'prisma' };
}
