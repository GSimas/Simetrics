import { scoreStudy } from './quality';
import type { ScreeningRecord } from './records';
import type { ExtractionField, ExtractionValue, ReviewState } from './types';

/**
 * Síntese dos estudos da seleção final: resumo de cada campo de extração (a "análise de
 * dados" do Parsifal) e as tabelas de características e de qualidade, que viram CSV e
 * entram no relatório.
 */

export type FieldSummary =
  | { kind: 'counts'; filled: number; counts: { label: string; count: number }[] }
  | { kind: 'numeric'; filled: number; mean: number; min: number; max: number }
  | { kind: 'filled'; filled: number };

function isFilled(value: ExtractionValue | undefined): boolean {
  if (value === undefined || value === '') return false;
  if (Array.isArray(value)) return value.length > 0;
  return !(typeof value === 'number' && Number.isNaN(value));
}

export function summarizeField(
  field: ExtractionField,
  studies: readonly ScreeningRecord[],
  review: ReviewState,
  labels: { yes: string; no: string },
): FieldSummary {
  const values = studies
    .map((study) => review.extraction[study.key]?.values[field.id])
    .filter(isFilled) as ExtractionValue[];

  if (field.type === 'select' || field.type === 'multiselect' || field.type === 'boolean') {
    const counts = new Map<string, number>(
      field.type === 'boolean' ? [[labels.yes, 0], [labels.no, 0]] : field.options.map((option) => [option, 0]),
    );
    for (const value of values) {
      const items = Array.isArray(value) ? value : [typeof value === 'boolean' ? (value ? labels.yes : labels.no) : String(value)];
      for (const item of items) counts.set(item, (counts.get(item) ?? 0) + 1);
    }
    return {
      kind: 'counts',
      filled: values.length,
      counts: [...counts.entries()].map(([label, count]) => ({ label, count })),
    };
  }

  if (field.type === 'number') {
    const numbers = values.map(Number).filter(Number.isFinite);
    if (numbers.length === 0) return { kind: 'filled', filled: 0 };
    return {
      kind: 'numeric',
      filled: numbers.length,
      mean: numbers.reduce((sum, value) => sum + value, 0) / numbers.length,
      min: Math.min(...numbers),
      max: Math.max(...numbers),
    };
  }

  return { kind: 'filled', filled: values.length };
}

export function formatExtractionValue(value: ExtractionValue | undefined, labels: { yes: string; no: string }): string {
  if (value === undefined) return '';
  if (Array.isArray(value)) return value.join('; ');
  if (typeof value === 'boolean') return value ? labels.yes : labels.no;
  return String(value);
}

function studyColumns(study: ScreeningRecord): Record<string, string | number> {
  return {
    title: study.title,
    authors: study.authors,
    year: study.year ?? '',
    venue: study.venue,
    doi: study.doi,
  };
}

/** Tabela de características: um estudo por linha, um campo de extração por coluna. */
export function extractionRows(
  studies: readonly ScreeningRecord[],
  review: ReviewState,
  labels: { yes: string; no: string },
): Record<string, string | number>[] {
  return studies.map((study) => {
    const extraction = review.extraction[study.key];
    const row: Record<string, string | number> = studyColumns(study);
    for (const field of review.extractionFields) {
      row[field.label || field.id] = formatExtractionValue(extraction?.values[field.id], labels);
    }
    return row;
  });
}

/** Tabela de qualidade: a resposta de cada pergunta e a nota total. */
export function qualityRows(studies: readonly ScreeningRecord[], review: ReviewState): Record<string, string | number>[] {
  const answers = new Map(review.qualityAnswers.map((answer) => [answer.id, answer.label]));
  return studies.map((study) => {
    const responses = review.quality[study.key] ?? {};
    const row: Record<string, string | number> = studyColumns(study);
    review.qualityQuestions.forEach((question, index) => {
      row[`Q${index + 1}`] = answers.get(responses[question.id] ?? '') ?? '';
    });
    const { score, max } = scoreStudy(review, study.key);
    row.score = score;
    row.max = max;
    return row;
  });
}
