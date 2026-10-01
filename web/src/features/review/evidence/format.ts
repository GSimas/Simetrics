import type { EvidenceTargetKey, ExtractionValue, ReviewState } from '@/core/review/types';

/** Rótulos para mostrar perguntas e respostas fora do formulário (cartões, planilha). */

export function formatValue(value: ExtractionValue | undefined, words: { yes: string; no: string }): string {
  if (value === undefined) return '';
  if (typeof value === 'boolean') return value ? words.yes : words.no;
  if (Array.isArray(value)) return value.join('; ');
  return String(value);
}

export function targetLabel(review: ReviewState, target: EvidenceTargetKey, exclusionLabel: string): string {
  if (target === 'ft-exclusion') return exclusionLabel;
  const [kind, id] = target.split(':') as ['extraction' | 'quality', string];
  if (kind === 'extraction') return review.extractionFields.find((field) => field.id === id)?.label || '—';
  return review.qualityQuestions.find((question) => question.id === id)?.text || '—';
}

/** Valor proposto pela IA como o revisor o lê: na qualidade, o rótulo da resposta, não o id. */
export function suggestionText(
  review: ReviewState,
  target: EvidenceTargetKey,
  value: ExtractionValue,
  words: { yes: string; no: string },
): string {
  if (target.startsWith('quality:')) return review.qualityAnswers.find((answer) => answer.id === value)?.label ?? String(value);
  return formatValue(value, words);
}

export function answerText(review: ReviewState, key: string, target: EvidenceTargetKey, words: { yes: string; no: string }): string {
  if (target === 'ft-exclusion') {
    const reason = review.decisions[key]?.ftReason;
    return review.criteria.find((criterion) => criterion.id === reason)?.text ?? '';
  }
  const [kind, id] = target.split(':') as ['extraction' | 'quality', string];
  if (kind === 'extraction') return formatValue(review.extraction[key]?.values[id], words);
  const answerId = review.quality[key]?.[id];
  return review.qualityAnswers.find((answer) => answer.id === answerId)?.label ?? '';
}
