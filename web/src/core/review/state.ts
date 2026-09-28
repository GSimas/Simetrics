import { DEFAULT_SEARCH_TARGETS } from './search-string';
import type { ScreeningRecord } from './records';
import {
  FRAMEWORKS,
  REVIEW_DEFAULTS,
  REVIEW_SCHEMA_VERSION,
  REVIEW_TYPES,
  type Criterion,
  type RecordScreening,
  type ReviewState,
  type ReviewType,
} from './types';

export function createReview(reviewerId: string, type: ReviewType = 'systematic', now = new Date()): ReviewState {
  const iso = now.toISOString();
  return {
    schemaVersion: REVIEW_SCHEMA_VERSION,
    type,
    title: '',
    objective: '',
    framework: REVIEW_DEFAULTS[type].framework,
    frameworkValues: {},
    questions: [],
    concepts: [],
    searchTargets: [...DEFAULT_SEARCH_TARGETS],
    criteria: [],
    decisions: {},
    reviewerId,
    createdAt: iso,
    updatedAt: iso,
  };
}

/** Há algo que valha salvar — um protocolo em branco não cria projeto. */
export function hasReviewContent(review: ReviewState | null): boolean {
  if (!review) return false;
  return (
    review.title.trim() !== '' ||
    review.objective.trim() !== '' ||
    Object.values(review.frameworkValues).some((value) => value.trim() !== '') ||
    review.questions.length > 0 ||
    review.concepts.length > 0 ||
    review.criteria.length > 0 ||
    Object.keys(review.decisions).length > 0
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function str(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : fallback;
}

function list<T>(value: unknown, guard: (item: unknown) => item is T): T[] {
  return Array.isArray(value) ? value.filter(guard) : [];
}

const hasId = (item: unknown): item is { id: string } => isRecord(item) && typeof item.id === 'string';

/**
 * Normaliza a revisão vinda de um projeto salvo ou importado. Campos ausentes ou de tipo
 * errado viram o padrão — o mesmo cuidado de `normalizeV1` com JSON editado à mão.
 */
export function normalizeReview(raw: unknown): ReviewState | null {
  if (!isRecord(raw)) return null;

  const type = REVIEW_TYPES.includes(raw.type as ReviewType) ? (raw.type as ReviewType) : 'systematic';
  const base = createReview(str(raw.reviewerId) || 'unknown', type);
  const framework = FRAMEWORKS.includes(raw.framework as ReviewState['framework'])
    ? (raw.framework as ReviewState['framework'])
    : base.framework;

  const frameworkValues: Record<string, string> = {};
  if (isRecord(raw.frameworkValues)) {
    for (const [key, value] of Object.entries(raw.frameworkValues)) {
      if (typeof value === 'string') frameworkValues[key] = value;
    }
  }

  const decisions: Record<string, RecordScreening> = {};
  if (isRecord(raw.decisions)) {
    for (const [key, value] of Object.entries(raw.decisions)) {
      if (!isRecord(value)) continue;
      const screening: RecordScreening = { updatedAt: str(value.updatedAt, base.updatedAt) };
      if (value.ta === 'include' || value.ta === 'exclude' || value.ta === 'maybe') screening.ta = value.ta;
      if (value.ft === 'include' || value.ft === 'exclude' || value.ft === 'not-retrieved') screening.ft = value.ft;
      if (typeof value.taReason === 'string') screening.taReason = value.taReason;
      if (typeof value.ftReason === 'string') screening.ftReason = value.ftReason;
      if (typeof value.note === 'string' && value.note) screening.note = value.note;
      decisions[key] = screening;
    }
  }

  return {
    ...base,
    title: str(raw.title),
    objective: str(raw.objective),
    framework,
    frameworkValues,
    questions: list(raw.questions, hasId).map((q) => ({ id: q.id, text: str((q as { text?: unknown }).text) })),
    concepts: list(raw.concepts, hasId).map((c) => {
      const concept = c as { id: string; label?: unknown; terms?: unknown };
      return {
        id: concept.id,
        label: str(concept.label),
        terms: Array.isArray(concept.terms) ? concept.terms.filter((t): t is string => typeof t === 'string') : [],
      };
    }),
    searchTargets: Array.isArray(raw.searchTargets)
      ? raw.searchTargets.filter((t): t is string => typeof t === 'string')
      : base.searchTargets,
    criteria: list(raw.criteria, hasId)
      .map((c) => c as { id: string; kind?: unknown; text?: unknown })
      .filter((c) => c.kind === 'inclusion' || c.kind === 'exclusion')
      .map((c) => ({ id: c.id, kind: c.kind as Criterion['kind'], text: str(c.text) })),
    decisions,
    createdAt: str(raw.createdAt, base.createdAt),
    updatedAt: str(raw.updatedAt, base.updatedAt),
  };
}

/** Linhas da planilha de decisões — uma por registro da base ativa. */
export function decisionRows(
  records: readonly ScreeningRecord[],
  review: ReviewState,
  labels: { decision: (value: string | undefined) => string },
): Record<string, string | number>[] {
  const criteria = new Map(review.criteria.map((criterion) => [criterion.id, criterion.text]));
  return records.map((record) => {
    const screening = review.decisions[record.key];
    return {
      key: record.key,
      title: record.title,
      authors: record.authors,
      year: record.year ?? '',
      venue: record.venue,
      doi: record.doi,
      database: record.database,
      title_abstract: labels.decision(screening?.ta),
      title_abstract_reason: screening?.taReason ? (criteria.get(screening.taReason) ?? '') : '',
      full_text: labels.decision(screening?.ft),
      full_text_reason: screening?.ftReason ? (criteria.get(screening.ftReason) ?? '') : '',
      note: screening?.note ?? '',
    };
  });
}
