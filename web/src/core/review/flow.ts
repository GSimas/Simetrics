import { FIELD } from '@/lib/schema';
import type { Dataset } from '@/lib/types';
import { isNullLike } from '../text';
import type { ScreeningRecord } from './records';
import type { Criterion, RecordScreening } from './types';

/**
 * Contagens do fluxo PRISMA 2020 derivadas do projeto: identificação vem dos arquivos
 * importados (base original, por base de dados), a remoção de duplicatas da deduplicação
 * do Simetrics, e a triagem das decisões registradas.
 */

export interface ReasonCount {
  id: string;
  label: string;
  count: number;
}

export interface ReviewFlow {
  identifiedBySource: { name: string; count: number }[];
  identified: number;
  duplicatesRemoved: number;
  screened: number;
  titleAbstract: { pending: number; include: number; maybe: number; exclude: number };
  /** Registros que seguem ao texto completo: incluídos e "talvez" da primeira etapa. */
  fullText: { eligible: number; pending: number; include: number; exclude: number; notRetrieved: number };
  fullTextExclusions: ReasonCount[];
  included: number;
}

export const NO_REASON_ID = '__none__';

export function advancesToFullText(screening: RecordScreening | undefined): boolean {
  return screening?.ta === 'include' || screening?.ta === 'maybe';
}

export function computeReviewFlow(
  original: Dataset,
  records: readonly ScreeningRecord[],
  decisions: Record<string, RecordScreening>,
  criteria: readonly Criterion[],
  noReasonLabel: string,
): ReviewFlow {
  const bySource = new Map<string, number>();
  for (const doc of original) {
    const raw = doc[FIELD.DATABASE];
    const name = isNullLike(raw) ? '—' : String(raw);
    bySource.set(name, (bySource.get(name) ?? 0) + 1);
  }

  const titleAbstract = { pending: 0, include: 0, maybe: 0, exclude: 0 };
  const fullText = { eligible: 0, pending: 0, include: 0, exclude: 0, notRetrieved: 0 };
  const reasons = new Map<string, number>();

  for (const record of records) {
    const screening = decisions[record.key];
    if (!screening?.ta) {
      titleAbstract.pending += 1;
      continue;
    }
    titleAbstract[screening.ta] += 1;
    if (!advancesToFullText(screening)) continue;

    fullText.eligible += 1;
    if (!screening.ft) fullText.pending += 1;
    else if (screening.ft === 'include') fullText.include += 1;
    else if (screening.ft === 'not-retrieved') fullText.notRetrieved += 1;
    else {
      fullText.exclude += 1;
      const reason = screening.ftReason ?? NO_REASON_ID;
      reasons.set(reason, (reasons.get(reason) ?? 0) + 1);
    }
  }

  const labels = new Map(criteria.map((criterion) => [criterion.id, criterion.text]));
  const fullTextExclusions = [...reasons.entries()]
    .map(([id, count]) => ({ id, label: labels.get(id) ?? noReasonLabel, count }))
    .sort((a, b) => b.count - a.count);

  const identified = original.length;
  return {
    identifiedBySource: [...bySource.entries()]
      .map(([name, count]) => ({ name, count }))
      .sort((a, b) => b.count - a.count),
    identified,
    duplicatesRemoved: Math.max(0, identified - records.length),
    screened: records.length,
    titleAbstract,
    fullText,
    fullTextExclusions,
    included: fullText.include,
  };
}

/** A triagem acabou quando não há registro pendente em nenhuma das duas etapas. */
export function isScreeningComplete(flow: ReviewFlow): boolean {
  return flow.screened > 0 && flow.titleAbstract.pending === 0 && flow.fullText.pending === 0;
}
