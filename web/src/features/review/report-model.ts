import { extractionTarget, FT_EXCLUSION, pilotMetrics, qualityTarget, type PilotMetrics } from '@/core/review/evidence';
import { finalSelection, includedStudies } from '@/core/review/quality';
import type { ScreeningRecord } from '@/core/review/records';
import { formatExtractionValue, summarizeField } from '@/core/review/synthesis';
import type { EvidenceTargetKey, ReviewState, TargetEvidence } from '@/core/review/types';
import type { ReviewCopy } from './copy';
import { EVIDENCE_COPY } from './evidence/copy';
import { fillText, reference, REPORT_TEXT, shortLabel } from './report-shared';

/**
 * O conteúdo das seções do relatório que vão além do protocolo e do PRISMA — formulário de
 * extração, exclusões, síntese, uso de IA e evidências por estudo —, montado uma vez e
 * desenhado igual no PDF e no Word. Puro: sem DOM, testável.
 */

export interface ReportQuote {
  page: number;
  /** Trecho citado; numa marcação de área, o rótulo da área. */
  text: string;
  /** "Marcado por você" / "Citado pela IA · Achado no PDF". */
  source: string;
}

export interface ReportAnswer {
  group: 'extraction' | 'quality' | 'exclusion';
  question: string;
  answer: string;
  /** Status da conferência; vazio quando a resposta nunca teve evidência nem proposta. */
  status: string;
  statusKind: TargetEvidence['status'] | null;
  /** "A IA sugeriu X (modelo)", quando houve proposta. */
  suggestion: string;
  quotes: ReportQuote[];
}

export interface StudyEvidenceBlock {
  key: string;
  label: string;
  reference: string;
  /** "arquivo.pdf · 12 p." ou vazio. */
  pdf: string;
  note: string;
  answers: ReportAnswer[];
}

export interface ReportModel {
  extractionForm: string[][];
  taReasons: { label: string; count: number }[];
  fullTextExcluded: { label: string; reason: string; quotes: ReportQuote[] }[];
  synthesis: { field: string; filled: string; summary: string }[];
  byYear: { year: string; count: number }[];
  ai: PilotMetrics & { models: string[]; studiesWithPdf: number };
  evidence: StudyEvidenceBlock[];
}

function quotesOf(entry: TargetEvidence | undefined, locale: 'pt' | 'en'): ReportQuote[] {
  const copy = EVIDENCE_COPY[locale];
  return (entry?.evidence ?? []).map((item) => ({
    page: item.page,
    text: item.quote || copy.areaLabel,
    source: item.origin === 'ai' ? `${copy.origin.ai} · ${copy.location[item.location]}` : copy.origin.manual,
  }));
}

export function buildReportModel(review: ReviewState, records: readonly ScreeningRecord[], copy: ReviewCopy, locale: 'pt' | 'en'): ReportModel {
  const evidenceCopy = EVIDENCE_COPY[locale];
  const labels = { yes: copy.yes, no: copy.no };
  const nf = locale === 'en' ? 'en' : 'pt-BR';
  const fmt = (value: number) => value.toLocaleString(nf, { maximumFractionDigits: 2 });
  const criteria = new Map(review.criteria.map((criterion) => [criterion.id, criterion.text.trim() || '—']));
  const answerLabels = new Map(review.qualityAnswers.map((answer) => [answer.id, answer.label]));
  const included = includedStudies(records, review);
  const selected = finalSelection(records, review);
  const includedKeys = new Set(included.map((study) => study.key));
  const selectedKeys = new Set(selected.map((study) => study.key));

  // --- Formulário de extração (protocolo) ---
  const extractionForm = review.extractionFields.map((field) => [
    field.label || '—',
    copy.fieldTypes[field.type] ?? field.type,
    field.options.join('; '),
  ]);

  // --- Exclusões ---
  const taCounts = new Map<string, number>();
  for (const screening of Object.values(review.decisions)) {
    if (screening.ta !== 'exclude') continue;
    const label = screening.taReason ? (criteria.get(screening.taReason) ?? copy.noReason) : copy.noReason;
    taCounts.set(label, (taCounts.get(label) ?? 0) + 1);
  }
  const taReasons = [...taCounts.entries()].map(([label, count]) => ({ label, count })).sort((a, b) => b.count - a.count);

  const fullTextExcluded = records
    .filter((record) => review.decisions[record.key]?.ft === 'exclude')
    .map((record) => {
      const reason = review.decisions[record.key]?.ftReason;
      return {
        label: `${shortLabel(record)} — ${record.title}`,
        reason: reason ? (criteria.get(reason) ?? copy.noReason) : copy.noReason,
        quotes: quotesOf(review.evidence[record.key]?.[FT_EXCLUSION], locale),
      };
    });

  // --- Síntese ---
  const synthesis = review.extractionFields.map((field) => {
    const summary = summarizeField(field, selected, review, labels);
    const filled = copy.filledIn.replace('{n}', fmt(summary.filled)).replace('{total}', fmt(selected.length));
    const text =
      summary.kind === 'counts'
        ? summary.counts.filter((item) => item.count > 0).map((item) => `${item.label}: ${fmt(item.count)}`).join(' · ') || '—'
        : summary.kind === 'numeric'
          ? copy.numericSummary.replace('{mean}', fmt(summary.mean)).replace('{min}', fmt(summary.min)).replace('{max}', fmt(summary.max))
          : '—';
    return { field: field.label || '—', filled, summary: text };
  });
  const years = new Map<string, number>();
  for (const study of selected) {
    const year = study.year ? String(study.year) : 's.d.';
    years.set(year, (years.get(year) ?? 0) + 1);
  }
  const byYear = [...years.entries()].map(([year, count]) => ({ year, count })).sort((a, b) => a.year.localeCompare(b.year));

  // --- Texto completo e IA ---
  const pdfKeys = records.filter((record) => review.documents[record.key] || review.evidence[record.key]).map((record) => record.key);
  const models = new Set<string>();
  for (const targets of Object.values(review.evidence)) {
    for (const entry of Object.values(targets)) if (entry?.suggestion?.model) models.add(entry.suggestion.model);
  }
  const ai = {
    ...pilotMetrics(review, pdfKeys),
    models: [...models].sort(),
    studiesWithPdf: records.filter((record) => review.documents[record.key]).length,
  };

  // --- Evidências por estudo ---
  const answerFor = (key: string, target: EvidenceTargetKey, group: ReportAnswer['group'], question: string, answer: string): ReportAnswer => {
    const entry = review.evidence[key]?.[target];
    const suggestion = entry?.suggestion;
    const suggested = suggestion
      ? `${evidenceCopy.aiSuggests}: ${
          target.startsWith('quality:') ? (answerLabels.get(String(suggestion.value)) ?? String(suggestion.value)) : formatExtractionValue(suggestion.value, labels)
        }${suggestion.model ? ` (${suggestion.model})` : ''}${suggestion.rationale ? ` — ${suggestion.rationale}` : ''}`
      : '';
    return {
      group,
      question,
      answer: answer || '—',
      status: entry ? evidenceCopy.status[entry.status] : '',
      statusKind: entry?.status ?? null,
      suggestion: suggested,
      quotes: quotesOf(entry, locale),
    };
  };

  const evidence: StudyEvidenceBlock[] = [];
  for (const record of records) {
    const document = review.documents[record.key];
    const targets = review.evidence[record.key];
    if (!document && !targets) continue;
    const answers: ReportAnswer[] = [];
    if (selectedKeys.has(record.key)) {
      for (const field of review.extractionFields) {
        const value = formatExtractionValue(review.extraction[record.key]?.values[field.id], labels);
        answers.push(answerFor(record.key, extractionTarget(field.id), 'extraction', field.label || '—', value));
      }
    }
    if (includedKeys.has(record.key)) {
      for (const question of review.qualityQuestions) {
        const value = answerLabels.get(review.quality[record.key]?.[question.id] ?? '') ?? '';
        answers.push(answerFor(record.key, qualityTarget(question.id), 'quality', question.text.trim() || '—', value));
      }
    }
    const screening = review.decisions[record.key];
    if (screening?.ft === 'exclude' || targets?.[FT_EXCLUSION]) {
      const reason = screening?.ftReason ? (criteria.get(screening.ftReason) ?? '') : '';
      answers.push(answerFor(record.key, FT_EXCLUSION, 'exclusion', evidenceCopy.exclusionTarget, reason));
    }
    // Um estudo sem pergunta a mostrar (ainda não incluído, sem exclusão) só tem o PDF: fica de fora.
    if (answers.length === 0) continue;
    evidence.push({
      key: record.key,
      label: shortLabel(record),
      reference: reference(record),
      pdf: document ? `${document.name} · ${evidenceCopy.pages.replace('{n}', String(document.pages))}` : '',
      note: screening?.note?.trim() ?? '',
      answers,
    });
  }

  return { extractionForm, taReasons, fullTextExcluded, synthesis, byYear, ai, evidence };
}

/** Trechos de uma resposta numa célula de tabela: um por linha, com página e origem. */
export function quotesCell(quotes: readonly ReportQuote[], pageLabel: string, empty: string): string {
  if (quotes.length === 0) return empty;
  return quotes.map((quote) => `${pageLabel} ${quote.page}: “${quote.text}” (${quote.source})`).join('\n');
}

/** Os parágrafos de método da seção de textos completos e IA (PRISMA 2020, item 9: automação na coleta). */
export function fullTextParagraphs(model: ReportModel, text: (typeof REPORT_TEXT)['pt'], locale: 'pt' | 'en'): string[] {
  const nf = locale === 'en' ? 'en' : 'pt-BR';
  const n = (value: number) => value.toLocaleString(nf);
  const { ai } = model;
  const paragraphs = [
    fillText(text.fullTextMethod, { pdfs: n(ai.studiesWithPdf), scanned: n(ai.scanned), answers: n(ai.answersWithEvidence) }),
  ];
  paragraphs.push(
    ai.suggested === 0
      ? text.aiNotUsed
      : fillText(text.aiUsed, {
          models: ai.models.join(', ') || '—',
          suggested: n(ai.suggested),
          located: n(ai.exact + ai.approximate),
          quotes: n(ai.aiQuotes),
          notFound: n(ai.notFound),
          confirmed: n(ai.confirmed),
          edited: n(ai.edited),
          rejected: n(ai.rejected),
          pending: ai.pending > 0 ? fillText(text.aiPending, { n: n(ai.pending) }) : '',
        }),
  );
  return paragraphs;
}
