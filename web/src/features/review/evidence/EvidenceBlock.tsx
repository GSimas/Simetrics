import { useEffect, useMemo, useRef, useState } from 'react';
import { Download, FilesIcon, Loader2 } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { downloadCsv, toCsv } from '@/core/export';
import { evidenceRows, matchPdfToStudy, pilotMetrics } from '@/core/review/evidence';
import type { ScreeningRecord } from '@/core/review/records';
import type { ReviewState } from '@/core/review/types';
import { ingestPdf, type IngestedPdf } from '@/lib/pdf';
import { storageEstimate } from '@/lib/pdf-store';
import { cn } from '@/lib/utils';
import { useLocale } from '@/state/locale.store';
import { useReview, useReviewReadOnly } from '@/state/review.store';
import { REVIEW_COPY } from '../copy';
import { Block } from '../parts';
import { fill, useEvidenceCopy } from './copy';
import { answerText, suggestionText, targetLabel } from './format';

function Metric({ label, value, tone }: { label: string; value: string; tone?: 'include' | 'exclude' | 'warning' | 'blue' | undefined }) {
  return (
    <div className="rounded-lg border border-border/80 px-3 py-2">
      <p
        className={cn(
          'text-lg font-medium tabular-nums leading-tight',
          tone === 'include' && 'text-include',
          tone === 'exclude' && 'text-exclude',
          tone === 'warning' && 'text-amber-600 dark:text-amber-300',
          tone === 'blue' && 'text-blue-600 dark:text-blue-300',
        )}
      >
        {value}
      </p>
      <p className="text-[11px] leading-snug text-muted-foreground">{label}</p>
    </div>
  );
}

interface Unmatched {
  file: File;
  ingested: IngestedPdf;
}

/**
 * Textos completos e evidências da etapa: envio de PDFs em lote (ligados aos estudos pelo
 * DOI ou pelo título), os números do piloto e a planilha das evidências.
 */
export function EvidenceBlock({ review, studies }: { review: ReviewState; studies: ScreeningRecord[] }) {
  const copy = useEvidenceCopy();
  const locale = useLocale((state) => state.locale);
  const reviewCopy = REVIEW_COPY[locale === 'en' ? 'en' : 'pt'];
  const readOnly = useReviewReadOnly();
  const attachDocument = useReview((state) => state.attachDocument);
  const input = useRef<HTMLInputElement | null>(null);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [unmatched, setUnmatched] = useState<Unmatched[]>([]);
  const [usage, setUsage] = useState<number | null>(null);

  const metrics = useMemo(() => pilotMetrics(review, studies.map((study) => study.key)), [review, studies]);
  const documentCount = Object.keys(review.documents).length;

  useEffect(() => {
    void storageEstimate().then((estimate) => setUsage(estimate?.usage ?? null));
  }, [documentCount]);

  const upload = async (files: File[]): Promise<void> => {
    const pdfs = files.filter((file) => file.type === 'application/pdf' || /\.pdf$/i.test(file.name));
    if (pdfs.length === 0) return;
    setResult(null);
    setProgress({ done: 0, total: pdfs.length });
    let matched = 0;
    const leftover: Unmatched[] = [];
    // Estudos ainda sem PDF primeiro: um arquivo não toma o lugar do PDF já anexado de outro.
    const pool = [...studies].sort((a, b) => Number(!!review.documents[a.key]) - Number(!!review.documents[b.key]));
    for (const [index, file] of pdfs.entries()) {
      try {
        const ingested = await ingestPdf(file);
        const study = matchPdfToStudy(pool, ingested);
        if (study) {
          attachDocument(study.key, ingested.document);
          matched += 1;
        } else leftover.push({ file, ingested });
      } catch {
        // Arquivo ilegível: não entra em nenhum estudo e fica fora da contagem de ligados.
      }
      setProgress({ done: index + 1, total: pdfs.length });
    }
    setProgress(null);
    setUnmatched(leftover);
    setResult(fill(copy.bulkResult, { matched, total: pdfs.length }));
  };

  const exportCsv = (): void => {
    const words = { yes: reviewCopy.yes, no: reviewCopy.no };
    const rows = evidenceRows(review, studies, {
      question: (target) => targetLabel(review, target, copy.exclusionTarget),
      answer: (key, target) => answerText(review, key, target, words),
      suggestion: (target, value) => suggestionText(review, target, value, words),
      status: (status) => copy.status[status],
      origin: (origin) => copy.origin[origin],
      location: (location) => copy.location[location],
    });
    downloadCsv('simetrics-evidencias.csv', toCsv(rows));
  };

  const fmt = (value: number) => value.toLocaleString(locale === 'en' ? 'en' : 'pt-BR');
  const located = metrics.exact + metrics.approximate;

  return (
    <Block title={copy.blockTitle} hint={copy.blockHint} tour="review-evidence">
      <div className="flex flex-wrap items-center gap-2">
        {!readOnly && (
          <>
            <input
              ref={input}
              type="file"
              accept="application/pdf,.pdf"
              multiple
              className="hidden"
              onChange={(event) => {
                void upload([...(event.target.files ?? [])]);
                event.target.value = '';
              }}
            />
            <Button type="button" variant="outline" size="sm" disabled={!!progress} className="active:scale-[0.97]" onClick={() => input.current?.click()}>
              {progress ? <Loader2 className="animate-spin" aria-hidden /> : <FilesIcon aria-hidden />}
              {progress ? fill(copy.bulkWorking, { done: progress.done, total: progress.total }) : copy.bulk}
            </Button>
          </>
        )}
        <Button type="button" variant="ghost" size="sm" className="active:scale-[0.97]" onClick={exportCsv} disabled={Object.keys(review.evidence).length === 0}>
          <Download aria-hidden />
          {copy.exportCsv}
        </Button>
        {usage !== null && documentCount > 0 && (
          <span className="ml-auto text-[11px] text-muted-foreground">{fill(copy.storage, { used: `${(usage / 1_048_576).toFixed(1)} MB` })}</span>
        )}
      </div>
      {result && <p className="text-xs animate-in fade-in-0 duration-200" role="status">{result}</p>}
      {unmatched.length > 0 && (
        <div className="space-y-1.5 rounded-lg border border-amber-500/40 bg-amber-500/5 p-3 text-xs">
          <p>{copy.bulkUnmatched}</p>
          {unmatched.map((item) => (
            <label key={item.ingested.document.hash} className="flex flex-wrap items-center gap-2">
              <span className="min-w-0 max-w-[40ch] truncate font-medium" title={item.file.name}>
                {item.file.name}
              </span>
              <select
                className="h-8 min-w-0 max-w-full flex-1 rounded-md border border-input bg-background px-2 text-xs"
                defaultValue=""
                onChange={(event) => {
                  if (!event.target.value) return;
                  attachDocument(event.target.value, item.ingested.document);
                  setUnmatched((list) => list.filter((entry) => entry !== item));
                }}
              >
                <option value="">{copy.bulkPick}</option>
                {studies.map((study) => (
                  <option key={study.key} value={study.key}>
                    {review.documents[study.key] ? '✓ ' : ''}
                    {(study.title || study.key).slice(0, 110)}
                  </option>
                ))}
              </select>
            </label>
          ))}
        </div>
      )}
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 xl:grid-cols-6">
        <Metric label={copy.metrics.withPdf} value={`${fmt(metrics.withPdf)}/${fmt(metrics.studies)}`} />
        <Metric label={copy.metrics.answersWithEvidence} value={fmt(metrics.answersWithEvidence)} />
        <Metric label={copy.metrics.suggested} value={fmt(metrics.suggested)} />
        <Metric label={copy.metrics.confirmed} value={fmt(metrics.confirmed)} tone="include" />
        <Metric label={copy.metrics.edited} value={fmt(metrics.edited)} tone="blue" />
        <Metric label={copy.metrics.rejected} value={fmt(metrics.rejected)} tone="exclude" />
        <Metric label={copy.metrics.pending} value={fmt(metrics.pending)} tone={metrics.pending > 0 ? 'warning' : undefined} />
        <Metric label={copy.metrics.aiQuotes} value={fmt(metrics.aiQuotes)} />
        <Metric label={copy.metrics.located} value={metrics.aiQuotes ? `${Math.round((located / metrics.aiQuotes) * 100)}%` : '—'} tone="include" />
        <Metric label={copy.metrics.notFound} value={fmt(metrics.notFound)} tone={metrics.notFound > 0 ? 'exclude' : undefined} />
        <Metric label={copy.metrics.scanned} value={fmt(metrics.scanned)} tone={metrics.scanned > 0 ? 'warning' : undefined} />
      </div>
      <p className="text-[11px] leading-relaxed text-muted-foreground">{copy.metricsHint}</p>
    </Block>
  );
}
