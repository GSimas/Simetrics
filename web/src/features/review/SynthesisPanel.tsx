import { useMemo } from 'react';
import { FileSpreadsheet } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { downloadCsv, timestampedFilename, toCsv } from '@/core/export';
import { finalSelection, hasQualityChecklist, includedStudies, scoreStudy } from '@/core/review/quality';
import type { ScreeningRecord } from '@/core/review/records';
import { extractionRows, formatExtractionValue, qualityRows, summarizeField, type FieldSummary } from '@/core/review/synthesis';
import type { ReviewState } from '@/core/review/types';
import { numberLocale } from '@/lib/i18n/labels';
import { useLocale } from '@/state/locale.store';
import { useScreeningRecords } from '@/state/review.store';
import type { ReviewCopy } from './copy';
import { Block } from './parts';
import { ScoreBadge } from './QualityPanel';
import { EmptyStep } from './StudyWorkspace';

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-border/80 px-3 py-2">
      <p className="text-[10.5px] font-medium uppercase tracking-[0.08em] text-muted-foreground">{label}</p>
      <p className="text-lg font-bold tabular-nums">{value}</p>
    </div>
  );
}

/** Lista com barras finas — o mesmo desenho dos rankings da aba Informações Principais. */
function BarList({ items, nf }: { items: { label: string; count: number }[]; nf: string }) {
  const max = Math.max(0, ...items.map((item) => item.count));
  return (
    <ol className="space-y-1.5">
      {items.map((item) => (
        <li key={item.label} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-2">
          <span className="min-w-0">
            <span className="block truncate text-xs">{item.label}</span>
            <span className="mt-1 block h-1.5 w-full overflow-hidden rounded-full bg-muted">
              <span
                className="block h-full rounded-full bg-highlight"
                style={{ width: `${max > 0 ? (item.count / max) * 100 : 0}%` }}
              />
            </span>
          </span>
          <span className="text-xs font-semibold tabular-nums">{item.count.toLocaleString(nf)}</span>
        </li>
      ))}
    </ol>
  );
}

function SummaryBody({ summary, total, copy, nf }: { summary: FieldSummary; total: number; copy: ReviewCopy; nf: string }) {
  const filled = copy.filledIn.replace('{n}', summary.filled.toLocaleString(nf)).replace('{total}', total.toLocaleString(nf));
  const fmt = (value: number) => value.toLocaleString(nf, { maximumFractionDigits: 2 });
  return (
    <div className="space-y-2">
      <p className="text-[11px] text-muted-foreground">{filled}</p>
      {summary.kind === 'counts' && <BarList items={summary.counts} nf={nf} />}
      {summary.kind === 'numeric' && (
        <p className="text-sm tabular-nums">
          {copy.numericSummary
            .replace('{mean}', fmt(summary.mean))
            .replace('{min}', fmt(summary.min))
            .replace('{max}', fmt(summary.max))}
        </p>
      )}
    </div>
  );
}

function studyLabel(study: ScreeningRecord): string {
  const firstAuthor = study.authors.split(';')[0]?.trim();
  return [firstAuthor, study.year].filter(Boolean).join(', ') || study.title;
}

export function SynthesisPanel({ review, copy }: { review: ReviewState; copy: ReviewCopy }) {
  const locale = useLocale((state) => state.locale);
  const nf = numberLocale(locale);
  const records = useScreeningRecords();
  const labels = { yes: copy.yes, no: copy.no };

  const included = useMemo(() => includedStudies(records, review), [records, review]);
  const studies = useMemo(() => finalSelection(records, review), [records, review]);
  const withChecklist = hasQualityChecklist(review);

  const quality = useMemo(() => {
    const scores = included.map((study) => scoreStudy(review, study.key)).filter((score) => score.complete);
    return {
      assessed: scores.length,
      mean: scores.length ? scores.reduce((sum, score) => sum + score.score, 0) / scores.length : null,
      passing: scores.filter((score) => score.passes === true).length,
    };
  }, [included, review]);

  const byYear = useMemo(() => {
    const counts = new Map<string, number>();
    for (const study of studies) {
      const year = study.year === null ? '—' : String(study.year);
      counts.set(year, (counts.get(year) ?? 0) + 1);
    }
    return [...counts.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([label, count]) => ({ label, count }));
  }, [studies]);

  if (studies.length === 0) return <EmptyStep message={copy.noIncluded} />;

  const extractedCount = studies.filter((study) => review.extraction[study.key]?.done).length;
  const baseName = review.title.trim() || 'revisao';
  const exportCsv = (rows: Record<string, string | number>[], suffix: string): void =>
    downloadCsv(timestampedFilename(`${baseName}-${suffix}`, 'csv'), toCsv(rows));

  return (
    <div className="space-y-4">
      <p className="max-w-3xl text-sm text-muted-foreground">{copy.synthesisIntro}</p>

      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">
        <Stat label={copy.finalSelection} value={studies.length.toLocaleString(nf)} />
        {withChecklist && (
          <>
            <Stat label={copy.assessed} value={`${quality.assessed.toLocaleString(nf)} / ${included.length.toLocaleString(nf)}`} />
            <Stat
              label={copy.meanScore}
              value={quality.mean === null ? '—' : quality.mean.toLocaleString(nf, { maximumFractionDigits: 2 })}
            />
            {review.qualityCutoff !== null && <Stat label={copy.passing} value={quality.passing.toLocaleString(nf)} />}
          </>
        )}
        {review.extractionFields.length > 0 && (
          <Stat label={copy.extracted} value={`${extractedCount.toLocaleString(nf)} / ${studies.length.toLocaleString(nf)}`} />
        )}
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Block title={copy.byYear}>
          <BarList items={byYear} nf={nf} />
        </Block>
        {review.extractionFields.length > 0 && (
          <Block title={copy.fieldsSummary}>
            <div className="space-y-4">
              {review.extractionFields.map((field) => (
                <div key={field.id} className="space-y-1">
                  <p className="text-xs font-semibold">{field.label || '—'}</p>
                  <SummaryBody summary={summarizeField(field, studies, review, labels)} total={studies.length} copy={copy} nf={nf} />
                </div>
              ))}
            </div>
          </Block>
        )}
      </div>

      {review.extractionFields.length > 0 && (
        <Block title={copy.characteristicsTable}>
          <div className="flex justify-end">
            <Button type="button" variant="outline" size="sm" onClick={() => exportCsv(extractionRows(studies, review, labels), 'caracteristicas')}>
              <FileSpreadsheet aria-hidden />
              {copy.exportCsv}
            </Button>
          </div>
          <div className="max-h-[60vh] overflow-auto border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>{copy.study}</TableHead>
                  {review.extractionFields.map((field) => (
                    <TableHead key={field.id}>{field.label || '—'}</TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {studies.map((study) => (
                  <TableRow key={study.key}>
                    <TableCell className="min-w-[12rem] font-medium" title={study.title}>
                      {studyLabel(study)}
                    </TableCell>
                    {review.extractionFields.map((field) => (
                      <TableCell key={field.id} className="min-w-[8rem] text-xs">
                        {formatExtractionValue(review.extraction[study.key]?.values[field.id], labels) || '—'}
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </Block>
      )}

      {withChecklist && (
        <Block title={copy.qualityTable}>
          <div className="flex justify-end">
            <Button type="button" variant="outline" size="sm" onClick={() => exportCsv(qualityRows(included, review), 'qualidade')}>
              <FileSpreadsheet aria-hidden />
              {copy.exportCsv}
            </Button>
          </div>
          <div className="max-h-[60vh] overflow-auto border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>{copy.study}</TableHead>
                  {review.qualityQuestions.map((question, index) => (
                    <TableHead key={question.id} title={question.text}>
                      Q{index + 1}
                    </TableHead>
                  ))}
                  <TableHead className="text-right">{copy.score}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {included.map((study) => {
                  const responses = review.quality[study.key] ?? {};
                  return (
                    <TableRow key={study.key}>
                      <TableCell className="min-w-[12rem] font-medium" title={study.title}>
                        {studyLabel(study)}
                      </TableCell>
                      {review.qualityQuestions.map((question) => (
                        <TableCell key={question.id} className="text-xs">
                          {review.qualityAnswers.find((answer) => answer.id === responses[question.id])?.label ?? '—'}
                        </TableCell>
                      ))}
                      <TableCell className="text-right">
                        <ScoreBadge review={review} studyKey={study.key} copy={copy} />
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
        </Block>
      )}
    </div>
  );
}
