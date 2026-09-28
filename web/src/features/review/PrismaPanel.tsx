import { useMemo, useState } from 'react';
import { AlertTriangle, ArrowDown, Download, ExternalLink, FileSpreadsheet, FileText, Loader2 } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { downloadBlob, downloadCsv, timestampedFilename, toCsv } from '@/core/export';
import { computeReviewFlow, isScreeningComplete, type ReviewFlow } from '@/core/review/flow';
import { PRISMALAB_URL, toPrismaLabProject } from '@/core/review/prismalab';
import { finalSelection, isExcludedByQuality } from '@/core/review/quality';
import { decisionRows } from '@/core/review/state';
import type { ReviewState } from '@/core/review/types';
import { numberLocale } from '@/lib/i18n/labels';
import { useDataset } from '@/state/dataset.store';
import { useLocale } from '@/state/locale.store';
import { useProjectStore } from '@/state/project.store';
import { useScreeningRecords } from '@/state/review.store';
import type { ReviewCopy } from './copy';

function FlowBox({ label, value, tone = 'main', children }: {
  label: string;
  value: number;
  tone?: 'main' | 'side';
  children?: React.ReactNode;
}) {
  const locale = useLocale((state) => state.locale);
  return (
    <div className={tone === 'main' ? 'rounded-lg border border-border p-3' : 'rounded-lg border border-dashed border-border/80 p-3'}>
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="text-xl font-bold tabular-nums">n = {value.toLocaleString(numberLocale(locale))}</p>
      {children}
    </div>
  );
}

function Row({ phase, main, side }: { phase: string; main: React.ReactNode; side?: React.ReactNode }) {
  return (
    <div className="grid gap-2 sm:grid-cols-[7rem_minmax(0,1fr)_minmax(0,1fr)] sm:items-stretch">
      <p className="eyebrow self-center">{phase}</p>
      {main}
      {side ?? <div className="hidden sm:block" />}
    </div>
  );
}

function FlowSummary({ flow, copy }: { flow: ReviewFlow; copy: ReviewCopy }) {
  const locale = useLocale((state) => state.locale);
  const nf = numberLocale(locale);
  const arrow = (
    <div className="grid sm:grid-cols-[7rem_minmax(0,1fr)_minmax(0,1fr)]">
      <span />
      <ArrowDown className="mx-auto size-4 text-muted-foreground" aria-hidden />
    </div>
  );

  return (
    <div className="space-y-1.5">
      <Row
        phase={copy.identification}
        main={
          <FlowBox label={copy.identifiedFrom} value={flow.identified}>
            <ul className="mt-1 space-y-0.5 text-[11px] text-muted-foreground">
              {flow.identifiedBySource.map((source) => (
                <li key={source.name}>
                  {source.name}: {source.count.toLocaleString(nf)}
                </li>
              ))}
            </ul>
          </FlowBox>
        }
        side={<FlowBox label={copy.duplicatesRemoved} value={flow.duplicatesRemoved} tone="side" />}
      />
      {arrow}
      <Row
        phase={copy.screening}
        main={<FlowBox label={copy.recordsScreened} value={flow.screened} />}
        side={<FlowBox label={copy.recordsExcluded} value={flow.titleAbstract.exclude} tone="side" />}
      />
      {arrow}
      <Row
        phase=""
        main={<FlowBox label={copy.reportsSought} value={flow.fullText.eligible} />}
        side={<FlowBox label={copy.reportsNotRetrieved} value={flow.fullText.notRetrieved} tone="side" />}
      />
      {arrow}
      <Row
        phase=""
        main={<FlowBox label={copy.reportsAssessed} value={flow.fullText.eligible - flow.fullText.notRetrieved} />}
        side={
          <FlowBox label={copy.reportsExcluded} value={flow.fullText.exclude} tone="side">
            <ul className="mt-1 space-y-0.5 text-[11px] text-muted-foreground">
              {flow.fullTextExclusions.map((reason) => (
                <li key={reason.id}>
                  {reason.label}: {reason.count.toLocaleString(nf)}
                </li>
              ))}
            </ul>
          </FlowBox>
        }
      />
      {arrow}
      <Row phase={copy.included} main={<FlowBox label={copy.studiesIncluded} value={flow.included} />} />
    </div>
  );
}

export function PrismaPanel({ review, copy }: { review: ReviewState; copy: ReviewCopy }) {
  const locale = useLocale((state) => state.locale);
  const original = useDataset((state) => state.original);
  const projectName = useProjectStore(
    (state) => state.projects.find((project) => project.id === state.activeProjectId)?.name ?? '',
  );
  const records = useScreeningRecords();

  const flow = useMemo(
    () =>
      computeReviewFlow(original ?? [], records, review.decisions, review.criteria, copy.noReason, {
        isExcluded: (key) => isExcludedByQuality(review, key),
        label: copy.qualityReason,
      }),
    [original, records, review, copy.noReason, copy.qualityReason],
  );
  const [generating, setGenerating] = useState(false);
  const complete = isScreeningComplete(flow);
  const baseName = review.title.trim() || projectName || 'revisao';

  const exportPrismaLab = (): void => {
    const project = toPrismaLabProject({ title: review.title || projectName, reviewType: review.type, flow, locale });
    downloadBlob(
      timestampedFilename(`${baseName}-prismalab`, 'json'),
      new Blob([JSON.stringify(project, null, 2)], { type: 'application/json' }),
    );
  };

  const exportDecisions = (onlyIncluded: boolean): void => {
    const rows = decisionRows(records, review, {
      decision: (value) => (value ? (copy.decisionLabels[value] ?? value) : ''),
    });
    const finalKeys = new Set(finalSelection(records, review).map((record) => record.key));
    const selected = onlyIncluded ? rows.filter((_, index) => finalKeys.has(records[index]!.key)) : rows;
    downloadCsv(timestampedFilename(`${baseName}-${onlyIncluded ? 'incluidos' : 'triagem'}`, 'csv'), toCsv(selected));
  };

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
      <section className="space-y-4 rounded-xl border border-border/80 p-4">
        <h3 className="text-sm font-semibold">{copy.flowTitle}</h3>
        {!complete && flow.screened > 0 && (
          <p className="flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs text-amber-900 dark:border-amber-900/60 dark:bg-amber-950/50 dark:text-amber-200">
            <AlertTriangle className="mt-0.5 size-4 shrink-0" aria-hidden />
            {copy.pendingWarning
              .replace('{ta}', flow.titleAbstract.pending.toLocaleString(numberLocale(locale)))
              .replace('{ft}', flow.fullText.pending.toLocaleString(numberLocale(locale)))}
          </p>
        )}
        <FlowSummary flow={flow} copy={copy} />
        <p className="text-[11px] leading-relaxed text-muted-foreground">{copy.dedupHint}</p>
      </section>

      <section className="space-y-4 rounded-xl border border-border/80 p-4">
        <h3 className="text-sm font-semibold">{copy.exportTitle}</h3>
        <div className="space-y-2">
          <div className="flex flex-wrap gap-2">
            <Button type="button" onClick={exportPrismaLab} disabled={flow.screened === 0}>
              <Download aria-hidden />
              {copy.exportPrisma}
            </Button>
            <Button type="button" variant="outline" asChild>
              <a href={PRISMALAB_URL} target="_blank" rel="noopener noreferrer">
                {copy.openPrismaLab}
                <ExternalLink aria-hidden />
              </a>
            </Button>
          </div>
          <p className="text-xs leading-relaxed text-muted-foreground">{copy.exportPrismaHint}</p>
        </div>
        <div className="space-y-2 border-t border-border/80 pt-4">
          <div className="flex flex-wrap gap-2">
            <Button type="button" variant="outline" onClick={() => exportDecisions(false)} disabled={records.length === 0}>
              <FileSpreadsheet aria-hidden />
              {copy.exportDecisions}
            </Button>
            <Button type="button" variant="outline" onClick={() => exportDecisions(true)} disabled={flow.included === 0}>
              <FileSpreadsheet aria-hidden />
              {copy.exportIncluded}
            </Button>
          </div>
          <p className="text-xs leading-relaxed text-muted-foreground">{copy.exportDecisionsHint}</p>
        </div>
        <div className="space-y-2 border-t border-border/80 pt-4">
          <h4 className="text-sm font-semibold">{copy.reportTitle}</h4>
          <Button
            type="button"
            variant="outline"
            disabled={generating || flow.screened === 0}
            onClick={() => {
              setGenerating(true);
              // O gerador (e a biblioteca docx) só baixam quando alguém pede o relatório.
              void import('./report-docx')
                .then(({ downloadReviewReport }) => downloadReviewReport({ review, flow, records, copy, locale }))
                .finally(() => setGenerating(false));
            }}
          >
            {generating ? <Loader2 className="animate-spin" aria-hidden /> : <FileText aria-hidden />}
            {generating ? copy.generating : copy.downloadReport}
          </Button>
          <p className="text-xs leading-relaxed text-muted-foreground">{copy.reportHint}</p>
        </div>
      </section>
    </div>
  );
}
