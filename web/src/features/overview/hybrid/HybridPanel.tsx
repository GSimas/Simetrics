import { useMemo, useState } from 'react';
import { AlertTriangle, Check, Copy, Layers, ListChecks, Loader2, Settings2, X } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { hybridMethodsText } from '@/core/hybrid/report';
import type { HybridRun } from '@/core/hybrid/types';
import { localizeProgress } from '@/lib/i18n/progress';
import { useDataset } from '@/state/dataset.store';
import { useFreeTier } from '@/state/free-tier.store';
import { useHybridConfig } from '@/state/hybrid-config.store';
import { freeDocsLimitError, useHybrid } from '@/state/hybrid.store';
import { useLocale } from '@/state/locale.store';
import { HYBRID_COPY, type HybridCopy } from './copy';
import { HybridSettingsModal } from './HybridSettingsModal';
import { TaxonomyEditor } from './TaxonomyEditor';

function formatPercent(value: number, locale: 'pt' | 'en'): string {
  return `${(value * 100).toLocaleString(locale === 'pt' ? 'pt-BR' : 'en-US', { maximumFractionDigits: 1 })}%`;
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-xl border border-border/80 px-3 py-2">
      <p className="text-[10.5px] font-medium uppercase tracking-[0.08em] text-muted-foreground">{label}</p>
      <p className="text-lg font-bold tabular-nums text-foreground">{value}</p>
      {hint && <p className="text-[11px] text-muted-foreground">{hint}</p>}
    </div>
  );
}

function CopyButton({ text, copy }: { text: string; copy: HybridCopy }) {
  const [copied, setCopied] = useState(false);
  return (
    <Button
      type="button"
      variant="outline"
      size="sm"
      onClick={() => {
        void navigator.clipboard.writeText(text).then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 1800);
        });
      }}
    >
      {copied ? <Check aria-hidden /> : <Copy aria-hidden />}
      {copied ? copy.copied : copy.copy}
    </Button>
  );
}

function RunResult({ run, copy }: { run: HybridRun; copy: HybridCopy }) {
  const locale = useLocale((state) => state.locale);
  const methods = useMemo(() => hybridMethodsText(run, locale), [run, locale]);
  const firstRound = run.validation[0];
  const lastRound = run.validation[run.validation.length - 1];
  const agreementById = new Map((lastRound?.perCategory ?? []).map((item) => [item.categoryId, item]));
  const { coverage } = run;
  const share = (value: number) => (coverage.total > 0 ? value / coverage.total : 0);

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">
        <Stat
          label={copy.agreement}
          value={lastRound && lastRound.compared > 0 ? formatPercent(lastRound.agreement, locale) : '—'}
          {...(lastRound && firstRound && lastRound !== firstRound
            ? { hint: `${formatPercent(firstRound.agreement, locale)} → ${formatPercent(lastRound.agreement, locale)}` }
            : lastRound ? { hint: `n = ${lastRound.compared}` } : { hint: copy.notMeasured })}
        />
        <Stat
          label={copy.kappa}
          value={lastRound && lastRound.compared > 0 ? lastRound.kappa.toFixed(2) : '—'}
        />
        <Stat label={copy.auto} value={coverage.auto.toLocaleString()} hint={formatPercent(share(coverage.auto), locale)} />
        <Stat label={copy.review} value={coverage.review.toLocaleString()} hint={formatPercent(share(coverage.review), locale)} />
        <Stat
          label={copy.unclassified}
          value={coverage.unclassified.toLocaleString()}
          hint={formatPercent(share(coverage.unclassified), locale)}
        />
      </div>

      <div className="overflow-x-auto border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>{copy.category}</TableHead>
              <TableHead className="text-right">{copy.documents}</TableHead>
              <TableHead className="text-right">{copy.share}</TableHead>
              <TableHead className="text-right">{copy.meanConfidence}</TableHead>
              <TableHead className="text-right">{copy.sampleAgreement}</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {run.categoryCounts.map((item) => {
              const agreement = agreementById.get(item.categoryId);
              const category = run.finalTaxonomy.find((entry) => entry.id === item.categoryId);
              return (
                <TableRow key={item.categoryId}>
                  <TableCell className="font-medium text-foreground" title={category?.what}>
                    {item.name}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">{item.documents.toLocaleString()}</TableCell>
                  <TableCell className="text-right tabular-nums">{formatPercent(share(item.documents), locale)}</TableCell>
                  <TableCell className="text-right tabular-nums">{item.meanConfidence.toFixed(2)}</TableCell>
                  <TableCell className="text-right tabular-nums">
                    {agreement ? (
                      <Badge variant={agreement.agreement >= 0.6 ? 'success' : 'warning'}>
                        {formatPercent(agreement.agreement, locale)} · n={agreement.support}
                      </Badge>
                    ) : (
                      '—'
                    )}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </div>

      <div className="space-y-2 rounded-xl border border-border/80 bg-muted/30 p-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className="text-xs font-semibold">{copy.methods}</p>
          <CopyButton text={methods} copy={copy} />
        </div>
        <p className="text-xs leading-relaxed text-muted-foreground">{methods}</p>
      </div>

      <details className="text-xs text-muted-foreground">
        <summary className="cursor-pointer select-none font-medium text-foreground">
          {copy.history} ({run.taxonomyVersions.length}) · {copy.usage}
        </summary>
        <ul className="mt-2 space-y-1">
          {run.taxonomyVersions.map((version) => (
            <li key={version.version}>
              v{version.version} · {copy.origin[version.origin]} · {version.categories.length}{' '}
              {version.categories.length === 1 ? copy.categoryUnit : copy.categoryUnitPlural}
            </li>
          ))}
        </ul>
        <p className="mt-2">
          {run.models.discovery ?? '—'}: {(run.usage.discoveryInputTokens + run.usage.discoveryOutputTokens).toLocaleString()}{' '}
          {copy.tokens} · {run.models.classifier}: {run.usage.classifierRequests.toLocaleString()} {copy.requests},{' '}
          {run.usage.classifierInputTokens.toLocaleString()} {copy.tokens}
        </p>
      </details>
    </div>
  );
}

/**
 * Painel da classificação híbrida: início, progresso, revisão da taxonomia e resultado.
 * Convive com o k-means do ThemePanel — os dois gravam o mesmo campo de tema, e o que
 * rodou por último é o que vale.
 */
export function HybridPanel() {
  const locale = useLocale((state) => state.locale);
  const copy = HYBRID_COPY[locale];
  const config = useHybridConfig((state) => state.config);
  const hybridRun = useDataset((state) => state.hybridRun);
  const kmeansBusy = useDataset((state) => state.isCategorizingThemes);
  const { stage, progress: rawProgress, draft, draftSupport, isManual, error, start, startManual, updateDraft, confirm, cancel, dismiss } =
    useHybrid();
  // As fases do k-means chegam do worker em português; as do fluxo híbrido já vêm no idioma.
  const progress = rawProgress && localizeProgress(rawProgress, locale);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [focus, setFocus] = useState('');

  const freeStatus = useFreeTier((state) => state.status);
  const ownGenerative = Boolean(config.generative.apiKey.trim());
  const ownJev = Boolean(config.jev.apiKey.trim());
  // Uma só cota de classificações para o DeepSeek e o Jev do servidor.
  const freeQuota = freeStatus && freeStatus.hybrid.limit > 0 ? freeStatus.hybrid : null;
  const missingGenerative = !ownGenerative && freeStatus !== null && !freeStatus.deepseek.available;
  // Status ainda desconhecido: deixa tentar, o servidor decide.
  const canDiscover =
    ownGenerative || freeStatus === null || (!missingGenerative && freeQuota !== null && freeQuota.remaining > 0);
  // O Jev do servidor também gasta uma classificação da cota do dispositivo.
  const jevAvailable = ownJev || freeStatus === null || (freeStatus.jev.available && freeStatus.hybrid.remaining > 0);
  const docCount = useDataset((state) => state.active?.length ?? 0);
  const maxDocs = (freeStatus?.freeMaxDocs ?? 0).toLocaleString(locale === 'pt' ? 'pt-BR' : 'en');
  const discoverTooLarge = freeDocsLimitError(config, docCount, true);
  const manualTooLarge = freeDocsLimitError(config, docCount, false);
  const fill = (text: string, quota: { remaining: number; limit: number }) =>
    text.replace('{remaining}', String(quota.remaining)).replace('{limit}', String(quota.limit));
  const running = stage === 'running';

  return (
    <section className="space-y-4 rounded-xl border border-border/80 p-4" aria-labelledby="hybrid-title">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="space-y-1">
          <h3 id="hybrid-title" className="flex items-center gap-2 text-sm font-semibold">
            <Layers className="size-4 text-purple-600" aria-hidden />
            {copy.title}
            <Badge variant="purple">beta</Badge>
          </h3>
          <p className="text-xs text-muted-foreground">{copy.subtitle}</p>
        </div>
        <Button type="button" variant="outline" size="sm" onClick={() => setSettingsOpen(true)} disabled={running}>
          <Settings2 aria-hidden />
          {copy.settings}
        </Button>
      </div>

      <p className="text-xs leading-relaxed text-muted-foreground">{copy.description}</p>

      {error && (
        <div
          role="alert"
          className="flex items-start gap-2 rounded-xl border border-red-200 bg-red-50 p-3 text-xs text-red-800 dark:border-red-900 dark:bg-red-950/60 dark:text-red-300"
        >
          <AlertTriangle className="mt-0.5 size-4 shrink-0" aria-hidden />
          <span className="flex-1">{error}</span>
          {stage === 'error' && (
            <button type="button" onClick={dismiss} aria-label={copy.dismiss} className="shrink-0">
              <X className="size-4" aria-hidden />
            </button>
          )}
        </div>
      )}

      {(stage === 'idle' || stage === 'done' || stage === 'error') && (
        <div className="space-y-3">
          {(!ownGenerative || !ownJev) && freeStatus && (
            <p className={freeQuota?.remaining && !missingGenerative ? 'text-xs text-muted-foreground' : 'text-xs text-amber-800 dark:text-amber-300'}>
              {missingGenerative
                ? copy.missingGenerative
                : !freeQuota
                  ? copy.missingJev
                  : freeQuota.remaining > 0
                    ? fill(copy.freeDiscovery, freeQuota)
                    : fill(copy.freeDiscoveryExhausted, freeQuota)}
            </p>
          )}
          {!ownJev && freeStatus && (
            <p className={freeStatus.jev.available ? 'text-xs text-muted-foreground' : 'text-xs text-amber-800 dark:text-amber-300'}>
              {freeStatus.jev.available ? copy.jevFree.replace('{max}', maxDocs) : copy.missingJev}
            </p>
          )}
          {discoverTooLarge && stage !== 'error' && (
            <p className="text-xs text-amber-800 dark:text-amber-300">{discoverTooLarge}</p>
          )}
          <div className="space-y-1">
            <Label htmlFor="hybrid-focus" className="text-xs">{copy.focusLabel}</Label>
            <Input
              id="hybrid-focus"
              value={focus}
              maxLength={300}
              onChange={(event) => setFocus(event.target.value)}
              placeholder={copy.focusPlaceholder}
              className="h-9 text-sm"
            />
          </div>
          <div className="flex flex-wrap gap-2">
            <Button type="button" onClick={() => void start(focus)} disabled={!canDiscover || !jevAvailable || kmeansBusy || Boolean(discoverTooLarge)}>
              <Layers aria-hidden />
              {copy.discover}
            </Button>
            <Button type="button" variant="outline" onClick={startManual} disabled={!jevAvailable || kmeansBusy || Boolean(manualTooLarge)}>
              <ListChecks aria-hidden />
              {copy.manual}
            </Button>
          </div>
          <p className="text-[11px] leading-relaxed text-muted-foreground">{copy.privacy}</p>
        </div>
      )}

      {running && progress && (
        <div className="space-y-2" aria-live="polite">
          <div className="flex items-center justify-between gap-3 text-xs">
            <span className="flex items-center gap-2 font-medium">
              <Loader2 className="size-3.5 animate-spin" aria-hidden />
              {progress.phase}
            </span>
            <span className="tabular-nums text-muted-foreground">{progress.detail}</span>
          </div>
          <Progress value={Math.round(progress.ratio * 100)} aria-label={progress.phase} />
          <Button type="button" variant="ghost" size="sm" onClick={cancel}>
            {copy.cancel}
          </Button>
        </div>
      )}

      {stage === 'review' && draft && (
        <div className="space-y-3">
          <div className="space-y-1">
            <h4 className="text-sm font-semibold">{copy.reviewTitle}</h4>
            <p className="text-xs text-muted-foreground">{isManual ? copy.manualHint : copy.reviewHint}</p>
          </div>
          <TaxonomyEditor categories={draft} support={draftSupport} onChange={updateDraft} copy={copy} />
          <div className="flex flex-wrap justify-end gap-2 border-t border-border/80 pt-3">
            <Button type="button" variant="ghost" onClick={cancel}>
              {copy.cancel}
            </Button>
            <Button type="button" onClick={() => void confirm()}>
              <Check aria-hidden />
              {isManual ? copy.confirmManual : copy.confirm}
            </Button>
          </div>
        </div>
      )}

      {hybridRun && stage !== 'review' && !running && (
        <div className="space-y-2 border-t border-border/80 pt-4">
          <h4 className="text-sm font-semibold">{copy.resultTitle}</h4>
          <RunResult run={hybridRun} copy={copy} />
        </div>
      )}

      <HybridSettingsModal open={settingsOpen} onOpenChange={setSettingsOpen} />
    </section>
  );
}
