import { useEffect, useMemo, useState } from 'react';
import { AlertTriangle, Bot, Check, FileText, Loader2, MousePointerSquareDashed, Quote, Sparkles, Trash2, User, X } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { canAccept, extractionTarget, FT_EXCLUSION, qualityTarget } from '@/core/review/evidence';
import type { Evidence, EvidenceTargetKey, ExtractionField, ReviewState, TargetEvidence, VerificationStatus } from '@/core/review/types';
import { getPdf } from '@/lib/pdf-store';
import { cn } from '@/lib/utils';
import { useLocale } from '@/state/locale.store';
import { useReview, useReviewReadOnly, useScreeningRecords } from '@/state/review.store';
import { REVIEW_COPY } from '../copy';
import { FieldInput } from '../ExtractionPanel';
import { chipClass, PRESS } from '../motion';
import { ReadOnlyScope } from '../parts';
import { fill, useEvidenceCopy, type EvidenceCopy } from './copy';
import { answerText, suggestionText } from './format';
import { PdfViewer, type ViewerHighlight } from './PdfViewer';
import { useEvidenceReader, type ReaderScope } from './reader-store';
import { useEvidenceAi } from './useEvidenceAi';

interface TargetItem {
  target: EvidenceTargetKey;
  label: string;
  field?: ExtractionField;
  questionId?: string;
}

function targetsOf(review: ReviewState, scope: ReaderScope, copy: EvidenceCopy): TargetItem[] {
  if (scope === 'extraction') {
    return review.extractionFields.map((field) => ({ target: extractionTarget(field.id), label: field.label || '—', field }));
  }
  if (scope === 'quality') {
    return review.qualityQuestions.map((question) => ({ target: qualityTarget(question.id), label: question.text || '—', questionId: question.id }));
  }
  return [{ target: FT_EXCLUSION, label: copy.exclusionTarget }];
}

const STATUS_VARIANT: Record<VerificationStatus, 'warning' | 'success' | 'blue' | 'destructive' | 'secondary'> = {
  suggested: 'warning',
  confirmed: 'success',
  edited: 'blue',
  rejected: 'destructive',
  manual: 'secondary',
};

function EvidenceItem({
  item,
  copy,
  onGo,
  onRemove,
}: {
  item: Evidence;
  copy: EvidenceCopy;
  onGo: () => void;
  onRemove?: () => void;
}) {
  const missing = item.location === 'not-found';
  return (
    <li className="group flex items-start gap-2 rounded-md border border-border/70 p-2 text-xs">
      <span className="mt-0.5 text-muted-foreground" title={copy.origin[item.origin]}>
        {item.origin === 'ai' ? <Bot className="size-3.5" aria-label={copy.origin.ai} /> : <User className="size-3.5" aria-label={copy.origin.manual} />}
      </span>
      <button
        type="button"
        onClick={onGo}
        disabled={missing && item.rects.length === 0}
        className={cn('min-w-0 flex-1 rounded-sm text-left hover:text-foreground disabled:cursor-default', PRESS)}
        title={missing ? copy.locationHint['not-found'] : copy.goTo}
      >
        <span className="font-mono text-[10.5px] text-muted-foreground">
          {copy.page} {item.page}
        </span>{' '}
        {item.quote ? <q className="italic leading-snug">{item.quote.length > 220 ? `${item.quote.slice(0, 220)}…` : item.quote}</q> : <span>{copy.areaLabel}</span>}
        {item.origin === 'ai' && (
          <span
            className={cn(
              'mt-1 flex items-center gap-1 text-[10.5px]',
              missing ? 'text-exclude' : item.location === 'approximate' ? 'text-amber-600 dark:text-amber-300' : 'text-include',
            )}
            title={copy.locationHint[item.location]}
          >
            {missing ? <AlertTriangle className="size-3" aria-hidden /> : <Check className="size-3" aria-hidden />}
            {copy.location[item.location]}
          </span>
        )}
      </button>
      {onRemove && (
        <button
          type="button"
          onClick={onRemove}
          aria-label={copy.removeEvidence}
          title={copy.removeEvidence}
          className={cn('rounded-sm p-0.5 text-muted-foreground opacity-60 hover:text-exclude group-hover:opacity-100', PRESS)}
        >
          <Trash2 className="size-3.5" aria-hidden />
        </button>
      )}
    </li>
  );
}

function TargetCard({
  item,
  studyKey,
  review,
  entry,
  active,
  onActivate,
  onGo,
  readOnly,
  copy,
}: {
  item: TargetItem;
  studyKey: string;
  review: ReviewState;
  entry: TargetEvidence | undefined;
  active: boolean;
  onActivate: () => void;
  onGo: (evidence: Evidence) => void;
  readOnly: boolean;
  copy: EvidenceCopy;
}) {
  const locale = useLocale((state) => state.locale);
  const reviewCopy = REVIEW_COPY[locale === 'en' ? 'en' : 'pt'];
  const words = { yes: reviewCopy.yes, no: reviewCopy.no };
  const setExtractionValue = useReview((state) => state.setExtractionValue);
  const answerQuality = useReview((state) => state.answerQuality);
  const acceptSuggestion = useReview((state) => state.acceptSuggestion);
  const rejectSuggestion = useReview((state) => state.rejectSuggestion);
  const removeEvidence = useReview((state) => state.removeEvidence);
  const suggestion = entry?.suggestion;
  const pending = entry?.status === 'suggested';
  const acceptable = canAccept(entry);

  return (
    <section
      onClick={onActivate}
      onFocusCapture={onActivate}
      aria-current={active || undefined}
      className={cn(
        'space-y-2.5 rounded-lg border p-3 transition-[border-color,box-shadow,background-color] duration-200',
        active ? 'border-highlight bg-highlight/5 shadow-[0_0_24px_-14px_var(--highlight)]' : 'border-border/80 hover:border-highlight/40',
      )}
    >
      <div className="flex items-start gap-2">
        <h4 className="min-w-0 flex-1 text-sm font-medium leading-snug">{item.label}</h4>
        {entry && (
          <Badge key={entry.status} variant={STATUS_VARIANT[entry.status]} className="shrink-0 animate-in fade-in-0 zoom-in-90 px-1.5 py-0 text-[10px] duration-200">
            {copy.status[entry.status]}
          </Badge>
        )}
      </div>

      {suggestion && (
        <div
          className={cn(
            'space-y-2 rounded-md border p-2.5 text-xs',
            pending ? 'border-amber-500/50 bg-amber-500/5' : 'border-border/70 bg-muted/30',
          )}
        >
          <p className="flex items-start gap-1.5">
            <Sparkles className="mt-0.5 size-3.5 shrink-0 text-highlight" aria-hidden />
            <span>
              <span className="text-muted-foreground">{copy.aiSuggests}:</span>{' '}
              <strong className="font-semibold">{suggestionText(review, item.target, suggestion.value, words)}</strong>
              {suggestion.rationale && <span className="mt-0.5 block text-muted-foreground">{suggestion.rationale}</span>}
              <span className="mt-0.5 block font-mono text-[10px] text-muted-foreground/80">{suggestion.model}</span>
            </span>
          </p>
          {pending && !readOnly && (
            <div className="flex flex-wrap items-center gap-1.5">
              <button
                type="button"
                disabled={!acceptable}
                title={acceptable ? undefined : copy.acceptBlocked}
                onClick={(event) => {
                  event.stopPropagation();
                  acceptSuggestion(studyKey, item.target);
                }}
                className={cn(chipClass(false, 'include', 'sm'), 'inline-flex items-center gap-1 disabled:cursor-not-allowed disabled:opacity-50')}
              >
                <Check className="size-3.5" aria-hidden />
                {copy.accept}
              </button>
              <button
                type="button"
                onClick={(event) => {
                  event.stopPropagation();
                  rejectSuggestion(studyKey, item.target);
                }}
                className={cn(chipClass(false, 'exclude', 'sm'), 'inline-flex items-center gap-1')}
              >
                <X className="size-3.5" aria-hidden />
                {copy.reject}
              </button>
              {!acceptable && <span className="text-[10.5px] text-exclude">{copy.acceptBlocked}</span>}
            </div>
          )}
        </div>
      )}

      <ReadOnlyScope readOnly={readOnly}>
        {item.field ? (
          <div className="space-y-1">
            <span id={`extraction-${item.field.id}-label`} className="block text-[11px] text-muted-foreground">
              {copy.yourAnswer}
            </span>
            <FieldInput
              field={item.field}
              value={review.extraction[studyKey]?.values[item.field.id]}
              onChange={(value) => setExtractionValue(studyKey, item.field!.id, value)}
              copy={reviewCopy}
            />
          </div>
        ) : item.questionId ? (
          <div className="flex flex-wrap gap-1.5" role="radiogroup" aria-label={item.label}>
            {review.qualityAnswers.map((answer) => {
              const checked = review.quality[studyKey]?.[item.questionId!] === answer.id;
              return (
                <button
                  key={answer.id}
                  type="button"
                  role="radio"
                  aria-checked={checked}
                  onClick={() => answerQuality(studyKey, item.questionId!, checked ? null : answer.id)}
                  className={chipClass(checked, 'highlight', 'sm')}
                >
                  {answer.label || '—'}
                </button>
              );
            })}
          </div>
        ) : (
          <p className="text-xs">
            {answerText(review, studyKey, item.target, words) || <span className="text-muted-foreground">{copy.exclusionNone}</span>}
          </p>
        )}
      </ReadOnlyScope>

      {entry && entry.evidence.length > 0 ? (
        <ul className="space-y-1.5">
          {entry.evidence.map((evidence) => (
            <EvidenceItem
              key={evidence.id}
              item={evidence}
              copy={copy}
              onGo={() => onGo(evidence)}
              {...(readOnly ? {} : { onRemove: () => removeEvidence(studyKey, item.target, evidence.id) })}
            />
          ))}
        </ul>
      ) : (
        <p className="text-[11px] text-muted-foreground">{copy.noEvidence}</p>
      )}
      {active && !readOnly && <p className="text-[10.5px] text-highlight">{copy.activeHint}</p>}
    </section>
  );
}

function AiBar({ studyKey, scope, title, copy }: { studyKey: string; scope: 'extraction' | 'quality'; title: string; copy: EvidenceCopy }) {
  const { run, running, messages, preview, refreshDestination } = useEvidenceAi(studyKey, scope, title, copy);
  const [confirming, setConfirming] = useState(false);
  const scanned = useReview((state) => state.review?.documents[studyKey]?.textLayer === 'none');
  const blocked = scanned ? copy.aiNoText : preview && !preview.destination.available ? copy.aiUnavailable : null;
  const hasSuggestions = useReview((state) =>
    Object.keys(state.review?.evidence[studyKey] ?? {}).some((key) => key.startsWith(scope) && state.review?.evidence[studyKey]?.[key as EvidenceTargetKey]?.suggestion),
  );
  const details = preview && {
    chars: preview.article.chars.toLocaleString(),
    refs: preview.article.droppedReferences ? copy.aiRefsDropped : '',
    provider: preview.destination.provider,
    model: preview.destination.model,
    route: preview.destination.ownKey ? copy.aiRouteOwn : copy.aiRouteServer,
  };

  return (
    <div className="space-y-2 border-b border-border/80 px-4 py-3">
      <div className="flex flex-wrap items-center gap-2">
        <Button
          type="button"
          size="sm"
          disabled={running || !preview || blocked !== null}
          onClick={() => {
            refreshDestination();
            setConfirming(true);
          }}
          className="active:scale-[0.97]"
        >
          {running ? <Loader2 className="animate-spin" aria-hidden /> : <Sparkles aria-hidden />}
          {running ? copy.aiRunning : hasSuggestions ? copy.aiRerun : copy.aiRun}
        </Button>
      </div>
      {blocked ? (
        <p className="text-[11px] leading-relaxed text-amber-700 dark:text-amber-300">{blocked}</p>
      ) : (
        details && <p className="text-[11px] leading-relaxed text-muted-foreground">{fill(copy.aiDisclosure, details)}</p>
      )}
      {messages.map((message) => (
        <p
          key={message.text}
          className={cn('text-xs animate-in fade-in-0 duration-200', message.tone === 'error' ? 'text-exclude' : 'text-foreground')}
          role={message.tone === 'error' ? 'alert' : 'status'}
        >
          {message.text}
        </p>
      ))}
      <Dialog open={confirming} onOpenChange={setConfirming}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>{copy.aiConfirmTitle}</DialogTitle>
            <DialogDescription>{details ? fill(copy.aiConfirmBody, details) : ''}</DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setConfirming(false)}>
              {copy.cancel}
            </Button>
            <Button
              onClick={() => {
                setConfirming(false);
                void run();
              }}
            >
              <Sparkles aria-hidden />
              {copy.aiConfirm}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

/** O leitor aberto (ver `useEvidenceReader`): PDF à esquerda, perguntas e conferência à direita. */
export function EvidenceReaderHost() {
  const request = useEvidenceReader((state) => state.request);
  const close = useEvidenceReader((state) => state.close);
  // Sem o PDF (removido, outro projeto aberto), não há o que ler: o leitor fica fechado.
  const hasDocument = useReview((state) => !!request && !!state.review?.documents[request.studyKey]);
  return (
    <Dialog open={request !== null && hasDocument} onOpenChange={(open) => !open && close()}>
      <DialogContent className="flex h-[94vh] w-[97vw] max-w-[1600px] flex-col gap-0 overflow-hidden p-0 sm:max-w-[1600px]">
        {request && <EvidenceReader key={`${request.studyKey}-${request.scope}`} />}
      </DialogContent>
    </Dialog>
  );
}

function EvidenceReader() {
  const copy = useEvidenceCopy();
  const request = useEvidenceReader((state) => state.request)!;
  const review = useReview((state) => state.review);
  const addEvidence = useReview((state) => state.addEvidence);
  const readOnly = useReviewReadOnly();
  const records = useScreeningRecords();
  const study = records.find((record) => record.key === request.studyKey);
  const document = review?.documents[request.studyKey];
  const targets = useMemo(() => (review ? targetsOf(review, request.scope, copy) : []), [review, request.scope, copy]);
  const [active, setActive] = useState<EvidenceTargetKey | null>(request.target ?? targets[0]?.target ?? null);
  // Num PDF escaneado não há texto para selecionar: a marcação começa por área.
  const [areaMode, setAreaMode] = useState(() => useReview.getState().review?.documents[request.studyKey]?.textLayer === 'none');
  const [loaded, setLoaded] = useState<{ hash: string; blob: Blob | null } | null>(null);
  const [focus, setFocus] = useState<{ id: string; page: number; nonce: number } | null>(null);
  const hash = document?.hash;
  // `undefined` enquanto lê; `null` quando o arquivo não está neste navegador.
  const blob = !hash ? null : loaded?.hash === hash ? loaded.blob : undefined;

  useEffect(() => {
    let cancelled = false;
    if (!hash) return;
    void getPdf(hash).then((stored) => {
      if (cancelled) return;
      setLoaded({ hash, blob: stored?.data ?? null });
      // Aberto num trecho: pula para ele assim que o PDF chega.
      const wanted = useEvidenceReader.getState().request?.evidenceId;
      const item = Object.values(useReview.getState().review?.evidence[request.studyKey] ?? {})
        .flatMap((entry) => entry?.evidence ?? [])
        .find((candidate) => candidate.id === wanted);
      if (item) setFocus({ id: item.id, page: item.page, nonce: Date.now() });
    });
    return () => {
      cancelled = true;
    };
  }, [hash, request.studyKey]);

  const evidence = review?.evidence[request.studyKey];

  const highlights = useMemo<ViewerHighlight[]>(() => {
    const list: ViewerHighlight[] = [];
    for (const item of targets) {
      for (const piece of evidence?.[item.target]?.evidence ?? []) {
        if (piece.location === 'not-found' && piece.rects.length === 0) continue;
        list.push({
          id: piece.id,
          page: piece.page,
          quote: piece.location === 'not-found' ? '' : piece.quote,
          prefix: piece.prefix,
          suffix: piece.suffix,
          rects: piece.rects,
          tone: item.target === active ? (piece.location === 'approximate' ? 'warning' : 'active') : 'muted',
        });
      }
    }
    return list;
  }, [targets, evidence, active]);

  if (!review || !document) return null;
  const activeLabel = targets.find((item) => item.target === active)?.label ?? '';
  const canMark = !readOnly && active !== null;

  return (
    <>
      <DialogHeader className="space-y-1 border-b border-border/80 px-5 py-3 pr-12 text-left">
        <DialogTitle className="line-clamp-2 text-base">{study?.title || copy.readerTitle}</DialogTitle>
        <DialogDescription className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
          <span className="inline-flex items-center gap-1">
            <FileText className="size-3.5" aria-hidden />
            {document.name} · {fill(copy.pages, { n: document.pages })}
          </span>
          {document.textLayer !== 'ok' && (
            <span className="inline-flex items-center gap-1 text-amber-700 dark:text-amber-300">
              <AlertTriangle className="size-3.5" aria-hidden />
              {document.textLayer === 'none' ? copy.scanned : copy.partial}
            </span>
          )}
        </DialogDescription>
      </DialogHeader>

      <div className="grid min-h-0 flex-1 grid-cols-[minmax(0,1fr)] grid-rows-[minmax(0,1fr)_minmax(0,1fr)] lg:grid-cols-[minmax(0,1fr)_minmax(22rem,30rem)] lg:grid-rows-1">
        <div className="flex min-h-0 flex-col border-b border-border/80 lg:border-b-0 lg:border-r">
          <div className="flex flex-wrap items-center gap-2 border-b border-border/80 px-3 py-1.5 text-[11px] text-muted-foreground">
            {!readOnly && (
              <button
                type="button"
                aria-pressed={areaMode}
                onClick={() => setAreaMode((mode) => !mode)}
                className={cn(chipClass(areaMode, 'highlight', 'sm'), 'inline-flex items-center gap-1')}
              >
                <MousePointerSquareDashed className="size-3.5" aria-hidden />
                {copy.areaMode}
              </button>
            )}
            <span className="min-w-0 flex-1">
              {readOnly ? '' : active ? fill(areaMode ? copy.areaModeHint : copy.textModeHint, { target: activeLabel }) : copy.noTarget}
            </span>
          </div>
          {blob === undefined ? (
            <p className="flex flex-1 items-center justify-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="size-4 animate-spin" aria-hidden />
              {copy.reading}
            </p>
          ) : blob === null ? (
            <p className="flex flex-1 items-center justify-center p-8 text-center text-sm text-muted-foreground">{copy.missingFile}</p>
          ) : (
            <PdfViewer
              data={blob}
              highlights={highlights}
              focus={focus}
              areaMode={areaMode && canMark}
              onArea={(page, rect) => {
                if (!active) return;
                addEvidence(request.studyKey, active, { page, quote: '', prefix: '', suffix: '', rects: [rect], origin: 'manual', location: 'exact' });
                // Num PDF com texto, a área é exceção (tabela, figura); escaneado, é o único jeito.
                if (document.textLayer !== 'none') setAreaMode(false);
              }}
              selectionAction={(selection, clear) =>
                canMark ? (
                  <Button
                    type="button"
                    size="sm"
                    className="max-w-64 shadow-lg active:scale-[0.97]"
                    onClick={() => {
                      addEvidence(request.studyKey, active!, { ...selection, origin: 'manual', location: 'exact' });
                      clear();
                    }}
                  >
                    <Quote aria-hidden />
                    <span className="truncate">{fill(copy.useSelection, { target: activeLabel })}</span>
                  </Button>
                ) : null
              }
              copy={copy}
              className="min-h-0 flex-1"
            />
          )}
        </div>

        <div className="flex min-h-0 flex-col">
          {request.scope !== 'full-text' && !readOnly && (
            <AiBar studyKey={request.studyKey} scope={request.scope} title={study?.title ?? ''} copy={copy} />
          )}
          {readOnly && request.scope !== 'full-text' && <p className="border-b border-border/80 px-4 py-2 text-[11px] text-muted-foreground">{copy.aiReadonly}</p>}
          <div className="min-h-0 flex-1 space-y-2.5 overflow-y-auto p-4">
            <p className="eyebrow">{copy.questions}</p>
            {targets.map((item) => (
              <TargetCard
                key={item.target}
                item={item}
                studyKey={request.studyKey}
                review={review}
                entry={evidence?.[item.target]}
                active={active === item.target}
                onActivate={() => setActive(item.target)}
                onGo={(piece) => {
                  setActive(item.target);
                  setFocus({ id: piece.id, page: piece.page, nonce: Date.now() });
                }}
                readOnly={readOnly}
                copy={copy}
              />
            ))}
          </div>
        </div>
      </div>
    </>
  );
}
