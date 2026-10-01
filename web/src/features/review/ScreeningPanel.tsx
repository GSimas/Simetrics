import { Fragment, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import { Ban, Check, CircleHelp, ExternalLink, Keyboard, SearchX, Undo2, X } from 'lucide-react';

import { Collapse } from '@/components/Collapse';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import { Textarea } from '@/components/ui/textarea';
import { advancesToFullText } from '@/core/review/flow';
import type { ScreeningRecord } from '@/core/review/records';
import { buildHighlighter } from '@/core/review/search-string';
import type { RecordScreening, ReviewState, ScreeningStage } from '@/core/review/types';
import { cn } from '@/lib/utils';
import { useNavigation } from '@/state/navigation.store';
import { useReview, useReviewReadOnly, useScreeningRecords } from '@/state/review.store';
import type { ReviewCopy } from './copy';
import { chipClass, decisionClass, PRESS, useFlash, type FlashKind, type Tone } from './motion';
import { DecisionFlash } from './motion-parts';
import { ReadOnlyScope } from './parts';
import { FT_EXCLUSION } from '@/core/review/evidence';
import { DocumentSlot, EvidenceInline } from './evidence/DocumentSlot';

type Status = 'pending' | 'include' | 'exclude' | 'maybe' | 'not-retrieved';
type Filter = Status | 'all';

const FILTERS: Record<ScreeningStage, Filter[]> = {
  'title-abstract': ['pending', 'include', 'maybe', 'exclude', 'all'],
  'full-text': ['pending', 'include', 'exclude', 'not-retrieved', 'all'],
};

const ROW_HEIGHT = 76;

function statusOf(screening: RecordScreening | undefined, stage: ScreeningStage): Status {
  return (stage === 'title-abstract' ? screening?.ta : screening?.ft) ?? 'pending';
}

/** Cor de cada estado: verde incluir, vermelho excluir, âmbar talvez, neutro não recuperado. */
const TONE: Record<Filter, Tone> = {
  include: 'include',
  exclude: 'exclude',
  maybe: 'warning',
  'not-retrieved': 'neutral',
  pending: 'highlight',
  all: 'highlight',
};

const FLASH: Record<Exclude<Status, 'pending'>, FlashKind> = {
  include: 'include',
  exclude: 'exclude',
  maybe: 'neutral',
  'not-retrieved': 'neutral',
};

const STATUS_VARIANT: Record<Status, 'success' | 'destructive' | 'warning' | 'secondary' | 'outline'> = {
  include: 'success',
  exclude: 'destructive',
  maybe: 'warning',
  'not-retrieved': 'secondary',
  pending: 'outline',
};

/** Ponto de estado na lista, aceso com a cor da decisão. */
const DOT: Record<Status, string> = {
  include: 'bg-include shadow-[0_0_8px_var(--glow-include)]',
  exclude: 'bg-exclude shadow-[0_0_8px_var(--glow-exclude)]',
  maybe: 'bg-amber-500 shadow-[0_0_8px_rgb(245_158_11)]',
  'not-retrieved': 'bg-muted-foreground',
  pending: 'bg-transparent',
};

function Highlighted({ text, regex }: { text: string; regex: RegExp | null }): ReactNode {
  if (!regex || !text) return text;
  const parts: ReactNode[] = [];
  let last = 0;
  for (const match of text.matchAll(regex)) {
    const start = match.index ?? 0;
    if (start > last) parts.push(text.slice(last, start));
    parts.push(
      <mark key={start} className="rounded-sm bg-highlight/25 px-0.5 text-foreground">
        {match[0]}
      </mark>,
    );
    last = start + match[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts.map((part, index) => <Fragment key={index}>{part}</Fragment>);
}

/** Atalhos só valem fora de campos de texto — digitar uma nota não pode triar registros. */
function isTyping(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName) || target.getAttribute('role') === 'combobox';
}

export function ScreeningPanel({ review, copy }: { review: ReviewState; copy: ReviewCopy }) {
  const records = useScreeningRecords();
  const decideTitleAbstract = useReview((state) => state.decideTitleAbstract);
  const decideFullText = useReview((state) => state.decideFullText);
  const setNote = useReview((state) => state.setNote);
  const setActiveTab = useNavigation((state) => state.setActiveTab);
  const readOnly = useReviewReadOnly();
  const [flash, triggerFlash] = useFlash();

  const [stage, setStage] = useState<ScreeningStage>('title-abstract');
  const [filter, setFilter] = useState<Filter>('pending');
  const [query, setQuery] = useState('');
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  // Exclusão no texto completo aguardando o motivo.
  const [choosingReason, setChoosingReason] = useState(false);

  const decisions = review.decisions;
  const exclusionCriteria = review.criteria.filter((criterion) => criterion.kind === 'exclusion' && criterion.text.trim());
  const highlighter = useMemo(() => buildHighlighter(review.concepts), [review.concepts]);

  const haystacks = useMemo(
    () => new Map(records.map((record) => [record.key, `${record.title} ${record.authors} ${record.doi}`.toLowerCase()])),
    [records],
  );

  const stageRecords = useMemo(
    () => (stage === 'title-abstract' ? records : records.filter((record) => advancesToFullText(decisions[record.key]))),
    [records, decisions, stage],
  );

  const counts = useMemo(() => {
    const result: Record<Filter, number> = { pending: 0, include: 0, exclude: 0, maybe: 0, 'not-retrieved': 0, all: 0 };
    for (const record of stageRecords) {
      result[statusOf(decisions[record.key], stage)] += 1;
      result.all += 1;
    }
    return result;
  }, [stageRecords, decisions, stage]);

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return stageRecords.filter(
      (record) =>
        (filter === 'all' || statusOf(decisions[record.key], stage) === filter) &&
        (!needle || haystacks.get(record.key)!.includes(needle)),
    );
  }, [stageRecords, decisions, stage, filter, query, haystacks]);

  // O registro aberto sai da lista quando é decidido no filtro "Pendentes": o foco vai para
  // quem ocupou o lugar dele, e não volta ao topo.
  const [anchor, setAnchor] = useState(0);
  const selectedPosition = visible.findIndex((record) => record.key === selectedKey);
  const position = selectedPosition >= 0 ? selectedPosition : Math.min(anchor, visible.length - 1);
  const selected: ScreeningRecord | undefined = visible[position];

  const scrollRef = useRef<HTMLDivElement>(null);
  // Mesma limitação de DossierDocuments: o React Compiler não memoiza o retorno, e nada
  // dele atravessa componentes memoizados aqui.
  // eslint-disable-next-line react-hooks/incompatible-library
  const virtualizer = useVirtualizer({
    count: visible.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 8,
  });

  useEffect(() => {
    if (position >= 0) virtualizer.scrollToIndex(position, { align: 'auto', behavior: 'smooth' });
  }, [position, virtualizer]);

  const select = (index: number): void => {
    const clamped = Math.max(0, Math.min(visible.length - 1, index));
    const record = visible[clamped];
    if (record) setSelectedKey(record.key);
    setAnchor(clamped);
    setChoosingReason(false);
  };

  /** Depois de decidir, segue para o próximo registro da lista. */
  const advance = (decision: Status): void => {
    setChoosingReason(false);
    // Saindo do filtro, o registro some da lista e o seguinte ocupa a mesma posição.
    if (filter !== 'all' && filter !== decision) {
      setSelectedKey(null);
      setAnchor(position);
      return;
    }
    select(position + 1);
  };

  const decide = (decision: Status | null, reason?: string): void => {
    if (!selected || readOnly) return;
    if (stage === 'title-abstract') {
      decideTitleAbstract(selected.key, decision as 'include' | 'exclude' | 'maybe' | null, reason);
    } else {
      if (decision === 'exclude' && reason === undefined && exclusionCriteria.length > 0) {
        setChoosingReason(true);
        return;
      }
      decideFullText(selected.key, decision as 'include' | 'exclude' | 'not-retrieved' | null, reason);
    }
    if (decision) {
      triggerFlash(FLASH[decision as Exclude<Status, 'pending'>]);
      advance(decision);
    } else setChoosingReason(false);
  };

  useEffect(() => {
    const onKey = (event: KeyboardEvent): void => {
      if (event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey || isTyping(event.target)) return;
      // Com o leitor de PDF (ou outro diálogo) aberto, as teclas são dele.
      if (document.querySelector('[role="dialog"]')) return;
      const key = event.key.toLowerCase();

      if (choosingReason) {
        const number = Number(key);
        if (number >= 1 && number <= exclusionCriteria.length) {
          event.preventDefault();
          decide('exclude', exclusionCriteria[number - 1]!.id);
        } else if (key === 'escape') {
          setChoosingReason(false);
        }
        return;
      }

      const navigation: Record<string, () => void> = {
        j: () => select(position + 1),
        arrowdown: () => select(position + 1),
        k: () => select(position - 1),
        arrowup: () => select(position - 1),
      };
      // No exemplo (só visualização), o teclado navega mas não decide.
      const decisionsByKey: Record<string, () => void> = readOnly
        ? {}
        : {
            i: () => decide('include'),
            e: () => decide('exclude'),
            u: () => decide(null),
            ...(stage === 'title-abstract'
              ? { t: () => decide('maybe'), m: () => decide('maybe') }
              : { n: () => decide('not-retrieved') }),
          };
      const action = navigation[key] ?? decisionsByKey[key];
      if (action) {
        event.preventDefault();
        action();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  if (records.length === 0) {
    return (
      <div className="flex flex-col items-center gap-4 rounded-xl border border-dashed border-border p-10 text-center animate-in fade-in-0 duration-300">
        <p className="max-w-xl text-sm text-muted-foreground">{copy.noDataset}</p>
        <Button type="button" variant="outline" className="active:scale-[0.97]" onClick={() => setActiveTab('data')}>
          {copy.goToImport}
        </Button>
      </div>
    );
  }

  const done = counts.all - counts.pending;
  const current = selected ? decisions[selected.key] : undefined;
  const currentStatus = statusOf(current, stage);
  const currentReason = stage === 'title-abstract' ? current?.taReason : current?.ftReason;

  return (
    <div className="space-y-4" data-tour="review-screening">
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex gap-1 rounded-md border border-border p-0.5" role="tablist" aria-label={copy.steps.screening}>
          {(['title-abstract', 'full-text'] as const).map((value) => (
            <button
              key={value}
              type="button"
              role="tab"
              aria-selected={stage === value}
              data-review-target={value === 'title-abstract' ? 'screen-ta' : 'screen-ft'}
              onClick={() => {
                setStage(value);
                setFilter('pending');
                setSelectedKey(null);
                setAnchor(0);
                setChoosingReason(false);
              }}
              className={cn(
                'rounded px-3 py-1.5 text-xs font-medium',
                PRESS,
                stage === value
                  ? 'bg-highlight/10 text-foreground shadow-[0_0_18px_-8px_var(--highlight)]'
                  : 'text-muted-foreground hover:bg-muted/60 hover:text-foreground',
              )}
            >
              {value === 'title-abstract' ? copy.stageTitleAbstract : copy.stageFullText}
            </button>
          ))}
        </div>
        <div className="min-w-[12rem] flex-1 space-y-1">
          <Progress value={counts.all ? Math.round((done / counts.all) * 100) : 0} aria-label={copy.steps.screening} />
          <p className="text-[11px] tabular-nums text-muted-foreground">
            {copy.progress.replace('{done}', done.toLocaleString()).replace('{total}', counts.all.toLocaleString())}
          </p>
        </div>
      </div>

      {stage === 'full-text' && stageRecords.length === 0 ? (
        <p className="rounded-xl border border-dashed border-border p-8 text-center text-sm text-muted-foreground animate-in fade-in-0 duration-300">
          {copy.fullTextLocked}
        </p>
      ) : (
        <div key={stage} className="grid gap-4 animate-in fade-in-0 duration-300 lg:grid-cols-[minmax(0,24rem)_minmax(0,1fr)]">
          <div className="space-y-2">
            <div className="flex flex-wrap gap-1" role="group" aria-label={copy.filterLabel}>
              {FILTERS[stage].map((value) => (
                <button
                  key={value}
                  type="button"
                  aria-pressed={filter === value}
                  onClick={() => {
                    setFilter(value);
                    setSelectedKey(null);
                    setAnchor(0);
                  }}
                  className={chipClass(filter === value, TONE[value], 'sm')}
                >
                  {copy.filters[value]}{' '}
                  <span key={counts[value]} className="inline-block tabular-nums animate-in fade-in-0 zoom-in-90 duration-200">
                    {counts[value].toLocaleString()}
                  </span>
                </button>
              ))}
            </div>
            <Input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder={copy.searchRecords}
              aria-label={copy.searchRecords}
              className="h-8 text-sm"
            />
            <div
              ref={scrollRef}
              className="h-[60vh] overflow-y-auto rounded-xl border border-border/80 transition-[border-color] duration-300 hover:border-highlight/30"
              role="listbox"
              aria-label={copy.steps.screening}
            >
              {visible.length === 0 ? (
                <p className="flex h-full flex-col items-center justify-center gap-2 p-6 text-center text-xs text-muted-foreground animate-in fade-in-0 duration-300">
                  <SearchX className="size-5" aria-hidden />
                  {filter === 'pending' && !query ? copy.allDone : copy.emptyList}
                </p>
              ) : (
                <div className="relative" style={{ height: virtualizer.getTotalSize() }}>
                  {virtualizer.getVirtualItems().map((item) => {
                    const record = visible[item.index]!;
                    const status = statusOf(decisions[record.key], stage);
                    const active = item.index === position;
                    return (
                      <button
                        key={record.key}
                        type="button"
                        role="option"
                        aria-selected={active}
                        onClick={() => select(item.index)}
                        className={cn(
                          'absolute inset-x-0 flex flex-col items-start justify-center gap-1 overflow-hidden border-b border-border/60 border-l-2 px-3 py-2 text-left transition-[background-color,border-color] duration-200',
                          active ? 'border-l-highlight bg-secondary' : 'border-l-transparent hover:bg-muted/40',
                        )}
                        style={{ top: item.start, height: ROW_HEIGHT }}
                      >
                        <span className="line-clamp-2 text-xs font-medium leading-snug text-foreground">
                          {record.title || '—'}
                        </span>
                        <span className="flex w-full items-center gap-2 text-[10.5px] text-muted-foreground">
                          <span className="truncate">{[record.year, record.database].filter(Boolean).join(' · ')}</span>
                          {status !== 'pending' && (
                            <span className="ml-auto flex shrink-0 items-center gap-1.5">
                              <span className={cn('size-1.5 rounded-full transition-colors duration-300', DOT[status])} aria-hidden />
                              <Badge
                                key={status}
                                variant={STATUS_VARIANT[status]}
                                className="px-1.5 py-0 text-[9.5px] animate-in fade-in-0 zoom-in-90 duration-200"
                              >
                                {copy.decisionLabels[status]}
                              </Badge>
                            </span>
                          )}
                        </span>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          </div>

          {selected ? (
            // O contêiner fica montado entre um registro e outro: o lampejo da decisão
            // continua visível enquanto o próximo registro entra.
            <div className="relative rounded-xl">
              <DecisionFlash flash={flash} />
              <article
                key={selected.key}
                className="space-y-4 rounded-xl border border-border/80 p-5 animate-in fade-in-0 slide-in-from-right-2 duration-200"
                aria-live="polite"
              >
                <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
                  <span className="tabular-nums">
                    {copy.recordOf.replace('{n}', (position + 1).toLocaleString()).replace('{total}', visible.length.toLocaleString())}
                  </span>
                  {selected.database && <Badge variant="outline">{selected.database}</Badge>}
                  <Badge
                    key={currentStatus}
                    variant={STATUS_VARIANT[currentStatus]}
                    className="ml-auto animate-in fade-in-0 zoom-in-90 duration-200"
                  >
                    {copy.decisionLabels[currentStatus]}
                  </Badge>
                </div>

                <div className="space-y-2">
                  <h4 className="text-lg font-semibold leading-snug">
                    <Highlighted text={selected.title || '—'} regex={highlighter} />
                  </h4>
                  {selected.authors && <p className="text-xs text-muted-foreground">{selected.authors.split(';').join('; ')}</p>}
                  <p className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
                    {[selected.venue, selected.year].filter(Boolean).join(' · ')}
                    {selected.doi && (
                      <a
                        href={`https://doi.org/${selected.doi.replace(/^https?:\/\/(dx\.)?doi\.org\//i, '')}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="group inline-flex items-center gap-1 text-highlight underline-offset-4 hover:underline"
                      >
                        {copy.openDoi}
                        <ExternalLink
                          className="size-3 transition-transform duration-200 group-hover:-translate-y-0.5 group-hover:translate-x-0.5"
                          aria-hidden
                        />
                      </a>
                    )}
                  </p>
                </div>

                <p className="max-h-[32vh] overflow-y-auto whitespace-pre-line text-sm leading-relaxed">
                  {selected.abstract ? (
                    <Highlighted text={selected.abstract} regex={highlighter} />
                  ) : (
                    <span className="text-muted-foreground">{copy.noAbstract}</span>
                  )}
                </p>

                {selected.keywords && (
                  <p className="text-xs text-muted-foreground">
                    <Highlighted text={selected.keywords} regex={highlighter} />
                  </p>
                )}

                <ReadOnlyScope readOnly={readOnly} className="space-y-3 border-t border-border/80 pt-4">
                  <div className="flex flex-wrap gap-2">
                    <button type="button" className={decisionClass(currentStatus === 'include', 'include')} onClick={() => decide('include')}>
                      <Check aria-hidden />
                      {copy.include}
                      <kbd className="font-mono text-[10px] opacity-60">I</kbd>
                    </button>
                    <button type="button" className={decisionClass(currentStatus === 'exclude', 'exclude')} onClick={() => decide('exclude')}>
                      <X aria-hidden />
                      {copy.exclude}
                      <kbd className="font-mono text-[10px] opacity-60">E</kbd>
                    </button>
                    {stage === 'title-abstract' ? (
                      <button type="button" className={decisionClass(currentStatus === 'maybe', 'warning')} onClick={() => decide('maybe')}>
                        <CircleHelp aria-hidden />
                        {copy.maybe}
                        <kbd className="font-mono text-[10px] opacity-60">T</kbd>
                      </button>
                    ) : (
                      <button
                        type="button"
                        className={decisionClass(currentStatus === 'not-retrieved', 'neutral')}
                        onClick={() => decide('not-retrieved')}
                      >
                        <Ban aria-hidden />
                        {copy.notRetrieved}
                        <kbd className="font-mono text-[10px] opacity-60">N</kbd>
                      </button>
                    )}
                    {currentStatus !== 'pending' && (
                      <Button
                        type="button"
                        variant="ghost"
                        className="animate-in fade-in-0 slide-in-from-left-1 duration-200 active:scale-[0.97]"
                        onClick={() => decide(null)}
                      >
                        <Undo2 aria-hidden />
                        {copy.clear}
                      </Button>
                    )}
                  </div>

                  <Collapse open={choosingReason} delayOpen={false}>
                    <div
                      className="space-y-2 rounded-lg border border-exclude/50 bg-exclude/5 p-3 shadow-[0_0_28px_-16px_var(--glow-exclude)]"
                      role="group"
                      aria-label={copy.reason}
                    >
                      <p className="text-xs font-medium">{copy.reasonRequired}</p>
                      <div className="flex flex-wrap gap-1.5">
                        {exclusionCriteria.map((criterion, index) => (
                          <button
                            key={criterion.id}
                            type="button"
                            onClick={() => decide('exclude', criterion.id)}
                            className={cn(chipClass(false, 'exclude'), 'inline-flex items-center gap-1.5')}
                          >
                            <kbd className="font-mono text-[10px] opacity-60">{index + 1}</kbd>
                            {criterion.text}
                          </button>
                        ))}
                      </div>
                    </div>
                  </Collapse>

                  <Collapse open={currentStatus === 'exclude' && !choosingReason} delayOpen={false}>
                    <div className="flex flex-wrap items-center gap-2 p-0.5 text-xs">
                      <span className="text-muted-foreground">
                        {stage === 'title-abstract' ? copy.reasonOptional : copy.reason}:
                      </span>
                      {exclusionCriteria.length === 0 ? (
                        <span className="text-muted-foreground">{copy.noCriteria}</span>
                      ) : (
                        exclusionCriteria.map((criterion) => (
                          <button
                            key={criterion.id}
                            type="button"
                            aria-pressed={currentReason === criterion.id}
                            onClick={() =>
                              stage === 'title-abstract'
                                ? decideTitleAbstract(selected.key, 'exclude', currentReason === criterion.id ? undefined : criterion.id)
                                : decideFullText(selected.key, 'exclude', criterion.id)
                            }
                            className={chipClass(currentReason === criterion.id, 'exclude', 'sm')}
                          >
                            {criterion.text}
                          </button>
                        ))
                      )}
                    </div>
                  </Collapse>

                  {stage === 'full-text' && (
                    <div className="space-y-2">
                      <DocumentSlot studyKey={selected.key} scope="full-text" />
                      <EvidenceInline studyKey={selected.key} target={FT_EXCLUSION} scope="full-text" />
                    </div>
                  )}

                  <div className="space-y-1">
                    <label htmlFor="screening-note" className="text-xs font-medium">
                      {copy.note}
                    </label>
                    <Textarea
                      id="screening-note"
                      key={selected.key}
                      defaultValue={current?.note ?? ''}
                      onBlur={(event) => {
                        if ((current?.note ?? '') !== event.target.value) setNote(selected.key, event.target.value);
                      }}
                      placeholder={copy.notePlaceholder}
                      rows={2}
                      className="text-sm"
                    />
                  </div>
                </ReadOnlyScope>

                <p className="flex items-center gap-2 text-[11px] text-muted-foreground">
                  <Keyboard className="size-3.5" aria-hidden />
                  {stage === 'title-abstract' ? copy.shortcutsTa : copy.shortcutsFt}
                </p>
              </article>
            </div>
          ) : (
            <div className="flex items-center justify-center rounded-xl border border-dashed border-border p-10 text-center text-sm text-muted-foreground animate-in fade-in-0 duration-300">
              {filter === 'pending' && !query ? copy.allDone : copy.emptyList}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
