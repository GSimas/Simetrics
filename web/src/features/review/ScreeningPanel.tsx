import { Fragment, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import { Ban, Check, CircleHelp, ExternalLink, Keyboard, SearchX, Undo2, X } from 'lucide-react';

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
import { useReview, useScreeningRecords } from '@/state/review.store';
import type { ReviewCopy } from './copy';

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

const STATUS_VARIANT: Record<Status, 'success' | 'destructive' | 'warning' | 'secondary' | 'outline'> = {
  include: 'success',
  exclude: 'destructive',
  maybe: 'warning',
  'not-retrieved': 'secondary',
  pending: 'outline',
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
    if (position >= 0) virtualizer.scrollToIndex(position, { align: 'auto' });
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
    if (!selected) return;
    if (stage === 'title-abstract') {
      decideTitleAbstract(selected.key, decision as 'include' | 'exclude' | 'maybe' | null, reason);
    } else {
      if (decision === 'exclude' && reason === undefined && exclusionCriteria.length > 0) {
        setChoosingReason(true);
        return;
      }
      decideFullText(selected.key, decision as 'include' | 'exclude' | 'not-retrieved' | null, reason);
    }
    if (decision) advance(decision);
    else setChoosingReason(false);
  };

  useEffect(() => {
    const onKey = (event: KeyboardEvent): void => {
      if (event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey || isTyping(event.target)) return;
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

      const actions: Record<string, () => void> = {
        j: () => select(position + 1),
        arrowdown: () => select(position + 1),
        k: () => select(position - 1),
        arrowup: () => select(position - 1),
        i: () => decide('include'),
        e: () => decide('exclude'),
        u: () => decide(null),
        ...(stage === 'title-abstract' ? { t: () => decide('maybe'), m: () => decide('maybe') } : { n: () => decide('not-retrieved') }),
      };
      const action = actions[key];
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
      <div className="flex flex-col items-center gap-4 rounded-xl border border-dashed border-border p-10 text-center">
        <p className="max-w-xl text-sm text-muted-foreground">{copy.noDataset}</p>
        <Button type="button" variant="outline" onClick={() => setActiveTab('overview')}>
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
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex rounded-md border border-border p-0.5" role="tablist" aria-label={copy.steps.screening}>
          {(['title-abstract', 'full-text'] as const).map((value) => (
            <button
              key={value}
              type="button"
              role="tab"
              aria-selected={stage === value}
              onClick={() => {
                setStage(value);
                setFilter('pending');
                setSelectedKey(null);
                setAnchor(0);
                setChoosingReason(false);
              }}
              className={cn(
                'rounded px-3 py-1.5 text-xs font-medium transition-colors',
                stage === value ? 'bg-secondary text-foreground' : 'text-muted-foreground hover:text-foreground',
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
        <p className="rounded-xl border border-dashed border-border p-8 text-center text-sm text-muted-foreground">
          {copy.fullTextLocked}
        </p>
      ) : (
        <div className="grid gap-4 lg:grid-cols-[minmax(0,24rem)_minmax(0,1fr)]">
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
                  className={cn(
                    'rounded-md border px-2 py-1 text-[11px] transition-colors',
                    filter === value ? 'border-highlight text-highlight' : 'border-border text-muted-foreground hover:text-foreground',
                  )}
                >
                  {copy.filters[value]} <span className="tabular-nums">{counts[value].toLocaleString()}</span>
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
            <div ref={scrollRef} className="h-[60vh] overflow-y-auto rounded-xl border border-border/80" role="listbox" aria-label={copy.steps.screening}>
              {visible.length === 0 ? (
                <p className="flex h-full flex-col items-center justify-center gap-2 p-6 text-center text-xs text-muted-foreground">
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
                          'absolute inset-x-0 flex flex-col items-start justify-center gap-1 overflow-hidden border-b border-border/60 px-3 py-2 text-left transition-colors',
                          active ? 'bg-secondary' : 'hover:bg-muted/40',
                        )}
                        style={{ top: item.start, height: ROW_HEIGHT }}
                      >
                        <span className="line-clamp-2 text-xs font-medium leading-snug text-foreground">
                          {record.title || '—'}
                        </span>
                        <span className="flex w-full items-center gap-2 text-[10.5px] text-muted-foreground">
                          <span className="truncate">{[record.year, record.database].filter(Boolean).join(' · ')}</span>
                          {status !== 'pending' && (
                            <Badge variant={STATUS_VARIANT[status]} className="ml-auto shrink-0 px-1.5 py-0 text-[9.5px]">
                              {copy.decisionLabels[status]}
                            </Badge>
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
            <article className="space-y-4 rounded-xl border border-border/80 p-5" aria-live="polite">
              <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
                <span className="tabular-nums">
                  {copy.recordOf.replace('{n}', (position + 1).toLocaleString()).replace('{total}', visible.length.toLocaleString())}
                </span>
                {selected.database && <Badge variant="outline">{selected.database}</Badge>}
                <Badge variant={STATUS_VARIANT[currentStatus]} className="ml-auto">
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
                      className="inline-flex items-center gap-1 text-highlight underline-offset-4 hover:underline"
                    >
                      {copy.openDoi}
                      <ExternalLink className="size-3" aria-hidden />
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

              <div className="space-y-3 border-t border-border/80 pt-4">
                <div className="flex flex-wrap gap-2">
                  <Button type="button" variant={currentStatus === 'include' ? 'default' : 'outline'} onClick={() => decide('include')}>
                    <Check aria-hidden />
                    {copy.include}
                    <kbd className="font-mono text-[10px] opacity-60">I</kbd>
                  </Button>
                  <Button type="button" variant={currentStatus === 'exclude' ? 'destructive' : 'outline'} onClick={() => decide('exclude')}>
                    <X aria-hidden />
                    {copy.exclude}
                    <kbd className="font-mono text-[10px] opacity-60">E</kbd>
                  </Button>
                  {stage === 'title-abstract' ? (
                    <Button type="button" variant={currentStatus === 'maybe' ? 'secondary' : 'outline'} onClick={() => decide('maybe')}>
                      <CircleHelp aria-hidden />
                      {copy.maybe}
                      <kbd className="font-mono text-[10px] opacity-60">T</kbd>
                    </Button>
                  ) : (
                    <Button
                      type="button"
                      variant={currentStatus === 'not-retrieved' ? 'secondary' : 'outline'}
                      onClick={() => decide('not-retrieved')}
                    >
                      <Ban aria-hidden />
                      {copy.notRetrieved}
                      <kbd className="font-mono text-[10px] opacity-60">N</kbd>
                    </Button>
                  )}
                  {currentStatus !== 'pending' && (
                    <Button type="button" variant="ghost" onClick={() => decide(null)}>
                      <Undo2 aria-hidden />
                      {copy.clear}
                    </Button>
                  )}
                </div>

                {choosingReason && (
                  <div className="space-y-2 rounded-lg border border-amber-300/60 p-3 dark:border-amber-800/60" role="group" aria-label={copy.reason}>
                    <p className="text-xs font-medium">{copy.reasonRequired}</p>
                    <div className="flex flex-wrap gap-1.5">
                      {exclusionCriteria.map((criterion, index) => (
                        <Button
                          key={criterion.id}
                          type="button"
                          variant="outline"
                          size="sm"
                          onClick={() => decide('exclude', criterion.id)}
                        >
                          <kbd className="font-mono text-[10px] opacity-60">{index + 1}</kbd>
                          {criterion.text}
                        </Button>
                      ))}
                    </div>
                  </div>
                )}

                {currentStatus === 'exclude' && !choosingReason && (
                  <div className="flex flex-wrap items-center gap-2 text-xs">
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
                          className={cn(
                            'rounded-md border px-2 py-0.5 transition-colors',
                            currentReason === criterion.id
                              ? 'border-highlight text-highlight'
                              : 'border-border text-muted-foreground hover:text-foreground',
                          )}
                        >
                          {criterion.text}
                        </button>
                      ))
                    )}
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

                <p className="flex items-center gap-2 text-[11px] text-muted-foreground">
                  <Keyboard className="size-3.5" aria-hidden />
                  {stage === 'title-abstract' ? copy.shortcutsTa : copy.shortcutsFt}
                </p>
              </div>
            </article>
          ) : (
            <div className="flex items-center justify-center rounded-xl border border-dashed border-border p-10 text-center text-sm text-muted-foreground">
              {filter === 'pending' && !query ? copy.allDone : copy.emptyList}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
