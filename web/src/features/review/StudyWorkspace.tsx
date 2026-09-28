import { useMemo, useState, type ReactNode } from 'react';
import { ChevronLeft, ChevronRight, ExternalLink, SearchX } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import type { ScreeningRecord } from '@/core/review/records';
import { cn } from '@/lib/utils';
import type { ReviewCopy } from './copy';

type Filter = 'pending' | 'done' | 'all';

/**
 * Lista dos estudos incluídos e o estudo aberto — o esqueleto comum da avaliação de
 * qualidade e da extração de dados. Cada etapa diz o que é "concluído" e desenha o
 * formulário do estudo.
 */
export function StudyWorkspace({
  studies,
  isDone,
  badge,
  copy,
  label,
  children,
}: {
  studies: ScreeningRecord[];
  isDone: (key: string) => boolean;
  badge?: (study: ScreeningRecord) => ReactNode;
  copy: ReviewCopy;
  /** Nome acessível da lista. */
  label: string;
  children: (study: ScreeningRecord) => ReactNode;
}) {
  const [filter, setFilter] = useState<Filter>('pending');
  const [query, setQuery] = useState('');
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [anchor, setAnchor] = useState(0);

  const done = studies.filter((study) => isDone(study.key)).length;
  const counts: Record<Filter, number> = { pending: studies.length - done, done, all: studies.length };

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return studies.filter(
      (study) =>
        (filter === 'all' || (filter === 'done') === isDone(study.key)) &&
        (!needle || `${study.title} ${study.authors} ${study.doi}`.toLowerCase().includes(needle)),
    );
  }, [studies, filter, query, isDone]);

  // Concluir um estudo no filtro "Pendentes" o tira da lista: o seguinte ocupa o lugar.
  const selectedPosition = visible.findIndex((study) => study.key === selectedKey);
  const position = selectedPosition >= 0 ? selectedPosition : Math.min(anchor, visible.length - 1);
  const selected = visible[position];

  const select = (index: number): void => {
    const clamped = Math.max(0, Math.min(visible.length - 1, index));
    setSelectedKey(visible[clamped]?.key ?? null);
    setAnchor(clamped);
  };

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,22rem)_minmax(0,1fr)]">
      <div className="space-y-2">
        <div className="space-y-1">
          <Progress value={studies.length ? Math.round((done / studies.length) * 100) : 0} aria-label={copy.studyFilters.done} />
          <p className="text-[11px] tabular-nums text-muted-foreground">
            {copy.studyProgress.replace('{done}', done.toLocaleString()).replace('{total}', studies.length.toLocaleString())}
          </p>
        </div>
        <div className="flex flex-wrap gap-1" role="group" aria-label={copy.filterLabel}>
          {(['pending', 'done', 'all'] as const).map((value) => (
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
              {copy.studyFilters[value]} <span className="tabular-nums">{counts[value].toLocaleString()}</span>
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
        <div className="max-h-[60vh] overflow-y-auto rounded-xl border border-border/80" role="listbox" aria-label={label}>
          {visible.length === 0 ? (
            <p className="flex flex-col items-center justify-center gap-2 p-6 text-center text-xs text-muted-foreground">
              <SearchX className="size-5" aria-hidden />
              {copy.emptyList}
            </p>
          ) : (
            visible.map((study, index) => (
              <button
                key={study.key}
                type="button"
                role="option"
                aria-selected={index === position}
                onClick={() => select(index)}
                className={cn(
                  'flex w-full flex-col items-start gap-1 border-b border-border/60 px-3 py-2 text-left transition-colors last:border-b-0',
                  index === position ? 'bg-secondary' : 'hover:bg-muted/40',
                )}
              >
                <span className="line-clamp-2 text-xs font-medium leading-snug">{study.title || '—'}</span>
                <span className="flex w-full items-center gap-2 text-[10.5px] text-muted-foreground">
                  <span className="truncate">{[study.year, study.venue].filter(Boolean).join(' · ')}</span>
                  <span className="ml-auto shrink-0">{badge?.(study)}</span>
                </span>
              </button>
            ))
          )}
        </div>
      </div>

      {selected ? (
        <article className="space-y-4 rounded-xl border border-border/80 p-5">
          <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
            <span className="tabular-nums">
              {copy.studyOf.replace('{n}', (position + 1).toLocaleString()).replace('{total}', visible.length.toLocaleString())}
            </span>
            <span className="ml-auto flex gap-1">
              <Button type="button" variant="ghost" size="sm" onClick={() => select(position - 1)} disabled={position <= 0}>
                <ChevronLeft aria-hidden />
                {copy.previous}
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={() => select(position + 1)}
                disabled={position >= visible.length - 1}
              >
                {copy.next}
                <ChevronRight aria-hidden />
              </Button>
            </span>
          </div>
          <div className="space-y-1.5">
            <h4 className="text-base font-semibold leading-snug">{selected.title || '—'}</h4>
            {selected.authors && <p className="text-xs text-muted-foreground">{selected.authors.split(';').join('; ')}</p>}
            <p className="flex flex-wrap items-center gap-x-3 text-xs text-muted-foreground">
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
            {selected.abstract && (
              <details className="text-sm">
                <summary className="cursor-pointer select-none text-xs text-muted-foreground">{copy.abstractLabel}</summary>
                <p className="mt-2 max-h-[24vh] overflow-y-auto whitespace-pre-line leading-relaxed">{selected.abstract}</p>
              </details>
            )}
          </div>
          <div className="border-t border-border/80 pt-4">{children(selected)}</div>
        </article>
      ) : (
        <div className="flex items-center justify-center rounded-xl border border-dashed border-border p-10 text-center text-sm text-muted-foreground">
          {copy.emptyList}
        </div>
      )}
    </div>
  );
}

/** Aviso de etapa vazia, com atalho opcional para o protocolo. */
export function EmptyStep({ message, action }: { message: string; action?: { label: string; onClick: () => void } }) {
  return (
    <div className="flex flex-col items-center gap-4 rounded-xl border border-dashed border-border p-10 text-center">
      <p className="max-w-xl text-sm text-muted-foreground">{message}</p>
      {action && (
        <Button type="button" variant="outline" onClick={action.onClick}>
          {action.label}
        </Button>
      )}
    </div>
  );
}
