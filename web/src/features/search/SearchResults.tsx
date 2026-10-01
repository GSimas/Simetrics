import { Fragment, useMemo } from 'react';
import { ExternalLink } from 'lucide-react';

import { matchEntities } from '@/core/search';
import { collectColumns, isNullLike, pickColumn, splitTokens, toNumeric } from '@/core/text';
import { entityTypeLabel, numberLocale } from '@/lib/i18n/labels';
import { FIELD, FIELD_CANDIDATES } from '@/lib/schema';
import type { Dataset, SimetricsDoc } from '@/lib/types';
import { identityKey, useAsyncResult } from '@/lib/use-async-result';
import { cn } from '@/lib/utils';
import { useDataset } from '@/state/dataset.store';
import { useLocale } from '@/state/locale.store';
import { useNavigation } from '@/state/navigation.store';
import { getAiWorker } from '@/workers/client';

import { cleanDoiUrl } from './doi';

/** Trecho do resumo exibido sob cada resultado, em caracteres. */
const SNIPPET = 260;

/** Termos da consulta a destacar: palavras de 3+ letras, escapadas para a regex. */
function highlighter(query: string): RegExp | null {
  const words = query
    .split(/\s+/)
    .filter((word) => word.length >= 3)
    .map((word) => word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  // Palavra inteira, como o índice tokeniza: "man" não acende dentro de "management".
  return words.length > 0 ? new RegExp(`(?<![\\p{L}\\p{N}])(${words.join('|')})(?![\\p{L}\\p{N}])`, 'giu') : null;
}

function Highlight({ text, pattern }: { text: string; pattern: RegExp | null }) {
  if (!pattern) return <>{text}</>;
  // split com grupo de captura: as posições ímpares são os trechos que casaram.
  return (
    <>
      {text.split(pattern).map((part, index) =>
        index % 2 === 1 ? (
          <mark key={index} className="bg-transparent font-semibold text-foreground">
            {part}
          </mark>
        ) : (
          <Fragment key={index}>{part}</Fragment>
        ),
      )}
    </>
  );
}

/** Janela do resumo em volta da primeira ocorrência de um termo, como num buscador. */
function snippet(text: string, pattern: RegExp | null): string {
  if (text.length <= SNIPPET) return text;
  const at = pattern ? text.search(pattern) : -1;
  const start = at > 80 ? at - 80 : 0;
  return `${start > 0 ? '…' : ''}${text.slice(start, start + SNIPPET).trim()}…`;
}

/**
 * Resultados de uma consulta: as entidades com esse nome (atalho para o dossiê) e os
 * documentos mais relevantes por BM25 — o mesmo índice que o assistente usa.
 */
export function SearchResults({ active }: { active: Dataset }) {
  const { t, locale } = useLocale();
  const query = useNavigation((state) => state.searchQuery);
  const selectEntity = useNavigation((state) => state.selectEntity);
  const searchOptions = useDataset((state) => state.searchOptions);
  const nf = numberLocale(locale);

  const { data: hits, loading } = useAsyncResult(`search ${identityKey(active)} ${query}`, () =>
    getAiWorker().searchDocuments(active, query),
  );
  const entities = useMemo(
    () => (searchOptions ? matchEntities(searchOptions, query) : []),
    [searchOptions, query],
  );
  const pattern = useMemo(() => highlighter(query), [query]);

  const columns = useMemo(() => {
    const all = collectColumns(active);
    return {
      title: pickColumn(all, FIELD_CANDIDATES.title),
      authors: pickColumn(all, FIELD_CANDIDATES.authors),
      venue: pickColumn(all, FIELD_CANDIDATES.venue),
      doi: pickColumn(all, FIELD_CANDIDATES.doi),
    };
  }, [active]);

  const field = (doc: SimetricsDoc, column: string | null): string => {
    const value = column ? String(doc[column] ?? '').trim() : '';
    return isNullLike(value) ? '' : value;
  };

  return (
    <div
      data-tour="search-results"
      className={cn('max-w-3xl space-y-8 transition-opacity duration-200', loading && 'opacity-60')}
      aria-busy={loading}
    >
      {entities.length > 0 && (
        <section className="space-y-2.5">
          <h4 className="eyebrow">{t('search_entities')}</h4>
          <div className="flex flex-wrap gap-1.5">
            {entities.map((match) => (
              <button
                key={`${match.type}:${match.term}`}
                type="button"
                onClick={() => selectEntity(match.type, match.term)}
                className="inline-flex max-w-full cursor-pointer items-center gap-2 border border-border px-2.5 py-1 text-xs text-foreground transition-colors hover:border-highlight hover:text-highlight"
              >
                <span className="truncate">{match.term}</span>
                <span className="eyebrow shrink-0 text-[9px]">
                  {match.type === 'Local de Publicação (Venue)' ? 'Venue' : entityTypeLabel(match.type, locale)}
                </span>
              </button>
            ))}
          </div>
        </section>
      )}

      {hits && (
        <section className="space-y-5">
          <h4 className="eyebrow" role="status">
            {hits.length > 0
              ? t('search_results_count').replace('{count}', hits.length.toLocaleString(nf))
              : t('search_no_results')}
          </h4>
          <ol className="space-y-6">
            {hits.map(({ doc: position }) => {
              const doc = active[position];
              if (!doc) return null;
              const title = field(doc, columns.title);
              const authors = columns.authors ? splitTokens(doc[columns.authors]) : [];
              const venue = field(doc, columns.venue);
              const year = toNumeric(doc[FIELD.YEAR_CLEAN]);
              const citations = toNumeric(doc[FIELD.TOTAL_CITATIONS]) ?? 0;
              const abstract = field(doc, FIELD.ABSTRACT);
              const doiUrl = cleanDoiUrl(columns.doi ? doc[columns.doi] : doc[FIELD.DOI]);
              const byline = authors.slice(0, 3).join('; ') + (authors.length > 3 ? ' et al.' : '');
              const meta = [byline, year, venue]
                .filter(Boolean)
                .join(' · ');
              return (
                <li key={position} className="space-y-1">
                  <p className="truncate text-xs text-muted-foreground" title={meta}>
                    {meta}
                  </p>
                  <button
                    type="button"
                    disabled={!title}
                    onClick={() => selectEntity('Documento', title)}
                    className="cursor-pointer text-left text-lg font-medium leading-snug text-foreground underline-offset-4 transition-colors hover:text-highlight hover:underline focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-ring"
                  >
                    <Highlight text={title || '—'} pattern={pattern} />
                  </button>
                  {abstract && (
                    <p className="text-sm leading-relaxed text-muted-foreground">
                      <Highlight text={snippet(abstract, pattern)} pattern={pattern} />
                    </p>
                  )}
                  <p className="eyebrow flex flex-wrap items-center gap-3 pt-0.5">
                    <span className="tabular-nums">
                      {citations.toLocaleString(nf)} {t('search_citations')}
                    </span>
                    {doiUrl && (
                      <a
                        href={doiUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1 transition-colors hover:text-highlight"
                      >
                        DOI <ExternalLink className="size-3" aria-hidden />
                      </a>
                    )}
                  </p>
                </li>
              );
            })}
          </ol>
        </section>
      )}
    </div>
  );
}
