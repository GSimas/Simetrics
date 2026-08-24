import { lazy, Suspense, useMemo, useState } from 'react';

import { KpiCard } from '@/components/KpiCard';
import { SearchableSelect } from '@/components/SearchableSelect';
import { Badge } from '@/components/ui/badge';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { availableTypes, filterByEntity, optionsForType } from '@/core/search';
import { computeIndices } from '@/core/scientometrics';
import { buildProfiles, findSimilar } from '@/core/similarity';
import { mean, sum } from '@/core/stats';
import { wordFrequencies } from '@/core/wordcloud';
import { FIELD, FIELD_CANDIDATES } from '@/lib/schema';
import type { SearchEntityType } from '@/lib/types';
import { collectColumns, pickColumn, toNumeric } from '@/core/text';
import { useDataset } from '@/state/dataset.store';
import { useLocale } from '@/state/locale.store';
import { EmptyState } from '@/features/EmptyState';

const WordCloud = lazy(() => import('@/components/charts/WordCloud'));

export default function SearchTab() {
  const active = useDataset((state) => state.active);
  const searchOptions = useDataset((state) => state.searchOptions);
  const { t, locale } = useLocale();

  const [type, setType] = useState<SearchEntityType>('Autor');
  const [term, setTerm] = useState<string | null>(null);

  const types = useMemo(
    () => (searchOptions ? availableTypes(searchOptions) : []),
    [searchOptions],
  );

  const rawOptions = useMemo(
    () => (searchOptions ? optionsForType(searchOptions, type) : []),
    [searchOptions, type],
  );

  const documents = useMemo(
    () => (active && term ? filterByEntity(active, term, type) : []),
    [active, term, type],
  );

  const profiles = useMemo(() => (active ? buildProfiles(active) : null), [active]);

  const dossier = useMemo(() => {
    if (documents.length === 0) return null;

    const citations = documents.map((doc) => toNumeric(doc[FIELD.TOTAL_CITATIONS]) ?? 0);
    const years = documents
      .map((doc) => toNumeric(doc[FIELD.YEAR_CLEAN]))
      .filter((year): year is number => year !== null && Number.isFinite(year));

    return {
      totalCitations: sum(citations),
      meanCitations: mean(citations),
      indices: computeIndices(citations, years),
      timespan:
        years.length > 0
          ? `${Math.min(...years)}–${Math.max(...years)}`
          : 'N/S',
    };
  }, [documents]);

  const similar = useMemo(
    () => (profiles && term ? findSimilar(profiles, term, type) : []),
    [profiles, term, type],
  );

  const cloudWords = useMemo(() => {
    if (!active || documents.length === 0) return [];
    const keywordsColumn = pickColumn(collectColumns(active), FIELD_CANDIDATES.keywords);
    return keywordsColumn ? wordFrequencies(documents, keywordsColumn, 120) : [];
  }, [active, documents]);

  if (!active || !searchOptions) {
    return <EmptyState title={t('tab_search')} />;
  }

  const titleColumn = pickColumn(collectColumns(active), FIELD_CANDIDATES.title);

  return (
    <div className="space-y-4">
      <Card className="border-t-4 border-t-cyan-500 shadow-xs">
        <CardHeader>
          <CardTitle className="text-base font-bold text-foreground">{t('search_title')}</CardTitle>
          <CardDescription>
            {t('search_desc')}
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-3">
          <div className="grid gap-3 sm:grid-cols-[200px_1fr]">
            <div className="space-y-1.5">
              <Label htmlFor="search-type">{t('search_type_label')}</Label>
              <Select
                value={type}
                onValueChange={(value) => {
                  setType(value as SearchEntityType);
                  setTerm(null);
                }}
              >
                <SelectTrigger id="search-type" className="h-10 rounded-xl">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {types.map((option) => (
                    <SelectItem key={option} value={option}>
                      {option}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-1.5">
              <Label>{locale === 'en' ? `Select ${type}` : `Selecionar ${type}`}</Label>
              <SearchableSelect
                options={rawOptions}
                value={term}
                onChange={setTerm}
                placeholder={
                  locale === 'en'
                    ? `Click to select ${type.toLowerCase()}...`
                    : `Clique para selecionar ${type.toLowerCase()}...`
                }
                searchPlaceholder={
                  locale === 'en'
                    ? `Type to filter ${type.toLowerCase()}...`
                    : `Digite para filtrar ${type.toLowerCase()}...`
                }
                emptyText={
                  locale === 'en'
                    ? 'No matching entity found.'
                    : 'Nenhuma entidade encontrada.'
                }
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {term && dossier && (
        <>
          <Card className="border-t-4 border-t-primary shadow-xs">
            <CardHeader>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <CardTitle className="text-lg font-bold text-foreground">{term}</CardTitle>
                <Badge variant="blue" className="text-xs">
                  {type}
                </Badge>
              </div>
              <CardDescription>
                {documents.length.toLocaleString('pt-BR')}{' '}
                {locale === 'en' ? 'documents' : 'documentos'} · {dossier.timespan}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-3 sm:gap-4 lg:grid-cols-4">
                <KpiCard title={t('kpi_docs')} value={documents.length} tone="blue" />
                <KpiCard
                  title={locale === 'en' ? 'Total Citations' : 'Citações'}
                  value={dossier.totalCitations}
                  tone="amber"
                />
                <KpiCard
                  title={locale === 'en' ? 'Mean Citations' : 'Média de citações'}
                  value={Number(dossier.meanCitations.toFixed(2))}
                  tone="amber"
                />
                <KpiCard
                  title="Índice h"
                  value={dossier.indices.h}
                  tone="purple"
                  info={
                    locale === 'en'
                      ? 'h-index: Number h of papers with at least h citations each. Simultaneously measures productivity and citation impact.'
                      : 'Índice h: Número h de publicações que receberam pelo menos h citações cada. Mede simultaneamente produtividade e impacto.'
                  }
                />
                <KpiCard
                  title="Índice g"
                  value={dossier.indices.g}
                  tone="purple"
                  info={
                    locale === 'en'
                      ? 'g-index: Highest rank g where the top g papers have together at least g² citations. Gives higher weight to highly-cited papers.'
                      : 'Índice g: Maior número g tal que os g artigos mais citados receberam juntos pelo menos g² citações. Dá mais peso a artigos de alto impacto.'
                  }
                />
                <KpiCard
                  title="Índice i10"
                  value={dossier.indices.i10}
                  tone="indigo"
                  info={
                    locale === 'en'
                      ? 'i10-index: Number of publications with at least 10 citations each (Google Scholar benchmark).'
                      : 'Índice i10: Número de publicações com pelo menos 10 citações cada (padrão Google Scholar).'
                  }
                />
                <KpiCard
                  title="Índice m"
                  value={dossier.indices.m}
                  tone="indigo"
                  info={
                    locale === 'en'
                      ? 'm-quotient: h-index divided by academic career length in years (h / Δt). Enables career-stage comparisons.'
                      : 'Índice m: Índice h dividido pelos anos de atividade acadêmica (h / Δt). Permite comparar pesquisadores em diferentes momentos da carreira.'
                  }
                />
                <KpiCard title={t('kpi_docs_sub')} value={dossier.timespan} tone="cyan" />
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-4 lg:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="text-base font-bold">{t('search_similar_title')}</CardTitle>
                <CardDescription>
                  {t('search_similar_desc')}
                </CardDescription>
              </CardHeader>
              <CardContent>
                {similar.length === 0 ? (
                  <p className="text-sm text-muted-foreground">
                    {locale === 'en'
                      ? 'No entities with shared traits found.'
                      : 'Nenhuma entidade com traços em comum — ou o tipo selecionado não tem perfil comparável.'}
                  </p>
                ) : (
                  <div className="max-h-96 overflow-auto rounded-xl border">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>{locale === 'en' ? 'Entity' : 'Entidade'}</TableHead>
                          <TableHead>{locale === 'en' ? 'Similarity' : 'Similaridade'}</TableHead>
                          <TableHead>{locale === 'en' ? 'Shared Traits' : 'Traços em comum'}</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {similar.map((hit) => (
                          <TableRow key={hit.item}>
                            <TableCell>
                              <button
                                type="button"
                                className="max-w-56 truncate text-left font-medium hover:underline text-primary"
                                title={hit.item}
                                onClick={() => setTerm(hit.item)}
                              >
                                {hit.item}
                              </button>
                            </TableCell>
                            <TableCell>
                              <Badge variant="blue" className="tabular-nums font-semibold">
                                {hit.similarity}%
                              </Badge>
                            </TableCell>
                            <TableCell
                              className="max-w-64 truncate text-xs text-muted-foreground"
                              title={hit.sharedTraits}
                            >
                              {hit.sharedTraits}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-base font-bold">{t('search_lexico_title')}</CardTitle>
                <CardDescription>
                  {t('search_lexico_desc')}
                </CardDescription>
              </CardHeader>
              <CardContent>
                {cloudWords.length === 0 ? (
                  <p className="text-sm text-muted-foreground">
                    {locale === 'en'
                      ? 'No keywords associated with this entity.'
                      : 'Os documentos desta entidade não trazem palavras-chave.'}
                  </p>
                ) : (
                  <Suspense
                    fallback={
                      <div className="grid h-72 place-items-center text-sm text-muted-foreground">
                        {locale === 'en' ? 'Rendering cloud...' : 'Montando nuvem…'}
                      </div>
                    }
                  >
                    <WordCloud words={cloudWords} height={340} exportName={`nuvem-${term}`} />
                  </Suspense>
                )}
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-base font-bold">{t('search_docs_title')}</CardTitle>
              <CardDescription>
                {t('search_docs_desc')}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="max-h-[32rem] overflow-auto rounded-xl border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>{locale === 'en' ? 'Title' : 'Título'}</TableHead>
                      <TableHead>{locale === 'en' ? 'Year' : 'Ano'}</TableHead>
                      <TableHead>{locale === 'en' ? 'Citations' : 'Citações'}</TableHead>
                      <TableHead>Venue</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {[...documents]
                      .sort(
                        (left, right) =>
                          (toNumeric(right[FIELD.TOTAL_CITATIONS]) ?? 0) -
                          (toNumeric(left[FIELD.TOTAL_CITATIONS]) ?? 0),
                      )
                      .map((doc, index) => {
                        const title = titleColumn ? String(doc[titleColumn] ?? '') : '';
                        return (
                          <TableRow key={`${title}-${index}`}>
                            <TableCell className="max-w-[28rem] truncate" title={title}>
                              {title}
                            </TableCell>
                            <TableCell className="tabular-nums">
                              {toNumeric(doc[FIELD.YEAR_CLEAN]) ?? '—'}
                            </TableCell>
                            <TableCell className="tabular-nums">
                              {toNumeric(doc[FIELD.TOTAL_CITATIONS]) ?? 0}
                            </TableCell>
                            <TableCell
                              className="max-w-64 truncate text-muted-foreground"
                              title={String(doc[FIELD.SECONDARY_TITLE] ?? '')}
                            >
                              {String(doc[FIELD.SECONDARY_TITLE] ?? '—')}
                            </TableCell>
                          </TableRow>
                        );
                      })}
                  </TableBody>
                </Table>
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
