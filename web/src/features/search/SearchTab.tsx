import { lazy, Suspense, useMemo, useState } from 'react';
import {
  ExternalLink,
  FileText,
  User,
  ChevronDown,
  BookOpen,
  Globe,
  Users,
} from 'lucide-react';

import { Collapse } from '@/components/Collapse';

import { SectionTitle } from '@/components/InfoTip';
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { availableTypes, filterByEntity, optionsForType } from '@/core/search';
import { computeIndices } from '@/core/scientometrics';
import { cachedProfiles, findSimilar } from '@/core/similarity';
import { mean, sum } from '@/core/stats';
import { keywordFrequencies } from '@/core/wordcloud';
import { FIELD, FIELD_CANDIDATES } from '@/lib/schema';
import type { SearchEntityType } from '@/lib/types';
import { collectColumns, isNullLike, pickColumn, splitTokens, toNumeric } from '@/core/text';
import { useDataset } from '@/state/dataset.store';
import { useLocale } from '@/state/locale.store';
import { openInSearch, resolveEntity, useNavigation } from '@/state/navigation.store';
import { EmptyState } from '@/features/EmptyState';
import { collaborationNetwork } from '@/core/viz/collaboration';
import { cn } from '@/lib/utils';

const WordCloud = lazy(() => import('@/components/charts/WordCloud'));
const PlotlyChart = lazy(() => import('@/components/charts/PlotlyChart'));
const WorldMap = lazy(() => import('@/components/charts/WorldMap'));

/** Coautores à vista antes do "ver todos". */
const VISIBLE_COAUTHORS = 12;

function cleanDoiUrl(rawDoi: unknown): string | null {
  if (!rawDoi || typeof rawDoi !== 'string') return null;
  const trimmed = rawDoi.trim();
  if (!trimmed || isNullLike(trimmed)) return null;
  if (trimmed.startsWith('http://') || trimmed.startsWith('https://')) return trimmed;
  const clean = trimmed.replace(/^doi:\s*/i, '');
  return `https://doi.org/${clean}`;
}

export default function SearchTab() {
  const active = useDataset((state) => state.active);
  const searchOptions = useDataset((state) => state.searchOptions);
  const { t, locale } = useLocale();

  // Tipo e termo vivem no store de navegação: gráficos de outras abas abrem perfis aqui,
  // e a busca continua no lugar ao voltar para esta aba.
  const type = useNavigation((state) => state.searchType);
  const term = useNavigation((state) => state.searchTerm);
  const setType = (searchType: SearchEntityType): void => useNavigation.setState({ searchType });
  const setTerm = (searchTerm: string | null): void => useNavigation.setState({ searchTerm });
  const [expandedAbstracts, setExpandedAbstracts] = useState<Set<number>>(new Set());
  const [showAllCoauthors, setShowAllCoauthors] = useState(false);

  const renderAuthorChip = ({ author, count }: { author: string; count: number }) => (
    <button
      key={author}
      type="button"
      onClick={() => {
        setType('Autor');
        setTerm(author);
        setShowAllCoauthors(false);
      }}
      className="inline-flex cursor-pointer items-center gap-1 border border-border px-2 py-0.5 text-xs text-foreground transition-colors hover:border-highlight hover:text-highlight"
      title={locale === 'en' ? `View dossier for ${author}` : `Ver dossiê de ${author}`}
    >
      <User className="size-3" />
      <span>{author}</span>
      <span className="tabular-nums text-muted-foreground">{count}</span>
    </button>
  );

  const toggleAbstract = (index: number): void => {
    setExpandedAbstracts((prev) => {
      const next = new Set(prev);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  };

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

  const profiles = useMemo(() => (active ? cachedProfiles(active) : null), [active]);

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
    return keywordsColumn ? keywordFrequencies(documents, keywordsColumn, 120) : [];
  }, [active, documents]);

  const timelineData = useMemo(() => {
    if (documents.length === 0) return { x: [], y: [] };
    const counts = new Map<number, number>();
    for (const doc of documents) {
      const y = toNumeric(doc[FIELD.YEAR_CLEAN]);
      if (y !== null && Number.isFinite(y)) {
        counts.set(y, (counts.get(y) ?? 0) + 1);
      }
    }
    const years = [...counts.keys()].sort((a, b) => a - b);
    return {
      x: years,
      y: years.map((y) => counts.get(y) ?? 0),
    };
  }, [documents]);

  // Extrai os países vinculados a um autor selecionado
  const authorCountries = useMemo(() => {
    if (type !== 'Autor' || !active || documents.length === 0) return [];
    const countryColumn = pickColumn(collectColumns(active), FIELD_CANDIDATES.countries);
    if (!countryColumn) return [];

    const countrySet = new Set<string>();
    for (const doc of documents) {
      const countries = splitTokens(doc[countryColumn]);
      for (const c of countries) {
        if (c && !isNullLike(c)) countrySet.add(c);
      }
    }
    return [...countrySet].sort();
  }, [type, active, documents]);

  // Coautores do autor selecionado, do parceiro mais frequente ao menos frequente.
  const authorCoauthors = useMemo(() => {
    if (type !== 'Autor' || !active || !term || documents.length === 0) return [];
    const authorColumn = pickColumn(collectColumns(active), FIELD_CANDIDATES.authors);
    if (!authorColumn) return [];

    const counts = new Map<string, number>();
    const self = term.trim().toLowerCase();
    for (const doc of documents) {
      for (const author of new Set(splitTokens(doc[authorColumn]))) {
        if (!author || isNullLike(author) || author.trim().toLowerCase() === self) continue;
        counts.set(author, (counts.get(author) ?? 0) + 1);
      }
    }
    return [...counts.entries()]
      .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
      .map(([author, count]) => ({ author, count }));
  }, [type, active, term, documents]);

  // Rede de colaboração dos documentos do país, para o mapa-múndi do perfil.
  const countryNetwork = useMemo(
    () => (type === 'País' && documents.length > 0 ? collaborationNetwork(documents, 30) : null),
    [type, documents],
  );
  const countryFocus = useMemo(() => {
    if (!countryNetwork || !term) return undefined;
    const needle = term.trim().toLowerCase();
    return countryNetwork.nodes.find(
      (node) => node.country.toLowerCase() === needle || node.label.toLowerCase() === needle,
    )?.country;
  }, [countryNetwork, term]);

  // Extrai os autores vinculados a um país selecionado
  const countryAuthors = useMemo(() => {
    if (type !== 'País' || !active || documents.length === 0) return [];
    const authorColumn = pickColumn(collectColumns(active), FIELD_CANDIDATES.authors);
    if (!authorColumn) return [];

    const authorMap = new Map<string, { count: number; citations: number }>();
    for (const doc of documents) {
      const authors = splitTokens(doc[authorColumn]);
      const cites = toNumeric(doc[FIELD.TOTAL_CITATIONS]) ?? 0;
      for (const author of authors) {
        const existing = authorMap.get(author) ?? { count: 0, citations: 0 };
        existing.count += 1;
        existing.citations += cites;
        authorMap.set(author, existing);
      }
    }

    return [...authorMap.entries()]
      .map(([author, metrics]) => ({
        author,
        count: metrics.count,
        citations: metrics.citations,
      }))
      .sort((a, b) => b.count - a.count || b.citations - a.citations);
  }, [type, active, documents]);

  // Autores do documento selecionado, com quantos documentos cada um tem na base.
  const documentAuthors = useMemo(() => {
    if (type !== 'Documento' || !active || documents.length === 0) return [];
    const authorColumn = pickColumn(collectColumns(active), FIELD_CANDIDATES.authors);
    if (!authorColumn) return [];
    return [...new Set(splitTokens(documents[0]?.[authorColumn]))].map((author) => ({
      author,
      count: filterByEntity(active, author, 'Autor').length,
    }));
  }, [type, active, documents]);

  if (!active || !searchOptions) {
    return <EmptyState title={t('tab_search')} />;
  }

  const titleColumn = pickColumn(collectColumns(active), FIELD_CANDIDATES.title);
  const keywordsColumn = pickColumn(collectColumns(active), FIELD_CANDIDATES.keywords);
  const abstractColumn = pickColumn(collectColumns(active), FIELD_CANDIDATES.abstract);
  const doiColumn = pickColumn(collectColumns(active), FIELD_CANDIDATES.doi);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <SectionTitle title={t('search_title')} info={t('search_desc')} />
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
                  setExpandedAbstracts(new Set());
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
                onChange={(val) => {
                  setTerm(val);
                  setExpandedAbstracts(new Set());
                }}
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
        // Chave por perfil: trocar de entidade remonta o dossiê, que entra com um fade curto.
        <div key={`${type}:${term}`} className="space-y-4 duration-300 animate-in fade-in-0 slide-in-from-bottom-2">
          <Card className="border-t-2 border-t-highlight shadow-xs">
            <CardHeader>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <CardTitle className="text-lg font-bold text-foreground break-words">{term}</CardTitle>
                <Badge variant="blue" className="text-xs">
                  {type}
                </Badge>
              </div>
              <CardDescription>
                {documents.length.toLocaleString('pt-BR')}{' '}
                {locale === 'en' ? 'documents' : 'documentos'} · {dossier.timespan}
              </CardDescription>

              {/* Se for Autor, exibe os Países vinculados como elementos clicáveis */}
              {type === 'Autor' && authorCountries.length > 0 && (
                <div className="flex flex-wrap items-center gap-1.5 pt-1.5">
                  <span className="text-xs font-semibold text-muted-foreground flex items-center gap-1">
                    <Globe className="size-3.5 text-primary" />
                    {locale === 'en' ? 'Affiliated countries:' : 'Países vinculados:'}
                  </span>
                  {authorCountries.map((country) => (
                    <button
                      key={country}
                      type="button"
                      onClick={() => {
                        setType('País');
                        setTerm(country);
                      }}
                      className="inline-flex cursor-pointer items-center gap-1 border border-border px-2 py-0.5 text-xs text-foreground transition-colors hover:border-highlight hover:text-highlight"
                      title={locale === 'en' ? `View dossier for ${country}` : `Ver dossiê de ${country}`}
                    >
                      <Globe className="size-3" />
                      <span>{country}</span>
                    </button>
                  ))}
                </div>
              )}

              {/* Se for Autor, lista os coautores — os primeiros à vista, o resto num collapse. */}
              {type === 'Autor' && authorCoauthors.length > 0 && (
                <div className="space-y-1.5 pt-1.5">
                  <div className="flex flex-wrap items-center gap-1.5">
                    <span className="text-xs font-semibold text-muted-foreground flex items-center gap-1">
                      <Users className="size-3.5 text-primary" />
                      {t('search_coauthors_label')}
                    </span>
                    {authorCoauthors.slice(0, VISIBLE_COAUTHORS).map(renderAuthorChip)}
                    {authorCoauthors.length > VISIBLE_COAUTHORS && (
                      <button
                        type="button"
                        onClick={() => setShowAllCoauthors((open) => !open)}
                        aria-expanded={showAllCoauthors}
                        className="eyebrow inline-flex cursor-pointer items-center gap-1 px-1 transition-colors hover:text-highlight"
                      >
                        {showAllCoauthors
                          ? t('search_show_less')
                          : `${t('search_show_all')} (+${authorCoauthors.length - VISIBLE_COAUTHORS})`}
                        <ChevronDown
                          className={cn('size-3 transition-transform duration-300', showAllCoauthors && 'rotate-180')}
                        />
                      </button>
                    )}
                  </div>
                  {authorCoauthors.length > VISIBLE_COAUTHORS && (
                    <Collapse open={showAllCoauthors} delayOpen={false}>
                      <div className="flex flex-wrap gap-1.5">
                        {authorCoauthors.slice(VISIBLE_COAUTHORS).map(renderAuthorChip)}
                      </div>
                    </Collapse>
                  )}
                </div>
              )}
            </CardHeader>
            <CardContent>
              <div
                className={cn(
                  'grid grid-cols-2 gap-3 sm:gap-4',
                  type === 'Autor' ? 'lg:grid-cols-5' : 'lg:grid-cols-4',
                )}
              >
                <KpiCard title={t('kpi_docs')} value={documents.length} tone="blue" />
                {type === 'Autor' && (
                  <KpiCard title={t('search_coauthors')} value={authorCoauthors.length} tone="cyan" />
                )}
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

              {/* Se for Documento selecionado, exibe Abstract, Palavras-chave e DOI em destaque na apresentação */}
              {type === 'Documento' && documents[0] && (
                (() => {
                  const selectedDoc = documents[0];
                  const rawSec = selectedDoc[FIELD.SECONDARY_TITLE];
                  const secTitle = rawSec ? String(rawSec) : '';
                  const rawDoi = doiColumn ? selectedDoc[doiColumn] : selectedDoc[FIELD.DOI];
                  const doiUrl = cleanDoiUrl(rawDoi);
                  const rawKw = keywordsColumn ? selectedDoc[keywordsColumn] : selectedDoc[FIELD.KEYWORDS];
                  const kwStr = rawKw ? String(rawKw).trim() : '';
                  const rawAbs = abstractColumn ? selectedDoc[abstractColumn] : selectedDoc[FIELD.ABSTRACT];
                  const absStr = rawAbs ? String(rawAbs).trim() : '';

                  return (
                    <div className="mt-4 space-y-3 pt-3 border-t border-border/70">
                      <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                        {secTitle && !isNullLike(secTitle) && (
                          <span className="font-semibold text-foreground flex items-center gap-1.5">
                            <BookOpen className="size-4 text-primary" />
                            <span>{secTitle}</span>
                          </span>
                        )}
                        {doiUrl && (
                          <a
                            href={doiUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-1.5 rounded-lg border border-cyan-300 bg-cyan-50 px-3 py-1 text-xs font-semibold text-cyan-800 hover:bg-cyan-100 dark:border-cyan-800 dark:bg-cyan-950/60 dark:text-cyan-300 transition-colors cursor-pointer"
                          >
                            <ExternalLink className="size-3.5" />
                            {t('search_doi_link')}
                          </a>
                        )}
                      </div>

                      {documentAuthors.length > 0 && (
                        <div className="space-y-1">
                          <span className="text-xs font-semibold text-muted-foreground flex items-center gap-1">
                            <Users className="size-3.5" />
                            {t('search_doc_authors')}:
                          </span>
                          <div className="flex flex-wrap gap-1.5">
                            {documentAuthors.map(({ author, count }) => renderAuthorChip({ author, count }))}
                          </div>
                        </div>
                      )}

                      {kwStr && !isNullLike(kwStr) && (
                        <div className="space-y-1">
                          <span className="text-xs font-semibold text-muted-foreground flex items-center gap-1">
                            <BookOpen className="size-3.5" />
                            {t('search_keywords')}:
                          </span>
                          <div className="flex flex-wrap gap-1.5">
                            {[...new Set(splitTokens(kwStr))].map((kw) => (
                              <button
                                key={kw}
                                type="button"
                                onClick={() => {
                                  setType('Palavra-chave');
                                  setTerm(kw);
                                }}
                                className="inline-flex cursor-pointer items-center border border-border px-2 py-0.5 text-xs text-foreground transition-colors hover:border-highlight hover:text-highlight"
                                title={locale === 'en' ? `Search keyword: ${kw}` : `Buscar palavra-chave: ${kw}`}
                              >
                                {kw}
                              </button>
                            ))}
                          </div>
                        </div>
                      )}

                      {absStr && !isNullLike(absStr) && (
                        <div className="space-y-1">
                          <span className="text-xs font-semibold text-muted-foreground flex items-center gap-1">
                            <FileText className="size-3.5" />
                            {t('search_abstract')}:
                          </span>
                          <div className="rounded-xl border border-border/80 bg-muted/30 p-4 text-xs text-foreground leading-relaxed">
                            {absStr}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })()
              )}
            </CardContent>
          </Card>

          <div className="grid gap-4 lg:grid-cols-2">
            {/* Lado Esquerdo: Se for País mostra Autores do País; se for outra entidade mostra Entidades Semelhantes */}
            {type === 'País' ? (
              <Card>
                <CardHeader>
                  <SectionTitle title={t('search_country_authors')} info={t('search_country_authors_desc')} />
                </CardHeader>
                <CardContent>
                  {countryAuthors.length === 0 ? (
                    <p className="text-sm text-muted-foreground">
                      {locale === 'en' ? 'No authors found for this country.' : 'Nenhum autor encontrado para este país.'}
                    </p>
                  ) : (
                    <div className="max-h-96 overflow-auto rounded-xl border">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>{locale === 'en' ? 'Author' : 'Autor'}</TableHead>
                            <TableHead className="text-right">{locale === 'en' ? 'Papers' : 'Artigos'}</TableHead>
                            <TableHead className="text-right">{locale === 'en' ? 'Citations' : 'Citações'}</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {countryAuthors.map(({ author, count, citations }) => (
                            <TableRow key={author}>
                              <TableCell>
                                <button
                                  type="button"
                                  className="max-w-56 truncate text-left font-medium hover:underline text-primary cursor-pointer flex items-center gap-1.5"
                                  title={author}
                                  onClick={() => {
                                    setType('Autor');
                                    setTerm(author);
                                  }}
                                >
                                  <User className="size-3.5 text-muted-foreground shrink-0" />
                                  <span className="truncate">{author}</span>
                                </button>
                              </TableCell>
                              <TableCell className="text-right tabular-nums font-semibold">
                                {count.toLocaleString('pt-BR')}
                              </TableCell>
                              <TableCell className="text-right tabular-nums text-muted-foreground">
                                {citations.toLocaleString('pt-BR')}
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </div>
                  )}
                </CardContent>
              </Card>
            ) : (
              <Card>
                <CardHeader>
                  <SectionTitle title={t('search_similar_title')} info={t('search_similar_desc')} />
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
                                  className="max-w-56 truncate text-left font-medium hover:underline text-primary cursor-pointer"
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
            )}

            {/* Lado Direito: Tabs com Lexicometria (Nuvem de Palavras) e Produção Histórica */}
            <Card>
              <CardHeader>
                <SectionTitle title={t('search_lexico_title')} info={t('search_lexico_desc')} />
              </CardHeader>
              <CardContent>
                <Tabs defaultValue="cloud" className="w-full">
                  <TabsList className="mb-3 w-full justify-start">
                    <TabsTrigger value="cloud" className="text-xs sm:text-sm">
                      {t('search_tab_cloud')}
                    </TabsTrigger>
                    <TabsTrigger value="timeline" className="text-xs sm:text-sm">
                      {t('search_tab_timeline')}
                    </TabsTrigger>
                    {type === 'País' && (
                      <TabsTrigger value="map" className="text-xs sm:text-sm">
                        {t('search_tab_map')}
                      </TabsTrigger>
                    )}
                  </TabsList>

                  <TabsContent value="cloud">
                    {cloudWords.length === 0 ? (
                      <p className="text-sm text-muted-foreground py-8 text-center">
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
                        <WordCloud
                          words={cloudWords}
                          height={320}
                          exportName={`nuvem-${term}`}
                          onWordClick={(word) => openInSearch(word, ['Palavra-chave'])}
                          isClickable={(word) => resolveEntity(word, ['Palavra-chave']) !== null}
                        />
                      </Suspense>
                    )}
                  </TabsContent>

                  <TabsContent value="timeline">
                    {timelineData.x.length === 0 ? (
                      <p className="text-sm text-muted-foreground py-8 text-center">
                        {locale === 'en'
                          ? 'No publication years available.'
                          : 'Anos de publicação não disponíveis para estes documentos.'}
                      </p>
                    ) : (
                      <Suspense
                        fallback={
                          <div className="grid h-72 place-items-center text-sm text-muted-foreground">
                            {locale === 'en' ? 'Rendering chart...' : 'Carregando gráfico…'}
                          </div>
                        }
                      >
                        <PlotlyChart
                          data={[
                            {
                              x: timelineData.x,
                              y: timelineData.y,
                              type: 'bar',
                              marker: { color: '#3FAE8F' },
                              name: t('search_timeline_docs'),
                            },
                          ]}
                          layout={{
                            title: { text: t('search_timeline_title'), font: { size: 13 } },
                            xaxis: { title: { text: locale === 'en' ? 'Year' : 'Ano' }, dtick: 1 },
                            yaxis: { title: { text: t('search_timeline_docs') }, rangemode: 'tozero' },
                            margin: { l: 45, r: 20, t: 35, b: 40 },
                          }}
                          height={320}
                          exportName={`producao-historica-${term}`}
                        />
                      </Suspense>
                    )}
                  </TabsContent>

                  {type === 'País' && (
                    <TabsContent value="map">
                      {countryNetwork && countryNetwork.nodes.length > 0 ? (
                        <Suspense
                          fallback={
                            <div className="grid h-72 place-items-center text-sm text-muted-foreground">
                              {locale === 'en' ? 'Loading map…' : 'Carregando mapa…'}
                            </div>
                          }
                        >
                          <WorldMap
                            key={term ?? ''}
                            nodes={countryNetwork.nodes.map((node) => ({
                              key: node.country,
                              label: node.label,
                              documents: node.documents,
                              latitude: node.latitude,
                              longitude: node.longitude,
                            }))}
                            edges={countryNetwork.edges}
                            focus={countryFocus}
                            onOpenProfile={(key) =>
                              openInSearch(
                                countryNetwork.nodes.find((node) => node.country === key)?.label ?? key,
                                ['País'],
                              )
                            }
                            exportName={`mapa-${term}`}
                          />
                        </Suspense>
                      ) : (
                        <p className="text-sm text-muted-foreground py-8 text-center">{t('map_no_data')}</p>
                      )}
                    </TabsContent>
                  )}
                </Tabs>
              </CardContent>
            </Card>

          {/* Documento já mostra autores e palavras-chave no topo; as demais entidades listam seus documentos. */}
          {type !== 'Documento' && (
            <Card className="lg:col-span-2">
              <CardHeader>
                <SectionTitle title={t('search_docs_title')} info={t('search_docs_desc')} />
              </CardHeader>
              <CardContent>
                <div className="max-h-[36rem] overflow-auto rounded-xl border">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>{locale === 'en' ? 'Title & Details' : 'Título e Detalhes'}</TableHead>
                        <TableHead className="w-24 text-center">{locale === 'en' ? 'Year' : 'Ano'}</TableHead>
                        <TableHead className="w-24 text-center">{locale === 'en' ? 'Citations' : 'Citações'}</TableHead>
                        <TableHead className="w-48">Venue</TableHead>
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
                          const title = titleColumn ? String(doc[titleColumn] ?? '').trim() : '';
                          const keywords = keywordsColumn ? String(doc[keywordsColumn] ?? '').trim() : '';
                          const abstract = abstractColumn ? String(doc[abstractColumn] ?? '').trim() : '';
                          const rawDoi = doiColumn ? doc[doiColumn] : doc[FIELD.DOI];
                          const doiUrl = cleanDoiUrl(rawDoi);
                          const isExpanded = expandedAbstracts.has(index);

                          return (
                            <TableRow key={`${title}-${index}`} className="align-top">
                              <TableCell className="space-y-2 py-3">
                                <div className="flex flex-wrap items-center gap-2">
                                  <button
                                    type="button"
                                    className="text-left font-semibold text-primary hover:underline cursor-pointer break-words leading-snug"
                                    title={title}
                                    onClick={() => {
                                      setType('Documento');
                                      setTerm(title);
                                    }}
                                  >
                                    {title || '—'}
                                  </button>

                                  {doiUrl && (
                                    <a
                                      href={doiUrl}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="inline-flex items-center gap-1 rounded-md border border-cyan-300 bg-cyan-50 px-2 py-0.5 text-[11px] font-medium text-cyan-800 hover:bg-cyan-100 dark:border-cyan-800 dark:bg-cyan-950/60 dark:text-cyan-300"
                                      title={doiUrl}
                                    >
                                      <ExternalLink className="size-3" />
                                      {t('search_doi_link')}
                                    </a>
                                  )}
                                </div>

                                {keywords && !isNullLike(keywords) && (
                                  <div className="flex flex-wrap items-center gap-1.5">
                                    <span className="text-[11px] font-semibold text-muted-foreground flex items-center gap-1">
                                      <BookOpen className="size-3" />
                                      {t('search_keywords')}:
                                    </span>
                                    {[...new Set(splitTokens(keywords))].slice(0, 6).map((kw) => (
                                      <Badge key={kw} variant="secondary" className="text-[10px] px-1.5 py-0">
                                        {kw}
                                      </Badge>
                                    ))}
                                  </div>
                                )}

                                {abstract && !isNullLike(abstract) && (
                                  <div className="pt-1">
                                    <button
                                      type="button"
                                      onClick={() => toggleAbstract(index)}
                                      aria-expanded={isExpanded}
                                      className="inline-flex items-center gap-1 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground cursor-pointer"
                                    >
                                      <FileText className="size-3" />
                                      <span>{t('search_abstract')}</span>
                                      <ChevronDown
                                        className={cn(
                                          'size-3 transition-transform duration-300',
                                          isExpanded && 'rotate-180',
                                        )}
                                      />
                                    </button>

                                    {/* Abre animando a altura, sem saltar o conteúdo abaixo. */}
                                    <Collapse open={isExpanded} delayOpen={false}>
                                      <p className="mt-1.5 border border-border/60 bg-muted/40 p-2.5 text-xs text-muted-foreground leading-relaxed">
                                        {abstract}
                                      </p>
                                    </Collapse>
                                  </div>
                                )}
                              </TableCell>
                              <TableCell className="text-center tabular-nums font-medium py-3">
                                {toNumeric(doc[FIELD.YEAR_CLEAN]) ?? '—'}
                              </TableCell>
                              <TableCell className="text-center tabular-nums font-semibold text-foreground py-3">
                                {toNumeric(doc[FIELD.TOTAL_CITATIONS]) ?? 0}
                              </TableCell>
                              <TableCell
                                className="max-w-48 truncate text-xs text-muted-foreground py-3"
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
          )}
          </div>
        </div>
      )}
    </div>
  );
}
