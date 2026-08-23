import { lazy, Suspense, useDeferredValue, useMemo, useState } from 'react';
import { Search } from 'lucide-react';

import { KpiCard } from '@/components/KpiCard';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Input } from '@/components/ui/input';
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
import { EmptyState } from '@/features/EmptyState';

const WordCloud = lazy(() => import('@/components/charts/WordCloud'));

/** Quantas opções o seletor lista antes de exigir refinamento do filtro. */
const MAX_LISTED_OPTIONS = 200;

export default function SearchTab() {
  const active = useDataset((state) => state.active);
  const searchOptions = useDataset((state) => state.searchOptions);

  const [type, setType] = useState<SearchEntityType>('Autor');
  const [query, setQuery] = useState('');
  const [term, setTerm] = useState<string | null>(null);

  // O filtro digitado percorre milhares de opções; adiar a lista mantém a digitação
  // fluida enquanto a filtragem acompanha em segundo plano.
  const deferredQuery = useDeferredValue(query);

  const types = useMemo(
    () => (searchOptions ? availableTypes(searchOptions) : []),
    [searchOptions],
  );

  const options = useMemo(() => {
    if (!searchOptions) return [];
    const all = optionsForType(searchOptions, type);
    if (!deferredQuery.trim()) return all.slice(0, MAX_LISTED_OPTIONS);

    const needle = deferredQuery.toLowerCase();
    return all.filter((option) => option.toLowerCase().includes(needle)).slice(0, MAX_LISTED_OPTIONS);
  }, [searchOptions, type, deferredQuery]);

  // Perfis de similaridade custam uma varredura completa da base, então são construídos
  // uma vez por dataset e reaproveitados a cada consulta.
  const profiles = useMemo(() => (active ? buildProfiles(active) : null), [active]);

  const documents = useMemo(
    () => (active && term ? filterByEntity(active, term, type) : []),
    [active, term, type],
  );

  const dossier = useMemo(() => {
    if (documents.length === 0) return null;

    const citations = documents.map((doc) => toNumeric(doc[FIELD.TOTAL_CITATIONS]) ?? 0);
    const years = documents.map((doc) => doc[FIELD.YEAR_CLEAN]);
    const indices = computeIndices(citations, years);

    const validYears = years
      .map((year) => toNumeric(year))
      .filter((year): year is number => year !== null);

    return {
      indices,
      totalCitations: sum(citations),
      meanCitations: mean(citations),
      timespan:
        validYears.length > 0
          ? `${Math.min(...validYears)}–${Math.max(...validYears)}`
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
    return <EmptyState title="Motor de Busca e Dossiê Científico" />;
  }

  const titleColumn = pickColumn(collectColumns(active), FIELD_CANDIDATES.title);

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Motor de busca</CardTitle>
          <CardDescription>
            Escolha uma entidade para montar seu dossiê: produção, impacto, documentos e
            perfis semelhantes.
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-3">
          <div className="grid gap-3 sm:grid-cols-[220px_1fr]">
            <div className="space-y-1.5">
              <Label htmlFor="search-type">Tipo</Label>
              <Select
                value={type}
                onValueChange={(value) => {
                  setType(value as SearchEntityType);
                  setTerm(null);
                  setQuery('');
                }}
              >
                <SelectTrigger id="search-type">
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
              <Label htmlFor="search-query">Buscar</Label>
              <div className="relative">
                <Search
                  className="pointer-events-none absolute left-2.5 top-2.5 size-4 text-muted-foreground"
                  aria-hidden
                />
                <Input
                  id="search-query"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder={`Digite para filtrar ${type.toLowerCase()}…`}
                  className="pl-8"
                />
              </div>
            </div>
          </div>

          <div className="flex flex-wrap gap-1.5">
            {options.length === 0 ? (
              <p className="text-sm text-muted-foreground">Nenhuma opção encontrada.</p>
            ) : (
              options.map((option) => (
                <Button
                  key={option}
                  size="sm"
                  variant={term === option ? 'default' : 'outline'}
                  className="h-7 max-w-96 justify-start truncate text-xs font-normal"
                  title={option}
                  onClick={() => setTerm(option)}
                >
                  {option}
                </Button>
              ))
            )}
          </div>

          {options.length === MAX_LISTED_OPTIONS && (
            <p className="text-xs text-muted-foreground">
              Exibindo as primeiras {MAX_LISTED_OPTIONS} opções — refine o filtro para ver
              outras.
            </p>
          )}
        </CardContent>
      </Card>

      {term && dossier && (
        <>
          <Card>
            <CardHeader>
              <CardTitle className="text-base">{term}</CardTitle>
              <CardDescription>
                {documents.length.toLocaleString('pt-BR')} documentos · {dossier.timespan}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
                <KpiCard title="Documentos" value={documents.length} tone="accent" />
                <KpiCard title="Citações" value={dossier.totalCitations} />
                <KpiCard
                  title="Média de citações"
                  value={Number(dossier.meanCitations.toFixed(2))}
                />
                <KpiCard title="Índice h" value={dossier.indices.h} />
                <KpiCard title="Índice g" value={dossier.indices.g} />
                <KpiCard title="Índice i10" value={dossier.indices.i10} />
                <KpiCard title="Índice m" value={dossier.indices.m} />
                <KpiCard title="Período" value={dossier.timespan} />
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-4 lg:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Entidades semelhantes</CardTitle>
                <CardDescription>
                  Similaridade de Jaccard sobre o &ldquo;DNA acadêmico&rdquo;: palavras-chave,
                  coautores e veículos em comum.
                </CardDescription>
              </CardHeader>
              <CardContent>
                {similar.length === 0 ? (
                  <p className="text-sm text-muted-foreground">
                    Nenhuma entidade com traços em comum — ou o tipo selecionado não tem
                    perfil comparável.
                  </p>
                ) : (
                  <div className="max-h-96 overflow-auto rounded-md border">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Entidade</TableHead>
                          <TableHead>Similaridade</TableHead>
                          <TableHead>Traços em comum</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {similar.map((hit) => (
                          <TableRow key={hit.item}>
                            <TableCell>
                              <button
                                type="button"
                                className="max-w-56 truncate text-left font-medium hover:underline"
                                title={hit.item}
                                onClick={() => setTerm(hit.item)}
                              >
                                {hit.item}
                              </button>
                            </TableCell>
                            <TableCell>
                              <Badge variant="secondary" className="tabular-nums">
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
                <CardTitle className="text-base">Lexicometria</CardTitle>
                <CardDescription>
                  Palavras-chave mais frequentes nos documentos desta entidade.
                </CardDescription>
              </CardHeader>
              <CardContent>
                {cloudWords.length === 0 ? (
                  <p className="text-sm text-muted-foreground">
                    Os documentos desta entidade não trazem palavras-chave.
                  </p>
                ) : (
                  <Suspense
                    fallback={
                      <div className="grid h-72 place-items-center text-sm text-muted-foreground">
                        Montando nuvem…
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
              <CardTitle className="text-base">Documentos</CardTitle>
              <CardDescription>
                Ordenados por citações, do mais citado ao menos citado.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="max-h-[32rem] overflow-auto rounded-md border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Título</TableHead>
                      <TableHead>Ano</TableHead>
                      <TableHead>Citações</TableHead>
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
