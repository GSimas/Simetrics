import { lazy, Suspense, useMemo, type ReactNode } from 'react';

import { ErrorBoundary } from '@/components/ErrorBoundary';
import { interpolateColor } from '@/components/charts/svg/scale';
import { CONTINENTS, continentOf, type Continent } from '@/core/continents';
import type { CooccurrenceReport } from '@/core/graph';
import type { EntityRow } from '@/core/tables';
import type { BoxplotSeries } from '@/core/viz/boxplot';
import type { CollaborationNetwork } from '@/core/viz/collaboration';
import type { ConceptTerm } from '@/core/viz/concept-map';
import type { KeywordGenetics } from '@/core/viz/genetics';
import type { HistoriographData } from '@/core/viz/historiograph';
import type { SankeyData } from '@/core/viz/sankey';
import type { ThematicMap } from '@/core/viz/thematic-map';
import { collectColumns, pickColumn, toNumeric } from '@/core/text';
import { rankingSvg } from '@/features/overview/ranking-export';
import { VISUAL_COPY } from '@/features/overview/visual-copy';
import { CITATION_SCALE_LIGHT, PALETTE, communityColor } from '@/features/overview/viz-shared';
import { numberLocale } from '@/lib/i18n/labels';
import type { TranslationKey } from '@/lib/i18n/translations';
import { FIELD, FIELD_CANDIDATES } from '@/lib/schema';
import type { Dataset } from '@/lib/types';
import { identityKey, useAsyncResult } from '@/lib/use-async-result';
import { useLocale } from '@/state/locale.store';
import { getAnalyticsWorker } from '@/workers/client';

import type { LegendItem } from './capture';
import { NetworkSvg } from './NetworkSvg';
import { reportLabel, type ReportChartId } from './report-catalog';

/**
 * Gráficos da prévia do relatório — os mesmos componentes das abas, com os mesmos dados
 * e os ajustes padrão de cada painel. O que aparece aqui é o que vai para o documento
 * (ver `capture.ts`); cada figura só calcula seus dados quando está selecionada.
 */

const TimeSeriesChart = lazy(() => import('@/components/charts/TimeSeriesChart'));
const WorldMap = lazy(() => import('@/components/charts/WorldMap'));
const RadialGraph = lazy(() => import('@/components/charts/RadialGraph'));
const WordCloud = lazy(() => import('@/components/charts/WordCloud'));
const SankeyChart = lazy(() => import('@/components/charts/SankeyChart'));
const BoxPlotChart = lazy(() => import('@/components/charts/BoxPlotChart'));
const ScatterChart = lazy(() => import('@/components/charts/ScatterChart'));
const Scatter3DChart = lazy(() => import('@/components/charts/Scatter3DChart'));
const LotkaChart = lazy(() => import('@/components/charts/LotkaChart'));

type FigureState = 'loading' | 'ready' | 'empty';

function Figure({
  id,
  state,
  legend,
  children,
}: {
  id: ReportChartId;
  state: FigureState;
  legend?: readonly LegendItem[] | undefined;
  children?: ReactNode;
}) {
  const locale = useLocale((s) => s.locale);
  const placeholder = <div className="h-56 animate-pulse bg-muted/60" aria-busy="true" />;
  return (
    <figure
      data-report-chart={id}
      data-state={state}
      data-legend={legend && legend.length > 0 ? JSON.stringify(legend) : undefined}
      className="space-y-2"
    >
      <figcaption className="eyebrow text-muted-foreground">{reportLabel(id, locale)}</figcaption>
      {state === 'loading' ? (
        placeholder
      ) : state === 'empty' ? (
        <p className="text-xs text-muted-foreground">
          {locale === 'en' ? 'Not enough data for this chart.' : 'Dados insuficientes para este gráfico.'}
        </p>
      ) : (
        <Suspense fallback={placeholder}>{children}</Suspense>
      )}
    </figure>
  );
}

const stateOf = <T,>(data: T | null | undefined, loading: boolean, hasData: (value: T) => boolean): FigureState =>
  loading || data === undefined ? 'loading' : data === null || !hasData(data) ? 'empty' : 'ready';

/** Resultado do worker amarrado à base (e só calculado quando a figura está na folha). */
function useWorker<T>(dataset: Dataset, key: string, compute: () => Promise<T>) {
  return useAsyncResult<T>(`report ${key} ${identityKey(dataset)}`, compute);
}

// ---------------------------------------------------------------------------------------
// Produção e rankings

export function ProductionFigure({ docsPerYear }: { docsPerYear: { year: number; count: number }[] | undefined }) {
  const { t } = useLocale();
  const state = docsPerYear === undefined ? 'loading' : docsPerYear.length === 0 ? 'empty' : 'ready';
  return (
    <Figure id="production" state={state}>
      <TimeSeriesChart
        exportName="producao"
        mode="bars-grouped"
        xLabel={t('prod_axis_year')}
        yLabel={t('prod_axis_docs')}
        unit={t('prod_unit_docs')}
        height={360}
        series={[
          {
            name: 'Total',
            color: PALETTE[0] as string,
            points: (docsPerYear ?? []).map((point) => ({ x: point.year, y: point.count })),
          },
        ]}
      />
    </Figure>
  );
}

interface RankRow {
  label: string;
  value: number;
  detail?: string | undefined;
}

/** Top 10 por soma de citações — a métrica padrão do painel de rankings. */
const topByCitations = (rows: readonly EntityRow[]): RankRow[] =>
  [...rows]
    .sort((a, b) => b.citations - a.citations || b.docCount - a.docCount)
    .slice(0, 10)
    .map((row) => ({ label: row.entity, value: row.citations }));

function topDocuments(dataset: Dataset): RankRow[] {
  const titleColumn = pickColumn(collectColumns(dataset), FIELD_CANDIDATES.title);
  if (!titleColumn) return [];
  return dataset
    .map((doc) => {
      const year = toNumeric(doc[FIELD.YEAR_CLEAN]);
      return {
        label: String(doc[titleColumn] ?? '').trim(),
        value: toNumeric(doc[FIELD.TOTAL_CITATIONS]) ?? 0,
        detail: year === null ? undefined : String(Math.trunc(year)),
      };
    })
    .filter((row) => row.label.length > 0)
    .sort((a, b) => b.value - a.value)
    .slice(0, 10);
}

const RANKING_TITLE: Record<'rankingAuthors' | 'rankingCountries' | 'rankingVenues' | 'rankingDocuments', TranslationKey> = {
  rankingAuthors: 'top_authors',
  rankingCountries: 'top_countries',
  rankingVenues: 'top_venues',
  rankingDocuments: 'top_documents',
};

export function RankingFigure({
  id,
  rows,
  dataset,
}: {
  id: keyof typeof RANKING_TITLE;
  rows: readonly EntityRow[] | undefined;
  dataset: Dataset;
}) {
  const { t, locale } = useLocale();
  const items = useMemo(
    () => (id === 'rankingDocuments' ? topDocuments(dataset) : rows ? topByCitations(rows) : undefined),
    [id, rows, dataset],
  );
  const svg = useMemo(() => {
    if (!items || items.length === 0) return '';
    const max = Math.max(...items.map((item) => item.value));
    return rankingSvg(
      `${t('top_title')} — ${t(RANKING_TITLE[id])}`,
      t('top_metric_cit_sum'),
      items.map((item) => ({
        label: item.label,
        detail: item.detail,
        value: item.value.toLocaleString(numberLocale(locale)),
        ratio: max > 0 ? item.value / max : 0,
      })),
    ).svg.replace('<svg ', `<svg role="img" aria-label="${reportLabel(id, locale)}" `);
  }, [items, id, t, locale]);
  const state = items === undefined ? 'loading' : items.length === 0 ? 'empty' : 'ready';
  return (
    <Figure id={id} state={state}>
      {/* SVG montado pelo próprio app (sem conteúdo externo): as cores seguem a folha. */}
      <div className="[&>svg]:h-auto [&>svg]:w-full" dangerouslySetInnerHTML={{ __html: svg }} />
    </Figure>
  );
}

// ---------------------------------------------------------------------------------------
// Geografia

export function WorldMapFigure({ collaboration }: { collaboration: CollaborationNetwork | null | undefined }) {
  const state = stateOf(collaboration, false, (value) => value.nodes.length > 0);
  return (
    <Figure id="worldMap" state={state}>
      {collaboration && (
        <WorldMap
          exportName="mapa"
          nodes={collaboration.nodes.map((node) => ({
            key: node.country,
            label: node.label,
            documents: node.documents,
            latitude: node.latitude,
            longitude: node.longitude,
          }))}
          edges={collaboration.edges.map((edge) => ({ source: edge.source, target: edge.target, documents: edge.documents }))}
        />
      )}
    </Figure>
  );
}

const CONTINENT_KEYS = {
  Africa: 'continent_Africa',
  'North America': 'continent_North_America',
  'South America': 'continent_South_America',
  Asia: 'continent_Asia',
  Europe: 'continent_Europe',
  Oceania: 'continent_Oceania',
} as const satisfies Record<Continent, TranslationKey>;

export function CollabChordFigure({ collaboration }: { collaboration: CollaborationNetwork | null | undefined }) {
  const t = useLocale((s) => s.t);
  const groupOf = (country: string): number => {
    const continent = continentOf(country);
    return continent ? CONTINENTS.indexOf(continent) : CONTINENTS.length;
  };
  const colorOf = (group: number): string => ((group < CONTINENTS.length && PALETTE[group]) || PALETTE[7]) as string;
  const groups = collaboration
    ? [...new Set(collaboration.nodes.map((node) => groupOf(node.country)))].sort((a, b) => a - b)
    : [];
  const legend = groups.map((group) => {
    const continent = CONTINENTS[group];
    return { label: t(continent ? CONTINENT_KEYS[continent] : 'continent_unknown'), color: colorOf(group) };
  });
  const state = stateOf(collaboration, false, (value) => value.nodes.length > 0);
  return (
    <Figure id="collabChord" state={state} legend={legend}>
      {collaboration && (
        <RadialGraph
          exportName="colaboracao-radial"
          weightLabel={t('radial_documents')}
          legend={legend}
          nodes={collaboration.nodes.map((node) => ({
            key: node.country,
            label: node.label,
            weight: node.documents,
            group: groupOf(node.country),
            color: colorOf(groupOf(node.country)),
          }))}
          edges={collaboration.edges.map((edge) => ({ source: edge.source, target: edge.target, weight: edge.documents }))}
        />
      )}
    </Figure>
  );
}

// ---------------------------------------------------------------------------------------
// Léxico, temas e redes

export function WordCloudFigure({ keywords }: { keywords: readonly EntityRow[] | undefined }) {
  const words = useMemo(
    () => keywords?.slice(0, 120).map((row) => ({ text: row.entity, value: row.docCount })),
    [keywords],
  );
  const state = words === undefined ? 'loading' : words.length === 0 ? 'empty' : 'ready';
  return (
    <Figure id="wordCloud" state={state}>
      {words && <WordCloud words={words} height={360} exportName="nuvem" />}
    </Figure>
  );
}

/** Temas: o gráfico de rosca desenhado em canvas (não há equivalente nas abas). */
export function ThemesFigure({ image, alt }: { image: string | null | undefined; alt: string }) {
  const state = image === undefined ? 'loading' : image === null ? 'empty' : 'ready';
  return (
    <Figure id="themesChart" state={state}>
      {image && <img data-report-img src={image} alt={alt} className="h-auto w-full" />}
    </Figure>
  );
}

export function NetworkFigure({ network }: { network: CooccurrenceReport | null }) {
  const state = stateOf(network, false, (value) => value.nodes.length > 0);
  return <Figure id="network" state={state}>{network && <NetworkSvg nodes={network.nodes} edges={network.edges} />}</Figure>;
}

export function NetworkChordFigure({ network }: { network: CooccurrenceReport | null }) {
  const t = useLocale((s) => s.t);
  const legend =
    network && network.communityCount <= 8
      ? Array.from({ length: network.communityCount }, (_, index) => ({
          label: `${t('radial_communities')} ${index + 1}`,
          color: communityColor(index),
        }))
      : undefined;
  const state = stateOf(network, false, (value) => value.nodes.length > 0);
  return (
    <Figure id="networkChord" state={state} legend={legend}>
      {network && (
        <RadialGraph
          exportName="rede-radial"
          weightLabel={t('radial_documents')}
          legend={legend}
          nodes={network.nodes.map((node) => ({
            key: node.key,
            label: node.label,
            weight: node.count,
            group: node.community,
            color: communityColor(node.community),
          }))}
          edges={network.edges}
        />
      )}
    </Figure>
  );
}

// ---------------------------------------------------------------------------------------
// Análises visuais avançadas

export function SankeyFigure({ dataset }: { dataset: Dataset }) {
  const c = VISUAL_COPY[useLocale((s) => s.locale)];
  const { data, loading } = useWorker<{ periods: readonly (readonly [number, number])[]; sankey: SankeyData | null } | null>(
    dataset,
    'sankey',
    async () => {
      const periods = await getAnalyticsWorker().sankeyPeriods(dataset);
      if (!periods) return null;
      return { periods, sankey: await getAnalyticsWorker().sankey(dataset, periods, 10) };
    },
  );
  const state = stateOf(data, loading, (value) => (value.sankey?.links.length ?? 0) > 0);
  return (
    <Figure id="sankey" state={state}>
      {data?.sankey && (
        <SankeyChart
          exportName={c.sankeyExport}
          height={520}
          columnLabels={data.periods.map(([start, end]) => `${start}–${end}`)}
          nodes={data.sankey.nodes.map((node) => ({
            term: node.term,
            column: node.period,
            color: PALETTE[node.period % PALETTE.length] as string,
          }))}
          links={data.sankey.links.map((link) => ({
            source: link.source,
            target: link.target,
            value: link.value,
            kind: link.kind === 'continuidade' ? c.continuity : c.intersection,
            color: link.kind === 'continuidade' ? 'rgba(63, 174, 143, 0.45)' : 'rgba(150, 160, 170, 0.28)',
          }))}
        />
      )}
    </Figure>
  );
}

export function BoxplotFigure({ dataset }: { dataset: Dataset }) {
  const locale = useLocale((s) => s.locale);
  const c = VISUAL_COPY[locale];
  const { data, loading } = useWorker<BoxplotSeries[]>(dataset, 'boxplot', async () => {
    const options = await getAnalyticsWorker().boxplotOptions(dataset, 'Países');
    return getAnalyticsWorker().boxplot(dataset, 'Países', 'Citações por documento', options.slice(0, 3));
  });
  const state = stateOf(data, loading, (value) => value.length > 0);
  return (
    <Figure id="boxplot" state={state}>
      {data && (
        <BoxPlotChart
          exportName={c.boxplotExport}
          height={380}
          log
          yLabel={locale === 'en' ? 'Citations per document' : 'Citações por documento'}
          series={data.map((entry, index) => ({
            name: entry.entity,
            color: PALETTE[index % PALETTE.length] as string,
            values: entry.values,
            labels: entry.labels,
          }))}
        />
      )}
    </Figure>
  );
}

export function GeneticsFigure({ dataset }: { dataset: Dataset }) {
  const locale = useLocale((s) => s.locale);
  const c = VISUAL_COPY[locale];
  const { data, loading } = useWorker<KeywordGenetics[]>(dataset, 'genetics', () => getAnalyticsWorker().genetics(dataset));
  const top = data?.slice(0, 150) ?? [];
  const citations = top.map((item) => item.citations);
  const min = Math.min(...citations);
  const max = Math.max(...citations);
  const state = stateOf(data, loading, (value) => value.length > 0);
  return (
    <Figure id="genetics" state={state}>
      <ScatterChart
        exportName={c.geneticsExport}
        ariaLabel={reportLabel('genetics', locale)}
        height={440}
        integerX
        xLabel={c.termBirthYear}
        yLabel={c.longevityYears}
        colorScale={{ label: c.citations, min, max, colors: CITATION_SCALE_LIGHT }}
        points={top.map((item) => ({
          id: item.keyword,
          x: item.birthYear,
          y: item.lifespan,
          r: Math.min(23, 4 + Math.sqrt(item.occurrences) * 1.5),
          color: interpolateColor(CITATION_SCALE_LIGHT, (item.citations - min) / (max - min || 1)),
          label: item.keyword,
          tooltip: null,
        }))}
      />
    </Figure>
  );
}

export function ConceptFigure({ dataset, dimensions }: { dataset: Dataset; dimensions: '2d' | '3d' }) {
  const locale = useLocale((s) => s.locale);
  const c = VISUAL_COPY[locale];
  const id = dimensions === '2d' ? 'concept2d' : 'concept3d';
  const { data, loading } = useWorker<ConceptTerm[]>(dataset, 'concept 4', () =>
    getAnalyticsWorker().conceptMap(dataset, { topTerms: 50, clusters: 4 }),
  );
  const terms = data ?? [];
  const groups = [...new Set(terms.map((term) => term.cluster))].sort((a, b) => a - b);
  const colorOf = (group: number): string => PALETTE[groups.indexOf(group) % PALETTE.length] as string;
  const legend = groups.map((group) => ({
    key: String(group),
    label: c.clusterN.replace('{n}', String(group + 1)),
    color: colorOf(group),
  }));
  const point = (term: ConceptTerm) => ({
    id: term.term,
    x: term.x,
    y: term.y,
    r: Math.min(17, 4 + Math.sqrt(term.frequency) * 1.25),
    color: colorOf(term.cluster),
    label: term.term,
    group: String(term.cluster),
    tooltip: null,
  });
  const dim = (n: number) => c.dimensionN.replace('{n}', String(n));
  const state = stateOf(data, loading, (value) => value.length > 0);
  return (
    <Figure id={id} state={state}>
      {dimensions === '2d' ? (
        <ScatterChart
          exportName={`${c.conceptExport}-2d`}
          ariaLabel={reportLabel(id, locale)}
          height={460}
          xLabel={dim(1)}
          yLabel={dim(2)}
          legend={legend}
          points={terms.map(point)}
        />
      ) : (
        <Scatter3DChart
          exportName={`${c.conceptExport}-3d`}
          ariaLabel={reportLabel(id, locale)}
          height={520}
          axisLabels={[dim(1), dim(2), dim(3)]}
          legend={legend}
          points={terms.map((term) => ({ ...point(term), z: term.z }))}
        />
      )}
    </Figure>
  );
}

export function ThematicMapFigure({ dataset }: { dataset: Dataset }) {
  const locale = useLocale((s) => s.locale);
  const c = VISUAL_COPY[locale];
  // Resumos dão o vocabulário mais rico; sem eles, as palavras-chave.
  const { data, loading } = useWorker<ThematicMap | null>(dataset, 'thematic', async () =>
    (await getAnalyticsWorker().thematicMap(dataset, 'abstract', 150)) ??
    getAnalyticsWorker().thematicMap(dataset, 'keywords', 150),
  );
  const state = stateOf(data, loading, (value) => value.clusters.length > 0);
  return (
    <Figure id="thematicMap" state={state}>
      {data && (
        <ScatterChart
          exportName={c.thematicExport}
          ariaLabel={reportLabel('thematicMap', locale)}
          height={500}
          labelPlacement="center"
          xLabel={c.centralityAxis}
          yLabel={c.densityAxis}
          reference={{ x: data.meanCentrality, y: data.meanDensity }}
          quadrants={{ topLeft: c.niches, topRight: c.motors, bottomLeft: c.emerging, bottomRight: c.basic }}
          points={data.clusters.map((cluster, index) => ({
            id: String(cluster.id),
            x: cluster.centrality,
            y: cluster.density,
            r: Math.min(45, 10 + Math.sqrt(cluster.frequency)),
            color: PALETTE[index % PALETTE.length] as string,
            opacity: 0.55,
            label: cluster.label.split('<br>').join('\n'),
            tooltip: null,
          }))}
        />
      )}
    </Figure>
  );
}

export function HistoriographFigure({ dataset }: { dataset: Dataset }) {
  const locale = useLocale((s) => s.locale);
  const c = VISUAL_COPY[locale];
  const { data, loading } = useWorker<HistoriographData | null>(dataset, `historiograph ${locale}`, () =>
    getAnalyticsWorker().historiograph(dataset, 30, locale),
  );
  const state = stateOf(data, loading, (value) => value.nodes.length > 0);
  return (
    <Figure id="historiograph" state={state}>
      {data && (
        <ScatterChart
          exportName="historiograph"
          ariaLabel={reportLabel('historiograph', locale)}
          height={500}
          integerX
          hideYAxis
          xLabel={c.timeline}
          edges={data.edges}
          points={data.nodes.map((node) => ({
            id: node.id,
            x: node.year,
            y: node.offset,
            r: Math.max(4, node.size / 4),
            color: PALETTE[0] as string,
            label: node.id,
            tooltip: null,
          }))}
        />
      )}
    </Figure>
  );
}

export function LotkaFigure({ lotka }: { lotka: Parameters<typeof LotkaChart>[0]['lotka'] | null | undefined }) {
  const t = useLocale((s) => s.t);
  // A legenda da Lotka é HTML, fora do SVG: vai junto na captura.
  const legend = [
    { label: t('lotka_observed'), color: PALETTE[0] as string },
    { label: t('lotka_theoretical'), color: PALETTE[1] as string },
  ];
  const state = lotka === undefined ? 'loading' : lotka === null ? 'empty' : 'ready';
  return (
    <Figure id="lotka" state={state} legend={legend}>
      {lotka && <LotkaChart lotka={lotka} exportName="lotka" />}
    </Figure>
  );
}

/** Figura isolada: um gráfico que quebre não derruba a folha. */
export function SafeFigure({ children }: { children: ReactNode }) {
  const locale = useLocale((s) => s.locale);
  return <ErrorBoundary label={locale === 'en' ? 'chart' : 'gráfico'}>{children}</ErrorBoundary>;
}
