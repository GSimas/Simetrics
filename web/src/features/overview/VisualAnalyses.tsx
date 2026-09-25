import { lazy, Suspense, useState } from 'react';

import BoxPlotChart from '@/components/charts/BoxPlotChart';
import SankeyChart from '@/components/charts/SankeyChart';
import Scatter3DChart from '@/components/charts/Scatter3DChart';
import ScatterChart from '@/components/charts/ScatterChart';
import { TipRow } from '@/components/charts/svg/chart-kit';
import { formatNumber, interpolateColor } from '@/components/charts/svg/scale';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  MAX_BOXPLOT_ITEMS,
  type BoxplotDimension,
  type BoxplotMetric,
  type BoxplotSeries,
} from '@/core/viz/boxplot';
import type { ConceptTerm } from '@/core/viz/concept-map';
import type { KeywordGenetics } from '@/core/viz/genetics';
import type { HistoriographData } from '@/core/viz/historiograph';
import type { Period, SankeyData } from '@/core/viz/sankey';
import type { ThematicMap } from '@/core/viz/thematic-map';
import type { Dataset, SearchEntityType } from '@/lib/types';
import { useAsyncResult } from '@/lib/use-async-result';
import { boxplotDimensionLabel, boxplotMetricLabel } from '@/lib/i18n/labels';
import { useLocale } from '@/state/locale.store';
import { getAnalyticsWorker } from '@/workers/client';
import { VISUAL_COPY } from './visual-copy';
import { PALETTE, QUADRANT_NOTE, chartMessage } from './viz-shared';
import { ReadingTip } from '@/components/InfoTip';
import { openInSearch } from '@/state/navigation.store';
import { useDataset } from '@/state/dataset.store';
import { usePreferences } from '@/state/preferences.store';

const LotkaChart = lazy(() => import('@/components/charts/LotkaChart'));

/**
 * Segunda onda de visualizações, agrupada em sub-abas.
 *
 * As sub-abas não são organização decorativa: cada gráfico aqui custa uma passagem
 * completa pela base no worker, e o conteúdo de cada aba só é calculado quando ela é
 * aberta. Empilhar todos na Visão Geral dispararia sete análises pesadas de uma vez.
 */

type PanelKey = 'boxplot' | 'sankey' | 'genetics' | 'concept' | 'thematic' | 'historiograph' | 'lotka';

const BOX_DIMENSIONS: BoxplotDimension[] = ['Países', 'Palavras-chave', 'Temas (IA)'];
const BOX_METRICS: BoxplotMetric[] = [
  'Citações por documento',
  'Citações por autor',
  'Citações por ano',
  'Documentos por autor',
  'Documentos por ano',
];

export interface VisualAnalysesProps {
  dataset: Dataset;
}

export function VisualAnalyses({ dataset }: VisualAnalysesProps) {
  const [panel, setPanel] = useState<PanelKey>('sankey');
  const t = useLocale((state) => state.t);

  return (
    <>
        <Tabs value={panel} onValueChange={(value) => setPanel(value as PanelKey)}>
          <TabsList className="h-auto w-full flex-wrap justify-start gap-x-2">
            <TabsTrigger value="sankey">{t('visual_tab_sankey')}</TabsTrigger>
            <TabsTrigger value="boxplot">{t('visual_tab_boxplot')}</TabsTrigger>
            <TabsTrigger value="genetics">{t('visual_tab_genetics')}</TabsTrigger>
            <TabsTrigger value="concept">{t('visual_tab_concept')}</TabsTrigger>
            <TabsTrigger value="thematic">{t('visual_tab_thematic')}</TabsTrigger>
            <TabsTrigger value="historiograph">{t('visual_tab_historiograph')}</TabsTrigger>
            <TabsTrigger value="lotka">{t('lotka_title')}</TabsTrigger>
          </TabsList>

          <TabsContent value="sankey">
            {panel === 'sankey' && <SankeyPanel dataset={dataset} />}
          </TabsContent>
          <TabsContent value="boxplot">
            {panel === 'boxplot' && <BoxplotPanel dataset={dataset} />}
          </TabsContent>
          <TabsContent value="genetics">
            {panel === 'genetics' && <GeneticsPanel dataset={dataset} />}
          </TabsContent>
          <TabsContent value="concept">
            {panel === 'concept' && <ConceptPanel dataset={dataset} />}
          </TabsContent>
          <TabsContent value="thematic">
            {panel === 'thematic' && <ThematicPanel dataset={dataset} />}
          </TabsContent>
          <TabsContent value="historiograph">
            {panel === 'historiograph' && <HistoriographPanel dataset={dataset} />}
          </TabsContent>
          <TabsContent value="lotka">{panel === 'lotka' && <LotkaPanel />}</TabsContent>
        </Tabs>
    </>
  );
}

/** Tipos de entidade do Motor de Busca para cada dimensão do boxplot. */
const BOX_SEARCH_TYPES: Record<BoxplotDimension, SearchEntityType[]> = {
  Países: ['País'],
  'Palavras-chave': ['Palavra-chave'],
  'Temas (IA)': ['Tema'],
};

const KEYWORD: SearchEntityType[] = ['Palavra-chave'];

/**
 * Escala de cor das citações na genética dos termos — no lugar do "Teal" do Plotly. Mais
 * citado = mais contraste com o fundo: escurece no tema claro e clareia no escuro.
 */
const CITATION_SCALE_LIGHT = ['#cdeee4', '#8fd3c1', '#3fae8f', '#236e5e', '#0f3b33'] as const;
const CITATION_SCALE_DARK = ['#1d4a40', '#236e5e', '#3fae8f', '#8fd3c1', '#d9ffa0'] as const;

/** Distribuição estatística comparativa. */
function BoxplotPanel({ dataset }: { dataset: Dataset }) {
  const locale = useLocale((state) => state.locale);
  const c = VISUAL_COPY[locale];
  const [dimension, setDimension] = useState<BoxplotDimension>('Países');
  const [metric, setMetric] = useState<BoxplotMetric>('Citações por documento');
  const [selected, setSelected] = useState<string[] | null>(null);
  // Logarítmica por padrão: citações têm cauda longa e a escala linear achata as caixas.
  const [logScale, setLogScale] = useState(true);

  const { data: options, loading: loadingOptions } = useAsyncResult<string[]>(
    `box-options ${dimension}`,
    () => getAnalyticsWorker().boxplotOptions(dataset, dimension),
  );

  // Pré-seleção derivada, e não escrita em efeito: enquanto o usuário não escolher nada,
  // o painel abre com as três entidades mais frequentes. `selected` só passa a valer
  // depois da primeira interação, e a troca de dimensão zera de volta para a derivada.
  const effectiveSelection = selected ?? (options ?? []).slice(0, 3);

  const { data: series } = useAsyncResult<BoxplotSeries[]>(
    `box ${dimension} ${metric} ${effectiveSelection.join('|')}`,
    () =>
      effectiveSelection.length === 0
        ? Promise.resolve([])
        : getAnalyticsWorker().boxplot(dataset, dimension, metric, effectiveSelection),
    // Na troca de dimensão, `options` ainda é da dimensão anterior: espera as novas.
    { enabled: !loadingOptions },
  );

  const toggle = (entity: string): void => {
    const current = effectiveSelection;
    setSelected(
      current.includes(entity)
        ? current.filter((item) => item !== entity)
        : current.length >= MAX_BOXPLOT_ITEMS
          ? current
          : [...current, entity],
    );
  };

  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-3">
        <div className="space-y-1.5">
          <Label htmlFor="box-dimension">{c.compareBy}</Label>
          <Select
            value={dimension}
            onValueChange={(value) => {
              setDimension(value as BoxplotDimension);
              setSelected(null);
            }}
          >
            <SelectTrigger id="box-dimension">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {BOX_DIMENSIONS.map((option) => (
                <SelectItem key={option} value={option}>
                  {boxplotDimensionLabel(option, locale)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-1.5">
          <Label htmlFor="box-metric">{c.metric}</Label>
          <Select value={metric} onValueChange={(value) => setMetric(value as BoxplotMetric)}>
            <SelectTrigger id="box-metric">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {BOX_METRICS.map((option) => (
                <SelectItem key={option} value={option}>
                  {boxplotMetricLabel(option, locale)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-1.5">
          <Label htmlFor="box-scale">{c.yScale}</Label>
          <Select
            value={logScale ? 'log' : 'linear'}
            onValueChange={(value) => setLogScale(value === 'log')}
          >
            <SelectTrigger id="box-scale">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="linear">{c.linear}</SelectItem>
              <SelectItem value="log">{c.logarithmic}</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {(options ?? []).length === 0 ? (
        chartMessage(
          dimension === 'Temas (IA)'
            ? c.noThemes
            : c.noDimensionData.replace('{dimension}', boxplotDimensionLabel(dimension, locale).toLowerCase()),
        )
      ) : (
        <>
          <div className="space-y-1.5">
            <Label>
              {c.selectUpTo
                .replace('{max}', String(MAX_BOXPLOT_ITEMS))
                .replace('{count}', String(effectiveSelection.length))}
            </Label>
            <div className="flex max-h-32 flex-wrap gap-1.5 overflow-y-auto rounded-md border p-2">
              {(options ?? []).slice(0, 60).map((option) => (
                <Button
                  key={option}
                  size="sm"
                  variant={effectiveSelection.includes(option) ? 'default' : 'outline'}
                  className="h-7 max-w-64 truncate text-xs font-normal"
                  title={option}
                  onClick={() => toggle(option)}
                >
                  {option}
                </Button>
              ))}
            </div>
          </div>

          {(series ?? []).length === 0 ? (
            chartMessage(c.selectAtLeastOne)
          ) : (
            <BoxPlotChart
              exportName={c.boxplotExport}
              onSeriesClick={(name) => openInSearch(name, BOX_SEARCH_TYPES[dimension])}
              height={440}
              log={logScale}
              yLabel={boxplotMetricLabel(metric, locale)}
              series={(series ?? []).map((entry, index) => ({
                name: entry.entity,
                color: PALETTE[index % PALETTE.length] as string,
                values: entry.values,
                labels: entry.labels,
              }))}
            />
          )}
        </>
      )}
    </div>
  );
}

/**
 * Fluxo de evolução temática entre três períodos.
 *
 * Os períodos partem de uma sugestão automática (três fatias de tamanho semelhante
 * cobrindo toda a base), mas o usuário pode ajustar o início/fim de cada um livremente
 * pelos sliders — os limites do slider são o próprio intervalo coberto pela sugestão,
 * então não dá para arrastar para um ano sem documento algum na base.
 */
function SankeyPanel({ dataset }: { dataset: Dataset }) {
  const locale = useLocale((state) => state.locale);
  const c = VISUAL_COPY[locale];
  const [topN, setTopN] = useState(10);
  const [periods, setPeriods] = useState<[Period, Period, Period] | null>(null);

  const { data: suggested, loading: loadingPeriods } = useAsyncResult<
    [Period, Period, Period] | null
  >('sankey-periods', () => getAnalyticsWorker().sankeyPeriods(dataset));

  // Os períodos ajustados pelo usuário prevalecem; a sugestão automática só define o
  // ponto de partida, antes da primeira interação com os sliders.
  const effectivePeriods = periods ?? suggested;

  const { data: sankey, loading: loadingSankey } = useAsyncResult<SankeyData | null>(
    `sankey ${topN} ${effectivePeriods ? effectivePeriods.flat().join('-') : 'none'}`,
    () =>
      effectivePeriods
        ? getAnalyticsWorker().sankey(dataset, effectivePeriods, topN)
        : Promise.resolve(null),
  );

  if (!suggested) {
    return chartMessage(
      loadingPeriods ? c.computingFlows : c.sankeyNeedsData,
    );
  }

  const [datasetStart] = suggested[0];
  const [, datasetEnd] = suggested[2];
  const activePeriods = effectivePeriods ?? suggested;

  const updatePeriod = (index: 0 | 1 | 2, next: Period): void => {
    const base = [...activePeriods] as [Period, Period, Period];
    base[index] = next;
    setPeriods(base);
  };

  return (
    <div className="space-y-4">
      <div className="space-y-1.5">
        <Label htmlFor="sankey-top">{c.termsPerPeriod}</Label>
        <Select value={String(topN)} onValueChange={(value) => setTopN(Number(value))}>
          <SelectTrigger id="sankey-top" className="w-40">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {[5, 8, 10, 15, 20].map((option) => (
              <SelectItem key={option} value={String(option)}>
                Top {option}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="grid gap-4 rounded-md border p-3 sm:grid-cols-3">
        {activePeriods.map(([start, end], index) => (
          <div key={index} className="space-y-2">
            <Label>
              {c.period.replace('{n}', String(index + 1))}: {start}–{end}
            </Label>
            <Slider
              min={datasetStart}
              max={datasetEnd}
              step={1}
              value={[start, end]}
              onValueChange={(value) => {
                const [nextStart, nextEnd] = value as [number, number];
                updatePeriod(index as 0 | 1 | 2, [nextStart, nextEnd]);
              }}
              className="py-1"
            />
          </div>
        ))}
      </div>

      <ReadingTip>{c.sankeyTip}</ReadingTip>

      {!sankey ? (
        chartMessage(
          loadingSankey ? c.computingFlows : c.noFlow,
        )
      ) : (
        <SankeyChart
          exportName={c.sankeyExport}
          onNodeClick={(term) => openInSearch(term, KEYWORD)}
          height={620}
          columnLabels={activePeriods.map(([start, end]) => `${start}–${end}`)}
          nodes={sankey.nodes.map((node) => ({
            term: node.term,
            column: node.period,
            color: PALETTE[node.period % PALETTE.length] as string,
          }))}
          links={sankey.links.map((link) => ({
            source: link.source,
            target: link.target,
            value: link.value,
            kind: link.kind === 'continuidade' ? c.continuity : c.intersection,
            color: link.kind === 'continuidade' ? 'rgba(63, 174, 143, 0.45)' : 'rgba(150, 160, 170, 0.28)',
          }))}
        />
      )}
    </div>
  );
}

/** Ciclo de vida das palavras-chave. */
function GeneticsPanel({ dataset }: { dataset: Dataset }) {
  const t = useLocale((state) => state.t);
  const locale = useLocale((state) => state.locale);
  const c = VISUAL_COPY[locale];
  const dark = usePreferences((state) => state.theme === 'dark');
  const CITATION_SCALE = dark ? CITATION_SCALE_DARK : CITATION_SCALE_LIGHT;
  const { data } = useAsyncResult<KeywordGenetics[]>('genetics', () =>
    getAnalyticsWorker().genetics(dataset),
  );

  if (!data) return chartMessage(c.computingGenetics);
  if (data.length === 0) {
    return chartMessage(c.geneticsNeedsData);
  }

  // Só os termos mais replicados: a cauda longa é composta de termos que aparecem uma vez
  // e formaria uma nuvem indistinta na origem do gráfico.
  const top = data.slice(0, 150);
  const minCitations = Math.min(...top.map((item) => item.citations));
  const maxCitations = Math.max(...top.map((item) => item.citations));

  return (
    <div className="space-y-3">
      <ReadingTip>{c.geneticsTip}</ReadingTip>

      <ScatterChart
        exportName={c.geneticsExport}
        ariaLabel={t('visual_tab_genetics')}
        height={480}
        integerX
        xLabel={c.termBirthYear}
        yLabel={c.longevityYears}
        colorScale={{ label: c.citations, min: minCitations, max: maxCitations, colors: CITATION_SCALE }}
        points={top.map((item) => ({
          id: item.keyword,
          x: item.birthYear,
          y: item.lifespan,
          // O Plotly media o diâmetro (8 + √n × 3, até 46); aqui é o raio.
          r: Math.min(23, 4 + Math.sqrt(item.occurrences) * 1.5),
          color: interpolateColor(
            CITATION_SCALE,
            (item.citations - minCitations) / (maxCitations - minCitations || 1),
          ),
          label: item.keyword,
          tooltip: (
            <>
              <p className="mb-1 font-semibold break-words">{item.keyword}</p>
              <TipRow label={c.bornIn} value={item.birthYear} />
              <TipRow label={c.longevity} value={c.years.replace('{n}', String(item.lifespan))} />
              <TipRow label={c.replications} value={formatNumber(item.occurrences, undefined, locale)} />
              <TipRow label={c.citations} value={formatNumber(item.citations, undefined, locale)} />
            </>
          ),
          onClick: () => openInSearch(item.keyword, KEYWORD),
        }))}
      />
    </div>
  );
}

/** Mapa conceitual por PCA, em 2D e 3D. */
function ConceptPanel({ dataset }: { dataset: Dataset }) {
  const t = useLocale((state) => state.t);
  const locale = useLocale((state) => state.locale);
  const c = VISUAL_COPY[locale];
  const [dimensions, setDimensions] = useState<'2d' | '3d'>('2d');
  const [clusters, setClusters] = useState(4);

  const { data: terms } = useAsyncResult<ConceptTerm[]>(`concept ${clusters}`, () =>
    getAnalyticsWorker().conceptMap(dataset, { topTerms: 50, clusters }),
  );

  if (!terms) return chartMessage(c.projectingTerms);
  if (terms.length === 0) {
    return chartMessage(c.conceptNeedsData);
  }

  const groups = [...new Set(terms.map((term) => term.cluster))].sort((a, b) => a - b);
  const colorOf = new Map(groups.map((group, index) => [group, PALETTE[index % PALETTE.length] as string]));
  const legend = groups.map((group) => ({
    key: String(group),
    label: c.clusterN.replace('{n}', String(group + 1)),
    color: colorOf.get(group) as string,
  }));
  const conceptPoint = (term: ConceptTerm) => ({
    id: term.term,
    x: term.x,
    y: term.y,
    r: Math.min(17, 4 + Math.sqrt(term.frequency) * 1.25),
    color: colorOf.get(term.cluster) as string,
    label: term.term,
    group: String(term.cluster),
    tooltip: (
      <>
        <p className="mb-1 font-semibold break-words">{term.term}</p>
        <TipRow label={c.cluster} value={term.cluster + 1} color={colorOf.get(term.cluster)} />
        <TipRow label={c.frequency} value={formatNumber(term.frequency, undefined, locale)} />
      </>
    ),
    onClick: () => openInSearch(term.term, KEYWORD),
  });

  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-2">
        <div className="space-y-1.5">
          <Label htmlFor="concept-dim">{c.projection}</Label>
          <Select
            value={dimensions}
            onValueChange={(value) => setDimensions(value as '2d' | '3d')}
          >
            <SelectTrigger id="concept-dim">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="2d">{c.dimensions.replace('{n}', '2')}</SelectItem>
              <SelectItem value="3d">{c.dimensions.replace('{n}', '3')}</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-1.5">
          <Label htmlFor="concept-clusters">{c.clusters}</Label>
          <Select value={String(clusters)} onValueChange={(value) => setClusters(Number(value))}>
            <SelectTrigger id="concept-clusters">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {[3, 4, 5, 6, 8].map((option) => (
                <SelectItem key={option} value={String(option)}>
                  {c.clustersN.replace('{n}', String(option))}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <ReadingTip>{c.conceptTip}</ReadingTip>

      {dimensions === '3d' ? (
        <Scatter3DChart
          exportName={`${c.conceptExport}-3d`}
          ariaLabel={t('visual_tab_concept')}
          height={620}
          axisLabels={[
            c.dimensionN.replace('{n}', '1'),
            c.dimensionN.replace('{n}', '2'),
            c.dimensionN.replace('{n}', '3'),
          ]}
          legend={legend}
          points={terms.map((term) => ({ ...conceptPoint(term), z: term.z }))}
        />
      ) : (
        <ScatterChart
          exportName={`${c.conceptExport}-2d`}
          ariaLabel={t('visual_tab_concept')}
          height={500}
          xLabel={c.dimensionN.replace('{n}', '1')}
          yLabel={c.dimensionN.replace('{n}', '2')}
          legend={legend}
          points={terms.map(conceptPoint)}
        />
      )}
    </div>
  );
}

/** Mapa temático de centralidade × densidade. */
function ThematicPanel({ dataset }: { dataset: Dataset }) {
  const t = useLocale((state) => state.t);
  const locale = useLocale((state) => state.locale);
  const c = VISUAL_COPY[locale];
  const [source, setSource] = useState<'abstract' | 'keywords'>('abstract');

  const { data: map, loading } = useAsyncResult<ThematicMap | null>(`thematic ${source}`, () =>
    getAnalyticsWorker().thematicMap(dataset, source, 150),
  );

  if (loading) return chartMessage(c.buildingNetwork);
  if (!map) return chartMessage(c.thematicNeedsData);

  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-2">
        <div className="space-y-1.5">
          <Label htmlFor="thematic-source">{c.textSource}</Label>
          <Select
            value={source}
            onValueChange={(value) => setSource(value as 'abstract' | 'keywords')}
          >
            <SelectTrigger id="thematic-source">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="abstract">{c.abstracts}</SelectItem>
              <SelectItem value="keywords">{c.keywords}</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <ReadingTip>{QUADRANT_NOTE[locale]}</ReadingTip>

      <ScatterChart
        exportName={c.thematicExport}
        ariaLabel={t('visual_tab_thematic')}
        height={560}
        labelPlacement="center"
        xLabel={c.centralityAxis}
        yLabel={c.densityAxis}
        reference={{ x: map.meanCentrality, y: map.meanDensity }}
        quadrants={{
          topLeft: c.niches,
          topRight: c.motors,
          bottomLeft: c.emerging,
          bottomRight: c.basic,
        }}
        points={map.clusters.map((cluster, index) => ({
          id: String(cluster.id),
          x: cluster.centrality,
          y: cluster.density,
          r: Math.min(45, 10 + Math.sqrt(cluster.frequency)),
          color: PALETTE[index % PALETTE.length] as string,
          opacity: 0.55,
          // O núcleo monta o rótulo com "<br>" (herança do Plotly): vira quebra de linha.
          label: cluster.label.split('<br>').join('\n'),
          tooltip: (
            <>
              <p className="mb-1 font-semibold break-words">{cluster.terms.slice(0, 3).join(' · ')}</p>
              <p className="mb-1 text-muted-foreground break-words">{cluster.terms.join(', ')}</p>
              <TipRow label={c.centrality} value={formatNumber(cluster.centrality, 0, locale)} />
              <TipRow label={c.density} value={formatNumber(cluster.density, 0, locale)} />
              <TipRow label={c.frequency} value={formatNumber(cluster.frequency, undefined, locale)} />
            </>
          ),
          onClick: () => openInSearch(cluster.terms[0], ['Palavra-chave', 'Tema']),
        }))}
      />
    </div>
  );
}

/** Linha do tempo de citações diretas. */
function HistoriographPanel({ dataset }: { dataset: Dataset }) {
  const t = useLocale((state) => state.t);
  const locale = useLocale((state) => state.locale);
  const c = VISUAL_COPY[locale];
  const [topN, setTopN] = useState(30);

  const { data, loading } = useAsyncResult<HistoriographData | null>(
    `historiograph ${topN} ${locale}`,
    () => getAnalyticsWorker().historiograph(dataset, topN, locale),
  );

  if (loading) return chartMessage(c.tracingCitations);

  if (!data) {
    return chartMessage(c.noReferences);
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end gap-3">
        <div className="space-y-1.5">
          <Label htmlFor="hist-top">{c.documents}</Label>
          <Select value={String(topN)} onValueChange={(value) => setTopN(Number(value))}>
            <SelectTrigger id="hist-top" className="w-40">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {[20, 30, 40, 50].map((option) => (
                <SelectItem key={option} value={String(option)}>
                  Top {option}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <ReadingTip>
          {c.historiographTip
            .replace('{edges}', String(data.edges.length))
            .replace('{nodes}', String(data.nodes.length))}
        </ReadingTip>
      </div>

      <ScatterChart
        exportName="historiograph"
        ariaLabel={t('visual_tab_historiograph')}
        height={560}
        integerX
        hideYAxis
        xLabel={c.timeline}
        edges={data.edges}
        points={data.nodes.map((node) => ({
          id: node.id,
          x: node.year,
          y: node.offset,
          r: Math.max(4, node.size / 4),
          color: PALETTE[0],
          label: node.id,
          tooltip: (
            <>
              <p className="mb-1 font-semibold break-words">{node.title}</p>
              <TipRow label={c.year} value={node.year} />
              <TipRow label={c.citations} value={formatNumber(node.citations, undefined, locale)} />
            </>
          ),
          onClick: () => openInSearch(node.title, ['Documento']),
        }))}
      />
    </div>
  );
}

/** Lei de Lotka: produtividade observada dos autores contra a curva teórica c/x². */
function LotkaPanel() {
  const t = useLocale((state) => state.t);
  const locale = useLocale((state) => state.locale);
  const c = VISUAL_COPY[locale];
  const lotka = useDataset((state) => state.overview?.lotka);
  if (!lotka) return chartMessage(c.computingLotka);
  return (
    <div className="space-y-3">
      <ReadingTip>{t('lotka_description')}</ReadingTip>
      <Suspense fallback={chartMessage(c.loadingChart)}>
        <LotkaChart lotka={lotka} />
      </Suspense>
    </div>
  );
}
