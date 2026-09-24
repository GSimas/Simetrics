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
import { useLocale } from '@/state/locale.store';
import { getAnalyticsWorker } from '@/workers/client';
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
          <Label htmlFor="box-dimension">Comparar por</Label>
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
                  {option}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-1.5">
          <Label htmlFor="box-metric">Métrica</Label>
          <Select value={metric} onValueChange={(value) => setMetric(value as BoxplotMetric)}>
            <SelectTrigger id="box-metric">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {BOX_METRICS.map((option) => (
                <SelectItem key={option} value={option}>
                  {option}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-1.5">
          <Label htmlFor="box-scale">Escala do eixo Y</Label>
          <Select
            value={logScale ? 'log' : 'linear'}
            onValueChange={(value) => setLogScale(value === 'log')}
          >
            <SelectTrigger id="box-scale">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="linear">Linear</SelectItem>
              <SelectItem value="log">Logarítmica</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {(options ?? []).length === 0 ? (
        chartMessage(
          dimension === 'Temas (IA)'
            ? 'Nenhum tema disponível. Use o mapeamento temático por IA acima para gerá-los.'
            : `A base não traz dados de ${dimension.toLowerCase()}.`,
        )
      ) : (
        <>
          <div className="space-y-1.5">
            <Label>
              Selecione até {MAX_BOXPLOT_ITEMS} itens ({effectiveSelection.length} selecionados)
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
            chartMessage('Selecione ao menos um item para comparar.')
          ) : (
            <BoxPlotChart
              exportName="distribuicao-comparativa"
              onSeriesClick={(name) => openInSearch(name, BOX_SEARCH_TYPES[dimension])}
              height={440}
              log={logScale}
              yLabel={metric}
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
      loadingPeriods
        ? 'Calculando fluxos temáticos…'
        : 'A base precisa de anos e palavras-chave para montar o fluxo.',
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
        <Label htmlFor="sankey-top">Termos por período</Label>
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
              Período {index + 1}: {start}–{end}
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

      <ReadingTip>
        As linhas mais grossas são termos que sobreviveram de um período ao seguinte; as
        finas, termos distintos que costumam aparecer nos mesmos documentos.
      </ReadingTip>

      {!sankey ? (
        chartMessage(
          loadingSankey
            ? 'Calculando fluxos temáticos…'
            : 'Nenhum fluxo para os períodos selecionados — tente um recorte mais amplo.',
        )
      ) : (
        <SankeyChart
          exportName="evolucao-tematica"
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
            kind: link.kind === 'continuidade' ? 'Continuidade' : 'Intersecção',
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
  const dark = usePreferences((state) => state.theme === 'dark');
  const CITATION_SCALE = dark ? CITATION_SCALE_DARK : CITATION_SCALE_LIGHT;
  const { data } = useAsyncResult<KeywordGenetics[]>('genetics', () =>
    getAnalyticsWorker().genetics(dataset),
  );

  if (!data) return chartMessage('Calculando ciclo de vida dos termos…');
  if (data.length === 0) {
    return chartMessage('A base precisa de palavras-chave e anos para esta análise.');
  }

  // Só os termos mais replicados: a cauda longa é composta de termos que aparecem uma vez
  // e formaria uma nuvem indistinta na origem do gráfico.
  const top = data.slice(0, 150);
  const minCitations = Math.min(...top.map((item) => item.citations));
  const maxCitations = Math.max(...top.map((item) => item.citations));

  return (
    <div className="space-y-3">
      <ReadingTip>
        Cada ponto é uma palavra-chave. O eixo X mostra quando ela apareceu pela primeira
        vez; o Y, por quantos anos permaneceu em uso; o tamanho, quantas vezes se replicou.
        Termos no alto e à esquerda são o núcleo estável da área; à direita e embaixo, as
        fronteiras recentes.
      </ReadingTip>

      <ScatterChart
        exportName="genetica-das-ideias"
        ariaLabel={t('visual_tab_genetics')}
        height={480}
        integerX
        xLabel="Ano de nascimento do termo"
        yLabel="Longevidade (anos)"
        colorScale={{ label: 'Citações', min: minCitations, max: maxCitations, colors: CITATION_SCALE }}
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
              <TipRow label="Nasceu em" value={item.birthYear} />
              <TipRow label="Longevidade" value={`${item.lifespan} anos`} />
              <TipRow label="Replicações" value={formatNumber(item.occurrences)} />
              <TipRow label="Citações" value={formatNumber(item.citations)} />
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
  const [dimensions, setDimensions] = useState<'2d' | '3d'>('2d');
  const [clusters, setClusters] = useState(4);

  const { data: terms } = useAsyncResult<ConceptTerm[]>(`concept ${clusters}`, () =>
    getAnalyticsWorker().conceptMap(dataset, { topTerms: 50, clusters }),
  );

  if (!terms) return chartMessage('Projetando termos…');
  if (terms.length === 0) {
    return chartMessage('A base precisa de palavras-chave suficientes para o mapa conceitual.');
  }

  const groups = [...new Set(terms.map((term) => term.cluster))].sort((a, b) => a - b);
  const colorOf = new Map(groups.map((group, index) => [group, PALETTE[index % PALETTE.length] as string]));
  const legend = groups.map((group) => ({
    key: String(group),
    label: `Agrupamento ${group + 1}`,
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
        <TipRow label="Agrupamento" value={term.cluster + 1} color={colorOf.get(term.cluster)} />
        <TipRow label="Frequência" value={formatNumber(term.frequency)} />
      </>
    ),
    onClick: () => openInSearch(term.term, KEYWORD),
  });

  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-2">
        <div className="space-y-1.5">
          <Label htmlFor="concept-dim">Projeção</Label>
          <Select
            value={dimensions}
            onValueChange={(value) => setDimensions(value as '2d' | '3d')}
          >
            <SelectTrigger id="concept-dim">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="2d">2 dimensões</SelectItem>
              <SelectItem value="3d">3 dimensões</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-1.5">
          <Label htmlFor="concept-clusters">Agrupamentos</Label>
          <Select value={String(clusters)} onValueChange={(value) => setClusters(Number(value))}>
            <SelectTrigger id="concept-clusters">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {[3, 4, 5, 6, 8].map((option) => (
                <SelectItem key={option} value={String(option)}>
                  {option} agrupamentos
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <ReadingTip>
        Termos próximos aparecem nos mesmos documentos. As ilhas são escolas de pensamento;
        os termos entre elas são pontes conceituais.
      </ReadingTip>

      {dimensions === '3d' ? (
        <Scatter3DChart
          exportName="mapa-conceitual-3d"
          ariaLabel={t('visual_tab_concept')}
          height={620}
          axisLabels={['Dimensão 1', 'Dimensão 2', 'Dimensão 3']}
          legend={legend}
          points={terms.map((term) => ({ ...conceptPoint(term), z: term.z }))}
        />
      ) : (
        <ScatterChart
          exportName="mapa-conceitual-2d"
          ariaLabel={t('visual_tab_concept')}
          height={500}
          xLabel="Dimensão 1"
          yLabel="Dimensão 2"
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
  const [source, setSource] = useState<'abstract' | 'keywords'>('abstract');

  const { data: map, loading } = useAsyncResult<ThematicMap | null>(`thematic ${source}`, () =>
    getAnalyticsWorker().thematicMap(dataset, source, 150),
  );

  if (loading) return chartMessage('Construindo a rede de coocorrência…');
  if (!map) return chartMessage('Não há texto suficiente para montar o mapa temático.');

  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-2">
        <div className="space-y-1.5">
          <Label htmlFor="thematic-source">Fonte do texto</Label>
          <Select
            value={source}
            onValueChange={(value) => setSource(value as 'abstract' | 'keywords')}
          >
            <SelectTrigger id="thematic-source">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="abstract">Resumos</SelectItem>
              <SelectItem value="keywords">Palavras-chave</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <ReadingTip>{QUADRANT_NOTE}</ReadingTip>

      <ScatterChart
        exportName="mapa-tematico"
        ariaLabel={t('visual_tab_thematic')}
        height={560}
        labelPlacement="center"
        xLabel="Centralidade (relevância externa)"
        yLabel="Densidade (desenvolvimento interno)"
        reference={{ x: map.meanCentrality, y: map.meanDensity }}
        quadrants={{
          topLeft: 'Nichos',
          topRight: 'Motores',
          bottomLeft: 'Emergentes / em declínio',
          bottomRight: 'Básicos / transversais',
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
              <TipRow label="Centralidade" value={formatNumber(cluster.centrality, 0)} />
              <TipRow label="Densidade" value={formatNumber(cluster.density, 0)} />
              <TipRow label="Frequência" value={formatNumber(cluster.frequency)} />
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
  const [topN, setTopN] = useState(30);

  const { data, loading } = useAsyncResult<HistoriographData | null>(
    `historiograph ${topN}`,
    () => getAnalyticsWorker().historiograph(dataset, topN),
  );

  if (loading) return chartMessage('Rastreando citações diretas…');

  if (!data) {
    return chartMessage(
      'Esta base não traz referências citadas, e sem elas não há como rastrear quais ' +
        'documentos citam quais. No Web of Science, exporte com "Full Record and Cited ' +
        'References"; no Scopus, marque "References" na exportação.',
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end gap-3">
        <div className="space-y-1.5">
          <Label htmlFor="hist-top">Documentos</Label>
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
          {data.edges.length} citações diretas entre os {data.nodes.length} documentos mais
          citados. A detecção casa sobrenome do primeiro autor e ano dentro do texto das
          referências, então erra em homônimos e em grafias divergentes.
        </ReadingTip>
      </div>

      <ScatterChart
        exportName="historiograph"
        ariaLabel={t('visual_tab_historiograph')}
        height={560}
        integerX
        hideYAxis
        xLabel="Linha do tempo"
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
              <TipRow label="Ano" value={node.year} />
              <TipRow label="Citações" value={formatNumber(node.citations)} />
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
  const lotka = useDataset((state) => state.overview?.lotka);
  if (!lotka) return chartMessage('Calculando a distribuição de produtividade…');
  return (
    <div className="space-y-3">
      <ReadingTip>{t('lotka_description')}</ReadingTip>
      <Suspense fallback={chartMessage('Carregando gráfico…')}>
        <LotkaChart lotka={lotka} />
      </Suspense>
    </div>
  );
}
