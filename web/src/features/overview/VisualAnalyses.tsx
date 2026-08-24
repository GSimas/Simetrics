import { lazy, Suspense, useState } from 'react';

import type { Data, Trace } from '@/components/charts/plotly';

import { Button } from '@/components/ui/button';
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
import type { SankeyData } from '@/core/viz/sankey';
import type { ThematicMap } from '@/core/viz/thematic-map';
import type { Dataset } from '@/lib/types';
import { useAsyncResult } from '@/lib/use-async-result';
import { useLocale } from '@/state/locale.store';
import { getAnalyticsWorker } from '@/workers/client';
import { PALETTE, QUADRANT_NOTE, chartMessage } from './viz-shared';

const PlotlyChart = lazy(() => import('@/components/charts/PlotlyChart'));

/**
 * Segunda onda de visualizações, agrupada em sub-abas.
 *
 * As sub-abas não são organização decorativa: cada gráfico aqui custa uma passagem
 * completa pela base no worker, e o conteúdo de cada aba só é calculado quando ela é
 * aberta. Empilhar todos na Visão Geral dispararia sete análises pesadas de uma vez.
 */

type PanelKey = 'boxplot' | 'sankey' | 'genetics' | 'concept' | 'thematic' | 'historiograph';

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
  const [panel, setPanel] = useState<PanelKey>('boxplot');
  const t = useLocale((state) => state.t);

  return (
    <Card className="border-t-4 border-t-blue-500 shadow-xs">
      <CardHeader>
        <CardTitle className="text-base font-bold text-foreground">{t('visual_title')}</CardTitle>
        <CardDescription>
          {t('visual_description')}
        </CardDescription>
      </CardHeader>

      <CardContent>
        <Tabs value={panel} onValueChange={(value) => setPanel(value as PanelKey)}>
          <TabsList className="h-auto flex-wrap gap-1 bg-slate-100 dark:bg-slate-800/80 p-1.5">
            <TabsTrigger value="boxplot">{t('visual_tab_boxplot')}</TabsTrigger>
            <TabsTrigger value="sankey">{t('visual_tab_sankey')}</TabsTrigger>
            <TabsTrigger value="genetics">{t('visual_tab_genetics')}</TabsTrigger>
            <TabsTrigger value="concept">{t('visual_tab_concept')}</TabsTrigger>
            <TabsTrigger value="thematic">{t('visual_tab_thematic')}</TabsTrigger>
            <TabsTrigger value="historiograph">{t('visual_tab_historiograph')}</TabsTrigger>
          </TabsList>

          <TabsContent value="boxplot">
            {panel === 'boxplot' && <BoxplotPanel dataset={dataset} />}
          </TabsContent>
          <TabsContent value="sankey">
            {panel === 'sankey' && <SankeyPanel dataset={dataset} />}
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
        </Tabs>
      </CardContent>
    </Card>
  );
}

/** Distribuição estatística comparativa. */
function BoxplotPanel({ dataset }: { dataset: Dataset }) {
  const [dimension, setDimension] = useState<BoxplotDimension>('Países');
  const [metric, setMetric] = useState<BoxplotMetric>('Citações por documento');
  const [selected, setSelected] = useState<string[] | null>(null);
  const [logScale, setLogScale] = useState(false);

  const { data: options } = useAsyncResult<string[]>(`box-options ${dimension}`, () =>
    getAnalyticsWorker().boxplotOptions(dataset, dimension),
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
            <Suspense fallback={chartMessage('Carregando gráfico…')}>
              <PlotlyChart
                exportName="distribuicao-comparativa"
                height={440}
                data={(series ?? []).map((entry, index): Trace => ({
                  type: 'box',
                  name: entry.entity,
                  y: entry.values,
                  text: entry.labels,
                  // Todos os pontos visíveis: com poucas entidades, ver cada observação
                  // individual é o que revela os outliers que a caixa apenas resume.
                  boxpoints: 'all',
                  jitter: 0.4,
                  pointpos: 0,
                  marker: {
                    color: PALETTE[index % PALETTE.length] as string,
                    size: 4,
                    opacity: 0.6,
                  },
                  line: { color: PALETTE[index % PALETTE.length] as string },
                  hovertemplate: '%{text}<br>%{y}<extra>%{x}</extra>',
                }))}
                layout={{
                  showlegend: false,
                  yaxis: { title: { text: metric }, type: logScale ? 'log' : 'linear' },
                }}
              />
            </Suspense>
          )}
        </>
      )}
    </div>
  );
}

/** Fluxo de evolução temática entre três períodos. */
function SankeyPanel({ dataset }: { dataset: Dataset }) {
  const [topN, setTopN] = useState(10);

  const { data, loading } = useAsyncResult(`sankey ${topN}`, async () => {
    const worker = getAnalyticsWorker();
    const suggested = await worker.sankeyPeriods(dataset);
    if (!suggested) return null;
    return { periods: suggested, sankey: await worker.sankey(dataset, suggested, topN) };
  });

  const periods = data?.periods ?? null;
  const sankey: SankeyData | null = data?.sankey ?? null;

  if (loading) return chartMessage('Calculando fluxos temáticos…');
  if (!periods || !sankey) {
    return chartMessage('A base precisa de anos e palavras-chave para montar o fluxo.');
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end gap-3">
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

        <p className="text-xs text-muted-foreground">
          Períodos: {periods.map(([start, end]) => `${start}–${end}`).join(' · ')}. As linhas
          mais grossas são termos que sobreviveram de um período ao seguinte; as finas, termos
          distintos que costumam aparecer nos mesmos documentos.
        </p>
      </div>

      <Suspense fallback={chartMessage('Carregando gráfico…')}>
        <PlotlyChart
          exportName="evolucao-tematica"
          height={620}
          data={[
            {
              type: 'sankey',
              orientation: 'h',
              node: {
                pad: 14,
                thickness: 18,
                line: { color: 'rgba(0,0,0,0.25)', width: 0.5 },
                label: sankey.nodes.map((node) => node.label),
                color: sankey.nodes.map((node) => PALETTE[node.period % PALETTE.length] as string),
              },
              link: {
                source: sankey.links.map((link) => link.source),
                target: sankey.links.map((link) => link.target),
                value: sankey.links.map((link) => link.value),
                color: sankey.links.map((link) =>
                  link.kind === 'continuidade'
                    ? 'rgba(18, 115, 185, 0.45)'
                    : 'rgba(150, 160, 170, 0.25)',
                ),
              },
            } as never,
          ]}
          layout={{ margin: { l: 10, r: 10, t: 10, b: 10 } }}
        />
      </Suspense>
    </div>
  );
}

/** Ciclo de vida das palavras-chave. */
function GeneticsPanel({ dataset }: { dataset: Dataset }) {
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

  return (
    <div className="space-y-3">
      <p className="text-xs text-muted-foreground">
        Cada ponto é uma palavra-chave. O eixo X mostra quando ela apareceu pela primeira
        vez; o Y, por quantos anos permaneceu em uso; o tamanho, quantas vezes se replicou.
        Termos no alto e à esquerda são o núcleo estável da área; à direita e embaixo, as
        fronteiras recentes.
      </p>

      <Suspense fallback={chartMessage('Carregando gráfico…')}>
        <PlotlyChart
          exportName="genetica-das-ideias"
          height={480}
          data={[
            {
              type: 'scatter',
              mode: 'markers',
              x: top.map((item) => item.birthYear),
              y: top.map((item) => item.lifespan),
              text: top.map((item) => item.keyword),
              customdata: top.map((item) => [item.occurrences, item.citations]) as never,
              marker: {
                size: top.map((item) => Math.min(46, 8 + Math.sqrt(item.occurrences) * 3)),
                color: top.map((item) => item.citations),
                colorscale: 'Teal',
                showscale: true,
                colorbar: { title: { text: 'Citações' }, thickness: 12 },
                line: { width: 1, color: 'rgba(255,255,255,0.7)' },
                opacity: 0.85,
              },
              hovertemplate:
                '<b>%{text}</b><br>Nasceu em %{x}<br>Longevidade: %{y} anos' +
                '<br>Replicações: %{customdata[0]}<br>Citações: %{customdata[1]}<extra></extra>',
            },
          ] as Data[]}
          layout={{
            xaxis: { title: { text: 'Ano de nascimento do termo' } },
            yaxis: { title: { text: 'Longevidade (anos)' } },
          }}
        />
      </Suspense>
    </div>
  );
}

/** Mapa conceitual por PCA, em 2D e 3D. */
function ConceptPanel({ dataset }: { dataset: Dataset }) {
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

      <p className="text-xs text-muted-foreground">
        Termos próximos aparecem nos mesmos documentos. As ilhas são escolas de pensamento;
        os termos entre elas são pontes conceituais.
      </p>

      <Suspense fallback={chartMessage('Carregando gráfico…')}>
        <PlotlyChart
          exportName={`mapa-conceitual-${dimensions}`}
          height={dimensions === '3d' ? 620 : 500}
          data={groups.map((group, index) => {
            const members = terms.filter((term) => term.cluster === group);
            const base = {
              name: `Agrupamento ${group + 1}`,
              mode: 'text+markers' as const,
              text: members.map((term) => term.term),
              textposition: 'top center' as const,
              textfont: { size: 9 },
              marker: {
                size: members.map((term) => Math.min(34, 8 + Math.sqrt(term.frequency) * 2.5)),
                color: PALETTE[index % PALETTE.length] as string,
                line: { width: 1, color: 'rgba(255,255,255,0.8)' },
                opacity: 0.85,
              },
              x: members.map((term) => term.x),
              y: members.map((term) => term.y),
            };

            return dimensions === '3d'
              ? { ...base, type: 'scatter3d' as const, z: members.map((term) => term.z) }
              : { ...base, type: 'scatter' as const };
          }) as never}
          layout={
            dimensions === '3d'
              ? {
                  scene: {
                    xaxis: { title: { text: 'Dimensão 1' } },
                    yaxis: { title: { text: 'Dimensão 2' } },
                    zaxis: { title: { text: 'Dimensão 3' } },
                  },
                }
              : {
                  xaxis: { title: { text: 'Dimensão 1' } },
                  yaxis: { title: { text: 'Dimensão 2' } },
                }
          }
        />
      </Suspense>
    </div>
  );
}

/** Mapa temático de centralidade × densidade. */
function ThematicPanel({ dataset }: { dataset: Dataset }) {
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

      <p className="text-xs text-muted-foreground">{QUADRANT_NOTE}</p>

      <Suspense fallback={chartMessage('Carregando gráfico…')}>
        <PlotlyChart
          exportName="mapa-tematico"
          height={560}
          data={map.clusters.map((cluster, index): Trace => ({
            type: 'scatter',
            mode: 'text+markers',
            name: `Tema ${cluster.id}`,
            x: [cluster.centrality],
            y: [cluster.density],
            text: [cluster.label],
            textposition: 'middle center',
            textfont: { size: 10 },
            customdata: [[cluster.terms.join(', '), cluster.frequency]] as never,
            marker: {
              size: [Math.min(90, 20 + Math.sqrt(cluster.frequency) * 2)],
              color: PALETTE[index % PALETTE.length] as string,
              opacity: 0.55,
              line: { width: 1.5, color: 'rgba(255,255,255,0.9)' },
            },
            hovertemplate:
              '<b>Tema %{fullData.name}</b><br>%{customdata[0]}' +
              '<br>Centralidade: %{x:.0f}<br>Densidade: %{y:.0f}' +
              '<br>Frequência: %{customdata[1]}<extra></extra>',
          }))}
          layout={{
            showlegend: false,
            xaxis: { title: { text: 'Centralidade (relevância externa)' } },
            yaxis: { title: { text: 'Densidade (desenvolvimento interno)' } },
            shapes: [
              {
                type: 'line',
                x0: map.meanCentrality,
                x1: map.meanCentrality,
                yref: 'paper',
                y0: 0,
                y1: 1,
                line: { dash: 'dash', width: 1, color: 'rgba(128,128,128,0.5)' },
              },
              {
                type: 'line',
                xref: 'paper',
                x0: 0,
                x1: 1,
                y0: map.meanDensity,
                y1: map.meanDensity,
                line: { dash: 'dash', width: 1, color: 'rgba(128,128,128,0.5)' },
              },
            ],
            annotations: ([
              { x: 0.99, y: 0.99, text: '<b>Motores</b>', xanchor: 'right', yanchor: 'top' },
              { x: 0.01, y: 0.99, text: '<b>Nichos</b>', xanchor: 'left', yanchor: 'top' },
              {
                x: 0.99,
                y: 0.01,
                text: '<b>Básicos / transversais</b>',
                xanchor: 'right',
                yanchor: 'bottom',
              },
              {
                x: 0.01,
                y: 0.01,
                text: '<b>Emergentes / em declínio</b>',
                xanchor: 'left',
                yanchor: 'bottom',
              },
            ] as const).map((annotation) => ({
              ...annotation,
              xref: 'paper' as const,
              yref: 'paper' as const,
              showarrow: false,
              font: { size: 10, color: 'rgba(128,128,128,0.9)' },
            })),
          }}
        />
      </Suspense>
    </div>
  );
}

/** Linha do tempo de citações diretas. */
function HistoriographPanel({ dataset }: { dataset: Dataset }) {
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

  const positions = new Map(data.nodes.map((node) => [node.id, node]));

  // Uma única série de linhas com `null` entre segmentos: o Plotly interpreta o nulo como
  // quebra, o que desenha N arestas com um traço só em vez de N traços.
  const edgeX: (number | null)[] = [];
  const edgeY: (number | null)[] = [];

  for (const edge of data.edges) {
    const from = positions.get(edge.from);
    const to = positions.get(edge.to);
    if (!from || !to) continue;

    edgeX.push(from.year, to.year, null);
    edgeY.push(from.offset, to.offset, null);
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

        <p className="text-xs text-muted-foreground">
          {data.edges.length} citações diretas entre os {data.nodes.length} documentos mais
          citados. A detecção casa sobrenome do primeiro autor e ano dentro do texto das
          referências, então erra em homônimos e em grafias divergentes.
        </p>
      </div>

      <Suspense fallback={chartMessage('Carregando gráfico…')}>
        <PlotlyChart
          exportName="historiograph"
          height={560}
          data={[
            {
              type: 'scatter',
              mode: 'lines',
              x: edgeX,
              y: edgeY,
              line: { width: 1, color: 'rgba(130,140,150,0.55)' },
              hoverinfo: 'skip',
              showlegend: false,
            },
            {
              type: 'scatter',
              mode: 'text+markers',
              x: data.nodes.map((node) => node.year),
              y: data.nodes.map((node) => node.offset),
              text: data.nodes.map((node) => node.id),
              textposition: 'top center',
              textfont: { size: 9 },
              customdata: data.nodes.map((node) => [node.title, node.citations]) as never,
              marker: {
                size: data.nodes.map((node) => node.size / 2),
                color: '#1273B9',
                opacity: 0.8,
                line: { width: 1, color: 'white' },
              },
              hovertemplate: '<b>%{customdata[0]}</b><br>%{customdata[1]} citações<extra></extra>',
              showlegend: false,
            },
          ] as Data[]}
          layout={{
            xaxis: { title: { text: 'Linha do tempo' }, dtick: 1 },
            yaxis: { showticklabels: false, showgrid: false, zeroline: false },
          }}
        />
      </Suspense>
    </div>
  );
}
