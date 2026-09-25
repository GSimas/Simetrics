import { lazy, Suspense, useEffect, useMemo, useState } from 'react';
import type { ColumnDef } from '@tanstack/react-table';
import { ChevronDown } from 'lucide-react';

import { Collapse } from '@/components/Collapse';
import { ErrorBoundary } from '@/components/ErrorBoundary';
import { DataTable } from '@/components/DataTable';
import { EntityChip } from '@/components/EntityChip';
import { InfoTip, SectionTitle } from '@/components/InfoTip';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import type { CooccurrenceKind, SizeMetric } from '@/core/graph';
import { localizeMetricText, type GlobalMetrics } from '@/core/graph/metrics';
import type { NodeKind, SearchEntityType, SnaNodeMetrics } from '@/lib/types';
import { cooccurrenceLabel, entityTypeLabel, numberLocale, sizeMetricLabel } from '@/lib/i18n/labels';
import type { Locale } from '@/lib/i18n/translations';
import { localizeProgress } from '@/lib/i18n/progress';
import { communityColor } from '@/features/overview/viz-shared';
import { useStickyValue } from '@/lib/use-sticky-value';
import { cn } from '@/lib/utils';
import { useDataset } from '@/state/dataset.store';
import { useLocale } from '@/state/locale.store';
import { openInSearch } from '@/state/navigation.store';
import { EmptyState } from '@/features/EmptyState';
import { CollaborationPanel } from './CollaborationPanel';

const SigmaGraph = lazy(() => import('@/components/charts/SigmaGraph'));
const RadialGraph = lazy(() => import('@/components/charts/RadialGraph'));

const NETWORK_KINDS: CooccurrenceKind[] = ['Coautoria', 'Palavras-chave', 'Países'];
const TOP_N_OPTIONS = [20, 30, 50, 75, 100] as const;

/** Tipo de entidade do Motor de Busca para cada tipo de nó do grafo heterogêneo. */
const SNA_SEARCH_TYPES: Record<string, SearchEntityType[]> = {
  Documento: ['Documento'],
  Autor: ['Autor'],
  País: ['País'],
  Venue: ['Local de Publicação (Venue)'],
  'Local de Publicação (Venue)': ['Local de Publicação (Venue)'],
};

/** Tipo de entidade do Motor de Busca para os nós de cada rede. */
const NETWORK_SEARCH_TYPES: Record<CooccurrenceKind, SearchEntityType[]> = {
  Coautoria: ['Autor'],
  'Palavras-chave': ['Palavra-chave'],
  Países: ['País'],
};
const SIZE_METRICS: SizeMetric[] = [
  'Tamanho Fixo',
  'Grau Absoluto',
  'Centralidade (Eigen)',
  'Betweenness',
  'Closeness',
];

const METRIC_LABELS: {
  key: keyof GlobalMetrics;
  label: string;
  labelEn: string;
  hint: string;
  hintEn: string;
}[] = [
  {
    key: 'density',
    label: 'Densidade',
    labelEn: 'Density',
    hint: 'Proporção de arestas existentes sobre todas as possíveis (0 a 1). Mede o quão integrada e coesa é a rede.',
    hintEn: 'Ratio of actual edges to all possible edges (0 to 1). Measures overall network cohesion.',
  },
  {
    key: 'clustering',
    label: 'Clustering médio',
    labelEn: 'Avg Clustering',
    hint: 'Tendência dos vizinhos de um nó também estarem conectados entre si (formação de triângulos). Indica densidade de grupos locais.',
    hintEn: 'Tendency of nodes to cluster together into tightly-knit groups or triangles.',
  },
  {
    key: 'entropy',
    label: 'Entropia de Shannon',
    labelEn: 'Shannon Entropy',
    hint: 'Mede o grau de incerteza ou desordem na distribuição de conexões. Valores altos indicam conectividade homogênea e distribuída.',
    hintEn: 'Degree distribution uncertainty. High values indicate distributed, decentralized connectivity.',
  },
  {
    key: 'efficiency',
    label: 'Eficiência global',
    labelEn: 'Global Efficiency',
    hint: 'Média do inverso dos caminhos mais curtos. Mede a rapidez e facilidade de tráfego de informação entre os nós da rede.',
    hintEn: 'Average inverse shortest path length. Quantifies how efficiently information traverses the network.',
  },
  {
    key: 'meanDegree',
    label: 'Grau médio',
    labelEn: 'Mean Degree',
    hint: 'Número médio de colaborações, citações ou relações diretas que cada nó possui.',
    hintEn: 'Average number of direct connections per node.',
  },
  {
    key: 'stdDegree',
    label: 'Desvio do grau',
    labelEn: 'Degree Std Dev',
    hint: 'Dispersão da conectividade. Valores altos revelam disparidades entre super-hubs e nós periféricos.',
    hintEn: 'Degree dispersion. High values indicate large gaps between central hubs and peripheral nodes.',
  },
  {
    key: 'meanPageRank',
    label: 'PageRank médio',
    labelEn: 'Mean PageRank',
    hint: 'Prestígio acadêmico médio. Nós conectados a outros nós influentes recebem maior pontuação.',
    hintEn: 'Average prestige centrality across nodes in the network.',
  },
  {
    key: 'meanEigenvector',
    label: 'Autovetor médio',
    labelEn: 'Mean Eigenvector',
    hint: 'Proximidade média aos principais centros (hubs) da rede.',
    hintEn: 'Average closeness and influence relative to primary network hubs.',
  },
  {
    key: 'assortativity',
    label: 'Assortatividade',
    labelEn: 'Assortativity',
    hint: 'Correlação entre graus de nós conectados. Negativa indica que grandes hubs se conectam predominantemente a nós menores.',
    hintEn: 'Degree correlation of linked nodes. Negative values mean hubs connect mostly to peripheral nodes.',
  },
  {
    key: 'powerLawExponent',
    label: 'Lei de potência',
    labelEn: 'Power Law Exponent',
    hint: 'Expoente da cauda longa. Valores entre 2 e 3 indicam uma rede livre de escala (Scale-Free) dominada por poucos super-hubs.',
    hintEn: 'Scale-free exponent. Values between 2 and 3 characterize heavy-tailed scale-free networks.',
  },
  {
    key: 'spearmanDegreeBetweenness',
    label: 'Spearman grau×ponte',
    labelEn: 'Degree-Betweenness Corr',
    hint: 'Correlação entre grau e intermediação. Alta correlação indica que os autores mais conectados também são as principais pontes entre grupos.',
    hintEn: 'Correlation between node degree and bridge centrality (betweenness).',
  },
];

function formatMetric(value: number | string, locale: Locale): string {
  if (typeof value === 'string') return String(localizeMetricText(value, locale));
  if (!Number.isFinite(value)) return '—';
  if (value === 0) return '0';
  return Math.abs(value) < 0.001 ? value.toExponential(2) : value.toFixed(Math.abs(value) < 1 ? 4 : 2);
}

function NetworkMetricCard({
  label,
  value,
  hint,
  locale,
}: {
  label: string;
  value: number | string;
  hint: string;
  locale: Locale;
}) {
  return (
    <div className="border border-border bg-card p-4 transition-colors hover:border-highlight/60">
      {/* O `dt` é a própria linha flexível: um `dt` dentro de outra div invalida a
          lista de definição para leitores de tela. */}
      <dt className="flex items-center justify-between gap-1.5">
        <span className="eyebrow truncate">{label}</span>
        <InfoTip label={label}>{hint}</InfoTip>
      </dt>
      <dd className="mt-2 text-xl font-medium tabular-nums tracking-tight text-foreground">
        {formatMetric(value, locale)}
      </dd>
    </div>
  );
}

/** Quantas métricas ficam à vista antes do "ver todas". */
const VISIBLE_METRICS = 4;

/**
 * Ecologia profunda da rede: o resumo (nós, arestas, componentes) e as primeiras
 * métricas ficam à vista; as demais abrem num collapse, para a aba não começar com uma
 * parede de números.
 */
function NetworkEcology({ global }: { global: GlobalMetrics }) {
  const { t, locale } = useLocale();
  const isEn = locale === 'en';
  const [expanded, setExpanded] = useState(false);

  const cards = METRIC_LABELS.map(({ key, label, labelEn, hint, hintEn }) => (
    <NetworkMetricCard
      key={key}
      label={isEn ? labelEn : label}
      value={global[key] as number | string}
      hint={isEn ? hintEn : hint}
      locale={locale}
    />
  ));
  const hidden = cards.length - VISIBLE_METRICS;

  return (
    <Card data-tour="net-ecology">
      <CardHeader className="space-y-1.5">
        <SectionTitle
          title={t('network_deep_title')}
          info={t('network_deep_desc').replace(/:\s*$/, '.')}
        />
        <p className="eyebrow">
          {global.nodeCount.toLocaleString(numberLocale(locale))} {t('network_nodes')} ·{' '}
          {global.edgeCount.toLocaleString(numberLocale(locale))} {t('network_edges')} ·{' '}
          {global.componentCount.toLocaleString(numberLocale(locale))} {t('network_components')}
        </p>
      </CardHeader>
      <CardContent>
        <dl className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
          {cards.slice(0, VISIBLE_METRICS)}
        </dl>

        {hidden > 0 && (
          <>
            <Collapse open={expanded} delayOpen={false}>
              <dl className="grid grid-cols-2 gap-3 pt-3 sm:grid-cols-3 lg:grid-cols-4">
                {cards.slice(VISIBLE_METRICS)}
              </dl>
            </Collapse>
            <button
              type="button"
              onClick={() => setExpanded((open) => !open)}
              aria-expanded={expanded}
              className="mt-3 inline-flex cursor-pointer items-center gap-1.5 border-b border-border pb-0.5 text-sm transition-colors hover:border-highlight hover:text-highlight"
            >
              {expanded ? t('network_show_less') : `${t('network_show_all_metrics')} (+${hidden})`}
              <ChevronDown
                className={cn('size-4 transition-transform duration-300', expanded && 'rotate-180')}
                aria-hidden
              />
            </button>
          </>
        )}
      </CardContent>
    </Card>
  );
}

/** Espaço do renderizador enquanto o chunk do grafo chega. */
function GraphFallback() {
  const t = useLocale((state) => state.t);
  return (
    <div className="grid h-[560px] place-items-center border text-sm text-muted-foreground" aria-busy="true">
      {t('network_loading_renderer')}
    </div>
  );
}

/**
 * Barra de progresso do cálculo de SNA. Componente próprio: o progresso muda a cada
 * poucas centenas de linhas processadas, e assinado na raiz da aba cada tique
 * re-renderizava o grafo radial, o mapa e a tabela de métricas.
 */
function SnaProgress() {
  const t = useLocale((state) => state.t);
  const locale = useLocale((state) => state.locale);
  const snaProgress = useDataset((state) => state.snaProgress);
  const sticky = useStickyValue(snaProgress).value;
  const shownProgress = sticky && localizeProgress(sticky, locale);
  return (
    // Fechada, a barra zera a própria margem do space-y (mb-0 vence o :where do
    // Tailwind) para não deixar um vão acima dos blocos.
    <Collapse open={snaProgress !== null} className={snaProgress ? '' : 'mb-0'}>
      {shownProgress && (
        <Card>
          <CardContent className="space-y-1.5 pt-6">
            <div className="flex justify-between text-xs text-muted-foreground" aria-live="polite">
              <span>{shownProgress.phase}</span>
              <span className="tabular-nums">{Math.round(shownProgress.ratio * 100)}%</span>
            </div>
            <Progress value={shownProgress.ratio * 100} aria-label={`${t('network_progress_label')}: ${shownProgress.phase}`} />
          </CardContent>
        </Card>
      )}
    </Collapse>
  );
}

export default function NetworksTab() {
  const active = useDataset((state) => state.active);
  const { value: sna } = useStickyValue(useDataset((state) => state.sna));
  const network = useDataset((state) => state.network);
  const computeSna = useDataset((state) => state.computeSna);
  const computeNetwork = useDataset((state) => state.computeNetwork);
  const { t, locale } = useLocale();

  const [kind, setKind] = useState<CooccurrenceKind>('Coautoria');
  const [topN, setTopN] = useState<number>(50);
  const [sizeMetric, setSizeMetric] = useState<SizeMetric>('Grau Absoluto');
  const [graphLayout, setGraphLayout] = useState<'force' | 'radial'>('force');

  const openNode = (key: string): void => {
    openInSearch(network?.nodes.find((node) => node.key === key)?.label ?? key, NETWORK_SEARCH_TYPES[kind]);
  };

  useEffect(() => {
    if (active) void computeSna();
  }, [active, computeSna]);

  // Memoizados: um array novo a cada render refazia o layout do diagrama de cordas.
  const radialNodes = useMemo(
    () =>
      (network?.nodes ?? []).map((node) => ({
        key: node.key,
        label: node.label,
        weight: node.count,
        group: node.community,
        color: communityColor(node.community),
      })),
    [network],
  );
  const radialLegend = useMemo(
    () =>
      // Com muitas comunidades a legenda viraria uma parede de rótulos.
      network && network.communityCount <= 8
        ? Array.from({ length: network.communityCount }, (_, index) => ({
            label: `${t('radial_communities')} ${index + 1}`,
            color: communityColor(index),
          }))
        : undefined,
    [network, t],
  );

  useEffect(() => {
    if (active) void computeNetwork(kind, topN, sizeMetric);
  }, [active, kind, topN, sizeMetric, computeNetwork]);

  const snaColumns = useMemo<ColumnDef<Record<string, unknown>, unknown>[]>(
    () =>
      [
        {
          accessorKey: 'item',
          header: 'Item',
          cell: ({ row }) => (
            <EntityChip
              label={String(row.original['item'])}
              types={SNA_SEARCH_TYPES[String(row.original['kind'])] ?? []}
            />
          ),
        },
        {
          accessorKey: 'kind',
          header: locale === 'en' ? 'Type' : 'Tipo',
          cell: ({ row }) => {
            const val = String(row.original['kind']) as NodeKind | 'Outro';
            const variant =
              val === 'Autor'
                ? 'purple'
                : val === 'País'
                  ? 'indigo'
                  : val === 'Local de Publicação (Venue)'
                    ? 'cyan'
                    : 'blue';
            return <Badge variant={variant}>{entityTypeLabel(val, locale)}</Badge>;
          },
        },
        { accessorKey: 'degreeAbsolute', header: locale === 'en' ? 'Absolute degree' : 'Grau absoluto' },
        { accessorKey: 'degreeCentrality', header: locale === 'en' ? 'Degree centrality' : 'Centralidade de grau' },
        { accessorKey: 'eigenvector', header: locale === 'en' ? 'Eigenvector' : 'Autovetor' },
        { accessorKey: 'betweenness', header: 'Betweenness' },
        { accessorKey: 'closeness', header: 'Closeness' },
      ] as ColumnDef<Record<string, unknown>, unknown>[],
    [locale],
  );

  if (!active) {
    return <EmptyState title={t('tab_networks')} />;
  }

  return (
    <div className="space-y-6">
      <SnaProgress />

        {sna && <NetworkEcology global={sna.global} />}

        {/* Coocorrência e colaboração lado a lado em telas largas. */}
        <div className="grid items-start gap-6 lg:grid-cols-2">
          <Card data-tour="net-cooccurrence">
            <CardHeader>
              <SectionTitle title={t('network_cooccurrence_title')} info={t('network_cooccurrence_desc')} />
            </CardHeader>
            <CardContent className="space-y-4">
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="space-y-1.5">
                <Label htmlFor="network-kind">{t('network_kind_label')}</Label>
                <Select value={kind} onValueChange={(value) => setKind(value as CooccurrenceKind)}>
                  <SelectTrigger id="network-kind">
                    <SelectValue>{cooccurrenceLabel(kind, locale)}</SelectValue>
                  </SelectTrigger>
                  <SelectContent>
                    {NETWORK_KINDS.map((option) => (
                      <SelectItem key={option} value={option}>
                        {cooccurrenceLabel(option, locale)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-1.5">
                <Label htmlFor="network-top">{t('network_top_label')}</Label>
                <Select value={String(topN)} onValueChange={(value) => setTopN(Number(value))}>
                  <SelectTrigger id="network-top">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {TOP_N_OPTIONS.map((option) => (
                      <SelectItem key={option} value={String(option)}>
                        Top {option}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-1.5">
                <Label htmlFor="network-size">{t('network_size_label')}</Label>
                <Select
                  value={sizeMetric}
                  onValueChange={(value) => setSizeMetric(value as SizeMetric)}
                >
                  <SelectTrigger id="network-size">
                    <SelectValue>{sizeMetricLabel(sizeMetric, locale)}</SelectValue>
                  </SelectTrigger>
                  <SelectContent>
                    {SIZE_METRICS.map((option) => (
                      <SelectItem key={option} value={option}>
                        {sizeMetricLabel(option, locale)}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

            </div>

            {network && (
              <>
                <p className="eyebrow">
                  {network.nodes.length} {t('network_nodes')} · {network.edges.length}{' '}
                  {t('network_edges')} · {network.communityCount} {locale === 'en' ? 'communities' : 'comunidades'}
                </p>
                <Tabs
                  data-tour="net-graph"
                  value={graphLayout}
                  onValueChange={(value) => setGraphLayout(value as 'force' | 'radial')}
                >
                  <TabsList className="w-full justify-start">
                    <TabsTrigger value="force">{t('network_layout_force')}</TabsTrigger>
                    <TabsTrigger value="radial">{t('network_radial_tab')}</TabsTrigger>
                  </TabsList>
                  {/* Suspense dentro de cada painel: fora, o painel ativo sumia do DOM enquanto o
                      renderizador baixava e a aba apontava (aria-controls) para um id inexistente. */}
                  <TabsContent value="force">
                    <ErrorBoundary label={t('error_chart')} resetKeys={[network]}>
                      <Suspense fallback={<GraphFallback />}>
                        <SigmaGraph
                          nodes={network.nodes}
                          edges={network.edges}
                          onNodeClick={openNode}
                          exportName={`${locale === 'en' ? 'network' : 'rede'}-${cooccurrenceLabel(kind, locale)}`}
                        />
                      </Suspense>
                    </ErrorBoundary>
                  </TabsContent>
                  <TabsContent value="radial">
                    <ErrorBoundary label={t('error_chart')} resetKeys={[network]}>
                      <Suspense fallback={<GraphFallback />}>
                        <RadialGraph
                          nodes={radialNodes}
                          edges={network.edges}
                          weightLabel={t('radial_documents')}
                          legend={radialLegend}
                          onNodeClick={openNode}
                          exportName={`${locale === 'en' ? 'radial-network' : 'rede-radial'}-${cooccurrenceLabel(kind, locale)}`}
                        />
                      </Suspense>
                    </ErrorBoundary>
                  </TabsContent>
                </Tabs>
              </>
            )}
            </CardContent>
          </Card>

          <Card data-tour="net-collab">
            <CardHeader>
              <SectionTitle title={t('network_collab_title')} info={t('network_collab_desc')} />
            </CardHeader>
            <CardContent className="space-y-4">
              <CollaborationPanel dataset={active} />
            </CardContent>
          </Card>
        </div>

        {sna && (
          <Card data-tour="net-metrics">
            <CardHeader>
              <SectionTitle title={t('network_nodes_metrics_title')} info={t('network_nodes_metrics_desc')} />
            </CardHeader>
            <CardContent className="space-y-4">
            <DataTable
              data={sna.nodes as unknown as Record<string, unknown>[]}
              columns={snaColumns}
              exportName={locale === 'en' ? 'sna-metrics' : 'metricas-sna'}
              filterPlaceholder={t('table_filter_placeholder')}
            />
            </CardContent>
          </Card>
        )}
    </div>
  );
}

export type { SnaNodeMetrics };
