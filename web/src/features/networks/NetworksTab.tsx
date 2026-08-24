import { lazy, Suspense, useEffect, useMemo, useState } from 'react';
import type { ColumnDef } from '@tanstack/react-table';
import { Info } from 'lucide-react';

import { DataTable } from '@/components/DataTable';
import { Badge } from '@/components/ui/badge';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import type { CooccurrenceKind, SizeMetric } from '@/core/graph';
import type { GlobalMetrics } from '@/core/graph/metrics';
import type { SnaNodeMetrics } from '@/lib/types';
import { useDataset } from '@/state/dataset.store';
import { useLocale } from '@/state/locale.store';
import { EmptyState } from '@/features/EmptyState';
import { CollaborationPanel } from './CollaborationPanel';

const SigmaGraph = lazy(() => import('@/components/charts/SigmaGraph'));

const NETWORK_KINDS: CooccurrenceKind[] = ['Coautoria', 'Palavras-chave', 'Países'];
const TOP_N_OPTIONS = [20, 30, 50, 75, 100] as const;
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

function formatMetric(value: number | string): string {
  if (typeof value === 'string') return value;
  if (!Number.isFinite(value)) return '—';
  if (value === 0) return '0';
  return Math.abs(value) < 0.001 ? value.toExponential(2) : value.toFixed(Math.abs(value) < 1 ? 4 : 2);
}

function NetworkMetricCard({
  label,
  value,
  hint,
}: {
  label: string;
  value: number | string;
  hint: string;
}) {
  const [showTooltip, setShowTooltip] = useState(false);

  return (
    <div className="relative rounded-xl border border-border/80 bg-gradient-to-br from-purple-500/[0.04] via-card to-card p-3 shadow-2xs transition-all hover:border-purple-300">
      <div className="flex items-center justify-between gap-1.5">
        <dt className="text-xs font-semibold text-muted-foreground truncate" title={hint}>
          {label}
        </dt>
        <div
          className="relative inline-flex items-center shrink-0"
          onMouseEnter={() => setShowTooltip(true)}
          onMouseLeave={() => setShowTooltip(false)}
        >
          <button
            type="button"
            className="text-muted-foreground/60 transition-colors hover:text-purple-600 focus:outline-hidden"
            title={hint}
            aria-label={hint}
            onClick={(e) => {
              e.stopPropagation();
              setShowTooltip((prev) => !prev);
            }}
          >
            <Info className="size-3.5" />
          </button>

          {showTooltip && (
            <div className="pointer-events-none absolute right-0 bottom-full mb-2 z-50 w-52 rounded-lg border border-border/90 bg-popover p-2.5 text-[11px] font-normal normal-case leading-snug text-popover-foreground shadow-xl backdrop-blur-xs animate-in fade-in-0 zoom-in-95">
              <p>{hint}</p>
            </div>
          )}
        </div>
      </div>
      <dd className="mt-1 text-base font-bold tabular-nums text-foreground">
        {formatMetric(value)}
      </dd>
    </div>
  );
}

export default function NetworksTab() {
  const active = useDataset((state) => state.active);
  const sna = useDataset((state) => state.sna);
  const network = useDataset((state) => state.network);
  const progress = useDataset((state) => state.progress);
  const computeSna = useDataset((state) => state.computeSna);
  const computeNetwork = useDataset((state) => state.computeNetwork);
  const { t, locale } = useLocale();
  const isEn = locale === 'en';

  const [kind, setKind] = useState<CooccurrenceKind>('Coautoria');
  const [topN, setTopN] = useState<number>(50);
  const [sizeMetric, setSizeMetric] = useState<SizeMetric>('Grau Absoluto');

  useEffect(() => {
    if (active) void computeSna();
  }, [active, computeSna]);

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
            <span className="block max-w-96 truncate font-medium" title={String(row.original['item'])}>
              {String(row.original['item'])}
            </span>
          ),
        },
        {
          accessorKey: 'kind',
          header: 'Tipo / Type',
          cell: ({ row }) => {
            const val = String(row.original['kind']);
            const variant =
              val === 'Autor'
                ? 'purple'
                : val === 'País'
                  ? 'indigo'
                  : val === 'Venue'
                    ? 'cyan'
                    : 'blue';
            return <Badge variant={variant}>{val}</Badge>;
          },
        },
        { accessorKey: 'degreeAbsolute', header: 'Grau absoluto' },
        { accessorKey: 'degreeCentrality', header: 'Centralidade de grau' },
        { accessorKey: 'eigenvector', header: 'Autovetor' },
        { accessorKey: 'betweenness', header: 'Betweenness' },
        { accessorKey: 'closeness', header: 'Closeness' },
      ] as ColumnDef<Record<string, unknown>, unknown>[],
    [],
  );

  if (!active) {
    return <EmptyState title={t('tab_networks')} />;
  }

  return (
    <div className="space-y-4">
      {progress && (
        <Card>
          <CardContent className="space-y-1.5 pt-6">
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>{progress.phase}</span>
              <span className="tabular-nums">{Math.round(progress.ratio * 100)}%</span>
            </div>
            <Progress value={progress.ratio * 100} />
          </CardContent>
        </Card>
      )}

      {sna && (
        <Card className="border-t-4 border-t-purple-500 shadow-xs">
          <CardHeader>
            <CardTitle className="text-base font-bold text-foreground">{t('network_deep_title')}</CardTitle>
            <CardDescription>
              {t('network_deep_desc')}{' '}
              <strong className="text-foreground">{sna.global.nodeCount.toLocaleString('pt-BR')}</strong> {t('network_nodes')},{' '}
              <strong className="text-foreground">{sna.global.edgeCount.toLocaleString('pt-BR')}</strong> {t('network_edges')} e{' '}
              <strong className="text-foreground">{sna.global.componentCount.toLocaleString('pt-BR')}</strong> {t('network_components')}.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <dl className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
              {METRIC_LABELS.map(({ key, label, labelEn, hint, hintEn }) => (
                <NetworkMetricCard
                  key={key}
                  label={isEn ? labelEn : label}
                  value={sna.global[key] as number | string}
                  hint={isEn ? hintEn : hint}
                />
              ))}
            </dl>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="text-base">{t('network_cooccurrence_title')}</CardTitle>
          <CardDescription>
            {t('network_cooccurrence_desc')}
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-4">
          <div className="grid gap-3 sm:grid-cols-3">
            <div className="space-y-1.5">
              <Label htmlFor="network-kind">{t('network_kind_label')}</Label>
              <Select value={kind} onValueChange={(value) => setKind(value as CooccurrenceKind)}>
                <SelectTrigger id="network-kind">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {NETWORK_KINDS.map((option) => (
                    <SelectItem key={option} value={option}>
                      {option}
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
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {SIZE_METRICS.map((option) => (
                    <SelectItem key={option} value={option}>
                      {option}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          {network && (
            <>
              <p className="text-xs text-muted-foreground">
                {network.nodes.length} nós, {network.edges.length} arestas,{' '}
                {network.communityCount} comunidades.
              </p>
              <Suspense
                fallback={
                  <div className="grid h-[560px] place-items-center rounded-lg border text-sm text-muted-foreground">
                    Carregando renderizador…
                  </div>
                }
              >
                <SigmaGraph nodes={network.nodes} edges={network.edges} />
              </Suspense>
            </>
          )}
        </CardContent>
      </Card>

      <CollaborationPanel dataset={active} />

      {sna && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">{t('network_nodes_metrics_title')}</CardTitle>
            <CardDescription>
              {t('network_nodes_metrics_desc')}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <DataTable
              data={sna.nodes as unknown as Record<string, unknown>[]}
              columns={snaColumns}
              exportName="metricas-sna"
              filterPlaceholder={t('table_filter_placeholder')}
            />
          </CardContent>
        </Card>
      )}
    </div>
  );
}

export type { SnaNodeMetrics };
