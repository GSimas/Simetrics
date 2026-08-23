import { lazy, Suspense, useEffect, useMemo, useState } from 'react';
import type { ColumnDef } from '@tanstack/react-table';

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
import { EmptyState } from '@/features/EmptyState';
import { CollaborationPanel } from './CollaborationPanel';

// Sigma e Graphology somam ~170 kB; só quem abre esta aba paga por eles.
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

/** Métricas globais com o rótulo e a leitura que cada uma pede. */
const METRIC_LABELS: { key: keyof GlobalMetrics; label: string; hint: string }[] = [
  { key: 'density', label: 'Densidade', hint: 'Arestas existentes sobre as possíveis' },
  { key: 'clustering', label: 'Clustering médio', hint: 'Tendência a formar triângulos' },
  { key: 'entropy', label: 'Entropia de Shannon', hint: 'Desordem na distribuição de graus' },
  { key: 'efficiency', label: 'Eficiência global', hint: 'Média do inverso das distâncias' },
  { key: 'meanDegree', label: 'Grau médio', hint: 'Conexões por nó' },
  { key: 'stdDegree', label: 'Desvio do grau', hint: 'Dispersão da conectividade' },
  { key: 'meanPageRank', label: 'PageRank médio', hint: 'Influência média' },
  { key: 'meanEigenvector', label: 'Autovetor médio', hint: 'Proximidade média aos hubs' },
  { key: 'assortativity', label: 'Assortatividade', hint: 'Negativa: hubs cercados de periféricos' },
  { key: 'powerLawExponent', label: 'Lei de potência', hint: 'Entre 2 e 3 indica rede livre de escala' },
  {
    key: 'spearmanDegreeBetweenness',
    label: 'Spearman grau×ponte',
    hint: 'Alta: quem tem muitos links também é ponte',
  },
];

function formatMetric(value: number | string): string {
  if (typeof value === 'string') return value;
  if (!Number.isFinite(value)) return '—';
  if (value === 0) return '0';
  return Math.abs(value) < 0.001 ? value.toExponential(2) : value.toFixed(Math.abs(value) < 1 ? 4 : 2);
}

export default function NetworksTab() {
  const active = useDataset((state) => state.active);
  const sna = useDataset((state) => state.sna);
  const network = useDataset((state) => state.network);
  const progress = useDataset((state) => state.progress);
  const computeSna = useDataset((state) => state.computeSna);
  const computeNetwork = useDataset((state) => state.computeNetwork);

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
          header: 'Tipo',
          cell: ({ row }) => <Badge variant="outline">{String(row.original['kind'])}</Badge>,
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
    return <EmptyState title="Redes e Grafos de Conhecimento" />;
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
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Ecologia profunda da rede</CardTitle>
            <CardDescription>
              Grafo heterogêneo ligando documentos a autores, países e venues:{' '}
              {sna.global.nodeCount.toLocaleString('pt-BR')} nós,{' '}
              {sna.global.edgeCount.toLocaleString('pt-BR')} arestas e{' '}
              {sna.global.componentCount.toLocaleString('pt-BR')} componentes.{' '}
              {sna.betweennessExact
                ? 'Betweenness calculado de forma exata.'
                : 'Betweenness estimado por amostragem determinística, dado o tamanho do grafo.'}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <dl className="grid grid-cols-2 gap-x-6 gap-y-3 sm:grid-cols-3 lg:grid-cols-4">
              {METRIC_LABELS.map(({ key, label, hint }) => (
                <div key={key}>
                  <dt className="text-xs text-muted-foreground" title={hint}>
                    {label}
                  </dt>
                  <dd className="text-sm font-medium tabular-nums">
                    {formatMetric(sna.global[key] as number | string)}
                  </dd>
                </div>
              ))}
            </dl>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Rede de coocorrência</CardTitle>
          <CardDescription>
            Entidades conectadas por aparecerem no mesmo documento. A espessura da aresta
            reflete a frequência da coocorrência; a cor, a comunidade detectada pelo
            Louvain.
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-4">
          <div className="grid gap-3 sm:grid-cols-3">
            <div className="space-y-1.5">
              <Label htmlFor="network-kind">Tipo de rede</Label>
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
              <Label htmlFor="network-top">Entidades exibidas</Label>
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
              <Label htmlFor="network-size">Tamanho dos nós</Label>
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
                {network.nodes.length} entidades, {network.edges.length} conexões,{' '}
                {network.communityCount} comunidades. Passe o cursor sobre um nó para ver
                suas métricas.
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
            <CardTitle className="text-base">Métricas por nó</CardTitle>
            <CardDescription>
              Todos os {sna.nodes.length.toLocaleString('pt-BR')} nós do grafo heterogêneo,
              ordenados por grau.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <DataTable
              data={sna.nodes as unknown as Record<string, unknown>[]}
              columns={snaColumns}
              exportName="metricas-sna"
              filterPlaceholder="Filtrar nós…"
            />
          </CardContent>
        </Card>
      )}
    </div>
  );
}

export type { SnaNodeMetrics };
