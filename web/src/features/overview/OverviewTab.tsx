import { lazy, Suspense, useEffect } from 'react';
import {
  BookOpen,
  Building2,
  CalendarRange,
  Copy,
  Globe2,
  Quote,
  TrendingUp,
  Users,
} from 'lucide-react';

import { KpiCard } from '@/components/KpiCard';
import { UploadPanel } from '@/components/UploadPanel';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import type { MetadataCompleteness } from '@/lib/types';
import { useDataset, type DedupStrategy } from '@/state/dataset.store';
import { EntityTables } from './EntityTables';
import { ThemePanel } from './ThemePanel';
import { VisualAnalyses } from './VisualAnalyses';

// Plotly pesa mais de 1 MB. Carregar sob demanda mantém a abertura do app leve para quem
// ainda nem subiu uma base.
const PlotlyChart = lazy(() => import('@/components/charts/PlotlyChart'));

const STATUS_VARIANT: Record<MetadataCompleteness['status'], 'success' | 'default' | 'warning' | 'destructive'> = {
  Excelente: 'success',
  Bom: 'default',
  Aceitável: 'warning',
  Ruim: 'destructive',
};

const DEDUP_LABELS: Record<DedupStrategy, string> = {
  none: 'Base completa',
  doi: 'Deduplicar por DOI',
  similarity: 'Deduplicar por similaridade',
};

export default function OverviewTab() {
  const active = useDataset((state) => state.active);
  const overview = useDataset((state) => state.overview);
  const tables = useDataset((state) => state.tables);
  const duplicates = useDataset((state) => state.duplicates);
  const dedupStrategy = useDataset((state) => state.dedupStrategy);
  const computeOverview = useDataset((state) => state.computeOverview);
  const computeTables = useDataset((state) => state.computeTables);
  const applyDedup = useDataset((state) => state.applyDedup);
  const busy = useDataset((state) => state.progress !== null);

  // As derivações são disparadas ao entrar na aba, e o store ignora o pedido se já tiver
  // o resultado — trocar de aba não recalcula.
  useEffect(() => {
    if (!active) return;
    void computeOverview();
    void computeTables();
  }, [active, computeOverview, computeTables]);

  if (!active) {
    return (
      <div className="space-y-4">
        <UploadPanel />
        <Card>
          <CardHeader>
            <CardTitle>Comece por aqui</CardTitle>
            <CardDescription>
              O Simetrics transforma exports de bases bibliográficas em indicadores
              cientométricos, redes de conhecimento e mapeamento temático. Envie seus
              arquivos acima ou carregue a base de exemplo para explorar.
            </CardDescription>
          </CardHeader>
          <CardContent className="text-sm text-muted-foreground">
            Todo o processamento acontece no seu navegador — os documentos não são
            enviados para nenhum servidor.
          </CardContent>
        </Card>
      </div>
    );
  }

  const summary = overview?.summary;
  const metrics = summary?.bibliometrix;

  return (
    <div className="space-y-4">
      <UploadPanel />

      {summary && metrics && (
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <KpiCard
            title="Documentos"
            value={summary.totalDocs}
            subtitle={`Período ${summary.timespan}`}
            Icon={BookOpen}
            tone="accent"
          />
          <KpiCard title="Autores" value={summary.authorsCount} Icon={Users} />
          <KpiCard title="Países" value={summary.countriesCount} Icon={Globe2} />
          <KpiCard title="Venues" value={summary.venuesCount} Icon={Building2} />
          <KpiCard
            title="Crescimento anual"
            value={`${metrics.growthRate.toLocaleString('pt-BR')}%`}
            subtitle="Taxa composta no período"
            Icon={TrendingUp}
          />
          <KpiCard
            title="Citações por ano"
            value={metrics.avgCitPerYear}
            subtitle="Média por documento"
            Icon={Quote}
          />
          <KpiCard
            title="Colaboração internacional"
            value={metrics.mcp}
            subtitle={`${metrics.scp.toLocaleString('pt-BR')} de país único`}
            Icon={Globe2}
          />
          <KpiCard
            title="Autores por documento"
            value={metrics.coauthIndex}
            subtitle={`${metrics.singleAuthorDocs.toLocaleString('pt-BR')} com autor único`}
            Icon={CalendarRange}
          />
        </div>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Deduplicação</CardTitle>
          <CardDescription>
            Bases diferentes indexam os mesmos artigos. O DOI é a evidência mais forte de
            identidade; a similaridade de título alcança os registros sem DOI, ao custo de
            algum risco de falso positivo.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex flex-wrap items-center gap-2">
            {(Object.keys(DEDUP_LABELS) as DedupStrategy[]).map((strategy) => (
              <Button
                key={strategy}
                size="sm"
                variant={dedupStrategy === strategy ? 'default' : 'outline'}
                disabled={busy}
                onClick={() => void applyDedup(strategy)}
              >
                {DEDUP_LABELS[strategy]}
              </Button>
            ))}

            {duplicates.length > 0 && (
              <Badge variant="warning">
                {duplicates.length.toLocaleString('pt-BR')} documentos removidos
              </Badge>
            )}
          </div>
        </CardContent>
      </Card>

      <ThemePanel />

      {overview && (
        <div className="grid gap-4 lg:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Produção ao longo do tempo</CardTitle>
              <CardDescription>Documentos publicados por ano.</CardDescription>
            </CardHeader>
            <CardContent>
              <Suspense fallback={<ChartSkeleton />}>
                <PlotlyChart
                  exportName="producao-por-ano"
                  height={320}
                  data={[
                    {
                      type: 'bar',
                      x: overview.docsPerYear.map((point) => point.year),
                      y: overview.docsPerYear.map((point) => point.count),
                      marker: { color: '#1273B9' },
                      hovertemplate: '%{x}: %{y} documentos<extra></extra>',
                    },
                  ]}
                  layout={{
                    xaxis: { title: { text: 'Ano' } },
                    yaxis: { title: { text: 'Documentos' } },
                  }}
                />
              </Suspense>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Lei de Lotka</CardTitle>
              <CardDescription>
                Produtividade observada contra a distribuição teórica c/x². O afastamento
                da curva indica concentração atípica da autoria.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Suspense fallback={<ChartSkeleton />}>
                {overview.lotka && (
                  <PlotlyChart
                    exportName="lei-de-lotka"
                    height={320}
                    data={[
                      {
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Observado',
                        x: overview.lotka.observed.map((point) => point.articles),
                        y: overview.lotka.observed.map((point) => point.frequency),
                        line: { color: '#1273B9', width: 2 },
                      },
                      {
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Teórico (Lotka)',
                        x: overview.lotka.theoretical.map((point) => point.articles),
                        y: overview.lotka.theoretical.map((point) => point.frequency),
                        line: { color: '#E8734A', width: 2, dash: 'dash' },
                      },
                    ]}
                    layout={{
                      xaxis: { title: { text: 'Artigos publicados' } },
                      yaxis: { title: { text: 'Proporção de autores' } },
                      legend: { x: 0.6, y: 0.95 },
                    }}
                  />
                )}
              </Suspense>
            </CardContent>
          </Card>
        </div>
      )}

      {overview && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Qualidade e completude dos metadados</CardTitle>
            <CardDescription>
              Campos ausentes limitam o que a análise consegue enxergar — sem afiliação não
              há mapa de colaboração, sem referências não há rede de cocitação.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto rounded-md border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Metadado</TableHead>
                    <TableHead>Faltantes</TableHead>
                    <TableHead>%</TableHead>
                    <TableHead>Status</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {overview.completeness.map((row) => (
                    <TableRow key={row.field}>
                      <TableCell className="font-medium">{row.field}</TableCell>
                      <TableCell className="tabular-nums">
                        {row.missing.toLocaleString('pt-BR')}
                      </TableCell>
                      <TableCell className="tabular-nums">
                        {row.missingPct.toLocaleString('pt-BR', {
                          minimumFractionDigits: 1,
                          maximumFractionDigits: 1,
                        })}
                        %
                      </TableCell>
                      <TableCell>
                        <Badge variant={STATUS_VARIANT[row.status]}>{row.status}</Badge>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      )}

      {duplicates.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Copy className="size-4" aria-hidden />
              Relatório de documentos excluídos
            </CardTitle>
            <CardDescription>
              Cada linha indica o documento removido e qual foi mantido em seu lugar.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="max-h-80 overflow-auto rounded-md border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Documento removido</TableHead>
                    <TableHead>Mantido no lugar</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {duplicates.slice(0, 200).map((doc, index) => (
                    <TableRow key={`${String(doc['TITLE'])}-${index}`}>
                      <TableCell className="max-w-96 truncate" title={String(doc['TITLE'])}>
                        {String(doc['TITLE'])}
                      </TableCell>
                      <TableCell
                        className="max-w-96 truncate"
                        title={doc['DOCUMENTO DE REFERÊNCIA (MANTIDO)']}
                      >
                        {doc['DOCUMENTO DE REFERÊNCIA (MANTIDO)']}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
            {duplicates.length > 200 && (
              <p className="mt-2 text-xs text-muted-foreground">
                Exibindo as 200 primeiras de {duplicates.length.toLocaleString('pt-BR')}.
              </p>
            )}
          </CardContent>
        </Card>
      )}

      <VisualAnalyses dataset={active} />

      {tables && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Tabelas analíticas</CardTitle>
            <CardDescription>
              Índices h, g, i10 e m por entidade. Clique nos cabeçalhos para ordenar;
              exporte em CSV o recorte filtrado.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <EntityTables tables={tables} />
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function ChartSkeleton() {
  return (
    <div className="grid h-80 place-items-center text-sm text-muted-foreground">
      Carregando gráfico…
    </div>
  );
}
