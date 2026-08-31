import { lazy, Suspense, useEffect, useState } from 'react';
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

import type { Trace } from '@/components/charts/plotly';
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
import type { ProductionCategory, ProductionSeries } from '@/core/viz/production-timeline';
import type { Dataset, MetadataCompleteness } from '@/lib/types';
import { useAsyncResult } from '@/lib/use-async-result';
import { useDataset, type DedupStrategy } from '@/state/dataset.store';
import { useLocale } from '@/state/locale.store';
import { getAnalyticsWorker } from '@/workers/client';
import { EntityTables } from './EntityTables';
import { ThemePanel } from './ThemePanel';
import { VisualAnalyses } from './VisualAnalyses';
import { PALETTE, chartMessage } from './viz-shared';

const PlotlyChart = lazy(() => import('@/components/charts/PlotlyChart'));

const PRODUCTION_CATEGORIES: readonly ProductionCategory[] = [
  'Total',
  'Países',
  'Base de Dados',
  'Tipo de Trabalho',
  'Temas (IA)',
];

type ProductionChartMode = 'bars-grouped' | 'bars-stacked' | 'line';

const STATUS_VARIANT: Record<MetadataCompleteness['status'], 'success' | 'default' | 'warning' | 'destructive'> = {
  Excelente: 'success',
  Bom: 'default',
  Aceitável: 'warning',
  Ruim: 'destructive',
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
  const isDeduplicating = useDataset((state) => state.isDeduplicating);
  const isIngesting = useDataset((state) => state.isIngesting);
  const busy = isDeduplicating || isIngesting;
  const t = useLocale((state) => state.t);

  const [selectedStrategy, setSelectedStrategy] = useState<DedupStrategy>(dedupStrategy);

  useEffect(() => {
    setSelectedStrategy(dedupStrategy);
  }, [dedupStrategy]);

  const dedupLabels: Record<DedupStrategy, string> = {
    none: t('dedup_none'),
    doi: t('dedup_doi'),
    similarity: t('dedup_similarity'),
    both: t('dedup_both'),
  };

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
            <CardTitle>{t('empty_start_title')}</CardTitle>
            <CardDescription>{t('empty_start_desc')}</CardDescription>
          </CardHeader>
          <CardContent className="text-sm text-muted-foreground">
            {t('empty_client_note')}
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
        <div className="grid grid-cols-2 gap-3 sm:gap-4 lg:grid-cols-4">
          <KpiCard
            title={t('kpi_docs')}
            value={summary.totalDocs}
            subtitle={`${t('kpi_docs_sub')} ${summary.timespan}`}
            Icon={BookOpen}
            tone="blue"
          />
          <KpiCard
            title={t('kpi_authors')}
            value={summary.authorsCount}
            subtitle={t('kpi_authors_sub')}
            Icon={Users}
            tone="purple"
          />
          <KpiCard
            title={t('kpi_countries')}
            value={summary.countriesCount}
            subtitle={t('kpi_countries_sub')}
            Icon={Globe2}
            tone="indigo"
          />
          <KpiCard
            title={t('kpi_venues')}
            value={summary.venuesCount}
            subtitle={t('kpi_venues_sub')}
            Icon={Building2}
            tone="cyan"
          />
          <KpiCard
            title={t('kpi_growth')}
            value={`${metrics.growthRate.toLocaleString('pt-BR')}%`}
            subtitle={t('kpi_growth_sub')}
            Icon={TrendingUp}
            tone="emerald"
          />
          <KpiCard
            title={t('kpi_citations_year')}
            value={metrics.avgCitPerYear}
            subtitle={t('kpi_citations_year_sub')}
            Icon={Quote}
            tone="amber"
          />
          <KpiCard
            title={t('kpi_collab')}
            value={metrics.mcp}
            subtitle={`${metrics.scp.toLocaleString('pt-BR')} ${t('kpi_collab_sub')}`}
            Icon={Globe2}
            tone="indigo"
          />
          <KpiCard
            title={t('kpi_authors_doc')}
            value={metrics.coauthIndex}
            subtitle={`${metrics.singleAuthorDocs.toLocaleString('pt-BR')} ${t('kpi_authors_doc_sub')}`}
            Icon={CalendarRange}
            tone="purple"
          />
        </div>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="text-base">{t('dedup_title')}</CardTitle>
          <CardDescription>{t('dedup_description')}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex flex-wrap items-center gap-3">
            <div className="w-72 sm:w-80">
              <Select
                value={selectedStrategy}
                onValueChange={(val) => setSelectedStrategy(val as DedupStrategy)}
                disabled={busy}
              >
                <SelectTrigger className="h-9">
                  <SelectValue placeholder={t('dedup_strategy_label')} />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">{t('dedup_none')}</SelectItem>
                  <SelectItem value="doi">{t('dedup_doi')}</SelectItem>
                  <SelectItem value="similarity">{t('dedup_similarity')}</SelectItem>
                  <SelectItem value="both">{t('dedup_both')}</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <Button
              size="sm"
              variant="default"
              disabled={busy}
              onClick={() => void applyDedup(selectedStrategy)}
              className="gap-2 cursor-pointer font-medium"
            >
              {t('dedup_execute_btn')}
            </Button>

            {dedupStrategy !== 'none' && (
              <Badge variant="blue" className="text-xs">
                {dedupLabels[dedupStrategy]}
              </Badge>
            )}

            {duplicates.length > 0 && (
              <Badge variant="warning" className="text-xs">
                {duplicates.length.toLocaleString('pt-BR')} {t('dedup_removed')}
              </Badge>
            )}
          </div>
        </CardContent>
      </Card>

      <ThemePanel />

      {overview && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">{t('meta_quality_title')}</CardTitle>
            <CardDescription>{t('meta_quality_description')}</CardDescription>
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

      {overview && (
        <div className="grid gap-4 lg:grid-cols-2">
          <ProductionTimelineCard dataset={active} />

          <Card>
            <CardHeader>
              <CardTitle className="text-base">{t('lotka_title')}</CardTitle>
              <CardDescription>{t('lotka_description')}</CardDescription>
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
            <CardTitle className="text-base">{t('tables_title')}</CardTitle>
            <CardDescription>{t('tables_description')}</CardDescription>
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

/**
 * Produção ao longo do tempo — categoria (país, base, tipo de trabalho, tema de IA) e
 * modo de visualização (barras separadas/agrupadas ou linha) são estado de UI puro;
 * só a categoria dispara um novo cálculo no worker (`core/viz/production-timeline.ts`).
 */
function ProductionTimelineCard({ dataset }: { dataset: Dataset }) {
  const t = useLocale((state) => state.t);
  const hasThemes = useDataset((state) => state.clustering !== null);
  const [category, setCategory] = useState<ProductionCategory>('Total');
  const [mode, setMode] = useState<ProductionChartMode>('bars-grouped');

  const { data: series } = useAsyncResult<ProductionSeries[]>(`production ${category}`, () =>
    getAnalyticsWorker().productionTimeline(dataset, category),
  );

  const resolvedSeries = series ?? [];
  const isLine = mode === 'line';

  const categoryLabels: Record<ProductionCategory, string> = {
    Total: t('prod_category_total'),
    Países: t('prod_category_country'),
    'Base de Dados': t('prod_category_database'),
    'Tipo de Trabalho': t('prod_category_doctype'),
    'Temas (IA)': t('prod_category_theme'),
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{t('prod_title')}</CardTitle>
        <CardDescription>{t('prod_description')}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid gap-3 sm:grid-cols-2">
          <div className="space-y-1.5">
            <Label htmlFor="prod-category">{t('prod_category_label')}</Label>
            <Select
              value={category}
              onValueChange={(value) => setCategory(value as ProductionCategory)}
            >
              <SelectTrigger id="prod-category">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {PRODUCTION_CATEGORIES.map((option) => (
                  <SelectItem
                    key={option}
                    value={option}
                    disabled={option === 'Temas (IA)' && !hasThemes}
                  >
                    {categoryLabels[option]}
                    {option === 'Temas (IA)' && !hasThemes ? ` (${t('prod_category_theme_locked')})` : ''}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="prod-mode">{t('prod_mode_label')}</Label>
            <Select value={mode} onValueChange={(value) => setMode(value as ProductionChartMode)}>
              <SelectTrigger id="prod-mode">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="bars-grouped">{t('prod_mode_bars_grouped')}</SelectItem>
                <SelectItem value="bars-stacked">{t('prod_mode_bars_stacked')}</SelectItem>
                <SelectItem value="line">{t('prod_mode_line')}</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        {resolvedSeries.length === 0 ? (
          chartMessage(
            category === 'Temas (IA)' && !hasThemes ? t('prod_empty_no_themes') : t('prod_empty_generic'),
          )
        ) : (
          <Suspense fallback={<ChartSkeleton />}>
            <PlotlyChart
              exportName="producao-por-ano"
              height={340}
              data={resolvedSeries.map((entry, index): Trace => {
                const color = PALETTE[index % PALETTE.length] as string;
                const x = entry.points.map((point) => point.year);
                const y = entry.points.map((point) => point.count);

                return isLine
                  ? {
                      type: 'scatter',
                      mode: 'lines+markers',
                      name: entry.category,
                      x,
                      y,
                      line: { color, width: 2 },
                      marker: { color, size: 5 },
                      hovertemplate: `${entry.category} — %{x}: %{y} documentos<extra></extra>`,
                    }
                  : {
                      type: 'bar',
                      name: entry.category,
                      x,
                      y,
                      marker: { color },
                      hovertemplate: `${entry.category} — %{x}: %{y} documentos<extra></extra>`,
                    };
              })}
              layout={{
                xaxis: { title: { text: 'Ano' } },
                yaxis: { title: { text: 'Documentos' } },
                barmode: mode === 'bars-stacked' ? 'stack' : 'group',
                showlegend: resolvedSeries.length > 1,
                legend: { orientation: 'h', y: -0.2 },
              }}
            />
          </Suspense>
        )}
      </CardContent>
    </Card>
  );
}
