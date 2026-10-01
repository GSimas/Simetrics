import { useEffect, useState } from 'react';
import {
  BookOpen,
  Building2,
  CalendarRange,
  Globe2,
  Quote,
  ShieldCheck,
  Sparkles,
  TrendingUp,
  Users,
} from 'lucide-react';

import TimeSeriesChart from '@/components/charts/TimeSeriesChart';
import { SectionTitle } from '@/components/InfoTip';
import { KpiCard } from '@/components/KpiCard';
import { Launcher, LauncherGrid } from '@/components/Launcher';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
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
import { numberLocale } from '@/lib/i18n/labels';
import type { Dataset, MetadataCompleteness, SearchEntityType } from '@/lib/types';
import { identityKey, useAsyncResult } from '@/lib/use-async-result';
import { useStickyValue } from '@/lib/use-sticky-value';
import { cn } from '@/lib/utils';
import { useDataset } from '@/state/dataset.store';
import { useLocale } from '@/state/locale.store';
import { getAnalyticsWorker } from '@/workers/client';
import { openInSearch } from '@/state/navigation.store';
import { EntityTables } from './EntityTables';
import { ThemePanel } from './ThemePanel';
import { TopRankings } from './TopRankings';
import { EmptyState } from '@/features/EmptyState';
import { PALETTE, chartMessage } from './viz-shared';

const PRODUCTION_CATEGORIES: readonly ProductionCategory[] = [
  'Total',
  'Países',
  'Base de Dados',
  'Tipo de Trabalho',
  'Temas (IA)',
];

type ProductionChartMode = 'bars-grouped' | 'bars-stacked' | 'line';

/** Categorias cujas séries são entidades do Motor de Busca — clicar abre o perfil. */
const PRODUCTION_SEARCH_TYPES: Partial<Record<ProductionCategory, SearchEntityType[]>> = {
  Países: ['País'],
  'Temas (IA)': ['Tema'],
};

const STATUS_VARIANT: Record<MetadataCompleteness['status'], 'success' | 'default' | 'warning' | 'destructive'> = {
  Excelente: 'success',
  Bom: 'default',
  Aceitável: 'warning',
  Ruim: 'destructive',
};

/** O status é calculado em português (core/summary.ts); a tela mostra no idioma escolhido. */
const STATUS_LABEL = {
  Excelente: 'meta_status_excellent',
  Bom: 'meta_status_good',
  Aceitável: 'meta_status_acceptable',
  Ruim: 'meta_status_poor',
} as const satisfies Record<MetadataCompleteness['status'], string>;

export default function OverviewTab() {
  const active = useDataset((state) => state.active);
  // Após deduplicar ou mapear temas, as análises voltam a `null` até o worker
  // recalcular. Mostrar as anteriores nesse intervalo (esmaecidas) evita que KPIs e
  // blocos sumam e reapareçam.
  const { value: overview, stale: overviewStale } = useStickyValue(
    useDataset((state) => state.overview),
  );
  const { value: tables } = useStickyValue(useDataset((state) => state.tables));
  const computeOverview = useDataset((state) => state.computeOverview);
  const computeTables = useDataset((state) => state.computeTables);
  const t = useLocale((state) => state.t);
  const locale = useLocale((state) => state.locale);
  const nf = numberLocale(locale);

  useEffect(() => {
    if (!active) return;
    void computeOverview();
    void computeTables();
  }, [active, computeOverview, computeTables]);

  // Sem base, os módulos ficam bloqueados na navegação; isto só cobre um acesso direto.
  if (!active) return <EmptyState title={t('empty_start_title')} />;

  const summary = overview?.summary;
  const metrics = summary?.bibliometrix;



  return (
    <div className="space-y-6">
      <LauncherGrid label={t('launcher_section')}>
        {overview && (
          <Launcher
            tour="launcher-meta"
            Icon={ShieldCheck}
            title={t('meta_quality_title')}
            summary={t('sum_meta')}
            info={t('meta_quality_description')}
          >
            <div className="overflow-x-auto border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>{t('meta_col_field')}</TableHead>
                    <TableHead>{t('meta_col_missing')}</TableHead>
                    <TableHead>%</TableHead>
                    <TableHead>{t('meta_col_status')}</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {overview.completeness.map((row) => (
                    <TableRow key={row.field}>
                      <TableCell className="font-medium">{row.field}</TableCell>
                      <TableCell className="tabular-nums">
                        {row.missing.toLocaleString(nf)}
                      </TableCell>
                      <TableCell className="tabular-nums">
                        {row.missingPct.toLocaleString(nf, {
                          minimumFractionDigits: 1,
                          maximumFractionDigits: 1,
                        })}
                        %
                      </TableCell>
                      <TableCell>
                        <Badge variant={STATUS_VARIANT[row.status]}>{t(STATUS_LABEL[row.status])}</Badge>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </Launcher>
        )}

        <Launcher tour="launcher-theme" Icon={Sparkles} title={t('theme_title')} summary={t('sum_theme')} info={t('theme_description')}>
          <ThemePanel />
        </Launcher>


      </LauncherGrid>

      {summary && metrics && (
        <div
          data-tour="kpis"
          aria-busy={overviewStale}
          className={cn(
            'grid grid-cols-2 gap-3 transition-opacity duration-300 animate-in fade-in-0 sm:gap-4 lg:grid-cols-4',
            overviewStale && 'opacity-60',
          )}
        >
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
            value={`${metrics.growthRate.toLocaleString(nf)}%`}
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
            subtitle={`${metrics.scp.toLocaleString(nf)} ${t('kpi_collab_sub')}`}
            Icon={Globe2}
            tone="indigo"
          />
          <KpiCard
            title={t('kpi_authors_doc')}
            value={metrics.coauthIndex}
            subtitle={`${metrics.singleAuthorDocs.toLocaleString(nf)} ${t('kpi_authors_doc_sub')}`}
            Icon={CalendarRange}
            tone="purple"
          />
        </div>
      )}

      {overview && (
        <Card data-tour="production">
          <CardHeader className="pb-3">
            <SectionTitle title={t('prod_title')} info={t('prod_description')} />
          </CardHeader>
          <CardContent>
            <ProductionTimeline dataset={active} />
          </CardContent>
        </Card>
      )}

      {tables && (
        <TopRankings
          data-tour="top10"
          dataset={active}
          tables={tables}
          className={cn('transition-opacity duration-300', overviewStale && 'opacity-60')}
        />
      )}

      {tables && (
        <Card data-tour="tables">
          <CardHeader className="pb-3">
            <SectionTitle title={t('tables_title')} info={t('tables_description')} />
          </CardHeader>
          <CardContent>
            <EntityTables tables={tables} />
          </CardContent>
        </Card>
      )}

    </div>
  );
}

/**
 * Produção ao longo do tempo — categoria (país, base, tipo de trabalho, tema de IA) e
 * modo de visualização (barras separadas/agrupadas ou linha) são estado de UI puro;
 * só a categoria dispara um novo cálculo no worker (`core/viz/production-timeline.ts`).
 */
function ProductionTimeline({ dataset }: { dataset: Dataset }) {
  const t = useLocale((state) => state.t);
  const locale = useLocale((state) => state.locale);
  const hasThemes = useDataset((state) => state.clustering !== null || state.hybridRun !== null);
  const [category, setCategory] = useState<ProductionCategory>('Total');
  const [mode, setMode] = useState<ProductionChartMode>('bars-grouped');

  const { data: series } = useAsyncResult<ProductionSeries[]>(`production ${identityKey(dataset)} ${category}`, () =>
    getAnalyticsWorker().productionTimeline(dataset, category),
  );

  const resolvedSeries = series ?? [];

  const categoryLabels: Record<ProductionCategory, string> = {
    Total: t('prod_category_total'),
    Países: t('prod_category_country'),
    'Base de Dados': t('prod_category_database'),
    'Tipo de Trabalho': t('prod_category_doctype'),
    'Temas (IA)': t('prod_category_theme'),
  };

  return (
    <div className="space-y-4">
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
          <TimeSeriesChart
            exportName={locale === 'en' ? 'production-per-year' : 'producao-por-ano'}
            mode={mode}
            xLabel={t('prod_axis_year')}
            yLabel={t('prod_axis_docs')}
            unit={t('prod_unit_docs')}
            height={440}
            onSeriesClick={
              PRODUCTION_SEARCH_TYPES[category]
                ? (name) => openInSearch(name, PRODUCTION_SEARCH_TYPES[category] ?? [])
                : undefined
            }
            series={resolvedSeries.map((entry, index) => ({
              name: entry.category,
              color: PALETTE[index % PALETTE.length] as string,
              points: entry.points.map((point) => ({ x: point.year, y: point.count })),
            }))}
          />
        )}
    </div>
  );
}
