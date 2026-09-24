import { useEffect, useState } from 'react';
import {
  BookOpen,
  Building2,
  CalendarRange,
  Globe2,
  Layers,
  Quote,
  ShieldCheck,
  Sparkles,
  Table2,
  TrendingUp,
  Users,
} from 'lucide-react';

import TimeSeriesChart from '@/components/charts/TimeSeriesChart';
import { SectionTitle } from '@/components/InfoTip';
import { Collapse } from '@/components/Collapse';
import { KpiCard } from '@/components/KpiCard';
import { Launcher, LauncherGrid } from '@/components/Launcher';
import { UploadPanel } from '@/components/UploadPanel';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
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
import type { Dataset, MetadataCompleteness, SearchEntityType } from '@/lib/types';
import { identityKey, useAsyncResult } from '@/lib/use-async-result';
import { useStickyValue } from '@/lib/use-sticky-value';
import { cn } from '@/lib/utils';
import { useDataset, type DedupStrategy } from '@/state/dataset.store';
import { useLocale } from '@/state/locale.store';
import { getAnalyticsWorker } from '@/workers/client';
import { openInSearch } from '@/state/navigation.store';
import { EntityTables } from './EntityTables';
import { ThemePanel } from './ThemePanel';
import { TopRankings } from './TopRankings';
import { VisualAnalyses } from './VisualAnalyses';
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

export default function OverviewTab() {
  const active = useDataset((state) => state.active);
  // Após deduplicar ou mapear temas, as análises voltam a `null` até o worker
  // recalcular. Mostrar as anteriores nesse intervalo (esmaecidas) evita que KPIs e
  // blocos sumam e reapareçam.
  const { value: overview, stale: overviewStale } = useStickyValue(
    useDataset((state) => state.overview),
  );
  const { value: tables } = useStickyValue(useDataset((state) => state.tables));
  const duplicates = useDataset((state) => state.duplicates);
  const shownDuplicates = useStickyValue(duplicates.length > 0 ? duplicates : null).value ?? [];
  const dedupStrategy = useDataset((state) => state.dedupStrategy);
  const computeOverview = useDataset((state) => state.computeOverview);
  const computeTables = useDataset((state) => state.computeTables);
  const applyDedup = useDataset((state) => state.applyDedup);
  const isDeduplicating = useDataset((state) => state.isDeduplicating);
  const isIngesting = useDataset((state) => state.isIngesting);
  const busy = isDeduplicating || isIngesting;
  const t = useLocale((state) => state.t);

  const [selectedStrategy, setSelectedStrategy] = useState<DedupStrategy>(dedupStrategy);

  // A estratégia aplicada mudou (outra base, outro projeto): o seletor acompanha. Ajuste
  // durante o render, e não num efeito — o efeito renderizava duas vezes a cada troca.
  const [appliedStrategy, setAppliedStrategy] = useState(dedupStrategy);
  if (appliedStrategy !== dedupStrategy) {
    setAppliedStrategy(dedupStrategy);
    setSelectedStrategy(dedupStrategy);
  }

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
        <div className="border border-dashed border-border p-10 text-center">
          <SectionTitle
            className="justify-center"
            title={t('empty_start_title')}
            info={
              <>
                <p>{t('empty_start_desc')}</p>
                <p>{t('empty_client_note')}</p>
              </>
            }
          />
        </div>
      </div>
    );
  }

  const summary = overview?.summary;
  const metrics = summary?.bibliometrix;

  const dedupCard = (
    <Card data-tour="dedup">
      <CardHeader className="pb-3">
        <SectionTitle title={t('dedup_title')} info={t('dedup_description')} />
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex flex-wrap items-center gap-3">
          <div className="w-72 sm:w-80">
            <Select
              value={selectedStrategy}
              onValueChange={(val) => setSelectedStrategy(val as DedupStrategy)}
              disabled={busy}
            >
              <SelectTrigger className="h-9" aria-label={t('dedup_strategy_aria')}>
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
            disabled={busy}
            onClick={() => void applyDedup(selectedStrategy)}
            className="cursor-pointer"
          >
            {t('dedup_execute_btn')}
          </Button>

          {dedupStrategy !== 'none' && <Badge variant="blue">{dedupLabels[dedupStrategy]}</Badge>}

          {duplicates.length > 0 && (
            <Badge variant="warning">
              {duplicates.length.toLocaleString('pt-BR')} {t('dedup_removed')}
            </Badge>
          )}
        </div>

        {/* O relatório abre e fecha animado; durante o fechamento mostra a última lista. */}
        <Collapse open={duplicates.length > 0} delayOpen={false}>
            <div className="space-y-3 pt-2">
              <SectionTitle
                title="Relatório de documentos excluídos"
                info="Cada linha indica o documento removido e qual foi mantido em seu lugar."
              />
              <div className="max-h-96 overflow-auto border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Documento removido</TableHead>
                      <TableHead>Mantido no lugar</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {shownDuplicates.slice(0, 200).map((doc, index) => (
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
              {shownDuplicates.length > 200 && (
                <p className="eyebrow">
                  Exibindo as 200 primeiras de {shownDuplicates.length.toLocaleString('pt-BR')}.
                </p>
              )}
            </div>
        </Collapse>
      </CardContent>
    </Card>
  );


  return (
    <div className="space-y-6">
      <UploadPanel />

      {dedupCard}

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
          </Launcher>
        )}

        <Launcher tour="launcher-visual" Icon={Layers} title={t('visual_title')} summary={t('sum_visual')} info={t('visual_description')}>
          <VisualAnalyses dataset={active} />
        </Launcher>

        <Launcher tour="launcher-theme" Icon={Sparkles} title={t('theme_title')} summary={t('sum_theme')} info={t('theme_description')}>
          <ThemePanel />
        </Launcher>

        {tables && (
          <Launcher tour="launcher-tables" Icon={Table2} title={t('tables_title')} summary={t('sum_tables')} info={t('tables_description')}>
            <EntityTables tables={tables} />
          </Launcher>
        )}


      </LauncherGrid>
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
  const hasThemes = useDataset((state) => state.clustering !== null);
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
            exportName="producao-por-ano"
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
