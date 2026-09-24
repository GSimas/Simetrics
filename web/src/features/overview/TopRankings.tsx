import { useMemo, useState } from 'react';

import { ExportImageButton } from '@/components/charts/ExportImageButton';
import { SectionTitle } from '@/components/InfoTip';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import type { EntityRow } from '@/core/tables';
import { collectColumns, pickColumn, toNumeric } from '@/core/text';
import { FIELD, FIELD_CANDIDATES } from '@/lib/schema';
import type { Dataset, SearchEntityType } from '@/lib/types';
import type { TranslationKey } from '@/lib/i18n/translations';
import { useLocale } from '@/state/locale.store';
import { openInSearch } from '@/state/navigation.store';
import type { EntityTables } from '@/workers/analytics.worker';
import { rankingImage } from './ranking-export';

const TOP_N = 10;

type Metric = 'citSum' | 'citMean' | 'citMedian' | 'docs' | 'dpaMean' | 'dpaMedian';

const METRIC_VALUE: Record<Metric, (row: EntityRow) => number> = {
  citSum: (row) => row.citations,
  citMean: (row) => row.meanCitations,
  citMedian: (row) => row.medianCitations,
  docs: (row) => row.docCount,
  dpaMean: (row) => row.meanDocsPerAuthor,
  dpaMedian: (row) => row.medianDocsPerAuthor,
};

const METRIC_LABEL: Record<Metric, TranslationKey> = {
  citSum: 'top_metric_cit_sum',
  citMean: 'top_metric_cit_mean',
  citMedian: 'top_metric_cit_median',
  docs: 'top_metric_docs',
  dpaMean: 'top_metric_dpa_mean',
  dpaMedian: 'top_metric_dpa_median',
};

/** Métricas com casas decimais: médias e medianas. */
const FRACTIONAL: ReadonlySet<Metric> = new Set(['citMean', 'citMedian', 'dpaMean', 'dpaMedian']);

const ENTITY_METRICS: readonly Metric[] = ['citSum', 'citMean', 'citMedian', 'docs'];
const COUNTRY_METRICS: readonly Metric[] = [...ENTITY_METRICS, 'dpaMean', 'dpaMedian'];

interface RankItem {
  label: string;
  value: number;
  /** Linha secundária — ano do documento, por exemplo. */
  detail?: string | undefined;
}

/** Os N maiores pela métrica; empates desfeitos pelo número de documentos. */
function rankEntities(rows: readonly EntityRow[], metric: Metric): RankItem[] {
  const value = METRIC_VALUE[metric];
  return [...rows]
    .sort((a, b) => value(b) - value(a) || b.docCount - a.docCount || b.citations - a.citations)
    .slice(0, TOP_N)
    .map((row) => ({ label: row.entity, value: value(row) }));
}

function rankDocuments(dataset: Dataset): RankItem[] {
  const columns = collectColumns(dataset);
  const titleColumn = pickColumn(columns, FIELD_CANDIDATES.title);
  if (!titleColumn) return [];

  return dataset
    .map((doc) => {
      const year = toNumeric(doc[FIELD.YEAR_CLEAN]);
      return {
        label: String(doc[titleColumn] ?? '').trim(),
        value: toNumeric(doc[FIELD.TOTAL_CITATIONS]) ?? 0,
        detail: year === null ? undefined : String(Math.trunc(year)),
      };
    })
    .filter((item) => item.label.length > 0)
    .sort((a, b) => b.value - a.value)
    .slice(0, TOP_N);
}

function formatValue(value: number, fractional: boolean): string {
  return value.toLocaleString('pt-BR', {
    minimumFractionDigits: fractional ? 2 : 0,
    maximumFractionDigits: fractional ? 2 : 0,
  });
}

interface RankingPanelProps {
  id: string;
  /** Nome base do arquivo exportado. */
  exportName: string;
  title: string;
  items: RankItem[];
  types: SearchEntityType[];
  metric?: Metric;
  metrics?: readonly Metric[];
  onMetricChange?: (metric: Metric) => void;
  /** Rótulo fixo da métrica, quando não há escolha (documentos). */
  fixedMetricLabel?: string;
}

function RankingPanel({
  id,
  exportName,
  title,
  items,
  types,
  metric,
  metrics,
  onMetricChange,
  fixedMetricLabel,
}: RankingPanelProps) {
  const t = useLocale((state) => state.t);
  const max = Math.max(...items.map((item) => item.value), 0);
  const fractional = metric !== undefined && FRACTIONAL.has(metric);
  const metricLabel = metric ? t(METRIC_LABEL[metric]) : (fixedMetricLabel ?? '');

  const getImage = () =>
    items.length === 0
      ? null
      : rankingImage(
          `${t('top_title')} — ${title}`,
          metricLabel,
          items.map((item) => ({
            label: item.label,
            detail: item.detail,
            value: formatValue(item.value, fractional),
            ratio: max > 0 ? item.value / max : 0,
          })),
        );

  return (
    <section aria-labelledby={`${id}-title`} className="min-w-0 space-y-3 border border-border p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h4 id={`${id}-title`} className="text-sm font-semibold tracking-tight">
          {title}
        </h4>
        <div className="flex min-w-0 flex-1 items-center justify-end gap-1.5 sm:flex-none">
          {metric && metrics && onMetricChange ? (
            <Select value={metric} onValueChange={(value) => onMetricChange(value as Metric)}>
              <SelectTrigger
                className="h-8 w-full min-w-0 text-xs sm:w-auto sm:min-w-52"
                aria-label={`${t('top_metric_label')} — ${title}`}
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {metrics.map((option) => (
                  <SelectItem key={option} value={option}>
                    {t(METRIC_LABEL[option])}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <span className="eyebrow">{fixedMetricLabel}</span>
          )}
          <ExportImageButton filename={exportName} getImage={getImage} />
        </div>
      </div>

      {items.length === 0 ? (
        <p className="py-6 text-center text-sm text-muted-foreground">{t('top_empty')}</p>
      ) : (
        <ol className="space-y-1.5">
          {items.map((item, index) => (
            <li key={`${item.label}-${index}`}>
              <button
                type="button"
                onClick={() => openInSearch(item.label, types)}
                title={`${item.label} — ${t('top_open_hint')}`}
                className="group grid w-full cursor-pointer grid-cols-[1.5rem_minmax(0,1fr)_auto] items-center gap-x-2 rounded-sm px-1 py-1 text-left transition-colors hover:bg-muted/60 focus-visible:outline-2 focus-visible:outline-ring"
              >
                <span className="text-xs tabular-nums text-muted-foreground">{index + 1}</span>
                <span className="min-w-0">
                  <span className="block truncate text-xs group-hover:underline sm:text-sm">
                    {item.label}
                    {item.detail && <span className="text-muted-foreground"> · {item.detail}</span>}
                  </span>
                  <span className="mt-1 block h-1.5 w-full overflow-hidden rounded-full bg-muted">
                    <span
                      className="block h-full rounded-full bg-highlight transition-[width] duration-500"
                      style={{ width: `${max > 0 ? (item.value / max) * 100 : 0}%` }}
                    />
                  </span>
                </span>
                <span className="text-xs font-semibold tabular-nums sm:text-sm">
                  {formatValue(item.value, fractional)}
                </span>
              </button>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}

export interface TopRankingsProps {
  dataset: Dataset;
  tables: EntityTables;
  className?: string;
  'data-tour'?: string;
}

/**
 * Top 10 de autores, documentos, países e venues. Cada ranking escolhe sua métrica —
 * soma, média ou mediana das citações, ou número de documentos; países também por
 * documentos por autor. Clicar num item abre o perfil no Motor de Busca.
 */
export function TopRankings({ dataset, tables, className, 'data-tour': tour }: TopRankingsProps) {
  const t = useLocale((state) => state.t);
  const [authorMetric, setAuthorMetric] = useState<Metric>('citSum');
  const [countryMetric, setCountryMetric] = useState<Metric>('citSum');
  const [venueMetric, setVenueMetric] = useState<Metric>('citSum');

  const documents = useMemo(() => rankDocuments(dataset), [dataset]);
  const authors = useMemo(() => rankEntities(tables.authors, authorMetric), [tables.authors, authorMetric]);
  const countries = useMemo(
    () => rankEntities(tables.countries, countryMetric),
    [tables.countries, countryMetric],
  );
  const venues = useMemo(() => rankEntities(tables.venues, venueMetric), [tables.venues, venueMetric]);

  return (
    <Card className={className} data-tour={tour}>
      <CardHeader className="pb-3">
        <SectionTitle title={t('top_title')} info={t('top_description')} />
      </CardHeader>
      <CardContent className="grid gap-4 lg:grid-cols-2">
        <RankingPanel
          id="top-authors"
          exportName="top10-autores"
          title={t('top_authors')}
          items={authors}
          types={['Autor']}
          metric={authorMetric}
          metrics={ENTITY_METRICS}
          onMetricChange={setAuthorMetric}
        />
        <RankingPanel
          id="top-documents"
          exportName="top10-documentos"
          title={t('top_documents')}
          items={documents}
          types={['Documento']}
          fixedMetricLabel={t('top_metric_citations')}
        />
        <RankingPanel
          id="top-countries"
          exportName="top10-paises"
          title={t('top_countries')}
          items={countries}
          types={['País']}
          metric={countryMetric}
          metrics={COUNTRY_METRICS}
          onMetricChange={setCountryMetric}
        />
        <RankingPanel
          id="top-venues"
          exportName="top10-venues"
          title={t('top_venues')}
          items={venues}
          types={['Local de Publicação (Venue)']}
          metric={venueMetric}
          metrics={ENTITY_METRICS}
          onMetricChange={setVenueMetric}
        />
      </CardContent>
    </Card>
  );
}
