import { useMemo, type ReactElement } from 'react';
import type { ColumnDef } from '@tanstack/react-table';

import { DataTable } from '@/components/DataTable';
import { EntityChip, EntityChips } from '@/components/EntityChip';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { localizeTableText, type EntityRow } from '@/core/tables';
import type { EntityTables as Tables } from '@/workers/analytics.worker';
import type { Dataset, SearchEntityType } from '@/lib/types';
import { entityTypeLabel, numberLocale } from '@/lib/i18n/labels';
import type { Locale } from '@/lib/i18n/translations';
import { useDataset } from '@/state/dataset.store';
import { useLocale } from '@/state/locale.store';

const COPY = {
  pt: {
    citations: 'Citações',
    h: 'Índice h',
    g: 'Índice g',
    i10: 'Índice i10',
    m: 'Índice m',
    mean: 'Média',
    median: 'Mediana',
    std: 'Desvio padrão',
    specialization: 'Especialização (maior QL)',
    coauthors: 'Coautores',
    topDocument: 'Documento mais citado',
  },
  en: {
    citations: 'Citations',
    h: 'h-index',
    g: 'g-index',
    i10: 'i10-index',
    m: 'm-index',
    mean: 'Mean',
    median: 'Median',
    std: 'Standard deviation',
    specialization: 'Specialization (highest LQ)',
    coauthors: 'Co-authors',
    topDocument: 'Most cited document',
  },
} as const;

/** Cabeçalhos em inglês das colunas da base cujo nome é português; em pt fica o nome cru. */
const DATASET_KEY_EN: Record<string, string> = {
  'BASE DE DADOS': 'DATABASE',
  TEMA_GEMINI: 'THEME',
  TEMA_CONFIANCA: 'THEME CONFIDENCE',
  TEMA_STATUS: 'THEME STATUS',
};

/** Valores gravados em TEMA_STATUS (core/hybrid/apply.ts) e como aparecem em inglês. */
const THEME_STATUS_EN: Record<string, string> = {
  auto: 'auto',
  revisar: 'review',
  nao_classificado: 'unclassified',
};

function numeric(value: number, locale: Locale, digits = 0): ReactElement {
  return (
    <span className="tabular-nums">
      {value.toLocaleString(numberLocale(locale), { minimumFractionDigits: digits, maximumFractionDigits: digits })}
    </span>
  );
}

function indexColumns(locale: Locale): ColumnDef<EntityRow, unknown>[] {
  const copy = COPY[locale];
  return [
    {
      accessorKey: 'docCount',
      header: 'Docs',
      cell: ({ row }) => numeric(row.original.docCount, locale),
    },
    {
      accessorKey: 'citations',
      header: copy.citations,
      cell: ({ row }) => numeric(row.original.citations, locale),
    },
    { accessorKey: 'h', header: copy.h, cell: ({ row }) => numeric(row.original.h, locale) },
    { accessorKey: 'g', header: copy.g, cell: ({ row }) => numeric(row.original.g, locale) },
    { accessorKey: 'i10', header: copy.i10, cell: ({ row }) => numeric(row.original.i10, locale) },
    {
      accessorKey: 'm',
      header: copy.m,
      cell: ({ row }) => numeric(row.original.m, locale, 3),
    },
    {
      accessorKey: 'meanCitations',
      header: copy.mean,
      cell: ({ row }) => numeric(row.original.meanCitations, locale, 2),
    },
    {
      accessorKey: 'medianCitations',
      header: copy.median,
      cell: ({ row }) => numeric(row.original.medianCitations, locale, 2),
    },
    {
      accessorKey: 'stdCitations',
      header: copy.std,
      cell: ({ row }) => numeric(row.original.stdCitations, locale, 2),
    },
  ];
}

type Extra = 'coauthors' | 'topDocument' | 'none';

/** "Tema (QL: 1.83)" → "Tema"; "Título (12 citações)" → "Título". Opera no valor cru de core/tables. */
const stripSuffix = (value: string): string => value.replace(/\s*\((?:QL: [\d.]+|\d+ citações)\)$/, '');

function buildColumns(
  type: SearchEntityType,
  extra: Extra,
  locale: Locale,
): ColumnDef<EntityRow, unknown>[] {
  const types = [type];
  const copy = COPY[locale];
  const columns: ColumnDef<EntityRow, unknown>[] = [
    {
      accessorKey: 'entity',
      header: entityTypeLabel(type, locale),
      cell: ({ row }) => <EntityChip label={row.original.entity} types={types} />,
    },
    ...indexColumns(locale),
    {
      accessorKey: 'topSpecialization',
      header: copy.specialization,
      cell: ({ row }) => (
        <EntityChip
          label={localizeTableText(row.original.topSpecialization, locale)}
          term={stripSuffix(row.original.topSpecialization)}
          types={['Tema']}
        />
      ),
    },
  ];

  if (extra === 'coauthors') {
    columns.push({
      id: 'coauthors',
      header: copy.coauthors,
      accessorFn: (row) => row.coauthors.join(', '),
      cell: ({ row }) => <EntityChips values={row.original.coauthors} types={['Autor']} />,
    });
  } else if (extra === 'topDocument') {
    columns.push({
      id: 'topDocument',
      accessorKey: 'topDocument',
      header: copy.topDocument,
      cell: ({ row }) => (
        <EntityChip
          label={localizeTableText(row.original.topDocument, locale)}
          term={stripSuffix(row.original.topDocument)}
          types={['Documento']}
        />
      ),
    });
  }

  return columns;
}

/** Colunas da lista completa cujos valores são entidades do Motor de Busca. */
const ENTITY_COLUMNS: Record<string, { types: SearchEntityType[]; multiple: boolean }> = {
  TITLE: { types: ['Documento'], multiple: false },
  AUTHORS: { types: ['Autor'], multiple: true },
  COUNTRY: { types: ['País'], multiple: true },
  KEYWORDS: { types: ['Palavra-chave'], multiple: true },
  'SECONDARY TITLE': { types: ['Local de Publicação (Venue)'], multiple: false },
  TEMA_GEMINI: { types: ['Tema'], multiple: false },
};

const PRIORITY_COLUMNS = [
  'TITLE',
  'AUTHORS',
  'YEAR CLEAN',
  'TOTAL CITATIONS',
  'SECONDARY TITLE',
  'DOI',
  'COUNTRY',
  'KEYWORDS',
  'ABSTRACT',
  'BASE DE DADOS',
  'TEMA_GEMINI',
] as const;

function buildAllDocsColumns(activeDocs: Dataset | null, locale: Locale): ColumnDef<Record<string, unknown>, unknown>[] {
  if (!activeDocs || activeDocs.length === 0) return [];

  const allKeys = new Set<string>();
  for (const doc of activeDocs) {
    for (const key of Object.keys(doc)) {
      allKeys.add(key);
    }
  }

  const orderedKeys: string[] = [];
  for (const key of PRIORITY_COLUMNS) {
    if (allKeys.has(key)) {
      orderedKeys.push(key);
      allKeys.delete(key);
    }
  }
  const remainingKeys = [...allKeys].sort();
  orderedKeys.push(...remainingKeys);

  return orderedKeys.map((key) => ({
    accessorKey: key,
    header: locale === 'en' ? (DATASET_KEY_EN[key] ?? key) : key,
    cell: ({ row }) => {
      const val = row.original[key];
      if (val === null || val === undefined || val === '') {
        return <span className="text-muted-foreground">—</span>;
      }
      if (typeof val === 'number') {
        // Ano não leva separador de milhar ("1.993").
        const text = /YEAR|ANO/.test(key) ? String(val) : val.toLocaleString(numberLocale(locale));
        return <span className="tabular-nums font-medium">{text}</span>;
      }
      const raw = String(val);
      const str = locale === 'en' && key === 'TEMA_STATUS' ? (THEME_STATUS_EN[raw] ?? raw) : raw;
      const entity = ENTITY_COLUMNS[key];
      if (entity) {
        return entity.multiple ? (
          <EntityChips values={str} types={entity.types} />
        ) : (
          <EntityChip label={str} types={entity.types} />
        );
      }
      return (
        <span className="block max-w-80 truncate text-xs" title={str}>
          {str}
        </span>
      );
    },
  }));
}

export interface EntityTablesProps {
  tables: Tables;
}

export function EntityTables({ tables }: EntityTablesProps) {
  const active = useDataset((state) => state.active);
  const t = useLocale((state) => state.t);
  const locale = useLocale((state) => state.locale);

  const entityColumns = useMemo(
    () => ({
      authors: buildColumns('Autor', 'coauthors', locale),
      countries: buildColumns('País', 'topDocument', locale),
      venues: buildColumns('Local de Publicação (Venue)', 'topDocument', locale),
      keywords: buildColumns('Palavra-chave', 'none', locale),
    }),
    [locale],
  );

  const allDocsColumns = useMemo(() => buildAllDocsColumns(active, locale), [active, locale]);

  const panels = [
    {
      value: 'all_docs',
      label: t('table_tab_all_docs'),
      rows: (active ?? []) as unknown as Record<string, unknown>[],
      export: locale === 'en' ? 'all-documents' : 'todos-documentos',
      columns: allDocsColumns,
    },
    {
      value: 'authors',
      label: t('table_tab_authors'),
      rows: tables.authors as unknown as Record<string, unknown>[],
      export: locale === 'en' ? 'authors' : 'autores',
      columns: entityColumns.authors as unknown as ColumnDef<Record<string, unknown>, unknown>[],
    },
    {
      value: 'countries',
      label: t('table_tab_countries'),
      rows: tables.countries as unknown as Record<string, unknown>[],
      export: locale === 'en' ? 'countries' : 'paises',
      columns: entityColumns.countries as unknown as ColumnDef<Record<string, unknown>, unknown>[],
    },
    {
      value: 'venues',
      label: t('table_tab_venues'),
      rows: tables.venues as unknown as Record<string, unknown>[],
      export: 'venues',
      columns: entityColumns.venues as unknown as ColumnDef<Record<string, unknown>, unknown>[],
    },
    {
      value: 'keywords',
      label: t('table_tab_keywords'),
      rows: tables.keywords as unknown as Record<string, unknown>[],
      export: 'keywords',
      columns: entityColumns.keywords as unknown as ColumnDef<Record<string, unknown>, unknown>[],
    },
  ] as const;

  return (
    <Tabs defaultValue="all_docs">
      <TabsList className="h-auto w-full flex-wrap justify-start gap-x-2">
        {panels.map((panel) => (
          <TabsTrigger key={panel.value} value={panel.value} className="gap-1.5 text-xs sm:text-sm">
            <span>{panel.label}</span>
            <span className="rounded-full bg-card px-2 py-0.2 text-[11px] font-semibold text-primary shadow-2xs tabular-nums">
              {panel.rows.length.toLocaleString(numberLocale(locale))}
            </span>
          </TabsTrigger>
        ))}
      </TabsList>

      {panels.map((panel) => (
        <TabsContent key={panel.value} value={panel.value}>
          <DataTable
            data={panel.rows}
            columns={panel.columns}
            exportName={panel.export}
            filterPlaceholder={`${t('table_filter_placeholder')} (${panel.label.toLowerCase()})`}
          />
        </TabsContent>
      ))}
    </Tabs>
  );
}
