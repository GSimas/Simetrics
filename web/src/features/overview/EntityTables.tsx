import { useMemo, type ReactElement } from 'react';
import type { ColumnDef } from '@tanstack/react-table';

import { DataTable } from '@/components/DataTable';
import { EntityChip, EntityChips } from '@/components/EntityChip';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import type { EntityRow } from '@/core/tables';
import type { EntityTables as Tables } from '@/workers/analytics.worker';
import type { Dataset, SearchEntityType } from '@/lib/types';
import { useDataset } from '@/state/dataset.store';
import { useLocale } from '@/state/locale.store';

function numeric(value: number, digits = 0): ReactElement {
  return (
    <span className="tabular-nums">
      {value.toLocaleString('pt-BR', { minimumFractionDigits: digits, maximumFractionDigits: digits })}
    </span>
  );
}

function indexColumns(): ColumnDef<EntityRow, unknown>[] {
  return [
    {
      accessorKey: 'docCount',
      header: 'Docs',
      cell: ({ row }) => numeric(row.original.docCount),
    },
    {
      accessorKey: 'citations',
      header: 'Citações',
      cell: ({ row }) => numeric(row.original.citations),
    },
    { accessorKey: 'h', header: 'Índice h', cell: ({ row }) => numeric(row.original.h) },
    { accessorKey: 'g', header: 'Índice g', cell: ({ row }) => numeric(row.original.g) },
    { accessorKey: 'i10', header: 'Índice i10', cell: ({ row }) => numeric(row.original.i10) },
    {
      accessorKey: 'm',
      header: 'Índice m',
      cell: ({ row }) => numeric(row.original.m, 3),
    },
    {
      accessorKey: 'meanCitations',
      header: 'Média',
      cell: ({ row }) => numeric(row.original.meanCitations, 2),
    },
    {
      accessorKey: 'medianCitations',
      header: 'Mediana',
      cell: ({ row }) => numeric(row.original.medianCitations, 2),
    },
    {
      accessorKey: 'stdCitations',
      header: 'Desvio padrão',
      cell: ({ row }) => numeric(row.original.stdCitations, 2),
    },
  ];
}

type Extra = 'coauthors' | 'topDocument' | 'none';

/** "Tema (QL: 1.83)" → "Tema"; "Título (12 citações)" → "Título". */
const stripSuffix = (value: string): string => value.replace(/\s*\((?:QL: [\d.]+|\d+ citações)\)$/, '');

function buildColumns(
  entityLabel: string,
  types: readonly SearchEntityType[],
  extra: Extra,
): ColumnDef<EntityRow, unknown>[] {
  const columns: ColumnDef<EntityRow, unknown>[] = [
    {
      accessorKey: 'entity',
      header: entityLabel,
      cell: ({ row }) => <EntityChip label={row.original.entity} types={types} />,
    },
    ...indexColumns(),
    {
      accessorKey: 'topSpecialization',
      header: 'Especialização (maior QL)',
      cell: ({ row }) => (
        <EntityChip
          label={row.original.topSpecialization}
          term={stripSuffix(row.original.topSpecialization)}
          types={['Tema']}
        />
      ),
    },
  ];

  if (extra === 'coauthors') {
    columns.push({
      id: 'coauthors',
      header: 'Coautores',
      accessorFn: (row) => row.coauthors.join(', '),
      cell: ({ row }) => <EntityChips values={row.original.coauthors} types={['Autor']} />,
    });
  } else if (extra === 'topDocument') {
    columns.push({
      id: 'topDocument',
      accessorKey: 'topDocument',
      header: 'Documento mais citado',
      cell: ({ row }) => (
        <EntityChip
          label={row.original.topDocument}
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

function buildAllDocsColumns(activeDocs: Dataset | null): ColumnDef<Record<string, unknown>, unknown>[] {
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
    header: key,
    cell: ({ row }) => {
      const val = row.original[key];
      if (val === null || val === undefined || val === '') {
        return <span className="text-muted-foreground">—</span>;
      }
      if (typeof val === 'number') {
        // Ano não leva separador de milhar ("1.993").
        const text = /YEAR|ANO/.test(key) ? String(val) : val.toLocaleString('pt-BR');
        return <span className="tabular-nums font-medium">{text}</span>;
      }
      const str = String(val);
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

  const entityColumns = useMemo(
    () => ({
      authors: buildColumns('Autor', ['Autor'], 'coauthors'),
      countries: buildColumns('País', ['País'], 'topDocument'),
      venues: buildColumns('Local de Publicação (Venue)', ['Local de Publicação (Venue)'], 'topDocument'),
      keywords: buildColumns('Palavra-chave', ['Palavra-chave'], 'none'),
    }),
    [],
  );

  const allDocsColumns = useMemo(() => buildAllDocsColumns(active), [active]);

  const panels = [
    {
      value: 'all_docs',
      label: t('table_tab_all_docs'),
      rows: (active ?? []) as unknown as Record<string, unknown>[],
      export: 'todos-documentos',
      columns: allDocsColumns,
    },
    {
      value: 'authors',
      label: t('table_tab_authors'),
      rows: tables.authors as unknown as Record<string, unknown>[],
      export: 'autores',
      columns: entityColumns.authors as unknown as ColumnDef<Record<string, unknown>, unknown>[],
    },
    {
      value: 'countries',
      label: t('table_tab_countries'),
      rows: tables.countries as unknown as Record<string, unknown>[],
      export: 'paises',
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
              {panel.rows.length.toLocaleString('pt-BR')}
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
