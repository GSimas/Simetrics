import { useMemo, type ReactElement } from 'react';
import type { ColumnDef } from '@tanstack/react-table';

import { DataTable } from '@/components/DataTable';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import type { EntityRow } from '@/core/tables';
import type { EntityTables as Tables } from '@/workers/analytics.worker';
import { useLocale } from '@/state/locale.store';

function numeric(value: number, digits = 0): ReactElement {
  return (
    <span className="tabular-nums">
      {value.toLocaleString('pt-BR', { minimumFractionDigits: digits, maximumFractionDigits: digits })}
    </span>
  );
}

function truncated(value: string): ReactElement {
  return (
    <span className="block max-w-80 truncate" title={value}>
      {value || '—'}
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

function buildColumns(entityLabel: string, extra: Extra): ColumnDef<EntityRow, unknown>[] {
  const columns: ColumnDef<EntityRow, unknown>[] = [
    {
      accessorKey: 'entity',
      header: entityLabel,
      cell: ({ row }) => (
        <span className="block max-w-96 truncate font-medium" title={row.original.entity}>
          {row.original.entity}
        </span>
      ),
    },
    ...indexColumns(),
    {
      accessorKey: 'topSpecialization',
      header: 'Especialização (maior QL)',
      cell: ({ row }) => truncated(row.original.topSpecialization),
    },
  ];

  if (extra === 'coauthors') {
    columns.push({
      id: 'coauthors',
      header: 'Coautores',
      accessorFn: (row) => row.coauthors.join(', '),
      cell: ({ row }) => truncated(row.original.coauthors.join(', ')),
    });
  } else if (extra === 'topDocument') {
    columns.push({
      id: 'topDocument',
      accessorKey: 'topDocument',
      header: 'Documento mais citado',
      cell: ({ row }) => truncated(row.original.topDocument),
    });
  }

  return columns;
}

export interface EntityTablesProps {
  tables: Tables;
}

export function EntityTables({ tables }: EntityTablesProps) {
  const t = useLocale((state) => state.t);

  const columns = useMemo(
    () => ({
      authors: buildColumns('Autor', 'coauthors'),
      countries: buildColumns('País', 'topDocument'),
      venues: buildColumns('Local de Publicação (Venue)', 'topDocument'),
      keywords: buildColumns('Palavra-chave', 'none'),
    }),
    [],
  );

  const panels = [
    { value: 'authors', label: t('table_tab_authors'), rows: tables.authors, export: 'autores' },
    { value: 'countries', label: t('table_tab_countries'), rows: tables.countries, export: 'paises' },
    { value: 'venues', label: t('table_tab_venues'), rows: tables.venues, export: 'venues' },
    { value: 'keywords', label: t('table_tab_keywords'), rows: tables.keywords, export: 'keywords' },
  ] as const;

  return (
    <Tabs defaultValue="authors">
      <TabsList className="bg-slate-100 dark:bg-slate-800/80 p-1">
        {panels.map((panel) => (
          <TabsTrigger key={panel.value} value={panel.value} className="gap-1.5">
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
            data={panel.rows as unknown as Record<string, unknown>[]}
            columns={columns[panel.value] as unknown as ColumnDef<Record<string, unknown>, unknown>[]}
            exportName={panel.export}
            filterPlaceholder={`${t('table_filter_placeholder')} (${panel.label.toLowerCase()})`}
          />
        </TabsContent>
      ))}
    </Tabs>
  );
}
