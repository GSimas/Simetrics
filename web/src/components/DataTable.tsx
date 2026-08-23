import { useMemo, useState } from 'react';
import {
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  useReactTable,
  type ColumnDef,
  type SortingState,
} from '@tanstack/react-table';
import { ArrowDown, ArrowUp, ArrowUpDown, ChevronLeft, ChevronRight, Download } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { downloadCsv, timestampedFilename, toCsv } from '@/core/export';
import { cn } from '@/lib/utils';

/**
 * Tabela analítica com ordenação, filtro, paginação e exportação.
 *
 * Substitui o `st.dataframe` do Streamlit. As tabelas do Simetrics chegam a milhares de
 * linhas (3.821 palavras-chave na base de exemplo), então a paginação não é enfeite: é o
 * que impede o navegador de montar milhares de nós de DOM de uma vez.
 */

export interface DataTableProps<T> {
  data: readonly T[];
  columns: ColumnDef<T, unknown>[];
  /** Nome-base do arquivo exportado. Sem ele, o botão de exportar não aparece. */
  exportName?: string;
  /** Linhas por página. */
  pageSize?: number;
  filterPlaceholder?: string;
  className?: string;
}

export function DataTable<T extends Record<string, unknown>>({
  data,
  columns,
  exportName,
  pageSize = 20,
  filterPlaceholder = 'Filtrar…',
  className,
}: DataTableProps<T>) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [globalFilter, setGlobalFilter] = useState('');

  // O React Compiler não consegue memoizar o retorno do TanStack Table, que expõe
  // funções recriadas a cada render. É limitação conhecida da biblioteca, não deste
  // componente, e sem consequência aqui: nada do retorno atravessa uma fronteira
  // memoizada.
  // eslint-disable-next-line react-hooks/incompatible-library
  const table = useReactTable({
    data: data as T[],
    columns,
    state: { sorting, globalFilter },
    onSortingChange: setSorting,
    onGlobalFilterChange: setGlobalFilter,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    initialState: { pagination: { pageSize } },
  });

  const filteredCount = table.getFilteredRowModel().rows.length;

  // A exportação leva as linhas FILTRADAS, e não a página visível: o usuário filtra para
  // recortar o que quer levar, não para escolher uma página.
  const exportRows = useMemo(
    () => table.getFilteredRowModel().rows.map((row) => row.original as Record<string, unknown>),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [table, globalFilter, data],
  );

  return (
    <div className={cn('space-y-3', className)}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <Input
          value={globalFilter}
          onChange={(event) => setGlobalFilter(event.target.value)}
          placeholder={filterPlaceholder}
          className="h-8 max-w-xs"
        />

        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground tabular-nums">
            {filteredCount.toLocaleString('pt-BR')}{' '}
            {filteredCount === 1 ? 'linha' : 'linhas'}
            {filteredCount !== data.length && ` de ${data.length.toLocaleString('pt-BR')}`}
          </span>

          {exportName && (
            <Button
              variant="outline"
              size="sm"
              onClick={() =>
                downloadCsv(timestampedFilename(exportName, 'csv'), toCsv(exportRows))
              }
            >
              <Download aria-hidden />
              CSV
            </Button>
          )}
        </div>
      </div>

      <div className="overflow-x-auto rounded-md border">
        <Table>
          <TableHeader>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => {
                  const sortDirection = header.column.getIsSorted();
                  const canSort = header.column.getCanSort();

                  return (
                    <TableHead key={header.id}>
                      {header.isPlaceholder ? null : canSort ? (
                        <button
                          type="button"
                          onClick={header.column.getToggleSortingHandler()}
                          className="flex items-center gap-1 hover:text-foreground"
                        >
                          {flexRender(header.column.columnDef.header, header.getContext())}
                          {sortDirection === 'asc' ? (
                            <ArrowUp className="size-3" aria-hidden />
                          ) : sortDirection === 'desc' ? (
                            <ArrowDown className="size-3" aria-hidden />
                          ) : (
                            <ArrowUpDown className="size-3 opacity-40" aria-hidden />
                          )}
                        </button>
                      ) : (
                        flexRender(header.column.columnDef.header, header.getContext())
                      )}
                    </TableHead>
                  );
                })}
              </TableRow>
            ))}
          </TableHeader>

          <TableBody>
            {table.getRowModel().rows.length === 0 ? (
              <TableRow>
                <TableCell
                  colSpan={columns.length}
                  className="h-20 text-center text-muted-foreground"
                >
                  Nenhum resultado.
                </TableCell>
              </TableRow>
            ) : (
              table.getRowModel().rows.map((row) => (
                <TableRow key={row.id}>
                  {row.getVisibleCells().map((cell) => (
                    <TableCell key={cell.id}>
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </TableCell>
                  ))}
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>

      {table.getPageCount() > 1 && (
        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground tabular-nums">
            Página {table.getState().pagination.pageIndex + 1} de {table.getPageCount()}
          </span>
          <div className="flex gap-1">
            <Button
              variant="outline"
              size="sm"
              onClick={() => table.previousPage()}
              disabled={!table.getCanPreviousPage()}
            >
              <ChevronLeft aria-hidden />
              Anterior
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => table.nextPage()}
              disabled={!table.getCanNextPage()}
            >
              Próxima
              <ChevronRight aria-hidden />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
