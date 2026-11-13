import * as React from 'react';

import { cn } from '@/lib/utils';

interface TableProps extends React.HTMLAttributes<HTMLTableElement> {
  /** Linhas visíveis antes da rolagem vertical própria; `false` desliga (folha do relatório). */
  maxRows?: number | false;
}

/**
 * Com mais de `maxRows` linhas, o contêiner da tabela (o elemento pai) fica com a altura do
 * cabeçalho + `maxRows` linhas e ganha rolagem vertical própria — a tabela não estica a
 * página. Mede as linhas de verdade, porque a altura varia (chips de autores quebram).
 */
function capRows(table: HTMLTableElement, maxRows: number): void {
  const box = table.parentElement;
  if (!box) return;
  const rows = table.tBodies[0]?.rows;
  const limit = rows && rows.length > maxRows ? rows[maxRows] : undefined;
  if (!limit) {
    box.style.maxHeight = '';
    box.style.overflowY = '';
    return;
  }
  const height = limit.getBoundingClientRect().top - table.getBoundingClientRect().top;
  box.style.maxHeight = `${Math.ceil(height + box.offsetHeight - box.clientHeight)}px`;
  box.style.overflowY = 'auto';
}

const Table = React.forwardRef<HTMLTableElement, TableProps>(({ className, maxRows = 10, ...props }, ref) => {
  const inner = React.useRef<HTMLTableElement>(null);
  React.useImperativeHandle(ref, () => inner.current as HTMLTableElement);
  React.useLayoutEffect(() => {
    const table = inner.current;
    if (!table || maxRows === false) return;
    capRows(table, maxRows);
    const observer = new ResizeObserver(() => capRows(table, maxRows));
    observer.observe(table);
    return () => observer.disconnect();
  });
  return (
    // O contêiner rolável é responsabilidade de quem usa: as tabelas analíticas têm
    // dezenas de colunas e precisam rolar horizontalmente sem empurrar a página.
    <table ref={inner} className={cn('w-full caption-bottom text-sm', className)} {...props} />
  );
});
Table.displayName = 'Table';

const TableHeader = React.forwardRef<
  HTMLTableSectionElement,
  React.HTMLAttributes<HTMLTableSectionElement>
>(({ className, ...props }, ref) => (
  <thead
    ref={ref}
    // Fixo no topo quando o contêiner rola na vertical; fundo opaco para as linhas não
    // aparecerem por baixo.
    className={cn('sticky top-0 z-10 bg-muted [&_tr]:border-b', className)}
    {...props}
  />
));
TableHeader.displayName = 'TableHeader';

const TableBody = React.forwardRef<
  HTMLTableSectionElement,
  React.HTMLAttributes<HTMLTableSectionElement>
>(({ className, ...props }, ref) => (
  <tbody ref={ref} className={cn('[&_tr:last-child]:border-0', className)} {...props} />
));
TableBody.displayName = 'TableBody';

const TableRow = React.forwardRef<
  HTMLTableRowElement,
  React.HTMLAttributes<HTMLTableRowElement>
>(({ className, ...props }, ref) => (
  <tr
    ref={ref}
    className={cn(
      'border-b transition-colors hover:bg-blue-500/[0.04] data-[state=selected]:bg-muted',
      className,
    )}
    {...props}
  />
));
TableRow.displayName = 'TableRow';

const TableHead = React.forwardRef<
  HTMLTableCellElement,
  React.ThHTMLAttributes<HTMLTableCellElement>
>(({ className, ...props }, ref) => (
  <th
    ref={ref}
    className={cn(
      'h-10 whitespace-nowrap px-3 text-left align-middle text-xs font-semibold text-foreground/75 tracking-tight',
      className,
    )}
    {...props}
  />
));
TableHead.displayName = 'TableHead';

const TableCell = React.forwardRef<
  HTMLTableCellElement,
  React.TdHTMLAttributes<HTMLTableCellElement>
>(({ className, ...props }, ref) => (
  <td ref={ref} className={cn('px-3 py-2 align-middle', className)} {...props} />
));
TableCell.displayName = 'TableCell';

export { Table, TableBody, TableCell, TableHead, TableHeader, TableRow };
