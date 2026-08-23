import { BOM } from './parsers/bom';

/**
 * Exportação de tabelas para CSV — ⇄ `gerar_csv_bytes` (utils.py:602).
 */

/** Escapa um valor conforme o RFC 4180. */
function escapeCell(value: unknown): string {
  if (value === null || value === undefined) return '';

  const text = String(value);
  // Aspas, vírgulas e quebras de linha obrigam a delimitar o campo — e resumos de artigo
  // contêm os três com frequência.
  if (/["\n\r,;]/.test(text)) return `"${text.replace(/"/g, '""')}"`;
  return text;
}

export interface CsvOptions {
  /** Colunas a exportar, na ordem desejada. Padrão: a união das chaves das linhas. */
  columns?: readonly string[];
  /** Separador de campos. Padrão `,`. */
  delimiter?: string;
}

/** Serializa linhas em CSV, com cabeçalho. */
export function toCsv(
  rows: readonly Record<string, unknown>[],
  options: CsvOptions = {},
): string {
  const delimiter = options.delimiter ?? ',';

  const columns =
    options.columns ??
    [...new Set(rows.flatMap((row) => Object.keys(row)))];

  if (columns.length === 0) return '';

  const lines = [columns.map(escapeCell).join(delimiter)];
  for (const row of rows) {
    lines.push(columns.map((column) => escapeCell(row[column])).join(delimiter));
  }

  return lines.join('\n');
}

/**
 * Dispara o download de um texto como arquivo.
 *
 * O BOM é adicionado de propósito: sem ele, o Excel no Windows interpreta o CSV como
 * Latin-1 e os acentos aparecem corrompidos — o destino mais provável destas tabelas.
 */
export function downloadCsv(filename: string, content: string): void {
  const blob = new Blob([`${BOM}${content}`], { type: 'text/csv;charset=utf-8;' });
  downloadBlob(filename, blob);
}

/** Dispara o download de um blob já montado. */
export function downloadBlob(filename: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');

  anchor.href = url;
  anchor.download = filename;
  anchor.style.display = 'none';

  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);

  // A URL do objeto segura o blob na memória até ser revogada.
  URL.revokeObjectURL(url);
}

/** Nome de arquivo seguro, com a data para diferenciar exportações sucessivas. */
export function timestampedFilename(base: string, extension: string): string {
  const stamp = new Date().toISOString().slice(0, 10);
  const safe = base
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-zA-Z0-9-]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .toLowerCase();
  return `simetrics-${safe}-${stamp}.${extension}`;
}
