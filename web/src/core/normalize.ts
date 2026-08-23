import { FIELD, FIELD_CANDIDATES } from '@/lib/schema';
import type { Dataset, SimetricsDoc } from '@/lib/types';
import { collectColumns, isNullLike, pickColumn, toNumeric } from './text';

export type RawRow = Record<string, unknown>;

/**
 * ⇄ `padronizar_base_bibliometrica` (utils.py:550).
 *
 * Ponto de convergência obrigatório: todo importador termina aqui, e é isto que garante
 * que o resto do pipeline possa assumir texto onde espera texto e número onde espera
 * número, sem checagem defensiva em cada função.
 */
export function normalizeDataset(rows: readonly RawRow[]): Dataset {
  if (rows.length === 0) return [];

  const columns = collectColumns(rows);

  // Unificação de referências citadas: o primeiro campo não-vazio vence, na ordem de
  // preferência. Bases diferentes usam rótulos diferentes para o mesmo dado.
  const referenceColumns = FIELD_CANDIDATES.references.filter((column) => columns.has(column));
  const hasReferences = referenceColumns.length > 0;

  const yearSource = columns.has(FIELD.YEAR_CLEAN)
    ? FIELD.YEAR_CLEAN
    : columns.has(FIELD.YEAR)
      ? FIELD.YEAR
      : null;

  return rows.map((row) => {
    const doc: RawRow = {};

    for (const [key, value] of Object.entries(row)) {
      // Listas e objetos soltos viram texto: o pipeline inteiro assume valores hasheáveis,
      // e um array escapando até um groupBy quebra a agregação.
      if (Array.isArray(value)) {
        doc[key] = value.map((item) => String(item)).join('; ');
      } else if (value !== null && typeof value === 'object') {
        doc[key] = JSON.stringify(value);
      } else {
        doc[key] = value;
      }
    }

    // Citações são garantidamente numéricas; ausência vira 0, nunca NaN.
    const citations = toNumeric(doc[FIELD.TOTAL_CITATIONS]);
    doc[FIELD.TOTAL_CITATIONS] = citations ?? 0;

    if (hasReferences) {
      let unified = '';
      for (const column of referenceColumns) {
        const value = doc[column];
        const text = value === null || value === undefined ? '' : String(value).trim();
        if (text) {
          unified = text;
          break;
        }
      }
      doc[FIELD.REFERENCES_UNIFIED] = unified;
    }

    // YEAR CLEAN é o ano numérico; `null` quando não parseável, para o Lotka e o m-index
    // conseguirem distinguir "sem ano" de "ano zero".
    if (yearSource) {
      const year = toNumeric(doc[yearSource]);
      doc[FIELD.YEAR_CLEAN] = year;
    } else if (!(FIELD.YEAR_CLEAN in doc)) {
      doc[FIELD.YEAR_CLEAN] = null;
    }

    // Campos de texto nunca são null/undefined depois daqui.
    for (const key of Object.keys(doc)) {
      if (key === FIELD.TOTAL_CITATIONS || key === FIELD.YEAR_CLEAN) continue;
      const value = doc[key];
      if (value === null || value === undefined) doc[key] = '';
      else if (typeof value !== 'string' && typeof value !== 'number') doc[key] = String(value);
    }

    return doc as SimetricsDoc;
  });
}

/**
 * Garante que os campos canônicos existam mesmo quando a base de origem não os traz,
 * para que a UI não precise checar presença a cada acesso.
 */
export function ensureCanonicalFields(rows: Dataset): Dataset {
  const columns = collectColumns(rows);

  const titleColumn = pickColumn(columns, FIELD_CANDIDATES.title);
  const authorsColumn = pickColumn(columns, FIELD_CANDIDATES.authors);
  const venueColumn = pickColumn(columns, FIELD_CANDIDATES.venue);
  const keywordsColumn = pickColumn(columns, FIELD_CANDIDATES.keywords);
  const doiColumn = pickColumn(columns, FIELD_CANDIDATES.doi);

  const alias = (row: RawRow, target: string, source: string | null): void => {
    if (row[target] !== undefined && !isNullLike(row[target])) return;
    row[target] = source && row[source] !== undefined ? row[source] : '';
  };

  for (const row of rows as unknown as RawRow[]) {
    alias(row, FIELD.TITLE, titleColumn);
    alias(row, FIELD.AUTHORS, authorsColumn);
    alias(row, FIELD.SECONDARY_TITLE, venueColumn);
    alias(row, FIELD.KEYWORDS, keywordsColumn);
    alias(row, FIELD.DOI, doiColumn);
    row[FIELD.COUNTRY] ??= '';
    row[FIELD.ABSTRACT] ??= '';
    row[FIELD.REFERENCES_UNIFIED] ??= '';
  }

  return rows;
}
