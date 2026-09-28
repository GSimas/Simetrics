import { FIELD, FIELD_CANDIDATES } from '@/lib/schema';
import type { Dataset } from '@/lib/types';
import { collectColumns, isNullLike, pickColumn, toNumeric } from '../text';

/**
 * Visão de um documento da base para a triagem, com uma chave estável.
 *
 * A chave vem do conteúdo, nunca da posição na base: DOI normalizado quando existe, senão
 * um hash do título normalizado com o ano. Posições mudam a cada deduplicação ou arquivo
 * acrescentado; o conteúdo não — e é ele que dois revisores com a mesma busca têm em comum.
 */
export interface ScreeningRecord {
  key: string;
  /** Posição na base ativa. */
  index: number;
  title: string;
  abstract: string;
  authors: string;
  year: number | null;
  venue: string;
  keywords: string;
  doi: string;
  database: string;
}

function text(value: unknown): string {
  return isNullLike(value) ? '' : String(value).trim();
}

/** DOI sem prefixo de URL nem `doi:`, em minúsculas — DOIs não diferenciam caixa. */
export function normalizeDoi(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/^https?:\/\/(dx\.)?doi\.org\//, '')
    .replace(/^doi:\s*/, '')
    .trim();
}

/** Título sem acentos, pontuação nem caixa: a mesma obra exportada por bases diferentes. */
export function normalizeTitle(value: string): string {
  return value
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

/** FNV-1a de 32 bits em base 36 — curto, determinístico e sem dependência. */
function hash(value: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < value.length; i += 1) {
    h ^= value.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(36);
}

export function recordKey(doi: string, title: string, year: number | null): string {
  const normalizedDoi = normalizeDoi(doi);
  if (normalizedDoi.startsWith('10.')) return `doi:${normalizedDoi}`;
  const normalizedTitle = normalizeTitle(title);
  if (normalizedTitle) return `ti:${hash(normalizedTitle)}:${year ?? ''}`;
  return '';
}

export function toScreeningRecords(rows: Dataset): ScreeningRecord[] {
  const columns = collectColumns(rows);
  const titleColumn = pickColumn(columns, FIELD_CANDIDATES.title);
  const abstractColumn = pickColumn(columns, FIELD_CANDIDATES.abstract);
  const authorsColumn = pickColumn(columns, FIELD_CANDIDATES.authors);
  const yearColumn = pickColumn(columns, FIELD_CANDIDATES.year);
  const venueColumn = pickColumn(columns, FIELD_CANDIDATES.venue);
  const keywordsColumn = pickColumn(columns, FIELD_CANDIDATES.keywords);
  const doiColumn = pickColumn(columns, FIELD_CANDIDATES.doi);

  // Sem deduplicação, a mesma obra pode aparecer mais de uma vez: a segunda ocorrência
  // ganha um sufixo, e cada cópia é triada por conta própria.
  const seen = new Map<string, number>();

  return rows.map((doc, index) => {
    const title = titleColumn ? text(doc[titleColumn]) : '';
    const doi = doiColumn ? text(doc[doiColumn]) : '';
    const rawYear = yearColumn ? toNumeric(doc[yearColumn]) : null;
    const year = rawYear === null ? null : Math.trunc(rawYear);

    const base = recordKey(doi, title, year) || `row:${index}`;
    const occurrence = (seen.get(base) ?? 0) + 1;
    seen.set(base, occurrence);

    return {
      key: occurrence === 1 ? base : `${base}#${occurrence}`,
      index,
      title,
      abstract: abstractColumn ? text(doc[abstractColumn]) : '',
      authors: authorsColumn ? text(doc[authorsColumn]) : '',
      year,
      venue: venueColumn ? text(doc[venueColumn]) : '',
      keywords: keywordsColumn ? text(doc[keywordsColumn]) : '',
      doi,
      database: text(doc[FIELD.DATABASE]),
    };
  });
}
