import Papa from 'papaparse';

import { FIELD } from '@/lib/schema';
import type { Dataset } from '@/lib/types';
import { normalizeDataset, type RawRow } from '../normalize';
import { toNumeric } from '../text';
import { stripBom } from './bom';

/**
 * Importador de CSV do Scopus — ⇄ `processar_csv_scopus` (utils.py:2328).
 */

/** Rótulos do Scopus para os nomes canônicos. */
const COLUMN_MAP: Readonly<Record<string, string>> = {
  Title: FIELD.TITLE,
  Year: FIELD.YEAR,
  'Source title': FIELD.SECONDARY_TITLE,
  Abstract: FIELD.ABSTRACT,
  'Document Type': FIELD.DOCUMENT_TYPE,
  DOI: FIELD.DOI,
};

/**
 * Extrai o país de uma string de afiliações do Scopus.
 *
 * O formato é "Instituição, Departamento, Cidade, País", com afiliações separadas por
 * ';'. O país é o último segmento; dígitos de CEP grudados são removidos.
 */
function extractCountries(value: unknown): string {
  if (value === null || value === undefined) return '';

  const text = String(value).trim();
  if (!text) return '';

  const countries = new Set<string>();
  for (const affiliation of text.split(';')) {
    const parts = affiliation.split(',');
    const last = parts[parts.length - 1];
    if (last === undefined) continue;

    const cleaned = last.replace(/\d/g, '').trim();
    if (cleaned) countries.add(cleaned);
  }

  return [...countries].join('; ');
}

export function processScopusCsv(text: string): Dataset {
  // O Scopus exporta com BOM; `skipEmptyLines` reproduz o `on_bad_lines='skip'`.
  const parsed = Papa.parse<Record<string, string>>(stripBom(text), {
    header: true,
    skipEmptyLines: true,
    dynamicTyping: false,
  });

  const rows: RawRow[] = [];

  for (const record of parsed.data) {
    if (!record || typeof record !== 'object') continue;

    const row: RawRow = {};
    for (const [key, value] of Object.entries(record)) {
      row[COLUMN_MAP[key] ?? key] = value;
    }

    // "References" alimenta o campo unificado de referências citadas.
    if ('References' in record) row[FIELD.REFERENCES_UNIFIED] = record['References'] ?? '';

    if ('Cited by' in record) {
      row[FIELD.TOTAL_CITATIONS] = toNumeric(record['Cited by']) ?? 0;
    }

    // O Scopus separa autores por vírgula depois da inicial ("Silva A., Santos B.");
    // o resto do pipeline espera ponto e vírgula.
    if ('Authors' in record) {
      row[FIELD.AUTHORS] = String(record['Authors'] ?? '').replace(/\.,/g, '.;');
    }

    // Palavras-chave de autor e indexadas entram no mesmo campo.
    const keywordParts = ['Author Keywords', 'Index Keywords']
      .filter((column) => column in record)
      .map((column) => String(record[column] ?? '').trim())
      .filter(Boolean);
    row[FIELD.KEYWORDS] = keywordParts.join('; ');

    if ('Affiliations' in record) {
      row[FIELD.COUNTRY] = extractCountries(record['Affiliations']);
    }

    rows.push(row);
  }

  return normalizeDataset(rows);
}
