import Papa from 'papaparse';

import { FIELD } from '@/lib/schema';
import type { Dataset } from '@/lib/types';
import { normalizeDataset, type RawRow } from '../normalize';
import { parseRis, type RisValue } from './ris';
import { stripBom } from './bom';

/**
 * Importador da Cochrane Library — ⇄ `processar_cochrane` (utils.py:141).
 * Aceita tanto o CSV quanto o RIS exportados pela plataforma.
 */

const CSV_COLUMN_MAP: Readonly<Record<string, string>> = {
  Title: FIELD.TITLE,
  'Author(s)': FIELD.AUTHORS,
  Source: FIELD.SECONDARY_TITLE,
  Year: FIELD.YEAR_CLEAN,
  Abstract: FIELD.ABSTRACT,
  Keywords: FIELD.KEYWORDS,
  DOI: FIELD.DOI,
};

/** Chaves do rispy para os nomes canônicos, no dialeto da Cochrane. */
const RIS_KEY_MAP: Readonly<Record<string, string>> = {
  title: FIELD.TITLE,
  primary_title: FIELD.TITLE,
  authors: FIELD.AUTHORS,
  journal_name: FIELD.SECONDARY_TITLE,
  year: FIELD.YEAR_CLEAN,
  abstract: FIELD.ABSTRACT,
  keywords: FIELD.KEYWORDS,
  doi: FIELD.DOI,
};

function flatten(value: RisValue): string {
  if (Array.isArray(value)) return value.join('; ');
  if (typeof value === 'object') return '';
  return String(value);
}

function processCsv(text: string): RawRow[] {
  const parsed = Papa.parse<Record<string, string>>(stripBom(text), {
    header: true,
    skipEmptyLines: true,
    dynamicTyping: false,
  });

  return parsed.data
    .filter((record): record is Record<string, string> => Boolean(record))
    .map((record) => {
      const row: RawRow = {};
      for (const [key, value] of Object.entries(record)) {
        row[CSV_COLUMN_MAP[key] ?? key] = value;
      }
      return row;
    });
}

function processRis(text: string): RawRow[] {
  // A Cochrane insere espaços extras nas tags ("A1  -  Autor"), quebrando o padrão
  // estrito de 6 colunas que o parser RIS exige. Normalizamos antes de entregar.
  const normalized = text.replace(/^([A-Z0-9]{2})\s+-\s+/gm, '$1  - ');

  return parseRis(normalized).map((record) => {
    const row: RawRow = {};

    for (const [key, value] of Object.entries(record)) {
      if (key === 'unknown_tag') continue;
      row[RIS_KEY_MAP[key] ?? key.toUpperCase().replace(/_/g, ' ')] = flatten(value);
    }

    // Na Cochrane a tag A1 carrega os autores, e o rispy a mapeia para `first_authors`.
    if (!row[FIELD.AUTHORS] && record['first_authors'] !== undefined) {
      row[FIELD.AUTHORS] = flatten(record['first_authors']);
    }

    return row;
  });
}

export function processCochrane(fileName: string, text: string): Dataset {
  const rows = fileName.toLowerCase().endsWith('.csv') ? processCsv(text) : processRis(text);

  for (const row of rows) {
    // A Cochrane não exporta contagem de citações recebidas.
    row[FIELD.TOTAL_CITATIONS] = 0;

    // Remove o asterisco que marca o descritor MeSH principal.
    if (row[FIELD.KEYWORDS] !== undefined) {
      row[FIELD.KEYWORDS] = String(row[FIELD.KEYWORDS]).replace(/\*/g, '');
    }
  }

  return normalizeDataset(rows);
}
