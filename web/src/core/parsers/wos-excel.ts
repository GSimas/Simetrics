import { read, utils as sheetUtils } from 'xlsx';

import { FIELD } from '@/lib/schema';
import type { Dataset } from '@/lib/types';
import { normalizeDataset, type RawRow } from '../normalize';
import { toNumeric } from '../text';

/**
 * Importador de Excel do Web of Science — ⇄ `processar_excel_wos` (utils.py:2262).
 *
 * O SheetJS detecta sozinho `.xls` (binário legado) e `.xlsx`, então o roteamento
 * openpyxl/xlrd do Python desaparece aqui.
 */

const COLUMN_MAP: Readonly<Record<string, string>> = {
  'Article Title': FIELD.TITLE,
  'Publication Year': FIELD.YEAR,
  'Source Title': FIELD.SECONDARY_TITLE,
  Abstract: FIELD.ABSTRACT,
  'Document Type': FIELD.DOCUMENT_TYPE,
  DOI: FIELD.DOI,
  Authors: FIELD.AUTHORS,
};

/** Colunas de citação do WoS, em ordem de preferência (o Core é o padrão de impacto). */
const CITATION_COLUMNS = ['Times Cited, WoS Core', 'Times Cited, All Databases'] as const;

/** Colunas de afiliação de onde sai o país. */
const ADDRESS_COLUMNS = ['Addresses', 'Affiliations', 'Author Address'] as const;

export function processWosExcel(buffer: ArrayBuffer): Dataset {
  const workbook = read(buffer, { type: 'array' });
  const firstSheetName = workbook.SheetNames[0];
  if (firstSheetName === undefined) return [];

  const sheet = workbook.Sheets[firstSheetName];
  if (sheet === undefined) return [];

  const records = sheetUtils.sheet_to_json<Record<string, unknown>>(sheet, { defval: '' });
  const rows: RawRow[] = [];

  for (const record of records) {
    const row: RawRow = {};
    for (const [key, value] of Object.entries(record)) {
      row[COLUMN_MAP[key] ?? key] = value;
    }

    // "Cited References" alimenta AMBOS os destinos.
    //
    // O Python tem um dicionário com a chave 'Cited References' repetida (utils.py:2279),
    // e em Python a segunda entrada sobrescreve a primeira — então `CITED REFERENCES`
    // nunca é criada e só `REFERENCES_UNIFIED` sobrevive. Aqui os dois são preenchidos,
    // que era a intenção evidente do código original.
    if ('Cited References' in record) {
      const references = record['Cited References'];
      row[FIELD.CITED_REFERENCES] = references;
      row[FIELD.REFERENCES_UNIFIED] = references;
    }

    let citations: number | null = null;
    for (const column of CITATION_COLUMNS) {
      if (column in record) {
        citations = toNumeric(record[column]);
        if (citations !== null) break;
      }
    }
    row[FIELD.TOTAL_CITATIONS] = citations ?? 0;

    // O WoS já entrega o país como último segmento de cada endereço, entre colchetes de
    // autoria: "[Silva, A] Univ Sao Paulo, Sao Paulo, Brazil".
    for (const column of ADDRESS_COLUMNS) {
      if (!(column in record)) continue;

      const countries = new Set<string>();
      for (const address of String(record[column] ?? '').split(';')) {
        const parts = address.split(',');
        const last = parts[parts.length - 1];
        if (last === undefined) continue;

        const cleaned = last.replace(/\d/g, '').replace(/[[\]]/g, '').trim();
        if (cleaned) countries.add(cleaned);
      }
      row[FIELD.COUNTRY] = [...countries].join('; ');
      break;
    }

    rows.push(row);
  }

  return normalizeDataset(rows);
}
