import { FIELD, FIELD_CANDIDATES } from '@/lib/schema';
import type { Dataset, WorkerProgress } from '@/lib/types';
import { normalizeDataset, type RawRow } from '../normalize';
import { collectColumns, pickColumn, toNumeric } from '../text';
import { stripBom } from './bom';
import { extractCountries } from './countries';
import { flattenRisRecord, parseRis } from './ris';

/** Padrões que o WoS e o Scopus escondem no campo de notas — ⇄ utils.py:2778. */
const CITED_BY_PATTERN = /Cited\s+By:\s*(\d+)/;
const TIMES_CITED_PATTERN = /Times\s+Cited(?:.*?):\s*(\d+)/;

export interface RisSource {
  name: string;
  text: string;
  database: string;
}

/** Fração do total reservada a cada fase — leitura, enriquecimento, padronização. */
const PHASE_READ_END = 0.3;
const PHASE_ENRICH_END = 0.6;

/**
 * Pipeline completo de RIS — ⇄ `process_multiple_ris` (utils.py:2734).
 * Lê múltiplos arquivos, achata, resolve citações e países, e normaliza.
 *
 * `onProgress`, quando presente, reporta três fases reais (não cosméticas): a leitura
 * dos arquivos de origem, a resolução de citações/tipo/país, e a padronização final —
 * cada uma com progresso proporcional ao volume de documentos, não um salto único.
 */
export function processRisFiles(
  sources: readonly RisSource[],
  onProgress?: (progress: WorkerProgress) => void,
): Dataset {
  const rows: RawRow[] = [];

  sources.forEach((source, index) => {
    onProgress?.({
      phase: 'Lendo arquivos RIS',
      ratio: (index / sources.length) * PHASE_READ_END,
      detail: source.name,
    });

    let records;
    try {
      // O BOM precisa sair antes do parse.
      //
      // Com ele, a primeira linha vira `\uFEFFTY  - JOUR`, que não casa com o padrão de
      // tag do RIS. O parser então segue procurando um `TY` e engole o registro inteiro
      // até o próximo. O app Python tem exatamente esse comportamento e perde,
      // silenciosamente, o primeiro documento de todo arquivo com BOM — dois documentos
      // só na base de exemplo.
      records = parseRis(stripBom(source.text));
    } catch {
      // O Python engole falhas por arquivo e segue com os demais.
      return;
    }

    for (const record of records) {
      const flat = flattenRisRecord(record) as RawRow;
      flat['BASE DE DADOS'] = source.database;
      rows.push(flat);
    }
  });

  if (rows.length === 0) return [];

  const total = rows.length;
  onProgress?.({
    phase: 'Resolvendo citações, tipos e países',
    ratio: PHASE_READ_END,
    detail: `${total} documentos`,
  });

  const columns = collectColumns(rows);
  const citationColumns = FIELD_CANDIDATES.citations.filter((column) => columns.has(column));
  const hasNotes = columns.has('NOTES');
  const addressColumn = pickColumn(columns, FIELD_CANDIDATES.affiliation);

  rows.forEach((row, index) => {
    // Citações: primeira coluna com valor numérico vence, depois o resgate via NOTES.
    let citations: number | null = null;
    for (const column of citationColumns) {
      citations = toNumeric(row[column]);
      if (citations !== null) break;
    }

    if (citations === null && hasNotes) {
      const notes = String(row['NOTES'] ?? '');
      const citedBy = CITED_BY_PATTERN.exec(notes);
      const timesCited = citedBy ? null : TIMES_CITED_PATTERN.exec(notes);
      const raw = citedBy?.[1] ?? timesCited?.[1];
      if (raw !== undefined) citations = Number(raw);
    }

    row[FIELD.TOTAL_CITATIONS] = citations ?? 0;

    // "label.ris.referenceType.JOURNAL_ARTICLE" → "Journal Article".
    const referenceType = row[FIELD.TYPE_OF_REFERENCE];
    if (referenceType !== undefined && referenceType !== null) {
      row[FIELD.TYPE_OF_REFERENCE] = String(referenceType)
        .replace('label.ris.referenceType.', '')
        .replace(/_/g, ' ')
        .replace(/\p{L}+/gu, (word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase());
    }

    row[FIELD.COUNTRY] = addressColumn ? (extractCountries(row[addressColumn]) ?? '') : '';

    if (onProgress && index % 250 === 0) {
      onProgress({
        phase: 'Resolvendo citações, tipos e países',
        ratio: PHASE_READ_END + (index / total) * (PHASE_ENRICH_END - PHASE_READ_END),
        detail: `${index} de ${total} documentos`,
      });
    }
  });

  onProgress?.({
    phase: 'Padronizando estrutura',
    ratio: PHASE_ENRICH_END,
    detail: `${total} documentos`,
  });

  return normalizeDataset(rows, (processed, processedTotal) => {
    onProgress?.({
      phase: 'Padronizando estrutura',
      ratio: PHASE_ENRICH_END + (processed / processedTotal) * (1 - PHASE_ENRICH_END),
      detail: `${processed} de ${processedTotal} documentos`,
    });
  });
}
